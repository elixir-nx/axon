defmodule Axon.Quantization do
  alias Axon.Quantization.Layers

  ## Transformation

  def quantize(%Axon{} = model, %Axon.ModelState{} = model_state) do
    quantized_model = rewrite_dense(model)
    quantized_model_state = quantize_dense_layers(model, model_state)
    {quantized_model, quantized_model_state}
  end

  defp rewrite_dense(%Axon{} = model) do
    # TODO: Make this easier
    Axon.map_nodes(model, fn
      %{op_name: :dense, args: args, parameters: parameters} = axon_node ->
        scales = Axon.param("scales", &quantized_dense_scale/1, initializer: :zeros, kind: :state)

        %{
          axon_node
          | op_name: :weight_only_quantized_dense,
            op: &Layers.weight_only_quantized_dense/5,
            args: args ++ [:parameter],
            parameters: parameters ++ [scales]
        }

      axon_node ->
        axon_node
    end)
  end

  defp quantize_dense_layers(model, model_state) do
    # TODO: Make these updates easier
    dense_layer_names =
      model
      |> Axon.properties()
      |> Enum.filter(fn {_, v} -> v == :dense end)
      |> Enum.map(fn {k, _} -> k end)

    Enum.reduce(dense_layer_names, model_state, fn layer_name, state ->
      state
      |> update_in([Access.key!(:data), layer_name], fn params ->
        quantize_dense_params(params)
      end)
      |> update_in([Access.key!(:state), layer_name], fn _ ->
        ["scales"]
      end)
    end)
  end

  defp quantize_dense_params(%{"kernel" => dense_kernel, "bias" => dense_bias}) do
    transposed_kernel = Nx.transpose(dense_kernel)

    {quant_kernel, scales, _zero} =
      dynamically_quantize_per_channel(transposed_kernel, -128, 127, {:s, 8})

    %{
      "kernel" => Nx.transpose(quant_kernel),
      "bias" => dense_bias,
      "scales" => scales
    }
  end

  ## Quantizers

  def dynamically_quantize_per_channel(%Nx.Tensor{} = x, quant_min, quant_max, target_dtype) do
    unless Nx.rank(x) == 2, do: raise("expected 2d tensor")

    eps = Nx.Constants.epsilon(:f32)
    block_size = {1, Nx.axis_size(x, 1)}
    zero_point_dtype = {:s, 64}

    {scale, zero_point} =
      choose_quantization_params_affine(x, :symmetric, block_size, target_dtype,
        quant_min: quant_min,
        quant_max: quant_max,
        eps: eps,
        zero_point_dtype: zero_point_dtype
      )

    quant = quantize_affine(x, block_size, scale, zero_point, target_dtype, quant_min, quant_max)

    {quant, scale, zero_point}
  end

  def quantize_affine(
        input,
        block_size,
        scale,
        zero_point,
        target_dtype,
        quant_min,
        quant_max,
        opts \\ []
      ) do
    opts = Keyword.validate!(opts, zero_point_domain: :int)
    zero_point_domain = opts[:zero_point_domain]

    {shape_for_reduction, reduction_dims} = get_reduction_params(block_size, Nx.shape(input))

    original_shape = Nx.shape(input)
    input = Nx.reshape(input, shape_for_reduction)

    scale_shape =
      Enum.reduce(reduction_dims, shape_for_reduction, fn i, shape ->
        put_elem(shape, i, 1)
      end)

    scale = Nx.reshape(scale, scale_shape)
    zero_point = Nx.reshape(zero_point, scale_shape)

    quant =
      case zero_point_domain do
        :int ->
          Nx.clip(
            Nx.add(Nx.round(Nx.multiply(input, Nx.divide(1, scale))), zero_point),
            quant_min,
            quant_max
          )

        other ->
          raise "unsupported zero point domain #{other}"
      end

    Nx.as_type(Nx.reshape(quant, original_shape), target_dtype)
  end

  def choose_quantization_params_affine(
        input,
        mapping_type,
        block_size,
        target_dtype,
        opts \\ []
      ) do
    opts =
      Keyword.validate!(opts, [
        :quant_min,
        :quant_max,
        :eps,
        :scale_dtype,
        :zero_point_dtype,
        :zero_point_domain,
        preserve_zero: true
      ])

    preserve_zero = opts[:preserve_zero]

    {quant_min, quant_max} =
      get_and_check_qmin_qmax(target_dtype, opts[:quant_min], opts[:quant_max])

    scale_dtype = opts[:scale_dtype] || Nx.type(input)
    zero_point_dtype = opts[:zero_point_dtype] || Nx.type(input)
    eps = opts[:eps] || Nx.Constants.epsilon(Nx.type(input))

    {shape_for_reduction, reduction_dims} = get_reduction_params(block_size, Nx.shape(input))
    input = Nx.reshape(input, shape_for_reduction)

    min_val = Nx.reduce_min(input, axes: reduction_dims, keep_axes: false)
    max_val = Nx.reduce_max(input, axes: reduction_dims, keep_axes: false)

    {min_val_neg, max_val_pos} =
      if preserve_zero do
        {Nx.min(min_val, Nx.broadcast(0, min_val)), Nx.max(max_val, Nx.broadcast(0, max_val))}
      else
        {min_val, max_val}
      end

    {scale, zero_point} =
      case mapping_type do
        :symmetric ->
          max_val_pos = Nx.max(Nx.negate(min_val_neg), max_val_pos)
          scale = Nx.divide(max_val_pos, Nx.divide(Nx.subtract(quant_max, quant_min), 2))
          zero_point = Nx.broadcast(trunc((quant_max + quant_min + 1) / 2), scale)
          {scale, zero_point}

        other ->
          raise "unsupported mapping #{other}"
      end

    scale = Nx.clip(scale, eps, Nx.reduce_max(scale))

    {Nx.as_type(scale, scale_dtype), Nx.as_type(zero_point, zero_point_dtype)}
  end

  def get_and_check_qmin_qmax(target_dtype, quant_min, quant_max) do
    {lower_bound, upper_bound} =
      case target_dtype do
        {:u, 8} -> {0, 255}
        {:s, 8} -> {-128, 127}
        {:s, 16} -> {-(2 ** 15), 2 ** 15 - 1}
        {:s, 32} -> {-(2 ** 31), 2 ** 31 - 1}
      end

    quant_min =
      cond do
        quant_min == nil ->
          lower_bound

        quant_min < lower_bound ->
          raise "quant_min out of bounds for target_dtype"

        true ->
          quant_min
      end

    quant_max =
      cond do
        quant_max == nil ->
          upper_bound

        quant_max > upper_bound ->
          raise "quant_max out of bounds for target_dtype"

        true ->
          quant_max
      end

    {quant_min, quant_max}
  end

  def get_reduction_params(block_size, input_size) do
    if tuple_size(block_size) != tuple_size(input_size) do
      raise "block_size and input_size must have the same length"
    end

    {shape_for_reduction, reduction_dims, _} =
      block_size
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(input_size))
      |> Enum.with_index()
      |> Enum.reduce({[], [], 0}, fn {{block, input}, i}, {shape, dims, cur_dim} ->
        if block != input and block > 1 do
          unless rem(input, block) == 0 do
            raise "Expecting input size at #{i} dimension: #{input} to be divisible by block_size at #{i} dimension: #{block}"
          end

          shape = [block, div(input, block) | shape]
          dims = [cur_dim + 1 | dims]
          cur_dim = cur_dim + 2

          {shape, dims, cur_dim}
        else
          shape = [input | shape]
          dims = if block != 1, do: [cur_dim | dims], else: dims
          cur_dim = cur_dim + 1

          {shape, dims, cur_dim}
        end
      end)

    {List.to_tuple(Enum.reverse(shape_for_reduction)), Enum.reverse(reduction_dims)}
  end

  ## Layers

  def weight_only_quantized_dense(input, units, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        :meta,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros
      ])

    kernel_shape = &Axon.Shape.dense_kernel(&1, units)
    bias_shape = &Axon.Shape.dense_bias(&1, units)
    scales_shape = &quantized_dense_scale/1

    kernel = Axon.param("kernel", kernel_shape, initializer: opts[:kernel_initializer])
    bias = Axon.param("bias", bias_shape, initializer: opts[:bias_initializer])
    # TODO: This requires dependent initializers
    scales = Axon.param("scales", scales_shape, initializer: :zeros)

    Axon.layer(&Layers.weight_only_quantized_dense/5, [input, kernel, bias, scales],
      meta: opts[:meta],
      name: opts[:name],
      op_name: :weight_only_quantized_dense
    )
  end

  defp quantized_dense_scale(input_shape) do
    Nx.axis_size(input_shape, -1)
  end

  ## Quantizers
end
