defmodule Axon.Quantization.QTensor do
  @moduledoc """
  Representation of a quantized tensor.

  A quantized tensor stores information about the quantized
  value, scale, and zero-point. This module contains lower-level
  functions for converting to and from quantized tensors.

  In most cases, you should prefer to use the public APIs in
  `Axon.Quantization`.
  """
  import Nx.Defn

  @derive {Nx.Container, containers: [:value, :scale, :zero_point]}
  defstruct [:value, :scale, :zero_point]

  @doc """
  Converts a regular float tensor into a quantized tensor.
  """
  deftransform from_tensor(x, opts \\ []) do
    opts = Keyword.validate!(opts, type: {:s, 8})

    case opts[:type] do
      {:s, 8} ->
        dynamically_quantize_per_channel(x, min: -128, max: 127, type: {:s, 8})

      other ->
        raise "unsupported quantization type #{inspect(other)}"
    end
  end

  deftransformp dynamically_quantize_per_channel(input, opts \\ []) do
    opts = Keyword.validate!(opts, [:min, :max, :type])

    unless Nx.type(input) == {:f, 32}, do: raise(ArgumentError, "expected a float tensor")
    unless Nx.rank(input) == 2, do: raise(ArgumentError, "expected a 2d tensor")

    target_dtype = opts[:type]
    eps = Nx.Constants.epsilon(:f32)
    block_size = {1, Nx.axis_size(input, 1)}
    zero_point_type = {:s, 64}

    {scale, zero_point} =
      choose_quantization_params_affine(input,
        mapping_type: :symmetric,
        block_size: block_size,
        type: opts[:type],
        min: opts[:min],
        max: opts[:max],
        eps: eps,
        zero_point_type: zero_point_type
      )

    quantized_value =
      quantize_affine(input, scale, zero_point,
        block_size: block_size,
        type: target_dtype,
        min: opts[:min],
        max: opts[:max]
      )

    struct(__MODULE__, value: quantized_value, scale: scale, zero_point: zero_point)
  end

  deftransformp quantize_affine(
                  input,
                  scale,
                  zero_point,
                  opts \\ []
                ) do
    opts = Keyword.validate!(opts, [:block_size, :type, :min, :max, zero_point_domain: :int])

    target_dtype = opts[:type]
    quant_min = opts[:min]
    quant_max = opts[:max]
    block_size = opts[:block_size]
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

  deftransformp choose_quantization_params_affine(input, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :mapping_type,
        :block_size,
        :type,
        :min,
        :max,
        :eps,
        :scale_type,
        :zero_point_type,
        :zero_point_domain,
        preserve_zero: true
      ])

    mapping_type = opts[:mapping_type]
    block_size = opts[:block_size]
    target_dtype = opts[:type]
    preserve_zero = opts[:preserve_zero]

    {quant_min, quant_max} =
      get_and_check_qmin_qmax(target_dtype, opts[:min], opts[:max])

    scale_dtype = opts[:scale_type] || Nx.type(input)
    zero_point_dtype = opts[:zero_point_type] || Nx.type(input)
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

  deftransformp get_and_check_qmin_qmax(target_dtype, quant_min, quant_max) do
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

  deftransformp get_reduction_params(block_size, input_size) do
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
end
