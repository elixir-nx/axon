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
        dynamically_quantize_per_channel(Nx.transpose(x), min: -128, max: 127, type: {:s, 8})

      other ->
        raise "unsupported quantization type #{inspect(other)}"
    end
  end

  deftransformp dynamically_quantize_per_channel(input, opts \\ []) do
    opts = Keyword.validate!(opts, [:min, :max, :type])

    input =
      case Nx.type(input) do
        {:f, 32} -> input
        {:f, _} -> Nx.as_type(input, {:f, 32})
        {:bf, _} -> Nx.as_type(input, {:f, 32})
        other -> raise ArgumentError, "expected a float tensor, got #{inspect(other)}"
      end

    unless Nx.rank(input) == 2, do: raise(ArgumentError, "expected a 2d tensor")

    target_dtype = opts[:type]
    block_size = {1, Nx.axis_size(input, 1)}

    {quant_min, quant_max} = get_and_check_qmin_qmax(target_dtype, opts[:min], opts[:max])
    zero_point_value = trunc((quant_max + quant_min + 1) / 2)

    {shape_for_reduction, reduction_dims} = get_reduction_params(block_size, Nx.shape(input))
    original_shape = Nx.shape(input)

    scale_shape =
      Enum.reduce(reduction_dims, shape_for_reduction, fn i, shape ->
        put_elem(shape, i, 1)
      end)

    input = Nx.reshape(input, shape_for_reduction)

    {value, scale, zero_point} =
      do_quantize_kernel(input,
        reduction_dims: reduction_dims,
        scale_shape: scale_shape,
        original_shape: original_shape,
        quant_min: quant_min,
        quant_max: quant_max,
        zero_point_value: zero_point_value,
        target_dtype: target_dtype
      )

    struct(__MODULE__, value: Nx.transpose(value), scale: scale, zero_point: zero_point)
  end

  defnp do_quantize_kernel(input, opts \\ []) do
    opts =
      keyword!(opts, [
        :reduction_dims,
        :scale_shape,
        :original_shape,
        :quant_min,
        :quant_max,
        :zero_point_value,
        :target_dtype
      ])

    min_val = Nx.reduce_min(input, axes: opts[:reduction_dims])
    max_val = Nx.reduce_max(input, axes: opts[:reduction_dims])

    min_val_neg = Nx.min(min_val, 0)
    max_val_pos = Nx.max(max_val, 0)
    max_val_pos = Nx.max(Nx.negate(min_val_neg), max_val_pos)

    divisor = (opts[:quant_max] - opts[:quant_min]) / 2
    scale = max_val_pos / divisor
    scale = Nx.clip(scale, Nx.Constants.epsilon(:f32), Nx.reduce_max(scale))
    zero_point = Nx.broadcast(opts[:zero_point_value], scale)

    scale_r = Nx.reshape(scale, opts[:scale_shape])
    zero_point_r = Nx.reshape(zero_point, opts[:scale_shape])

    value =
      input
      |> Nx.multiply(Nx.divide(1, scale_r))
      |> Nx.round()
      |> Nx.add(zero_point_r)
      |> Nx.clip(opts[:quant_min], opts[:quant_max])
      |> Nx.reshape(opts[:original_shape])
      |> Nx.as_type(opts[:target_dtype])

    {value, scale, Nx.as_type(zero_point, {:s, 64})}
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
