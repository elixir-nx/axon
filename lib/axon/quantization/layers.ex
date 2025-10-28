defmodule Axon.Quantization.Layers do
  @moduledoc """
  Quantized Layer Implementations.
  """
  alias Axon.Quantization.QTensor
  import Nx.Defn

  @doc """
  Weight-only quantized version of a dense layer.

  It expects the input kernel to be an `Axon.Quantization.QTensor`.
  """
  deftransform weight_only_quantized_dense(input, kernel, bias \\ 0, opts \\ []) do
    {bias, opts} =
      case bias do
        %Nx.Tensor{} = bias ->
          {bias, opts}

        bias when is_number(bias) ->
          {bias, opts}

        opts when is_list(opts) ->
          {Nx.tensor(0), opts}

        other ->
          raise ArgumentError, "invalid bias, expected a tensor, got #{inspect(other)}"
      end

    weight_only_quantized_dense_impl(input, kernel, bias, opts)
  end

  defnp weight_only_quantized_dense_impl(
          x,
          %QTensor{value: w_int8, scale: scales},
          bias,
          _opts
        ) do
    x_view = Nx.reshape(x, {:auto, Nx.axis_size(x, -1)})

    y = Nx.dot(x_view, Nx.as_type(w_int8, Nx.type(x)))
    y = Nx.multiply(y, reshape_scales(scales, y))
    y = reshape_output(y, Nx.shape(x))

    Nx.add(y, bias)
  end

  deftransformp reshape_scales(scales, y) do
    ones = List.to_tuple(List.duplicate(1, Nx.rank(y) - 1))
    Nx.reshape(scales, :erlang.append_element(ones, :auto))
  end

  deftransformp reshape_output(output, x_shape) do
    all_but_last = Tuple.delete_at(x_shape, tuple_size(x_shape) - 1)
    Nx.reshape(output, :erlang.append_element(all_but_last, :auto))
  end
end
