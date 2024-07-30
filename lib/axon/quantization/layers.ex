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
    x_shape = Nx.shape(x)
    last_dim = Nx.axis_size(x, -1)

    x_view = Nx.reshape(x, {:auto, last_dim})

    y = Nx.dot(x_view, Nx.as_type(Nx.transpose(w_int8), Nx.type(x)))
    y = Nx.multiply(y, scales)
    y = reshape_output(y, x_shape)

    Nx.add(y, bias)
  end

  deftransformp reshape_output(output, x_shape) do
    all_but_last = Tuple.delete_at(x_shape, tuple_size(x_shape) - 1)
    new_shape = Tuple.append(all_but_last, :auto)
    Nx.reshape(output, new_shape)
  end
end
