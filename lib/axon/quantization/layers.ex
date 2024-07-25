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
          input,
          %QTensor{value: kernel, scale: scale},
          bias,
          _opts
        ) do
    input
    |> Nx.dot([Nx.rank(input) - 1], Nx.as_type(kernel, Nx.type(input)), [0])
    |> Nx.multiply(scale)
    |> Nx.add(bias)
  end
end
