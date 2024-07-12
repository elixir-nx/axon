defmodule Axon.Quantization.Layers do
  @moduledoc """
  Quantized Layer Implementations.
  """

  import Nx.Defn

  # TODO: Make this more general

  defn weight_only_quantized_dense(x, kernel, bias, scales, _opts \\ []) do
    # TODO: Flatten x if necessary

    x
    |> Nx.dot(Nx.as_type(kernel, Nx.type(x)))
    |> Nx.multiply(scales)
    |> Nx.add(bias)
  end
end
