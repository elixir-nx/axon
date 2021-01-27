defmodule Axon.Layers do
  @moduledoc """
  Functional implementations of common neural network layers.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` backend.
  """

  import Nx.Defn

  @doc ~S"""
  Dense layer.

  Linear transformation of the input such that:

  $$y = xW^T + b$$
  """
  # TODO: Optional bias
  defn dense(input, weight, bias) do
    input
    |> Nx.dot(weight)
    |> Nx.add(bias)
  end
end
