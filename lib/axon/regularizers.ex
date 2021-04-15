defmodule Axon.Regularizers do
  @moduledoc """
  Collection of parameter regularizers.
  """

  import Nx.Defn

  @doc """
  L1 Regularization.
  """
  defn l1(x, opts \\ []) do
    opts = keyword!(opts, penalty: 0.01)
    opts[:penalty] * Nx.sum(Nx.abs(x))
  end

  @doc """
  L2 Regularization.
  """
  defn l2(x, opts \\ []) do
    opts = keyword!(opts, penalty: 0.01)
    opts[:penalty] * Nx.sum(x * x)
  end

  @doc """
  L1L2 Regularization.
  """
  defn l1l2(x, opts \\ []) do
    opts = keyword!(opts, l1_penalty: 0.01, l2_penalty: 0.01)
    l1_penalty = opts[:l1_penalty]
    l2_penalty = opts[:l2_penalty]
    l1(x, penalty: l1_penalty) + l2(x, penalty: l2_penalty)
  end
end