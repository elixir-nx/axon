defmodule Axon.Shared do
  @doc """
  Asserts the lhs shape is equal to the rhs shape.
  """
  defmacro assert_shape!(lhs, rhs) do
    quote do
      Nx.Defn.Kernel.transform(
        {unquote(lhs), unquote(rhs)},
        &assert_shape_impl/1
      )
    end
  end

  @doc false
  def assert_shape_impl({lhs, rhs}) do
    unless lhs.shape == rhs.shape do
      raise ArgumentError,
            "expected input shapes to be equal," <>
              " got #{inspect(lhs)} != #{inspect(rhs)}"
    end
  end

  @doc """
  Asserts the given boolean expression on is true.
  """
  defmacro assert_rank!(lhs, rhs) do
    quote do
      Nx.Defn.Kernel.transform(
        {unquote(lhs), unquote(rhs)},
        &assert_rank_impl/1
      )
    end
  end

  @doc false
  def assert_rank_impl({lhs, rhs}) do
    unless tuple_size(lhs.shape) == tuple_size(rhs.shape) do
      raise ArgumentError,
            "expected input ranks to be equal," <>
              " got #{lhs} != #{rhs}"
    end
  end
end
