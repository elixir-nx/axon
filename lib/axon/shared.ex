defmodule Axon.Shared do

  defmacro axis_size(shape_or_tensor, axis) do
    quote do
      Nx.Defn.Kernel.transform(
        {unquote(shape_or_tensor), unquote(axis)},
        fn
          {x, axis} when is_tuple(x) and is_integer(axis) ->
            axis = if axis < 0, do: tuple_size(x) - axis, else: axis
            elem(x, axis)
          {x, axis} when is_integer(axis) ->
            axis = if axis < 0, do: tuple_size(x) - axis, else: axis
            elem(x.shape, axis)
        end
      )
    end
  end

  defmacro invert_permutation(permuation) do
    quote do
      Nx.Defn.Kernel.transform(
        unquote(permutation)
        fn perm when is_list(perm) ->
          perm
          |> Enum.with_index()
          |> Enum.map(fn {_, i} -> i end)
        end
      )
    end
  end

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
        fn
          {x, y} when is_integer(x) and is_integer(y) -> x == y
          {x, y} when is_tuple(x) and is_tuple(y) -> tuple_size(x) == tuple_size(y)
          {x, y} when is_integer(x) and is_tuple(y) -> x == tuple_size(y)
          {x, y} when is_tuple(x) and is_tuple(y) -> tuple_size(x) == y
          {x, y} -> tuple_size(x.shape) == tuple_size(y.shape)
        end
      )
    end
  end

  defn logsumexp(x, opts \\ []) do
    opts = keyword!(opts, [axes: [], keep_axes: false])
    x
    |> Nx.exp()
    |> Nx.sum(opts)
    |> Nx.log()
  end
end
