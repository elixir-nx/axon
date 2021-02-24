defmodule Axon.Shared do

  import Nx.Defn

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

  defmacro invert_permutation(permutation) do
    quote do
      Nx.Defn.Kernel.transform(
        unquote(permutation),
        fn perm when is_list(perm) ->
          perm
          |> Enum.with_index()
          |> Enum.map(fn {_, i} -> i end)
        end
      )
    end
  end

  @doc """
  Asserts `lhs` has same shape as `rhs`.
  """
  defmacro assert_shape!(lhs, rhs) do
    quote do
      Nx.Defn.Kernel.transform(
        {unquote(lhs), unquote(rhs)},
        fn
          {lhs, rhs} when is_tuple(lhs) and is_tuple(rhs) and lhs == rhs -> :ok
          {lhs, rhs} when is_tuple(lhs) and lhs == rhs.shape -> :ok
          {lhs, rhs} when is_tuple(rhs) and lhs.shape == rhs -> :ok
          {lhs, rhs} when lhs.shape == rhs.shape -> :ok
          {lhs, rhs} ->
            raise ArgumentError, "expected input shapes to be equal," <>
                                 " got #{inspect(lhs)} != #{inspect(rhs)}"
        end
      )
    end
  end

  @doc """
  Asserts `lhs` has same rank as `rhs`.
  """
  defmacro assert_rank!(lhs, rhs) do
    quote do
      Nx.Defn.Kernel.transform(
        {unquote(lhs), unquote(rhs)},
        fn
          {x, y} when is_integer(x) and is_integer(y) and x == y -> :ok
          {x, y} when is_tuple(x) and is_tuple(y) and tuple_size(x) == tuple_size(y) -> :ok
          {x, y} when is_integer(x) and is_tuple(y) and x == tuple_size(y) -> :ok
          {x, y} when is_tuple(x) and is_tuple(y) and tuple_size(x) == y -> :ok
          {x, y} when tuple_size(x.shape) == tuple_size(y.shape) -> :ok
          {x, y} ->
            raise ArgumentError, "expected input shapes to have equal rank," <>
                                 " got #{inspect(x)} and #{inspect(y)}"
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
