defmodule Axon.Shared do
  @moduledoc false

  # Collection of private helper functions and
  # macros for enforcing shape/type constraints,
  # doing shape calculations, and even some
  # helper numerical definitions.

  import Nx.Defn

  @doc """
  Returns the size of the given axis.
  """
  defmacro axis_size(shape_or_tensor, axis) do
    quote do
      Nx.Defn.Kernel.transform(
        {unquote(shape_or_tensor), unquote(axis)},
        fn
          {x, axis} when is_tuple(x) and is_integer(axis) ->
            axis = if axis < 0, do: tuple_size(x) - axis, else: axis
            elem(x, axis)
          {x, axis} when is_integer(axis) ->
            axis = if axis < 0, do: tuple_size(Nx.shape(x)) - axis, else: axis
            elem(Nx.shape(x), axis)
          _ ->
            raise ArgumentError, "input axis must be an integer"
        end
      )
    end
  end

  @doc """
  Inverts the give permutation. Used in certain
  shape determining transpose permuation.
  """
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
        fn {lhs, rhs} ->
          lhs = Nx.shape(lhs)
          rhs = Nx.shape(rhs)
          unless lhs == rhs do
            raise ArgumentError, "expected input shapes to be equal," <>
                                 " got #{inspect(lhs)} != #{inspect(rhs)}"
          end
        end
      )
    end
  end

  @doc """
  Asserts `lhs` has same rank as `rhs`.
  """
  defmacro assert_equal_rank!(lhs, rhs) do
    quote do
      Nx.Defn.Kernel.transform(
        {unquote(lhs), unquote(rhs)},
        fn {x, y} ->
            x = if is_integer(x), do: x, else: Nx.rank(x)
            y = if is_integer(y), do: y, else: Nx.rank(y)
            unless x >= y do
              raise ArgumentError, "expected input ranks to be equal," <>
                                   " got #{x} != #{y}"
            end
        end
      )
    end
  end

  @doc """
  Asserts `lhs` has at least rank `rhs`.
  """
  defmacro assert_greater_equal_rank!(lhs, rhs) do
    quote do
      Nx.Defn.Kernel.transform(
        {unquote(lhs), unquote(rhs)},
        fn {x, y} ->
            x = if is_integer(x), do: x, else: Nx.rank(x)
            y = if is_integer(y), do: y, else: Nx.rank(y)
            unless x >= y do
              raise ArgumentError, "expected input shape to have at least rank #{y}," <>
                                   " got rank #{x}"
            end
        end
      )
    end
  end

  @doc """
  Transforms the given Elixir value into a scalar predicate.
  """
  defmacro to_predicate(term) do
    quote do
      Nx.Defn.Kernel.transform(
        unquote(term),
        fn term -> if term, do: 1, else: 0 end
      )
    end
  end

  ## Numerical Helpers

  # TODO: These should be contained somewhere else

  defn logsumexp(x, opts \\ []) do
    opts = keyword!(opts, [axes: [], keep_axes: false])
    x
    |> Nx.exp()
    |> Nx.sum(opts)
    |> Nx.log()
  end

  defn xlogy(x, y) do
    x_ok = Nx.not_equal(x, 0.0)
    safe_x = Nx.select(x_ok, x, Nx.tensor(1, type: Nx.type(x)))
    safe_y = Nx.select(x_ok, y, Nx.tensor(1, type: Nx.type(y)))
    Nx.select(x_ok, safe_x * Nx.log(safe_y), Nx.tensor(0, type: Nx.type(x)))
  end

end
