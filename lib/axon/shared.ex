defmodule Axon.Shared do
  @moduledoc false

  # Collection of private helper functions and
  # macros for enforcing shape/type constraints,
  # doing shape calculations, and even some
  # helper numerical definitions.

  import Nx.Defn

  @doc """
  Asserts `lhs` has same shape as `rhs`.
  """
  defn assert_shape!(lhs, rhs) do
    transform(
      {lhs, rhs},
      fn {lhs, rhs} ->
        lhs = Nx.shape(lhs)
        rhs = Nx.shape(rhs)

        unless Elixir.Kernel.==(lhs, rhs) do
          raise ArgumentError,
                "expected input shapes to be equal," <>
                  " got #{inspect(lhs)} != #{inspect(rhs)}"
        end
      end
    )
  end

  @doc """
  Asserts `lhs` has same rank as `rhs`.
  """
  defn assert_equal_rank!(lhs, rhs) do
    transform(
      {lhs, rhs},
      fn {x, y} ->
        x = if is_integer(x), do: x, else: Nx.rank(x)
        y = if is_integer(y), do: y, else: Nx.rank(y)

        unless Elixir.Kernel.>=(x, y) do
          raise ArgumentError, "expected input ranks to be equal, got #{x} != #{y}"
        end
      end
    )
  end

  @doc """
  Asserts `lhs` has at least rank `rhs`.
  """
  defn assert_greater_equal_rank!(lhs, rhs) do
    transform(
      {lhs, rhs},
      fn {x, y} ->
        x = if is_integer(x), do: x, else: Nx.rank(x)
        y = if is_integer(y), do: y, else: Nx.rank(y)

        unless Elixir.Kernel.>=(x, y) do
          raise ArgumentError, "expected input shape to have at least rank #{y}, got rank #{x}"
        end
      end
    )
  end

  @doc """
  Transforms the given Elixir value into a scalar predicate.
  """
  defn to_predicate(term) do
    transform(term, fn term -> if term, do: 1, else: 0 end)
  end

  @doc """
  Creates a zeros-like structure which matches the structure
  of the input.
  """
  defn zeros_like(params) do
    transform(
      params,
      &deep_new(&1, fn x -> Axon.Initializers.zeros(shape: Nx.shape(x)) end)
    )
  end

  @doc """
  Creates a fulls-like tuple of inputs.
  """
  defn fulls_like(params, value) do
    transform(
      params,
      &deep_new(&1, fn x -> Axon.Initializers.full(value, shape: Nx.shape(x)) end)
    )
  end

  @doc """
  Deep merges two possibly nested maps, applying fun to leaf values.
  """
  def deep_merge(left, right, fun) do
    case Nx.Container.traverse(left, leaves(right), &recur_merge(&1, &2, fun)) do
      {merged, []} ->
        merged

      {_merged, _leftover} ->
        raise ArgumentError,
              "unable to merge arguments with incompatible" <>
                " structure"
    end
  end

  defp leaves(container) do
    container
    |> Nx.Container.reduce([], fn x, acc -> [x | acc] end)
    |> Enum.reverse()
  end

  defp recur_merge(left, [right | right_leaves], fun) do
    case {left, right} do
      {%Nx.Tensor{} = left, %Nx.Tensor{} = right} ->
        {fun.(left, right), right_leaves}

      {left, right} ->
        {deep_merge(left, right, fun), right_leaves}
    end
  end

  @doc """
  Creates a new map-like structure from a possible nested map, applying `fun`
  to each leaf.
  """
  def deep_new(map, fun) do
    {cont, :ok} = Nx.Container.traverse(map, :ok, &recur_traverse(&1, &2, fun))
    cont
  end

  defp recur_traverse(item, :ok, fun) do
    case item do
      %Nx.Tensor{} = t ->
        {fun.(t), :ok}

      container ->
        {deep_new(container, fun), :ok}
    end
  end

  @doc """
  Deep reduces a map with an accumulator.
  """
  def deep_reduce(map, acc, fun) do
    Nx.Container.reduce(map, acc, &recur_deep_reduce(&1, &2, fun))
  end

  defp recur_deep_reduce(value, acc, fun) do
    case value do
      %Axon{} = val ->
        fun.(val, acc)

      %Nx.Tensor{} = val ->
        fun.(val, acc)

      val ->
        deep_reduce(val, acc, fun)
    end
  end

  @doc """
  Deep map-reduce a nested container with an accumulator.
  """
  def deep_map_reduce(container, acc, fun) do
    Nx.Container.traverse(container, acc, &recur_deep_map_reduce(&1, &2, fun))
  end

  defp recur_deep_map_reduce(leaf, acc, fun) do
    case leaf do
      %Axon{} = leaf ->
        fun.(leaf, acc)

      %Nx.Tensor{} = leaf ->
        fun.(leaf, acc)

      container ->
        deep_map_reduce(container, acc, fun)
    end
  end

  @doc """
  JIT given function with args and opts or apply it inside defn.
  """
  def jit_or_apply(caller, fun, args, opts \\ []) do
    if Nx.Defn.Compiler.current() do
      if opts != [] do
        raise ArgumentError,
              "cannot pass execution options to Axon.#{caller} inside defn, got: #{inspect(opts)}"
      end

      apply(fun, args)
    else
      Nx.Defn.jit(fun, args, opts)
    end
  end

  ## Numerical Helpers

  # TODO: These should be contained somewhere else, like another library

  defn logsumexp(x, opts \\ []) do
    opts = keyword!(opts, axes: [], keep_axes: false)

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

  defn reciprocal(x), do: Nx.divide(1, x)

  defn normalize(input, mean, variance, gamma, bias, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-6)

    scale =
      variance
      |> Nx.add(opts[:epsilon])
      |> Nx.rsqrt()
      |> Nx.multiply(gamma)

    input
    |> Nx.subtract(mean)
    |> Nx.multiply(scale)
    |> Nx.add(bias)
  end

  defn mean_and_variance(input, opts \\ []) do
    opts = keyword!(opts, [:axes])
    mean = Nx.mean(input, axes: opts[:axes], keep_axes: true)
    mean_of_squares = Nx.mean(input * input, axes: opts[:axes], keep_axes: true)
    {mean, mean_of_squares - mean * mean}
  end
end
