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
  deftransform assert_shape!(caller, lhs_name, lhs, rhs_name, rhs) do
    lhs = Nx.shape(lhs)
    rhs = Nx.shape(rhs)

    unless lhs == rhs do
      raise ArgumentError,
            "#{caller}: expected input shapes #{lhs_name} and #{rhs_name}" <>
              " to be equal, got #{inspect(lhs)} != #{inspect(rhs)}"
    end
  end

  @doc """
  Asserts all shapes are equal.
  """
  deftransform assert_shape!(caller, shape_names, [shape | shapes]) do
    equal? =
      Enum.all?(shapes, fn cur_shape ->
        Nx.shape(cur_shape) == Nx.shape(shape)
      end)

    unless equal? do
      raise ArgumentError,
            "#{caller}: expected all input shapes #{inspect(shape_names)}" <>
              " to be equal, got #{inspect(shapes)}"
    end
  end

  @doc """
  Asserts `inp` has explicit rank `rank`.
  """
  deftransform assert_rank!(caller, inp_name, inp, rank) do
    x = Nx.rank(inp)

    unless x == rank do
      raise ArgumentError,
            "#{caller}: expected #{inp_name} to have rank equal to #{rank}," <>
              " got #{x} != #{rank}"
    end
  end

  @doc """
  Asserts `lhs` has same rank as `rhs`.
  """
  deftransform assert_equal_rank!(caller, lhs_name, lhs, rhs_name, rhs) do
    x = if is_integer(lhs), do: lhs, else: Nx.rank(lhs)
    y = if is_integer(rhs), do: rhs, else: Nx.rank(rhs)

    unless x >= y do
      raise ArgumentError,
            "#{caller}: expected #{lhs_name} and #{rhs_name} ranks to be equal" <>
              " got #{x} != #{y}"
    end
  end

  @doc """
  Asserts all ranks are equal.
  """
  deftransform assert_equal_rank!(caller, rank_names, [rank | ranks]) do
    equal? =
      Enum.all?(ranks, fn cur_rank ->
        Nx.rank(cur_rank) == Nx.rank(rank)
      end)

    unless equal? do
      raise ArgumentError,
            "#{caller}: expected all input ranks #{inspect(rank_names)}" <>
              " to be equal, got #{inspect(ranks)}"
    end
  end

  @doc """
  Asserts `lhs` has at least rank `rhs`.
  """
  deftransform assert_min_rank!(caller, name, lhs, rhs) do
    x = if is_integer(lhs), do: lhs, else: Nx.rank(lhs)
    y = if is_integer(rhs), do: rhs, else: Nx.rank(rhs)

    unless x >= y do
      raise ArgumentError,
            "#{caller}: expected #{name} shape to have at least rank #{y}, got rank #{x}"
    end
  end

  @doc """
  Creates a zeros-like structure which matches the structure
  of the input.
  """
  deftransform zeros_like(params, opts \\ []) do
    opts = Keyword.validate!(opts, [:type])
    fun = Axon.Initializers.zeros()

    deep_new(params, fn x ->
      type = opts[:type] || Nx.type(x)
      fun.(Nx.shape(x), type)
    end)
  end

  @doc """
  Creates a fulls-like tuple of inputs.
  """
  deftransform fulls_like(params, value, opts \\ []) do
    opts = Keyword.validate!(opts, [:type])
    fun = Axon.Initializers.full(value)

    deep_new(params, fn x ->
      type = opts[:type] || Nx.type(x)
      fun.(Nx.shape(x), type)
    end)
  end

  @doc """
  Deep merges two possibly nested maps, applying fun to leaf values.
  """
  deftransform deep_merge(left, right, fun) do
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
  deftransform deep_new(item, fun) when is_integer(item) do
    fun.(item)
  end

  deftransform deep_new(%Nx.Tensor{} = item, fun) do
    fun.(item)
  end

  deftransform deep_new(map, fun) do
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
  deftransform deep_reduce(item, acc, fun) when is_integer(item) do
    fun.(item, acc)
  end

  deftransform deep_reduce(map, acc, fun) do
    Nx.Container.reduce(map, acc, &recur_deep_reduce(&1, &2, fun))
  end

  defp recur_deep_reduce(value, acc, fun) do
    case value do
      %Nx.Tensor{} = val ->
        fun.(val, acc)

      %{axon: :axon} = val ->
        fun.(val, acc)

      val ->
        deep_reduce(val, acc, fun)
    end
  end

  @doc """
  Deep map-reduce a nested container with an accumulator.
  """
  deftransform deep_map_reduce(leaf, acc, fun) when is_integer(leaf), do: fun.(leaf, acc)

  deftransform deep_map_reduce(container, acc, fun) do
    Nx.Container.traverse(container, acc, &recur_deep_map_reduce(&1, &2, fun))
  end

  defp recur_deep_map_reduce(leaf, acc, fun) do
    case leaf do
      %Nx.Tensor{} = leaf ->
        fun.(leaf, acc)

      container ->
        deep_map_reduce(container, acc, fun)
    end
  end

  ## List transforms in defn

  deftransform list_duplicate(value, size) do
    List.duplicate(value, size)
  end

  deftransform list_wrap(value), do: List.wrap(value)

  ## Numerical Helpers

  # TODO: These should be contained somewhere else, like another library

  defn xlogy(x, y) do
    x_ok = Nx.not_equal(x, 0.0)
    safe_x = Nx.select(x_ok, x, Nx.tensor(1, type: Nx.type(x)))
    safe_y = Nx.select(x_ok, y, Nx.tensor(1, type: Nx.type(y)))
    Nx.select(x_ok, safe_x * Nx.log(safe_y), Nx.tensor(0, type: Nx.type(x)))
  end

  defn reciprocal(x), do: Nx.divide(1, x)

  defn normalize(input, mean, variance, gamma, bias, opts \\ []) do
    [epsilon: epsilon] = keyword!(opts, epsilon: 1.0e-6)
    scale = gamma * Nx.rsqrt(variance + epsilon)
    scale * (input - mean) + bias
  end

  defn mean_and_variance(input, opts \\ []) do
    opts = keyword!(opts, [:axes])
    mean = Nx.mean(input, axes: opts[:axes], keep_axes: true)
    var = Nx.variance(input, axes: opts[:axes], keep_axes: true)
    {mean, var}
  end
end
