defmodule Axon.Initializers do
  @moduledoc """
  Common parameter initializers.
  """

  # TODO: These should all be defn
  # TODO: Add random keys

  import Nx.Defn

  @doc """
  Initializes parameters to 0.

  ## Examples

      iex> Axon.Initializers.zeros(shape: {2, 2})
      #Nx.Tensor<
        f32[2][2]
        [
          [0.0, 0.0],
          [0.0, 0.0]
        ]
      >
  """
  defn zeros(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.broadcast(Nx.tensor(0, type: opts[:type]), opts[:shape])
  end

  @doc """
  Initializes parameters to 1.

  ## Examples

      iex> Axon.Initializers.ones(shape: {2, 2})
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 1.0],
          [1.0, 1.0]
        ]
      >
  """
  defn ones(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.broadcast(Nx.tensor(1, type: opts[:type]), opts[:shape])
  end

  @doc """
  Initializes parameters to an identity matrix.

  ## Examples

      iex> Axon.Initializers.identity(shape: {2, 2})
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 0.0],
          [0.0, 1.0]
        ]
      >
  """
  defn identity(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    transform(opts[:shape], &assert_rank(&1, 2))

    Nx.as_type(
      Nx.equal(Nx.iota(opts[:shape], axis: 0), Nx.iota(opts[:shape], axis: 1)),
      opts[:type]
    )
  end

  @doc """
  Initializes parameters with a random uniform distribution.

  ## Examples

      iex> t = Axon.Initializers.uniform(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}
  """
  defn uniform(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0e-2])
    Nx.random_uniform(opts[:shape], type: opts[:type]) * opts[:scale]
  end

  @doc """
  Initializes parameters with a random normal distribution.

  ## Examples

      iex> t = Axon.Initializers.normal(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}
  """
  defn normal(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0e-2, mean: 0.0])
    Nx.random_normal(opts[:shape], opts[:mean], opts[:scale], type: opts[:type])
  end

  @doc """
  Initializes parameters with a [Lecun uniform](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) initializer.
  """
  def lecun_uniform(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    scale = opts[:scale] || 1.0
    variance_scaling(shape, type: type, scale: scale, mode: :fan_in, distribution: :uniform)
  end

  @doc """
  Initializers parameters with a [Lecun normal](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) initializer.
  """
  def lecun_normal(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    scale = opts[:scale] || 1.0

    variance_scaling(shape,
      type: type,
      scale: scale,
      mode: :fan_in,
      distribution: :truncated_normal
    )
  end

  @doc """
  Initializes parameters with a [Glorot uniform](http://proceedings.mlr.press/v9/glorot10a.html) initializer.
  """
  def glorot_uniform(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    scale = opts[:scale] || 1.0
    variance_scaling(shape, type: type, scale: scale, mode: :fan_avg, distribution: :uniform)
  end

  @doc """
  Initializes parameters with a [Glorot normal](http://proceedings.mlr.press/v9/glorot10a.html) initializer.
  """
  def glorot_normal(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    scale = opts[:scale] || 1.0

    variance_scaling(shape,
      type: type,
      scale: scale,
      mode: :fan_avg,
      distribution: :truncated_normal
    )
  end

  @doc """
  Initializes parameters with a [He uniform](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) initializer.
  """
  def he_uniform(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    scale = opts[:scale] || 2.0
    variance_scaling(shape, type: type, scale: scale, mode: :fan_in, distribution: :uniform)
  end

  @doc """
  Initializes parameters with a [He normal](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) initializer.
  """
  def he_normal(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    scale = opts[:scale] || 2.0

    variance_scaling(shape,
      type: type,
      scale: scale,
      mode: :fan_in,
      distribution: :truncated_normal
    )
  end

  @doc """
  Initializes parameters with variance scaling according to
  the given distribution and mode.
  """
  def variance_scaling(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    scale = opts[:scale] || 1.0e-2
    mode = opts[:mode] || :fan_in
    distribution = opts[:distribution] || :normal

    {fan_in, fan_out} = compute_fans(shape)

    denominator =
      case mode do
        :fan_in -> fan_in
        :fan_out -> fan_out
        :fan_avg -> (fan_in + fan_out) / 2.0
      end

    variance = Nx.divide(Nx.tensor(scale, type: type), Nx.tensor(denominator, type: type))

    case distribution do
      :normal ->
        Nx.random_normal(shape, type: type) * Nx.sqrt(variance)

      :uniform ->
        Nx.random_uniform(shape, type: type) * Nx.sqrt(Nx.multiply(3.0, variance))

      :truncated_normal ->
        stddev = Nx.divide(Nx.sqrt(variance), Nx.tensor(0.87962566103423978, type: type))
        Nx.multiply(Nx.clip(Nx.random_normal(shape, type: type), -2.0, 2.0), stddev)
    end
  end

  # Helpers

  defp assert_rank(shape, rank) do
    n = tuple_size(shape)

    unless n == rank,
      do:
        raise(
          ArgumentError,
          "invalid rank for shape #{inspect(shape)}" <>
            " expected shape to have rank #{rank}, got" <>
            " #{n}"
        )
  end

  defp compute_fans(shape) do
    receptive_field_size =
      tuple_product(shape, tuple_size(shape)) / elem(shape, 0) / elem(shape, 1)

    fan_in = elem(shape, 0) * receptive_field_size
    fan_out = elem(shape, 1) * receptive_field_size
    {fan_in, fan_out}
  end

  # TODO: Replace with Tuple.product on Elixir v1.12
  defp tuple_product(_tuple, 0), do: 1
  defp tuple_product(tuple, i), do: :erlang.element(i, tuple) * tuple_product(tuple, i - 1)
end
