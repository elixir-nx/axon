defmodule Axon.Initializers do
  @moduledoc """
  Common parameter initializers.
  """

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
    Nx.eye(opts[:shape], type: opts[:type])
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
  defn lecun_uniform(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0])
    variance_scaling(
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_in,
      distribution: :uniform
    )
  end

  @doc """
  Initializers parameters with a [Lecun normal](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) initializer.
  """
  defn lecun_normal(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0])
    variance_scaling(
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_in,
      distribution: :truncated_normal
    )
  end

  @doc """
  Initializes parameters with a [Glorot uniform](http://proceedings.mlr.press/v9/glorot10a.html) initializer.
  """
  defn glorot_uniform(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0])
    variance_scaling(
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_avg,
      distribution: :uniform
    )
  end

  @doc """
  Initializes parameters with a [Glorot normal](http://proceedings.mlr.press/v9/glorot10a.html) initializer.
  """
  defn glorot_normal(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0])
    variance_scaling(
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_avg,
      distribution: :truncated_normal
    )
  end

  @doc """
  Initializes parameters with a [He uniform](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) initializer.
  """
  defn he_uniform(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 2.0])
    variance_scaling(
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_in,
      distribution: :uniform
    )
  end

  @doc """
  Initializes parameters with a [He normal](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html) initializer.
  """
  defn he_normal(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 2.0])
    variance_scaling(
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_in,
      distribution: :truncated_normal
    )
  end

  @doc """
  Initializes parameters with variance scaling according to
  the given distribution and mode.
  """
  defn variance_scaling(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0e-2, mode: :fan_in, distribution: :normal])

    fans = transform(opts[:shape], &compute_fans/1)
    denominator = transform({fans, opts[:mode]},
      fn
        {{fan_in, _}, :fan_in} -> fan_in
        {{_, fan_out}, :fan_out} -> fan_out
        {{fan_in, fan_out}, :fan_avg} -> (fan_in + fan_out) / 2.0
      end)

    variance = Nx.divide(Nx.tensor(opts[:scale], type: opts[:type]), denominator)

    var_opts = transform(opts, &Keyword.take(&1, [:shape, :type]))

    transform({opts[:distribution], variance, var_opts},
      fn
        {:normal, variance, opts} ->
          var_normal(variance, opts)
        {:uniform, variance, opts} ->
          var_uniform(variance, opts)
        {:truncated_normal, variance, opts} ->
          var_uniform(variance, opts)
        {dist, _, _} ->
          raise ArgumentError, "invalid distribution #{inspect(dist)} passed to variance_scaling/1"
      end
    )
  end

  # Variance scaling branches
  defnp var_normal(variance, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.random_normal(opts[:shape], type: opts[:type]) * Nx.sqrt(variance)
  end

  defnp var_uniform(variance, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.random_uniform(opts[:shape], type: opts[:type]) * Nx.sqrt(Nx.multiply(3.0, variance))
  end

  defnp var_truncated(variance, opts \\ []) do
    stddev = Nx.divide(Nx.sqrt(variance), 0.87962566103423978)
    Nx.multiply(Nx.clip(Nx.random_normal(opts[:shape], type: opts[:type]), -2.0, 2.0), stddev)
  end

  # Helpers

  defp compute_fans(shape) do
    receptive_field_size =
      Nx.size(shape) / elem(shape, 0) / elem(shape, 1)

    fan_in = elem(shape, 0) * receptive_field_size
    fan_out = elem(shape, 1) * receptive_field_size
    {fan_in, fan_out}
  end
end
