defmodule Axon.Initializers do
  @moduledoc """
  Parameter initializers.

  Parameter initializers are used to initialize the weights
  and biases of a neural network. Because most deep learning
  optimization algorithms are iterative, they require an initial
  point to iterate from.

  Sometimes the initialization of a model can determine whether
  or not a model converges. In some cases, the initial point is
  unstable, and therefore the model has no chance of converging
  using common first-order optimization methods. In cases where
  the model will converge, initialization can have a significant
  impact on how quickly the model converges.

  Most initialization strategies are built from intuition and
  heuristics rather than theory. It's commonly accepted that
  the parameters of different layers should be different -
  motivating the use of random initialization for each layer's
  parameters. Usually, only the weights of a layer are initialized
  using a random distribution - while the biases are initialized
  to a uniform constant (like 0).

  Most initializers use Gaussian (normal) or uniform distributions
  with variations on scale. The output scale of an initializer
  should generally be large enough to avoid information loss but
  small enough to avoid exploding values. The initializers in
  this module have a default scale known to work well with
  the initialization strategy.

  The functions in this module return initialization functions which
  take shapes and types and return tensors:

      init_fn = Axon.Initializers.zeros()
      init_fn.({1, 2}, {:f, 32})

  You may use these functions from within `defn` or outside.
  """

  import Nx.Defn
  import Axon.Shared

  @doc """
  Initializes parameters to 0.

  ## Examples

      iex> init_fn = Axon.Initializers.zeros()
      iex> out = init_fn.({2, 2}, {:f, 32})
      iex> out
      #Nx.Tensor<
        f32[2][2]
        [
          [0.0, 0.0],
          [0.0, 0.0]
        ]
      >
  """
  def zeros() do
    fn shape, type ->
      zeros_impl(shape: shape, type: type)
    end
  end

  defnp zeros_impl(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.broadcast(Nx.tensor(0, type: opts[:type]), opts[:shape])
  end

  @doc """
  Initializes parameters to 1.

  ## Examples

      iex> init_fn = Axon.Initializers.ones()
      iex> out = init_fn.({2, 2}, {:f, 32})
      iex> out
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 1.0],
          [1.0, 1.0]
        ]
      >
  """
  def ones() do
    fn shape, type ->
      ones_impl(shape: shape, type: type)
    end
  end

  defnp ones_impl(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.broadcast(Nx.tensor(1, type: opts[:type]), opts[:shape])
  end

  @doc """
  Initializes parameters to value.

  ## Examples

      iex> init_fn = Axon.Initializers.full(1.00)
      iex> out = init_fn.({2, 2}, {:f, 32})
      iex> out
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 1.0],
          [1.0, 1.0]
        ]
      >
  """
  def full(value) do
    fn shape, type ->
      full_impl(value, shape: shape, type: type)
    end
  end

  defnp full_impl(value, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.as_type(Nx.broadcast(value, opts[:shape]), opts[:type])
  end

  @doc """
  Initializes parameters to an identity matrix.

  ## Examples

      iex> init_fn = Axon.Initializers.identity()
      iex> out = init_fn.({2, 2}, {:f, 32})
      iex> out
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 0.0],
          [0.0, 1.0]
        ]
      >
  """
  def identity() do
    fn shape, type ->
      identity_impl(shape: shape, type: type)
    end
  end

  defnp identity_impl(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.eye(opts[:shape], type: opts[:type])
  end

  @doc """
  Initializes parameters with a random uniform distribution.

  ## Options

    * `:scale` - scale of the output distribution. Defaults to `1.0e-2`

  ## Examples

      iex> init_fn = Axon.Initializers.uniform()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.uniform(scale: 1.0e-3)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  """
  def uniform(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 1.0e-2
      uniform_impl(key, shape: shape, type: type, scale: scale)
    end
  end

  defnp uniform_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0e-2])
    shape = Nx.shape(opts[:shape])

    Nx.Random.uniform_split(key, Nx.negate(opts[:scale]), opts[:scale],
      type: opts[:type],
      shape: shape
    )
  end

  @doc """
  Initializes parameters with a random normal distribution.

  ## Options

    * `:mean` - mean of the output distribution. Defaults to `0.0`
    * `:scale` - scale of the output distribution. Defaults to `1.0e-2`

  ## Examples

      iex> init_fn = Axon.Initializers.normal()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.normal(mean: 1.0, scale: 1.0)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  """
  def normal(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 1.0e-2
      mean = opts[:mean] || 0.0
      normal_impl(key, shape: shape, type: type, scale: scale, mean: mean)
    end
  end

  defnp normal_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0e-2, mean: 0.0])
    Nx.Random.normal_split(key, opts[:mean], opts[:scale], shape: opts[:shape], type: opts[:type])
  end

  @doc """
  Initializes parameters with the Lecun uniform initializer.

  The Lecun uniform initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_in`
  and `distribution: :uniform`.

  ## Options

    * `:scale` - scale of the output distribution. Defaults to `1.0`

  ## Examples

      iex> init_fn = Axon.Initializers.lecun_uniform()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.lecun_uniform(scale: 1.0e-3)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ## References

    * [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

  """
  def lecun_uniform(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 1.0
      lecun_uniform_impl(key, shape: shape, type: type, scale: scale)
    end
  end

  defnp lecun_uniform_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0])

    variance_scaling_impl(
      key,
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_in,
      distribution: :uniform
    )
  end

  @doc """
  Initializes parameters with the Lecun normal initializer.

  The Lecun normal initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_in`
  and `distribution: :truncated_normal`.

  ## Options

    * `:scale` - scale of the output distribution. Defaults to `1.0`

  ## Examples

      iex> init_fn = Axon.Initializers.lecun_normal()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.lecun_normal(scale: 1.0e-3)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ## References

    * [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

  """
  def lecun_normal(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 1.0
      lecun_normal_impl(key, shape: shape, type: type, scale: scale)
    end
  end

  defnp lecun_normal_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0])

    variance_scaling_impl(
      key,
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_in,
      distribution: :truncated_normal
    )
  end

  @doc """
  Initializes parameters with the Glorot uniform initializer.

  The Glorot uniform initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_avg`
  and `distribution: :uniform`.

  The Glorot uniform initializer is also called the Xavier
  uniform initializer.

  ## Options

    * `:scale` - scale of the output distribution. Defaults to `1.0`

  ## Examples

      iex> init_fn = Axon.Initializers.glorot_uniform()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.glorot_uniform(scale: 1.0e-3)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ## References

    * [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)

  """
  def glorot_uniform(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 1.0
      glorot_uniform_impl(key, shape: shape, type: type, scale: scale)
    end
  end

  defnp glorot_uniform_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0])

    variance_scaling_impl(
      key,
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_avg,
      distribution: :uniform
    )
  end

  @doc """
  Initializes parameters with the Glorot normal initializer.

  The Glorot normal initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_avg`
  and `distribution: :truncated_normal`.

  The Glorot normal initializer is also called the Xavier
  normal initializer.

  ## Options

    * `:scale` - scale of the output distribution. Defaults to `1.0`

  ## Examples

      iex> init_fn = Axon.Initializers.glorot_normal()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.glorot_normal(scale: 1.0e-3)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ## References

    * [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)

  """
  def glorot_normal(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 1.0
      glorot_normal_impl(key, shape: shape, type: type, scale: scale)
    end
  end

  defnp glorot_normal_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0])

    variance_scaling_impl(
      key,
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_avg,
      distribution: :truncated_normal
    )
  end

  @doc """
  Initializes parameters with the He uniform initializer.

  The He uniform initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_ni`
  and `distribution: :uniform`.

  ## Options

    * `:scale` - scale of the output distribution. Defaults to `2.0`

  ## Examples

      iex> init_fn = Axon.Initializers.he_uniform()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.he_uniform(scale: 1.0e-3)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ## References

    * [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)

  """
  def he_uniform(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 2.0
      he_uniform_impl(key, shape: shape, type: type, scale: scale)
    end
  end

  defnp he_uniform_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 2.0])

    variance_scaling_impl(
      key,
      shape: opts[:shape],
      type: opts[:type],
      scale: opts[:scale],
      mode: :fan_in,
      distribution: :uniform
    )
  end

  @doc """
  Initializes parameters with the He normal initializer.

  The He normal initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_in`
  and `distribution: :truncated_normal`.

  ## Options

    * `:scale` - scale of the output distribution. Defaults to `2.0`

  ## Examples

      iex> init_fn = Axon.Initializers.he_normal()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.he_normal(scale: 1.0e-3)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ## References

    * [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)

  """
  def he_normal(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 2.0
      he_normal_impl(key, shape: shape, type: type, scale: scale)
    end
  end

  defnp he_normal_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 2.0])

    variance_scaling_impl(
      key,
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

  Variance scaling adapts scale to the weights of the output
  tensor.

  ## Options

    * `:scale` - scale of the output distribution. Defaults to `1.0e-2`
    * `:mode` - compute fan mode. One of `:fan_in`, `:fan_out`, or `:fan_avg`.
      Defaults to `:fan_in`
    * `:distribution` - output distribution. One of `:normal`, `:truncated_normal`,
      or `:uniform`. Defaults to `:normal`

  ## Examples

      iex> init_fn = Axon.Initializers.variance_scaling()
      iex> t = init_fn.({2, 2}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> init_fn = Axon.Initializers.variance_scaling(mode: :fan_out, distribution: :truncated_normal)
      iex> t = init_fn.({2, 2}, {:bf, 16}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

      iex> init_fn = Axon.Initializers.variance_scaling(mode: :fan_out, distribution: :normal)
      iex> t = init_fn.({64, 3, 32, 32}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.shape(t)
      {64, 3, 32, 32}
      iex> Nx.type(t)
      {:f, 32}

  """
  def variance_scaling(opts \\ []) do
    fn shape, type, key ->
      scale = opts[:scale] || 1.0
      mode = opts[:mode] || :fan_in
      distribution = opts[:distribution] || :normal

      variance_scaling_impl(
        key,
        shape: shape,
        type: type,
        scale: scale,
        mode: mode,
        distribution: distribution
      )
    end
  end

  defnp variance_scaling_impl(key, opts \\ []) do
    opts =
      keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0, mode: :fan_in, distribution: :normal])

    fans = compute_fans(opts[:shape])
    denominator = compute_denominator(fans, opts[:mode])

    variance = Nx.divide(Nx.tensor(opts[:scale], type: opts[:type]), Nx.max(denominator, 1.0))

    apply_distribution(key, opts[:distribution], variance, shape: opts[:shape], type: opts[:type])
  end

  deftransformp compute_fans(shape) do
    rank = Nx.rank(shape)

    {in_size, out_size} =
      cond do
        rank < 1 ->
          {1, 1}

        rank == 1 ->
          {elem(shape, 0), elem(shape, 0)}

        rank == 2 ->
          {elem(shape, 0), elem(shape, 1)}

        true ->
          {elem(shape, rank - 2), elem(shape, rank - 1)}
      end

    receptive_field_size = Nx.size(shape) / in_size / out_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size

    {fan_in, fan_out}
  end

  deftransformp compute_denominator(fans, mode) do
    case {fans, mode} do
      {{fan_in, _}, :fan_in} ->
        fan_in

      {{_, fan_out}, :fan_out} ->
        fan_out

      {{fan_in, fan_out}, :fan_avg} ->
        (fan_in + fan_out) / 2.0

      {{_, _}, mode} ->
        raise ArgumentError, "invalid mode #{inspect(mode)} passed to variance_scaling/1"
    end
  end

  deftransformp apply_distribution(key, distribution, variance, opts) do
    case distribution do
      :normal ->
        var_normal(key, variance, opts)

      :uniform ->
        var_uniform(key, variance, opts)

      :truncated_normal ->
        var_truncated(key, variance, opts)

      dist ->
        raise ArgumentError,
              "invalid distribution #{inspect(dist)} passed to variance_scaling/1"
    end
  end

  @doc """
  Initializes a tensor with an orthogonal distribution.

  For 2-D tensors, the initialization is generated through the QR decomposition of a random distribution
  For tensors with more than 2 dimensions, a 2-D tensor with shape `{shape[0] * shape[1] * ... * shape[n-2], shape[n-1]}`
  is initialized and then reshaped accordingly.

  ## Options

    * `:distribution` - output distribution. One of [`:normal`, `:uniform`].
      Defaults to `:normal`

  ## Examples

      iex> init_fn = Axon.Initializers.orthogonal()
      iex> t = init_fn.({3, 3}, {:f, 32}, Nx.Random.key(1))
      iex> Nx.type(t)
      {:f, 32}
      iex> Nx.shape(t)
      {3, 3}

      iex> init_fn = Axon.Initializers.orthogonal()
      iex> t = init_fn.({1, 2, 3, 4}, {:f, 64}, Nx.Random.key(1))
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.shape(t)
      {1, 2, 3, 4}
  """
  def orthogonal(opts \\ []) do
    fn shape, type, key ->
      distribution = opts[:distribution] || :normal
      orthogonal_impl(key, shape: shape, type: type, distribution: distribution)
    end
  end

  defnp orthogonal_impl(key, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, distribution: :normal])

    shape = opts[:shape]
    distribution = opts[:distribution]
    type = opts[:type]

    assert_min_rank!("Axon.Initializers.orthogonal", "input_shape", shape, 2)

    {{m, n}, random_seed} =
      transform({key, shape, distribution, type}, fn {key, shape, distribution, type} ->
        flat_shape =
          if tuple_size(shape) > 2 do
            tuple_list = shape |> Tuple.to_list() |> Enum.reverse()
            n = hd(tuple_list)
            m = Enum.reduce(tl(tuple_list), 1, &(&1 * &2))
            {m, n}
          else
            shape
          end

        out =
          case distribution do
            :uniform ->
              Nx.Random.uniform_split(key, 0.0, 1.0, shape: flat_shape, type: type)

            :normal ->
              Nx.Random.normal_split(key, 0.0, 1.0, shape: flat_shape, type: type)

            dist ->
              raise ArgumentError,
                    "invalid distribution #{inspect(dist)} passed to orthogonal/1"
          end

        {flat_shape, out}
      end)

    {q, _r} = Nx.LinAlg.qr(random_seed, mode: :complete)

    rand =
      q
      |> Nx.slice([0, 0], [m, n])
      |> Nx.reshape(shape)

    rand
  end

  # Variance scaling branches

  defnp var_normal(key, variance, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    shape = opts[:shape]
    type = opts[:type]

    sigma = Nx.sqrt(variance)

    Nx.Random.normal_split(key, 0.0, sigma, shape: shape, type: type)
  end

  defnp var_uniform(key, variance, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    shape = opts[:shape]
    type = opts[:type]

    limit = Nx.sqrt(3 * variance)
    Nx.Random.uniform_split(key, -limit, limit, shape: shape, type: type)
  end

  defnp var_truncated(key, variance, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    shape = opts[:shape]
    type = opts[:type]

    sigma =
      variance
      |> Nx.sqrt()
      |> Nx.divide(0.87962566103423978)
      |> Nx.as_type(type)

    truncated_normal(key, -2, 2, shape: shape, type: type) * sigma
  end

  defnp truncated_normal(key, lower, upper, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    shape = opts[:shape]
    type = opts[:type]

    sqrt2 = Nx.sqrt(2) |> Nx.as_type(type)
    lower = Nx.as_type(lower, type)
    upper = Nx.as_type(upper, type)

    a = Nx.erf(lower / sqrt2)
    b = Nx.erf(upper / sqrt2)

    u = Nx.Random.uniform_split(key, a, b, shape: shape, type: type)
    out = sqrt2 * Nx.erf_inv(u)

    Nx.clip(out, lower, upper)
  end
end
