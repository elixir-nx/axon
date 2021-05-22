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

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  # TODO: Add random keys

  import Nx.Defn
  import Axon.Shared

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
  Initializes parameters to value.

  ## Examples

      iex> Axon.Initializers.full(1.00, shape: {2, 2})
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 1.0],
          [1.0, 1.0]
        ]
      >
  """
  defn full(value, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    Nx.as_type(Nx.broadcast(value, opts[:shape]), opts[:type])
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

  ## Options

    * `:shape` - output shape
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:scale` - scale of the output distribution. Defaults to `1.0e-2`

  ## Examples

      iex> t = Axon.Initializers.uniform(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.uniform(shape: {2, 2}, type: {:bf, 16}, scale: 1.0e-3)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  """
  defn uniform(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0e-2])
    shape = Nx.shape(opts[:shape])
    Nx.random_uniform(shape, Nx.negate(opts[:scale]), opts[:scale], type: opts[:type])
  end

  @doc """
  Initializes parameters with a random normal distribution.

  ## Options

    * `:shape` - output shape
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:mean` - mean of the output distribution. Defaults to `0.0`
    * `:scale` - scale of the output distribution. Defaults to `1.0e-2`

  ## Examples

      iex> t = Axon.Initializers.normal(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.normal(shape: {2, 2}, type: {:bf, 16}, mean: 1.0, scale: 1.0)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  """
  defn normal(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0e-2, mean: 0.0])
    Nx.random_normal(opts[:shape], opts[:mean], opts[:scale], type: opts[:type])
  end

  @doc """
  Initializes parameters with the Lecun uniform initializer.

  The Lecun uniform initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_in`
  and `distribution: :uniform`.

  ## Options

    * `:shape` - output shape. Must be at least rank `2`
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:scale` - scale of the output distribution. Defaults to `1.0`

  ## Examples

      iex> t = Axon.Initializers.lecun_uniform(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.lecun_uniform(shape: {2, 2}, type: {:bf, 16}, scale: 1.0e-3)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ### Error cases

      iex> Axon.Initializers.lecun_uniform(shape: {2})
      ** (ArgumentError) expected input shape to have at least rank 2, got rank 1

  ## References

    * [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

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
  Initializes parameters with the Lecun normal initializer.

  The Lecun normal initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_in`
  and `distribution: :truncated_normal`.

  ## Options

    * `:shape` - output shape. Must be at least rank `2`
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:scale` - scale of the output distribution. Defaults to `1.0`

  ## Examples

      iex> t = Axon.Initializers.lecun_uniform(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.lecun_uniform(shape: {2, 2}, type: {:bf, 16}, scale: 1.0e-3)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ### Error cases

      iex> Axon.Initializers.lecun_normal(shape: {2})
      ** (ArgumentError) expected input shape to have at least rank 2, got rank 1

  ## References

    * [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

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
  Initializes parameters with the Glorot uniform initializer.

  The Glorot uniform initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_avg`
  and `distribution: :uniform`.

  The Glorot uniform initializer is also called the Xavier
  uniform initializer.

  ## Options

    * `:shape` - output shape. Must be at least rank `2`
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:scale` - scale of the output distribution. Defaults to `1.0`

  ## Examples

      iex> t = Axon.Initializers.glorot_uniform(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.glorot_uniform(shape: {2, 2}, type: {:bf, 16}, scale: 1.0e-3)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ### Error cases

      iex> Axon.Initializers.glorot_uniform(shape: {2})
      ** (ArgumentError) expected input shape to have at least rank 2, got rank 1

  ## References

    * [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)

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
  Initializes parameters with the Glorot normal initializer.

  The Glorot normal initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_avg`
  and `distribution: :truncated_normal`.

  The Glorot normal initializer is also called the Xavier
  normal initializer.

  ## Options

    * `:shape` - output shape. Must be at least rank `2`
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:scale` - scale of the output distribution. Defaults to `1.0`

  ## Examples

      iex> t = Axon.Initializers.glorot_normal(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.glorot_normal(shape: {2, 2}, type: {:bf, 16}, scale: 1.0e-3)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ### Error cases

      iex> Axon.Initializers.glorot_normal(shape: {2})
      ** (ArgumentError) expected input shape to have at least rank 2, got rank 1

  ## References

    * [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)

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
  Initializes parameters with the He uniform initializer.

  The He uniform initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_ni`
  and `distribution: :uniform`.

  ## Options

    * `:shape` - output shape. Must be at least rank `2`
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:scale` - scale of the output distribution. Defaults to `2.0`

  ## Examples

      iex> t = Axon.Initializers.he_uniform(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.he_uniform(shape: {2, 2}, type: {:bf, 16}, scale: 1.0e-3)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ### Error cases

      iex> Axon.Initializers.he_uniform(shape: {2})
      ** (ArgumentError) expected input shape to have at least rank 2, got rank 1

  ## References

    * [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)

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
  Initializes parameters with the He normal initializer.

  The He normal initializer is equivalent to calling
  `Axon.Initializers.variance_scaling` with `mode: :fan_ni`
  and `distribution: :truncated_normal`.

  ## Options

    * `:shape` - output shape. Must be at least rank `2`
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:scale` - scale of the output distribution. Defaults to `2.0`

  ## Examples

      iex> t = Axon.Initializers.he_normal(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.he_normal(shape: {2, 2}, type: {:bf, 16}, scale: 1.0e-3)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

  ### Error cases

      iex> Axon.Initializers.he_normal(shape: {2})
      ** (ArgumentError) expected input shape to have at least rank 2, got rank 1

  ## References

    * [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)

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

  Variance scaling adapts scale to the weights of the output
  tensor.

  ## Options

    * `:shape` - output shape. Must be at least rank `2`
    * `:type` - output type. Defaults to `{:f, 32}`
    * `:scale` - scale of the output distribution. Defaults to `1.0e-2`
    * `:mode` - compute fan mode. One of `:fan_in`, `:fan_out`, or `:fan_avg`.
      Defaults to `:fan_in`
    * `:distribution` - output distribution. One of `:normal`, `:truncated_normal`,
      or `:uniform`. Defaults to `:normal`

  ## Examples

      iex> t = Axon.Initializers.variance_scaling(shape: {2, 2})
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:f, 32}

      iex> t = Axon.Initializers.variance_scaling(shape: {2, 2}, type: {:bf, 16}, mode: :fan_out, distribution: :truncated_normal)
      iex> Nx.shape(t)
      {2, 2}
      iex> Nx.type(t)
      {:bf, 16}

      iex> t = Axon.Initializers.variance_scaling(shape: {64, 3, 32, 32}, mode: :fan_out, distribution: :normal)
      iex> Nx.shape(t)
      {64, 3, 32, 32}
      iex> Nx.type(t)
      {:f, 32}

  ### Error cases

      iex> Axon.Initializers.variance_scaling(shape: {2})
      ** (ArgumentError) expected input shape to have at least rank 2, got rank 1

      iex> Axon.Initializers.variance_scaling(shape: {2, 2}, mode: :not_a_mode)
      ** (ArgumentError) invalid mode :not_a_mode passed to variance_scaling/1

      iex> Axon.Initializers.variance_scaling(shape: {2, 2}, distribution: :not_a_dist)
      ** (ArgumentError) invalid distribution :not_a_dist passed to variance_scaling/1

  """
  defn variance_scaling(opts \\ []) do
    opts =
      keyword!(opts, [:shape, type: {:f, 32}, scale: 1.0, mode: :fan_in, distribution: :normal])

    assert_greater_equal_rank!(opts[:shape], 2)

    fans = transform(opts[:shape], &compute_fans/1)

    denominator =
      transform(
        {fans, opts[:mode]},
        fn
          {{fan_in, _}, :fan_in} ->
            fan_in

          {{_, fan_out}, :fan_out} ->
            fan_out

          {{fan_in, fan_out}, :fan_avg} ->
            (fan_in + fan_out) / 2.0

          {{_, _}, mode} ->
            raise ArgumentError, "invalid mode #{inspect(mode)} passed to variance_scaling/1"
        end
      )

    variance = Nx.divide(Nx.tensor(opts[:scale], type: opts[:type]), Nx.max(denominator, 1.0))

    var_opts = transform(opts, &Keyword.take(&1, [:shape, :type]))

    transform(
      {opts[:distribution], variance, var_opts},
      fn
        {:normal, variance, opts} ->
          var_normal(variance, opts)

        {:uniform, variance, opts} ->
          var_uniform(variance, opts)

        {:truncated_normal, variance, opts} ->
          var_uniform(variance, opts)

        {dist, _, _} ->
          raise ArgumentError,
                "invalid distribution #{inspect(dist)} passed to variance_scaling/1"
      end
    )
  end

  @doc """
  Initializes a tensor with an orthogonal distribution.

  For 2-D tensors, the initialization is generated through the QR decomposition of a random distribution
  For tensors with more than 2 dimensions, a 2-D tensor with shape `{shape[0] * shape[1] * ... * shape[n-2], shape[n-1]}`
  is initialized and then reshaped accordingly.

  ## Options

    * `:shape` - output shape. Must be at least rank `2`
    * `:type` - random seed's type. Defaults to `{:f, 32}`
    * `:distribution` - output distribution. One of [`:normal`, `:uniform`]. Defaults to `:normal`

  ## Examples

      iex> t = Axon.Initializers.orthogonal(shape: {3, 3})
      iex> Nx.type(t)
      {:f, 32}
      iex> Nx.shape(t)
      {3, 3}

      iex> t = Axon.Initializers.orthogonal(shape: {1, 2, 3, 4}, type: {:f, 64})
      iex> Nx.type(t)
      {:f, 64}
      iex> Nx.shape(t)
      {1, 2, 3, 4}
  """
  defn orthogonal(opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}, distribution: :normal])

    shape = opts[:shape]
    distribution = opts[:distribution]
    type = opts[:type]

    assert_greater_equal_rank!(shape, 2)

    {{m, n}, random_seed} =
      transform({shape, distribution, type}, fn {shape, distribution, type} ->
        flat_shape =
          if tuple_size(shape) > 2 do
            tuple_list = shape |> Tuple.to_list() |> Enum.reverse()
            n = hd(tuple_list)
            m = Enum.reduce(tl(tuple_list), 1, &(&1 * &2))
            {m, n}
          else
            shape
          end

        random_seed =
          case distribution do
            :uniform ->
              orthogonal_random_uniform(flat_shape, type)

            :normal ->
              orthogonal_random_normal(flat_shape, type)

            dist ->
              raise ArgumentError,
                    "invalid distribution #{inspect(dist)} passed to orthogonal/1"
          end

        {flat_shape, random_seed}
      end)

    {q, _r} = Nx.LinAlg.qr(random_seed, mode: :complete)

    q
    |> Nx.slice([0, 0], [m, n])
    |> Nx.reshape(shape)
  end

  defnp orthogonal_random_normal(shape, type) do
    Nx.random_normal(shape, type: type)
  end

  defnp orthogonal_random_uniform(shape, type) do
    Nx.random_uniform(shape, type: type)
  end

  # Variance scaling branches

  defnp var_normal(variance, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    shape = opts[:shape]
    type = opts[:type]

    sigma = Nx.sqrt(variance)

    Nx.random_normal(shape, 0.0, sigma, type: type)
  end

  defnp var_uniform(variance, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    shape = opts[:shape]
    type = opts[:type]

    limit = Nx.sqrt(3 * variance)
    Nx.random_uniform(shape, -limit, limit, type: type)
  end

  defnp var_truncated(variance, opts \\ []) do
    opts = keyword!(opts, [:shape, type: {:f, 32}])
    shape = opts[:shape]
    type = opts[:type]

    sigma =
      variance
      |> Nx.sqrt()
      |> Nx.divide(0.87962566103423978)

    Nx.clip(Nx.random_normal(shape, 0.0, sigma, type: type), -2, 2)
  end

  defp compute_fans(shape) do
    rank = Nx.rank(shape)

    {fan_in, fan_out} =
      cond do
        rank < 1 ->
          {1, 1}

        rank == 1 ->
          {elem(shape, 0), elem(shape, 0)}

        rank == 2 ->
          {elem(shape, 0), elem(shape, 1)}

        true ->
          receptive_field_size = Nx.size(shape) / elem(shape, 0) / elem(shape, 1)

          fan_in = elem(shape, 0) * receptive_field_size
          fan_out = elem(shape, 1) * receptive_field_size
          {fan_in, fan_out}
      end

    {fan_in, fan_out}
  end
end
