defmodule Axon.Updates do
  @moduledoc ~S"""
  Parameter update methods.

  Update methods transform the input tensor in some way,
  usually by scaling or shifting the input with respect
  to some input state. Update methods are composed
  to create more advanced optimization methods such as AdaGrad
  or Adam. Each update returns a tuple:

      {init_fn, update_fn}

  Which represent a state initialization and state update
  function respectively. While each method in the Updates
  API is a regular Elixir function, the two methods they
  return are implemented as `defn`, so they can be accelerated
  using any Nx backend or compiler.

  Update methods are just combinators that can be arbitrarily
  composed to create complex optimizers. For example, the Adam
  optimizer in Axon.Optimizers is implemented as:

      def adam(learning_rate, opts \\ []) do
        Updates.scale_by_adam(opts)
        |> Updates.scale(-learning_rate)
      end

  ## Custom combinators

  You can create your own combinators using the `stateless/2` and
  `stateful/3` primitives. Every update method in this module is
  implemented in terms of one of these two primitives.

  `stateless/2` represents a stateless update:

      def scale(combinator \\ Axon.Updates.identity(), step_size) do
        stateless(combinator, &apply_scale(&1, &2, step_size))
      end

      defnp apply_scale(x, _params, step) do
        transform(
          {x, step},
          fn {updates, step} ->
            updates
            |> Tuple.to_list()
            |> Enum.map(&Nx.multiply(&1, step))
            |> List.to_tuple()
          end
        )
      end

  Notice how the function given to `stateless/2` is defined within `defn`.
  This is what allows the anonymous functions returned by `Axon.Updates`
  to be used inside `defn`.

  `stateful/3` represents a stateful update and follows the same pattern:

      def my_stateful_update(updates) do
        Axon.Updates.stateful(updates, &init_my_update/1, &apply_my_update/2)
      end

      defnp init_my_update(params) do
        state = zeros_like(params)
        {state}
      end

      defnp apply_my_update(updates, {state}) do
        new_state = apply_map(state, &Nx.add(&1, 0.01))
        updates = transform({updates, new_state}, fn {updates, state} ->
          updates
          |> Tuple.to_list()
          |> Enum.zip(Tuple.to_list(state))
          |> Enum.map(fn {g, z} -> Nx.multiply(g, z) end)
          |> List.to_tuple()
        end)
        {updates, {new_state}}
      end

  """
  import Nx.Defn
  import Axon.Shared

  @doc ~S"""
  Scales input by a fixed step size.

  $$f(x_i) = \alpha x_i$$
  """
  def scale(combinator \\ identity(), step_size) do
    stateless(combinator, &apply_scale(&1, &2, step_size))
  end

  defnp apply_scale(x, _params, step) do
    transform(
      {x, step},
      fn {updates, step} ->
        updates
        |> Tuple.to_list()
        |> Enum.map(&Nx.multiply(&1, step))
        |> List.to_tuple()
      end
    )
  end

  @doc """
  Scales input according to Adam algorithm.

  ## Options

      * `:b1` - first moment decay. Defaults to `0.9`
      * `:b2` - second moment decay. Defaults to `0.999`
      * `:eps` - numerical stability term. Defaults to `1.0e-8`
      * `:eps_root` - numerical stability term. Defaults to `0.0`

  ## References

    * [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

  """
  def scale_by_adam(combinator \\ identity(), opts) do
    stateful(
      combinator,
      &init_scale_by_adam/1,
      &apply_scale_by_adam(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_adam(params) do
    mus = zeros_like(params)
    nus = zeros_like(params)
    count = Nx.tensor(0)
    {mus, nus, count}
  end

  defnp apply_scale_by_adam(x, {mu, nu, count}, _params, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 1.0e-6, eps_root: 1.0e-5)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)
    nu = update_moment(x, nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x =
      transform({mu_hat, nu_hat, eps, eps_root}, fn {mu_hat, nu_hat, eps, eps_root} ->
        mu_hat
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(nu_hat))
        |> Enum.map(fn {z, t} -> z / (Nx.sqrt(t + eps_root) + eps) end)
        |> List.to_tuple()
      end)

    {x, {mu, nu, count + 1}}
  end

  @doc """
  Scales input by the root of all prior squared inputs.

  ## Options

      * `:eps` - numerical stability term. Defaults to `1.0e-7`

  """
  def scale_by_rss(combinator \\ identity(), opts) do
    {initial, opts} = Keyword.pop(opts, :initial_accumulator_value, 0.1)

    stateful(
      combinator,
      &init_scale_by_rss(&1, initial),
      &apply_scale_by_rss(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_rss(params, value) do
    sum_of_squares = fulls_like(params, value)
    {sum_of_squares}
  end

  defnp apply_scale_by_rss(x, {sum_of_squares}, _params, opts \\ []) do
    opts = keyword!(opts, eps: 1.0e-7)
    eps = opts[:eps]

    sum_of_squares =
      transform({x, sum_of_squares}, fn {x, sum_of_squares} ->
        x
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(sum_of_squares))
        |> Enum.map(fn {g, z} -> Nx.power(g, 2) + z end)
        |> List.to_tuple()
      end)

    inv_sqrt_squares =
      transform({sum_of_squares, eps}, fn {sum_of_squares, eps} ->
        sum_of_squares
        |> Tuple.to_list()
        |> Enum.map(fn z -> Nx.rsqrt(z + eps) end)
        |> List.to_tuple()
      end)

    inv_sqrt_x_square =
      transform({sum_of_squares, inv_sqrt_squares}, fn {sum_of_squares, inv_sqrt_squares} ->
        sum_of_squares
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(inv_sqrt_squares))
        |> Enum.map(fn {z, t} -> Nx.select(Nx.greater(z, 0), t, 0.0) end)
        |> List.to_tuple()
      end)

    x =
      transform({x, inv_sqrt_x_square}, fn {x, inv_sqrt_x_square} ->
        x
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(inv_sqrt_x_square))
        |> Enum.map(fn {g, t} -> g * t end)
        |> List.to_tuple()
      end)

    {x, {sum_of_squares}}
  end

  @doc """
  Scales input by the root of the EMA of squared inputs.

  ## Options

      * `:decay` - EMA decay rate. Defaults to `0.9`
      * `:eps` - numerical stability term. Defaults to `1.0e-8`

  ## References

    * [Overview of mini-batch gradient descent](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  """
  def scale_by_rms(combinator \\ identity(), opts) do
    {initial, opts} = Keyword.pop(opts, :initial_scale, 0.0)

    stateful(
      combinator,
      &init_scale_by_rms(&1, initial),
      &apply_scale_by_rms(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_rms(params, scale) do
    nu = fulls_like(params, scale)
    {nu}
  end

  defnp apply_scale_by_rms(x, {nu}, _params, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, eps: 1.0e-8)
    decay = opts[:decay]
    eps = opts[:eps]

    nu = update_moment(x, nu, decay, 2)

    x =
      transform({x, nu, eps}, fn {x, nu, eps} ->
        x
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(nu))
        |> Enum.map(fn {g, t} -> Nx.rsqrt(t + eps) * g end)
        |> List.to_tuple()
      end)

    {x, {nu}}
  end

  @doc """
  Scales input according to the AdaBelief algorithm.

  ## Options

      * `:b1` - first moment decay. Defaults to `0.9`
      * `:b2` - second moment decay. Defaults to `0.999`
      * `:eps` - numerical stability term. Defaults to `0.0`
      * `:eps_root` - numerical stability term. Defaults to `1.0e-16`

  ## References

    * [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468)

  """
  def scale_by_belief(combinator \\ identity(), opts) do
    stateful(
      combinator,
      &init_scale_by_belief/1,
      &apply_scale_by_belief(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_belief(params) do
    mus = zeros_like(params)
    nus = zeros_like(params)
    count = Nx.tensor(0)
    {mus, nus, count}
  end

  defnp apply_scale_by_belief(x, {mu, nu, count}, _params, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 0.0, eps_root: 1.0e-16)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)
    nu = update_moment(x, nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x =
      transform({mu_hat, nu_hat, eps, eps_root}, fn {mu_hat, nu_hat, eps, eps_root} ->
        mu_hat
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(nu_hat))
        |> Enum.map(fn {z, t} -> 1 / (Nx.sqrt(t + eps_root) + eps) * z end)
        |> List.to_tuple()
      end)

    {x, {mu, nu, count + 1}}
  end

  @doc """
  Scales input by the root of the centered EMA of squared inputs.

  ## Options

      * `:decay` - EMA decay rate. Defaults to `0.9`
      * `:eps` - numerical stability term. Defaults to `1.0e-8`

  ## References

    * [Overview of mini-batch gradient descent](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  """
  def scale_by_stddev(combinator \\ identity(), opts) do
    {initial, opts} = Keyword.pop(opts, :initial_scale, 0.0)

    stateful(
      combinator,
      &init_scale_by_stddev(&1, initial),
      &apply_scale_by_stddev(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_stddev(params, value) do
    mu = zeros_like(params)
    nu = fulls_like(params, value)
    {mu, nu}
  end

  defnp apply_scale_by_stddev(x, {mu, nu}, _params, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, eps: 1.0e-8)
    decay = opts[:decay]
    eps = opts[:eps]

    mu = update_moment(x, mu, decay, 1)
    nu = update_moment(x, nu, decay, 2)

    x =
      transform({x, mu, nu, eps}, fn {x, mu, nu, eps} ->
        [Tuple.to_list(x), Tuple.to_list(mu), Tuple.to_list(nu)]
        |> Enum.zip()
        |> Enum.map(fn {g, z, t} -> g * Nx.rsqrt(-Nx.power(z, 2) + t + eps) end)
        |> List.to_tuple()
      end)

    {x, {mu, nu}}
  end

  @doc """
  Scales input using the given schedule function.
  """
  def scale_by_schedule(combinator \\ identity(), schedule_fn) when is_function(schedule_fn) do
    stateful(
      combinator,
      &init_scale_by_schedule/1,
      &apply_scale_by_schedule(&1, &2, &3, schedule_fn)
    )
  end

  defnp init_scale_by_schedule(_) do
    {Nx.tensor(0)}
  end

  defnp apply_scale_by_schedule(x, {count}, _params, schedule_fn) do
    step_size = schedule_fn.(count)

    updates =
      transform({x, step_size}, fn {x, step_size} ->
        x
        |> Tuple.to_list()
        |> Enum.map(fn x -> x * step_size end)
        |> List.to_tuple()
      end)

    {updates, {count + 1}}
  end

  @doc """
  Scale input according to the Rectified Adam algorithm.

  ## Options

      * `:b1` - first moment decay. Defaults to `0.9`
      * `:b2` - second moment decay. Defaults to `0.999`
      * `:eps` - numerical stability term. Defaults to `1.0e-8`
      * `:eps_root` - numerical stability term. Defaults to `0.0`
      * `:threshold` - threshold for variance. Defaults to `5.0`

  ## References

    * [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)

  """
  def scale_by_radam(combinator \\ identity(), opts) do
    stateful(
      combinator,
      &init_scale_by_radam/1,
      &apply_scale_by_radam(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_radam(params) do
    mu = zeros_like(params)
    nu = zeros_like(params)
    count = Nx.tensor(0)
    {mu, nu, count}
  end

  defnp apply_scale_by_radam(x, {mu, nu, count}, _params, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 1.0e-8, eps_root: 0.0, threshold: 5.0)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]
    threshold = opts[:threshold]

    ro_inf =
      1
      |> Nx.subtract(b2)
      |> reciprocal()
      |> Nx.multiply(2)
      |> Nx.subtract(1)

    mu = update_moment(x, mu, b1, 1)
    nu = update_moment(x, nu, b2, 2)

    b2t =
      b2
      |> Nx.power(count + 1)

    ro =
      ro_inf
      |> Nx.subtract(2)
      |> Nx.multiply(count + 1)
      |> Nx.multiply(b2t)
      |> Nx.divide(1 - b2t)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x =
      if Nx.greater_equal(ro, threshold) do
        radam_update(ro, ro_inf, mu_hat, nu_hat, eps_root, eps)
      else
        mu_hat
      end

    {x, {mu, nu, count + 1}}
  end

  defnp radam_update(ro, ro_inf, mu, nu, eps_root, eps) do
    top =
      ro
      |> Nx.subtract(4)
      |> Nx.multiply(Nx.subtract(ro, 2))
      |> Nx.multiply(ro_inf)

    bottom =
      ro_inf
      |> Nx.subtract(4)
      |> Nx.multiply(Nx.subtract(ro, 2))
      |> Nx.multiply(ro)

    nu_hat =
      transform({nu, eps, eps_root}, fn {nu, eps, eps_root} ->
        nu
        |> Tuple.to_list()
        |> Enum.map(fn t -> Nx.sqrt(t + eps_root) + eps end)
        |> List.to_tuple()
      end)

    transform({mu, nu_hat, top, bottom}, fn {mu, nu_hat, top, bottom} ->
      mu
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(nu_hat))
      |> Enum.map(fn {z, t} -> Nx.sqrt(top / bottom) * (z / t) end)
      |> List.to_tuple()
    end)
  end

  @doc """
  Trace inputs with past inputs.

  ## Options

    * `:decay` - decay rate for tracing past updates. Defaults
      to `0.9`
    * `:nesterov` - whether to use Nesterov momentum. Defaults
      to `false`

  """
  def trace(combinator \\ identity(), opts) do
    stateful(
      combinator,
      &init_trace/1,
      &apply_trace(&1, &2, &3, opts)
    )
  end

  defnp init_trace(params) do
    trace = zeros_like(params)
    {trace}
  end

  defnp apply_trace(x, {trace}, _params, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, nesterov: false)
    decay = opts[:decay]

    update_trace =
      transform({x, trace, decay}, fn {x, trace, decay} ->
        x
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(trace))
        |> Enum.map(fn {g, t} -> t * decay + g end)
        |> List.to_tuple()
      end)

    x =
      transform({x, update_trace, decay, opts}, fn {x, trace, decay, opts} ->
        if opts[:nesterov] do
          x
          |> Tuple.to_list()
          |> Enum.zip(Tuple.to_list(trace))
          |> Enum.map(fn {g, t} -> t * decay + g end)
          |> List.to_tuple()
        else
          trace
        end
      end)

    {x, {update_trace}}
  end

  @doc """
  Clips input between -delta and delta.

  ## Options

    * `:delta` - maximum absolute value of the input. Defaults
      to `2.0`
  """
  def clip(combinator \\ identity(), opts) do
    stateless(combinator, &apply_clip(&1, &2, opts))
  end

  defnp apply_clip(x, _params, opts \\ []) do
    opts = keyword!(opts, delta: 2.0)
    delta = opts[:delta]

    transform({x, delta}, fn {x, delta} ->
      x
      |> Tuple.to_list()
      |> Enum.map(fn g -> Nx.clip(g, -delta, delta) end)
      |> List.to_tuple()
    end)
  end

  @doc """
  Clips input using input global norm.

  ## Options

    * `:max_norm` - maximum norm value of input. Defaults to
      `1.0`
  """
  def clip_by_global_norm(combinator \\ identity(), opts) do
    stateless(combinator, &apply_clip_by_global_norm(&1, &2, opts))
  end

  defnp apply_clip_by_global_norm(x, _params, opts \\ []) do
    opts = keyword!(opts, max_norm: 1.0)
    max_norm = opts[:max_norm]

    g_norm = apply_map(x, fn z -> Nx.sqrt(Nx.sum(Nx.power(z, 2))) end)

    transform({x, g_norm, max_norm}, fn {x, g_norm, max_norm} ->
      x
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(g_norm))
      |> Enum.map(fn {z, g} -> Nx.select(Nx.less(g, max_norm), z, z / g * max_norm) end)
      |> List.to_tuple()
    end)
  end

  @doc """
  Centralize input.
  """
  def centralize(combinator \\ identity()) do
    stateless(combinator, &apply_centralize/2)
  end

  defnp apply_centralize(x, _params) do
    apply_map(x, fn z -> -Nx.mean(z) + z end)
  end

  @doc """
  Weight decay.
  """
  def add_decayed_weights(combinator \\ identity(), opts) do
    stateless(combinator, &apply_weight_decay(&1, &2, opts))
  end

  defnp apply_weight_decay(updates, params, opts \\ []) do
    opts = keyword!(opts, decay: 0.0)
    decay = opts[:decay]

    transform({updates, params, decay}, fn {updates, params, decay} ->
      updates
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(params))
      |> Enum.map(fn {g, p} -> g + decay * p end)
      |> List.to_tuple()
    end)
  end

  @doc """
  Scale by trust ratio.
  """
  def scale_by_trust_ratio(combinator \\ identity(), opts) do
    stateless(combinator, &apply_scale_by_trust_ratio(&1, &2, opts))
  end

  defnp apply_scale_by_trust_ratio(x, params, opts \\ []) do
    opts = keyword!(opts, min_norm: 0.0)
    min_norm = opts[:min_norm]

    param_norm = safe_norm(params, min_norm)
    update_norm = safe_norm(x, min_norm)

    trust_ratios =
      transform({param_norm, update_norm}, fn {param_norm, update_norm} ->
        param_norm
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(update_norm))
        |> Enum.map(fn {p, g} -> p / g end)
        |> List.to_tuple()
      end)

    zero_norms =
      transform({param_norm, update_norm}, fn {param_norm, update_norm} ->
        param_norm
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(update_norm))
        |> Enum.map(fn {p, g} -> Nx.logical_or(Nx.equal(p, 0), Nx.equal(g, 0)) end)
        |> List.to_tuple()
      end)

    transform({zero_norms, trust_ratios, x}, fn {zero_norms, trust_ratios, x} ->
      [Tuple.to_list(zero_norms), Tuple.to_list(trust_ratios), Tuple.to_list(x)]
      |> Enum.zip()
      |> Enum.map(fn {z, t, g} -> g * Nx.select(z, 1, t) end)
      |> List.to_tuple()
    end)
  end

  @doc """
  Add noise.
  """
  def add_noise(combinator \\ identity(), opts) do
    stateful(combinator, &init_add_noise/1, &apply_add_noise(&1, &2, &3, opts))
  end

  defnp init_add_noise(_params) do
    {Nx.tensor(0)}
  end

  defnp apply_add_noise(x, {count}, _params, opts \\ []) do
    opts = keyword!(opts, eta: 0.01, gamma: 0.55)
    var = opts[:eta] / Nx.power(count + 1, opts[:gamma])

    noise = apply_map(x, fn x -> Nx.random_normal(x) end)

    updates =
      transform({x, noise, var}, fn {x, noise, var} ->
        x
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(noise))
        |> Enum.map(fn {g, n} -> g + var * n end)
        |> List.to_tuple()
      end)

    {updates, {count + 1}}
  end

  @doc """
  Scale by yogi.
  """
  def scale_by_yogi(combinator \\ identity(), opts) do
    {initial, opts} = Keyword.pop(opts, :initial_accumulator_value, 1.0e-6)

    stateful(
      combinator,
      &init_scale_by_yogi(&1, initial),
      &apply_scale_by_yogi(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_yogi(params, value) do
    value = fulls_like(params, value)
    mu = value
    nu = value
    count = Nx.tensor(0)
    {mu, nu, count}
  end

  defnp apply_scale_by_yogi(x, {mu, nu, count}, _params, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 1.0e-3, eps_root: 0.0)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)

    signed_sq =
      transform({x, nu}, fn {x, nu} ->
        x
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(nu))
        |> Enum.map(fn {g, v} -> Nx.sign(v - Nx.power(g, 2)) * Nx.power(g, 2) end)
        |> List.to_tuple()
      end)

    nu = update_moment(signed_sq, nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    updates =
      transform({mu_hat, nu_hat, eps, eps_root}, fn {mu_hat, nu_hat, eps, eps_root} ->
        mu_hat
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(nu_hat))
        |> Enum.map(fn {m, v} -> m / (Nx.sqrt(v + eps_root) + eps) end)
        |> List.to_tuple()
      end)

    {updates, {mu, nu, count + 1}}
  end

  @doc """
  Represents a stateless update.
  """
  def stateless({parent_init_fn, parent_apply_fn} \\ identity(), apply_fn) do
    apply_fn = fn updates, state, params ->
      {updates, state} = parent_apply_fn.(updates, state, params)
      {apply_fn.(updates, params), state}
    end

    {parent_init_fn, apply_fn}
  end

  @doc """
  Returns the identity update.

  This is often as the initial update in many functions in this module.
  """
  def identity() do
    {fn _params -> {} end, fn updates, state, _params -> {updates, state} end}
  end

  def identity(combinator) do
    combinator
  end

  @doc """
  Represents a stateful update.
  """
  def stateful({parent_init_fn, parent_apply_fn} \\ identity(), init_fn, apply_fn) do
    init_fn = fn params ->
      state = parent_init_fn.(params)
      Tuple.insert_at(state, 0, init_fn.(params))
    end

    apply_fn = fn updates, state, params ->
      this_state = elem(state, 0)
      other_state = Tuple.delete_at(state, 0)
      {updates, new_other_state} = parent_apply_fn.(updates, other_state, params)
      {updates, new_this_state} = apply_fn.(updates, this_state, params)
      {updates, Tuple.insert_at(new_other_state, 0, new_this_state)}
    end

    {init_fn, apply_fn}
  end

  @doc """
  Applies updates to params.
  """
  defn apply_updates(params, updates) do
    transform({params, updates}, fn {params, updates} ->
      params
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(updates))
      |> Enum.map(fn {x, u} -> Nx.add(x, u) end)
      |> List.to_tuple()
    end)
  end

  ## Helpers

  defnp update_moment(x, moment, decay, order) do
    transform({x, moment, decay, order}, fn {x, moment, decay, order} ->
      x
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(moment))
      |> Enum.map(fn {g, z} -> (1 - decay) * Nx.power(g, order) + decay * z end)
      |> List.to_tuple()
    end)
  end

  defnp bias_correction(moment, decay, count) do
    transform({moment, decay, count}, fn {moment, decay, count} ->
      moment
      |> Tuple.to_list()
      |> Enum.map(fn z -> z / (1 - Nx.power(decay, count)) end)
      |> List.to_tuple()
    end)
  end

  defnp safe_norm(x, min_norm) do
    transform({x, min_norm}, fn {x, min_norm} ->
      x
      |> Tuple.to_list()
      |> Enum.map(fn g ->
        norm = Nx.LinAlg.norm(g)
        z = Nx.select(Nx.less(norm, min_norm), 1, g)
        Nx.select(Nx.less(norm, min_norm), min_norm, Nx.LinAlg.norm(z))
      end)
      |> List.to_tuple()
    end)
  end

  defnp empty_state(_) do
    {}
  end
end
