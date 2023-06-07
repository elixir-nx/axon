defmodule Axon.Updates do
  @moduledoc false

  import Nx.Defn
  import Axon.Shared

  @doc ~S"""
  Scales input by a fixed step size.

  $$f(x_i) = \alpha x_i$$
  """
  @deprecated "Use Optimus.Updates.scale/2 instead"
  def scale(combinator \\ identity(), step_size) do
    stateless(combinator, &apply_scale(&1, &2, step_size))
  end

  defnp apply_scale(updates, _params, step) do
    deep_new(updates, fn v -> Nx.multiply(v, step) end)
  end

  @doc ~S"""
  Scales input by a tunable learning rate which can be
  manipulated by external APIs such as Axon's Loop API.

  $$f(x_i) = \alpha x_i$$
  """
  @deprecated "Use Optimus.Updates.scale_by_state/1 instead"
  def scale_by_state(combinator_or_step)

  def scale_by_state(step) when is_number(step) do
    scale_by_state(identity(), step)
  end

  def scale_by_state({init_fn, apply_fn} = combinator, step)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_number(step) do
    stateful(combinator, &init_scale_by_state(&1, init_scale: step), &apply_scale_by_state/3)
  end

  defnp init_scale_by_state(_params, opts \\ []) do
    opts = keyword!(opts, [:init_scale])
    %{scale: opts[:init_scale]}
  end

  defnp apply_scale_by_state(x, %{scale: scale} = state, params) do
    {apply_scale(x, params, scale), state}
  end

  @doc """
  Scales input according to Adam algorithm.

  ## Options

      * `:b1` - first moment decay. Defaults to `0.9`

      * `:b2` - second moment decay. Defaults to `0.999`

      * `:eps` - numerical stability term. Defaults to `1.0e-8`

      * `:eps_root` - numerical stability term. Defaults to `1.0e-15`

  ## References

    * [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

  """
  @deprecated "Use Optimus.Updates.scale_by_adam/1 instead"
  def scale_by_adam(combinator_or_opts \\ [])

  def scale_by_adam(opts) when is_list(opts) do
    scale_by_adam(identity(), opts)
  end

  def scale_by_adam({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    scale_by_adam(combinator, [])
  end

  def scale_by_adam({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateful(
      combinator,
      &init_scale_by_adam/1,
      &apply_scale_by_adam(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_adam(params) do
    mus = zeros_like(params, type: :f32)
    nus = zeros_like(params, type: :f32)
    count = Nx.tensor(0)
    %{mu: mus, nu: nus, count: count}
  end

  defnp apply_scale_by_adam(x, %{mu: mu, nu: nu, count: count}, _params, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 1.0e-8, eps_root: 1.0e-15)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)
    nu = update_moment(x, nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x = deep_merge(mu_hat, nu_hat, fn z, t -> z / (Nx.sqrt(t + eps_root) + eps) end)
    {x, %{mu: mu, nu: nu, count: count + 1}}
  end

  @doc """
  Scales input by the root of all prior squared inputs.

  ## Options

      * `:eps` - numerical stability term. Defaults to `1.0e-7`

  """
  @deprecated "Use Optimus.Updates.scale_by_rss/1 instead"
  def scale_by_rss(combinator_or_opts \\ [])

  def scale_by_rss(opts) when is_list(opts) do
    scale_by_rss(identity(), opts)
  end

  def scale_by_rss({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    scale_by_rss(combinator, [])
  end

  def scale_by_rss({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    {initial, opts} = Keyword.pop(opts, :initial_accumulator_value, 0.1)

    stateful(
      combinator,
      &init_scale_by_rss(&1, initial),
      &apply_scale_by_rss(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_rss(params, value) do
    sum_of_squares = fulls_like(params, value, type: :f32)
    %{sum_of_squares: sum_of_squares}
  end

  defnp apply_scale_by_rss(x, %{sum_of_squares: sum_of_squares}, _params, opts \\ []) do
    opts = keyword!(opts, eps: 1.0e-7)
    eps = opts[:eps]

    sum_of_squares = deep_merge(x, sum_of_squares, fn g, z -> Nx.pow(g, 2) + z end)

    inv_sqrt_squares = deep_new(sum_of_squares, fn z -> Nx.rsqrt(z + eps) end)

    inv_sqrt_x_square =
      deep_merge(sum_of_squares, inv_sqrt_squares, fn z, t ->
        Nx.select(Nx.greater(z, 0), t, 0.0)
      end)

    x = deep_merge(x, inv_sqrt_x_square, fn g, t -> g * t end)

    {x, %{sum_of_squares: sum_of_squares}}
  end

  @doc """
  Scales input by the root of the EMA of squared inputs.

  ## Options

      * `:decay` - EMA decay rate. Defaults to `0.9`.

      * `:eps` - numerical stability term. Defaults to `1.0e-8`.

  ## References

    * [Overview of mini-batch gradient descent](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  """
  @deprecated "Use Optimus.Updates.scale_by_rms/1 instead"
  def scale_by_rms(combinator_or_opts \\ [])

  def scale_by_rms(opts) when is_list(opts) do
    scale_by_rms(identity(), opts)
  end

  def scale_by_rms({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    scale_by_rms(combinator, [])
  end

  def scale_by_rms({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    {initial, opts} = Keyword.pop(opts, :initial_scale, 0.0)

    stateful(
      combinator,
      &init_scale_by_rms(&1, initial),
      &apply_scale_by_rms(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_rms(params, scale) do
    nu = fulls_like(params, scale, type: :f32)
    %{nu: nu}
  end

  defnp apply_scale_by_rms(x, %{nu: nu}, _params, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, eps: 1.0e-8)
    decay = opts[:decay]
    eps = opts[:eps]

    nu = update_moment(x, nu, decay, 2)

    x = deep_merge(x, nu, fn g, t -> Nx.rsqrt(t + eps) * g end)

    {x, %{nu: nu}}
  end

  @doc """
  Scales input according to the AdaBelief algorithm.

  ## Options

      * `:b1` - first moment decay. Defaults to `0.9`.

      * `:b2` - second moment decay. Defaults to `0.999`.

      * `:eps` - numerical stability term. Defaults to `0.0`.

      * `:eps_root` - numerical stability term. Defaults to `1.0e-16`.

  ## References

    * [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468)

  """
  @deprecated "Use Optimus.Updates.scale_by_belief/1 instead"
  def scale_by_belief(combinator_or_opts \\ [])

  def scale_by_belief(opts) when is_list(opts) do
    scale_by_belief(identity(), opts)
  end

  def scale_by_belief({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    scale_by_belief(combinator, [])
  end

  def scale_by_belief({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateful(
      combinator,
      &init_scale_by_belief/1,
      &apply_scale_by_belief(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_belief(params) do
    mus = zeros_like(params, type: :f32)
    nus = zeros_like(params, type: :f32)
    count = Nx.tensor(0)
    %{mu: mus, nu: nus, count: count}
  end

  defnp apply_scale_by_belief(x, %{mu: mu, nu: nu, count: count}, _params, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 0.0, eps_root: 1.0e-16)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)
    nu = update_moment(x, nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x = deep_merge(mu_hat, nu_hat, fn z, t -> 1 / (Nx.sqrt(t + eps_root) + eps) * z end)

    {x, %{mu: mu, nu: nu, count: count + 1}}
  end

  @doc """
  Scales input by the root of the centered EMA of squared inputs.

  ## Options

      * `:decay` - EMA decay rate. Defaults to `0.9`.

      * `:eps` - numerical stability term. Defaults to `1.0e-8`.

  ## References

    * [Overview of mini-batch gradient descent](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  """
  @deprecated "Use Optimus.Updates.scale_by_stddev/1 instead"
  def scale_by_stddev(combinator_or_opts \\ [])

  def scale_by_stddev(opts) when is_list(opts) do
    scale_by_stddev(identity(), opts)
  end

  def scale_by_stddev({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    scale_by_stddev(combinator, [])
  end

  def scale_by_stddev({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    {initial, opts} = Keyword.pop(opts, :initial_scale, 0.0)

    stateful(
      combinator,
      &init_scale_by_stddev(&1, initial),
      &apply_scale_by_stddev(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_stddev(params, value) do
    mu = zeros_like(params, type: :f32)
    nu = fulls_like(params, value, type: :f32)
    %{mu: mu, nu: nu}
  end

  defnp apply_scale_by_stddev(x, %{mu: mu, nu: nu}, _params, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, eps: 1.0e-8)
    decay = opts[:decay]
    eps = opts[:eps]

    mu = update_moment(x, mu, decay, 1)
    nu = update_moment(x, nu, decay, 2)

    mu_nu =
      deep_merge(mu, nu, fn m, n ->
        Nx.rsqrt(-Nx.pow(m, 2) + n + eps)
      end)

    x = deep_merge(x, mu_nu, fn g, mn -> g * mn end)

    {x, %{mu: mu, nu: nu}}
  end

  @doc """
  Scales input using the given schedule function.

  This can be useful for implementing learning rate schedules.
  The number of update iterations is tracked by an internal
  counter. You might need to update the schedule to operate
  on per-batch schedule rather than per-epoch.
  """
  @deprecated "Use Optimus.Updates.scale_by_schedule/2 instead"
  def scale_by_schedule(combinator \\ identity(), schedule_fn) when is_function(schedule_fn, 1) do
    stateful(
      combinator,
      &init_scale_by_schedule/1,
      &apply_scale_by_schedule(&1, &2, &3, schedule_fn)
    )
  end

  defnp init_scale_by_schedule(_) do
    %{count: Nx.tensor(0)}
  end

  defnp apply_scale_by_schedule(x, %{count: count}, _params, schedule_fn) do
    step_size = schedule_fn.(count)

    updates = deep_new(x, fn x -> x * step_size end)

    {updates, %{count: count + 1}}
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
  @deprecated "Use Optimus.Updates.scale_by_radam/1 instead"
  def scale_by_radam(combinator_or_opts \\ [])

  def scale_by_radam(opts) when is_list(opts) do
    scale_by_radam(identity(), opts)
  end

  def scale_by_radam({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    scale_by_radam(combinator, [])
  end

  def scale_by_radam({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateful(
      combinator,
      &init_scale_by_radam/1,
      &apply_scale_by_radam(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_radam(params) do
    mu = zeros_like(params, type: :f32)
    nu = zeros_like(params, type: :f32)
    count = Nx.tensor(0)
    %{mu: mu, nu: nu, count: count}
  end

  defnp apply_scale_by_radam(x, %{mu: mu, nu: nu, count: count}, _params, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 1.0e-8, eps_root: 0.0, threshold: 5.0)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]
    threshold = opts[:threshold]

    ro_inf = 2.0 / (1 - b1) - 1

    mu = update_moment(x, mu, b1, 1)
    nu = update_moment(x, nu, b2, 2)
    count_inc = count + 1

    b2t = Nx.pow(b2, count_inc)
    ro = ro_inf - 2 * count_inc * b2t / (1 - b2t)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x =
      if Nx.all(Nx.greater_equal(ro, threshold)) do
        radam_update(ro, ro_inf, mu_hat, nu_hat, eps_root, eps)
      else
        mu_hat
      end

    {x, %{mu: mu, nu: nu, count: count + 1}}
  end

  defnp radam_update(ro, ro_inf, mu, nu, eps_root, eps) do
    r = Nx.sqrt((ro - 4) * (ro - 2) * ro_inf / ((ro_inf - 4) * (ro_inf - 2) * ro))

    deep_merge(mu, nu, fn m, v ->
      r * m / (Nx.sqrt(v + eps_root) + eps)
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
  @deprecated "Use Optimus.Updates.trace/1 instead"
  def trace(combinator_or_opts \\ [])

  def trace(opts) when is_list(opts) do
    trace(identity(), opts)
  end

  def trace({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    trace(combinator, [])
  end

  def trace({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateful(
      combinator,
      &init_trace/1,
      &apply_trace(&1, &2, &3, opts)
    )
  end

  defnp init_trace(params) do
    trace = zeros_like(params, type: :f32)
    %{trace: trace}
  end

  defnp apply_trace(x, %{trace: trace}, _params, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, nesterov: false)
    decay = opts[:decay]

    update_trace = deep_merge(x, trace, fn g, t -> t * decay + g end)

    x =
      if opts[:nesterov] do
        deep_merge(x, update_trace, fn g, t -> t * decay + g end)
      else
        update_trace
      end

    {x, %{trace: update_trace}}
  end

  @doc """
  Clips input between -delta and delta.

  ## Options

    * `:delta` - maximum absolute value of the input. Defaults
      to `2.0`
  """
  @deprecated "Use Optimus.Updates.clip/1 instead"
  def clip(combinator_or_opts \\ [])

  def clip(opts) when is_list(opts) do
    clip(identity(), opts)
  end

  def clip({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    clip(combinator, [])
  end

  def clip({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateless(combinator, &apply_clip(&1, &2, opts))
  end

  defnp apply_clip(x, _params, opts \\ []) do
    opts = keyword!(opts, delta: 2.0)
    delta = opts[:delta]

    deep_new(x, fn g -> Nx.clip(g, -delta, delta) end)
  end

  @doc """
  Clips input using input global norm.

  ## Options

    * `:max_norm` - maximum norm value of input. Defaults to
      `1.0`
  """
  @deprecated "Use Optimus.Updates.clip_by_global_norm/1 instead"
  def clip_by_global_norm(combinator_or_opts \\ [])

  def clip_by_global_norm(opts) when is_list(opts) do
    clip_by_global_norm(identity(), opts)
  end

  def clip_by_global_norm({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    clip_by_global_norm(combinator, [])
  end

  def clip_by_global_norm({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateless(combinator, &apply_clip_by_global_norm(&1, &2, opts))
  end

  defnp apply_clip_by_global_norm(x, _params, opts \\ []) do
    opts = keyword!(opts, max_norm: 1.0)
    max_norm = opts[:max_norm]

    sum_gs =
      deep_reduce(x, Nx.tensor(0.0), fn leaf, acc ->
        leaf
        |> Nx.pow(2)
        |> Nx.sum()
        |> Nx.add(acc)
      end)

    g_norm = Nx.sqrt(sum_gs)

    deep_new(x, fn z ->
      Nx.select(Nx.less(g_norm, max_norm), z, z / g_norm * max_norm)
    end)
  end

  @doc """
  Centralizes input by shifting updates by their mean.
  """
  @deprecated "Use Optimus.Updates.centralize/1 instead"
  def centralize(combinator_or_opts \\ [])

  def centralize(opts) when is_list(opts) do
    centralize(identity(), opts)
  end

  def centralize({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    centralize(combinator, [])
  end

  def centralize({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateless(combinator, &apply_centralize(&1, &2, opts))
  end

  defnp apply_centralize(x, _params, _opts \\ []) do
    deep_new(x, &centralize_for_rank/1)
  end

  deftransformp centralize_for_rank(input) do
    if Nx.rank(input) > 1 do
      input
      |> Nx.subtract(Nx.mean(input, axes: tl(Nx.axes(input)), keep_axes: true))
    else
      input
    end
  end

  @doc """
  Adds decayed weights to updates.

  Commonly used as a regularization strategy.

  ## Options

      * `:decay` - Rate of decay. Defaults to `0.0`.
  """
  @deprecated "Use Optimus.Updates.add_decayed_weights/1 instead"
  def add_decayed_weights(combinator_or_opts \\ [])

  def add_decayed_weights(opts) when is_list(opts) do
    add_decayed_weights(identity(), opts)
  end

  def add_decayed_weights({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    add_decayed_weights(combinator, [])
  end

  def add_decayed_weights({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateless(combinator, fn updates, params ->
      opts = Nx.Defn.Kernel.keyword!(opts, decay: 0.0)
      # Decay can be a tensor, that's why we preprocess it before-hand
      # and pass it as argument to defn instead of as an option.
      apply_weight_decay(updates, params, opts[:decay])
    end)
  end

  defnp apply_weight_decay(updates, params, decay) do
    deep_merge(updates, params, fn g, p -> g + decay * p end)
  end

  @doc """
  Scale by trust ratio.

  ## Options

      * `:min_norm` - Min norm to clip. Defaults to
        `0.0`.

      * `:trust_coefficient` - Trust coefficient. Defaults
        to `1.0`.

      * `:eps` - Numerical stability term. Defaults to `0.0`.
  """
  @deprecated "Use Optimus.Updates.scale_by_trust_ratio/1 instead"
  def scale_by_trust_ratio(combinator_or_opts \\ [])

  def scale_by_trust_ratio(opts) when is_list(opts) do
    scale_by_trust_ratio(identity(), opts)
  end

  def scale_by_trust_ratio({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    scale_by_trust_ratio(combinator, [])
  end

  def scale_by_trust_ratio({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    stateless(combinator, fn update, params ->
      opts = Nx.Defn.Kernel.keyword!(opts, min_norm: 0.0, trust_coefficient: 1.0, eps: 0.0)

      apply_scale_by_trust_ratio(
        update,
        params,
        opts[:min_norm],
        opts[:trust_coefficient],
        opts[:eps]
      )
    end)
  end

  defnp apply_scale_by_trust_ratio(updates, params, min_norm, trust_coefficient, eps) do
    deep_merge(updates, params, fn g, p ->
      param_norm = safe_norm(p, min_norm)
      update_norm = safe_norm(g, min_norm)

      trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

      zero_norm = param_norm == 0.0 or update_norm == 0.0
      safe_trust_ratio = Nx.select(zero_norm, 1, trust_ratio)
      g * safe_trust_ratio
    end)
  end

  @doc """
  Adds random Gaussian noise to the input.

  ## Options

      * `:seed` - Random seed to use. Defaults to the
        current system time.

      * `:eta` - Controls amount of noise to add.
        Defaults to `0.01`.

      * `:gamma` - Controls amount of noise to add.
        Defaults to `0.55`.
  """
  @deprecated "Use Optimus.Updates.add_noise/1 instead"
  def add_noise(combinator_or_opts \\ [])

  def add_noise(opts) when is_list(opts) do
    add_noise(identity(), opts)
  end

  def add_noise({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    add_noise(combinator, [])
  end

  def add_noise({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) and is_list(opts) do
    {seed, opts} = Keyword.pop_lazy(opts, :seed, fn -> :erlang.system_time() end)
    stateful(combinator, &init_add_noise(&1, seed: seed), &apply_add_noise(&1, &2, &3, opts))
  end

  defnp init_add_noise(_params, opts \\ []) do
    %{count: Nx.tensor(0), key: Nx.Random.key(opts[:seed])}
  end

  defnp apply_add_noise(x, %{count: count, key: key}, _params, opts \\ []) do
    opts = keyword!(opts, eta: 0.01, gamma: 0.55)
    var = opts[:eta] / Nx.pow(count + 1, opts[:gamma])

    {noise, key} =
      deep_map_reduce(x, key, fn z, key ->
        Nx.Random.normal(key, shape: Nx.shape(z), type: Nx.type(z))
      end)

    updates = deep_merge(x, noise, fn g, n -> g + var * n end)

    {updates, %{count: count + 1, key: key}}
  end

  @doc """
  Scale input according to the Yogi algorithm.

  ## Options

      * `:initial_accumulator_value` - Initial state accumulator value.

      * `:b1` - first moment decay. Defaults to `0.9`

      * `:b2` - second moment decay. Defaults to `0.999`

      * `:eps` - numerical stability term. Defaults to `1.0e-8`

      * `:eps_root` - numerical stability term. Defaults to `0.0`

  ## References

      * [Adaptive Methods for Nonconvex Optimization](https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf)
  """
  @deprecated "Use Optimus.Updates.scale_by_yogi/1 instead"
  def scale_by_yogi(combinator_or_opts \\ [])

  def scale_by_yogi(opts) when is_list(opts) do
    scale_by_yogi(identity(), opts)
  end

  def scale_by_yogi({init_fn, apply_fn} = combinator)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    scale_by_yogi(combinator, [])
  end

  def scale_by_yogi({init_fn, apply_fn} = combinator, opts)
      when is_function(init_fn, 1) and is_function(apply_fn, 3) do
    {initial, opts} = Keyword.pop(opts, :initial_accumulator_value, 1.0e-6)

    stateful(
      combinator,
      &init_scale_by_yogi(&1, initial),
      &apply_scale_by_yogi(&1, &2, &3, opts)
    )
  end

  defnp init_scale_by_yogi(params, value) do
    value = fulls_like(params, value, type: :f32)
    mu = value
    nu = value
    count = Nx.tensor(0)
    %{mu: mu, nu: nu, count: count}
  end

  defnp apply_scale_by_yogi(x, %{mu: mu, nu: nu, count: count}, _params, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 1.0e-3, eps_root: 0.0)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)

    nu =
      deep_merge(x, nu, fn g, v ->
        v - (1 - b2) * Nx.sign(v - Nx.pow(g, 2)) * Nx.pow(g, 2)
      end)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    updates = deep_merge(mu_hat, nu_hat, fn m, v -> m / (Nx.sqrt(v + eps_root) + eps) end)

    {updates, %{mu: mu, nu: nu, count: count + 1}}
  end

  @doc """
  Represents a stateless update.

  Stateless updates do not depend on an update state and thus
  only require an implementation of an update function.
  """
  @deprecated "Use Optimus.Updates.stateless/2 instead"
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
  @deprecated "Use Optimus.Updates.identity/1 instead"
  def identity() do
    {fn _params -> {} end, fn updates, state, _params -> {updates, state} end}
  end

  def identity(combinator) do
    combinator
  end

  @doc """
  Composes two updates. This is useful for extending optimizers
  without having to reimplement them. For example, you can implement
  gradient centralization:

      import Axon.Updates

      Axon.Updates.compose(Axon.Updates.centralize(), Axon.Optimizers.rmsprop())

  This is equivalent to:

      Axon.Updates.centralize()
      |> Axon.Updates.scale_by_rms()
  """
  @deprecated "Use Optimus.Updates.compose/2 instead"
  def compose({init_fn1, apply_fn1}, {init_fn2, apply_fn2}) do
    init_fn = fn params ->
      state = init_fn1.(params)
      Tuple.insert_at(state, 0, init_fn2.(params))
    end

    apply_fn = fn updates, state, params ->
      this_state = elem(state, 0)
      other_state = Tuple.delete_at(state, 0)
      {updates, new_other_state} = apply_fn1.(updates, other_state, params)
      {updates, new_this_state} = apply_fn2.(updates, this_state, params)
      {updates, Tuple.insert_at(new_other_state, 0, new_this_state)}
    end

    {init_fn, apply_fn}
  end

  @doc """
  Represents a stateful update.

  Stateful updates require some update state, such as
  momentum or RMS of previous updates. Therefore you must
  implement some initialization function as well as an update
  function.
  """
  @deprecated "Use Optimus.Updates.stateful/3 instead"
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
  Applies updates to params and updates state parameters with
  given state map.
  """
  defn apply_updates(params, updates, state \\ nil) do
    new_params =
      deep_merge(params, updates, fn x, u ->
        Nx.add(x, Nx.as_type(u, Nx.type(x)))
      end)

    merge_state(new_params, state)
  end

  deftransformp merge_state(params, state) do
    case {params, state} do
      {params, nil} ->
        params

      {params, state} ->
        merge_inner(params, state)
    end
  end

  defp merge_inner(%Nx.Tensor{}, %Nx.Tensor{} = state) do
    state
  end

  defp merge_inner(params, state) when is_map(params) and is_map(state) do
    Map.merge(params, state, fn _, s1, s2 -> merge_inner(s1, s2) end)
  end

  ## Helpers

  defnp update_moment(x, moment, decay, order) do
    deep_merge(x, moment, fn g, z -> (1 - decay) * Nx.pow(g, order) + decay * z end)
  end

  defnp bias_correction(moment, decay, count) do
    deep_new(moment, fn z -> z / (1 - Nx.pow(decay, count)) end)
  end

  defnp safe_norm(g, min_norm) do
    norm = Nx.LinAlg.norm(g)
    z = Nx.select(Nx.less_equal(norm, min_norm), 1, g)
    masked_norm = Nx.LinAlg.norm(z)
    Nx.select(Nx.less_equal(norm, min_norm), min_norm, masked_norm)
  end
end
