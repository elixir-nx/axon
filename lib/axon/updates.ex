defmodule Axon.Updates do
  @moduledoc """
  Parameter update methods.

  Update methods transform the input tensor in some way,
  usually by scaling or shifting the input with respect
  to some input state. Update methods are typically composed
  to create more advanced optimization methods such as AdaGrad
  or Adam; however, they can also be applied to model parameters.

  """
  import Nx.Defn
  import Axon.Shared

  @doc ~S"""
  Scales input by a fixed step size.

  $$f(x_i) = \alpha x_i$$
  """
  def scale(transform, step_size) do
    stateless(transform, &apply_scale(&1, step_size))
  end

  def scale(step_size) do
    stateless(&apply_scale(&1, step_size))
  end

  defnp apply_scale(x, step) do
    transform({x, step},
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
  def scale_by_adam(transform, opts) do
    b1 = opts[:b1] || 0.9
    b2 = opts[:b2] || 0.999
    eps = opts[:eps] || 1.0e-6
    eps_root = opts[:eps_root] || 1.0e-5

    stateful(transform, &init_scale_by_adam/1, &apply_scale_by_adam(&1, &2, [b1: b1, b2: b2, eps: eps, eps_root: eps_root]))
  end

  def scale_by_adam(opts) do
    b1 = opts[:b1] || 0.9
    b2 = opts[:b2] || 0.999
    eps = opts[:eps] || 1.0e-6
    eps_root = opts[:eps_root] || 1.0e-5

    stateful(&init_scale_by_adam/1, &apply_scale_by_adam(&1, &2, [b1: b1, b2: b2, eps: eps, eps_root: eps_root]))
  end

  defnp init_scale_by_adam(params) do
    mus = zeros_like(params)
    nus = zeros_like(params)
    count = Nx.tensor(0)
    {mus, nus, count}
  end

  defnp apply_scale_by_adam(x, {mu, nu, count}, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 1.0e-6, eps_root: 1.0e-5)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)
    nu = update_moment(x, nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x = transform({mu_hat, nu_hat, eps, eps_root}, fn {mu_hat, nu_hat, eps, eps_root} ->
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
  def scale_by_rss(combinator, opts) do
    initial_accumulator_value = opts[:initial_accumulator_value] || 0.1
    eps = opts[:eps] || 1.0e-7

    stateful(combinator, &init_scale_by_rss(&1, initial_accumulator_value), &apply_scale_by_rss(&1, &2, [eps: eps]))
  end

  def scale_by_rss(opts) do
    initial_accumulator_value = opts[:initial_accumulator_value] || 0.1
    eps = opts[:eps] || 1.0e-7

    stateful(&init_scale_by_rss(&1, initial_accumulator_value), &apply_scale_by_rss(&1, &2, [eps: eps]))
  end

  defnp init_scale_by_rss(params, value) do
    sum_of_squares = fulls_like(params, value)
    {sum_of_squares}
  end

  defnp apply_scale_by_rss(x, {sum_of_squares}, opts \\ []) do
    opts = keyword!(opts, eps: 1.0e-7)
    eps = opts[:eps]

    sum_of_squares = transform({x, sum_of_squares}, fn {x, sum_of_squares} ->
      x
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(sum_of_squares))
      |> Enum.map(fn {g, z} -> Nx.power(g, 2) + z end)
      |> List.to_tuple()
    end)

    inv_sqrt_squares = transform({sum_of_squares, eps}, fn {sum_of_squares, eps} ->
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

    x = transform({x, inv_sqrt_x_square}, fn {x, inv_sqrt_x_square} ->
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
  def scale_by_rms(combinator, opts) do
    decay = opts[:decay] || 0.9
    eps = opts[:eps] || 1.0e-8
    initial_scale = opts[:initial_scale] || 0.0

    stateful(combinator, &init_scale_by_rms(&1, initial_scale), &apply_scale_by_rms(&1, &2, [decay: decay, eps: eps]))
  end

  def scale_by_rms(opts) do
    decay = opts[:decay] || 0.9
    eps = opts[:eps] || 1.0e-8
    initial_scale = opts[:initial_scale] || 0.0

    stateful(&init_scale_by_rms(&1, initial_scale), &apply_scale_by_rms(&1, &2, [decay: decay, eps: eps]))
  end

  defnp init_scale_by_rms(params, scale) do
    nu = fulls_like(params, scale)
    {nu}
  end

  defnp apply_scale_by_rms(x, {nu}, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, eps: 1.0e-8)
    decay = opts[:decay]
    eps = opts[:eps]

    nu = update_moment(x, nu, decay, 2)

    x = transform({x, nu, eps}, fn {x, nu, eps} ->
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
  def scale_by_belief(combinator, opts) do
    b1 = opts[:b1] || 0.9
    b2 = opts[:b2] || 0.999
    eps = opts[:eps] || 0.0
    eps_root = opts[:eps_root] || 1.0e-16

    stateful(combinator, &init_scale_by_belief/1, &apply_scale_by_belief(&1, &2, [b1: b1, b2: b2, eps: eps, eps_root: eps_root]))
  end

  def scale_by_belief(opts) do
    b1 = opts[:b1] || 0.9
    b2 = opts[:b2] || 0.999
    eps = opts[:eps] || 0.0
    eps_root = opts[:eps_root] || 1.0e-16

    stateful(&init_scale_by_belief/1, &apply_scale_by_belief(&1, &2, [b1: b1, b2: b2, eps: eps, eps_root: eps_root]))
  end

  defnp init_scale_by_belief(params) do
    mus = zeros_like(params)
    nus = zeros_like(params)
    count = Nx.tensor(0)
    {mus, nus, count}
  end

  defnp apply_scale_by_belief(x, {mu, nu, count}, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 0.0, eps_root: 1.0e-16)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)
    nu = update_moment(x, nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x = transform({mu_hat, nu_hat, eps, eps_root}, fn {mu_hat, nu_hat, eps, eps_root} ->
      mu_hat
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(nu_hat))
      |> Enum.map(fn {z, t} -> (1 / (Nx.sqrt(t + eps_root) + eps)) * z end)
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
  def scale_by_stddev(combinator, opts) do
    decay = opts[:decay] || 0.9
    eps = opts[:eps] || 1.0e-8
    initial_scale = opts[:initial_scale] || 0.0

    stateful(combinator, &init_scale_by_stddev(&1, initial_scale), &apply_scale_by_stddev(&1, &2, [decay: decay, eps: eps]))
  end

  def scale_by_stddev(opts) do
    decay = opts[:decay] || 0.9
    eps = opts[:eps] || 1.0e-8
    initial_scale = opts[:initial_scale] || 0.0

    stateful(&init_scale_by_stddev(&1, initial_scale), &apply_scale_by_stddev(&1, &2, [decay: decay, eps: eps]))
  end

  defnp init_scale_by_stddev(params, value) do
    mu = zeros_like(params)
    nu = fulls_like(params, value)
    {mu, nu}
  end

  defnp apply_scale_by_stddev(x, {mu, nu}, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, eps: 1.0e-8)
    decay = opts[:decay]
    eps = opts[:eps]

    mu = update_moment(x, mu, decay, 1)
    nu = update_moment(x, nu, decay, 2)

    x = transform({x, mu, nu, eps}, fn {x, mu, nu, eps} ->
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
  def scale_by_schedule(combinator, schedule_fn) when is_function(schedule_fn) do
    stateful(combinator, &init_scale_by_schedule/1, &apply_scale_by_schedule(&1, &2, schedule_fn))
  end

  def scale_by_schedule(schedule_fn) when is_function(schedule_fn) do
    stateful(&init_scale_by_schedule/1, &apply_scale_by_schedule(&1, &2, schedule_fn))
  end

  defnp init_scale_by_schedule(_) do
    {Nx.tensor(0)}
  end

  defnp apply_scale_by_schedule(x, {count}, schedule_fn) do
    step_size = schedule_fn.(count)
    transform({x, step_size}, fn {x, step_size} ->
      x
      |> Tuple.to_list()
      |> Enum.map(fn x -> x * step_size end)
      |> List.to_tuple()
    end)
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
  def scale_by_radam(combinator, opts) do
    b1 = opts[:b1] || 0.9
    b2 = opts[:b2] || 0.999
    eps = opts[:eps] || 1.0e-8
    eps_root = opts[:eps_root] || 0.0
    threshold = opts[:threshold] || 5.0

    stateful(combinator, &init_scale_by_radam/1, &apply_scale_by_radam(&1, &2, [b1: b1, b2: b2, eps: eps, eps_root: eps_root, threshold: threshold]))
  end

  def scale_by_radam(opts) do
    b1 = opts[:b1] || 0.9
    b2 = opts[:b2] || 0.999
    eps = opts[:eps] || 1.0e-8
    eps_root = opts[:eps_root] || 0.0
    threshold = opts[:threshold] || 5.0

    stateful(&init_scale_by_radam/1, &apply_scale_by_radam(&1, &2, [b1: b1, b2: b2, eps: eps, eps_root: eps_root, threshold: threshold]))
  end

  defnp init_scale_by_radam(params) do
    mu = zeros_like(params)
    nu = zeros_like(params)
    count = Nx.tensor(0)
    {mu, nu, count}
  end

  defnp apply_scale_by_radam(x, {mu, nu, count}, opts \\ []) do
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

    nu_hat = transform({nu, eps, eps_root}, fn {nu, eps, eps_root} ->
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

  Returns `{traced_inputs, updated_trace}`.

  ## Options

    * `:decay` - decay rate for tracing past updates. Defaults
      to `0.9`
    * `:nesterov` - whether to use Nesterov momentum. Defaults
      to `false`

  """
  def trace(combinator, opts) do
    decay = opts[:decay] || 0.9
    nesterov = opts[:nesterov] || false

    stateful(combinator, &init_trace/1, &apply_trace(&1, &2, [decay: decay, nesterov: nesterov]))
  end

  def trace(opts) do
    decay = opts[:decay] || 0.9
    nesterov = opts[:nesterov] || false

    stateful(&init_trace/1, &apply_trace(&1, &2, [decay: decay, nesterov: nesterov]))
  end

  defnp init_trace(params) do
    trace = zeros_like(params)
    {trace}
  end

  defnp apply_trace(x, {trace}, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, nesterov: false)
    decay = opts[:decay]
    nesterov? = to_predicate(opts[:nesterov])

    update_trace = transform({x, trace, decay}, fn {x, trace, decay} ->
      x
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(trace))
      |> Enum.map(fn {g, t} -> t * decay + g end)
      |> List.to_tuple()
    end)

    x =
      if nesterov? do
        transform({x, update_trace, decay}, fn {x, trace, decay} ->
          x
          |> Tuple.to_list()
          |> Enum.zip(Tuple.to_list(trace))
          |> Enum.map(fn {g, t} -> t * decay + g end)
          |> List.to_tuple()
        end)
      else
        update_trace
      end

    {x, {update_trace}}
  end

  @doc """
  Clips input between -delta and delta.

  ## Options

    * `:delta` - maximum absolute value of the input. Defaults
      to `2.0`
  """
  def clip(combinator, opts) do
    delta = opts[:delta] || 2.0
    stateless(combinator, &apply_clip(&1, [delta: delta]))
  end

  def clip(opts) do
    delta = opts[:delta] || 2.0
    stateless(&apply_clip(&1, [delta: delta]))
  end

  defnp apply_clip(x, opts \\ []) do
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
  def clip_by_global_norm(combinator, opts) do
    max_norm = opts[:max_norm] || 1.0
    stateless(combinator, &apply_clip_by_global_norm(&1, [max_norm: max_norm]))
  end

  def clip_by_global_norm(opts) do
    max_norm = opts[:max_norm] || 1.0
    stateless(&apply_clip_by_global_norm(&1, [max_norm: max_norm]))
  end

  defnp apply_clip_by_global_norm(x, opts \\ []) do
    opts = keyword!(opts, max_norm: 1.0)
    max_norm = opts[:max_norm]

    g_norm = apply_map(x, fn z -> Nx.sqrt(Nx.sum(Nx.power(z, 2))) end)

    transform({x, g_norm, max_norm}, fn {x, g_norm, max_norm} ->
      x
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(g_norm))
      |> Enum.map(fn {z, g} -> Nx.select(Nx.less(g, max_norm), z, (z / g) * max_norm) end)
      |> List.to_tuple()
    end)
  end

  @doc """
  Centralize input.
  """
  def centralize(combinator) do
    stateless(combinator, &apply_centralize/1)
  end

  def centralize() do
    stateless(&apply_centralize/1)
  end

  defnp apply_centralize(x) do
    apply_map(x, fn z -> -Nx.mean(z) + z end)
  end

  ## Helpers

  defp stateless({parent_init_fn, parent_apply_fn}, apply_fn) do
    apply_fn =
      fn updates, state ->
        {updates, state} = parent_apply_fn.(updates, state)
        {apply_fn.(updates), state}
      end
    {parent_init_fn, apply_fn}
  end

  defp stateless(apply_fn) do
    init_fn = &empty_state/1
    apply_fn =
      fn updates, state ->
        updates = apply_fn.(updates)
        {updates, state}
      end

    {init_fn, apply_fn}
  end

  defp stateful({parent_init_fn, parent_apply_fn}, init_fn, apply_fn) do
    init_fn =
      fn params ->
        state = parent_init_fn.(params)
        Tuple.insert_at(state, 0, init_fn.(params))
      end

    apply_fn =
      fn updates, state ->
        this_state = elem(state, 0)
        other_state = Tuple.delete_at(state, 0)
        {updates, new_other_state} = parent_apply_fn.(updates, other_state)
        {updates, new_this_state} = apply_fn.(updates, this_state)
        {updates, Tuple.insert_at(new_other_state, 0, new_this_state)}
      end

    {init_fn, apply_fn}
  end

  defp stateful(init_fn, apply_fn) do
    init_fn = fn params -> {init_fn.(params)} end
    apply_fn =
      fn updates, state ->
        apply_fn.(updates, elem(state, 0))
      end

    {init_fn, apply_fn}
  end

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

  defnp empty_state(_) do
    {}
  end
end
