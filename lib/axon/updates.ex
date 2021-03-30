defmodule Axon.Updates do
  @moduledoc """
  Parameter update methods.

  Update methods transform the input tensor in some way,
  usually by scaling or shifting the input with respect
  to some input state. Update methods are typically composed
  to create more advanced optimization methods such as AdaGrad
  or Adam; however, they can also be applied to model parameters.

  These methods are the building blocks of common gradient descent
  methods. For example, a basic gradient descent algorithm
  would look something like:

      g_param = grad(param, loss_fun(...))
      param - 0.01 * g_param

  With these methods, you can write that as:

      g_param = grad(param, loss_fun(...))
      param + scale(g_param, step: -0.01)

  The benefits of this module are more easily seen as optimizers
  get more complex. For example, you can implement the Adam optimizer:

      g_param = grad(param, loss_fun(...))
      {updates, mu_new, nu_new} =
        g_param
        |> scale_by_adam(mu, nu)

      g_param + scale(updates, step: -0.01)

  In the example above, by `mu` and `nu` are the 1st and 2nd moment
  respectively. Normally, they would be initialized and maintained
  as optimizer parameters; however, because these are stateless
  implementations, they are updated along with the input updates.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.

  """
  import Nx.Defn
  import Axon.Shared

  @doc ~S"""
  Scales input by a fixed step size.

  $$f(x_i) = \alpha x_i$$

  ## Examples

      iex> Axon.Updates.scale(Nx.tensor([-1.0, 0.0, 1.0]), 0.01)
      #Nx.Tensor<
        f32[3]
        [-0.01, 0.0, 0.01]
      >

      iex> Axon.Updates.scale(Nx.tensor([[-5, 2, 1, 4, 2], [0, 2, 1, 4, 1]]), 0.1)
      #Nx.Tensor<
        f32[2][5]
        [
          [-0.5, 0.2, 0.1, 0.4, 0.2],
          [0.0, 0.2, 0.1, 0.4, 0.1]
        ]
      >

  """
  def scale({init_fn, apply_fn}, step_size) do
    {_, scale_apply_fn} = scale(step_size)
    {init_fn, &scale_apply_fn.(apply_fn.(&1, &2))}
  end

  def scale(step_size) do
    init_fn = &empty/1
    apply_fn = &apply_scale(&1, step_size)

    {init_fn, apply_fn}
  end

  defnp apply_scale({x, state}, step) do
    updates = transform({x, step},
      fn {updates, step} ->
        updates
        |> Tuple.to_list()
        |> Enum.map(&Nx.multiply(&1, step))
        |> List.to_tuple()
      end
    )

    {updates, state}
  end

  @doc """
  Scales input according to Adam algorithm.

  Returns `{scaled_input, updated_mu, update_nu}`.

  ## Options

      * `:b1` - first moment decay. Defaults to `0.9`
      * `:b2` - second moment decay. Defaults to `0.999`
      * `:eps` - numerical stability term. Defaults to `1.0e-8`
      * `:eps_root` - numerical stability term. Defaults to `0.0`

  ## References

    * [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

  """
  def scale_by_adam(opts \\ []) do
    b1 = opts[:b1] || 0.9
    b2 = opts[:b2] || 0.999
    eps = opts[:eps] || 1.0e-6
    eps_root = opts[:eps_root] || 1.0e-5

    init_fn = &init_scale_by_adam(&1)
    apply_fn = &apply_scale_by_adam(&1, &2, [b1: b1, b2: b2, eps: eps, eps_root: eps_root])

    {init_fn, apply_fn}
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

    mu = transform({x, mu, b1}, fn {x, mu, b1} ->
      x
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(mu))
      |> Enum.map(fn {g, z} -> update_moment(g, z, b1, 1) end)
      |> List.to_tuple()
    end)

    nu = transform({x, nu, b2}, fn {x, nu, b2} ->
      x
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(nu))
      |> Enum.map(fn {g, z} -> update_moment(g, z, b2, 2) end)
      |> List.to_tuple()
    end)

    mu_hat = transform({mu, b1, count}, fn {mu, b1, count} ->
      mu
      |> Tuple.to_list()
      |> Enum.map(&bias_correction(&1, b1, count + 1))
      |> List.to_tuple()
    end)

    nu_hat = transform({nu, b2, count}, fn {nu, b2, count} ->
      nu
      |> Tuple.to_list()
      |> Enum.map(&bias_correction(&1, b2, count + 1))
      |> List.to_tuple()
    end)

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

  Returns `{scaled_input, updated_sum_of_squares}`.

  ## Options

      * `:eps` - numerical stability term. Defaults to `1.0e-7`

  """
  defn scale_by_rss(x, sum_of_squares, opts \\ []) do
    opts = keyword!(opts, eps: 1.0e-7)
    eps = opts[:eps]

    sum_of_squares =
      x
      |> Nx.power(2)
      |> Nx.add(sum_of_squares)

    inv_sqrt_squares =
      sum_of_squares
      |> Nx.add(eps)
      |> Nx.rsqrt()

    inv_sqrt_x_square = Nx.select(Nx.greater(sum_of_squares, 0), inv_sqrt_squares, 0.0)

    x =
      x
      |> Nx.multiply(inv_sqrt_x_square)

    {x, sum_of_squares}
  end

  @doc """
  Scales input by the root of the EMA of squared inputs.

  Returns `{scaled_input, updated_nu}`

  ## Options

      * `:decay` - EMA decay rate. Defaults to `0.9`
      * `:eps` - numerical stability term. Defaults to `1.0e-8`

  ## References

    * [Overview of mini-batch gradient descent](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  """
  defn scale_by_rms(x, nu, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, eps: 1.0e-8)
    decay = opts[:decay]
    eps = opts[:eps]

    nu = update_moment(x, nu, decay, 2)

    x =
      nu
      |> Nx.add(eps)
      |> Nx.rsqrt()
      |> Nx.multiply(x)

    {x, nu}
  end

  @doc """
  Scales input according to the AdaBelief algorithm.

  Returns `{scaled_input, update_mu, updated_nu}`.

  ## Options

      * `:b1` - first moment decay. Defaults to `0.9`
      * `:b2` - second moment decay. Defaults to `0.999`
      * `:eps` - numerical stability term. Defaults to `0.0`
      * `:eps_root` - numerical stability term. Defaults to `1.0e-16`

  ## References

    * [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468)

  """
  defn scale_by_belief(x, mu, nu, count, opts \\ []) do
    opts = keyword!(opts, b1: 0.9, b2: 0.999, eps: 0.0, eps_root: 1.0e-16)
    b1 = opts[:b1]
    b2 = opts[:b2]
    eps = opts[:eps]
    eps_root = opts[:eps_root]

    mu = update_moment(x, mu, b1, 1)

    nu =
      x
      |> Nx.subtract(mu)
      |> update_moment(nu, b2, 2)

    mu_hat = bias_correction(mu, b1, count + 1)
    nu_hat = bias_correction(nu, b2, count + 1)

    x =
      nu_hat
      |> Nx.add(eps_root)
      |> Nx.sqrt()
      |> Nx.add(eps)
      |> reciprocal()
      |> Nx.multiply(mu_hat)

    {x, mu, nu}
  end

  @doc """
  Scales input by the root of the centered EMA of squared inputs.

  Returns `{scaled_input, updated_mu, updated_nu}`

  ## Options

      * `:decay` - EMA decay rate. Defaults to `0.9`
      * `:eps` - numerical stability term. Defaults to `1.0e-8`

  ## References

    * [Overview of mini-batch gradient descent](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  """
  defn scale_by_stddev(x, mu, nu, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, eps: 1.0e-8)
    decay = opts[:decay]
    eps = opts[:eps]

    mu = update_moment(x, mu, decay, 1)
    nu = update_moment(x, nu, decay, 2)

    x =
      mu
      |> Nx.power(2)
      |> Nx.negate()
      |> Nx.add(nu)
      |> Nx.add(eps)
      |> Nx.rsqrt()
      |> Nx.multiply(x)

    {x, mu, nu}
  end

  @doc """
  Scales input using the given schedule function.
  """
  def scale_by_schedule(x, count, schedule_fn) when is_function(schedule_fn) do
    step_size = schedule_fn.(count)
    Nx.multiply(x, step_size)
  end

  @doc """
  Scales input by trust ratio.

  Returns `scaled_input`.

  ## Options

      * `:min_norm` - minimum norm for inputs. Defaults to `0.0`

  ## References

      [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)

  """
  defn scale_by_trust_ratio(x, g, opts \\ []) do
    opts = keyword!(opts, min_norm: 0.0)
    min_norm = opts[:min_norm]

    param_norm = safe_norm(x, min_norm)
    update_norm = safe_norm(g, min_norm)
    zero_norm = Nx.logical_or(Nx.equal(param_norm, 0), Nx.equal(update_norm, 0))

    trust_ratio = Nx.divide(param_norm, update_norm)
    safe_trust_ratio = Nx.select(zero_norm, 1, trust_ratio)

    Nx.multiply(x, safe_trust_ratio)
  end

  @doc """
  Scale input according to the Rectified Adam algorithm.

  Returns `{scaled_input, updated_mu, updated_nu}`.

  ## Options

      * `:b1` - first moment decay. Defaults to `0.9`
      * `:b2` - second moment decay. Defaults to `0.999`
      * `:eps` - numerical stability term. Defaults to `1.0e-8`
      * `:eps_root` - numerical stability term. Defaults to `0.0`
      * `:threshold` - threshold for variance. Defaults to `5.0`

  ## References

    * [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)

  """
  defn scale_by_radam(x, mu, nu, count, opts \\ []) do
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

    {x, mu, nu}
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
      nu
      |> Nx.add(eps_root)
      |> Nx.sqrt()
      |> Nx.add(eps)

    top
    |> Nx.divide(bottom)
    |> Nx.sqrt()
    |> Nx.multiply(mu)
    |> Nx.divide(nu_hat)
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
  defn trace(x, trace, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, nesterov: false)
    decay = opts[:decay]
    nesterov? = to_predicate(opts[:nesterov])

    update_trace =
      trace
      |> Nx.multiply(decay)
      |> Nx.add(x)

    x =
      if nesterov? do
        update_trace
        |> Nx.multiply(decay)
        |> Nx.add(x)
      else
        update_trace
      end

    {x, update_trace}
  end

  @doc """
  Clips input between -delta and delta.

  ## Options

    * `:delta` - maximum absolute value of the input. Defaults
      to `2.0`

  ## Examples

      iex> Axon.Updates.clip(Nx.tensor([-3.0, -2.5, 0.0, 2.0, 1.0]))
      #Nx.Tensor<
        f32[5]
        [-2.0, -2.0, 0.0, 2.0, 1.0]
      >

      iex> Axon.Updates.clip(Nx.tensor([-5, -3, -1, 0, 2, 10, 4]), delta: 2.5)
      #Nx.Tensor<
        f32[7]
        [-2.5, -2.5, -1.0, 0.0, 2.0, 2.5, 2.5]
      >

  """
  defn clip(x, opts \\ []) do
    opts = keyword!(opts, delta: 2.0)
    delta = opts[:delta]
    Nx.clip(x, -delta, delta)
  end

  @doc """
  Clips input using input global norm.

  ## Options

    * `:max_norm` - maximum norm value of input. Defaults to
      `1.0`

  ## Examples

      iex> Axon.Updates.clip_by_global_norm(Nx.tensor([-3.0, -2.5, 0.0, 2.0, 1.0]))
      #Nx.Tensor<
        f32[5]
        [-0.6666666865348816, -0.5555555820465088, 0.0, 0.4444444477558136, 0.2222222238779068]
      >

  """
  defn clip_by_global_norm(x, opts \\ []) do
    opts = keyword!(opts, max_norm: 1.0)
    max_norm = opts[:max_norm]

    g_norm =
      x
      |> Nx.power(2)
      |> Nx.sum()
      |> Nx.sqrt()

    x_norm =
      x
      |> Nx.divide(g_norm)
      |> Nx.multiply(max_norm)

    Nx.select(Nx.less(g_norm, max_norm), x, x_norm)
  end

  @doc """
  Centralize input.

  ## Examples

    iex> Axon.Updates.centralize(Nx.tensor([2.0, -3.0, 1.0, 2.0, -3.0]))
    #Nx.Tensor<
      f32[5]
      [2.2, -2.8, 1.2, 2.2, -2.8]
    >

    iex> Axon.Updates.centralize(Nx.tensor([[1.0, -2.0, 5.0, 10.0], [2.0, 3.0, 4.0, 5.0]]))
    #Nx.Tensor<
      f32[2][4]
      [
        [-2.5, -5.5, 1.5, 6.5],
        [-1.5, -0.5, 0.5, 1.5]
      ]
    >

  """
  defn centralize(x) do
    x
    |> Nx.mean()
    |> Nx.negate()
    |> Nx.add(x)
  end

  ## Helpers

  defnp update_moment(x, moment, decay, order) do
    (1 - decay) * Nx.power(x, order) + Nx.multiply(decay, moment)
  end

  defnp bias_correction(moment, decay, count) do
    correction = 1 - Nx.power(decay, count)
    Nx.divide(moment, correction)
  end

  defnp safe_norm(x, min_norm) do
    norm = Nx.norm(x)
    x = Nx.select(Nx.less(norm, min_norm), 1, x)
    Nx.select(Nx.less(norm, min_norm), min_norm, Nx.norm(x))
  end

  defnp empty(_) do
    {}
  end
end
