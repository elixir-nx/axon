defmodule Axon.Schedules do
  @moduledoc false
  import Nx.Defn

  @doc """
  Linear decay schedule.

  ## Options

    * `:warmup` - scheduler warmup steps. Defaults to `0`

    * `:steps` - total number of decay steps. Defaults to `1000`
  """
  @deprecated "Use Polaris.Schedules.linear_decay/2 instead"
  def linear_decay(init_value, opts \\ []) do
    &apply_linear_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_linear_decay(step, opts \\ []) do
    opts =
      keyword!(opts,
        init_value: 1.0e-2,
        warmup: 0,
        steps: 1000
      )

    if step < opts[:warmup] do
      step / Nx.max(1, opts[:warmup])
    else
      Nx.max(0.0, (opts[:steps] - step) / Nx.max(1, opts[:steps] - opts[:warmup]))
    end
  end

  @doc ~S"""
  Exponential decay schedule.

  $$\gamma(t) = \gamma_0 * r^{\frac{t}{k}}$$

  ## Options

    * `:decay_rate` - rate of decay. $r$ in above formulation.
      Defaults to `0.95`

    * `:transition_steps` - steps per transition. $k$ in above
      formulation. Defaults to `10`

    * `:transition_begin` - step to begin transition. Defaults to `0`

    * `:staircase` - discretize outputs. Defaults to `false`

  """
  @deprecated "Use Polaris.Schedules.exponential_decay/2 instead"
  def exponential_decay(init_value, opts \\ []) do
    &apply_exponential_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_exponential_decay(step, opts \\ []) do
    opts =
      keyword!(opts,
        init_value: 1.0e-2,
        decay_rate: 0.95,
        transition_steps: 10,
        transition_begin: 0,
        staircase: false
      )

    init_value = opts[:init_value]
    rate = opts[:decay_rate]
    staircase? = opts[:staircase]
    k = opts[:transition_steps]
    start = opts[:transition_begin]

    t = Nx.subtract(step, start)

    p =
      if staircase? do
        t
        |> Nx.divide(k)
        |> Nx.floor()
      else
        t
        |> Nx.divide(k)
      end

    decayed_value =
      rate
      |> Nx.pow(p)
      |> Nx.multiply(init_value)

    Nx.select(
      Nx.less_equal(t, 0),
      init_value,
      decayed_value
    )
  end

  @doc ~S"""
  Cosine decay schedule.

  $$\gamma(t) = \gamma_0 * \left(\frac{1}{2}(1 - \alpha)(1 + \cos\pi \frac{t}{k}) + \alpha\right)$$

  ## Options

    * `:decay_steps` - number of steps to apply decay for.
      $k$ in above formulation. Defaults to `10`

    * `:alpha` - minimum value of multiplier adjusting learning rate.
      $\alpha$ in above formulation. Defaults to `0.0`

  ## References

    * [SGDR: Stochastic Gradient Descent with Warm Restarts](https://openreview.net/forum?id=Skq89Scxx&noteId=Skq89Scxx)

  """
  @deprecated "Use Polaris.Schedules.cosine_decay/2 instead"
  def cosine_decay(init_value, opts \\ []) do
    &apply_cosine_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_cosine_decay(step, opts \\ []) do
    opts = keyword!(opts, init_value: 1.0e-2, decay_steps: 10, alpha: 0.0)
    init_value = opts[:init_value]
    decay_steps = opts[:decay_steps]
    alpha = opts[:alpha]

    step
    |> Nx.min(decay_steps)
    |> Nx.divide(decay_steps)
    |> Nx.multiply(3.1415926535897932384626433832795028841971)
    |> Nx.cos()
    |> Nx.add(1)
    |> Nx.divide(2)
    |> Nx.multiply(1 - alpha)
    |> Nx.add(alpha)
    |> Nx.multiply(init_value)
  end

  @doc ~S"""
  Constant schedule.

  $$\gamma(t) = \gamma_0$$

  """
  @deprecated "Use Polaris.Schedules.constant/2 instead"
  def constant(init_value, opts \\ []) do
    &apply_constant(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_constant(_step, opts \\ []) do
    opts = keyword!(opts, init_value: 0.01)
    opts[:init_value]
  end

  @doc ~S"""
  Polynomial schedule.

  $$\gamma(t) = (\gamma_0 - \gamma_n) * (1 - \frac{t}{k})^p$$

  ## Options

    * `:end_value` - end value of annealed scalar. $\gamma_n$ in above formulation.
      Defaults to `1.0e-3`

    * `:power` - power of polynomial. $p$ in above formulation. Defaults to `2`

    * `:transition_steps` - number of steps over which annealing takes place.
      $k$ in above formulation. Defaults to `10`

  """
  @deprecated "Use Polaris.Schedules.polynomial_decay/2 instead"
  def polynomial_decay(init_value, opts \\ []) do
    &apply_polynomial_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_polynomial_decay(step, opts \\ []) do
    opts =
      keyword!(opts,
        init_value: 1.0e-2,
        end_value: 1.0e-3,
        power: 2,
        transition_steps: 10,
        transition_begin: 0
      )

    init_value = opts[:init_value]
    end_value = opts[:end_value]
    start = opts[:transition_begin]
    k = opts[:transition_steps]
    p = opts[:power]

    step
    |> Nx.subtract(start)
    |> Nx.clip(0, k)
    |> Nx.divide(k)
    |> Nx.negate()
    |> Nx.add(1)
    |> Nx.pow(p)
    |> Nx.multiply(Nx.subtract(init_value, end_value))
    |> Nx.add(end_value)
  end
end
