defmodule Axon.Updates do
  @moduledoc """
  Common update methods.
  """
  import Nx.Defn

  @doc """
  Scales input by a fixed step-size.
  """
  defn scale(x, opts \\ []) do
    opts = keyword!(opts, [:step])
    x
    |> Nx.multiply(opts[:step])
  end

  @doc """
  Scale input according to Adam algorithm.
  """
  defn scale_by_adam(x, mu, nu, count, opts \\ []) do
    opts = keyword!(opts, [b1: 0.9, b2: 0.999, eps: 1.0e-3, eps_root: 0.0])
    mu = update_moment(x, mu, opts[:b1], 1)
    nu = update_moment(x, nu, opts[:b2], 2)
    mu_hat = bias_correction(mu, opts[:b1], count + 1)
    nu_hat = bias_correction(nu, opts[:b2], count + 1)

    x = Nx.divide(mu_hat, Nx.sqrt(nu_hat + opts[:eps_root]) + opts[:eps])
    {x, mu, nu}
  end

  @doc """
  Scale the input by the root of all prior squared inputs.
  """
  defn scale_by_rss(x, sum_of_squares, opts \\ []) do
    opts = keyword!(opts, [eps: 1.0e-7])
    sum_of_squares = Nx.square(x) + sum_of_squares
    inv_sqrt_x_square = Nx.select(Nx.greater(sum_of_squares, 0), Nx.rsqrt(sum_of_squares + opts[:eps]), 0.0)

    x = inv_sqrt_x_square * x

    {x, sum_of_squares}
  end

  @doc """
  Scale the input by the root of the EMA of the squared inputs.
  """
  defn scale_by_rms(x, nu, opts \\ []) do
    opts = keyword!(opts, [decay: 0.9, eps: 1.0e-8])
    nu = update_moment(x, nu, opts[:decay], 2)

    x = x * Nx.rsqrt(nu + opts[:eps])

    {x, nu}
  end

  @doc """
  Scale the input according to the AdaBelief algorithm.
  """
  defn scale_by_belief(x, mu, nu, count, opts \\ []) do
    opts = keyword!(opts, [b1: 0.9, b2: 0.999, eps: 0.0, eps_root: 1.0e-16])
    mu = update_moment(x, mu, opts[:b1], 1)
    pred_error = x - mu
    nu = update_moment(pred_error, nu, opts[:b2], 2)
    mu_hat = bias_correction(mu, opts[:b1], count + 1)
    nu_hat = bias_correction(nu, opts[:b2], count + 1)

    Nx.divide(mu, Nx.sqrt(nu + opts[:eps_root]) + opts[:eps])
  end

  @doc """
  Scale the input by the root of the centered EMA of squared inputs.
  """
  defn scale_by_stddev(x, mu, nu, opts \\ []) do
    opts = keyword!(opts, [decay: 0.9, eps: 1.0e-8])
    mu = update_moment(x, mu, opts[:decay], 1)
    nu = update_moment(x, nu, opts[:decay], 2)

    x = x * Nx.rsqrt(nu - Nx.power(mu, 2) + opts[:eps])
  end

  @doc """
  Scale the input by the given schedule function.

  TODO: This should be defn
  """
  def scale_by_schedule(x, count, schedule_fn) when is_function(schedule_fn) do
    step_size = schedule_fn.(count)
    Nx.multiply(x, step_size)
  end

  @doc """
  Scale the input by trust ratio.
  """
  defn scale_by_trust_ratio(x, g, opts \\ []) do
    opts = keyword!(opts, [min_norm: 0.0])
    param_norm = safe_norm(x, opts[:min_norm])
    update_norm = safe_norm(g, opts[:min_norm])
    trust_ratio = Nx.divide(param_norm, update_norm)

    zero_norm = Nx.logical_or(Nx.equal(param_norm, 0), Nx.equal(update_norm, 0))

    safe_trust_ratio = Nx.select(zero_norm, 1, trust_ratio)

    Nx.multiply(x, safe_trust_ratio)
  end

  @doc """
  Trace updates.
  """
  defn trace(x, trace, opts \\ []) do
    opts = keyword!(opts, decay: 0.9, nesterov: false)
    update_trace = x + opts[:decay] * trace

    x = transform({x, update_trace, opts[:nesterov], opts},
      fn
        {_, t, false, _} -> t
        {g, t, true, opts} -> g + opts[:decay] * t
      end
    )

    {x, update_trace}
  end

  @doc """
  Clips input between -delta and delta
  """
  defn clip(x, opts \\ []) do
    opts = keyword!(opts, [delta: 2.0])
    Nx.clip(x, -opts[:delta], opts[:delta])
  end

  @doc """
  Clips input by global norm.
  """
  defn clip_by_global_norm(x, max_norm) do
    g_norm =
      x
      |> Nx.power(2)
      |> Nx.sum()
      |> Nx.sqrt()

    Nx.select(Nx.less(g_norm, max_norm), x, (x / g_norm) * max_norm)
  end

  @doc """
  Centralize input.
  """
  defn centralize(x) do
    x
    |> Nx.mean()
    |> Nx.negate()
    |> Nx.add(x)
  end

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

end