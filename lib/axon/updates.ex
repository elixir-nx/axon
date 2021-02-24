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
    opts = keyword!(opts, [b1: 0.9, b2: 0.999, eps: 1.0e-8, eps_root: 0.0])
    mu = update_moment(x, mu, opts[:b1], 1)
    nu = update_moment(x, nu, opts[:b2], 2)
    mu_hat = bias_correction(mu, opts[:b1], count + 1)
    nu_hat = bias_correction(nu, opts[:b2], count + 1)

    x = Nx.divide(mu_hat, Nx.sqrt(nu_hat + opts[:eps_root]) + opts[:eps])
    {x, mu, nu}
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

end