defmodule Axon.Normalization do
  @moduledoc """
  Normalization calculations.
  """
  import Nx.Defn

  defn batch_norm_stats(x, old_mean, old_var, opts \\ []) do
    opts = keyword!(opts, momentum: 0.5)
    beta = 1 - opts[:momentum]
    mean = Nx.mean(x, axes: [0], keep_axes: true)
    mean2 = Nx.mean(Nx.power(mean, 2), axes: [0], keep_axes: true)
    var = mean2 - Nx.power(mean, 2)

    {stop_grad(Nx.add(old_mean, beta * (Nx.squeeze(mean) - old_mean))),
     stop_grad(Nx.add(old_var, beta * (Nx.squeeze(var) - old_var)))}
  end
end
