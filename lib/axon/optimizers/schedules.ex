defmodule Axon.Optimizers.Schedules do
  @moduledoc """
  A collection of common learning-rate schedules.
  """

  import Nx.Defn

  @doc """
  An exponential decay scheduler.
  """
  defn exponential_decay(step, initial_learning_rate, decay_steps, decay_rate) do
    initial_learning_rate * Nx.power(decay_rate, step / decay_steps)
  end
end