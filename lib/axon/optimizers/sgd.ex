defmodule Axon.Optimizers.SGD do
  @moduledoc """
  Stochastic-Gradient Descent optimizer.
  """

  import Nx.Defn

  @behaviour Axon.Optimizers

  @impl true
  def initialize(), do: :ok

  @impl true
  def learning_rate(_epoch), do: 1.0e-2

  defn update_parameter(parameter, gradient, step) do
    parameter - gradient * step
  end

  @impl true
  def apply_gradients(parameters, gradients, epoch) do
    lr = learning_rate(epoch)
    parameters
    |> Enum.zip(gradients)
    |> Enum.map(fn {param, grad} -> update_parameter(param, grad, lr) end)
  end
end