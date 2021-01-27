defmodule Axon.Optimizers do
  @moduledoc """
  Optimizer behaviour.
  """

  @type tensor :: Nx.Tensor.t()

  @callback initialize() :: :ok
  @callback learning_rate(epoch :: integer) :: tensor
  @callback apply_gradients(
              params :: Enum.t(tensor),
              gradients :: Enum.t(tensor),
              epoch :: number
            ) :: Enum.t(tensor)
end
