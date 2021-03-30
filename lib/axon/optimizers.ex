defmodule Axon.Optimizers do
  @moduledoc """
  Common optimizers.
  """
  alias Axon.Updates

  @doc """
  SGD optimizer.
  """
  def sgd(learning_rate) do
    Updates.scale(-learning_rate)
  end

  @doc """
  Adam optimizer.
  """
  def adam(learning_rate, opts \\ []) do
    Updates.scale_by_adam(opts)
    |> Updates.scale(-learning_rate)
  end
end