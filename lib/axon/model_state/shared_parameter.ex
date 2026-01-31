defmodule Axon.ModelState.SharedParameter do
  # Represents a tied or shared parameter for layers whose
  # weights are connected but don't necessarily perform the
  # same operation. This implements the Nx.Container behavior
  # and contains an access path to the parameter that holds the
  # original weight.

  @moduledoc false

  @derive {Nx.Container, containers: [], keep: [:path, :transform]}
  defstruct [:path, :transform]

  def new(path, opts \\ []) do
    %__MODULE__{
      path: path,
      transform: Keyword.get(opts, :transform)
    }
  end
end
