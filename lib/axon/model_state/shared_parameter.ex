defmodule Axon.ModelState.SharedParameter do
  @moduledoc false

  # Represents a tied or shared parameter for layers who's
  # weights are connected but don't necessarily perform the
  # same operation. This implements the Nx.Container behavior
  # and contains an access path to the parameter that holds the
  # original weight

  @derive {
    Nx.Container,
    keep: [:path], containers: []
  }
  defstruct [:path]

  def new(path) do
    %__MODULE__{path: path}
  end
end
