defmodule Axon.MixedPrecision.Policy do
  @moduledoc false

  # Represents a mixed precision policy for a single layer
  defstruct [:params, :compute, :output]
end
