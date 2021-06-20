defmodule Axon.MixedPrecision.Policy do
  @moduledoc false

  # Represents a mixed precision policy for a single layer
  defstruct params: {:f, 32}, compute: {:f, 32}, output: {:f, 32}
end
