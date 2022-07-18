defmodule Axon.None do
  @moduledoc """
  Represents a missing value of an optional node.

  See `Axon.input/2` and `Axon.optional/2` for more details.
  """

  @derive {Inspect, except: [:__propagate__]}
  @derive {Nx.Container, containers: []}
  defstruct __propagate__: true
end
