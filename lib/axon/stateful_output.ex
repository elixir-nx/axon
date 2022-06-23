defmodule Axon.StatefulOutput do
  @moduledoc """
  Container for returning stateful outputs from Axon layers.

  Some layers, such as `Axon.batch_norm/2`, keep a running internal
  state which is updated continuously at train time and used statically
  at inference time. In order for the Axon compiler to differentiate
  ordinary layer outputs from internal state, you must mark output
  as stateful.

  Stateful Outputs consist of two fields:

      :output - Actual layer output to be forwarded to next layer
      :state - Internal layer state to be tracked and updated

  `:output` is simply forwarded to the next layer. `:state` is aggregated
  with other stateful outputs, and then is treated specially by internal
  Axon training functions such that update state parameters reflect returned
  values from stateful outputs.

  `:state` must be a map with keys that map directly to layer internal
  state names. For example, `Axon.Layers.batch_norm` returns StatefulOutput
  with `:state` keys of `"mean"` and `"var"`.
  """

  @derive {
    Nx.Container,
    keep: [], containers: [:output, :state]
  }
  defstruct [:output, :state]
end
