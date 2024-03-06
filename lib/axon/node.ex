defmodule Axon.Node do
  @moduledoc false

  defstruct [
    :id,
    :name,
    :mode,
    :parent,
    :parameters,
    :args,
    :op,
    :policy,
    :hooks,
    :opts,
    :global_options,
    :op_name,
    :stacktrace
  ]
end
