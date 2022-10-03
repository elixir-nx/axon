defmodule Axon.Node do
  @moduledoc false

  defstruct [
    :id,
    :name,
    :parent,
    :parameters,
    :args,
    :op,
    :policy,
    :hooks,
    :opts,
    :op_name,
    :stacktrace
  ]
end
