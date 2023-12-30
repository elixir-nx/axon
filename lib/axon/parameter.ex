defmodule Axon.Parameter do
  @moduledoc false
  defstruct [
    :name,
    :shape,
    :initializer,
    :children,
    type: {:f, 32},
    frozen: false,
    kind: :parameter
  ]
end
