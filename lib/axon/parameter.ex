defmodule Axon.Parameter do
  @moduledoc false
  defstruct [
    :name,
    :template,
    :shape,
    :initializer,
    :children,
    type: {:f, 32},
    frozen: false,
    kind: :parameter
  ]
end
