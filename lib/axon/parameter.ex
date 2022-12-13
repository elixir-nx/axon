defmodule Axon.Parameter do
  @moduledoc false
  defstruct [:id, :name, :shape, :initializer, type: {:f, 32}, frozen: false]
end
