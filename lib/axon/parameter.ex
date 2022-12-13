defmodule Axon.Parameter do
  @moduledoc false
  defstruct [:name, :shape, :initializer, type: {:f, 32}, frozen: false]
end
