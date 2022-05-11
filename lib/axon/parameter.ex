defmodule Axon.Parameter do
  @moduledoc false
  defstruct [:id, :name, :shape, :initializer, frozen: false]
end
