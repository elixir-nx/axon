defmodule Axon.Parameter do
  @moduledoc false
  defstruct [:id, :name, :shape, :initializer, :regularizer, frozen: false]
end
