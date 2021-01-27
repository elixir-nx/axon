defmodule Axon.Layers do
  @moduledoc """
  Layer behaviour.
  """

  @type tensor :: Nx.Tensor.t()

  # TODO: This callback needs to support a generic map or some other input of parameters
  @callback forward(input :: tensor, weight :: tensor, bias :: tensor) :: tensor

  defmacro __using__(_opts) do
    quote do
      @behaviour Axon.Layers

      import Axon.{Activations, Initializers}
      import Nx.Defn
    end
  end
end
