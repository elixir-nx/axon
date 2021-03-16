defmodule Axon.Shape do
  @moduledoc """
  Layer shape calculations.
  """

  @doc """
  Flatten shape.
  """
  def flatten(shape) do
    out_units = div(Nx.size(shape), elem(shape, 0))
    {elem(shape, 0), out_units}
  end
end
