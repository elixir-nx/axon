defmodule Axon.Layers.Dense do
  @moduledoc """
  Dense layer.
  """
  use Axon.Layers

  def initialize(in_features, out_features) do
    %{
      "weight" => zeros({in_features, out_features}),
      "bias" => zeros({out_features})
    }
  end

  defn forward(x, weight, bias) do
    x
    |> Nx.dot(weight)
    |> linear()
    |> Nx.add(bias)
  end
end
