defmodule Axon.ActivationsTest do
  use ExUnit.Case, async: true
  doctest Axon.Activations

  import Nx.Defn

  describe "relu" do
    defn grad_relu(x), do: grad(x, &Nx.mean(Axon.Activations.relu(&1)))

    test "returns correct gradient with custom grad" do
      assert Nx.all_close?(
               grad_relu(Nx.iota({1, 3}, type: {:f, 32})),
               Nx.tensor([[0.0, 0.3333333432674408, 0.3333333432674408]])
             ) == Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "softmax" do
    test "raises on bad axis" do
      assert_raise ArgumentError, ~r/softmax axis must be within rank of tensor/, fn ->
        Axon.Activations.softmax(Nx.iota({1, 3}), axis: 2)
      end
    end
  end
end
