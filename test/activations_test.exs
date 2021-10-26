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

  describe "sigmoid" do
    defn value_and_grad_sigmoid(x), do: value_and_grad(x, &Axon.Activations.sigmoid(&1))

    defn value_and_grad_sum_sigmoid(x),
      do: value_and_grad(x, &Nx.sum(Axon.Activations.sigmoid(&1)))

    test "value_and_grad" do
      assert {value, grad} = value_and_grad_sigmoid(Nx.tensor(5.0))
      assert Nx.all_close?(value, Nx.tensor(0.9933072))
      assert Nx.all_close?(grad, Nx.tensor(0.00664803))

      assert {value, grad} =
               value_and_grad_sum_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))

      assert Nx.all_close?(value, Nx.tensor(3.5))

      assert Nx.all_close?(
               grad,
               Nx.tensor([
                 0.04517666,
                 0.10499358,
                 0.19661194,
                 0.25,
                 0.19661193,
                 0.10499363,
                 0.04517666
               ])
             )
    end
  end

  describe "softmax" do
    test "raises on bad axis" do
      assert_raise ArgumentError, ~r/softmax axis must be within rank of tensor/, fn ->
        Axon.Activations.softmax(Nx.iota({1, 3}), axis: 2)
      end
    end
  end

  describe "softplus" do
    defn value_and_grad_softplus(x), do: value_and_grad(x, &Axon.Activations.softplus(&1))

    defn value_and_grad_sum_softplus(x),
      do: value_and_grad(x, &Nx.sum(Axon.Activations.softplus(&1)))

    test "value_and_grad" do
      assert {value, grad} = value_and_grad_softplus(Nx.tensor(5.0))
      assert Nx.all_close?(value, Nx.tensor(5.0067153))
      assert Nx.all_close?(grad, Nx.tensor(0.9933072))

      assert {value, grad} =
               value_and_grad_sum_softplus(
                 Nx.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
               )

      assert Nx.all_close?(value, Nx.tensor(11.707001))

      assert Nx.all_close?(
               grad,
               Nx.tensor([
                 0.01798621,
                 0.04742587,
                 0.11920291,
                 0.2689414,
                 0.5,
                 0.73105854,
                 0.880797,
                 0.95257413,
                 0.9820139
               ])
             )

      assert {value, grad} =
               value_and_grad_sum_softplus(
                 Nx.tensor([
                   [
                     3.91343785e-02,
                     2.02403838e-02,
                     1.12020537e-02,
                     2.83025027e-02,
                     2.39001730e-02,
                     3.32025503e-02,
                     1.57335941e-02,
                     3.85219835e-02,
                     4.17842921e-02,
                     3.35262782e-02,
                     7.44309500e-03,
                     4.03720339e-02,
                     4.42154224e-02,
                     3.90086390e-02,
                     4.04843100e-02,
                     4.23467114e-02,
                     2.15607638e-02,
                     3.81104307e-02,
                     4.93991938e-02,
                     4.31956985e-02,
                     3.86686089e-02,
                     2.52724580e-02,
                     5.28243431e-02,
                     4.63339678e-02,
                     5.84869638e-02,
                     4.19255988e-02,
                     3.79695809e-02,
                     4.37996404e-02,
                     6.22997236e-05,
                     5.94581819e-02,
                     4.50271399e-02,
                     4.27128846e-02
                   ]
                 ])
               )

      assert Nx.all_close?(value, Nx.tensor(22.758692))

      assert Nx.all_close?(
               grad,
               Nx.tensor([
                 [
                   0.5097823,
                   0.50505996,
                   0.5028005,
                   0.50707513,
                   0.50597477,
                   0.5082999,
                   0.5039333,
                   0.5096293,
                   0.5104446,
                   0.5083808,
                   0.5018608,
                   0.5100916,
                   0.5110521,
                   0.5097509,
                   0.5101197,
                   0.5105851,
                   0.50539,
                   0.50952643,
                   0.5123473,
                   0.51079726,
                   0.50966597,
                   0.5063178,
                   0.513203,
                   0.5115814,
                   0.51461756,
                   0.51047987,
                   0.50949126,
                   0.5109482,
                   0.50001556,
                   0.51486015,
                   0.5112549,
                   0.5106766
                 ]
               ])
             )
    end
  end
end
