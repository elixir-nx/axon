defmodule Axon.LossesTest do
  use ExUnit.Case, async: true
  doctest Axon.Losses

  describe "binary_cross_entropy" do
    test "supports class weights" do
      y_true = Nx.tensor([0, 1, 0, 1, 0])
      y_pred = Nx.tensor([0.2, 0.8, 0.5, 0.3, 0.8])
      pos = 0.5
      neg = 0.8

      assert Axon.Losses.binary_cross_entropy(y_true, y_pred,
               positive_weight: pos,
               negative_weight: neg,
               reduction: :mean
             ) ==
               Nx.tensor(0.5468282103538513)
    end

    test "supports from_logits" do
      y_true = Nx.tensor([0, 1, 0, 1, 0])
      y_pred = Nx.tensor([15.0, -10.0, 6.0, 2.0, -1.0])

      assert Axon.Losses.binary_cross_entropy(y_true, y_pred, from_logits: true, reduction: :mean) ==
               Nx.tensor(6.293759822845459)
    end
  end

  describe "categorical_cross_entropy" do
    test "supports class weights" do
      y_true = Nx.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
      y_pred = Nx.tensor([[0.1, 0.8, 0.1], [0.4, 0.2, 0.4], [0.15, 0.25, 0.6]])
      weights = [0.3, 0.2, 0.2]

      assert Axon.Losses.categorical_cross_entropy(y_true, y_pred,
               class_weights: weights,
               reduction: :none
             ) == Nx.tensor([0.04462870582938194, 0.27488723397254944, 0.1021651178598404])

      assert Axon.Losses.categorical_cross_entropy(y_true, y_pred,
               class_weights: weights,
               reduction: :mean
             ) == Nx.tensor(0.6024014353752136)

      assert Axon.Losses.categorical_cross_entropy(y_true, y_pred,
               class_weights: weights,
               reduction: :sum
             ) == Nx.tensor(0.4216810464859009)
    end

    test "supports from_logits" do
      y_true = Nx.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
      y_pred = Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])

      assert Axon.Losses.categorical_cross_entropy(y_true, y_pred,
               from_logits: true,
               reduction: :mean
             ) ==
               Nx.tensor(7.562242031097412)
    end
  end
end
