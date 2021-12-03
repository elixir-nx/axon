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
               Nx.tensor(0.546828031539917)
    end

    test "supports from_logits" do
      y_true = Nx.tensor([0, 1, 0, 1, 0])
      y_pred = Nx.tensor([15.0, -10.0, 6.0, 2.0, -1.0])

      assert Axon.Losses.binary_cross_entropy(y_true, y_pred, from_logits: true, reduction: :mean) ==
               Nx.tensor(6.2885422706604)
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

    test "supports sparse 1-d targets, from logits" do
      y_true = Nx.tensor([1, 0, 2])
      y_pred = Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])

      assert Axon.Losses.categorical_cross_entropy(y_true, y_pred,
               from_logits: true,
               sparse: true,
               reduction: :mean
             ) ==
               Nx.tensor(7.562242031097412)
    end

    test "supports sparse 2-d targets, from logits" do
      y_true = Nx.tensor([[1], [0], [2]])
      y_pred = Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])

      assert Axon.Losses.categorical_cross_entropy(y_true, y_pred,
               from_logits: true,
               sparse: true,
               reduction: :mean
             ) ==
               Nx.tensor(7.562242031097412)
    end

    test "supports sparse 1-d targets, from softmax" do
      y_true = Nx.tensor([1, 0, 2])

      y_pred =
        Axon.Activations.softmax(
          Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])
        )

      assert Axon.Losses.categorical_cross_entropy(y_true, y_pred,
               from_logits: false,
               sparse: true,
               reduction: :mean
             ) ==
               Nx.tensor(7.562242031097412)
    end

    test "supports sparse 2-d targets, from softmax" do
      y_true = Nx.tensor([[1], [0], [2]])

      y_pred =
        Axon.Activations.softmax(
          Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])
        )

      assert Axon.Losses.categorical_cross_entropy(y_true, y_pred,
               from_logits: false,
               sparse: true,
               reduction: :mean
             ) ==
               Nx.tensor(7.562242031097412)
    end
  end

  describe "ctcloss" do
    test "value for basic case" do
      y_true =
        Nx.tensor([
          [0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0],
          [0, 2, 0, 2, 0, 4, 0, 1, 0, 0, 0, 0, 0]
        ])

      y_pred = Nx.broadcast(-1.6094379425048828, {2, 10, 5})

      assert Axon.Losses.ctcloss(y_true, y_pred) ==
               Nx.tensor([8.08701229095459, 8.934309959411621])
    end

    test "trailing blanks doesn't contribute" do
      y_true1 =
        Nx.tensor([
          [0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0],
          [0, 2, 0, 2, 0, 4, 0, 1, 0, 0, 0, 0, 0]
        ])

      y_true2 = Nx.tensor([[0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0], [0, 2, 0, 2, 0, 4, 0, 1, 0, 0, 0]])
      y_pred = Nx.broadcast(-1.6094379425048828, {2, 10, 5})

      loss1 = Axon.Losses.ctcloss(y_true1, y_pred)
      loss2 = Axon.Losses.ctcloss(y_true2, y_pred)

      assert loss1 == loss2
    end

    test "value for complex case" do
      y_true = Nx.tensor([[0, 2, 0, 3, 0, 4, 0, 1, 0, 0, 0, 0, 0]])

      y_pred =
        Nx.tensor([
          [
            [-0.9714, -0.9640, -2.9779, -3.0708, -1.9462],
            [-3.0251, -2.2230, -0.4155, -2.4767, -2.3113],
            [-1.5946, -1.2221, -2.1956, -2.1702, -1.2839],
            [-0.2618, -2.5582, -2.9246, -3.6593, -2.6112],
            [-2.2140, -0.4346, -3.3434, -3.2254, -1.7830],
            [-2.9288, -2.9082, -2.0663, -0.7128, -1.2909],
            [-3.3950, -1.9521, -0.2629, -4.4295, -3.1303],
            [-0.9938, -1.1471, -2.5672, -2.3438, -1.9693],
            [-2.1427, -3.1549, -3.6807, -1.1077, -0.7246],
            [-2.2734, -2.6316, -3.7343, -0.4695, -1.7381]
          ]
        ])

      assert Axon.Losses.ctcloss(y_true, y_pred) == Nx.tensor([10.782031059265137])
    end
  end
end
