defmodule Axon.LossesTest do
  use Axon.Case, async: true
  doctest Axon.Losses

  describe "binary_cross_entropy" do
    test "supports class weights" do
      y_true = Nx.tensor([0, 1, 0, 1, 0])
      y_pred = Nx.tensor([0.2, 0.8, 0.5, 0.3, 0.8])
      pos = 0.5
      neg = 0.8

      assert_equal(
        Axon.Losses.binary_cross_entropy(y_true, y_pred,
          positive_weight: pos,
          negative_weight: neg,
          reduction: :mean
        ),
        Nx.tensor(0.546828031539917)
      )
    end

    test "supports from_logits" do
      y_true = Nx.tensor([0, 1, 0, 1, 0])
      y_pred = Nx.tensor([15.0, -10.0, 6.0, 2.0, -1.0])

      assert_all_close(
        Axon.Losses.binary_cross_entropy(y_true, y_pred, from_logits: true, reduction: :mean),
        Nx.tensor(6.2885422706604)
      )
    end

    test "raises on y_true shape not equal to y_pred" do
      y_true = Nx.iota({1})
      y_pred = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Losses.binary_cross_entropy: expected input shapes y_true and y_pred/,
                   fn ->
                     Axon.Losses.binary_cross_entropy(y_true, y_pred)
                   end
    end
  end

  describe "categorical_cross_entropy" do
    test "supports class weights" do
      y_true = Nx.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
      y_pred = Nx.tensor([[0.1, 0.8, 0.1], [0.4, 0.2, 0.4], [0.15, 0.25, 0.6]])
      weights = [0.3, 0.2, 0.2]

      assert_equal(
        Axon.Losses.categorical_cross_entropy(y_true, y_pred,
          class_weights: weights,
          reduction: :none
        ),
        Nx.tensor([0.04462870582938194, 0.27488723397254944, 0.1021651178598404])
      )

      assert_all_close(
        Axon.Losses.categorical_cross_entropy(y_true, y_pred,
          class_weights: weights,
          reduction: :mean
        ),
        Nx.tensor(0.6024014353752136)
      )

      assert_equal(
        Axon.Losses.categorical_cross_entropy(y_true, y_pred,
          class_weights: weights,
          reduction: :sum
        ),
        Nx.tensor(0.4216810464859009)
      )
    end

    test "supports from_logits" do
      y_true = Nx.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
      y_pred = Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])

      assert_equal(
        Axon.Losses.categorical_cross_entropy(y_true, y_pred,
          from_logits: true,
          reduction: :mean
        ),
        Nx.tensor(7.562242031097412)
      )
    end

    test "supports sparse 1-d targets, from logits" do
      y_true = Nx.tensor([1, 0, 2])
      y_pred = Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])

      assert_equal(
        Axon.Losses.categorical_cross_entropy(y_true, y_pred,
          from_logits: true,
          sparse: true,
          reduction: :mean
        ),
        Nx.tensor(7.562242031097412)
      )
    end

    test "supports sparse 2-d targets, from logits" do
      y_true = Nx.tensor([[1], [0], [2]])
      y_pred = Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])

      assert_equal(
        Axon.Losses.categorical_cross_entropy(y_true, y_pred,
          from_logits: true,
          sparse: true,
          reduction: :mean
        ),
        Nx.tensor(7.562242031097412)
      )
    end

    test "supports sparse 1-d targets, from softmax" do
      y_true = Nx.tensor([1, 0, 2])

      y_pred =
        Axon.Activations.softmax(
          Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])
        )

      assert_equal(
        Axon.Losses.categorical_cross_entropy(y_true, y_pred,
          from_logits: false,
          sparse: true,
          reduction: :mean
        ),
        Nx.tensor(7.562242031097412)
      )
    end

    test "supports sparse 2-d targets, from softmax" do
      y_true = Nx.tensor([[1], [0], [2]])

      y_pred =
        Axon.Activations.softmax(
          Nx.tensor([[15.0, 2.0, -22.0], [-5.0, -2.0, 3.0], [2.0, 1.96, 1.20]])
        )

      assert_equal(
        Axon.Losses.categorical_cross_entropy(y_true, y_pred,
          from_logits: false,
          sparse: true,
          reduction: :mean
        ),
        Nx.tensor(7.562242031097412)
      )
    end
  end

  describe "ctcloss" do
    test "value for basic case" do
      y_true =
        Nx.tensor([
          [2, 3, 4, 1],
          [2, 2, 4, 1]
        ])

      l_true = Nx.tensor([4, 4])
      y_pred = Nx.broadcast(-1.6094379425048828, {2, 10, 5})

      assert_equal(
        Axon.Losses.connectionist_temporal_classification({l_true, y_true}, y_pred),
        Nx.tensor([8.08642292022705, 8.933040618896484])
      )
    end

    test "oversize don't contribute" do
      y_true1 =
        Nx.tensor([
          [2, 3, 4, 1],
          [2, 2, 4, 1]
        ])

      y_true2 =
        Nx.tensor([
          [2, 3, 4, 1, 1],
          [2, 2, 4, 1, 1]
        ])

      l_true = Nx.tensor([4, 4])
      y_pred = Nx.broadcast(-1.6094379425048828, {2, 10, 5})

      loss1 = Axon.Losses.connectionist_temporal_classification({l_true, y_true1}, y_pred)
      loss2 = Axon.Losses.connectionist_temporal_classification({l_true, y_true2}, y_pred)

      assert_equal(loss1, loss2)
    end

    test "value for complex case" do
      y_true = Nx.tensor([[2, 3, 4, 1]])
      l_true = Nx.tensor([4])

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

      assert_equal(
        Axon.Losses.connectionist_temporal_classification({l_true, y_true}, y_pred),
        Nx.tensor([10.772387504577637])
      )
    end

    test ":mean reduction" do
      y_true =
        Nx.tensor([
          [2, 3, 4, 1],
          [2, 2, 4, 1]
        ])

      l_true = Nx.tensor([4, 3])

      y_pred =
        Nx.tensor([
          [
            [-1.0309, -2.4399, -1.4435, -4.6214, -1.1705],
            [-1.6151, -1.1580, -1.2058, -2.6540, -2.1439],
            [-1.7822, -1.0288, -1.3551, -1.8287, -2.8867],
            [-3.3948, -0.6075, -2.2001, -1.7393, -2.0002],
            [-0.4375, -3.7533, -3.0314, -3.7273, -1.3524],
            [-2.7503, -1.0192, -2.7091, -2.6811, -0.8207],
            [-3.9056, -2.3302, -1.3263, -1.6814, -0.8416],
            [-1.9398, -2.7478, -1.0155, -2.4359, -1.0716],
            [-1.2053, -3.1142, -1.0739, -2.7737, -1.3789],
            [-3.7676, -1.0243, -2.9354, -2.1129, -0.8123]
          ],
          [
            [-1.4757, -3.1694, -0.8400, -2.6604, -1.4796],
            [-4.2809, -1.5615, -0.8630, -1.9280, -1.5652],
            [-2.0013, -2.0925, -2.8260, -1.7255, -0.6849],
            [-1.2511, -1.4673, -1.4363, -1.7486, -2.6389],
            [-1.5776, -1.4469, -1.6252, -2.3894, -1.3106],
            [-1.6266, -1.0977, -1.6670, -2.0491, -1.8831],
            [-1.3675, -2.4313, -3.0910, -2.4530, -0.6427],
            [-1.7390, -1.4423, -0.9968, -2.2811, -2.1482],
            [-1.5861, -1.2293, -1.3651, -1.5892, -3.1388],
            [-0.8559, -2.7733, -1.0032, -2.5221, -2.7232]
          ]
        ])

      assert_equal(
        Axon.Losses.connectionist_temporal_classification({l_true, y_true}, y_pred,
          reduction: :mean
        ),
        Nx.tensor(2.472621440887451)
      )
    end
  end

  describe "cosine_similarity" do
    test "supports eps" do
      y_true = Nx.tensor([[0.0, 1.0], [1.0, 1.0]])
      y_pred = Nx.tensor([[1.0, 0.0], [1.0, 1.0]])
      eps = 1.0e-3

      assert_all_close(
        Axon.Losses.cosine_similarity(y_true, y_pred, eps: eps),
        Nx.tensor([0.0, 1.0])
      )
    end

    test "supports axes" do
      y_true = Nx.tensor([[0.0, 1.0], [1.0, 1.0]])
      y_pred = Nx.tensor([[1.0, 0.0], [1.0, 1.0]])
      axes = [0]

      assert_all_close(
        Axon.Losses.cosine_similarity(y_true, y_pred, axes: axes),
        Nx.tensor([0.7071067690849304, 0.7071067690849304])
      )
    end
  end
end
