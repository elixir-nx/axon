defmodule Axon.LayersTest do
  use ExUnit.Case, async: true
  doctest Axon.Layers

  import Nx.Defn

  describe "dense" do
    test "forward matches tensorflow with input rank-2" do
      inp = Nx.tensor([[0.5818032]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      expected = Nx.tensor([[0.2905983, -0.12873293, -0.1240975, 0.50677955]])
      actual = Axon.Layers.dense(inp, kernel, bias)

      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches tensorflow with input rank-3" do
      inp = Nx.tensor([[[0.7022208], [0.17015481]], [[0.13636208], [0.05458272]]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      expected =
        Nx.tensor([
          [
            [0.35074434, -0.1553772, -0.14978234, 0.61166924],
            [0.08498871, -0.03764938, -0.03629369, 0.14821331]
          ],
          [
            [0.06810995, -0.03017221, -0.02908577, 0.11877815],
            [0.0272629, -0.01207727, -0.01164239, 0.04754426]
          ]
        ])

      actual = Axon.Layers.dense(inp, kernel, bias)

      assert Nx.all_close(actual, expected) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches tensorflow with input rank-4" do
      inp = Nx.tensor([[[[0.8079568], [0.61292243]]], [[[0.07109654], [0.29420567]]]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      expected =
        Nx.tensor([
          [
            [
              [0.4035572, -0.17877291, -0.17233564, 0.7037706],
              [0.3061417, -0.13561855, -0.13073517, 0.53388596]
            ]
          ],
          [
            [
              [0.03551121, -0.01573121, -0.01516476, 0.06192862],
              [0.14694947, -0.06509755, -0.06275351, 0.2562678]
            ]
          ]
        ])

      actual = Axon.Layers.dense(inp, kernel, bias)

      assert Nx.all_close(actual, expected) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches tensorflow with input rank-2" do
      inp = Nx.tensor([[0.0335058]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      {expected_k_grad, expected_b_grad} =
        jit(
          fn kernel, bias ->
            grad({kernel, bias}, fn {k, b} ->
              Nx.sum(Axon.Layers.dense(inp, k, b))
            end)
          end,
          [kernel, bias]
        )

      actual_k_grad = Nx.tensor([[0.0335058, 0.0335058, 0.0335058, 0.0335058]])
      actual_b_grad = Nx.tensor([1.0, 1.0, 1.0, 1.0])

      assert Nx.all_close(actual_k_grad, expected_k_grad) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(actual_b_grad, expected_b_grad) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches tensorflow with input rank-3" do
      inp = Nx.tensor([[[0.29668367], [0.7820021]], [[0.75896287], [0.651641]]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      {expected_k_grad, expected_b_grad} =
        jit(
          fn kernel, bias ->
            grad({kernel, bias}, fn {k, b} ->
              Nx.sum(Axon.Layers.dense(inp, k, b))
            end)
          end,
          [kernel, bias]
        )

      actual_k_grad = Nx.tensor([[2.4892898, 2.4892898, 2.4892898, 2.4892898]])
      actual_b_grad = Nx.tensor([4.0, 4.0, 4.0, 4.0])

      assert Nx.all_close(actual_k_grad, expected_k_grad) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(actual_b_grad, expected_b_grad) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches tensorflow with input rank-4" do
      inp = Nx.tensor([[[[0.5623027], [0.41169107]]], [[[0.86306655], [0.14902902]]]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      {expected_k_grad, expected_b_grad} =
        jit(
          fn kernel, bias ->
            grad({kernel, bias}, fn {k, b} ->
              Nx.sum(Axon.Layers.dense(inp, k, b))
            end)
          end,
          [kernel, bias]
        )

      actual_k_grad = Nx.tensor([[1.9860893, 1.9860893, 1.9860893, 1.9860893]])
      actual_b_grad = Nx.tensor([4.0, 4.0, 4.0, 4.0])

      assert Nx.all_close(actual_k_grad, expected_k_grad) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(actual_b_grad, expected_b_grad) == Nx.tensor(1, type: {:u, 8})
    end

    test "raises with input rank less than 2" do
      inp = Nx.tensor([1.0, 2.0, 3.0])
      kernel = Nx.tensor([[1.0], [2.0], [3.0]])
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError, ~r/expected input shape to have at least rank 2/, fn ->
        Axon.Layers.dense(inp, kernel, bias)
      end
    end
  end

  describe "conv_transpose" do
    test "correct valid padding, no strides" do
      inp = Nx.iota({1, 1, 4}, type: {:f, 32})
      kernel = Nx.iota({3, 1, 2}, type: {:f, 32})
      bias = 0.0

      assert Axon.Layers.conv_transpose(inp, kernel, bias, padding: :valid) ==
               Nx.tensor([
                 [
                   [0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.0, 3.0, 8.0, 13.0, 6.0],
                   [0.0, 5.0, 14.0, 23.0, 12.0]
                 ]
               ])
    end

    test "correct with valid padding, strides" do
      inp = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({3, 1, 2, 2}, type: {:f, 32})
      bias = 0.0

      assert Axon.Layers.conv_transpose(inp, kernel, bias, padding: :valid, strides: [2, 1]) ==
               Nx.tensor([
                 [
                   [[0.0, 3.0, 2.0], [0.0, 1.0, 0.0], [6.0, 13.0, 6.0], [2.0, 3.0, 0.0]],
                   [[0.0, 7.0, 6.0], [0.0, 5.0, 4.0], [14.0, 33.0, 18.0], [10.0, 23.0, 12.0]],
                   [[0.0, 11.0, 10.0], [0.0, 9.0, 8.0], [22.0, 53.0, 30.0], [18.0, 43.0, 24.0]]
                 ]
               ])
    end

    test "correct with 3 spatial dimensions" do
      inp = Nx.iota({1, 1, 2, 2, 1}, type: {:f, 32})
      kernel = Nx.iota({3, 1, 2, 2, 1}, type: {:f, 32})
      bias = 0.0

      assert Axon.Layers.conv_transpose(inp, kernel, bias, padding: :valid, strides: [1, 1, 2]) ==
               Nx.tensor([
                 [
                   [
                     [[0.0, 0.0], [3.0, 0.0], [2.0, 0.0]],
                     [[6.0, 0.0], [14.0, 0.0], [6.0, 0.0]],
                     [[2.0, 0.0], [3.0, 0.0], [0.0, 0.0]]
                   ],
                   [
                     [[0.0, 0.0], [7.0, 0.0], [6.0, 0.0]],
                     [[14.0, 0.0], [38.0, 0.0], [22.0, 0.0]],
                     [[10.0, 0.0], [23.0, 0.0], [12.0, 0.0]]
                   ],
                   [
                     [[0.0, 0.0], [11.0, 0.0], [10.0, 0.0]],
                     [[22.0, 0.0], [62.0, 0.0], [38.0, 0.0]],
                     [[18.0, 0.0], [43.0, 0.0], [24.0, 0.0]]
                   ]
                 ]
               ])
    end

    test "correct with same padding, no strides" do
      inp = Nx.iota({3, 1, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      bias = 0.0

      assert Axon.Layers.conv_transpose(inp, kernel, bias, padding: :same) ==
               Nx.tensor([
                 [[[0.0, 3.0], [6.0, 14.0]]],
                 [[[12.0, 23.0], [22.0, 38.0]]],
                 [[[24.0, 43.0], [38.0, 62.0]]]
               ])
    end

    test "correct with same padding, strides" do
      inp = Nx.iota({1, 3, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({4, 3, 1, 2}, type: {:f, 32})
      bias = 0.0

      assert Axon.Layers.conv_transpose(inp, kernel, bias, padding: :same, strides: [2, 1]) ==
               Nx.tensor([
                 [
                   [[52.0, 101.0], [0.0, 0.0], [70.0, 131.0], [0.0, 0.0]],
                   [[124.0, 263.0], [0.0, 0.0], [178.0, 365.0], [0.0, 0.0]],
                   [[196.0, 425.0], [0.0, 0.0], [286.0, 599.0], [0.0, 0.0]],
                   [[268.0, 587.0], [0.0, 0.0], [394.0, 833.0], [0.0, 0.0]]
                 ]
               ])
    end

    test "correct with custom padding, no strides" do
      inp = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({1, 1, 2, 1}, type: {:f, 32})
      bias = 0.0

      assert Axon.Layers.conv_transpose(inp, kernel, bias, padding: [{0, 1}, {1, 2}]) ==
               Nx.tensor([[[[0.0, 2.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]])
    end

    test "correct with custom padding, strides" do
      inp = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({1, 1, 2, 1}, type: {:f, 32})
      bias = 0.0

      assert Axon.Layers.conv_transpose(inp, kernel, bias,
               padding: [{0, 1}, {1, 2}],
               strides: [2, 1]
             ) ==
               Nx.tensor([
                 [
                   [
                     [0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 2.0, 3.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0]
                   ]
                 ]
               ])
    end

    test "correct with kernel dilation" do
      inp = Nx.iota({1, 1, 2, 4}, type: {:f, 32})
      kernel = Nx.iota({1, 1, 2, 3}, type: {:f, 32})
      bias = 0.0

      assert Axon.Layers.conv_transpose(inp, kernel, bias,
               kernel_dilation: [2, 1],
               padding: [{0, 1}, {1, 2}],
               strides: [2, 1]
             ) ==
               Nx.tensor([[[[43.0, 67.0, 82.0, 49.0, 21.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]])
    end
  end

  describe "resize" do
    test "bilinear without aligned corners" do
      input = Nx.iota({1, 1, 3, 4}, type: {:f, 32})

      assert Axon.Layers.resize(input, shape: {5, 2}, method: :bilinear, align_corners: false) ==
               Nx.tensor([
                 [
                   [
                     [0.5, 2.5],
                     [2.1000001430511475, 4.100000381469727],
                     [4.5, 6.5],
                     [6.900000095367432, 8.899999618530273],
                     [8.5, 10.5]
                   ]
                 ]
               ])
    end

    test "bilinear with aligned corners" do
      input = Nx.iota({1, 1, 3, 4}, type: {:f, 32})

      assert Axon.Layers.resize(input, shape: {5, 2}, method: :bilinear, align_corners: true) ==
               Nx.tensor([
                 [
                   [
                     [0.0, 3.0],
                     [2.0, 5.0],
                     [4.0, 7.0],
                     [6.0, 9.0],
                     [8.0, 11.0]
                   ]
                 ]
               ])
    end
  end

  describe "conv" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = Nx.random_uniform({3, 1, 4, 4})
      bias = Nx.tensor(0.0)

      first = Axon.Layers.conv(input, kernel, bias)
      last = Axon.Layers.conv(t_input, kernel, bias, channels: :last)

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end
  end

  describe "depthwise conv" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 3, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = Nx.random_uniform({6, 1, 4, 4})
      bias = Nx.tensor(0.0)

      first = Axon.Layers.depthwise_conv(input, kernel, bias)
      last = Axon.Layers.depthwise_conv(t_input, kernel, bias, channels: :last)

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end
  end

  describe "max pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.max_pool(input, kernel_size: {2, 2})
      last = Axon.Layers.max_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end

    test "channels last same as channels first with dilation" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.max_pool(input, kernel_size: {2, 2}, window_dilations: [2, 2])

      last =
        Axon.Layers.max_pool(t_input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :last
        )

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end
  end

  describe "avg pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.avg_pool(input, kernel_size: {2, 2})
      last = Axon.Layers.avg_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end

    test "channels last same as channels first with dilation" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.avg_pool(input, kernel_size: {2, 2}, window_dilations: [2, 2])

      last =
        Axon.Layers.avg_pool(t_input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :last
        )

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end
  end

  describe "lp pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.lp_pool(input, kernel_size: {2, 2})
      last = Axon.Layers.lp_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end

    test "channels last same as channels first with dilation" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.lp_pool(input, kernel_size: {2, 2}, window_dilations: [2, 2])

      last =
        Axon.Layers.lp_pool(t_input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :last
        )

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end
  end

  describe "adaptive avg pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.adaptive_avg_pool(input, output_size: {25, 25})
      last = Axon.Layers.adaptive_avg_pool(t_input, output_size: {25, 25}, channels: :last)

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end
  end

  describe "adaptive max pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.adaptive_max_pool(input, output_size: {25, 25})
      last = Axon.Layers.adaptive_max_pool(t_input, output_size: {25, 25}, channels: :last)

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end
  end

  describe "adaptive lp pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.adaptive_lp_pool(input, output_size: {25, 25})
      last = Axon.Layers.adaptive_lp_pool(t_input, output_size: {25, 25}, channels: :last)

      assert first == Nx.transpose(last, axes: [0, 3, 1, 2])
    end
  end

  describe "global max pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.global_max_pool(input)
      last = Axon.Layers.global_max_pool(t_input, channels: :last)

      assert first == last
    end
  end

  describe "global avg pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.global_avg_pool(input)
      last = Axon.Layers.global_avg_pool(t_input, channels: :last)

      assert first == last
    end
  end

  describe "global lp pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.global_lp_pool(input)
      last = Axon.Layers.global_lp_pool(t_input, channels: :last)

      assert first == last
    end
  end
end
