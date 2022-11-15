defmodule Axon.LayersTest do
  use Axon.Case, async: true
  doctest Axon.Layers

  import Nx.Defn

  describe "dense" do
    test "forward matches tensorflow with input rank-2" do
      inp = Nx.tensor([[0.5818032]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      expected = Nx.tensor([[0.2905983, -0.12873293, -0.1240975, 0.50677955]])
      actual = Axon.Layers.dense(inp, kernel, bias)

      assert_all_close(expected, actual)
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

      assert_all_close(actual, expected)
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

      assert_all_close(actual, expected)
    end

    test "backward matches tensorflow with input rank-2" do
      inp = Nx.tensor([[0.0335058]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      {expected_k_grad, expected_b_grad} =
        jit(fn kernel, bias, inp ->
          grad({kernel, bias}, fn {k, b} ->
            Nx.sum(Axon.Layers.dense(inp, k, b))
          end)
        end).(kernel, bias, inp)

      actual_k_grad = Nx.tensor([[0.0335058, 0.0335058, 0.0335058, 0.0335058]])
      actual_b_grad = Nx.tensor([1.0, 1.0, 1.0, 1.0])

      assert_all_close(actual_k_grad, expected_k_grad)
      assert_all_close(actual_b_grad, expected_b_grad)
    end

    test "backward matches tensorflow with input rank-3" do
      inp = Nx.tensor([[[0.29668367], [0.7820021]], [[0.75896287], [0.651641]]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      {expected_k_grad, expected_b_grad} =
        jit(fn kernel, bias, inp ->
          grad({kernel, bias}, fn {k, b} ->
            Nx.sum(Axon.Layers.dense(inp, k, b))
          end)
        end).(kernel, bias, inp)

      actual_k_grad = Nx.tensor([[2.4892898, 2.4892898, 2.4892898, 2.4892898]])
      actual_b_grad = Nx.tensor([4.0, 4.0, 4.0, 4.0])

      assert_all_close(actual_k_grad, expected_k_grad)
      assert_all_close(actual_b_grad, expected_b_grad)
    end

    test "backward matches tensorflow with input rank-4" do
      inp = Nx.tensor([[[[0.5623027], [0.41169107]]], [[[0.86306655], [0.14902902]]]])
      kernel = Nx.tensor([[0.4994787, -0.22126544, -0.21329808, 0.87104976]])
      bias = Nx.tensor([0.0, 0.0, 0.0, 0.0])

      {expected_k_grad, expected_b_grad} =
        jit(fn kernel, bias, inp ->
          grad({kernel, bias}, fn {k, b} ->
            Nx.sum(Axon.Layers.dense(inp, k, b))
          end)
        end).(kernel, bias, inp)

      actual_k_grad = Nx.tensor([[1.9860893, 1.9860893, 1.9860893, 1.9860893]])
      actual_b_grad = Nx.tensor([4.0, 4.0, 4.0, 4.0])

      assert_all_close(actual_k_grad, expected_k_grad)
      assert_all_close(actual_b_grad, expected_b_grad)
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

  describe "bilinear" do
    test "raises on input ranks less than 2" do
      inp1 = Nx.tensor([1.0, 2.0, 3.0])
      inp2 = Nx.tensor([[1.0, 2.0, 3.0]])
      kernel = Nx.tensor([[[1.0, 2.0, 3.0]]])
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.bilinear: expected input1 shape to have at least rank 2/,
                   fn ->
                     Axon.Layers.bilinear(inp1, inp2, kernel, bias)
                   end

      inp1 = Nx.tensor([[1.0]])
      inp2 = Nx.tensor([2.0])
      kernel = Nx.tensor([[[1.0, 2.0, 3.0]]])
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.bilinear: expected input2 shape to have at least rank 2/,
                   fn ->
                     Axon.Layers.bilinear(inp1, inp2, kernel, bias)
                   end
    end

    test "raises on not equal input ranks" do
      inp1 = Nx.tensor([[1.0]])
      inp2 = Nx.tensor([[[2.0]]])
      kernel = Nx.tensor([[[1.0]]])
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.bilinear: expected input1 and input2 ranks to be equal/,
                   fn ->
                     Axon.Layers.bilinear(inp1, inp2, kernel, bias)
                   end
    end

    test "raises on kernel rank not equal to 3" do
      inp1 = Nx.tensor([[1.0]])
      inp2 = Nx.tensor([[2.0]])
      kernel = Nx.tensor([[1.0, 2.0]])
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.bilinear: expected kernel to have rank equal to 3/,
                   fn ->
                     Axon.Layers.bilinear(inp1, inp2, kernel, bias)
                   end
    end
  end

  describe "conv" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = Nx.random_uniform({3, 1, 4, 4})
      t_kernel = Nx.transpose(kernel, axes: [2, 3, 1, 0])
      bias = Nx.tensor(0.0)

      first = Axon.Layers.conv(input, kernel, bias, channels: :first)
      last = Axon.Layers.conv(t_input, t_kernel, bias, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})
      kernel = Nx.iota({2, 1, 1})
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.conv: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.conv(inp, kernel, bias)
                   end
    end

    test "raises on not equal input, kernel ranks" do
      inp = Nx.iota({1, 1, 1})
      kernel = Nx.iota({2, 1, 1, 1})
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.conv: expected input and kernel ranks to be equal/,
                   fn ->
                     Axon.Layers.conv(inp, kernel, bias)
                   end
    end
  end

  describe "conv_transpose" do
    test "correct valid padding, no strides" do
      inp = Nx.iota({1, 1, 4}, type: {:f, 32})
      kernel = Nx.iota({3, 1, 2}, type: {:f, 32})
      bias = 0.0

      assert_equal(
        Axon.Layers.conv_transpose(inp, kernel, bias, padding: :valid, channels: :first),
        Nx.tensor([
          [
            [0.0, 1.0, 2.0, 3.0, 0.0],
            [0.0, 3.0, 8.0, 13.0, 6.0],
            [0.0, 5.0, 14.0, 23.0, 12.0]
          ]
        ])
      )
    end

    test "correct with valid padding, strides" do
      inp = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({3, 1, 2, 2}, type: {:f, 32})
      bias = 0.0

      assert_equal(
        Axon.Layers.conv_transpose(inp, kernel, bias,
          padding: :valid,
          strides: [2, 1],
          channels: :first
        ),
        Nx.tensor([
          [
            [[0.0, 3.0, 2.0], [0.0, 1.0, 0.0], [6.0, 13.0, 6.0], [2.0, 3.0, 0.0]],
            [[0.0, 7.0, 6.0], [0.0, 5.0, 4.0], [14.0, 33.0, 18.0], [10.0, 23.0, 12.0]],
            [[0.0, 11.0, 10.0], [0.0, 9.0, 8.0], [22.0, 53.0, 30.0], [18.0, 43.0, 24.0]]
          ]
        ])
      )
    end

    test "correct with 3 spatial dimensions" do
      inp = Nx.iota({1, 1, 2, 2, 1}, type: {:f, 32})
      kernel = Nx.iota({3, 1, 2, 2, 1}, type: {:f, 32})
      bias = 0.0

      assert_equal(
        Axon.Layers.conv_transpose(inp, kernel, bias,
          padding: :valid,
          strides: [1, 1, 2],
          channels: :first
        ),
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
      )
    end

    test "correct with same padding, no strides" do
      inp = Nx.iota({3, 1, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      bias = 0.0

      assert_equal(
        Axon.Layers.conv_transpose(inp, kernel, bias, padding: :same, channels: :first),
        Nx.tensor([
          [[[0.0, 3.0], [6.0, 14.0]]],
          [[[12.0, 23.0], [22.0, 38.0]]],
          [[[24.0, 43.0], [38.0, 62.0]]]
        ])
      )
    end

    test "correct with same padding, strides" do
      inp = Nx.iota({1, 3, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({4, 3, 1, 2}, type: {:f, 32})
      bias = 0.0

      assert_equal(
        Axon.Layers.conv_transpose(inp, kernel, bias,
          padding: :same,
          strides: [2, 1],
          channels: :first
        ),
        Nx.tensor([
          [
            [[52.0, 101.0], [0.0, 0.0], [70.0, 131.0], [0.0, 0.0]],
            [[124.0, 263.0], [0.0, 0.0], [178.0, 365.0], [0.0, 0.0]],
            [[196.0, 425.0], [0.0, 0.0], [286.0, 599.0], [0.0, 0.0]],
            [[268.0, 587.0], [0.0, 0.0], [394.0, 833.0], [0.0, 0.0]]
          ]
        ])
      )
    end

    test "correct with custom padding, no strides" do
      inp = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({1, 1, 2, 1}, type: {:f, 32})
      bias = 0.0

      assert_equal(
        Axon.Layers.conv_transpose(inp, kernel, bias, padding: [{0, 1}, {1, 2}], channels: :first),
        Nx.tensor([[[[0.0, 2.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]])
      )
    end

    test "correct with custom padding, strides" do
      inp = Nx.iota({1, 1, 2, 2}, type: {:f, 32})
      kernel = Nx.iota({1, 1, 2, 1}, type: {:f, 32})
      bias = 0.0

      assert_equal(
        Axon.Layers.conv_transpose(inp, kernel, bias,
          padding: [{0, 1}, {1, 2}],
          strides: [2, 1],
          channels: :first
        ),
        Nx.tensor([
          [
            [
              [0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 2.0, 3.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0]
            ]
          ]
        ])
      )
    end

    test "correct with kernel dilation" do
      inp = Nx.iota({1, 1, 2, 4}, type: {:f, 32})
      kernel = Nx.iota({1, 1, 2, 3}, type: {:f, 32})
      bias = 0.0

      assert_equal(
        Axon.Layers.conv_transpose(inp, kernel, bias,
          kernel_dilation: [2, 1],
          padding: [{0, 1}, {1, 2}],
          strides: [2, 1],
          channels: :first
        ),
        Nx.tensor([[[[43.0, 67.0, 82.0, 49.0, 21.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]])
      )
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})
      kernel = Nx.iota({2, 1, 1})
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.conv_transpose: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.conv_transpose(inp, kernel, bias)
                   end
    end

    test "raises on not equal input, kernel ranks" do
      inp = Nx.iota({1, 1, 1})
      kernel = Nx.iota({2, 1, 1, 1})
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.conv_transpose: expected input and kernel ranks to be equal/,
                   fn ->
                     Axon.Layers.conv_transpose(inp, kernel, bias)
                   end
    end
  end

  describe "depthwise conv" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 3, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = Nx.random_uniform({6, 1, 4, 4})
      t_kernel = Nx.transpose(kernel, axes: [2, 3, 1, 0])
      bias = Nx.tensor(0.0)

      first = Axon.Layers.depthwise_conv(input, kernel, bias, channels: :first)
      last = Axon.Layers.depthwise_conv(t_input, t_kernel, bias, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})
      kernel = Nx.iota({2, 1, 1})
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.depthwise_conv: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.depthwise_conv(inp, kernel, bias)
                   end
    end

    test "raises on not equal input, kernel ranks" do
      inp = Nx.iota({1, 1, 1})
      kernel = Nx.iota({2, 1, 1, 1})
      bias = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.depthwise_conv: expected input and kernel ranks to be equal/,
                   fn ->
                     Axon.Layers.depthwise_conv(inp, kernel, bias)
                   end
    end
  end

  describe "separable_conv2d" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 3, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      k1 = Nx.random_uniform({6, 1, 4, 1})
      t_k1 = Nx.transpose(k1, axes: [2, 3, 1, 0])
      k2 = Nx.random_uniform({6, 1, 1, 4})
      t_k2 = Nx.transpose(k2, axes: [2, 3, 1, 0])
      b1 = Nx.tensor(0.0)
      b2 = Nx.tensor(0.0)

      first = Axon.Layers.separable_conv2d(input, k1, b1, k2, b2, channels: :first)
      last = Axon.Layers.separable_conv2d(t_input, t_k1, b1, t_k2, b2, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank not equal to 4" do
      inp = Nx.iota({1, 1, 1})
      k1 = k2 = Nx.iota({1, 1, 1})
      b1 = b2 = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.separable_conv2d: expected input to have rank equal to 4/,
                   fn ->
                     Axon.Layers.separable_conv2d(inp, k1, b1, k2, b2)
                   end
    end

    test "raises on not equal input, kernel ranks" do
      inp = Nx.iota({1, 1, 1, 1})
      k1 = Nx.iota({1, 1, 1, 1})
      k2 = Nx.iota({1, 1, 1, 1, 1})
      b1 = b2 = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.separable_conv2d: expected all input ranks/,
                   fn ->
                     Axon.Layers.separable_conv2d(inp, k1, b1, k2, b2)
                   end
    end
  end

  describe "separable_conv3d" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 3, 8, 8, 8})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 4, 1])
      k1 = Nx.random_uniform({6, 1, 4, 1, 1})
      t_k1 = Nx.transpose(k1, axes: [2, 3, 4, 1, 0])
      k2 = Nx.random_uniform({6, 1, 1, 4, 1})
      t_k2 = Nx.transpose(k2, axes: [2, 3, 4, 1, 0])
      k3 = Nx.random_uniform({6, 1, 1, 1, 4})
      t_k3 = Nx.transpose(k3, axes: [2, 3, 4, 1, 0])
      b1 = b2 = b3 = Nx.tensor(0.0)

      first = Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3, channels: :first)
      last = Axon.Layers.separable_conv3d(t_input, t_k1, b1, t_k2, b2, t_k3, b3, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 4, 1, 2, 3]))
    end

    test "raises on input rank not equal to 5" do
      inp = Nx.iota({1, 1, 1, 1})
      k1 = k2 = k3 = Nx.iota({1, 1, 1, 1})
      b1 = b2 = b3 = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.separable_conv3d: expected input to have rank equal to 5/,
                   fn ->
                     Axon.Layers.separable_conv3d(inp, k1, b1, k2, b2, k3, b3)
                   end
    end

    test "raises on not equal input, kernel ranks" do
      inp = Nx.iota({1, 1, 1, 1, 1})
      k1 = Nx.iota({1, 1, 1, 1, 1})
      k2 = Nx.iota({1, 1, 1, 1, 1})
      k3 = Nx.iota({1, 1, 1, 1, 1, 1})
      b1 = b2 = b3 = Nx.tensor([0.0])

      assert_raise ArgumentError,
                   ~r/Axon.Layers.separable_conv3d: expected all input ranks/,
                   fn ->
                     Axon.Layers.separable_conv3d(inp, k1, b1, k2, b2, k3, b3)
                   end
    end
  end

  describe "max_pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.max_pool(input, kernel_size: {2, 2}, channels: :first)
      last = Axon.Layers.max_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels last same as channels first with dilation" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first =
        Axon.Layers.max_pool(input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :first
        )

      last =
        Axon.Layers.max_pool(t_input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :last
        )

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.max_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.max_pool(inp)
                   end
    end
  end

  describe "avg_pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.avg_pool(input, kernel_size: {2, 2}, channels: :first)
      last = Axon.Layers.avg_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels last same as channels first with dilation" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first =
        Axon.Layers.avg_pool(input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :first
        )

      last =
        Axon.Layers.avg_pool(t_input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :last
        )

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.avg_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.avg_pool(inp)
                   end
    end
  end

  describe "lp pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.lp_pool(input, kernel_size: {2, 2}, channels: :first)
      last = Axon.Layers.lp_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels last same as channels first with dilation" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first =
        Axon.Layers.lp_pool(input, kernel_size: {2, 2}, window_dilations: [2, 2], channels: :first)

      last =
        Axon.Layers.lp_pool(t_input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :last
        )

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.lp_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.lp_pool(inp)
                   end
    end
  end

  describe "adaptive avg pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.adaptive_avg_pool(input, output_size: {25, 25}, channels: :first)
      last = Axon.Layers.adaptive_avg_pool(t_input, output_size: {25, 25}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.adaptive_avg_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.adaptive_avg_pool(inp)
                   end
    end
  end

  describe "adaptive max pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.adaptive_max_pool(input, output_size: {25, 25}, channels: :first)
      last = Axon.Layers.adaptive_max_pool(t_input, output_size: {25, 25}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.adaptive_max_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.adaptive_max_pool(inp)
                   end
    end
  end

  describe "adaptive lp pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.adaptive_lp_pool(input, output_size: {25, 25}, channels: :first)
      last = Axon.Layers.adaptive_lp_pool(t_input, output_size: {25, 25}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.adaptive_lp_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.adaptive_lp_pool(inp)
                   end
    end
  end

  describe "spatial_dropout" do
    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.spatial_dropout: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.spatial_dropout(inp)
                   end
    end
  end

  describe "feature_alpha_dropout" do
    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.feature_alpha_dropout: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.feature_alpha_dropout(inp)
                   end
    end
  end

  describe "global_max_pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.global_max_pool(input, channels: :first)
      last = Axon.Layers.global_max_pool(t_input, channels: :last)

      assert_equal(first, last)
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.global_max_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.global_max_pool(inp)
                   end
    end
  end

  describe "global_avg_pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.global_avg_pool(input, channels: :first)
      last = Axon.Layers.global_avg_pool(t_input, channels: :last)

      assert_equal(first, last)
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.global_avg_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.global_avg_pool(inp)
                   end
    end
  end

  describe "global_lp_pool" do
    test "channels last same as channels first" do
      input = Nx.random_uniform({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.global_lp_pool(input, channels: :first)
      last = Axon.Layers.global_lp_pool(t_input, channels: :last)

      assert_all_close(first, last)
    end

    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.global_lp_pool: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.global_lp_pool(inp)
                   end
    end
  end

  describe "embedding" do
    test "raises on kernel rank not equal to 2" do
      inp = Nx.iota({1})
      kernel = Nx.iota({1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.embedding: expected kernel to have rank equal to 2/,
                   fn ->
                     Axon.Layers.embedding(inp, kernel)
                   end
    end
  end

  describe "resize" do
    test "raises on input rank not equal to 4" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.resize: expected input to have rank equal to 4, got 2 != 4/,
                   fn ->
                     Axon.Layers.resize(inp)
                   end
    end
  end

  describe "dynamic_unroll" do
    test "computes carry and output identical to static_unroll" do
      input = Nx.iota({1, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({1, 1, 8}, type: {:f, 32})}

      input_kernel =
        {Nx.iota({2, 8}, type: {:f, 32}), Nx.iota({2, 8}, type: {:f, 32}),
         Nx.iota({2, 8}, type: {:f, 32})}

      hidden_kernel =
        {Nx.iota({8, 8}, type: {:f, 32}), Nx.iota({8, 8}, type: {:f, 32}),
         Nx.iota({8, 8}, type: {:f, 32})}

      bias =
        {Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}),
         Nx.iota({}, type: {:f, 32})}

      cell_fn = &Axon.Layers.gru_cell/5

      {s_output, {s_carry}} =
        Axon.Layers.static_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, bias)

      {d_output, {d_carry}} =
        Axon.Layers.dynamic_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, bias)

      assert_equal(s_carry, d_carry)
      assert_equal(s_output, d_output)
    end

    defn grad_static_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {output, _} = Axon.Layers.static_unroll(cell_fn, input, carry, input_kernel, x, bias)

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {output, _} = Axon.Layers.dynamic_unroll(cell_fn, input, carry, input_kernel, x, bias)

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for hidden kernel w.r.t. output" do
      input = Nx.iota({1, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({1, 1, 8}, type: {:f, 32})}

      input_kernel =
        {Nx.iota({2, 8}, type: {:f, 32}), Nx.iota({2, 8}, type: {:f, 32}),
         Nx.iota({2, 8}, type: {:f, 32})}

      hidden_kernel =
        {Nx.iota({8, 8}, type: {:f, 32}), Nx.iota({8, 8}, type: {:f, 32}),
         Nx.iota({8, 8}, type: {:f, 32})}

      bias =
        {Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}),
         Nx.iota({}, type: {:f, 32})}

      cell_fn = &Axon.Layers.gru_cell/5

      assert_equal(
        grad_static_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_hidden_output(
          input,
          carry,
          input_kernel,
          hidden_kernel,
          bias,
          cell_fn
        )
      )
    end

    defn grad_static_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {_, {carry}} = Axon.Layers.static_unroll(cell_fn, input, carry, input_kernel, x, bias)

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {_, {carry}} = Axon.Layers.dynamic_unroll(cell_fn, input, carry, input_kernel, x, bias)

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static_unroll for hidden kernel w.r.t carry" do
      input = Nx.iota({1, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({1, 1, 8}, type: {:f, 32})}

      input_kernel =
        {Nx.iota({2, 8}, type: {:f, 32}), Nx.iota({2, 8}, type: {:f, 32}),
         Nx.iota({2, 8}, type: {:f, 32})}

      hidden_kernel =
        {Nx.iota({8, 8}, type: {:f, 32}), Nx.iota({8, 8}, type: {:f, 32}),
         Nx.iota({8, 8}, type: {:f, 32})}

      bias =
        {Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}),
         Nx.iota({}, type: {:f, 32})}

      cell_fn = &Axon.Layers.gru_cell/5

      assert_equal(
        grad_static_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end

    defn grad_static_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {output, _} = Axon.Layers.static_unroll(cell_fn, input, carry, x, hidden_kernel, bias)

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {output, _} = Axon.Layers.dynamic_unroll(cell_fn, input, carry, x, hidden_kernel, bias)

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for input kernel w.r.t. output" do
      input = Nx.iota({1, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({1, 1, 8}, type: {:f, 32})}

      input_kernel =
        {Nx.iota({2, 8}, type: {:f, 32}), Nx.iota({2, 8}, type: {:f, 32}),
         Nx.iota({2, 8}, type: {:f, 32})}

      hidden_kernel =
        {Nx.iota({8, 8}, type: {:f, 32}), Nx.iota({8, 8}, type: {:f, 32}),
         Nx.iota({8, 8}, type: {:f, 32})}

      bias =
        {Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}),
         Nx.iota({}, type: {:f, 32})}

      cell_fn = &Axon.Layers.gru_cell/5

      assert_equal(
        grad_static_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end

    defn grad_static_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {_, {carry}} = Axon.Layers.static_unroll(cell_fn, input, carry, x, hidden_kernel, bias)

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {_, {carry}} = Axon.Layers.dynamic_unroll(cell_fn, input, carry, x, hidden_kernel, bias)

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static unroll for input kernel w.r.t. carry" do
      input = Nx.iota({1, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({1, 1, 8}, type: {:f, 32})}

      input_kernel =
        {Nx.iota({2, 8}, type: {:f, 32}), Nx.iota({2, 8}, type: {:f, 32}),
         Nx.iota({2, 8}, type: {:f, 32})}

      hidden_kernel =
        {Nx.iota({8, 8}, type: {:f, 32}), Nx.iota({8, 8}, type: {:f, 32}),
         Nx.iota({8, 8}, type: {:f, 32})}

      bias =
        {Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}),
         Nx.iota({}, type: {:f, 32})}

      cell_fn = &Axon.Layers.gru_cell/5

      assert_equal(
        grad_static_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end

    defn grad_static_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {output, _} =
          Axon.Layers.static_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, x)

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {output, _} =
          Axon.Layers.dynamic_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, x)

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for bias w.r.t. output" do
      input = Nx.iota({1, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({1, 1, 8}, type: {:f, 32})}

      input_kernel =
        {Nx.iota({2, 8}, type: {:f, 32}), Nx.iota({2, 8}, type: {:f, 32}),
         Nx.iota({2, 8}, type: {:f, 32})}

      hidden_kernel =
        {Nx.iota({8, 8}, type: {:f, 32}), Nx.iota({8, 8}, type: {:f, 32}),
         Nx.iota({8, 8}, type: {:f, 32})}

      bias =
        {Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}),
         Nx.iota({}, type: {:f, 32})}

      cell_fn = &Axon.Layers.gru_cell/5

      assert_equal(
        grad_static_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end

    defn grad_static_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {_, {carry}} =
          Axon.Layers.static_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, x)

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {_, {carry}} =
          Axon.Layers.dynamic_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, x)

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static unroll for bias w.r.t. carry" do
      input = Nx.iota({1, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({1, 1, 8}, type: {:f, 32})}

      input_kernel =
        {Nx.iota({2, 8}, type: {:f, 32}), Nx.iota({2, 8}, type: {:f, 32}),
         Nx.iota({2, 8}, type: {:f, 32})}

      hidden_kernel =
        {Nx.iota({8, 8}, type: {:f, 32}), Nx.iota({8, 8}, type: {:f, 32}),
         Nx.iota({8, 8}, type: {:f, 32})}

      bias =
        {Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}), Nx.iota({}, type: {:f, 32}),
         Nx.iota({}, type: {:f, 32})}

      cell_fn = &Axon.Layers.gru_cell/5

      assert_equal(
        grad_static_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end
  end
end
