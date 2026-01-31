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
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = random({3, 1, 4, 4})
      t_kernel = Nx.transpose(kernel, axes: [2, 3, 1, 0])
      bias = Nx.tensor(0.0)

      first = Axon.Layers.conv(input, kernel, bias, channels: :first)
      last = Axon.Layers.conv(t_input, t_kernel, bias, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels last same as channels first with strides" do
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = random({3, 1, 4, 4})
      t_kernel = Nx.transpose(kernel, axes: [2, 3, 1, 0])
      bias = Nx.tensor(0.0)

      first = Axon.Layers.conv(input, kernel, bias, channels: :first, strides: [1, 2])
      last = Axon.Layers.conv(t_input, t_kernel, bias, channels: :last, strides: [1, 2])

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
    test "channels first same as channels last" do
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = random({3, 1, 4, 4})
      t_kernel = Nx.transpose(kernel, axes: [2, 3, 1, 0])
      bias = Nx.tensor(0.0)

      first = Axon.Layers.conv_transpose(input, kernel, bias, channels: :first)
      last = Axon.Layers.conv_transpose(t_input, t_kernel, bias, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels first same as channels last with strides" do
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = random({3, 1, 4, 4})
      t_kernel = Nx.transpose(kernel, axes: [2, 3, 1, 0])
      bias = Nx.tensor(0.0)

      first = Axon.Layers.conv_transpose(input, kernel, bias, channels: :first, strides: [1, 2])
      last = Axon.Layers.conv_transpose(t_input, t_kernel, bias, channels: :last, strides: [1, 2])

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

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
        Axon.Layers.conv_transpose(
          inp,
          kernel,
          bias,
          padding: [{0, 1}, {1, 2}],
          channels: :first
        ),
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
      input = random({1, 3, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      kernel = random({6, 1, 4, 4})
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
      input = random({1, 3, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])
      k1 = random({6, 1, 4, 1})
      t_k1 = Nx.transpose(k1, axes: [2, 3, 1, 0])
      k2 = random({6, 1, 1, 4})
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
      input = random({1, 3, 8, 8, 8})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 4, 1])
      k1 = random({6, 1, 4, 1, 1})
      t_k1 = Nx.transpose(k1, axes: [2, 3, 4, 1, 0])
      k2 = random({6, 1, 1, 4, 1})
      t_k2 = Nx.transpose(k2, axes: [2, 3, 4, 1, 0])
      k3 = random({6, 1, 1, 1, 4})
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
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.max_pool(input, kernel_size: {2, 2}, channels: :first)
      last = Axon.Layers.max_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels last same as channels first with dilation" do
      input = random({1, 1, 28, 28})
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

    test "channels last same as channels first with custom padding" do
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first =
        Axon.Layers.max_pool(input,
          kernel_size: {2, 2},
          channels: :first,
          padding: [{2, 2}, {1, 2}]
        )

      last =
        Axon.Layers.max_pool(t_input,
          kernel_size: {2, 2},
          channels: :last,
          padding: [{2, 2}, {1, 2}]
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
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.avg_pool(input, kernel_size: {2, 2}, channels: :first)
      last = Axon.Layers.avg_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels last same as channels first with dilation" do
      input = random({1, 1, 28, 28})
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

    test "channels last same as channels first with custom padding" do
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first =
        Axon.Layers.max_pool(input,
          kernel_size: {2, 2},
          channels: :first,
          padding: [{2, 2}, {1, 2}]
        )

      last =
        Axon.Layers.max_pool(t_input,
          kernel_size: {2, 2},
          channels: :last,
          padding: [{2, 2}, {1, 2}]
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
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first = Axon.Layers.lp_pool(input, kernel_size: {2, 2}, channels: :first)
      last = Axon.Layers.lp_pool(t_input, kernel_size: {2, 2}, channels: :last)

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels last same as channels first with dilation" do
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first =
        Axon.Layers.lp_pool(input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :first
        )

      last =
        Axon.Layers.lp_pool(t_input,
          kernel_size: {2, 2},
          window_dilations: [2, 2],
          channels: :last
        )

      assert_equal(first, Nx.transpose(last, axes: [0, 3, 1, 2]))
    end

    test "channels last same as channels first with custom padding" do
      input = random({1, 1, 28, 28})
      t_input = Nx.transpose(input, axes: [0, 2, 3, 1])

      first =
        Axon.Layers.max_pool(input,
          kernel_size: {2, 2},
          channels: :first,
          padding: [{2, 2}, {1, 2}]
        )

      last =
        Axon.Layers.max_pool(t_input,
          kernel_size: {2, 2},
          channels: :last,
          padding: [{2, 2}, {1, 2}]
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
      input = random({1, 1, 28, 28})
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
      input = random({1, 1, 28, 28})
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
      input = random({1, 1, 28, 28})
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
                     Axon.Layers.spatial_dropout(inp, Nx.Random.key(0))
                   end
    end
  end

  describe "feature_alpha_dropout" do
    test "raises on input rank less than 3" do
      inp = Nx.iota({1, 1})

      assert_raise ArgumentError,
                   ~r/Axon.Layers.feature_alpha_dropout: expected input shape to have at least rank 3/,
                   fn ->
                     Axon.Layers.feature_alpha_dropout(inp, Nx.Random.key(0))
                   end
    end
  end

  describe "global_max_pool" do
    test "channels last same as channels first" do
      input = random({1, 1, 28, 28})
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
      input = random({1, 1, 28, 28})
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
      input = random({1, 1, 28, 28})
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

    # Adapted from NxImage
    test "methods" do
      # Reference values computed in jax

      image = Nx.iota({1, 2, 2, 3}, type: :f32)

      assert_equal(
        Axon.Layers.resize(image, size: {3, 3}, method: :nearest),
        Nx.tensor([
          [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0], [9.0, 10.0, 11.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0], [9.0, 10.0, 11.0]]
          ]
        ])
      )

      assert_equal(
        Axon.Layers.resize(image, size: {3, 3}, method: :bilinear),
        Nx.tensor([
          [
            [[0.0, 1.0, 2.0], [1.5, 2.5, 3.5], [3.0, 4.0, 5.0]],
            [[3.0, 4.0, 5.0], [4.5, 5.5, 6.5], [6.0, 7.0, 8.0]],
            [[6.0, 7.0, 8.0], [7.5, 8.5, 9.5], [9.0, 10.0, 11.0]]
          ]
        ])
      )

      assert_all_close(
        Axon.Layers.resize(image, size: {3, 3}, method: :bicubic),
        Nx.tensor([
          [
            [[-0.5921, 0.4079, 1.4079], [1.1053, 2.1053, 3.1053], [2.8026, 3.8026, 4.8026]],
            [[2.8026, 3.8026, 4.8026], [4.5, 5.5, 6.5], [6.1974, 7.1974, 8.1974]],
            [[6.1974, 7.1974, 8.1974], [7.8947, 8.8947, 9.8947], [9.5921, 10.5921, 11.5921]]
          ]
        ]),
        atol: 1.0e-3
      )

      assert_all_close(
        Axon.Layers.resize(image, size: {3, 3}, method: :lanczos3),
        Nx.tensor([
          [
            [[-1.1173, -0.1173, 0.8827], [0.7551, 1.7551, 2.7551], [2.6276, 3.6276, 4.6276]],
            [[2.6276, 3.6276, 4.6276], [4.5, 5.5, 6.5], [6.3724, 7.3724, 8.3724]],
            [[6.3724, 7.3724, 8.3724], [8.2449, 9.2449, 10.2449], [10.1173, 11.1173, 12.1173]]
          ]
        ]),
        atol: 1.0e-3
      )

      assert_all_close(
        Axon.Layers.resize(image, size: {3, 3}, method: :lanczos5),
        Nx.tensor([
          [
            [[-1.3525, -0.3525, 0.6475], [0.5984, 1.5984, 2.5984], [2.5492, 3.5492, 4.5492]],
            [[2.5492, 3.5492, 4.5492], [4.5, 5.5, 6.5], [6.4508, 7.4508, 8.4508]],
            [[6.4508, 7.4508, 8.4508], [8.4016, 9.4016, 10.4016], [10.3525, 11.3525, 12.3525]]
          ]
        ]),
        atol: 1.0e-3
      )
    end

    test "without anti-aliasing" do
      # Upscaling

      image = Nx.iota({1, 4, 4, 3}, type: :f32)

      assert_all_close(
        Axon.Layers.resize(image, size: {3, 3}, method: :bicubic, antialias: false),
        Nx.tensor([
          [
            [
              [[1.5427, 2.5427, 3.5427], [5.7341, 6.7341, 7.7341], [9.9256, 10.9256, 11.9256]],
              [[18.3085, 19.3085, 20.3085], [22.5, 23.5, 24.5], [26.6915, 27.6915, 28.6915]],
              [
                [35.0744, 36.0744, 37.0744],
                [39.2659, 40.2659, 41.2659],
                [43.4573, 44.4573, 45.4573]
              ]
            ]
          ]
        ]),
        atol: 1.0e-3
      )

      # Downscaling (no effect)

      image = Nx.iota({1, 2, 2, 3}, type: :f32)

      assert_all_close(
        Axon.Layers.resize(image, size: {3, 3}, method: :bicubic, antialias: false),
        Nx.tensor([
          [
            [[-0.5921, 0.4079, 1.4079], [1.1053, 2.1053, 3.1053], [2.8026, 3.8026, 4.8026]],
            [[2.8026, 3.8026, 4.8026], [4.5, 5.5, 6.5], [6.1974, 7.1974, 8.1974]],
            [[6.1974, 7.1974, 8.1974], [7.8947, 8.8947, 9.8947], [9.5921, 10.5921, 11.5921]]
          ]
        ]),
        atol: 1.0e-3
      )
    end
  end

  describe "lstm_cell" do
    test "cell function matches results expected from pytorch" do
      seq =
        File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_input_seq.npy")
        |> Nx.load_numpy!()

      c =
        File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_input_c.npy") |> Nx.load_numpy!()

      h =
        File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_input_h.npy") |> Nx.load_numpy!()

      wii = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_wii.npy") |> Nx.load_numpy!()
      wif = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_wif.npy") |> Nx.load_numpy!()
      wig = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_wig.npy") |> Nx.load_numpy!()
      wio = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_wio.npy") |> Nx.load_numpy!()
      whi = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_whi.npy") |> Nx.load_numpy!()
      whf = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_whf.npy") |> Nx.load_numpy!()
      whg = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_whg.npy") |> Nx.load_numpy!()
      who = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_who.npy") |> Nx.load_numpy!()
      bi = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_bi.npy") |> Nx.load_numpy!()
      bf = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_bf.npy") |> Nx.load_numpy!()
      bg = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_bg.npy") |> Nx.load_numpy!()
      bo = File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_bo.npy") |> Nx.load_numpy!()

      expected_c =
        File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_output_c.npy") |> Nx.load_numpy!()

      expected_h =
        File.read!("test/fixtures/lstm_cell_test/test_lstm_cell_output_h.npy") |> Nx.load_numpy!()

      {_, {new_c, new_h}} =
        Axon.Layers.lstm_cell(
          seq,
          {c, h},
          Nx.tensor(0),
          %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio},
          %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who},
          %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}
        )

      assert_all_close(new_c, expected_c)
      assert_all_close(new_h, expected_h)
    end
  end

  describe "lstm" do
    test "matches results expected from pytorch with dynamic unroll" do
      seq = File.read!("test/fixtures/lstm_test/test_lstm_input_seq.npy") |> Nx.load_numpy!()

      c =
        File.read!("test/fixtures/lstm_test/test_lstm_input_c.npy")
        |> Nx.load_numpy!()
        |> Nx.squeeze()

      h =
        File.read!("test/fixtures/lstm_test/test_lstm_input_h.npy")
        |> Nx.load_numpy!()
        |> Nx.squeeze()

      wii = File.read!("test/fixtures/lstm_test/test_lstm_wii.npy") |> Nx.load_numpy!()
      wif = File.read!("test/fixtures/lstm_test/test_lstm_wif.npy") |> Nx.load_numpy!()
      wig = File.read!("test/fixtures/lstm_test/test_lstm_wig.npy") |> Nx.load_numpy!()
      wio = File.read!("test/fixtures/lstm_test/test_lstm_wio.npy") |> Nx.load_numpy!()
      whi = File.read!("test/fixtures/lstm_test/test_lstm_whi.npy") |> Nx.load_numpy!()
      whf = File.read!("test/fixtures/lstm_test/test_lstm_whf.npy") |> Nx.load_numpy!()
      whg = File.read!("test/fixtures/lstm_test/test_lstm_whg.npy") |> Nx.load_numpy!()
      who = File.read!("test/fixtures/lstm_test/test_lstm_who.npy") |> Nx.load_numpy!()
      bi = File.read!("test/fixtures/lstm_test/test_lstm_bi.npy") |> Nx.load_numpy!()
      bf = File.read!("test/fixtures/lstm_test/test_lstm_bf.npy") |> Nx.load_numpy!()
      bg = File.read!("test/fixtures/lstm_test/test_lstm_bg.npy") |> Nx.load_numpy!()
      bo = File.read!("test/fixtures/lstm_test/test_lstm_bo.npy") |> Nx.load_numpy!()

      expected_seq =
        File.read!("test/fixtures/lstm_test/test_lstm_output_seq.npy") |> Nx.load_numpy!()

      expected_c =
        File.read!("test/fixtures/lstm_test/test_lstm_output_c.npy") |> Nx.load_numpy!()

      expected_h =
        File.read!("test/fixtures/lstm_test/test_lstm_output_h.npy") |> Nx.load_numpy!()

      {new_seq, {new_c, new_h}} =
        Axon.Layers.lstm(
          seq,
          {c, h},
          Nx.tensor(0),
          %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio},
          %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who},
          %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo},
          unroll: :dynamic
        )

      assert_all_close(new_seq, expected_seq, atol: 1.0e-3)
      assert_all_close(new_c, expected_c, atol: 1.0e-3)
      assert_all_close(new_h, expected_h, atol: 1.0e-3)
    end
  end

  describe "dynamic_unroll" do
    test "computes carry and output identical to static_unroll" do
      input = Nx.iota({2, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({2, 8}, type: {:f, 32})}

      input_kernel =
        %{
          "wir" => Nx.iota({2, 8}, type: {:f, 32}),
          "wiz" => Nx.iota({2, 8}, type: {:f, 32}),
          "win" => Nx.iota({2, 8}, type: {:f, 32})
        }

      hidden_kernel =
        %{
          "whr" => Nx.iota({8, 8}, type: {:f, 32}),
          "whz" => Nx.iota({8, 8}, type: {:f, 32}),
          "whn" => Nx.iota({8, 8}, type: {:f, 32})
        }

      bias =
        %{
          "br" => Nx.iota({}, type: {:f, 32}),
          "bz" => Nx.iota({}, type: {:f, 32}),
          "bin" => Nx.iota({}, type: {:f, 32}),
          "bhn" => Nx.iota({}, type: {:f, 32})
        }

      cell_fn = &Axon.Layers.gru_cell/6

      {s_output, {s_carry}} =
        Axon.Layers.static_unroll(
          cell_fn,
          input,
          carry,
          Nx.tensor(0),
          input_kernel,
          hidden_kernel,
          bias
        )

      {d_output, {d_carry}} =
        Axon.Layers.dynamic_unroll(
          cell_fn,
          input,
          carry,
          Nx.tensor(0),
          input_kernel,
          hidden_kernel,
          bias
        )

      assert_equal(s_carry, d_carry)
      assert_equal(s_output, d_output)
    end

    defn grad_static_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {output, _} =
          Axon.Layers.static_unroll(cell_fn, input, carry, Nx.tensor(0), input_kernel, x, bias)

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {output, _} =
          Axon.Layers.dynamic_unroll(cell_fn, input, carry, Nx.tensor(0), input_kernel, x, bias)

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for hidden kernel w.r.t. output" do
      input = Nx.iota({2, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({2, 8}, type: {:f, 32})}

      input_kernel =
        %{
          "wir" => Nx.iota({2, 8}, type: {:f, 32}),
          "wiz" => Nx.iota({2, 8}, type: {:f, 32}),
          "win" => Nx.iota({2, 8}, type: {:f, 32})
        }

      hidden_kernel =
        %{
          "whr" => Nx.iota({8, 8}, type: {:f, 32}),
          "whz" => Nx.iota({8, 8}, type: {:f, 32}),
          "whn" => Nx.iota({8, 8}, type: {:f, 32})
        }

      bias =
        %{
          "br" => Nx.iota({}, type: {:f, 32}),
          "bz" => Nx.iota({}, type: {:f, 32}),
          "bin" => Nx.iota({}, type: {:f, 32}),
          "bhn" => Nx.iota({}, type: {:f, 32})
        }

      cell_fn = &Axon.Layers.gru_cell/6

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
        {_, {carry}} =
          Axon.Layers.static_unroll(
            cell_fn,
            input,
            carry,
            Nx.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
            input_kernel,
            x,
            bias
          )

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {_, {carry}} =
          Axon.Layers.dynamic_unroll(
            cell_fn,
            input,
            carry,
            Nx.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
            input_kernel,
            x,
            bias
          )

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static_unroll for hidden kernel w.r.t carry" do
      input = Nx.iota({2, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({2, 8}, type: {:f, 32})}

      input_kernel =
        %{
          "wir" => Nx.iota({2, 8}, type: {:f, 32}),
          "wiz" => Nx.iota({2, 8}, type: {:f, 32}),
          "win" => Nx.iota({2, 8}, type: {:f, 32})
        }

      hidden_kernel =
        %{
          "whr" => Nx.iota({8, 8}, type: {:f, 32}),
          "whz" => Nx.iota({8, 8}, type: {:f, 32}),
          "whn" => Nx.iota({8, 8}, type: {:f, 32})
        }

      bias =
        %{
          "br" => Nx.iota({}, type: {:f, 32}),
          "bz" => Nx.iota({}, type: {:f, 32}),
          "bin" => Nx.iota({}, type: {:f, 32}),
          "bhn" => Nx.iota({}, type: {:f, 32})
        }

      cell_fn = &Axon.Layers.gru_cell/6

      assert_equal(
        grad_static_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end

    defn grad_static_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {output, _} =
          Axon.Layers.static_unroll(cell_fn, input, carry, Nx.tensor(0), x, hidden_kernel, bias)

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {output, _} =
          Axon.Layers.dynamic_unroll(cell_fn, input, carry, Nx.tensor(0), x, hidden_kernel, bias)

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for input kernel w.r.t. output" do
      input = Nx.iota({2, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({2, 8}, type: {:f, 32})}

      input_kernel =
        %{
          "wir" => Nx.iota({2, 8}, type: {:f, 32}),
          "wiz" => Nx.iota({2, 8}, type: {:f, 32}),
          "win" => Nx.iota({2, 8}, type: {:f, 32})
        }

      hidden_kernel =
        %{
          "whr" => Nx.iota({8, 8}, type: {:f, 32}),
          "whz" => Nx.iota({8, 8}, type: {:f, 32}),
          "whn" => Nx.iota({8, 8}, type: {:f, 32})
        }

      bias =
        %{
          "br" => Nx.iota({}, type: {:f, 32}),
          "bz" => Nx.iota({}, type: {:f, 32}),
          "bin" => Nx.iota({}, type: {:f, 32}),
          "bhn" => Nx.iota({}, type: {:f, 32})
        }

      cell_fn = &Axon.Layers.gru_cell/6

      assert_equal(
        grad_static_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end

    defn grad_static_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {_, {carry}} =
          Axon.Layers.static_unroll(
            cell_fn,
            input,
            carry,
            Nx.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
            x,
            hidden_kernel,
            bias
          )

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {_, {carry}} =
          Axon.Layers.dynamic_unroll(
            cell_fn,
            input,
            carry,
            Nx.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
            x,
            hidden_kernel,
            bias
          )

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static unroll for input kernel w.r.t. carry" do
      input = Nx.iota({2, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({2, 8}, type: {:f, 32})}

      input_kernel =
        %{
          "wir" => Nx.iota({2, 8}, type: {:f, 32}),
          "wiz" => Nx.iota({2, 8}, type: {:f, 32}),
          "win" => Nx.iota({2, 8}, type: {:f, 32})
        }

      hidden_kernel =
        %{
          "whr" => Nx.iota({8, 8}, type: {:f, 32}),
          "whz" => Nx.iota({8, 8}, type: {:f, 32}),
          "whn" => Nx.iota({8, 8}, type: {:f, 32})
        }

      bias =
        %{
          "br" => Nx.iota({}, type: {:f, 32}),
          "bz" => Nx.iota({}, type: {:f, 32}),
          "bin" => Nx.iota({}, type: {:f, 32}),
          "bhn" => Nx.iota({}, type: {:f, 32})
        }

      cell_fn = &Axon.Layers.gru_cell/6

      assert_equal(
        grad_static_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end

    defn grad_static_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {output, _} =
          Axon.Layers.static_unroll(
            cell_fn,
            input,
            carry,
            Nx.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
            input_kernel,
            hidden_kernel,
            x
          )

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {output, _} =
          Axon.Layers.dynamic_unroll(
            cell_fn,
            input,
            carry,
            Nx.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
            input_kernel,
            hidden_kernel,
            x
          )

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for bias w.r.t. output" do
      input = Nx.iota({2, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({2, 8}, type: {:f, 32})}

      input_kernel =
        %{
          "wir" => Nx.iota({2, 8}, type: {:f, 32}),
          "wiz" => Nx.iota({2, 8}, type: {:f, 32}),
          "win" => Nx.iota({2, 8}, type: {:f, 32})
        }

      hidden_kernel =
        %{
          "whr" => Nx.iota({8, 8}, type: {:f, 32}),
          "whz" => Nx.iota({8, 8}, type: {:f, 32}),
          "whn" => Nx.iota({8, 8}, type: {:f, 32})
        }

      bias =
        %{
          "br" => Nx.iota({}, type: {:f, 32}),
          "bz" => Nx.iota({}, type: {:f, 32}),
          "bin" => Nx.iota({}, type: {:f, 32}),
          "bhn" => Nx.iota({}, type: {:f, 32})
        }

      cell_fn = &Axon.Layers.gru_cell/6

      assert_equal(
        grad_static_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end

    defn grad_static_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {_, {carry}} =
          Axon.Layers.static_unroll(
            cell_fn,
            input,
            carry,
            Nx.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
            input_kernel,
            hidden_kernel,
            x
          )

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {_, {carry}} =
          Axon.Layers.dynamic_unroll(
            cell_fn,
            input,
            carry,
            Nx.tensor([[0, 0, 0, 1], [0, 0, 1, 1]]),
            input_kernel,
            hidden_kernel,
            x
          )

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static unroll for bias w.r.t. carry" do
      input = Nx.iota({2, 4, 2}, type: {:f, 32})
      carry = {Nx.iota({2, 8}, type: {:f, 32})}

      input_kernel =
        %{
          "wir" => Nx.iota({2, 8}, type: {:f, 32}),
          "wiz" => Nx.iota({2, 8}, type: {:f, 32}),
          "win" => Nx.iota({2, 8}, type: {:f, 32})
        }

      hidden_kernel =
        %{
          "whr" => Nx.iota({8, 8}, type: {:f, 32}),
          "whz" => Nx.iota({8, 8}, type: {:f, 32}),
          "whn" => Nx.iota({8, 8}, type: {:f, 32})
        }

      bias =
        %{
          "br" => Nx.iota({}, type: {:f, 32}),
          "bz" => Nx.iota({}, type: {:f, 32}),
          "bin" => Nx.iota({}, type: {:f, 32}),
          "bhn" => Nx.iota({}, type: {:f, 32})
        }

      cell_fn = &Axon.Layers.gru_cell/6

      assert_equal(
        grad_static_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn),
        grad_dynamic_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
      )
    end
  end

  describe "group_norm" do
    test "matches pytorch" do
      a =
        Nx.tensor([
          [
            0.8423,
            1.9226,
            -1.1295,
            -1.3154,
            1.2963,
            -0.6821,
            -0.0519,
            0.6875,
            -0.0313,
            -0.3328,
            -0.2821,
            -2.3289,
            -1.7641,
            -1.3184,
            -0.0890,
            0.0625
          ],
          [
            -1.0853,
            0.8060,
            -0.1397,
            -0.2169,
            0.9605,
            0.3947,
            0.4760,
            0.8097,
            0.0380,
            -0.6314,
            0.5761,
            1.9309,
            0.5038,
            -0.1892,
            1.8476,
            0.0517
          ]
        ])

      b =
        Nx.tensor([
          -0.3101,
          -1.5896,
          -1.4963,
          0.1278,
          -1.4580,
          1.3832,
          0.5709,
          0.5531,
          -0.0588,
          1.0411,
          1.3503,
          -1.2166,
          0.7133,
          0.0694,
          0.3150,
          -0.1306
        ])

      c =
        Nx.tensor([
          1.6585,
          2.3515,
          -1.3456,
          0.2376,
          -0.1333,
          0.5068,
          0.2441,
          1.0382,
          0.6879,
          -0.5402,
          -1.8304,
          -0.8906,
          -0.5329,
          -0.3390,
          -0.1877,
          0.1405
        ])

      expected =
        Nx.tensor([
          [
            1.4768,
            -0.1375,
            0.4536,
            0.0623,
            -1.5881,
            -0.5951,
            0.1157,
            1.2847,
            0.6378,
            -0.0194,
            -1.0751,
            1.3407,
            -1.3700,
            -0.3844,
            0.0597,
            0.0149
          ],
          [
            2.2986,
            0.9877,
            -0.4434,
            0.1453,
            -1.7321,
            0.8146,
            0.4430,
            1.5159,
            0.7202,
            -1.9153,
            -1.7368,
            -2.8723,
            -0.5429,
            -0.3954,
            0.2952,
            0.2103
          ]
        ])

      actual = Axon.Layers.group_norm(a, b, c, num_groups: 2)

      assert_all_close(expected, actual, atol: 1.0e-3)
    end
  end

  describe "rms_norm" do
    test "matches pytorch 2D input" do
      input =
        Nx.tensor([
          [1.9269, 1.4873, 0.9007, -2.1055, 0.6784, -1.2345, -0.0431, -1.6047],
          [-0.7521, 1.6487, -0.3925, -1.4036, -0.7279, -0.5594, -0.7688, 0.7624]
        ])

      gamma =
        Nx.tensor([0.4617, 0.2674, 0.5349, 0.8094, 1.1103, -1.6898, -0.9890, 0.9580])

      expected =
        Nx.tensor([
          [0.6344, 0.2836, 0.3436, -1.2153, 0.5372, 1.4877, 0.0304, -1.0962],
          [-0.3605, 0.4576, -0.2179, -1.1793, -0.8390, 0.9814, 0.7893, 0.7582]
        ])

      actual = Axon.Layers.rms_norm(input, gamma, epsilon: 1.0e-6, channel_index: -1)
      assert_all_close(expected, actual, atol: 1.0e-3)
    end

    test "matches pytorch 3D input" do
      input =
        Nx.tensor([
          [
            [-1.3847, -0.8712, -0.2234, 1.7174, 0.3189, -0.4245],
            [0.3057, -0.7746, -1.5576, 0.9956, -0.8798, -0.6011],
            [-1.2742, 2.1228, -1.2347, -0.4879, -0.9138, -0.6581],
            [0.0780, 0.5258, -0.4880, 1.1914, -0.8140, -0.7360]
          ],
          [
            [-1.4032, 0.0360, -0.0635, 0.6756, -0.0978, 1.8446],
            [-1.1845, 1.3835, 1.4451, 0.8564, 2.2181, 0.5232],
            [0.3466, -0.1973, -1.0546, 1.2780, -0.1722, 0.5238],
            [0.0566, 0.4263, 0.5750, -0.6417, -2.2064, -0.7508]
          ]
        ])

      gamma =
        Nx.tensor([0.4679, -0.2049, -0.7409, 0.3618, 1.9199, -0.2254])

      expected =
        Nx.tensor([
          [
            [-0.6502, 0.1792, 0.1661, 0.6236, 0.6144, 0.0960],
            [0.1530, 0.1698, 1.2341, 0.3853, -1.8064, 0.1449],
            [-0.4825, -0.3521, 0.7403, -0.1429, -1.4199, 0.1201],
            [0.0504, -0.1488, 0.4994, 0.5955, -2.1588, 0.2291]
          ],
          [
            [-0.6653, -0.0075, 0.0477, 0.2477, -0.1903, -0.4213],
            [-0.4033, -0.2063, -0.7791, 0.2255, 3.0986, -0.0858],
            [0.2218, 0.0553, 1.0685, 0.6324, -0.4521, -0.1614],
            [0.0257, -0.0849, -0.4138, -0.2255, -4.1147, 0.1644]
          ]
        ])

      actual = Axon.Layers.rms_norm(input, gamma, epsilon: 1.0e-6, channel_index: -1)
      assert_all_close(expected, actual, atol: 1.0e-3)
    end

    test "matches pytorch with ones weight" do
      input =
        Nx.tensor([
          [0.6127, -1.1754, -0.7646, -0.6666],
          [0.7444, -0.6453, -1.3890, -0.2730]
        ])

      gamma =
        Nx.tensor([1.0000, 1.0000, 1.0000, 1.0000])

      expected =
        Nx.tensor([
          [0.7342, -1.4084, -0.9163, -0.7987],
          [0.8632, -0.7483, -1.6108, -0.3165]
        ])

      actual = Axon.Layers.rms_norm(input, gamma, epsilon: 1.0e-6, channel_index: -1)
      assert_all_close(expected, actual, atol: 1.0e-3)
    end
  end

  describe "batch_norm" do
    test "matches pytorch when variance < epsilon" do
      input_val = -0.002805
      mean = -0.008561
      variance = 0.000412
      weight = 1.0
      bias = -0.144881
      epsilon = 0.001

      expected = Nx.tensor([0.0083])

      actual =
        Axon.Layers.batch_norm(
          Nx.tensor([[[[input_val]]]]),
          Nx.tensor([weight]),
          Nx.tensor([bias]),
          Nx.tensor([mean]),
          Nx.tensor([variance]),
          mode: :inference,
          epsilon: epsilon
        )

      assert_all_close(expected, actual, atol: 1.0e-3)
    end
  end
end
