defmodule Axon.LayersTest do
  use ExUnit.Case, async: true
  doctest Axon.Layers

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
end
