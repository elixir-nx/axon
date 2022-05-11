defmodule Axon.ShapeTest do
  use ExUnit.Case
  doctest Axon.Shape

  test "conv raises on bad input shape" do
    assert_raise ArgumentError, ~r/conv/, fn ->
      Axon.Shape.conv({nil, 1, 1}, {3, 1, 2}, [1], :valid, [1], [1], :first)
    end
  end
end
