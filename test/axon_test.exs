defmodule AxonTest do
  use ExUnit.Case
  doctest Axon

  test "greets the world" do
    assert Axon.hello() == :world
  end
end
