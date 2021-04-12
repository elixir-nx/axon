defmodule AxonTest do
  use ExUnit.Case
  doctest Axon

  describe "conv" do
    test "works with defaults" do
      assert %Axon{} = Axon.input({nil, 1, 28, 28}) |> Axon.conv(64)
    end
  end
end
