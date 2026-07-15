defmodule Axon.DisplayTest do
  use ExUnit.Case, async: true

  describe "as_table/2" do
    test "renders layers with concrete parameter templates" do
      model = Axon.input("input") |> Axon.dense(32)
      input = Nx.template({1, 16}, :f32)

      table = Axon.Display.as_table(model, input)

      assert table =~ ~s|dense_0 ( dense )|
      assert table =~ ~s|kernel: f32[16][32]|
      assert table =~ ~s|bias: f32[32]|
      assert table =~ ~s|Total Parameters: 544|
      assert table =~ ~s|Total Parameters Memory: 2.18 kilobytes|
    end
  end
end
