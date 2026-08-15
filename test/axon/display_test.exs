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

    test "renders container outputs" do
      model =
        Axon.container(%{
          logits: Axon.input("input") |> Axon.dense(2),
          hidden: Axon.input("input") |> Axon.dense(4)
        })

      input = Nx.template({1, 16}, :f32)

      table = Axon.Display.as_table(model, input)

      assert table =~ ~s|container_0 ( container )|
      assert table =~ ~s|%{hidden: f32[1][4], logits: f32[1][2]}|
    end

    test "renders optional outputs which resolve to none" do
      model =
        Axon.container(%{
          out: Axon.input("input") |> Axon.dense(2),
          maybe: Axon.input("optional", optional: true) |> Axon.optional()
        })

      input = %{"input" => Nx.template({1, 16}, :f32)}

      table = Axon.Display.as_table(model, input)

      assert table =~ ~s|%{maybe: none, out: f32[1][2]}|
    end
  end

  describe "as_graph/3" do
    test "renders container outputs" do
      model =
        Axon.container({
          Axon.input("input") |> Axon.dense(2),
          Axon.input("optional", optional: true) |> Axon.optional()
        })

      input = %{"input" => Nx.template({1, 16}, :f32)}

      assert %Kino.JS{} = Axon.Display.as_graph(model, input)
    end
  end
end
