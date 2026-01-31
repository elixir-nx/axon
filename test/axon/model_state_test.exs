defmodule Axon.ModelStateTest do
  use ExUnit.Case, async: true

  alias Axon.ModelState

  describe "tie/4" do
    test "creates shared parameter at destination path" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(4, name: "dense_0")
        |> Axon.dense(4, name: "dense_1")

      {init_fn, _} = Axon.build(model)
      model_state = init_fn.(Nx.template({1, 2}, :f32), ModelState.empty())

      tied = ModelState.tie(model_state, ["dense_1", "kernel"], ["dense_0", "kernel"])

      assert %Axon.ModelState.SharedParameter{path: ["dense_0", "kernel"], transform: nil} =
               tied.data["dense_1"]["kernel"]
    end

    test "stores transform function" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(4, name: "dense_0")
        |> Axon.dense(4, name: "dense_1")

      {init_fn, _} = Axon.build(model)
      model_state = init_fn.(Nx.template({1, 2}, :f32), ModelState.empty())

      tied =
        ModelState.tie(
          model_state,
          ["dense_1", "kernel"],
          ["dense_0", "kernel"],
          transform: &Nx.transpose/1
        )

      assert %Axon.ModelState.SharedParameter{transform: transform} =
               tied.data["dense_1"]["kernel"]

      assert is_function(transform, 1)
    end
  end

  describe "trainable_parameters/1 with tied weights" do
    test "excludes tied parameters" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(4, name: "dense_0")
        |> Axon.dense(4, name: "dense_1")

      {init_fn, _} = Axon.build(model)
      model_state = init_fn.(Nx.template({1, 2}, :f32), ModelState.empty())

      # Before tying, both layers have kernel in trainable params
      trainable_before = ModelState.trainable_parameters(model_state)
      assert Map.has_key?(trainable_before["dense_0"], "kernel")
      assert Map.has_key?(trainable_before["dense_1"], "kernel")

      # After tying, dense_1 kernel should be excluded
      tied = ModelState.tie(model_state, ["dense_1", "kernel"], ["dense_0", "kernel"])
      trainable_after = ModelState.trainable_parameters(tied)

      assert Map.has_key?(trainable_after["dense_0"], "kernel")
      refute Map.has_key?(trainable_after["dense_1"], "kernel")
      assert Map.has_key?(trainable_after["dense_1"], "bias")
    end

    test "excludes layer when all parameters are tied" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(4, name: "dense_0", use_bias: false)
        |> Axon.dense(4, name: "dense_1", use_bias: false)

      {init_fn, _} = Axon.build(model)
      model_state = init_fn.(Nx.template({1, 2}, :f32), ModelState.empty())

      tied = ModelState.tie(model_state, ["dense_1", "kernel"], ["dense_0", "kernel"])
      trainable = ModelState.trainable_parameters(tied)

      assert Map.has_key?(trainable, "dense_0")
      refute Map.has_key?(trainable, "dense_1")
    end
  end
end
