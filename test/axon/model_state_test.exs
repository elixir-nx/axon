defmodule Axon.ModelStateTest do
  use ExUnit.Case, async: true

  import Nx.Testing, only: [assert_equal: 2]

  alias Axon.ModelState

  defmodule InDefn do
    import Nx.Defn

    defn empty(), do: Axon.ModelState.empty()

    defn new(data), do: Axon.ModelState.new(data)

    defn update(model_state, parameters), do: Axon.ModelState.update(model_state, parameters)

    defn update(model_state, parameters, state),
      do: Axon.ModelState.update(model_state, parameters, state)

    defn merge(lhs, rhs, fun), do: Axon.ModelState.merge(lhs, rhs, fun)

    defn freeze(model_state), do: Axon.ModelState.freeze(model_state)

    defn freeze(model_state, mask), do: Axon.ModelState.freeze(model_state, mask)

    defn unfreeze(model_state), do: Axon.ModelState.unfreeze(model_state)

    defn unfreeze(model_state, mask), do: Axon.ModelState.unfreeze(model_state, mask)

    defn trainable_parameters(model_state), do: Axon.ModelState.trainable_parameters(model_state)

    defn frozen_parameters(model_state), do: Axon.ModelState.frozen_parameters(model_state)

    defn trainable_state(model_state), do: Axon.ModelState.trainable_state(model_state)

    defn frozen_state(model_state), do: Axon.ModelState.frozen_state(model_state)

    defn tie(model_state, destination, source),
      do: Axon.ModelState.tie(model_state, destination, source)
  end

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

  describe "inside defn" do
    setup do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(4, name: "dense_0")
        |> Axon.batch_norm(name: "batch_norm_0")

      {init_fn, _} = Axon.build(model, mode: :train)
      model_state = init_fn.(Nx.template({1, 2}, :f32), ModelState.empty())

      %{model_state: model_state}
    end

    test "empty/0 and new/1" do
      assert %ModelState{data: %{}, parameters: %{}} = InDefn.empty()

      data = %{"dense_0" => %{"kernel" => Nx.iota({2, 2}, type: :f32)}}
      assert %ModelState{data: %{"dense_0" => %{"kernel" => kernel}}} = InDefn.new(data)
      assert_equal(kernel, Nx.iota({2, 2}, type: :f32))
    end

    test "trainable_parameters/1 and frozen_parameters/1", %{model_state: model_state} do
      assert_equal(
        InDefn.trainable_parameters(model_state),
        ModelState.trainable_parameters(model_state)
      )

      frozen = ModelState.freeze(model_state, &match?(["dense_0", _], &1))

      assert_equal(InDefn.trainable_parameters(frozen), ModelState.trainable_parameters(frozen))
      assert_equal(InDefn.frozen_parameters(frozen), ModelState.frozen_parameters(frozen))
    end

    test "trainable_state/1 and frozen_state/1", %{model_state: model_state} do
      assert_equal(InDefn.trainable_state(model_state), ModelState.trainable_state(model_state))
      assert_equal(InDefn.frozen_state(model_state), ModelState.frozen_state(model_state))
    end

    test "freeze/1,2 and unfreeze/1,2", %{model_state: model_state} do
      assert InDefn.freeze(model_state).frozen_parameters ==
               ModelState.freeze(model_state).frozen_parameters

      mask = &match?(["dense_0", _], &1)

      frozen = InDefn.freeze(model_state, mask)
      assert frozen.frozen_parameters == ModelState.freeze(model_state, mask).frozen_parameters

      assert InDefn.unfreeze(frozen).frozen_parameters ==
               ModelState.unfreeze(frozen).frozen_parameters

      assert InDefn.unfreeze(frozen, mask).frozen_parameters ==
               ModelState.unfreeze(frozen, mask).frozen_parameters
    end

    test "update/2,3", %{model_state: model_state} do
      parameters =
        model_state
        |> ModelState.trainable_parameters()
        |> deep_map(&Nx.add(&1, 1))

      state = deep_map(ModelState.trainable_state(model_state), &Nx.add(&1, 1))

      assert_equal(
        InDefn.update(model_state, parameters),
        ModelState.update(model_state, parameters)
      )

      assert_equal(
        InDefn.update(model_state, parameters, state),
        ModelState.update(model_state, parameters, state)
      )
    end

    test "merge/3", %{model_state: model_state} do
      fun = fn _key, lhs, rhs -> Nx.add(lhs, rhs) end

      assert_equal(
        InDefn.merge(model_state, model_state, fun),
        ModelState.merge(model_state, model_state, fun)
      )
    end

    test "tie/3", %{model_state: model_state} do
      tied = InDefn.tie(model_state, ["batch_norm_0", "gamma"], ["dense_0", "bias"])

      assert %Axon.ModelState.SharedParameter{path: ["dense_0", "bias"]} =
               tied.data["batch_norm_0"]["gamma"]
    end
  end

  defp deep_map(%Nx.Tensor{} = tensor, fun), do: fun.(tensor)

  defp deep_map(map, fun) when is_map(map) do
    Map.new(map, fn {k, v} -> {k, deep_map(v, fun)} end)
  end
end
