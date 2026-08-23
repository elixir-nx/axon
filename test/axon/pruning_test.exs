defmodule Axon.PruningTest do
  use Axon.Case, async: true

  doctest Axon.Pruning

  alias Axon.ModelState
  alias Axon.Pruning

  defmodule InDefn do
    import Nx.Defn

    defn global_mask(model_state), do: Axon.Pruning.magnitude_mask(model_state, 0.5)

    defn dense_mask(model_state), do: Axon.Pruning.magnitude_mask(model_state, 0.0)

    defn tensor_mask(model_state),
      do: Axon.Pruning.magnitude_mask(model_state, 0.5, scope: :tensor)

    defn apply_mask(model_state, mask), do: Axon.Pruning.apply_mask(model_state, mask)

    defn prune(model_state), do: Axon.Pruning.prune(model_state, 0.5)
  end

  @kernel Nx.tensor([[1.0, -8.0, 3.0, 0.5], [2.0, 7.0, -0.1, 4.0]])
  @kernel_mask Nx.tensor([[0, 1, 1, 0], [0, 1, 0, 1]], type: :u8)

  defp init(model, template, opts \\ []) do
    {init_fn, _} = Axon.build(model, opts)
    init_fn.(template, ModelState.empty())
  end

  defp dense_state do
    Axon.input("x", shape: {nil, 2})
    |> Axon.dense(4, name: "d")
    |> init(Nx.template({1, 2}, :f32))
    |> put_in([Access.key!(:data), "d", "kernel"], @kernel)
  end

  defp two_dense_state do
    Axon.input("x", shape: {nil, 4})
    |> Axon.dense(4, name: "d0")
    |> Axon.dense(4, name: "d1")
    |> init(Nx.template({1, 4}, :f32))
    |> put_in([Access.key!(:data), "d0", "kernel"], Nx.broadcast(0.1, {4, 4}))
    |> put_in([Access.key!(:data), "d1", "kernel"], Nx.broadcast(5.0, {4, 4}))
  end

  defp lstm_state do
    Axon.input("x", shape: {nil, 2, 3})
    |> Axon.lstm(4, name: "l")
    |> elem(0)
    |> init(Nx.template({1, 2, 3}, :f32))
  end

  describe "magnitude_mask/3" do
    test "prunes the smallest magnitudes of each tensor with scope: :tensor" do
      mask = Pruning.magnitude_mask(dense_state(), 0.5, scope: :tensor)

      assert %{"d" => %{"kernel" => kernel_mask}} = mask
      refute Map.has_key?(mask["d"], "bias")
      assert Nx.type(kernel_mask) == {:u, 8}
      assert Nx.shape(kernel_mask) == {2, 4}
      assert_equal(kernel_mask, @kernel_mask)
    end

    test "uses a single threshold across tensors with scope: :global" do
      mask = Pruning.magnitude_mask(two_dense_state(), 0.5, scope: :global)

      assert_equal(mask["d0"]["kernel"], Nx.broadcast(Nx.tensor(0, type: :u8), {4, 4}))
      assert_equal(mask["d1"]["kernel"], Nx.broadcast(Nx.tensor(1, type: :u8), {4, 4}))

      mask = Pruning.magnitude_mask(two_dense_state(), 0.5, scope: :tensor)

      assert_equal(Nx.sum(mask["d0"]["kernel"]), Nx.tensor(8))
      assert_equal(Nx.sum(mask["d1"]["kernel"]), Nx.tensor(8))
    end

    test "defaults to scope: :global" do
      assert_equal(
        Pruning.magnitude_mask(two_dense_state(), 0.5)["d0"]["kernel"],
        Nx.broadcast(Nx.tensor(0, type: :u8), {4, 4})
      )
    end

    test "prunes an exact number of entries when magnitudes tie" do
      state =
        put_in(dense_state(), [Access.key!(:data), "d", "kernel"], Nx.broadcast(1.0, {2, 4}))

      mask = Pruning.magnitude_mask(state, 0.25, scope: :tensor)
      assert_equal(Nx.sum(mask["d"]["kernel"]), Nx.tensor(8 - round(0.25 * 8)))

      mask = Pruning.magnitude_mask(state, 0.25, scope: :global)
      assert_equal(Nx.sum(mask["d"]["kernel"]), Nx.tensor(8 - round(0.25 * 8)))
    end

    test "keeps everything at sparsity 0 and nothing at sparsity 1" do
      for scope <- [:global, :tensor] do
        mask = Pruning.magnitude_mask(dense_state(), 0.0, scope: scope)
        assert Nx.type(mask["d"]["kernel"]) == {:u, 8}
        assert_equal(mask["d"]["kernel"], Nx.broadcast(Nx.tensor(1, type: :u8), {2, 4}))

        mask = Pruning.magnitude_mask(dense_state(), 1.0, scope: scope)
        assert Nx.type(mask["d"]["kernel"]) == {:u, 8}
        assert_equal(mask["d"]["kernel"], Nx.broadcast(Nx.tensor(0, type: :u8), {2, 4}))
      end
    end

    test "default filter selects kernels and skips norm parameters and state" do
      state =
        Axon.input("x", shape: {nil, 2})
        |> Axon.dense(4, name: "d")
        |> Axon.batch_norm(name: "bn")
        |> init(Nx.template({1, 2}, :f32), mode: :train)

      mask = Pruning.magnitude_mask(state, 0.5)

      assert Map.keys(mask) == ["d"]
      assert Map.keys(mask["d"]) == ["kernel"]

      pruned = Pruning.apply_mask(state, mask)
      assert_equal(pruned.data["bn"]["mean"], state.data["bn"]["mean"])
      assert_equal(pruned.data["bn"]["var"], state.data["bn"]["var"])
      assert_equal(pruned.data["bn"]["gamma"], state.data["bn"]["gamma"])
      assert_equal(pruned.data["d"]["bias"], state.data["d"]["bias"])
    end

    test "default filter selects composite recurrent kernels" do
      mask = Pruning.magnitude_mask(lstm_state(), 0.5)

      assert Map.keys(mask) == ["l"]
      assert Map.keys(mask["l"]) == ["hidden_kernel", "input_kernel"]
      assert Map.keys(mask["l"]["input_kernel"]) == ["wif", "wig", "wii", "wio"]
      assert Nx.type(mask["l"]["input_kernel"]["wii"]) == {:u, 8}
      assert Nx.shape(mask["l"]["input_kernel"]["wii"]) == {3, 4}
      assert Nx.shape(mask["l"]["hidden_kernel"]["whi"]) == {4, 4}
    end

    test "accepts a custom :filter" do
      mask = Pruning.magnitude_mask(dense_state(), 0.5, filter: &match?(["d", "bias"], &1))

      assert Map.keys(mask["d"]) == ["bias"]
      assert Nx.shape(mask["d"]["bias"]) == {4}
    end

    test "returns an empty map when nothing matches the filter" do
      assert Pruning.magnitude_mask(dense_state(), 0.5, filter: fn _ -> false end) == %{}
      assert Pruning.magnitude_mask(ModelState.empty(), 0.5) == %{}
    end

    test "skips tied parameters" do
      state = ModelState.tie(two_dense_state(), ["d1", "kernel"], ["d0", "kernel"])

      mask = Pruning.magnitude_mask(state, 0.5, scope: :tensor, filter: fn _ -> true end)

      assert Map.keys(mask["d0"]) == ["bias", "kernel"]
      assert Map.keys(mask["d1"]) == ["bias"]
    end

    test "skips quantized parameters" do
      model = Axon.input("x", shape: {nil, 2}) |> Axon.dense(4, name: "d")

      state =
        Axon.Quantization.quantize_model_state(model, init(model, Nx.template({1, 2}, :f32)))

      mask = Pruning.magnitude_mask(state, 0.5, filter: fn _ -> true end)

      assert Map.keys(mask["d"]) == ["bias"]
    end

    test "skips non-float parameters" do
      state =
        put_in(dense_state(), [Access.key!(:data), "d", "kernel"], Nx.iota({2, 4}, type: :s32))

      mask = Pruning.magnitude_mask(state, 0.5, filter: fn _ -> true end)

      assert Map.keys(mask["d"]) == ["bias"]
    end

    test "raises on invalid sparsity" do
      assert_raise ArgumentError,
                   "expected sparsity to be a number between 0 and 1, got: 1.5",
                   fn -> Pruning.magnitude_mask(dense_state(), 1.5) end

      assert_raise ArgumentError,
                   "expected sparsity to be a number between 0 and 1, got: -0.1",
                   fn -> Pruning.magnitude_mask(dense_state(), -0.1) end
    end

    test "raises on invalid options" do
      assert_raise ArgumentError,
                   "expected :scope to be :global or :tensor, got: :layer",
                   fn -> Pruning.magnitude_mask(dense_state(), 0.5, scope: :layer) end

      assert_raise ArgumentError, ~r/unknown keys \[:method\]/, fn ->
        Pruning.magnitude_mask(dense_state(), 0.5, method: :random)
      end
    end
  end

  describe "apply_mask/2" do
    test "zeroes masked entries and leaves everything else untouched" do
      state = dense_state()
      mask = %{"d" => %{"kernel" => @kernel_mask}}

      pruned = Pruning.apply_mask(state, mask)

      assert_equal(pruned.data["d"]["kernel"], Nx.select(@kernel_mask, @kernel, 0))
      assert Nx.type(pruned.data["d"]["kernel"]) == {:f, 32}
      assert_equal(pruned.data["d"]["bias"], state.data["d"]["bias"])
      assert pruned.parameters == state.parameters
      assert pruned.state == state.state
      assert pruned.frozen_parameters == state.frozen_parameters
    end

    test "works on a plain parameter map and ignores mask keys missing from it" do
      state = ModelState.freeze(two_dense_state(), &match?(["d0" | _], &1))
      mask = Pruning.magnitude_mask(two_dense_state(), 0.5, scope: :tensor)
      params = ModelState.trainable_parameters(state)

      masked = Pruning.apply_mask(params, mask)

      assert Map.keys(masked) == ["d1"]

      assert_equal(
        masked["d1"]["kernel"],
        Nx.select(mask["d1"]["kernel"], params["d1"]["kernel"], 0)
      )

      assert_equal(masked["d1"]["bias"], params["d1"]["bias"])
    end

    test "applies masks to composite parameters" do
      state = lstm_state()
      {pruned, mask} = Pruning.prune(state, 0.5, scope: :tensor)

      assert_equal(
        pruned.data["l"]["input_kernel"]["wii"],
        Nx.select(mask["l"]["input_kernel"]["wii"], state.data["l"]["input_kernel"]["wii"], 0)
      )

      assert_equal(pruned.data["l"]["bias"]["bi"], state.data["l"]["bias"]["bi"])
    end
  end

  describe "prune/3" do
    test "returns the pruned state and the mask" do
      state = dense_state()

      {pruned, mask} = Pruning.prune(state, 0.5)

      assert_equal(mask["d"]["kernel"], @kernel_mask)
      assert_equal(pruned.data["d"]["kernel"], Nx.select(@kernel_mask, @kernel, 0))
      assert_equal(pruned.data["d"]["bias"], state.data["d"]["bias"])
      assert pruned.parameters == state.parameters
    end

    test "works inside defn" do
      state = dense_state()

      assert_equal(InDefn.global_mask(state), Pruning.magnitude_mask(state, 0.5))
      assert_equal(InDefn.dense_mask(state), Pruning.magnitude_mask(state, 0.0))
      assert_equal(InDefn.tensor_mask(state), Pruning.magnitude_mask(state, 0.5, scope: :tensor))

      mask = Pruning.magnitude_mask(state, 0.5)
      assert_equal(InDefn.apply_mask(state, mask), Pruning.apply_mask(state, mask))

      {pruned, mask} = InDefn.prune(state)
      {expected_pruned, expected_mask} = Pruning.prune(state, 0.5)
      assert_equal(pruned, expected_pruned)
      assert_equal(mask, expected_mask)
    end
  end

  describe "sparsity/1 and global_sparsity/1" do
    test "report zero sparsity for a dense state" do
      state = dense_state()

      assert Pruning.sparsity(state) == %{"d" => %{"kernel" => 0.0, "bias" => 1.0}}
      assert Pruning.global_sparsity(state) == 4 / 12
    end

    test "report the size-weighted fraction of zeros after pruning" do
      state = put_in(dense_state(), [Access.key!(:data), "d", "bias"], Nx.broadcast(1.0, {4}))
      {pruned, _mask} = Pruning.prune(state, 0.5, scope: :tensor)

      assert Pruning.sparsity(pruned) == %{"d" => %{"kernel" => 0.5, "bias" => 0.0}}
      assert Pruning.global_sparsity(pruned) == 4 / 12
    end

    test "handle composite parameters and empty states" do
      {pruned, _mask} = Pruning.prune(lstm_state(), 0.5, scope: :tensor)

      assert %{"l" => %{"input_kernel" => %{"wii" => 0.5}, "hidden_kernel" => %{"whi" => 0.5}}} =
               Pruning.sparsity(pruned)

      assert Pruning.global_sparsity(ModelState.empty()) == 0.0
    end
  end

  describe "masked_optimizer/2" do
    defp train(optimizer, model, state, iterations) do
      data =
        Stream.repeatedly(fn -> {Nx.broadcast(1.0, {2, 4}), Nx.broadcast(0.5, {2, 1})} end)

      %Axon.Loop.State{step_state: %{model_state: trained}} =
        model
        |> Axon.Loop.trainer(:mean_squared_error, optimizer)
        |> Axon.Loop.run(data, state, iterations: iterations)

      trained
    end

    defp pruned_model do
      model =
        Axon.input("x", shape: {nil, 4})
        |> Axon.dense(8, name: "d")
        |> Axon.dense(1, name: "out")

      state =
        model
        |> init(Nx.template({1, 4}, :f32))
        |> put_in([Access.key!(:data), "d", "kernel"], Nx.iota({4, 8}, type: :f32))

      {pruned, mask} = Pruning.prune(state, 0.5, filter: &match?(["d", "kernel"], &1))
      {model, pruned, mask}
    end

    test "keeps pruned parameters at zero while training" do
      {model, pruned, mask} = pruned_model()

      for optimizer <- [:adam, Polaris.Optimizers.sgd(learning_rate: 0.1)] do
        trained = train(Pruning.masked_optimizer(optimizer, mask), model, pruned, 5)

        assert_equal(
          Nx.select(mask["d"]["kernel"], 0, trained.data["d"]["kernel"]),
          Nx.broadcast(0.0, {4, 8})
        )

        assert_not_equal(trained.data["d"]["kernel"], pruned.data["d"]["kernel"])
        assert_not_equal(trained.data["d"]["bias"], pruned.data["d"]["bias"])
        assert Pruning.sparsity(trained)["d"]["kernel"] == Pruning.sparsity(pruned)["d"]["kernel"]
      end
    end

    test "an unmasked optimizer does not keep pruned parameters at zero" do
      {model, pruned, mask} = pruned_model()

      trained = train(:adam, model, pruned, 5)

      assert_not_equal(
        Nx.select(mask["d"]["kernel"], 0, trained.data["d"]["kernel"]),
        Nx.broadcast(0.0, {4, 8})
      )
    end

    test "raises on an invalid optimizer" do
      message =
        "invalid optimizer :nope, expected an atom matching the name of an optimizer" <>
          " in Polaris.Optimizers or a tuple of {init_fn, update_fn}"

      assert_raise ArgumentError, message, fn -> Pruning.masked_optimizer(:nope, %{}) end

      assert_raise ArgumentError, ~r/invalid optimizer :module_info/, fn ->
        Pruning.masked_optimizer(:module_info, %{})
      end
    end
  end
end
