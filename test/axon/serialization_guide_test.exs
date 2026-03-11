defmodule Axon.SerializationGuideTest do
  @moduledoc """
  Tests that validate the examples in guides/serialization/saving_and_loading.livemd.
  Run with: mix test test/axon/serialization_guide_test.exs
  """
  use Axon.Case, async: false

  @tmp_path Path.join(
              System.tmp_dir!(),
              "axon_serialization_guide_test_#{:erlang.unique_integer([:positive])}"
            )

  setup do
    File.mkdir_p!(@tmp_path)
    on_exit(fn -> File.rm_rf!(@tmp_path) end)
    [tmp_path: @tmp_path]
  end

  describe "saving and loading guide examples" do
    test "full flow: train → save params → load → predict", %{tmp_path: tmp_path} do
      # Same model as the guide
      model =
        Axon.input("data")
        |> Axon.dense(8)
        |> Axon.relu()
        |> Axon.dense(4)
        |> Axon.relu()
        |> Axon.dense(1)

      loop = Axon.Loop.trainer(model, :mean_squared_error, :sgd, log: 0)

      train_data =
        Stream.repeatedly(fn ->
          {xs, _} =
            Nx.Random.normal(
              Nx.Random.key(:erlang.phash2({self(), System.unique_integer([:monotonic])})),
              shape: {8, 1}
            )

          {xs, Nx.sin(xs)}
        end)

      # Train
      trained_model_state =
        Axon.Loop.run(loop, train_data, Axon.ModelState.empty(), epochs: 2, iterations: 50)

      # Extract and save params (as in guide)
      params =
        case trained_model_state do
          %Axon.ModelState{data: data} -> data
          params when is_map(params) -> params
        end

      params_path = Path.join(tmp_path, "model_params.axon")
      params = Nx.backend_transfer(params)
      params_bytes = Nx.serialize(params)
      File.write!(params_path, params_bytes)

      # Load and predict (input shape must match training: {batch, 1} for 1 feature)
      loaded_params = File.read!(params_path) |> Nx.deserialize()
      input = Nx.tensor([[1.0]])

      prediction = Axon.predict(model, loaded_params, %{"data" => input})

      assert Nx.rank(prediction) == 2
      assert Nx.shape(prediction) == {1, 1}
    end

    test "checkpoint and resume flow", %{tmp_path: tmp_path} do
      model =
        Axon.input("data")
        |> Axon.dense(4)
        |> Axon.relu()
        |> Axon.dense(1)

      checkpoint_path = Path.join(tmp_path, "checkpoints")

      loop =
        model
        |> Axon.Loop.trainer(:mean_squared_error, :sgd, log: 0)
        |> Axon.Loop.checkpoint(path: checkpoint_path, event: :epoch_completed)

      train_data = [
        {Nx.tensor([[1.0, 2.0, 3.0, 4.0]]), Nx.tensor([[1.0]])},
        {Nx.tensor([[2.0, 3.0, 4.0, 5.0]]), Nx.tensor([[2.0]])}
      ]

      Axon.Loop.run(loop, train_data, Axon.ModelState.empty(), epochs: 2)

      # Verify checkpoint was saved
      ckpt_files = File.ls!(checkpoint_path) |> Enum.sort()
      assert length(ckpt_files) == 2
      assert Enum.any?(ckpt_files, &String.contains?(&1, "checkpoint_"))

      # Load checkpoint and extract params for inference
      ckpt_file = Path.join(checkpoint_path, List.first(ckpt_files))
      state = File.read!(ckpt_file) |> Axon.Loop.deserialize_state()

      %{model_state: model_state} = state.step_state
      params = model_state.data

      # Run inference with extracted params
      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      prediction = Axon.predict(model, params, %{"data" => input})

      assert Nx.rank(prediction) == 2
      assert Nx.shape(prediction) == {1, 1}
    end

    test "resume from checkpoint with from_state", %{tmp_path: tmp_path} do
      model =
        Axon.input("data")
        |> Axon.dense(2)
        |> Axon.dense(1)

      checkpoint_path = Path.join(tmp_path, "checkpoints_resume")

      loop =
        model
        |> Axon.Loop.trainer(:mean_squared_error, :sgd, log: 0)
        |> Axon.Loop.checkpoint(path: checkpoint_path, event: :epoch_completed)

      train_data = [{Nx.tensor([[1.0, 2.0]]), Nx.tensor([[1.0]])}]

      # Run for 1 epoch
      Axon.Loop.run(loop, train_data, Axon.ModelState.empty(), epochs: 1)

      # Load checkpoint and resume
      [ckpt_file] = File.ls!(checkpoint_path)
      state = File.read!(Path.join(checkpoint_path, ckpt_file)) |> Axon.Loop.deserialize_state()

      resumed_loop =
        model
        |> Axon.Loop.trainer(:mean_squared_error, :sgd, log: 0)
        |> Axon.Loop.from_state(state)

      # Resume - should complete without error
      result = Axon.Loop.run(resumed_loop, train_data, Axon.ModelState.empty(), epochs: 2)

      assert %Axon.ModelState{} = result
    end
  end
end
