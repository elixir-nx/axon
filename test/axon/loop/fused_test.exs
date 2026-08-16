defmodule Axon.Loop.FusedTest do
  use Axon.Case, async: true

  alias Axon.Loop
  alias Axon.Loop.State

  setup do
    model = Axon.input("input", shape: {nil, 4}) |> Axon.dense(2)

    data =
      for i <- 1..10 do
        x = Nx.iota({2, 4}, type: :f32) |> Nx.add(i)
        {x, Nx.multiply(Nx.iota({2, 2}, type: :f32), i)}
      end

    %{model: model, data: data}
  end

  defp trainer(model) do
    Loop.trainer(model, :mean_squared_error, :sgd, log: 0, seed: 0)
  end

  defp run(loop, data, opts) do
    Loop.run(loop, data, Axon.ModelState.empty(), opts ++ [compiler: test_compiler()])
  end

  # Runs the loop with and without fusion, returning both final states
  defp run_both(loop, data, opts \\ []) do
    loop = %{loop | output_transform: & &1}
    {fuse, opts} = Keyword.pop(opts, :fuse, true)
    opts = Keyword.put_new(opts, :epochs, 2)

    {run(loop, data, opts), run(loop, data, [fuse: fuse] ++ opts)}
  end

  defp values_close?(left, right) do
    Nx.Defn.Composite.compatible?(left, right, fn l, r ->
      Nx.to_number(Nx.all_close(l, r, atol: 1.0e-4)) == 1
    end)
  end

  describe "equivalence with the unfused loop" do
    test "trains to the same parameters and metrics", %{model: model, data: data} do
      loop = model |> trainer() |> Loop.metric(:mean_absolute_error, "mae")

      {unfused, fused} = run_both(loop, data)

      assert_all_close(unfused.step_state[:model_state], fused.step_state[:model_state])
      assert_all_close(unfused.metrics, fused.metrics)
      assert unfused.epoch == fused.epoch
      assert unfused.iteration == fused.iteration
      assert unfused.status == fused.status
      assert unfused.event_counts == fused.event_counts
    end

    test "matches when the loop is bounded by :iterations", %{model: model, data: data} do
      loop = model |> trainer() |> Loop.metric(:mean_absolute_error, "mae")

      {unfused, fused} = run_both(loop, data, iterations: 4)

      assert_all_close(unfused.step_state[:model_state], fused.step_state[:model_state])
      assert_all_close(unfused.metrics, fused.metrics)
      assert unfused.event_counts == fused.event_counts
      assert fused.event_counts[:iteration_completed][:total] == 8
    end

    test "matches when the chunk size does not divide the data", %{model: model, data: data} do
      loop = model |> trainer() |> Loop.metric(:mean_absolute_error, "mae")

      # 10 batches in chunks of 3, so the last chunk of each epoch is padded
      {unfused, fused} = run_both(loop, data, fuse: 3)

      assert_all_close(unfused.step_state[:model_state], fused.step_state[:model_state])
      assert_all_close(unfused.metrics, fused.metrics)
      assert unfused.event_counts == fused.event_counts
    end

    test "matches for an evaluation loop", %{model: model, data: data} do
      %{step_state: %{model_state: model_state}} =
        model |> trainer() |> Map.put(:output_transform, & &1) |> run(data, epochs: 1)

      loop = model |> Loop.evaluator() |> Loop.metric(:mean_absolute_error, "mae")

      {unfused, fused} =
        ExUnit.CaptureIO.capture_io(fn ->
          unfused = Loop.run(loop, data, model_state, compiler: test_compiler())
          fused = Loop.run(loop, data, model_state, compiler: test_compiler(), fuse: true)
          send(self(), {unfused, fused})
        end)
        |> then(fn _ -> receive do: (both -> both) end)

      assert_all_close(unfused, fused)
    end

    test "matches for a loop with no metrics", %{data: data} do
      loop =
        Loop.loop(
          fn {x, _}, state -> %{state | total: Nx.add(state.total, Nx.sum(x))} end,
          fn _, _ -> %{total: Nx.tensor(0.0)} end
        )

      {unfused, fused} = run_both(loop, data)

      assert_all_close(unfused.step_state, fused.step_state)
    end
  end

  describe "data" do
    test "stops at data exhaustion when :iterations is not set", %{model: model, data: data} do
      %State{} =
        state =
        model
        |> trainer()
        |> Map.put(:output_transform, & &1)
        |> run(data, epochs: 1, fuse: true)

      assert state.event_counts[:iteration_completed][:total] == length(data)
    end

    test "pulls exactly as many batches as it steps", %{model: model} do
      {:ok, counter} = Agent.start_link(fn -> 0 end)

      data =
        Stream.repeatedly(fn ->
          Agent.update(counter, &(&1 + 1))
          {Nx.iota({2, 4}, type: :f32), Nx.iota({2, 2}, type: :f32)}
        end)

      model
      |> trainer()
      |> run(data, epochs: 2, iterations: 3, fuse: true)

      # One batch is taken up front by Axon.Loop.run to infer shapes
      assert Agent.get(counter, & &1) == 1 + 2 * 3
    end

    test "halts the enumeration of unfinished data", %{model: model} do
      {:ok, halted} = Agent.start_link(fn -> false end)

      data =
        Stream.resource(
          fn -> 0 end,
          fn i -> {[{Nx.iota({2, 4}, type: :f32), Nx.iota({2, 2}, type: :f32)}], i + 1} end,
          fn _ -> Agent.update(halted, fn _ -> true end) end
        )

      model
      |> trainer()
      |> run(data, epochs: 1, iterations: 2, fuse: true)

      assert Agent.get(halted, & &1)
    end
  end

  describe "events" do
    test "fires epoch level events with a full loop state", %{model: model, data: data} do
      pid = self()

      handler = fn event ->
        fn state ->
          send(pid, {event, state.epoch, state.step_state[:model_state]})
          {:continue, state}
        end
      end

      model
      |> trainer()
      |> Loop.handle_event(:epoch_started, handler.(:epoch_started))
      |> Loop.handle_event(:epoch_completed, handler.(:epoch_completed))
      |> run(data, epochs: 2, fuse: true)

      for epoch <- 0..1, event <- [:epoch_started, :epoch_completed] do
        assert_received {^event, ^epoch, %Axon.ModelState{}}
      end
    end

    test "fires iteration events once per iteration", %{model: model, data: data} do
      pid = self()

      model
      |> trainer()
      |> Loop.handle_event(:iteration_started, fn state ->
        send(pid, {:iteration_started, state.iteration})
        {:continue, state}
      end)
      |> Loop.handle_event(:iteration_completed, fn state ->
        send(pid, {:iteration_completed, state.iteration, state.metrics})
        {:continue, state}
      end)
      |> run(data, epochs: 1, fuse: true)

      for i <- 0..9 do
        assert_received {:iteration_started, ^i}
        assert_received {:iteration_completed, ^i, %{"loss" => _}}
      end

      refute_received {:iteration_started, _}
      refute_received {:iteration_completed, _, _}
    end

    test "gives iteration handlers a nil step state", %{model: model, data: data} do
      pid = self()

      model
      |> trainer()
      |> Loop.handle_event(:iteration_completed, fn state ->
        send(pid, {:step_state, state.step_state})
        {:continue, state}
      end)
      |> run(data, epochs: 1, iterations: 1, fuse: true)

      assert_received {:step_state, nil}
    end

    test "counts iteration events the same as the unfused loop", %{model: model, data: data} do
      loop =
        model
        |> trainer()
        |> Loop.handle_event(:iteration_completed, &{:continue, &1})

      {unfused, fused} = run_both(loop, data)

      assert unfused.event_counts == fused.event_counts
    end

    test "keeps metrics and handler metadata returned by iteration handlers", %{
      model: model,
      data: data
    } do
      state =
        model
        |> trainer()
        |> Loop.handle_event(:iteration_completed, fn state ->
          metrics = Map.put(state.metrics, "loss", Nx.tensor(1.0))
          metadata = Map.update(state.handler_metadata, :seen, 1, &(&1 + 1))
          {:continue, %{state | metrics: metrics, handler_metadata: metadata}}
        end)
        |> Map.put(:output_transform, & &1)
        |> run(data, epochs: 1, fuse: true)

      assert state.handler_metadata[:seen] == 10
      assert_equal(state.metrics[0]["loss"], Nx.tensor(1.0))
    end
  end

  describe "halting" do
    test "halts the epoch from an iteration handler", %{model: model, data: data} do
      loop =
        model
        |> trainer()
        |> Loop.handle_event(:iteration_completed, fn state ->
          if state.iteration == 2, do: {:halt_epoch, state}, else: {:continue, state}
        end)

      {unfused, fused} = run_both(loop, data)

      assert fused.event_counts == unfused.event_counts
      assert fused.event_counts[:epoch_halted] == 2
    end

    test "halting from an iteration handler takes effect at the chunk boundary", %{
      model: model,
      data: data
    } do
      loop =
        model
        |> trainer()
        |> Loop.handle_event(:iteration_completed, fn state ->
          if state.iteration == 2, do: {:halt_epoch, state}, else: {:continue, state}
        end)

      {unfused, fused} = run_both(loop, data, fuse: 1)

      # A chunk of one batch halts exactly where the unfused loop does
      assert fused.event_counts == unfused.event_counts
      assert_all_close(fused.step_state[:model_state], unfused.step_state[:model_state])

      # A larger chunk finishes the chunk it was halted in, so it trains further
      {_, chunked} = run_both(loop, data, fuse: 32)
      assert chunked.event_counts == unfused.event_counts
      refute values_close?(chunked.step_state[:model_state], unfused.step_state[:model_state])
    end

    test "halts the loop from an iteration handler", %{model: model, data: data} do
      loop =
        model
        |> trainer()
        |> Loop.handle_event(:iteration_completed, fn state ->
          if state.iteration == 1, do: {:halt_loop, state}, else: {:continue, state}
        end)

      {unfused, fused} = run_both(loop, data, epochs: 5)

      assert fused.status == :halted
      assert fused.epoch == 0
      assert fused.event_counts[:iteration_completed][:total] == 2
      assert fused.event_counts == unfused.event_counts
    end

    test "halts from an epoch handler", %{model: model, data: data} do
      loop =
        model
        |> trainer()
        |> Loop.handle_event(:epoch_completed, fn state ->
          if state.epoch == 0, do: {:halt_loop, state}, else: {:continue, state}
        end)

      {unfused, fused} = run_both(loop, data, epochs: 5)

      assert fused.status == :halted
      assert fused.epoch == 0
      assert fused.event_counts == unfused.event_counts
    end
  end

  describe "validation" do
    test "raises when jit compilation is disabled", %{model: model, data: data} do
      assert_raise ArgumentError, ~r/cannot fuse a loop with jit_compile\?: false/, fn ->
        model |> trainer() |> run(data, epochs: 1, fuse: true, jit_compile?: false)
      end
    end

    test "raises on an invalid chunk size", %{model: model, data: data} do
      assert_raise ArgumentError, ~r/invalid value 0 for :fuse/, fn ->
        model |> trainer() |> run(data, epochs: 1, fuse: 0)
      end
    end
  end
end
