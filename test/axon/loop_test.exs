defmodule Axon.LoopTest do
  use Axon.Case, async: true
  import ExUnit.CaptureLog

  alias Axon.Loop
  alias Axon.Loop.State

  describe "factories" do
    test "loop/2 creates a basic loop with defaults" do
      step_fn = fn _, _ -> Nx.tensor(1) end

      assert %Loop{init: init_fn, step: update_fn} =
               Loop.loop(step_fn)

      assert_equal(init_fn.(Nx.tensor(1), %{}), %{})
      assert_equal(update_fn.({}, %{}), Nx.tensor(1))
    end

    test "trainer/3 returns a supervised training loop with basic case" do
      model = Axon.input("input", shape: {nil, 1})

      valid_axon_losses = [
        :binary_cross_entropy,
        :categorical_cross_entropy,
        :categorical_hinge,
        :hinge,
        :kl_divergence,
        :log_cosh,
        :mean_absolute_error,
        :mean_squared_error,
        :poisson,
        :soft_margin
      ]

      valid_axon_optimizers =
        Polaris.Optimizers.__info__(:functions)
        |> Enum.map(fn {k, _} -> k end)
        |> Enum.uniq()

      for loss <- valid_axon_losses do
        for optimizer <- valid_axon_optimizers do
          assert %Loop{init: init_fn, step: update_fn} =
                   Loop.trainer(model, loss, optimizer)

          assert %{model_state: %Axon.ModelState{}} =
                   pstate =
                   init_fn.({Nx.tensor([[1]]), Nx.tensor([[1]])}, Axon.ModelState.empty())

          assert %{model_state: %{}, y_true: tar, y_pred: pred} =
                   apply(Nx.Defn.jit(update_fn), [{Nx.tensor([[1]]), Nx.tensor([[1]])}, pstate])

          assert_equal(tar, Nx.tensor([[1]]))
          assert_equal(pred, Nx.tensor([[1]]))
        end
      end
    end

    test "trainer/3 returns a supervised training loop with custom loss" do
      model = Axon.input("input", shape: {nil, 1})
      custom_loss_fn = fn _, _ -> Nx.tensor(5.0, backend: Nx.Defn.Expr) end

      assert %Loop{init: init_fn, step: update_fn} =
               Loop.trainer(model, custom_loss_fn, :adam)

      assert %{model_state: %{}} =
               pstate = init_fn.({Nx.tensor([[1]]), Nx.tensor([[1]])}, Axon.ModelState.empty())

      assert %{model_state: %{}, y_true: tar, y_pred: pred, loss: loss} =
               apply(Nx.Defn.jit(update_fn), [{Nx.tensor([[1]]), Nx.tensor([[1]])}, pstate])

      assert_equal(tar, Nx.tensor([[1]]))
      assert_equal(pred, Nx.tensor([[1]]))
      assert_equal(loss, Nx.tensor(5.0))
    end

    test "trainer/3 returns a supervised training loop with custom optimizer" do
      model = Axon.input("input", shape: {nil, 1})
      optimizer = Polaris.Optimizers.rmsprop(learning_rate: 1.0e-3)

      assert %Loop{init: init_fn, step: update_fn} =
               Loop.trainer(model, :mean_squared_error, optimizer)

      assert %{model_state: %{}} =
               pstate = init_fn.({Nx.tensor([[1]]), Nx.tensor([[1]])}, Axon.ModelState.empty())

      assert %{model_state: %{}, y_true: tar, y_pred: pred} =
               apply(Nx.Defn.jit(update_fn), [{Nx.tensor([[1]]), Nx.tensor([[1]])}, pstate])

      assert_equal(tar, Nx.tensor([[1]]))
      assert_equal(pred, Nx.tensor([[1]]))
    end

    test "trainer/3 returns a supervised training loop with custom model" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.build(mode: :train)

      assert %Loop{init: init_fn, step: update_fn} =
               Loop.trainer(model, :mean_squared_error, :adam)

      assert %{model_state: %{}} =
               pstate = init_fn.({Nx.tensor([[1]]), Nx.tensor([[1]])}, Axon.ModelState.empty())

      assert %{model_state: %{}, y_true: tar, y_pred: pred} =
               apply(Nx.Defn.jit(update_fn), [{Nx.tensor([[1]]), Nx.tensor([[1]])}, pstate])

      assert_equal(tar, Nx.tensor([[1]]))
      assert_equal(pred, Nx.tensor([[1]]))
    end

    test "trainer/3 returns a supervised training loop with multi-loss" do
      model =
        {Axon.input("input_0", shape: {nil, 1}), Axon.input("input_1", shape: {nil, 1})}
        |> Axon.container()

      assert %Loop{init: init_fn, step: update_fn} =
               Loop.trainer(model, [mean_squared_error: 0.5, mean_absolute_error: 0.5], :adam)

      assert %{model_state: %{}} =
               pstate =
               init_fn.(
                 {%{"input_0" => Nx.tensor([[2]]), "input_1" => Nx.tensor([[2]])},
                  {Nx.tensor([[2]]), Nx.tensor([[2]])}},
                 Axon.ModelState.empty()
               )

      assert %{model_state: %{}, y_true: tar, y_pred: pred, loss: loss} =
               apply(Nx.Defn.jit(update_fn), [
                 {%{"input_0" => Nx.tensor([[1]]), "input_1" => Nx.tensor([[1]])},
                  {Nx.tensor([[2]]), Nx.tensor([[2]])}},
                 pstate
               ])

      assert_equal(tar, {Nx.tensor([[2]]), Nx.tensor([[2]])})
      assert_equal(pred, {Nx.tensor([[1]]), Nx.tensor([[1]])})
      assert_equal(loss, Nx.tensor(1.0))
    end

    test "trainer/3 raises on bad inputs" do
      assert_raise ArgumentError, ~r/Invalid/, fn ->
        apply(Axon.Loop, :trainer, [:foo, :mean_squared_error, :adam])
      end

      assert_raise ArgumentError, ~r/Invalid/, fn ->
        apply(Axon.Loop, :trainer, [Axon.input("input", shape: {nil, 1}), :foo, :adam])
      end

      assert_raise ArgumentError, ~r/Invalid/, fn ->
        apply(Axon.Loop, :trainer, [
          Axon.input("input", shape: {nil, 1}),
          :mean_squared_error,
          :foo
        ])
      end
    end

    test "evaluator/1 returns a supervised evaluator loop" do
      inp = Nx.tensor([[1]])

      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(1)
      {init_fn, _} = Axon.build(model)

      model_state = init_fn.(inp, Axon.ModelState.empty())

      expected_pred = Axon.predict(model, model_state, inp)

      assert %Loop{init: init_fn, step: update_fn} =
               Loop.evaluator(model)

      assert %{model_state: _, y_true: _, y_pred: _} =
               pstate = init_fn.({Nx.tensor([[1]]), Nx.tensor([[2]])}, model_state)

      assert %{y_true: tar, y_pred: pred} =
               apply(Nx.Defn.jit(update_fn), [{Nx.tensor([[1]]), Nx.tensor([[2]])}, pstate])

      assert_equal(tar, Nx.tensor([[2]]))
      assert_equal(pred, expected_pred)
    end

    test "evaluator/1 runs a supervised evaluator loop" do
      inp = Nx.tensor([[1]])

      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(1)
      {init_fn, _} = Axon.build(model)

      model_state = init_fn.(inp, Axon.ModelState.empty())
      data = [{Nx.tensor([[1]]), Nx.tensor([[2]])}]

      assert %Loop{} = loop = Loop.evaluator(model)
      assert %Loop{} = loop = Loop.metric(loop, :mean_absolute_error)

      assert ExUnit.CaptureIO.capture_io(fn ->
               assert %State{metrics: %{0 => %{"mean_absolute_error" => _}}} =
                        Loop.run(loop, data, model_state)
             end) =~ "Batch"
    end

    test "eval_step/1 evaluates model on a single batch" do
      inp = Nx.tensor([0, 1, 0, 1, 0, 1]) |> Nx.new_axis(-1)
      tar = Nx.tensor([1, 0, 1, 0, 1, 0]) |> Nx.new_axis(-1)

      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(1)
      {init_fn, _} = Axon.build(model)

      model_state = init_fn.(inp, Axon.ModelState.empty())

      {init_fn, step_fn} = Axon.Loop.eval_step(model)
      pstate = apply(Nx.Defn.jit(init_fn), [{Nx.tensor([[1]]), Nx.tensor([[1]])}, model_state])

      # Older versions of the loop API had backend mismatches,
      # so just verify there was a successful result here
      assert %{y_true: _, y_pred: _} = apply(Nx.Defn.jit(step_fn), [{inp, tar}, pstate])
    end

    test "eval_step/1 evaluates models which output a container" do
      inp = Nx.tensor([[1.0]])
      tar = Nx.tensor([[2.0]])

      dense = Axon.input("input", shape: {nil, 1}) |> Axon.dense(1)

      model =
        Axon.container(%{
          logits: dense,
          maybe: Axon.input("optional", shape: {nil, 1}, optional: true) |> Axon.optional()
        })

      {init_fn, _} = Axon.build(model)
      model_state = init_fn.(%{"input" => inp}, Axon.ModelState.empty())

      {eval_init_fn, eval_step_fn} = Axon.Loop.eval_step(model)

      assert %{y_pred: %{logits: logits, maybe: %Axon.None{}}} =
               pstate = apply(Nx.Defn.jit(eval_init_fn), [{%{"input" => inp}, tar}, model_state])

      assert_equal(logits, Nx.tensor([[0.0]]))

      assert %{y_pred: %{logits: logits, maybe: %Axon.None{}}} =
               apply(Nx.Defn.jit(eval_step_fn), [{%{"input" => inp}, tar}, pstate])

      assert_equal(logits, Axon.predict(model, model_state, %{"input" => inp}).logits)
    end

    test "validate/4 works with models which output a container" do
      inp = Nx.tensor([[1.0]])
      tar = Nx.tensor([[2.0]])
      data = List.duplicate({inp, tar}, 2)

      model =
        Axon.input("input", shape: {nil, 1})
        |> Axon.dense(1)
        |> then(&Axon.container(%{logits: &1}))

      loss = fn y_true, %{logits: logits} ->
        Axon.Losses.mean_squared_error(y_true, logits, reduction: :mean)
      end

      metric = fn y_true, %{logits: logits} ->
        Axon.Metrics.mean_absolute_error(y_true, logits)
      end

      loop =
        model
        |> Axon.Loop.trainer(loss, :sgd)
        |> Axon.Loop.metric(metric, "mae")
        |> Axon.Loop.validate(model, data)

      ExUnit.CaptureIO.capture_io(fn ->
        assert %Axon.Loop.State{step_state: %{model_state: %Axon.ModelState{}}} =
                 Axon.Loop.run(loop, data, Axon.ModelState.empty(), epochs: 1)
      end)
    end

    test "train_step/3 updates stateful layers after single step" do
      val = Nx.broadcast(1, {1, 8})

      model = Axon.constant(val) |> Axon.batch_norm(name: "batch_norm")
      {init_fn, step_fn} = Axon.Loop.train_step(model, :mean_squared_error, :adam)

      state = init_fn.({val, val}, Axon.ModelState.empty())
      state = step_fn.({val, val}, state)

      assert_all_close(state.model_state.data["batch_norm"]["mean"], Nx.broadcast(0.9, {8}))
      assert_all_close(state.model_state.data["batch_norm"]["var"], Nx.broadcast(0.1, {8}))

      val = Nx.broadcast(1, {1, 1, 8})
      model = Axon.constant(val) |> Axon.instance_norm(name: "instance_norm")
      {init_fn, step_fn} = Axon.Loop.train_step(model, :mean_squared_error, :adam)

      state = init_fn.({val, val}, Axon.ModelState.empty())
      state = step_fn.({val, val}, state)

      assert_all_close(state.model_state.data["instance_norm"]["mean"], Nx.broadcast(0.9, {8}))
      assert_all_close(state.model_state.data["instance_norm"]["var"], Nx.broadcast(0.1, {8}))
    end

    test "train_step/3 initializes the step state in the types the step produces" do
      model = precision_model({:f, 64})
      batch = precision_batch()

      for loss_scale <- [:identity, :dynamic, :static] do
        {init_fn, step_fn} =
          Loop.train_step(model, :mean_squared_error, :adam, loss_scale: loss_scale)

        state = init_fn.(batch, Axon.ModelState.empty())
        next = step_fn.(batch, state)

        assert Nx.type(state.loss) == {:f, 64}
        assert Nx.type(elem(state.optimizer_state, 1).mu["dense_0"]["kernel"]) == {:f, 64}
        assert Nx.type(state.model_state.data["dense_0"]["kernel"]) == {:f, 64}
        assert leaf_types(state) == leaf_types(next)
      end
    end

    test "train_step/3 keeps f32 state under an f16 policy" do
      model = precision_model({:f, 16})
      batch = precision_batch()

      {init_fn, step_fn} = Loop.train_step(model, :mean_squared_error, :adam)

      state = init_fn.(batch, Axon.ModelState.empty())
      next = step_fn.(batch, state)

      assert Nx.type(state.loss) == {:f, 32}
      assert Nx.type(elem(state.optimizer_state, 1).mu["dense_0"]["kernel"]) == {:f, 32}
      assert Nx.type(state.model_state.data["dense_0"]["kernel"]) == {:f, 16}
      assert leaf_types(state) == leaf_types(next)
    end
  end

  describe "metrics" do
    test "uses default names with out of the box metrics" do
      step_fn = fn _, _ -> 1 end

      loop =
        step_fn
        |> Loop.loop()
        |> Loop.metric(:accuracy)

      %Loop{metrics: metrics} = loop

      assert Map.has_key?(metrics, "accuracy")
    end

    test "raises when no name provided for custom metric" do
      step_fn = fn _, _ -> 1 end

      assert_raise ArgumentError, ~r/must provide/, fn ->
        step_fn
        |> Loop.loop()
        |> Loop.metric(&Axon.Metrics.accuracy/2)
      end
    end

    test "warns on duplicate metrics" do
      step_fn = fn _, _ -> 1 end

      assert capture_log(fn ->
               step_fn
               |> Axon.Loop.loop()
               |> Axon.Loop.metric(:accuracy)
               |> Axon.Loop.metric(:accuracy)
             end) =~ "Metric accuracy declared twice in loop."
    end

    test "computes running average by default with supervised transform" do
      step_fn = fn _, _ -> 1 end

      loop =
        step_fn
        |> Loop.loop()
        |> Loop.metric(:accuracy)

      assert %Loop{metrics: %{"accuracy" => {avg_acc_fun, _}}} = loop

      # Torchx cannot compute u64-output sum, so we need to force a signed type
      output = %{
        foo: 1,
        y_true: Nx.tensor([[1, 0, 1]]),
        y_pred: Nx.tensor([[0.8, 0.2, 0.8]])
      }

      cur_avg_acc = 0.5
      i = 1

      assert_equal(avg_acc_fun.(cur_avg_acc, List.wrap(output), i), Nx.tensor(0.75))
    end

    test "computes a running sum with custom transform" do
      step_fn = fn _, _ -> 1 end

      loop =
        step_fn
        |> Loop.loop()
        |> Loop.metric(:true_positives, "tp", :running_sum, &Tuple.to_list/1)

      assert %Loop{metrics: %{"tp" => {sum_tp_fun, _}}} = loop

      # the type for torchx needs to be signed because u64 is not supported there
      type =
        if Nx.default_backend() == Torchx.Backend do
          {:s, 8}
        end

      output = {Nx.tensor([1, 0, 1], type: type), Nx.tensor([0, 1, 1], type: type)}
      cur_sum = 25
      i = 10

      assert_equal(sum_tp_fun.(cur_sum, List.wrap(output), i), Nx.tensor(26))
    end
  end

  # A dense model with every parameter, computation and output in `type`.
  defp precision_model(type) do
    policy = Axon.MixedPrecision.create_policy(params: type, compute: type, output: type)

    Axon.input("input", shape: {nil, 4})
    |> Axon.dense(8, activation: :relu)
    |> Axon.dense(1)
    |> Axon.MixedPrecision.apply_policy(policy)
  end

  defp precision_batch(target_type \\ {:f, 32}) do
    {Nx.iota({2, 4}, type: :f32), Nx.tensor([[1.0], [0.0]], type: target_type)}
  end

  defp leaf_types(container) do
    container
    |> List.wrap()
    |> Nx.Defn.Composite.flatten_list()
    |> Enum.map(&Nx.type/1)
  end

  describe "step state types" do
    test "runs a strict loop with an f64 mixed precision policy" do
      model = precision_model({:f, 64})
      data = List.duplicate(precision_batch(), 3)

      ExUnit.CaptureIO.capture_io(fn ->
        state =
          model
          |> Loop.trainer(:mean_squared_error, :adam)
          |> Loop.metric(:mean_absolute_error)
          |> Loop.run(data, Axon.ModelState.empty(), epochs: 2)

        assert %State{epoch: 2, metrics: metrics, step_state: step_state} = state
        assert Nx.type(step_state.loss) == {:f, 64}
        assert Nx.type(step_state.model_state.data["dense_0"]["kernel"]) == {:f, 64}
        assert Nx.type(metrics[1]["loss"]) == {:f, 64}
        assert Nx.type(metrics[1]["mean_absolute_error"]) == {:f, 64}
      end)
    end

    test "runs a strict loop with f64 targets and an f32 model" do
      model = Axon.input("input", shape: {nil, 4}) |> Axon.dense(8) |> Axon.dense(1)
      data = List.duplicate(precision_batch({:f, 64}), 3)

      for optimizer <- [:sgd, :adam] do
        ExUnit.CaptureIO.capture_io(fn ->
          state =
            model
            |> Loop.trainer(:mean_squared_error, optimizer)
            |> Loop.run(data, Axon.ModelState.empty(), epochs: 1)

          assert %State{epoch: 1, metrics: metrics, step_state: step_state} = state
          assert Nx.type(step_state.loss) == {:f, 64}
          assert Nx.type(step_state.model_state.data["dense_0"]["kernel"]) == {:f, 32}
          assert Nx.type(metrics[0]["loss"]) == {:f, 64}
        end)
      end
    end
  end

  describe "looping" do
    test "returns initial state with epochs 0" do
      step_fn = fn _, _ -> 1 end

      state =
        step_fn
        |> Loop.loop()
        |> Loop.run([Nx.tensor(1)], %{}, epochs: 0)

      assert %State{epoch: 0, iteration: 0, times: %{}, metrics: %{}, step_state: pstate} = state

      assert pstate == %{}
    end

    test "propagates user-defined numerical data inside step_state" do
      Axon.input("input", shape: {nil, 1})
      |> Axon.dense(1)
      |> Loop.trainer(:binary_cross_entropy, :sgd, log: 0)
      |> Loop.handle_event(
        :epoch_completed,
        fn %State{step_state: pstate} = state ->
          {
            :continue,
            %State{
              state
              | step_state:
                  case pstate[:counter] do
                    nil -> Map.put(pstate, :counter, 0)
                    counter -> %{pstate | counter: Nx.to_number(counter) + 1}
                  end
            }
          }
        end
      )
      |> Loop.handle_event(
        :completed,
        fn %State{step_state: %{counter: counter}} = state ->
          assert 4 = counter

          {:continue, state}
        end
      )
      |> Loop.run(
        [{Nx.tensor([[1.0]]), Nx.tensor([[1.0]])}],
        Axon.ModelState.empty(),
        epochs: 5,
        strict?: false
      )
    end

    test "propagates user-defined numerical data inside step_state when it is nested into a tuple" do
      Axon.input("input", shape: {nil, 1})
      |> Axon.dense(1)
      |> Loop.trainer(:binary_cross_entropy, :sgd, log: 0)
      |> Loop.handle_event(
        :epoch_completed,
        fn %State{step_state: pstate} = state ->
          {
            :continue,
            %State{
              state
              | step_state:
                  case pstate[:counter] do
                    nil ->
                      Map.put(pstate, :counter, {{0}, 0})

                    {{counter}, _} ->
                      next_counter_value = Nx.to_number(counter) + 1
                      %{pstate | counter: {{next_counter_value}, next_counter_value}}
                  end
            }
          }
        end
      )
      |> Loop.handle_event(
        :completed,
        fn %State{step_state: %{counter: counter}} = state ->
          assert {{4}, 4} = counter

          {:continue, state}
        end
      )
      |> Loop.run(
        [{Nx.tensor([[1.0]]), Nx.tensor([[1.0]])}],
        Axon.ModelState.empty(),
        epochs: 5,
        strict?: false
      )
    end
  end

  describe "trainer" do
    test "returns clear error on bad inputs" do
      model = Axon.input("input")
      data = Stream.repeatedly(fn -> Nx.tensor(5) end)

      assert_raise ArgumentError, ~r/invalid arguments/, fn ->
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
        |> Axon.Loop.run(data, %{})
      end
    end
  end

  describe "evaluator" do
    test "returns clear error on bad inputs" do
      model = Axon.input("input")
      data = Stream.repeatedly(fn -> Nx.tensor(5) end)

      assert_raise ArgumentError, ~r/invalid arguments/, fn ->
        model
        |> Axon.Loop.evaluator()
        |> Axon.Loop.run(data, %{})
      end
    end
  end

  describe "events" do
    def run_dummy_loop!(event, epochs, iterations) do
      model = Axon.input("foo")

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
      |> send_handler(event)
      |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: epochs, iterations: iterations)
    end

    def send_handler(loop, event) do
      Axon.Loop.handle_event(loop, event, fn state ->
        send(self(), event)
        {:continue, state}
      end)
    end

    def send_event_counts_handler(loop, event) do
      Axon.Loop.handle_event(loop, event, fn state ->
        send(self(), {event, state.event_counts})
        {:continue, state}
      end)
    end

    def continue_handler(loop, event) do
      Axon.Loop.handle_event(loop, event, &{:continue, &1})
    end

    test "fires correctly on :started" do
      ExUnit.CaptureIO.capture_io(fn ->
        run_dummy_loop!(:started, 5, 10)

        assert_received :started
        refute_received :started
      end)
    end

    test "fires correctly on :epoch_started" do
      ExUnit.CaptureIO.capture_io(fn ->
        run_dummy_loop!(:epoch_started, 5, 10)
      end)

      for _ <- 1..5 do
        assert_received :epoch_started
      end

      refute_received :epoch_started
    end

    test "fires correctly on :epoch_completed" do
      ExUnit.CaptureIO.capture_io(fn ->
        run_dummy_loop!(:epoch_completed, 5, 10)
      end)

      for _ <- 1..5 do
        assert_received :epoch_completed
      end

      refute_received :epoch_completed
    end

    test "fires correctly on :iteration_started" do
      ExUnit.CaptureIO.capture_io(fn ->
        run_dummy_loop!(:iteration_started, 5, 10)
      end)

      for _ <- 1..50 do
        assert_received :iteration_started
      end

      refute_received :iteration_started
    end

    test "fires correctly on :iteration_completed" do
      ExUnit.CaptureIO.capture_io(fn ->
        run_dummy_loop!(:iteration_completed, 5, 10)
      end)

      for _ <- 1..50 do
        assert_received :iteration_completed
      end

      refute_received :iteration_completed
    end

    test "fires correctly on :epoch_halted" do
      model = Axon.input("foo")

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      ExUnit.CaptureIO.capture_io(fn ->
        model
        |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
        |> Axon.Loop.handle_event(:iteration_completed, fn state ->
          {:halt_epoch, state}
        end)
        |> send_handler(:epoch_halted)
        |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 5, iterations: 10)
      end)

      for _ <- 1..5 do
        assert_received :epoch_halted
      end

      refute_received :epoch_halted
    end

    test "events fire in order" do
      model = Axon.input("foo")

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      ExUnit.CaptureIO.capture_io(fn ->
        model
        |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
        |> send_handler(:started)
        |> send_handler(:epoch_started)
        |> send_handler(:iteration_started)
        |> send_handler(:iteration_completed)
        |> send_handler(:epoch_completed)
        |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 1, iterations: 1)
      end)

      assert_received :started
      assert_received :epoch_started
      assert_received :iteration_started
      assert_received :iteration_completed
      assert_received :epoch_completed

      refute_received _
    end

    test "events are counted correctly" do
      model = Axon.input("foo")

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      ExUnit.CaptureIO.capture_io(fn ->
        # loop with multiple :iteration_started handlers to test that
        # the event counts are only incremented once per event
        model
        |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
        |> send_event_counts_handler(:started)
        |> send_event_counts_handler(:epoch_started)
        |> continue_handler(:iteration_started)
        |> continue_handler(:iteration_started)
        |> send_event_counts_handler(:epoch_completed)
        |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 2, iterations: 10)
      end)

      assert_received {:started,
                       %{
                         started: 1
                       }}

      assert_received {:epoch_started,
                       %{
                         started: 1,
                         epoch_started: 1
                       }}

      assert_received {:epoch_completed,
                       %{
                         started: 1,
                         epoch_started: 1,
                         epoch_completed: 1,
                         iteration_started: %{total: 10, epoch: 0},
                         iteration_completed: %{total: 10, epoch: 0}
                       }}

      assert_received {:epoch_started,
                       %{
                         started: 1,
                         epoch_started: 2,
                         epoch_completed: 1,
                         iteration_started: %{total: 10, epoch: 0},
                         iteration_completed: %{total: 10, epoch: 0}
                       }}

      assert_received {:epoch_completed,
                       %{
                         started: 1,
                         epoch_started: 2,
                         epoch_completed: 2,
                         iteration_started: %{total: 20, epoch: 0},
                         iteration_completed: %{total: 20, epoch: 0}
                       }}

      refute_received _
    end
  end

  describe "filters" do
    def run_dummy_loop!(event, filter, epochs, iterations) do
      model = Axon.input("foo")

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :sgd, log: -1)
      |> send_handler(event, filter)
      |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: epochs, iterations: iterations)
    end

    def send_handler(loop, event, filter) do
      Axon.Loop.handle_event(
        loop,
        event,
        fn state ->
          send(self(), event)
          {:continue, state}
        end,
        filter
      )
    end

    test "supports an :always filter" do
      run_dummy_loop!(:iteration_started, :always, 5, 10)

      for _ <- 1..50 do
        assert_received :iteration_started
      end

      refute_received :iteration_started
    end

    test "supports an every: n filter" do
      run_dummy_loop!(:iteration_started, [every: 2], 5, 10)

      for _ <- 1..25 do
        assert_received :iteration_started
      end

      refute_received :iteration_started

      run_dummy_loop!(:iteration_completed, [every: 3], 3, 10)

      for _ <- 1..10 do
        assert_received :iteration_completed
      end

      refute_received :iteration_completed
    end

    test "supports after: n filter" do
      run_dummy_loop!(:iteration_started, [after: 10], 5, 10)

      for _ <- 1..40 do
        assert_received :iteration_started
      end

      refute_received :iteration_started

      run_dummy_loop!(:iteration_completed, [after: 10], 5, 10)

      for _ <- 1..40 do
        assert_received :iteration_completed
      end

      refute_received :iteration_completed
    end

    test "supports before: n filter" do
      run_dummy_loop!(:iteration_started, [before: 10], 5, 10)

      for _ <- 1..9 do
        assert_received :iteration_started
      end

      refute_received :iteration_started

      run_dummy_loop!(:iteration_completed, [before: 10], 5, 10)

      for _ <- 1..9 do
        assert_received :iteration_completed
      end

      refute_received :iteration_completed
    end

    test "supports once: n filter" do
      run_dummy_loop!(:iteration_started, [once: 30], 5, 10)

      assert_received :iteration_started
      refute_received :iteration_started

      run_dummy_loop!(:iteration_completed, [once: 30], 5, 10)

      assert_received :iteration_completed
      refute_received :iteration_completed
    end

    test "supports hybrid filter" do
      run_dummy_loop!(:iteration_started, [every: 2, after: 10, before: 40], 5, 10)

      for _ <- 1..15 do
        assert_received :iteration_started
      end

      refute_received :iteration_started
    end

    test "supports :first filter" do
      run_dummy_loop!(:iteration_started, :first, 5, 10)

      assert_received :iteration_started
      refute_received :iteration_started
    end

    test "supports function filter" do
      fun = fn
        %{event_counts: counts}, event -> counts[event][:total] == 5
      end

      run_dummy_loop!(:iteration_started, fun, 5, 10)

      assert_received :iteration_started
      refute_received :iteration_started
    end
  end

  describe "serialization" do
    test "serialize_state/deserialize_state preserve loop state" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      optimizer = Polaris.Optimizers.adam(learning_rate: 1.0e-2)
      loss = :binary_cross_entropy

      {init_fn, _} = Axon.Loop.train_step(model, loss, optimizer)
      step_state = init_fn.({Nx.tensor([[1]]), Nx.tensor([[1, 0]])}, Axon.ModelState.empty())
      state = %State{step_state: step_state}

      serialized = Axon.Loop.serialize_state(state)
      %State{step_state: deserialized_step_state} = Axon.Loop.deserialize_state(serialized)

      assert_equal(step_state, deserialized_step_state)
    end

    test "serialize_state/deserialize_state preserve loop state with step state serialization" do
      serialize_fn = fn step_state, opts -> :erlang.term_to_binary(step_state, opts) end
      deserialize_fn = fn binary, opts -> :erlang.binary_to_term(binary, opts) end

      init_fn = fn _data, _state -> %{foo: Nx.tensor(1)} end
      step_state = init_fn.(Nx.tensor(1), Axon.ModelState.empty())
      state = %State{step_state: step_state}

      serialized = Axon.Loop.serialize_state(state, serialize_step_state: serialize_fn)

      %State{step_state: deserialized_step_state} =
        Axon.Loop.deserialize_state(serialized, deserialize_step_state: deserialize_fn)

      assert_equal(step_state, deserialized_step_state)
    end
  end

  describe "checkpoint" do
    setup do
      File.rm_rf!("checkpoint")
      File.rm_rf!("checkpoint_custom_path")

      loop =
        Axon.input("input", shape: {nil, 1})
        |> Axon.dense(1)
        |> Loop.trainer(:binary_cross_entropy, :sgd, log: 0)

      [loop: loop]
    end

    test "saves a checkpoint on each epoch", %{loop: loop} do
      loop
      |> Loop.checkpoint()
      |> Loop.run([{Nx.tensor([[1]]), Nx.tensor([[2]])}], Axon.ModelState.empty(), epochs: 3)

      assert ["checkpoint_0_1.ckpt", "checkpoint_1_1.ckpt", "checkpoint_2_1.ckpt"] ==
               File.ls!("checkpoint") |> Enum.sort()
    end

    test "saves a checkpoint on custom events", %{loop: loop} do
      data = List.duplicate({Nx.iota({1, 1}), Nx.iota({1, 1})}, 5)

      assert %Axon.Loop.State{
               epoch: 3,
               iteration: 0,
               event_counts: %{iteration_completed: %{total: 15}}
             } =
               loop
               |> Loop.checkpoint(event: :iteration_completed, filter: [every: {:epoch, 2}])
               |> Loop.run(data, Axon.ModelState.empty(), epochs: 3)

      assert [
               "checkpoint_0_0.ckpt",
               "checkpoint_0_2.ckpt",
               "checkpoint_0_4.ckpt",
               "checkpoint_1_0.ckpt",
               "checkpoint_1_2.ckpt",
               "checkpoint_1_4.ckpt",
               "checkpoint_2_0.ckpt",
               "checkpoint_2_2.ckpt",
               "checkpoint_2_4.ckpt"
             ] ==
               File.ls!("checkpoint") |> Enum.sort()
    end

    test "uses the custom file_pattern function", %{loop: loop} do
      loop
      |> Loop.checkpoint(file_pattern: &"ckp_#{&1.epoch}.ckpt")
      |> Loop.run([{Nx.tensor([[1]]), Nx.tensor([[2]])}], Axon.ModelState.empty(), epochs: 3)

      assert ["ckp_0.ckpt", "ckp_1.ckpt", "ckp_2.ckpt"] ==
               File.ls!("checkpoint") |> Enum.sort()
    end

    test "uses a custom path", %{loop: loop} do
      loop
      |> Loop.checkpoint(path: "checkpoint_custom_path")
      |> Loop.run([{Nx.tensor([[1]]), Nx.tensor([[2]])}], Axon.ModelState.empty(), epochs: 3)

      assert ["checkpoint_0_1.ckpt", "checkpoint_1_1.ckpt", "checkpoint_2_1.ckpt"] ==
               File.ls!("checkpoint_custom_path") |> Enum.sort()
    end

    test "with :criteria, saves checkpoint after :patience is exceeded when metric stops improving",
         %{loop: loop} do
      loop
      |> Loop.handle_event(:epoch_completed, fn
        %{epoch: 0} = state ->
          state = %{state | metrics: Map.put(state.metrics, "loss", Nx.tensor(15))}
          {:continue, state}

        %{epoch: 1} = state ->
          # loss is improved (15 -> 10)
          # since_last_improvement = 0
          state = %{state | metrics: Map.put(state.metrics, "loss", Nx.tensor(10))}
          {:continue, state}

        %{epoch: 2} = state ->
          # loss is improved (10 -> 5)
          # since_last_improvement = 0
          state = %{state | metrics: Map.put(state.metrics, "loss", Nx.tensor(5))}
          {:continue, state}

        %{epoch: 3} = state ->
          # loss is NOT improved (5 -> 5)
          # since_last_improvement = 1
          state = %{state | metrics: Map.put(state.metrics, "loss", Nx.tensor(5))}
          {:continue, state}

        %{epoch: 4} = state ->
          # loss is NOT improved (5 -> 5)
          # since_last_improvement = 2
          state = %{state | metrics: Map.put(state.metrics, "loss", Nx.tensor(5))}
          {:continue, state}

        %{epoch: 5} = state ->
          # loss is NOT improved (5 -> 5)
          # since_last_improvement = 0 (goes back to 0 and waits for improvement)
          state = %{state | metrics: Map.put(state.metrics, "loss", Nx.tensor(5))}
          {:continue, state}
      end)
      |> Loop.checkpoint(criteria: "loss", mode: :min, patience: 2)
      |> Loop.run([{Nx.tensor([[0]]), Nx.tensor([[1]])}], Axon.ModelState.empty(), epochs: 6)

      # checkpoint_{epoch}_{iteration}.ckpt
      assert ["checkpoint_5_1.ckpt"] == File.ls!("checkpoint")
    end
  end

  describe "from_state" do
    test "resumes training from state" do
      model = Axon.input("input") |> Axon.dense(1)

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      ExUnit.CaptureIO.capture_io(fn ->
        state1 =
          model
          |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
          |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 3, iterations: 5)

        model
        |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
        |> Axon.Loop.from_state(state1)
        |> Axon.Loop.handle_event(:epoch_completed, fn %{epoch: epoch} = state ->
          assert epoch >= 3
          {:continue, state}
        end)
        |> Axon.Loop.run(data, epochs: 5, iterations: 5)
      end)
    end
  end

  describe "validate" do
    test "adds validation_* metrics to metrics map" do
      model = Axon.input("input") |> Axon.dense(1)

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      ExUnit.CaptureIO.capture_io(fn ->
        model
        |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, Enum.take(data, 5))
        |> Axon.Loop.handle_event(
          :epoch_completed,
          fn %{metrics: metrics} = state ->
            assert Map.has_key?(metrics, "validation_accuracy")
            {:continue, state}
          end,
          fn %{epoch: epoch}, _ -> epoch == 1 end
        )
        |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 5, iterations: 5)
      end)
    end
  end

  describe "buffer donation" do
    defp donation_model,
      do: Axon.input("input", shape: {nil, 4}) |> Axon.dense(8) |> Axon.dense(1)

    defp donation_data do
      for i <- 1..4 do
        xs = Nx.iota({2, 4}, type: :f32) |> Nx.add(i)
        {xs, Nx.sum(xs, axes: [1], keep_axes: true)}
      end
    end

    defp donation_init_state(model, data) do
      {init_fn, _} = Axon.build(model)
      [{xs, _} | _] = data
      Nx.Defn.jit_apply(init_fn, [xs, Axon.ModelState.empty()])
    end

    # Runs every tensor in `container` through `fun`, returning the results.
    defp map_leaves(container, fun) do
      {_, acc} =
        Nx.Defn.Composite.traverse(container, [], fn tensor, acc ->
          {tensor, [fun.(tensor) | acc]}
        end)

      acc
    end

    test "the state a donating loop returns is not itself donatable" do
      model = donation_model()
      data = donation_data()

      ExUnit.CaptureIO.capture_io(fn ->
        result =
          model
          |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: 0.01))
          |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 2, donate_state?: true)

        send(self(), {:result, result})
      end)

      assert_receive {:result, result}

      # Otherwise the mark would follow the caller out of the loop and donate
      # buffers they never offered on their next `Nx.Defn.jit` call.
      assert map_leaves(result.step_state, &Nx.donatable?/1) |> Enum.uniq() == [false]
    end

    test "donating state does not change the parameters a loop converges to" do
      model = donation_model()
      data = donation_data()
      init_state = donation_init_state(model, data)

      run = fn donate? ->
        ExUnit.CaptureIO.capture_io(fn ->
          %Axon.Loop.State{step_state: %{model_state: model_state}} =
            model
            |> Axon.Loop.trainer(
              :mean_squared_error,
              Polaris.Optimizers.adam(learning_rate: 0.01)
            )
            |> Axon.Loop.run(data, Nx.backend_copy(init_state),
              epochs: 3,
              donate_state?: donate?
            )

          send(self(), {:result, model_state})
        end)

        assert_receive {:result, result}
        result
      end

      assert_equal(run.(true), run.(false))
    end

    test "donating state works when the loop is not JIT compiled" do
      model = donation_model()
      data = donation_data()

      ExUnit.CaptureIO.capture_io(fn ->
        assert %Axon.Loop.State{step_state: %{model_state: %Axon.ModelState{}}} =
                 model
                 |> Axon.Loop.trainer(:mean_squared_error, :sgd)
                 |> Axon.Loop.run(data, Axon.ModelState.empty(),
                   epochs: 2,
                   donate_state?: true,
                   jit_compile?: false
                 )
      end)
    end

    test "donating state works with frozen parameters and stateful layers" do
      model =
        Axon.input("input", shape: {nil, 4})
        |> Axon.dense(8)
        |> Axon.batch_norm()
        |> Axon.dense(1)

      data = donation_data()

      frozen_state =
        model
        |> donation_init_state(data)
        |> Axon.ModelState.freeze(fn [layer | _] -> layer == "dense_0" end)

      run = fn donate? ->
        ExUnit.CaptureIO.capture_io(fn ->
          %Axon.Loop.State{step_state: %{model_state: model_state}} =
            model
            |> Axon.Loop.trainer(:mean_squared_error, :sgd)
            |> Axon.Loop.run(data, Nx.backend_copy(frozen_state),
              epochs: 2,
              donate_state?: donate?
            )

          send(self(), {:result, model_state})
        end)

        assert_receive {:result, result}
        result
      end

      # Frozen parameters are read but never updated, so they can only be
      # donated to an output which is the argument itself, and the running
      # statistics of a stateful layer are updated outside the optimizer.
      assert_equal(run.(true), run.(false))
    end

    test "donating state does not affect loops run from event handlers" do
      model = donation_model()
      data = donation_data()

      ExUnit.CaptureIO.capture_io(fn ->
        model
        |> Axon.Loop.trainer(:mean_squared_error, :sgd)
        |> Axon.Loop.metric(:mean_absolute_error)
        |> Axon.Loop.validate(model, data)
        |> Axon.Loop.handle_event(:epoch_completed, fn %{metrics: metrics} = state ->
          assert Map.has_key?(metrics, "validation_mean_absolute_error")
          {:continue, state}
        end)
        |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 2, donate_state?: true)
      end)
    end

    @tag :exla_only
    test "donated model state and optimizer state buffers are consumed by the next step" do
      model = donation_model()
      data = donation_data()

      # Capture the state produced by the first iteration, then check whether
      # the buffers behind it survive the iterations that follow.
      {:ok, captured} = Agent.start_link(fn -> nil end)

      capture = fn %State{iteration: iteration, step_state: step_state} = state ->
        if iteration == 0 do
          Agent.update(captured, fn nil ->
            Map.take(step_state, [:model_state, :optimizer_state])
          end)
        end

        {:continue, state}
      end

      run = fn donate? ->
        Agent.update(captured, fn _ -> nil end)

        ExUnit.CaptureIO.capture_io(fn ->
          model
          |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: 0.01))
          |> Axon.Loop.handle_event(:iteration_completed, capture)
          |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 1, donate_state?: donate?)
        end)

        captured
        |> Agent.get(& &1)
        |> map_leaves(&Nx.backend_deallocate/1)
        |> Enum.uniq()
      end

      assert run.(true) == [:already_deallocated]
      assert run.(false) == [:ok]
    end

    @tag :exla_only
    test "donating state does not consume the state the loop was given" do
      model = donation_model()
      data = donation_data()
      init_state = donation_init_state(model, data)

      ExUnit.CaptureIO.capture_io(fn ->
        model
        |> Axon.Loop.trainer(:mean_squared_error, :sgd)
        |> Axon.Loop.run(data, init_state, epochs: 1, donate_state?: true)
      end)

      # `init_fn` copies the state it is given into the step state, so the
      # tensors the caller passed in are not the ones being donated.
      assert map_leaves(init_state, &Nx.backend_deallocate/1) |> Enum.uniq() == [:ok]
    end
  end

  describe "early_stop" do
    test "adds correct handler metadata to loop state" do
      model = Axon.input("input") |> Axon.dense(1)

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      ExUnit.CaptureIO.capture_io(fn ->
        model
        |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, Enum.take(data, 5))
        |> Axon.Loop.early_stop("validation_accuracy", mode: :max)
        |> Axon.Loop.handle_event(
          :epoch_completed,
          fn %{handler_metadata: meta} = state ->
            assert %{early_stop: %{"validation_accuracy" => _, :since_last_improvement => _}} =
                     meta

            {:continue, state}
          end,
          fn %{epoch: epoch}, _ -> epoch == 1 end
        )
        |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 5, iterations: 5)
      end)
    end

    test "early stops if metric does not improve with mode max" do
      model = Axon.input("input") |> Axon.dense(1)

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      # metric is a counter when combined with running_sum,
      # so in mode min we can always guarantee this increases
      my_metric = fn _, _ -> 1 end

      ExUnit.CaptureIO.capture_io(fn ->
        state =
          model
          |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
          |> Axon.Loop.metric(my_metric, "counter", :running_sum)
          |> Axon.Loop.early_stop("counter", mode: :min, patience: 2)
          # TODO: This API needs to change
          |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 5, iterations: 5)

        assert %{epoch: 3} = state
      end)
    end

    test "early stops if metric does not improve with mode min" do
      model = Axon.input("input") |> Axon.dense(1)

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      # metric is a counter when combined with running_sum,
      # so in mode min we can always guarantee this increases
      my_metric = fn _, _ -> -1 end

      ExUnit.CaptureIO.capture_io(fn ->
        state =
          model
          |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
          |> Axon.Loop.metric(my_metric, "counter", :running_sum)
          |> Axon.Loop.early_stop("counter", mode: :max, patience: 2)
          # TODO: This API needs to change
          |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 5, iterations: 5)

        assert %{epoch: 3} = state
      end)
    end
  end

  describe "reduce_lr_on_plateau" do
    test "adds correct handler metadata to loop state" do
      model = Axon.input("input") |> Axon.dense(1)

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      ExUnit.CaptureIO.capture_io(fn ->
        model
        |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, Enum.take(data, 5))
        |> Axon.Loop.reduce_lr_on_plateau("validation_accuracy", mode: :max)
        |> Axon.Loop.handle_event(
          :epoch_completed,
          fn %{handler_metadata: meta} = state ->
            assert %{reduce_lr: %{"validation_accuracy" => _, :since_last_improvement => _}} =
                     meta

            {:continue, state}
          end,
          fn %{epoch: epoch}, _ -> epoch == 1 end
        )
        |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 5, iterations: 5)
      end)
    end

    test "reduces stateful learning rate by factor with mode min" do
      initial_lr = 10.0

      model = Axon.input("input") |> Axon.dense(1)

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      # metric is a counter when combined with running_sum,
      # so in mode min we can always guarantee this increases
      my_metric = fn _, _ -> 1 end

      ExUnit.CaptureIO.capture_io(fn ->
        state =
          model
          |> Axon.Loop.trainer(
            :binary_cross_entropy,
            Polaris.Optimizers.sgd(learning_rate: initial_lr)
          )
          |> Axon.Loop.metric(my_metric, "counter", :running_sum)
          |> Axon.Loop.reduce_lr_on_plateau("counter", factor: 0.5, mode: :min, patience: 2)
          # TODO: This API needs to change
          |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 7, iterations: 5)

        assert %{step_state: %{optimizer_state: optimizer_state}} = state
        # TODO: This is a strong assumption
        assert %{scale: lr} = elem(optimizer_state, 0)
        assert_all_close(lr, Nx.tensor(-2.5))
      end)
    end

    test "reduces stateful learning rate by factor with mode max" do
      initial_lr = 10.0

      model = Axon.input("input") |> Axon.dense(1)

      data =
        Stream.repeatedly(fn ->
          xs = Nx.tensor([[Enum.random(0..10)]])
          ys = Nx.greater(xs, 5)
          {xs, ys}
        end)

      # metric is a counter when combined with running_sum,
      # so in mode min we can always guarantee this increases
      my_metric = fn _, _ -> -1 end

      ExUnit.CaptureIO.capture_io(fn ->
        state =
          model
          |> Axon.Loop.trainer(
            :binary_cross_entropy,
            Polaris.Optimizers.sgd(learning_rate: initial_lr)
          )
          |> Axon.Loop.metric(my_metric, "counter", :running_sum)
          |> Axon.Loop.reduce_lr_on_plateau("counter", factor: 0.5, mode: :max, patience: 2)
          # TODO: This API needs to change
          |> Axon.Loop.run(data, Axon.ModelState.empty(), epochs: 7, iterations: 5)

        assert %{step_state: %{optimizer_state: optimizer_state}} = state
        # TODO: This is a strong assumption
        assert %{scale: lr} = elem(optimizer_state, 0)
        assert_all_close(lr, Nx.tensor(-2.5))
      end)
    end
  end
end
