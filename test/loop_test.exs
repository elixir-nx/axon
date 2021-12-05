defmodule Axon.LoopTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias Axon.Loop
  alias Axon.Loop.State

  describe "factories" do
    test "loop/3 creates a basic loop with defaults" do
      step_fn = fn _, _ -> 1 end

      assert %Loop{init: init_fn, step: update_fn, output_transform: transform} =
               Loop.loop(step_fn)

      assert init_fn.() == %{}
      assert update_fn.({}, %{}) == 1
      assert transform.(%{}) == %{}
    end

    test "trainer/3 returns a supervised training loop with basic case" do
      model = Axon.input({nil, 1})

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
        Axon.Optimizers.__info__(:functions)
        |> Enum.map(fn {k, _} -> k end)
        |> Enum.uniq()

      for loss <- valid_axon_losses do
        for optimizer <- valid_axon_optimizers do
          assert %Loop{init: init_fn, step: update_fn, output_transform: transform} =
                   Loop.trainer(model, loss, optimizer)

          assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

          state = %State{step_state: pstate}

          assert %{model_state: %{}, y_true: tar, y_pred: pred} =
                   Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[1]])}, pstate])

          assert tar == Nx.tensor([[1]])
          assert pred == Nx.tensor([[1]])

          assert transform.(state) == %{}
        end
      end
    end

    test "trainer/3 returns a supervised training loop with custom loss" do
      model = Axon.input({nil, 1})
      custom_loss_fn = fn _, _ -> Nx.tensor(5.0, backend: Nx.Defn.Expr) end

      assert %Loop{init: init_fn, step: update_fn, output_transform: transform} =
               Loop.trainer(model, custom_loss_fn, :adam)

      assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{step_state: pstate}

      assert %{model_state: %{}, y_true: tar, y_pred: pred, loss: loss} =
               Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[1]])}, pstate])

      assert tar == Nx.tensor([[1]])
      assert pred == Nx.tensor([[1]])
      assert loss == Nx.tensor(5.0)

      assert transform.(state) == %{}
    end

    test "trainer/3 returns a supervised training loop with custom optimizer" do
      model = Axon.input({nil, 1})
      optimizer = Axon.Optimizers.rmsprop(1.0e-3)

      assert %Loop{init: init_fn, step: update_fn, output_transform: transform} =
               Loop.trainer(model, :mean_squared_error, optimizer)

      assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{step_state: pstate}

      assert %{model_state: %{}, y_true: tar, y_pred: pred} =
               Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[1]])}, pstate])

      assert tar == Nx.tensor([[1]])
      assert pred == Nx.tensor([[1]])

      assert transform.(state) == %{}
    end

    test "trainer/3 returns a supervised training loop with custom model" do
      model = Axon.input({nil, 1}) |> Axon.compile()

      assert %Loop{init: init_fn, step: update_fn, output_transform: transform} =
               Loop.trainer(model, :mean_squared_error, :adam)

      assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{step_state: pstate}

      assert %{model_state: %{}, y_true: tar, y_pred: pred} =
               Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[1]])}, pstate])

      assert tar == Nx.tensor([[1]])
      assert pred == Nx.tensor([[1]])

      assert transform.(state) == %{}
    end

    test "trainer/3 returns a supervised training loop with multi-loss" do
      model = {Axon.input({nil, 1}), Axon.input({nil, 1})}

      assert %Loop{init: init_fn, step: update_fn, output_transform: transform} =
               Loop.trainer(model, [mean_squared_error: 0.5, mean_absolute_error: 0.5], :adam)

      assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{step_state: pstate}

      assert %{model_state: %{}, y_true: tar, y_pred: pred, loss: loss} =
               Nx.Defn.jit(update_fn, [
                 {{Nx.tensor([[1]]), Nx.tensor([[1]])}, {Nx.tensor([[2]]), Nx.tensor([[2]])}},
                 pstate
               ])

      assert tar == {Nx.tensor([[2]]), Nx.tensor([[2]])}
      assert pred == {Nx.tensor([[1]]), Nx.tensor([[1]])}
      assert loss == Nx.tensor(1.0)

      assert transform.(state) == %{}
    end

    test "trainer/3 raises on bad inputs" do
      assert_raise ArgumentError, ~r/Invalid/, fn ->
        Axon.Loop.trainer(:foo, :mean_squared_error, :adam)
      end

      assert_raise ArgumentError, ~r/Invalid/, fn ->
        Axon.Loop.trainer(Axon.input({nil, 1}), :foo, :adam)
      end

      assert_raise ArgumentError, ~r/Invalid/, fn ->
        Axon.Loop.trainer(Axon.input({nil, 1}), :mean_squared_error, :foo)
      end
    end

    test "evaluator/3 returns a supervised evaluator loop" do
      model = Axon.input({nil, 1})
      model_state = %{}

      assert %Loop{init: init_fn, step: update_fn, output_transform: transform} =
               Loop.evaluator(model, model_state)

      assert %{y_true: _, y_pred: _} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{step_state: pstate, metrics: %{"my_metric" => {}}}

      assert %{y_true: tar, y_pred: pred} =
               Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[2]])}, pstate])

      assert tar == Nx.tensor([[2]])
      assert pred == Nx.tensor([[1]])

      assert transform.(state) == %{"my_metric" => {}}
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

    test "computes running average by default with supervised output transform" do
      step_fn = fn _, _ -> 1 end

      loop =
        step_fn
        |> Loop.loop()
        |> Loop.metric(:accuracy)

      assert %Loop{metrics: %{"accuracy" => avg_acc_fun}} = loop

      output = %{foo: 1, y_true: Nx.tensor([1, 0, 1]), y_pred: Nx.tensor([0.8, 0.2, 0.8])}
      cur_avg_acc = 0.5
      i = 1

      assert avg_acc_fun.(cur_avg_acc, List.wrap(output), i) == Nx.tensor(0.75)
    end

    test "computes a running sum with custom output transform" do
      step_fn = fn _, _ -> 1 end

      loop =
        step_fn
        |> Loop.loop()
        |> Loop.metric(:true_positives, "tp", :running_sum, &Tuple.to_list/1)

      assert %Loop{metrics: %{"tp" => sum_tp_fun}} = loop

      output = {Nx.tensor([1, 0, 1]), Nx.tensor([0, 1, 1])}
      cur_sum = 25
      i = 10

      assert sum_tp_fun.(cur_sum, List.wrap(output), i) == Nx.tensor(26)
    end
  end

  describe "looping" do
    test "returns initial state with epochs 0" do
      step_fn = fn _, _ -> 1 end

      state =
        step_fn
        |> Loop.loop()
        |> Loop.run([], epochs: 0)

      assert %State{epoch: 0, iteration: 0, times: %{}, metrics: %{}, step_state: pstate} = state

      assert pstate == %{}
    end

    test "propagates user-defined numerical data inside step_state" do
      Axon.input({nil, 1})
      |> Axon.dense(1)
      |> Loop.trainer(:binary_cross_entropy, :sgd)
      |> Loop.handle(
        :epoch_completed,
        fn %State{step_state: pstate} = state ->
          {
            :continue,
            %State{
              state
              | step_state:
                  case pstate[:counter] do
                    nil -> Map.put(pstate, :counter, 0)
                    counter -> %{pstate | counter: Nx.to_scalar(counter) + 1}
                  end
            }
          }
        end
      )
      |> Loop.handle(
        :completed,
        fn %State{step_state: %{counter: counter}} = state ->
          assert 4 = counter

          {:continue, state}
        end
      )
      |> Loop.run(
        [{Nx.tensor([[1.0]]), Nx.tensor([[1.0]])}],
        epochs: 5
      )
    end

    test "propagates user-defined numerical data inside step_state when it is nested into a tuple" do
      Axon.input({nil, 1})
      |> Axon.dense(1)
      |> Loop.trainer(:binary_cross_entropy, :sgd)
      |> Loop.handle(
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
                      next_counter_value = Nx.to_scalar(counter) + 1
                      %{pstate | counter: {{next_counter_value}, next_counter_value}}
                  end
            }
          }
        end
      )
      |> Loop.handle(
        :completed,
        fn %State{step_state: %{counter: counter}} = state ->
          assert {{4}, 4} = counter

          {:continue, state}
        end
      )
      |> Loop.run(
        [{Nx.tensor([[1.0]]), Nx.tensor([[1.0]])}],
        epochs: 5
      )
    end
  end
end
