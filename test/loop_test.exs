defmodule Axon.LoopTest do
  use ExUnit.Case, async: true

  alias Axon.Loop
  alias Axon.Loop.Process
  alias Axon.Loop.State

  describe "factories" do
    test "loop/3 creates a basic loop with defaults" do
      step_fn = fn _, _ -> 1 end

      assert %Loop{process: process, output_transform: transform} = Loop.loop(step_fn)
      assert %Process{init: init_fn, update: update_fn} = process

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
          assert %Loop{process: process, output_transform: transform} =
                   Loop.trainer(model, loss, optimizer)

          assert %Process{init: init_fn, update: update_fn} = process

          assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

          state = %State{process_state: pstate}

          assert %{model_state: %{}, y_true: tar, y_pred: pred} =
                   Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[1]])}, state])

          assert tar == Nx.tensor([[1]])
          assert pred == Nx.tensor([[1]])

          assert transform.(state) == %{}
        end
      end
    end

    test "trainer/3 returns a supervised training loop with custom loss" do
      model = Axon.input({nil, 1})
      custom_loss_fn = fn _, _ -> Nx.tensor(5.0, backend: Nx.Defn.Expr) end

      assert %Loop{process: process, output_transform: transform} =
               Loop.trainer(model, custom_loss_fn, :adam)

      assert %Process{init: init_fn, update: update_fn} = process

      assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{process_state: pstate}

      assert %{model_state: %{}, y_true: tar, y_pred: pred, loss: loss} =
               Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[1]])}, state])

      assert tar == Nx.tensor([[1]])
      assert pred == Nx.tensor([[1]])
      assert loss == Nx.tensor(5.0)

      assert transform.(state) == %{}
    end

    test "trainer/3 returns a supervised training loop with custom optimizer" do
      model = Axon.input({nil, 1})
      optimizer = Axon.Optimizers.rmsprop(1.0e-3)

      assert %Loop{process: process, output_transform: transform} =
               Loop.trainer(model, :mean_squared_error, optimizer)

      assert %Process{init: init_fn, update: update_fn} = process

      assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{process_state: pstate}

      assert %{model_state: %{}, y_true: tar, y_pred: pred} =
               Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[1]])}, state])

      assert tar == Nx.tensor([[1]])
      assert pred == Nx.tensor([[1]])

      assert transform.(state) == %{}
    end

    test "trainer/3 returns a supervised training loop with custom model" do
      model = Axon.input({nil, 1}) |> Axon.compile()

      assert %Loop{process: process, output_transform: transform} =
               Loop.trainer(model, :mean_squared_error, :adam)

      assert %Process{init: init_fn, update: update_fn} = process

      assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{process_state: pstate}

      assert %{model_state: %{}, y_true: tar, y_pred: pred} =
               Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[1]])}, state])

      assert tar == Nx.tensor([[1]])
      assert pred == Nx.tensor([[1]])

      assert transform.(state) == %{}
    end

    test "trainer/3 returns a supervised training loop with multi-loss" do
      model = {Axon.input({nil, 1}), Axon.input({nil, 1})}

      assert %Loop{process: process, output_transform: transform} =
               Loop.trainer(model, [mean_squared_error: 0.5, mean_absolute_error: 0.5], :adam)

      assert %Process{init: init_fn, update: update_fn} = process

      assert %{model_state: %{}} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{process_state: pstate}

      assert %{model_state: %{}, y_true: tar, y_pred: pred, loss: loss} =
               Nx.Defn.jit(update_fn, [
                 {{Nx.tensor([[1]]), Nx.tensor([[1]])}, {Nx.tensor([[2]]), Nx.tensor([[2]])}},
                 state
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

      assert %Loop{process: process, output_transform: transform} =
               Loop.evaluator(model, model_state)

      assert %Process{init: init_fn, update: update_fn} = process

      assert %{y_true: _, y_pred: _} = pstate = Nx.Defn.jit(init_fn, [])

      state = %State{process_state: pstate, metrics: %{"my_metric" => {}}}

      assert %{y_true: tar, y_pred: pred} =
               Nx.Defn.jit(update_fn, [{Nx.tensor([[1]]), Nx.tensor([[2]])}, state])

      assert tar == Nx.tensor([[2]])
      assert pred == Nx.tensor([[1]])

      assert transform.(state) == %{"my_metric" => {}}
    end
  end

  describe "looping" do
    test "returns initial state with epochs 0" do
      step_fn = fn _, _ -> 1 end

      state =
        step_fn
        |> Loop.loop()
        |> Loop.run([], epochs: 0)

      assert %State{epoch: 0, iteration: 0, times: %{}, metrics: %{}, process_state: pstate} =
               state

      assert pstate == %{}
    end
  end
end
