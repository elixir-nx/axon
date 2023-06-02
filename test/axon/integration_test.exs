defmodule Axon.IntegrationTest do
  use Axon.Case, async: true

  import AxonTestUtil

  @moduletag :integration

  test "vector classification test" do
    {train, _test} = get_test_data(100, 0, 10, {10}, 2, 1337)

    train =
      train
      |> Stream.map(fn {xs, ys} ->
        {xs, one_hot(ys, num_classes: 2)}
      end)
      |> Enum.to_list()

    [{x_test, _}] = Enum.take(train, 1)

    model =
      Axon.input("input")
      |> Axon.dense(16)
      |> Axon.dropout(rate: 0.1)
      |> Axon.dense(2, activation: :softmax)

    ExUnit.CaptureIO.capture_io(fn ->
      results =
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(5.0e-3))
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, %{}, epochs: 10)

      assert %{step_state: %{model_state: model_state}, metrics: %{9 => last_epoch_metrics}} =
               results

      eval_results =
        model
        |> Axon.Loop.evaluator()
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.run(train, model_state)

      assert %{0 => %{"accuracy" => final_model_val_accuracy}} = eval_results

      assert_greater_equal(last_epoch_metrics["validation_accuracy"], 0.7)
      assert_all_close(final_model_val_accuracy, last_epoch_metrics["validation_accuracy"])
      assert Nx.shape(Axon.predict(model, model_state, x_test)) == {10, 2}
    end)
  end

  test "image classification test" do
    {train, _test} = get_test_data(100, 0, 10, {10, 10, 3}, 2, 1337)

    train =
      train
      |> Stream.map(fn {xs, ys} ->
        {xs, one_hot(ys, num_classes: 2)}
      end)
      |> Enum.to_list()

    [{x_test, _}] = Enum.take(train, 1)

    model =
      Axon.input("input")
      |> Axon.conv(4, kernel_size: 3, padding: :same, activation: :relu)
      |> Axon.conv(8, kernel_size: 3, padding: :same)
      |> Axon.batch_norm()
      |> Axon.conv(8, kernel_size: 3, padding: :same)
      |> Axon.flatten()
      |> Axon.dense(2, activation: :softmax)

    ExUnit.CaptureIO.capture_io(fn ->
      results =
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(5.0e-3))
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, %{}, epochs: 10)

      assert %{step_state: %{model_state: model_state}, metrics: %{9 => last_epoch_metrics}} =
               results

      eval_results =
        model
        |> Axon.Loop.evaluator()
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.run(train, model_state)

      assert %{0 => %{"accuracy" => final_model_val_accuracy}} = eval_results

      assert_greater_equal(last_epoch_metrics["validation_accuracy"], 0.7)
      assert_all_close(final_model_val_accuracy, last_epoch_metrics["validation_accuracy"])
      assert Nx.shape(Axon.predict(model, model_state, x_test)) == {10, 2}
    end)
  end

  test "time series classification test" do
    {train, _test} = get_test_data(100, 0, 10, {4, 10}, 2, 1337)

    train =
      train
      |> Stream.map(fn {xs, ys} ->
        {xs, one_hot(ys, num_classes: 2)}
      end)
      |> Enum.to_list()

    [{x_test, _}] = Enum.take(train, 1)

    model =
      Axon.input("input")
      |> Axon.lstm(5)
      |> elem(0)
      |> Axon.gru(2, activation: :softmax)
      |> elem(0)
      |> Axon.nx(fn seq -> Nx.squeeze(seq[[0..-1//1, -1, 0..-1//1]]) end)

    ExUnit.CaptureIO.capture_io(fn ->
      results =
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(5.0e-3))
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, %{}, epochs: 10)

      assert %{step_state: %{model_state: model_state}, metrics: %{9 => last_epoch_metrics}} =
               results

      eval_results =
        model
        |> Axon.Loop.evaluator()
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.run(train, model_state)

      assert %{0 => %{"accuracy" => final_model_val_accuracy}} = eval_results

      assert_greater_equal(last_epoch_metrics["validation_accuracy"], 0.7)
      assert_all_close(final_model_val_accuracy, last_epoch_metrics["validation_accuracy"])
      assert Nx.shape(Axon.predict(model, model_state, x_test)) == {10, 2}
    end)
  end

  test "gradient accumulation test" do
    {train, _test} = get_test_data(100, 0, 10, {10}, 2, 1337)

    train =
      train
      |> Stream.map(fn {xs, ys} ->
        {xs, one_hot(ys, num_classes: 2)}
      end)
      |> Enum.to_list()

    [{x_test, _}] = Enum.take(train, 1)

    model =
      Axon.input("input")
      |> Axon.dense(16)
      |> Axon.dropout(rate: 0.1)
      |> Axon.dense(2, activation: :softmax)

    ExUnit.CaptureIO.capture_io(fn ->
      results =
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(5.0e-3),
          gradient_accumulation_steps: 3
        )
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, %{}, epochs: 10)

      assert %{step_state: %{model_state: model_state}, metrics: %{9 => last_epoch_metrics}} =
               results

      eval_results =
        model
        |> Axon.Loop.evaluator()
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.run(train, model_state)

      assert %{0 => %{"accuracy" => final_model_val_accuracy}} = eval_results

      assert_greater_equal(last_epoch_metrics["validation_accuracy"], 0.7)
      assert_all_close(final_model_val_accuracy, last_epoch_metrics["validation_accuracy"])
      assert Nx.shape(Axon.predict(model, model_state, x_test)) == {10, 2}
    end)
  end

  test "deterministic training test" do
    {train, _test} = get_test_data(100, 0, 10, {10}, 2, 1337)

    train =
      train
      |> Stream.map(fn {xs, ys} ->
        {xs, one_hot(ys, num_classes: 2)}
      end)
      |> Enum.to_list()

    model =
      Axon.input("input")
      |> Axon.dense(16)
      |> Axon.dropout(rate: 0.1)
      |> Axon.dense(2, activation: :softmax)

    ExUnit.CaptureIO.capture_io(fn ->
      %{metrics: metrics1, step_state: step_state1} =
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(5.0e-3), seed: 1)
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, %{}, epochs: 10)

      %{metrics: metrics2, step_state: step_state2} =
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(5.0e-3), seed: 1)
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, %{}, epochs: 10)

      assert_equal(metrics1, metrics2)
      assert_equal(step_state1, step_state2)
    end)
  end

  describe "optimizer integration" do
    @optimizers_and_args [
      {:adabelief, [5.0e-3, []]},
      {:adagrad, [5.0e-3, []]},
      {:adam, [5.0e-3, []]},
      {:adamw, [5.0e-3, []]},
      {:adamw, [5.0e-3, [decay: 0.9]]},
      {:lamb, [5.0e-3, []]},
      {:lamb, [5.0e-3, [decay: 0.9]]},
      {:lamb, [5.0e-3, [min_norm: 0.1]]},
      {:lamb, [5.0e-3, [decay: 0.9, min_norm: 0.1]]},
      {:noisy_sgd, [5.0e-3, []]},
      {:radam, [5.0e-3, []]},
      {:rmsprop, [5.0e-3, []]},
      {:rmsprop, [5.0e-3, [centered: true]]},
      {:rmsprop, [5.0e-3, [momentum: 0.9]]},
      {:rmsprop, [5.0e-3, [nesterov: true, momentum: 0.9]]},
      {:rmsprop, [5.0e-3, [centered: true, nesterov: true, momentum: 0.9]]},
      {:sgd, [5.0e-3, []]},
      {:sgd, [5.0e-3, [momentum: 0.9]]},
      {:sgd, [5.0e-3, [momentum: 0.9, nesterov: true]]}
    ]

    for {optimizer, [lr, opts] = args} <- @optimizers_and_args do
      test "#{optimizer}, learning_rate: #{lr}, opts: #{inspect(opts)} trains simple model with dropout" do
        {train, _test} = get_test_data(100, 0, 10, {10}, 2, 1337)

        train =
          train
          |> Stream.map(fn {xs, ys} ->
            {xs, one_hot(ys, num_classes: 2)}
          end)
          |> Enum.to_list()

        [{x_test, _}] = Enum.take(train, 1)

        model =
          Axon.input("input")
          |> Axon.dense(16)
          |> Axon.dropout(rate: 0.1)
          |> Axon.dense(2, activation: :softmax)

        ExUnit.CaptureIO.capture_io(fn ->
          results =
            model
            |> Axon.Loop.trainer(
              :categorical_cross_entropy,
              apply(Axon.Optimizers, unquote(optimizer), unquote(args))
            )
            # TODO: Fix default output transform
            |> Map.update(:output_transform, nil, fn _ -> & &1 end)
            |> Axon.Loop.metric(:accuracy)
            |> Axon.Loop.validate(model, train)
            |> Axon.Loop.run(train, %{}, epochs: 10)

          assert %{step_state: %{model_state: model_state}, metrics: %{9 => last_epoch_metrics}} =
                   results

          eval_results =
            model
            |> Axon.Loop.evaluator()
            |> Axon.Loop.metric(:accuracy)
            |> Axon.Loop.run(train, model_state)

          assert %{0 => %{"accuracy" => final_model_val_accuracy}} = eval_results

          assert_greater_equal(last_epoch_metrics["validation_accuracy"], 0.7)
          assert_all_close(final_model_val_accuracy, last_epoch_metrics["validation_accuracy"])
          assert Nx.shape(Axon.predict(model, model_state, x_test)) == {10, 2}
        end)
      end
    end
  end

  describe "mixed precision training integration" do
    @policies [
      {"compute f16", Axon.MixedPrecision.create_policy(compute: {:f, 16})},
      {"compute f16, params f16",
       Axon.MixedPrecision.create_policy(compute: {:f, 16}, params: {:f, 16})},
      {"compute f16, params f16, output f16",
       Axon.MixedPrecision.create_policy(params: {:f, 16}, compute: {:f, 16}, output: {:f, 16})}
    ]

    @scales [:identity, :dynamic, :static]

    for {name, policy} <- @policies, scale <- @scales do
      test "trains simple model with policy #{name}, scale #{inspect(scale)}" do
        {train, _test} = get_test_data(100, 0, 10, {10}, 2, 1337)

        train =
          train
          |> Stream.map(fn {xs, ys} ->
            {xs, one_hot(ys, num_classes: 2)}
          end)
          |> Enum.to_list()

        [{x_test, _}] = Enum.take(train, 1)

        model =
          Axon.input("input")
          |> Axon.dense(16)
          |> Axon.dropout(rate: 0.1)
          |> Axon.dense(2, activation: :softmax)
          |> Axon.MixedPrecision.apply_policy(unquote(Macro.escape(policy)))

        ExUnit.CaptureIO.capture_io(fn ->
          results =
            model
            |> Axon.Loop.trainer(:categorical_cross_entropy, :adam, loss_scale: unquote(scale))
            # TODO: Fix default output transform
            |> Map.update(:output_transform, nil, fn _ -> & &1 end)
            |> Axon.Loop.metric(:accuracy)
            |> Axon.Loop.validate(model, train)
            |> Axon.Loop.run(train, %{}, epochs: 10)

          assert %{step_state: %{model_state: model_state}, metrics: %{9 => last_epoch_metrics}} =
                   results

          eval_results =
            model
            |> Axon.Loop.evaluator()
            |> Axon.Loop.metric(:accuracy)
            |> Axon.Loop.run(train, model_state)

          assert %{0 => %{"accuracy" => final_model_val_accuracy}} = eval_results

          assert_greater_equal(last_epoch_metrics["validation_accuracy"], 0.60)
          assert_all_close(final_model_val_accuracy, last_epoch_metrics["validation_accuracy"])
          assert Nx.shape(Axon.predict(model, model_state, x_test)) == {10, 2}
          assert Nx.type(model_state["dense_0"]["kernel"]) == unquote(Macro.escape(policy)).params
        end)
      end

      test "trains model with batch norm with policy #{name}, scale #{inspect(scale)}" do
        {train, _test} = get_test_data(100, 0, 10, {10}, 2, 1337)

        train =
          train
          |> Stream.map(fn {xs, ys} ->
            {xs, one_hot(ys, num_classes: 2)}
          end)
          |> Enum.to_list()

        [{x_test, _}] = Enum.take(train, 1)

        model =
          Axon.input("input")
          |> Axon.dense(16)
          |> Axon.batch_norm()
          |> Axon.dropout(rate: 0.1)
          |> Axon.dense(2, activation: :softmax)
          |> Axon.MixedPrecision.apply_policy(unquote(Macro.escape(policy)), except: [:batch_norm])

        ExUnit.CaptureIO.capture_io(fn ->
          results =
            model
            |> Axon.Loop.trainer(:categorical_cross_entropy, :adam, loss_scale: unquote(scale))
            # TODO: Fix default output transform
            |> Map.update(:output_transform, nil, fn _ -> & &1 end)
            |> Axon.Loop.metric(:accuracy)
            |> Axon.Loop.validate(model, train)
            |> Axon.Loop.run(train, %{}, epochs: 10)

          assert %{step_state: %{model_state: model_state}, metrics: %{9 => last_epoch_metrics}} =
                   results

          eval_results =
            model
            |> Axon.Loop.evaluator()
            |> Axon.Loop.metric(:accuracy)
            |> Axon.Loop.run(train, model_state)

          assert %{0 => %{"accuracy" => final_model_val_accuracy}} = eval_results

          assert_greater_equal(last_epoch_metrics["validation_accuracy"], 0.60)
          assert_all_close(final_model_val_accuracy, last_epoch_metrics["validation_accuracy"])
          assert Nx.shape(Axon.predict(model, model_state, x_test)) == {10, 2}
          assert Nx.type(model_state["dense_0"]["kernel"]) == unquote(Macro.escape(policy)).params
        end)
      end
    end

    test "mixed precision downcasts model when state is given to train" do
      policy =
        Axon.MixedPrecision.create_policy(
          params: {:f, 16},
          compute: {:f, 16},
          output: {:f, 16}
        )

      {train, _test} = get_test_data(100, 0, 10, {10}, 2, 1337)

      train =
        train
        |> Stream.map(fn {xs, ys} ->
          {xs, one_hot(ys, num_classes: 2)}
        end)
        |> Enum.to_list()

      [{x_test, _}] = Enum.take(train, 1)

      model =
        Axon.input("input")
        |> Axon.dense(16)
        |> Axon.dropout(rate: 0.1)
        |> Axon.dense(2, activation: :softmax)

      {init_fn, _} = Axon.build(model)
      initial_state = init_fn.(Nx.template({1, 10}, :f32), %{})

      mp_model = Axon.MixedPrecision.apply_policy(model, policy)

      ExUnit.CaptureIO.capture_io(fn ->
        results =
          mp_model
          |> Axon.Loop.trainer(:categorical_cross_entropy, :adam, loss_scale: :dynamic)
          # TODO: Fix default output transform
          |> Map.update(:output_transform, nil, fn _ -> & &1 end)
          |> Axon.Loop.metric(:accuracy)
          |> Axon.Loop.validate(model, train)
          |> Axon.Loop.run(train, initial_state, epochs: 10)

        assert %{step_state: %{model_state: model_state}, metrics: %{9 => last_epoch_metrics}} =
                 results

        eval_results =
          model
          |> Axon.Loop.evaluator()
          |> Axon.Loop.metric(:accuracy)
          |> Axon.Loop.run(train, model_state)

        assert %{0 => %{"accuracy" => final_model_val_accuracy}} = eval_results

        assert_greater_equal(last_epoch_metrics["validation_accuracy"], 0.60)
        assert_all_close(final_model_val_accuracy, last_epoch_metrics["validation_accuracy"])
        assert Nx.shape(Axon.predict(model, model_state, x_test)) == {10, 2}
        assert Nx.type(model_state["dense_0"]["kernel"]) == policy.params
      end)
    end
  end
end
