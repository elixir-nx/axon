defmodule Axon.IntegrationTest do
  use Axon.Case, async: true

  import AxonTestUtil

  @moduletag :integration

  test "bce with simple xor model" do
    x1_input = Axon.input("x1", shape: {nil, 1})
    x2_input = Axon.input("x2", shape: {nil, 1})

    model =
      x1_input
      |> Axon.concatenate(x2_input)
      |> Axon.dense(8, activation: :tanh)
      |> Axon.dense(1, activation: :sigmoid)

    batch_size = 32

    data =
      Stream.unfold(Nx.Random.key(42), fn key ->
        {x1, key} = Nx.Random.uniform(key, 0, 1, shape: {batch_size, 1})
        {x2, key} = Nx.Random.uniform(key, 0, 1, shape: {batch_size, 1})

        {x1, x2} = {Nx.round(x1), Nx.round(x2)}
        y = Nx.logical_xor(x1, x2)

        {{%{"x1" => x1, "x2" => x2}, y}, key}
      end)

    ExUnit.CaptureIO.capture_io(fn ->
      model_state =
        model
        |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
        |> Axon.Loop.run(data, Axon.ModelState.empty(), iterations: 100, epochs: 10)

      eval_results =
        model
        |> Axon.Loop.evaluator()
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.run(data, model_state, iterations: 100)

      assert_greater_equal(get_in(eval_results, [0, "accuracy"]), 0.9)
    end)
  end

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
        |> Axon.Loop.trainer(
          :categorical_cross_entropy,
          Polaris.Optimizers.adam(learning_rate: 5.0e-3)
        )
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

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

  test "f64 input test" do
    {train, _test} = get_test_data(100, 0, 10, {10}, 2, 1337)

    train =
      train
      |> Stream.map(fn {xs, ys} ->
        {Nx.as_type(xs, :f64), one_hot(ys, num_classes: 2)}
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
          Polaris.Optimizers.adam(learning_rate: 5.0e-3)
        )
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

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
        |> Axon.Loop.trainer(
          :categorical_cross_entropy,
          Polaris.Optimizers.adam(learning_rate: 5.0e-3)
        )
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

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
        |> Axon.Loop.trainer(
          :categorical_cross_entropy,
          Polaris.Optimizers.adam(learning_rate: 5.0e-3)
        )
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

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
        |> Axon.Loop.trainer(
          :categorical_cross_entropy,
          Polaris.Optimizers.adam(learning_rate: 5.0e-3),
          gradient_accumulation_steps: 3
        )
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

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
        |> Axon.Loop.trainer(
          :categorical_cross_entropy,
          Polaris.Optimizers.adam(learning_rate: 5.0e-3),
          seed: 1
        )
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

      %{metrics: metrics2, step_state: step_state2} =
        model
        |> Axon.Loop.trainer(
          :categorical_cross_entropy,
          Polaris.Optimizers.adam(learning_rate: 5.0e-3),
          seed: 1
        )
        # TODO: Fix default output transform
        |> Map.update(:output_transform, nil, fn _ -> & &1 end)
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, train)
        |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

      assert_equal(metrics1, metrics2)
      assert_equal(step_state1, step_state2)
    end)
  end

  describe "optimizer integration" do
    @optimizers_and_args [
      {:adabelief, [[learning_rate: 5.0e-3]]},
      {:adagrad, [[learning_rate: 5.0e-3]]},
      {:adam, [[learning_rate: 5.0e-3]]},
      {:adamw, [[learning_rate: 5.0e-3]]},
      {:adamw, [[learning_rate: 5.0e-3, decay: 0.9]]},
      {:lamb, [[learning_rate: 5.0e-3]]},
      {:lamb, [[learning_rate: 5.0e-3, decay: 0.9]]},
      {:lamb, [[learning_rate: 5.0e-3, min_norm: 0.1]]},
      {:lamb, [[learning_rate: 5.0e-3, decay: 0.9, min_norm: 0.1]]},
      {:noisy_sgd, [[learning_rate: 5.0e-3]]},
      {:radam, [[learning_rate: 5.0e-3]]},
      {:rmsprop, [[learning_rate: 5.0e-3]]},
      {:rmsprop, [[learning_rate: 5.0e-3, centered: true]]},
      {:rmsprop, [[learning_rate: 5.0e-3, momentum: 0.9]]},
      {:rmsprop, [[learning_rate: 5.0e-3, nesterov: true, momentum: 0.9]]},
      {:rmsprop, [[learning_rate: 5.0e-3, centered: true, nesterov: true, momentum: 0.9]]},
      {:sgd, [[learning_rate: 5.0e-3]]},
      {:sgd, [[learning_rate: 5.0e-3, momentum: 0.9]]},
      {:sgd, [[learning_rate: 5.0e-3, momentum: 0.9, nesterov: true]]}
    ]

    for {optimizer, [opts] = args} <- @optimizers_and_args do
      lr = opts[:learning_rate]

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
              Polaris.Optimizers.unquote(optimizer)(unquote_splicing(args))
            )
            # TODO: Fix default output transform
            |> Map.update(:output_transform, nil, fn _ -> & &1 end)
            |> Axon.Loop.metric(:accuracy)
            |> Axon.Loop.validate(model, train)
            |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

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

  describe "rnns" do
    test "static and dynamic unroll match" do
      key = Nx.Random.key(42)
      {data, _} = Nx.Random.randint(key, 1, 2, shape: {2, 16})

      train =
        data
        |> Nx.to_batched(2)
        |> Stream.map(fn x ->
          # if over half are greater than 64
          {x, Nx.new_axis(Nx.greater(Nx.sum(Nx.greater(x, 64), axes: [-1]), 8), -1)}
        end)

      input = Axon.input("input")

      dynamic_model =
        input
        |> Axon.embedding(2, 8)
        |> Axon.lstm(5, recurrent_initializer: :zeros)
        |> elem(0)
        |> Axon.nx(fn seq -> Nx.squeeze(seq[[0..-1//1, -1, 0..-1//1]]) end)

      static_model =
        input
        |> Axon.embedding(2, 8)
        |> Axon.lstm(5, unroll: :static, recurrent_initializer: :zeros)
        |> elem(0)
        |> Axon.nx(fn seq -> Nx.squeeze(seq[[0..-1//1, -1, 0..-1//1]]) end)

      ExUnit.CaptureIO.capture_io(fn ->
        %Axon.ModelState{data: dynamic} =
          dynamic_model
          |> Axon.Loop.trainer(
            :mean_squared_error,
            Polaris.Optimizers.adam(learning_rate: 1.0e-3),
            seed: 10
          )
          |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 1)

        %Axon.ModelState{data: static} =
          static_model
          |> Axon.Loop.trainer(
            :mean_squared_error,
            Polaris.Optimizers.adam(learning_rate: 1.0e-3),
            seed: 10
          )
          |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 1)

        # After a single step, initialized to the same seed with exact same configuration
        # and inputs, these should be exactly the same
        assert_all_close(dynamic, static, atol: 1.0e-3)
      end)
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
            |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

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

          assert Nx.type(model_state.data["dense_0"]["kernel"]) ==
                   unquote(Macro.escape(policy)).params
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
          |> Axon.MixedPrecision.apply_policy(
            unquote(Macro.escape(policy)),
            except: [:batch_norm]
          )

        ExUnit.CaptureIO.capture_io(fn ->
          results =
            model
            |> Axon.Loop.trainer(:categorical_cross_entropy, :adam, loss_scale: unquote(scale))
            # TODO: Fix default output transform
            |> Map.update(:output_transform, nil, fn _ -> & &1 end)
            |> Axon.Loop.metric(:accuracy)
            |> Axon.Loop.validate(model, train)
            |> Axon.Loop.run(train, Axon.ModelState.empty(), epochs: 10)

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

          assert Nx.type(model_state.data["dense_0"]["kernel"]) ==
                   unquote(Macro.escape(policy)).params
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
      initial_state = init_fn.(Nx.template({1, 10}, :f32), Axon.ModelState.empty())

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
        assert Nx.type(model_state.data["dense_0"]["kernel"]) == policy.params
      end)
    end
  end
end
