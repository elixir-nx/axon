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
end
