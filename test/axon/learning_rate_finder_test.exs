defmodule Axon.LearningRateFinderTest do
  use Axon.Case, async: true

  alias Axon.LearningRateFinder

  describe "suggestion/2" do
    test "picks the rate where the loss falls fastest, skipping the edges" do
      # flat, then a steady drop between points 20 and 40, then flat
      losses =
        List.duplicate(2.0, 20) ++ Enum.map(1..20, &(2.0 - &1 * 0.075)) ++ List.duplicate(0.5, 20)

      learning_rates = Enum.map(0..(length(losses) - 1), &(&1 * 1.0))

      picked = LearningRateFinder.suggestion(%{learning_rates: learning_rates, losses: losses})
      assert picked >= 20.0 and picked <= 40.0

      # skipping past the drop leaves only the flat tail and its numerical noise
      picked =
        LearningRateFinder.suggestion(%{learning_rates: learning_rates, losses: losses},
          skip_begin: 45
        )

      assert picked >= 45.0
    end

    test "is nil when too few points are left" do
      assert LearningRateFinder.suggestion(%{learning_rates: [1.0, 2.0], losses: [1.0, 0.5]}) ==
               nil

      assert LearningRateFinder.suggestion(%{learning_rates: [1.0, 2.0], losses: [1.0, 0.5]},
               skip_begin: 0,
               skip_end: 0
             ) == 1.0
    end
  end

  describe "run/4" do
    defp regression_data do
      key = Nx.Random.key(7)
      {x, key} = Nx.Random.normal(key, shape: {64, 4})
      {noise, _key} = Nx.Random.normal(key, shape: {64, 1})
      y = x |> Nx.dot(Nx.tensor([[1.0], [-2.0], [0.5], [3.0]])) |> Nx.add(Nx.multiply(noise, 0.1))

      Enum.zip(Nx.to_batched(x, 16), Nx.to_batched(y, 16))
    end

    test "sweeps the rate exponentially and suggests one inside the range" do
      model = Axon.input("x", shape: {nil, 4}) |> Axon.dense(1)

      result =
        LearningRateFinder.run(model, :mean_squared_error, regression_data(),
          min_learning_rate: 1.0e-4,
          max_learning_rate: 1.0,
          num_training: 40,
          seed: 1
        )

      assert length(result.learning_rates) == length(result.losses)
      assert length(result.losses) <= 40
      assert_in_delta hd(result.learning_rates), 1.0e-4, 1.0e-9

      # each step multiplies the rate by the same factor
      ratios =
        result.learning_rates
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [a, b] -> b / a end)

      assert Enum.all?(ratios, &(abs(&1 - :math.pow(1.0e4, 1 / 40)) < 1.0e-6))

      assert Enum.all?(result.losses, &is_float/1)
      assert result.suggestion >= 1.0e-4 and result.suggestion <= 1.0
    end

    test "stops once the loss has blown up" do
      model = Axon.input("x", shape: {nil, 4}) |> Axon.dense(1)

      result =
        LearningRateFinder.run(model, :mean_squared_error, regression_data(),
          min_learning_rate: 1.0e-2,
          max_learning_rate: 1.0e4,
          num_training: 60,
          seed: 1
        )

      assert length(result.losses) < 60

      whole_curve =
        LearningRateFinder.run(model, :mean_squared_error, regression_data(),
          min_learning_rate: 1.0e-2,
          max_learning_rate: 1.0e4,
          num_training: 60,
          early_stop_threshold: nil,
          seed: 1
        )

      assert length(whole_curve.losses) == 60
    end

    test "uses the given optimizer" do
      model = Axon.input("x", shape: {nil, 4}) |> Axon.dense(1)
      parent = self()

      optimizer = fn schedule ->
        send(parent, :optimizer_built)
        Polaris.Optimizers.sgd(learning_rate: schedule)
      end

      result =
        LearningRateFinder.run(model, :mean_squared_error, regression_data(),
          optimizer: optimizer,
          num_training: 20,
          seed: 1
        )

      assert_received :optimizer_built
      assert length(result.losses) <= 20
    end
  end
end
