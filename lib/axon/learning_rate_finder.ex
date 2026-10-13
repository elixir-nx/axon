defmodule Axon.LearningRateFinder do
  @moduledoc """
  Learning rate range test.

  Trains a model for a small number of iterations while the learning rate
  grows exponentially from `:min_learning_rate` to `:max_learning_rate`,
  records the training loss at every step and suggests the learning rate
  at which the loss was falling fastest. This is the range test from
  Leslie Smith's "Cyclical Learning Rates for Training Neural Networks"
  and the same procedure as PyTorch Lightning's and fast.ai's `lr_find`.

  The suggested rate is a good peak for a one-cycle schedule and a
  reasonable constant rate to start from:

      model = Axon.input("x", shape: {nil, 4}) |> Axon.dense(1)

      %{suggestion: learning_rate} =
        Axon.LearningRateFinder.run(model, :mean_squared_error, data)

      model
      |> Axon.Loop.trainer(:mean_squared_error, Polaris.Optimizers.adam(learning_rate: learning_rate))
      |> Axon.Loop.run(data, %{}, epochs: 10)

  The recorded curve is returned too, so it can be plotted or fed back to
  `suggestion/2` with different options.
  """

  alias Axon.Loop

  @doc """
  Runs the range test and suggests a learning rate.

  The model is trained from a fresh initialization on `:num_training`
  batches of `data` (cycled if it has fewer) with the learning rate
  climbing exponentially from `:min_learning_rate` to `:max_learning_rate`.
  The loss of every batch is recorded before the update that batch drives,
  paired with the learning rate of that update, and smoothed with an
  exponential moving average. The run stops early once the smoothed loss
  exceeds `:early_stop_threshold` times the best loss so far, since past
  that point the model has diverged and the rest of the curve is noise.

  ## Options

    * `:optimizer` - a function from a learning rate schedule to a Polaris
      optimizer, so the test can drive the rate. Defaults to
      `&Polaris.Optimizers.adam(learning_rate: &1)`. Use the optimizer you
      will train with.

    * `:min_learning_rate` - the rate at the first step. Defaults to `1.0e-6`.

    * `:max_learning_rate` - the rate at the last step. Defaults to `10.0`.

    * `:num_training` - how many steps to run. Defaults to `100`.

    * `:smoothing` - the exponential moving average factor applied to the
      losses, `0.0` for none. Defaults to `0.98`.

    * `:early_stop_threshold` - stop when the smoothed loss exceeds this
      many times the best loss so far, `nil` to always run every step.
      Defaults to `4.0`.

    * `:seed` - the seed for model initialization.

    * `:skip_begin`, `:skip_end` - see `suggestion/2`.

  Any other option is forwarded to `Axon.Loop.run/4`, for example
  `compiler: EXLA`.

  ## Returns

  A map with:

    * `:learning_rates` - the rate at every recorded step
    * `:losses` - the smoothed loss at every recorded step
    * `:suggestion` - the suggested learning rate, or `nil` when the curve
      is too short to pick one

  """
  @spec run(Axon.t(), atom() | function() | list(), Enumerable.t(), keyword()) :: %{
          learning_rates: list(float()),
          losses: list(float()),
          suggestion: float() | nil
        }
  def run(%Axon{} = model, loss, data, opts \\ []) do
    {finder_opts, run_opts} =
      Keyword.split(opts, [
        :optimizer,
        :min_learning_rate,
        :max_learning_rate,
        :num_training,
        :smoothing,
        :early_stop_threshold,
        :seed,
        :skip_begin,
        :skip_end
      ])

    finder_opts =
      Keyword.validate!(finder_opts,
        optimizer: &Polaris.Optimizers.adam(learning_rate: &1),
        min_learning_rate: 1.0e-6,
        max_learning_rate: 10.0,
        num_training: 100,
        smoothing: 0.98,
        early_stop_threshold: 4.0,
        seed: nil,
        skip_begin: 10,
        skip_end: 1
      )

    min_learning_rate = finder_opts[:min_learning_rate]
    max_learning_rate = finder_opts[:max_learning_rate]
    num_training = finder_opts[:num_training]

    schedule =
      Polaris.Schedules.exponential_decay(min_learning_rate,
        decay_rate: max_learning_rate / min_learning_rate,
        transition_steps: num_training
      )

    loss_fn = Loop.build_loss_fn(loss)

    trainer_opts =
      case finder_opts[:seed] do
        nil -> [log: 0]
        seed -> [log: 0, seed: seed]
      end

    loop =
      model
      |> Loop.trainer(loss_fn, finder_opts[:optimizer].(schedule), trainer_opts)
      |> Loop.handle_event(:iteration_completed, fn state ->
        record(state, loss_fn, schedule, finder_opts)
      end)

    batches = data |> Stream.cycle() |> Stream.take(num_training)

    %Loop.State{handler_metadata: metadata} =
      Loop.run(
        loop,
        batches,
        Axon.ModelState.empty(),
        [epochs: 1, iterations: num_training] ++ run_opts
      )

    record = Map.get(metadata, :learning_rate_finder, empty_record())

    result = %{
      learning_rates: Enum.reverse(record.learning_rates),
      losses: Enum.reverse(record.losses)
    }

    Map.put(
      result,
      :suggestion,
      suggestion(result, Keyword.take(finder_opts, [:skip_begin, :skip_end]))
    )
  end

  @doc """
  Picks the learning rate where the loss falls fastest.

  Takes the `:learning_rates` and `:losses` of a `run/4` result, drops
  `:skip_begin` points at the start (the loss barely moves at tiny rates
  and its gradient there is noise) and `:skip_end` points at the end (the
  last points before divergence look better than they are), and returns
  the rate at the most negative gradient of what is left. Returns `nil`
  when fewer than two points remain.

  ## Options

    * `:skip_begin` - points to ignore at the start. Defaults to `10`.
    * `:skip_end` - points to ignore at the end. Defaults to `1`.

  """
  @spec suggestion(%{learning_rates: list(float()), losses: list(float())}, keyword()) ::
          float() | nil
  def suggestion(%{learning_rates: learning_rates, losses: losses}, opts \\ []) do
    opts = Keyword.validate!(opts, skip_begin: 10, skip_end: 1)

    candidates =
      losses
      |> Enum.zip(learning_rates)
      |> Enum.drop(opts[:skip_begin])
      |> Enum.drop(-opts[:skip_end])

    if length(candidates) < 2 do
      nil
    else
      {candidate_losses, candidate_rates} = Enum.unzip(candidates)

      {_steepest, index} =
        candidate_losses
        |> gradient()
        |> Enum.with_index()
        |> Enum.min_by(fn {slope, _index} -> slope end)

      Enum.at(candidate_rates, index)
    end
  end

  defp record(%Loop.State{} = state, loss_fn, schedule, opts) do
    record = Map.get(state.handler_metadata, :learning_rate_finder, empty_record())
    step = length(record.losses)

    %{y_true: y_true, y_pred: y_pred} = state.step_state
    batch_loss = loss_fn.(y_true, y_pred) |> Nx.to_number()

    if is_number(batch_loss) do
      smoothing = opts[:smoothing]
      average = smoothing * record.average + (1 - smoothing) * batch_loss
      smoothed = average / (1 - :math.pow(smoothing, step + 1))
      best = min(record.best || smoothed, smoothed)
      learning_rate = schedule.(step) |> Nx.to_number()

      record = %{
        record
        | losses: [smoothed | record.losses],
          learning_rates: [learning_rate | record.learning_rates],
          average: average,
          best: best
      }

      state = put_in(state.handler_metadata[:learning_rate_finder], record)

      if diverged?(smoothed, best, opts[:early_stop_threshold]) do
        {:halt_loop, state}
      else
        {:continue, state}
      end
    else
      {:halt_loop, state}
    end
  end

  defp diverged?(_smoothed, _best, nil), do: false
  defp diverged?(smoothed, best, threshold), do: smoothed > threshold * best

  defp empty_record, do: %{losses: [], learning_rates: [], average: 0.0, best: nil}

  # Central differences, one-sided at the ends, like numpy's gradient
  defp gradient(values) do
    tuple = List.to_tuple(values)
    last = tuple_size(tuple) - 1

    Enum.map(0..last, fn index ->
      cond do
        index == 0 -> elem(tuple, 1) - elem(tuple, 0)
        index == last -> elem(tuple, last) - elem(tuple, last - 1)
        true -> (elem(tuple, index + 1) - elem(tuple, index - 1)) / 2
      end
    end)
  end
end
