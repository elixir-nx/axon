defmodule Axon.Training do
  @moduledoc """
  Abstractions for training machine learning models.
  """

  require Axon
  require Axon.Updates

  @doc """
  Represents a single training step.

  It expects a pair of tuples:

    * The first tuples contains the model initialization function,
      the model predict function, and the objective function. For a
      Neural Network, the objective function is the loss function of
      the Neural Network prediction

    * The second pairs contains the updater initialization function
      and the update function itself

  """
  def step({init_model_fn, predict_fn, objective_fn}, {init_update_fn, update_fn}, opts)
      when is_function(init_model_fn, 0) and is_function(objective_fn, 3) and
             is_function(init_update_fn, 1) and is_function(update_fn, 3) and is_list(opts) do
    metrics = opts[:metrics] || []

    update_metrics_fn = fn old_metrics, step, y_true, y_pred ->
      metrics
      |> Enum.map(fn
        {key, fun} ->
          batch_metric = fun.(y_true, y_pred)

          avg_metric =
            old_metrics[key]
            |> Nx.multiply(step)
            |> Nx.add(batch_metric)
            |> Nx.divide(Nx.add(step, 1))

          {key, avg_metric}

        key ->
          batch_metric = apply(Axon.Metrics, key, [y_true, y_pred])

          avg_metric =
            old_metrics[key]
            |> Nx.multiply(step)
            |> Nx.add(batch_metric)
            |> Nx.divide(Nx.add(step, 1))

          {key, avg_metric}
      end)
      |> Map.new()
    end

    init_fn = fn ->
      params = init_model_fn.()
      optim_params = init_update_fn.(params)
      init_metrics = Enum.map(metrics, fn k -> {k, Nx.Defn.Expr.tensor(0.0)} end) |> Map.new()

      %{
        epoch: Nx.Defn.Expr.tensor(0),
        epoch_step: Nx.Defn.Expr.tensor(0),
        epoch_loss: Nx.Defn.Expr.tensor(0.0),
        params: params,
        optimizer_state: optim_params,
        metrics: init_metrics
      }
    end

    step_fn = fn model_state, input, target ->
      {batch_loss, gradients} =
        Nx.Defn.Kernel.value_and_grad(model_state[:params], &objective_fn.(&1, input, target))

      new_metrics =
        case metrics do
          [] ->
            %{}

          _ ->
            preds = predict_fn.(model_state[:params], input)
            update_metrics_fn.(model_state[:metrics], model_state[:epoch_step], target, preds)
        end

      epoch_avg_loss =
        model_state[:epoch_loss]
        |> Nx.multiply(model_state[:epoch_step])
        |> Nx.add(batch_loss)
        |> Nx.divide(Nx.add(model_state[:epoch_step], 1))

      {updates, new_update_state} =
        update_fn.(gradients, model_state[:optimizer_state], model_state[:params])

      %{
        epoch: model_state[:epoch],
        epoch_step: Nx.add(model_state[:epoch_step], 1),
        epoch_loss: epoch_avg_loss,
        params: Axon.Updates.apply_updates(model_state[:params], updates),
        optimizer_state: new_update_state,
        metrics: new_metrics
      }
    end

    {init_fn, step_fn}
  end

  @doc """
  Represents a single training step using an Axon `model`,
  `loss` function, and `optimizer`.

  The `loss` function is either an atom or a two arity
  anonymous function.
  """
  def step(%Axon{} = model, loss, optimizer, opts)
      when is_function(loss, 2) and is_list(opts) do
    {init_fn, predict_fn} = Axon.compile(model)

    objective_fn = fn params, input, target ->
      preds = predict_fn.(params, input)
      Nx.add(loss.(target, preds), Axon.penalty(model, params))
    end

    step({init_fn, predict_fn, objective_fn}, optimizer, opts)
  end

  def step(%Axon{} = model, loss, optimizer, opts) when is_atom(loss) and is_list(opts) do
    loss_fn = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
    step(model, loss_fn, optimizer, opts)
  end

  @doc """
  Represents a single training step using an Axon `model`,
  initial state `model_state`, `loss` function and `optimizer`.

  The `loss` function is either an atom or a two arity anonymous
  function.
  """
  def step(%Axon{} = model, model_state, loss, optimizer, opts)
      when is_function(loss, 2) and is_list(opts) do
    # TODO: I don't think we should do this, but it seems
    # to be the workaround with the fewest implications
    # that I'm aware of
    init_fn = fn ->
      model_state
      |> Tuple.to_list()
      |> Enum.map(&Nx.Defn.Expr.tensor/1)
      |> List.to_tuple()
    end

    objective_fn = fn params, input, target ->
      preds = Axon.predict(model, params, input)
      Nx.add(loss.(target, preds), Axon.penalty(model, params))
    end

    step({init_fn, objective_fn}, optimizer, opts)
  end

  def step(%Axon{} = model, model_state, loss, optimizer, opts)
      when is_atom(loss) and is_list(opts) do
    loss_fn = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
    step(model, model_state, loss_fn, optimizer, opts)
  end

  @doc """
  Implements a common training loop.

  Its arguments are:

    * A tuple with the initialization function and the step function.
      Often retrieved from `step/3` but it could also be manually provided.

    * The inputs tensors

    * The targets tensors

    * A list of options

  ## Options

    * `:epochs` - number of epochs to train for. Defaults to `5`.
    * `:compiler` - `defn` compiler to use to run training loop.
      Defaults to `Nx.Defn.Evaluator`.
    * `:log_every` - frequency with which to log training loss.
      Accepts an integer referring to number of batches, `:epoch`,
      or `:none`. Defaults to `:epoch`.

  All other options are given to the underlying compiler.

  ## A note on Nx and anonymous functions

  When training, both `init_fn` and `step_fn` are executed within
  the given Nx `:compiler`. Therefore, it is required that `init_fn`
  and `step_fn` work on tensor expressions instead of tensor values.

  For example, let's suppose you want to initialize the values with:

      Nx.random_uniform({40, 28}, 0, 1)

  The following won't work:

      params = Nx.random_uniform({40, 28}, 0, 1)
      init_fn = fn -> params end

  Instead, we want to build the values inside the given compiler.
  The correct way to build those values is by compuing them inside
  a defn:

      defn init_values, do: Nx.random_uniform({40, 28}, 0, 1)

  And then:

      init_fn = &init_values/0

  """
  def train({init_fn, step_fn}, inputs, targets, opts \\ []) do
    epochs = opts[:epochs] || 5
    compiler = opts[:compiler] || Nx.Defn.Evaluator
    log_every = opts[:log_every] || 50

    jit_opts = [compiler: compiler, log_every: log_every] ++ opts
    model_state = Nx.Defn.jit(init_fn, [], jit_opts)

    for epoch <- 1..epochs, reduce: model_state do
      model_state ->
        {time, model_state} =
          :timer.tc(
            &train_epoch/6,
            [step_fn, model_state, inputs, targets, epoch, jit_opts]
          )

        epoch_avg_loss =
          model_state[:epoch_loss]
          |> Nx.to_scalar()

        zero_metrics =
          model_state[:metrics]
          |> Enum.map(fn {k, _} -> {k, 0.0} end)
          |> Map.new()

        IO.puts("\n")
        IO.puts("Epoch #{epoch} time: #{time / 1_000_000}s")
        IO.puts("Epoch #{epoch} loss: #{:io_lib.format("~.5f", [epoch_avg_loss])}")
        model_state[:metrics]
        |> Enum.each(fn {k, v} -> IO.puts("Epoch #{epoch} #{Atom.to_string(k)}: #{:io_lib.format("~.5f", [Nx.to_scalar(v)])}") end)
        IO.puts("\n")

        %{model_state | metrics: zero_metrics, epoch: epoch + 1, epoch_step: 0, epoch_loss: 0.0}
    end
  end

  ## Helpers

  defp train_epoch(step_fn, model_state, inputs, targets, epoch, opts) do
    {log_every, jit_opts} = Keyword.pop(opts, :log_every)

    dataset =
      inputs
      |> Stream.zip(targets)

    model_state =
      for {inp, tar} <- dataset, reduce: model_state do
        model_state ->
          model_state = Nx.Defn.jit(step_fn, [model_state, inp, tar], jit_opts)

          if is_integer(log_every) and Nx.remainder(model_state[:epoch_step], log_every) == Nx.tensor(0) do
            log_batch(epoch, model_state)
          end

          model_state
      end

    model_state
  end

  defp log_batch(epoch, model_state) do
    batch_num = model_state[:epoch_step]
    avg_loss = model_state[:epoch_loss]

    metrics =
      model_state[:metrics]
      |> Enum.map(fn {k, v} -> "Average #{Atom.to_string(k)}: #{:io_lib.format("~.5f", [Nx.to_scalar(v)])}" end)

    metrics = Enum.join(["Average Loss: #{:io_lib.format("~.5f", [Nx.to_scalar(avg_loss)])}" | metrics], " - ")

    IO.write(
      "\rEpoch #{epoch}, batch #{Nx.to_scalar(batch_num)} - " <>
        "#{metrics}"
    )
  end
end
