defmodule Axon.Training do
  @moduledoc """
  Abstractions for training machine learning models.
  """

  require Axon
  require Axon.Updates

  @doc """
  Represents a single training step.

  It expects a pair of 2-element tuples:

    * The first pair contains the model initialization function
      and the objective function. For a Neural Network, the objective
      function is the loss function of the Neural Network prediction

    * The second pairs contains the updater initialization function
      and the update function itself

  """
  def step({init_model_fn, objective_fn}, {init_update_fn, update_fn})
      when is_function(init_model_fn, 0) and is_function(objective_fn, 3) and
             is_function(init_update_fn, 1) and is_function(update_fn, 3) do
    init_fn = fn ->
      params = init_model_fn.()
      optim_params = init_update_fn.(params)
      {params, optim_params}
    end

    step_fn = fn model_state, input, target ->
      {params, update_state} = model_state

      {batch_loss, gradients} =
        Nx.Defn.Kernel.value_and_grad(params, &objective_fn.(&1, input, target))

      {updates, new_update_state} = update_fn.(gradients, update_state, params)
      {{Axon.Updates.apply_updates(params, updates), new_update_state}, batch_loss}
    end

    {init_fn, step_fn}
  end

  @doc """
  Represents a single training step using an Axon `model`,
  `loss` function, and `optimizer`.

  The `loss` function is either an atom or a two arity
  anonymous function.
  """
  def step(%Axon{} = model, loss, optimizer) when is_function(loss, 2) do
    {init_fn, predict_fn} = Axon.compile(model)

    objective_fn = fn params, input, target ->
      preds = predict_fn.(params, input)
      loss.(target, preds)
    end

    step({init_fn, objective_fn}, optimizer)
  end

  def step(%Axon{} = model, loss, optimizer) when is_atom(loss) do
    loss_fn = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
    step(model, loss_fn, optimizer)
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

    jit_opts = [compiler: compiler] ++ opts
    model_state = Nx.Defn.jit(init_fn, [], jit_opts)

    for epoch <- 1..epochs, reduce: model_state do
      model_state ->
        {time, {model_state, avg_loss}} =
          :timer.tc(&train_epoch/7, [
            step_fn,
            model_state,
            inputs,
            targets,
            compiler,
            epoch,
            jit_opts
          ])

        epoch_avg_loss =
          avg_loss
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        IO.puts("\n")
        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        IO.puts("Epoch #{epoch} Loss: #{epoch_avg_loss}")
        IO.puts("\n")
        model_state
    end
  end

  ## Helpers

  defp train_epoch(step_fn, model_state, inputs, targets, compiler, epoch, jit_opts) do
    total_batches = Enum.count(inputs)

    dataset =
      inputs
      |> Enum.zip(targets)
      |> Enum.with_index()

    for {{inp, tar}, i} <- dataset, reduce: {model_state, Nx.tensor(0.0)} do
      {model_state, state} ->
        {model_state, batch_loss} =
          Nx.Defn.jit(step_fn, [model_state, inp, tar], jit_opts)

        avg_loss =
          state
          |> Nx.multiply(i)
          |> Nx.add(Nx.backend_transfer(batch_loss))
          |> Nx.divide(i + 1)

        IO.write(
          "\rEpoch #{epoch}, batch #{i + 1} of #{total_batches} - " <>
            "Average Loss: #{Nx.to_scalar(avg_loss)}"
        )

        {model_state, avg_loss}
    end
  end
end
