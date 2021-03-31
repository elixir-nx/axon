defmodule Axon.Training do
  @moduledoc """
  Abstractions for training and validating machine learning models.
  """
  require Axon

  @doc """
  Represents a single training step.
  """
  def step({init_model_fn, objective_fn}, {init_update_fn, update_fn})
      when is_function(init_model_fn) and is_function(objective_fn) and
             is_function(init_update_fn) and is_function(update_fn) do
    init_fn = fn ->
      params = init_model_fn.()
      optim_params = init_update_fn.(params)
      {params, optim_params}
    end

    step_fn = fn params, update_state, input, target ->
      {batch_loss, gradients} =
        Nx.Defn.Kernel.value_and_grad(params, &objective_fn.(&1, input, target))

      {updates, new_update_state} = update_fn.(gradients, update_state)
      {Axon.apply_updates(params, updates), new_update_state, batch_loss}
    end

    {init_fn, step_fn}
  end

  @doc """
  Represents a single training step using an Axon model,
  loss function, and optimizer.
  """
  def step(%Axon{} = model, loss, optimizer) when is_function(loss, 2) do
    init_fn = fn -> Axon.init(model) end

    objective_fn = fn params, input, target ->
      preds = Axon.predict(model, params, input)
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

  ## Options

    * `epochs` - number of epochs to train for. Defaults to `5`.
    * `compiler` - `defn` compiler to use to run training loop.
      Defaults to `Nx.Defn.Evaluator`.
  """
  def train({init_fn, step_fn}, inputs, targets, opts \\ []) do
    epochs = opts[:epochs] || 5
    compiler = opts[:compiler] || Nx.Defn.Evaluator

    {params, optim_state} = Nx.Defn.jit(init_fn, [], compiler: compiler)

    for epoch <- 1..epochs, reduce: {params, optim_state} do
      {cur_params, cur_optim_state} ->
        {time, {new_params, new_optim_state, avg_loss}} =
          :timer.tc(&train_epoch/7, [
            step_fn,
            cur_params,
            cur_optim_state,
            inputs,
            targets,
            compiler,
            epoch
          ])

        epoch_avg_loss =
          avg_loss
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        IO.puts("\n")
        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        IO.puts("Epoch #{epoch} Loss: #{epoch_avg_loss}")
        IO.puts("\n")
        {new_params, new_optim_state}
    end
  end

  ## Helpers

  defp train_epoch(step_fn, cur_params, cur_optim_state, inputs, targets, compiler, epoch) do
    total_batches = Enum.count(inputs)

    dataset =
      inputs
      |> Enum.zip(targets)
      |> Enum.with_index()

    for {{inp, tar}, i} <- dataset, reduce: {cur_params, cur_optim_state, Nx.tensor(0.0)} do
      {params, optim_state, state} ->
        {new_model_state, new_optim_state, batch_loss} =
          Nx.Defn.jit(step_fn, [params, optim_state, inp, tar], compiler: compiler)

        avg_loss =
          state
          |> Nx.multiply(i)
          |> Nx.add(Nx.backend_transfer(batch_loss))
          |> Nx.divide(i + 1)

        IO.write(
          "\rEpoch #{epoch}, batch #{i + 1} of #{total_batches} - Average Loss: #{
            Nx.to_scalar(avg_loss)
          }"
        )

        {new_model_state, new_optim_state, avg_loss}
    end
  end
end
