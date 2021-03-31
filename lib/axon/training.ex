defmodule Axon.Training do
  @moduledoc """
  Training Loop.
  """
  require Axon

  def step(objective_fn, update_fn) do
    fn params, update_state, input, target ->
      {batch_loss, gradients} = Nx.Defn.Kernel.value_and_grad(params, &objective_fn.(&1, input, target))
      {updates, new_update_state} = update_fn.(gradients, update_state)
      {Axon.apply_updates(params, updates), new_update_state, batch_loss}
    end
  end

  def step(%Axon{} = model, loss, update_fn) when is_function(loss, 2) do
    objective_fn =
      fn params, input, target ->
        preds = Axon.predict(model, params, input)
        loss.(target, preds)
      end
    step(objective_fn, update_fn)
  end

  def step(%Axon{} = model, loss, update_fn) when is_atom(loss) do
    loss_fn = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
    step(model, loss_fn, update_fn)
  end

  def train(step_fn, model, optim_init_fn, inputs, targets, opts \\ []) do
    epochs = opts[:epochs] || 5
    compiler = opts[:compiler] || Nx.Defn.Evaluator

    params = Axon.init(model, compiler: compiler)
    optim_state = Nx.Defn.jit(optim_init_fn, [params], compiler: compiler)

    for epoch <- 1..epochs, reduce: {params, optim_state} do
      {cur_params, cur_optim_state} ->
        {time, {new_params, new_optim_state, avg_loss}} =
          :timer.tc(&train_epoch/7, [step_fn, cur_params, cur_optim_state, inputs, targets, compiler, epoch])

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
        {new_model_state, new_optim_state, batch_loss} = Nx.Defn.jit(step_fn, [params, optim_state, inp, tar], compiler: compiler)

        avg_loss =
          state
          |> Nx.multiply(i)
          |> Nx.add(Nx.backend_transfer(batch_loss))
          |> Nx.divide(i + 1)

        IO.write("\rEpoch #{epoch}, batch #{i + 1} of #{total_batches} - Average Loss: #{Nx.to_scalar(avg_loss)}")

        {new_model_state, new_optim_state, avg_loss}
    end
  end
end