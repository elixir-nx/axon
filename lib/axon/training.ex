defmodule Axon.Training do
  @moduledoc """
  Training Loop.
  """
  require Axon

  def step(objective_fn) do
    fn params, input, target ->
      {batch_loss, gradients} = Nx.Defn.Kernel.value_and_grad(params, &objective_fn.(&1, input, target))
      updates = Axon.Updates.scale(gradients, -0.01)
      {Axon.apply_updates(params, updates), batch_loss}
    end
  end

  def step(%Axon{} = model, loss) when is_function(loss, 2) do
    objective_fn =
      fn params, input, target ->
        preds = Axon.predict(model, params, input)
        loss.(target, preds)
      end
    step(objective_fn)
  end

  def step(%Axon{} = model, loss) when is_atom(loss) do
    loss_fn = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
    step(model, loss_fn)
  end

  defp train_epoch(step_fn, cur_params, inputs, targets, compiler, epoch) do
    total_batches = Enum.count(inputs)

    dataset =
      inputs
      |> Enum.zip(targets)
      |> Enum.with_index()

    step_fn =
      fn params, inp, tar ->
        {new_model_state, batch_loss} = step_fn.(params, inp, tar)
        {new_model_state, batch_loss}
      end

    for {{inp, tar}, i} <- dataset, reduce: {cur_params, Nx.tensor(0.0)} do
      {params, state} ->
        {new_model_state, batch_loss} = Nx.Defn.jit(step_fn, [params, inp, tar], compiler: compiler)
        avg_loss =
          state
          |> Nx.multiply(i)
          |> Nx.add(Nx.backend_transfer(batch_loss))
          |> Nx.divide(i + 1)

        IO.write("\rEpoch #{epoch}, batch #{i + 1} of #{total_batches} - Average Loss: #{Nx.to_scalar(avg_loss)}")

        {new_model_state, avg_loss}
    end
  end

  def train(step_fn, model, inputs, targets, opts \\ []) do
    epochs = opts[:epochs] || 5
    compiler = opts[:compiler] || Nx.Defn.Evaluator

    IO.puts("Initializing parameters...\n")
    params = Axon.init(model, compiler: compiler)

    for epoch <- 1..epochs, reduce: params do
      cur_params ->
        {time, {new_params, avg_loss}} =
          :timer.tc(&train_epoch/6, [step_fn, cur_params, inputs, targets, compiler, epoch])

        epoch_avg_loss =
          avg_loss
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        IO.puts("\n")
        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        IO.puts("Epoch #{epoch} Loss: #{epoch_avg_loss}")
        IO.puts("\n")
        new_params
    end
  end
end