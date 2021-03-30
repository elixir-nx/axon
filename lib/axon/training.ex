defmodule Axon.Training do
  @moduledoc """
  Training Loop.
  """
  require Axon

  def step(objective_fn, _opts \\ []) do
    fn params, input, target ->
      {batch_loss, gradients} = Nx.Defn.Kernel.value_and_grad(params, &objective_fn.(&1, input, target))
      updates = Axon.map(gradients, &Axon.Updates.scale(&1, -0.01))
      {Axon.apply_updates(params, updates), batch_loss}
    end
  end

  def step(%Axon{} = x, loss, _opts \\ []) when is_atom(loss) do
    loss_fn = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
    objective_fn =
      fn params, input, target ->
        preds = Axon.predict(model, params, input)
        loss_fn.(target, preds)
      end
    step(objective_fn)
  end

  def train_epoch(step_fn, cur_params, inputs, targets, compiler) do
    total_batches = Enum.count(inputs)

    dataset =
      inputs
      |> Enum.zip(targets)

    step_fn =
      fn params, inp, tar, state ->
        {new_model_state, batch_loss} = step_fn.(params, inp, tar)
        state = Nx.add(Nx.divide(batch_loss, total_batches), state)
        {new_model_state, state}
      end

    for {inp, tar} <- dataset, reduce: {cur_params, Nx.tensor(0.0)} do
      {params, state} ->
        Nx.Defn.jit(step_fn, [params, inp, tar, state], compiler: compiler)
    end
  end

  def train(model, step_fn, inputs, targets, opts \\ []) do
    epochs = opts[:epochs] || 5
    compiler = opts[:compiler] || Nx.Defn.Evaluator

    IO.puts("Initializing parameters...\n")
    params = Axon.init(model, compiler: compiler)

    for epoch <- 1..epochs, reduce: params do
      cur_params ->
        {time, {new_params, avg_loss}} =
          :timer.tc(__MODULE__, :train_epoch, [step_fn, cur_params, inputs, targets, compiler])

        epoch_avg_loss =
          avg_loss
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        IO.puts("Epoch #{epoch} Loss: #{epoch_avg_loss}")
        IO.puts("\n")
        new_params
    end
  end
end