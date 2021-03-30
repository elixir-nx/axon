defmodule Axon.Training do
  @moduledoc """
  Training Loop.
  """
  @type axon :: Axon.t
  @type tensor :: Nx.Tensor.t
  @type state :: Tuple.t(tensor)

  @callback loss(model_state :: state, input :: tensor, targets :: tensor) :: state
  @callback train_step(model_state :: state, inputs :: tensor, targets :: tensor) :: {state, state}

  defmacro __using__(_opts) do
    quote do
      @behaviour Axon.Training
      use Axon

      @impl true
      defn train_step(params, input, target) do
        {batch_loss, gradients} = value_and_grad(params, &__MODULE__.loss(&1, input, target))
        updates = Axon.map(gradients, &Axon.Updates.scale(&1, -0.01))
        {Axon.apply_updates(params, updates), batch_loss}
      end

      def train_epoch(cur_params, inputs, targets, compiler) do
        total_batches = Enum.count(inputs)

        dataset =
          inputs
          |> Enum.zip(targets)

        step_fn =
          fn params, inp, tar, state ->
            {new_model_state, batch_loss} = __MODULE__.train_step(params, inp, tar)
            state = Nx.add(Nx.divide(batch_loss, total_batches), state)
            {new_model_state, state}
          end

        for {inp, tar} <- dataset, reduce: {cur_params, Nx.tensor(0.0)} do
          {params, state} ->
            Nx.Defn.jit(step_fn, [params, inp, tar, state], compiler: compiler)
        end
      end

      def train(model, inputs, targets, opts \\ []) do
        epochs = opts[:epochs] || 5
        compiler = opts[:compiler] || Nx.Defn.Evaluator

        IO.puts("Initializing parameters...\n")
        params = Axon.init(model, compiler: compiler)

        for epoch <- 1..epochs, reduce: params do
          cur_params ->
            {time, {new_params, avg_loss}} =
              :timer.tc(__MODULE__, :train_epoch, [cur_params, inputs, targets, compiler])

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

      defoverridable train_step: 3
    end
  end
end