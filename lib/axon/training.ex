defmodule Axon.Training do
  @moduledoc """
  Training Loop.
  """
  @type axon :: Axon.t
  @type tensor :: Nx.Tensor.t
  @type state :: Tuple.t(tensor)

  @callback loss(model_state :: state, input :: tensor, targets :: tensor) :: state
  @callback train_step(model_state :: state, inputs :: tensor, targets :: tensor) :: state

  defmacro __using__(_opts) do
    quote do
      @behaviour Axon.Training
      use Axon

      @impl true
      defn train_step(params, input, target) do
        gradients = grad(params, &__MODULE__.loss(&1, input, target))
        updates = Axon.map(gradients, &Axon.Updates.scale(&1, -0.01))
        Axon.apply_updates(params, updates)
      end

      def train_epoch(cur_params, inputs, targets, compiler) do
        dataset =
          inputs
          |> Enum.zip(targets)

        for {inp, tar} <- dataset, reduce: cur_params do
          params ->
            Nx.Defn.jit(&__MODULE__.train_step/3, [params, inp, tar], compiler: compiler)
        end
      end

      def train(model, inputs, targets, opts \\ []) do
        epochs = opts[:epochs] || 5
        compiler = opts[:compiler] || Nx.Defn.Evaluator

        IO.puts("Initializing parameters...\n")
        params = Axon.init(model, compiler: compiler)

        for epoch <- 1..epochs, reduce: params do
          cur_params ->
            {time, new_params} =
              :timer.tc(__MODULE__, :train_epoch, [cur_params, inputs, targets, compiler])

            IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
            IO.puts("\n")
            new_params
        end
      end

      defoverridable train_step: 3
    end
  end
end