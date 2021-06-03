defmodule Axon.Training.Callback do
  @type state :: map
  @type options :: keyword

  @callback before_train(state, options) :: state
  @callback before_epoch(state, options) :: state
  @callback before_batch(state, options) :: state
  @callback after_batch(state, options) :: state
  @callback after_epoch(state, options) :: state
  @callback after_train(state, options) :: state

  defmacro __using__(_) do
    quote do
      @behaviour Axon.Training.Callback

      @impl true
      def before_train(state, _), do: state

      @impl true
      def before_epoch(state, _), do: state

      @impl true
      def before_batch(state, _), do: state

      @impl true
      def after_batch(state, _), do: state

      @impl true
      def after_epoch(state, _), do: state

      @impl true
      def after_train(state, _), do: state

      defoverridable before_train: 2,
                     before_epoch: 2,
                     before_batch: 2,
                     after_batch: 2,
                     after_epoch: 2,
                     after_train: 2
    end
  end
end
