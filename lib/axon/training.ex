defmodule Axon.Training do
  @moduledoc """
  Abstractions for training machine learning models.
  """
  require Axon
  require Axon.Updates

  alias Axon.Training.Step

  @doc false
  def step({_, _} = model, {_, _} = update), do: step(model, update, [])

  @doc """
  Represents a single training step.

  The first two arguments are tuples:

    * The first tuple contains the model initialization function
      and the objective function. For a Neural Network, the objective
      function is the loss function of the Neural Network prediction

    * The second pairs contains the updater initialization function
      and the update function itself

  ## Options

    * `:metrics` - metrics to track during each training step. Can be an
      atom representing a function in `Axon.Metrics`, or a 2-arity function
      taking `y_true` and `y_pred` as args.

  """
  def step({init_model_fn, objective_fn}, {init_update_fn, update_fn}, opts)
      when is_function(init_model_fn, 0) and is_function(objective_fn, 3) and
             is_function(init_update_fn, 1) and is_function(update_fn, 3) and is_list(opts) do
    metrics = opts[:metrics] || []

    update_metrics_fn = fn old_metrics, step, y_true, y_pred ->
      Map.new(metrics, fn
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
    end

    init_fn = fn ->
      params = init_model_fn.()
      optim_params = init_update_fn.(params)

      init_metrics = Map.new(metrics, fn k -> {k, Nx.tensor(0.0, backend: Nx.Defn.Expr)} end)

      %{
        epoch: Nx.tensor(0, backend: Nx.Defn.Expr),
        epoch_step: Nx.tensor(0, backend: Nx.Defn.Expr),
        epoch_loss: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        params: params,
        optimizer_state: optim_params,
        metrics: init_metrics
      }
    end

    step_fn = fn train_state, input, target ->
      {{preds, batch_loss}, gradients} =
        Nx.Defn.Kernel.value_and_grad(
          train_state[:params],
          &objective_fn.(&1, input, target),
          fn x -> elem(x, 1) end
        )

      new_metrics =
        case metrics do
          [] ->
            %{}

          _ ->
            update_metrics_fn.(train_state[:metrics], train_state[:epoch_step], target, preds)
        end

      epoch_avg_loss =
        train_state[:epoch_loss]
        |> Nx.multiply(train_state[:epoch_step])
        |> Nx.add(batch_loss)
        |> Nx.divide(Nx.add(train_state[:epoch_step], 1))

      {updates, new_update_state} =
        update_fn.(gradients, train_state[:optimizer_state], train_state[:params])

      %{
        epoch: train_state[:epoch],
        epoch_step: Nx.add(train_state[:epoch_step], 1),
        epoch_loss: epoch_avg_loss,
        params: Axon.Updates.apply_updates(train_state[:params], updates),
        optimizer_state: new_update_state,
        metrics: new_metrics
      }
    end

    %Step{init: init_fn, step: step_fn, callbacks: []}
  end

  @doc false
  def step(%Axon{} = model, loss, {_, _} = optimizer) when is_function(loss, 2) or is_atom(loss),
    do: step(model, loss, optimizer, [])

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
      loss = Nx.add(loss.(target, preds), Axon.penalty(model, params))
      {preds, loss}
    end

    step({init_fn, objective_fn}, optimizer, opts)
  end

  def step(%Axon{} = model, loss, optimizer, opts) when is_atom(loss) and is_list(opts) do
    loss_fn = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
    step(model, loss_fn, optimizer, opts)
  end

  @doc false
  def step(%Axon{} = model, train_state, loss, {_, _} = optimizer)
      when is_function(loss, 2) or is_atom(loss),
      do: step(model, train_state, loss, optimizer, [])

  @doc """
  Represents a single training step using an Axon `model`,
  initial state `train_state`, `loss` function and `optimizer`.

  The `loss` function is either an atom or a two arity anonymous
  function.
  """
  def step(%Axon{} = model, train_state, loss, optimizer, opts)
      when is_function(loss, 2) and is_list(opts) do
    init_fn = fn ->
      train_state
      |> Tuple.to_list()
      |> Enum.map(&Nx.tensor(&1, backend: Nx.Defn.Expr))
      |> List.to_tuple()
    end

    objective_fn = fn params, input, target ->
      preds = Axon.predict(model, params, input)
      Nx.add(loss.(target, preds), Axon.penalty(model, params))
    end

    step({init_fn, objective_fn}, optimizer, opts)
  end

  def step(%Axon{} = model, train_state, loss, optimizer, opts)
      when is_atom(loss) and is_list(opts) do
    loss_fn = &apply(Axon.Losses, loss, [&1, &2, [reduction: :mean]])
    step(model, train_state, loss_fn, optimizer, opts)
  end

  @valid_callbacks [:early_stopping]

  @doc false
  def callback(%Step{} = step, callback) when callback in @valid_callbacks do
    callback(step, callback, [])
  end

  @doc """
  Adds a callback from `Axon.Training.Callbacks` to the training step.
  """
  def callback(%Step{} = step, callback, opts) when callback in @valid_callbacks do
    fun = &apply(Axon.Training.Callbacks, callback, [&1, &2, &3])
    callback(step, fun, :all, opts)
  end

  @doc """
  Adds a callback function to the training step.

  Callback functions instrument specific points in the training loop.
  You can specify an `event` which is one of:

    - `:before_{train, epoch, batch}`
    - `:after_{train, epoch, batch}`

  The default `event` is `:all`, meaning the callback will run at every
  callback point.

  Callback functions have the following signature:

      callback_fn(train_state :: map, event :: atom, opts :: keyword) ::
        {:cont, train_state} | {:halt, train_state}

  You can trigger event-specific behavior using pattern matching:

      def my_callback(train_state, :before_epoch, _opts) do
        {:cont, %{train_state | my_metadata: 0}}
      end

      def my_callback(train_state, :after_epoch, _opts) do
        {:cont, %{train_state | my_metadata: train_state[:metadata] + 1}}
      end

      def my_callback(train_state, _event, _opts), do: {:cont, train_state}

  Returning `{:halt, train_state}` will immediately terminate the training loop:

      def early_stopping(train_state, :after_epoch, opts) do
        if stop?(train_state, opts) do
          {:halt, train_state}
        else
          {:cont, train_state}
        end
      end
  """
  def callback(%Step{callbacks: callbacks} = step, function, event \\ :all, opts \\ [])
      when is_function(function, 3) and is_atom(event) and is_list(opts) do
    %{step | callbacks: [{function, event, opts} | callbacks]}
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
  def train(
        %Step{init: init_fn, step: step_fn, callbacks: callbacks},
        inputs,
        targets,
        opts \\ []
      ) do
    epochs = opts[:epochs] || 5
    compiler = opts[:compiler] || Nx.Defn.Evaluator
    log_every = opts[:log_every] || 50

    callbacks = [
      {&Axon.Training.Callbacks.standard_io_logger(&1, &2, &3), :all, log_every: log_every}
      | Enum.reverse(callbacks)
    ]

    jit_opts = [compiler: compiler] ++ opts
    train_state = Nx.Defn.jit(init_fn, [], jit_opts)

    train_state =
      case apply_callback(callbacks, train_state, jit_opts, :before_train) do
        {:cont, train_state} ->
          Enum.reduce_while(1..epochs, train_state, fn epoch, train_state ->
            case apply_callback(callbacks, train_state, jit_opts, :before_epoch) do
              {:cont, train_state} ->
                {time, train_state} =
                  :timer.tc(&train_epoch/6, [
                    step_fn,
                    train_state,
                    inputs,
                    targets,
                    callbacks,
                    jit_opts
                  ])

                zero_metrics = Map.new(train_state[:metrics], fn {k, _} -> {k, 0.0} end)

                case apply_callback(
                       callbacks,
                       Map.put(train_state, :time, time),
                       jit_opts,
                       :after_epoch
                     ) do
                  {:cont, train_state} ->
                    train_state = %{
                      Map.delete(train_state, :time)
                      | metrics: zero_metrics,
                        epoch: epoch,
                        epoch_step: 0,
                        epoch_loss: 0.0
                    }

                    {:cont, train_state}

                  {:halt, train_state} ->
                    {:halt, train_state}
                end

              {:halt, train_state} ->
                {:halt, train_state}
            end
          end)

        {:halt, train_state} ->
          train_state
      end

    {_, train_state} = apply_callback(callbacks, train_state, jit_opts, :after_train)

    train_state
  end

  ## Helpers

  defp train_epoch(step_fn, train_state, inputs, targets, callbacks, opts) do
    dataset = Stream.zip(inputs, targets)

    Enum.reduce_while(dataset, train_state, fn {inp, tar}, train_state ->
      case apply_callback(callbacks, train_state, opts, :before_batch) do
        {:cont, train_state} ->
          train_state = Nx.Defn.jit(step_fn, [train_state, inp, tar], opts)
          apply_callback(callbacks, train_state, opts, :after_batch)

        {:halt, train_state} ->
          {:halt, train_state}
      end
    end)
  end

  defp apply_callback([], train_state, _, _), do: {:cont, train_state}

  defp apply_callback(callbacks, train_state, train_opts, event) do
    result =
      Enum.reduce_while(callbacks, train_state, fn
        {callback, :all, opts}, train_state ->
          case apply(callback, [train_state, event, opts ++ train_opts]) do
            {:halt, acc} ->
              {:halt, {:stopped, acc}}

            {:cont, acc} ->
              {:cont, acc}

            other ->
              raise "invalid return from callback #{inspect(other)}"
          end

        {callback, event, opts}, train_state ->
          case apply(callback, [train_state, event, opts ++ train_opts]) do
            {:halt, acc} ->
              {:halt, {:halt, acc}}

            {:cont, acc} ->
              {:cont, {:cont, acc}}

            other ->
              raise "invalid return from callback #{inspect(other)}"
          end
      end)

    case result do
      {:stopped, acc} ->
        {:halt, acc}

      acc ->
        {:cont, acc}
    end
  end
end
