defmodule Axon.Loop do
  @moduledoc """
  Abstractions for training machine learning models.
  """
  require Axon
  require Axon.Updates
  require Logger

  import Axon.Shared

  alias __MODULE__, as: Loop
  alias Axon.Loop.Process
  alias Axon.Loop.State

  @all_events [
    :started,
    :epoch_started,
    :iteration_started,
    :iteration_completed,
    :epoch_completed,
    :epoch_terminated,
    :terminated,
    :completed
  ]

  @default_handlers %{
    started: [],
    epoch_started: [],
    iteration_started: [],
    iteration_completed: [],
    epoch_completed: [],
    epoch_terminated: [],
    terminated: [],
    completed: []
  }

  defstruct [:process, metrics: %{}, handlers: @default_handlers]

  ## Loop Factories

  @doc """
  Creates a supervised trainer from a model, loss function,
  and optimizer.
  """
  def trainer(model, loss, optimizer) do
    {init_model_fn, forward_model_fn} = build_model_fns(model)
    loss_fn = build_loss_fn(loss)
    {init_optimizer_fn, update_optimizer_fn} = build_optimizer_fns(optimizer)

    init_fn = fn ->
      model_state = init_model_fn.()
      optimizer_state = init_optimizer_fn.(model_state)

      %{
        y_true: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        y_pred: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        loss: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        model_state: model_state,
        optimizer_state: optimizer_state
      }
    end

    objective_fn = fn state, inp, tar ->
      y_pred = forward_model_fn.(state, inp)
      {y_pred, loss_fn.(tar, y_pred)}
    end

    step_fn = fn state, {inp, tar} ->
      %State{process_state: process_state, iteration: iter} = state
      %{model_state: model_state, optimizer_state: optimizer_state, loss: loss} = process_state

      {{preds, batch_loss}, gradients} =
        Nx.Defn.value_and_grad(
          model_state,
          &objective_fn.(&1, inp, tar),
          fn x -> elem(x, 1) end
        )

      new_loss = running_average(loss, batch_loss, iter)

      {updates, new_optimizer_state} =
        update_optimizer_fn.(gradients, optimizer_state, model_state)

      updates = deep_merge(updates, model_state, fn g, x -> Nx.as_type(g, Nx.type(x)) end)

      %{
        y_true: tar,
        y_pred: preds,
        loss: new_loss,
        model_state: Axon.Updates.apply_updates(model_state, updates),
        optimizer_state: new_optimizer_state
      }
    end

    %Loop{process: %Process{init: init_fn, step: step_fn}}
  end

  @doc """
  Creates a supervised evaluator from a model and model state.
  """
  def evaluator(model, model_state) do
    init_fn = fn ->
      %{
        y_true: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        y_pred: Nx.tensor(0.0, backend: Nx.Defn.Expr)
      }
    end

    step_fn = fn state, {inp, tar} ->
      %{
        y_true: tar,
        y_pred: Axon.predict(model, model_state, inp)
      }
    end

    %Loop{process: %Process{init: init_fn, step: step_fn}}
  end

  @doc """
  Adds a metric to the given loop.
  """
  def metric(
        %Loop{metrics: metric_fns} = loop,
        metric,
        name,
        transform_or_fields \\ [:y_true, :y_pred]
      ) do
    case metric_fns do
      %{^name => _} ->
        Logger.warning(
          "Metric #{name} declared twice in loop. Original metric will be overriden."
        )

      _ ->
        :ok
    end

    metric_fn = build_metric_fn(metric, transform_or_fields)
    %Loop{loop | metrics: Map.put(metric_fns, name, metric_fn)}
  end

  @doc """
  Adds a handler to the given loop.
  """
  # TODO(seanmor5): Bikeshed on name
  # TODO(seanmor5): Handle bad event names gracefully
  # TODO(seanmor5): Add event filters
  def handle(%Loop{handlers: handle_fns} = loop, event, handler) do
    add_event_handler = fn event, handle_fns ->
      Map.update!(handle_fns, event, fn event_funs -> [handler | event_funs] end)
    end

    handler_fns =
      case event do
        [_ | _] = events ->
          Enum.reduce(events, handle_fns, add_event_handler)

        :all ->
          Enum.reduce(@all_events, handle_fns, add_event_handler)

        event when is_atom(event) ->
          add_event_handler.(event, handle_fns)
      end

    %Loop{loop | handlers: handler_fns}
  end

  @doc """
  Runs the given loop.
  """
  def run(loop, data, opts \\ []) do
    {max_epochs, opts} = Keyword.pop(opts, :epochs, 1)
    {max_iterations, opts} = Keyword.pop(opts, :iterations)
    {compiler, jit_opts} = Keyword.pop(opts, :compiler)

    %Loop{process: process, handlers: handler_fns, metrics: metric_fns} = loop

    %Process{init: init_fn, step: step_fn} = process

    metrics = Map.new(metric_fns, fn {k, _} -> {k, Nx.tensor(0.0)} end)
    process_state = maybe_jit(init_fn, [], compiler, jit_opts)

    loop_state = %State{
      epoch: 1,
      iteration: 0,
      process_state: process_state,
      metrics: metrics,
      times: %{}
    }

    Enum.reduce_while(1..max_epochs, loop_state, fn epoch, loop_state ->
      fire_event(:epoch_started, handler_fns, loop_state)

      {time, loop_state} =
        :timer.tc(&run_epoch/8, [
          step_fn,
          metric_fns,
          handler_fns,
          loop_state,
          data,
          max_iterations,
          compiler,
          jit_opts
        ])

      new_times = Map.put(loop_state.times, Nx.to_scalar(epoch), time)
      new_loop_state = %State{loop_state | times: new_times}

      fire_event(:epoch_completed, handler_fns, new_loop_state)

      {:cont, %State{new_loop_state | epoch: epoch + 1, iteration: 0}}
    end)
  end

  ## Helpers

  defp run_epoch(
         step_fn,
         metric_fns,
         handler_fns,
         loop_state,
         data,
         max_iterations,
         compiler,
         jit_opts
       ) do
    Enum.reduce_while(data, loop_state, fn data, state ->
      fire_event(:iteration_started, handler_fns, state)

      batch_fn = build_batch_fn(step_fn, metric_fns)

      %State{iteration: iters} =
        new_state = maybe_jit(batch_fn, [state, data], compiler, jit_opts)

      fire_event(:iteration_completed, handler_fns, new_state)

      if Nx.to_scalar(iters) >= max_iterations do
        {:halt, new_state}
      else
        {:cont, new_state}
      end
    end)
  end

  defp fire_event(event, handler_fns, state) do
    Enum.each(handler_fns[event], & &1.(state))
  end

  defp build_batch_fn(step_fn, metric_fns) do
    fn state, data ->
      %State{metrics: metrics, iteration: iter} = state
      new_process_state = step_fn.(state, data)

      new_metrics =
        metrics
        |> Enum.zip_with(metric_fns, fn {k, avg}, {k, v} ->
          {k, running_average(avg, v.(new_process_state), iter)}
        end)
        |> Map.new()

      %State{
        state
        | iteration: Nx.add(iter, 1),
          process_state: new_process_state,
          metrics: new_metrics
      }
    end
  end

  # Builds a loss function from an atom, function, or list of. Valid loss
  # functions must be one of an atom matching the name of a function in
  # Axon.Losses, an arity-2 function of the form loss(y_true, y_pred),
  # or a list of 2-tuples of {loss, weight} for constructing a simple
  # joint, multi-objective loss function.
  # TODO(seanmor5): Configurable per-batch reductions
  # TODO(seanmor5): Configurable multi-objective reductions
  # TODO(seanmor5): Should we trace custom loss functions and provide a
  # more clear error if the output shape is wrong?
  defp build_loss_fn(loss) do
    case loss do
      loss_name when is_atom(loss_name) ->
        &apply(Axon.Losses, loss_name, [&1, &2, [reduction: :mean]])

      loss_fn when is_function(loss, 2) ->
        loss_fn

      [{_, _} | _] = losses ->
        fn y_true, y_pred ->
          {_, loss} =
            Enum.reduce(losses, {0, Nx.tensor(0, backend: Nx.Defn.Expr)}, fn {loss, weight},
                                                                             {i, acc_loss} ->
              loss_fn = build_loss_fn(loss)

              y_true_i = elem(y_true, i)
              y_pred_i = elem(y_pred, i)

              new_acc_loss =
                y_true_i
                |> loss_fn.(y_pred_i)
                |> Nx.multiply(weight)
                |> Nx.add(acc_loss)

              {i + 1, new_acc_loss}
            end)

          loss
        end

      invalid ->
        raise ArgumentError,
              "Invalid loss function #{inspect(invalid)}, a valid loss" <>
                " function is an atom which matches a function in Axon.Losses," <>
                " an arity-2 function of the form loss(y_true, y_pred), or a list" <>
                " of 2-tuples of {loss, weight} for multi-objective models"
    end
  end

  # Builds model init and forward functions from an Axon struct
  # or a tuple of init / forward functions. Model functions are
  # essentially just model init / apply functions.
  defp build_model_fns(%Axon{} = model) do
    Axon.compile(model)
  end

  defp build_model_fns({init_fn, forward_fn})
       when is_function(init_fn, 0) and is_function(forward_fn, 2) do
    {init_fn, forward_fn}
  end

  defp build_model_fns(invalid) do
    raise ArgumentError,
          "Invalid model #{inspect(invalid)}, a valid model" <>
            " is an Axon struct, or a tuple of {init_fn, forward_fn}" <>
            " with signatures init_fn() :: model_state, forward_fn(" <>
            "model_state, inp) :: prediction"
  end

  # Builds optimizer init and update functions either from an atom
  # or a tuple of init / update functions. The init and update functions
  # match the signatures of those defined in Axon.Updates. If the
  # optimizer is an atom, it must match the name of a function in
  # Axon.Optimizers.
  defp build_optimizer_fns(optimizer) when is_atom(optimizer) do
    # TODO(seanmor5): Fall back to optimizer defaults rather
    # than this global default.
    apply(Axon.Optimizers, optimizer, [1.0e-2])
  end

  defp build_optimizer_fns({init_optimizer_fn, update_optimizer_fn})
       when is_function(init_optimizer_fn, 1) and is_function(update_optimizer_fn, 3) do
    {init_optimizer_fn, update_optimizer_fn}
  end

  defp build_optimizer_fns(invalid) do
    raise ArgumentError,
          "Invalid optimizer #{inspect(invalid)}, a valid optimizer" <>
            " is an atom matching the name of an optimizer in Axon.Optimizers" <>
            " or a tuple of {init_fn, update_fn}. See Axon.Updates for more" <>
            " information on building optimizers using the low-level API"
  end

  # Builds a metric function from an atom or function and an output transform.
  # A valid metric is an atom which matches the name of a function in
  # Axon.Metrics or a function which takes an arbitrary number of parameters
  # and returns an output of arbitrary shape/type. Output transforms are field(s)
  # to extract from the process state, or a function which transforms the process
  # state before it is passed to the metric function.
  defp build_metric_fn(metric, transform_or_fields) do
    transform_fn =
      case transform_or_fields do
        [_ | _] = fields ->
          fn output ->
            # TODO(seanmor5): Assert map to raise a clear error
            fields
            |> Enum.reduce([], fn field, acc -> [output[field] | acc] end)
            |> Enum.reverse()
          end

        field when is_atom(field) ->
          fn output ->
            # TODO(seanmor5): Assert map
            output[field]
          end

        transform when is_function(transform, 1) ->
          transform

        invalid ->
          raise ArgumentError,
                "Invalid output transform #{inspect(invalid)}, a valid output" <>
                  " transform is an atom or list of atoms specifying field(s)" <>
                  " to extract from the process state, or an arity-1 function" <>
                  " applied to the process state"
      end

    case metric do
      metric when is_atom(metric) ->
        fn output ->
          # TODO(seanmor5): Flatten all containers
          output
          |> transform_fn.()
          |> then(&apply(Axon.Metrics, metric, &1))
        end

      metric_fn when is_function(metric) ->
        fn output ->
          output
          |> transform_fn.()
          |> metric_fn.()
        end

      invalid ->
        raise ArgumentError,
              "Invalid metric #{inspect(invalid)}, a valid metric" <>
                " is an atom which matches the name of a function in" <>
                " Axon.Metrics or a function which takes a transformed" <>
                " process state and returns a value"
    end
  end

  # JIT-compiles the given function if the given compiler is a
  # valid defn compiler, otherwise applies the function with
  # the given arguments.
  defp maybe_jit(fun, args, compiler, jit_opts) do
    if compiler do
      Nx.Defn.jit(fun, args, [compiler: compiler] ++ jit_opts)
    else
      fun.(args)
    end
  end
end
