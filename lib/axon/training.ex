defmodule Axon.Training do
  @moduledoc """
  Abstractions for training machine learning models.
  """
  require Axon
  require Axon.Updates
  require Logger

  import Axon.Shared

  alias Axon.Training.Loop
  alias Axon.Training.Process
  alias Axon.Training.State

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

  @doc """
  Creates a supervised training step from a model, loss function,
  and optimizer.
  """
  # TODO(seanmor5): Bikeshed on step v train_step?
  # TODO(seanmor5): Handle more patterns for creating training step
  def step(%Axon{} = model, loss, {init_optimizer_fn, update_fn}) do
    init_fn = fn ->
      model_state = Axon.init(model)
      optimizer_state = init_optimizer_fn.(model_state)

      %{
        predictions: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        loss: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        model_state: model_state,
        optimizer_state: optimizer_state
      }
    end

    objective_fn = fn state, inp, tar ->
      y_pred = Axon.predict(model, state, inp)
      {y_pred, apply(Axon.Losses, loss, [tar, y_pred, [reduction: :mean]])}
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

      {updates, new_optimizer_state} = update_fn.(gradients, optimizer_state, model_state)

      updates = deep_merge(updates, model_state, fn g, x -> Nx.as_type(g, Nx.type(x)) end)

      %{
        predictions: preds,
        loss: new_loss,
        model_state: Axon.Updates.apply_updates(model_state, updates),
        optimizer_state: new_optimizer_state
      }
    end

    %Loop{process: %Process{init: init_fn, step: step_fn}}
    |> handle(:iteration_completed, &log_iteration_metrics/1)
    |> handle(:epoch_completed, &log_epoch_metrics/1)
  end

  @doc """
  Adds a metric to the given loop.
  """
  def metric(%Loop{metrics: metric_fns} = loop, metric, name) when is_atom(metric) do
    case metric_fns do
      %{^name => _} ->
        Logger.warning(
          "Metric #{name} declared twice in loop. Original metric will be overriden."
        )

      _ ->
        :ok
    end

    metric_fn = &apply(Axon.Metrics, metric, [&1, &2])
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
    {verbosity, opts} = Keyword.pop(opts, :verbosity, 2)
    {compiler, jit_opts} = Keyword.pop(opts, :compiler)

    %Loop{process: process, handlers: handler_fns, metrics: metric_fns} = loop

    %Process{init: init_fn, step: step_fn} = process

    metrics = Map.new(metric_fns, fn {k, _} -> {k, Nx.tensor(0.0)} end)
    process_state = maybe_jit(init_fn, [], compiler, jit_opts)
    loop_state = %State{epoch: 1, iteration: 0, process_state: process_state, metrics: metrics}

    Enum.reduce_while(1..max_epochs, loop_state, fn epoch, loop_state ->
      fire_event(:epoch_started, handler_fns, loop_state)

      {time, loop_state} =
        :timer.tc(&run_epoch/7, [
          step_fn,
          metric_fns,
          handler_fns,
          loop_state,
          data,
          compiler,
          jit_opts
        ])

      fire_event(:epoch_completed, handler_fns, loop_state)

      {:cont, %State{loop_state | epoch: epoch + 1, iteration: 0}}
    end)
  end

  ## Helpers

  defp fire_event(event, handler_fns, state) do
    Enum.each(handler_fns[event], & &1.(state))
  end

  defp run_epoch(step_fn, metric_fns, handler_fns, loop_state, data, compiler, jit_opts) do
    Enum.reduce_while(data, loop_state, fn data, state ->
      fire_event(:iteration_started, handler_fns, state)

      batch_fn = build_batch_fn(step_fn, metric_fns)
      new_state = maybe_jit(batch_fn, [state, data], compiler, jit_opts)

      fire_event(:iteration_completed, handler_fns, new_state)

      {:cont, new_state}
    end)
  end

  defp build_batch_fn(step_fn, metric_fns) do
    fn state, {_, tar} = data ->
      %State{metrics: metrics, iteration: iter} = state
      %{predictions: preds} = new_process_state = step_fn.(state, data)

      new_metrics =
        metrics
        |> Enum.zip_with(metric_fns, fn {k, avg}, {k, v} ->
          {k, running_average(avg, v.(preds, tar), iter)}
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

  # TODO(seanmor5): This should be a defn combinator with other running averages/metrics
  # stuff.
  defp running_average(avg, new_data, i) do
    avg
    |> Nx.multiply(i)
    |> Nx.add(new_data)
    |> Nx.divide(Nx.add(i, 1))
  end

  defp maybe_jit(fun, args, compiler, jit_opts) do
    if compiler do
      Nx.Defn.jit(fun, args, [compiler: compiler] ++ jit_opts)
    else
      fun.(args)
    end
  end

  defp log_iteration_metrics(%State{
         epoch: epoch,
         iteration: iter,
         metrics: metrics,
         process_state: process_state
       }) do
    %{loss: loss} = process_state

    iter = Nx.to_scalar(iter)
    epoch = Nx.to_scalar(epoch)

    if rem(iter, 50) == 0 do
      loss = "Loss: #{:io_lib.format("~.5f", [Nx.to_scalar(loss)])}"

      metrics =
        metrics
        |> Enum.map(fn {k, v} -> "#{k}: #{:io_lib.format("~.5f", [Nx.to_scalar(v)])}" end)
        |> Enum.join(" ")

      IO.write("\rEpoch #{epoch}, batch #{iter} - #{loss} #{metrics}")
    end
  end

  defp log_epoch_metrics(%State{epoch: epoch, process_state: process_state}) do
    %{loss: loss} = process_state
    IO.write("\n\n")
    IO.write("Epoch #{inspect(Nx.to_scalar(epoch))}")
    IO.write(" - Loss: #{:io_lib.format("~.5f", [Nx.to_scalar(loss)])}")
    # IO.write(" Time: #{:io_lib.format("~.5f", [time * 1.0e-6])}s")
    IO.write("\n\n")
  end
end
