defmodule Axon.Loop.Fused do
  @moduledoc false

  # Fuses many iterations of a loop into a single computation.
  #
  # The unfused loop dispatches one computation per batch. Every iteration pays
  # for a call into the compiler's runtime, and the step state (model
  # parameters, optimizer state) is handed back to the host and passed in again.
  # For small models that overhead is far larger than the training step itself.
  #
  # A fused loop instead moves `:fuse` batches to the device at once and runs
  # them in a device-side `while` loop which carries the step state and the
  # metrics, so the loop only synchronizes with the host once every `:fuse`
  # iterations. The same number of bytes crosses to the device either way, they
  # just cross in one transfer instead of one per batch.
  #
  # Iteration events cannot fire from inside the computation: a host callback
  # per iteration costs more than the dispatch it would save. Instead, the
  # metrics of every iteration in a chunk are recorded into a device-side buffer
  # and the events are fired on the host, in order, once the chunk is done.

  require Logger

  import Nx.Defn

  alias Axon.Loop.State
  alias Nx.Defn.Composite

  @default_chunk 32

  @doc """
  Validates and normalizes the `:fuse` option into a chunk size.
  """
  def chunk_size!(true), do: @default_chunk

  def chunk_size!(chunk) when is_integer(chunk) and chunk > 0, do: chunk

  def chunk_size!(invalid) do
    raise ArgumentError,
          "invalid value #{inspect(invalid)} for :fuse, expected true or a positive" <>
            " integer of the number of batches to fuse into a single computation"
  end

  ## Fused chunk

  defnp chunk(step_state, metrics, batches, log, counters, opts \\ []) do
    opts = keyword!(opts, [:batch_fn, :record?])
    {iteration, count} = counters

    while {step_state, metrics, batches, log, i = Nx.tensor(0, type: :s32), iteration, count},
          i < count do
      batch = index_batch(batches, i)
      {step_state, metrics} = apply_batch_fn(opts, batch, iteration + i, step_state, metrics)
      log = record_metrics(opts, log, metrics, i)
      {step_state, metrics, batches, log, i + 1, iteration, count}
    end
  end

  deftransformp index_batch(batches, i) do
    Composite.traverse(batches, & &1[i])
  end

  deftransformp apply_batch_fn(opts, batch, iteration, step_state, metrics) do
    opts[:batch_fn].(batch, iteration, step_state, metrics)
  end

  deftransformp record_metrics(opts, log, metrics, i) do
    if opts[:record?] do
      leaves = Composite.flatten_list([metrics])

      {log, []} =
        Composite.traverse(log, leaves, fn buffer, [metric | leaves] ->
          start = [i | List.duplicate(0, Nx.rank(metric))]
          {Nx.put_slice(buffer, start, Nx.new_axis(metric, 0)), leaves}
        end)

      log
    else
      log
    end
  end

  ## Loop integration

  @doc """
  Builds the function which runs a single fused chunk.

  `batch_fn` is the same batch function the unfused loop uses, so both paths run
  exactly the same step and metric computations.
  """
  def build_chunk_fn(batch_fn, record?) do
    opts = [batch_fn: batch_fn, record?: record?]

    fn step_state, metrics, batches, log, counters ->
      chunk(step_state, metrics, batches, log, counters, opts)
    end
  end

  @doc """
  Runs a single epoch as a series of fused chunks.

  Mirrors the contract of the unfused epoch runner: returns the loop status, the
  now compiled chunk function, and the updated loop state.
  """
  def run_epoch(chunk_fn, loop_state, data, debug?, ctx) do
    reducer = fn batch, _acc -> {:suspend, batch} end
    cont = &Enumerable.reduce(data, &1, reducer)

    run_chunks(chunk_fn, loop_state, cont, debug?, ctx)
  end

  defp run_chunks(chunk_fn, loop_state, cont, debug?, ctx) do
    %State{iteration: iteration, max_iteration: max_iteration} = loop_state

    remaining =
      if max_iteration > 0, do: max(max_iteration - iteration, 0), else: ctx.chunk

    case take_batches(cont, min(ctx.chunk, remaining)) do
      {[], _cont} ->
        {:continue, chunk_fn, loop_state}

      {batches, cont} ->
        metrics = loop_state.metrics
        {chunk_fn, loop_state, log} = run_chunk(chunk_fn, loop_state, batches, ctx, debug?)

        {status, loop_state} =
          fire_iteration_events(loop_state, metrics, log, length(batches), ctx)

        cond do
          status != :continue ->
            halt_enumeration(cont)
            {status, chunk_fn, loop_state}

          # A chunk which could not be filled means the data is exhausted
          cont == nil ->
            {:continue, chunk_fn, loop_state}

          max_iteration > 0 and loop_state.iteration >= max_iteration ->
            halt_enumeration(cont)
            {:continue, chunk_fn, loop_state}

          true ->
            if ctx.gc?, do: :erlang.garbage_collect()
            run_chunks(chunk_fn, loop_state, cont, debug?, ctx)
        end
    end
  end

  defp run_chunk(chunk_fn, loop_state, batches, ctx, debug?) do
    %State{step_state: step_state, metrics: metrics, iteration: iteration} = loop_state

    count = length(batches)
    stacked = stack_batches(batches, ctx.chunk)
    log = new_log(metrics, ctx)

    counters = {Nx.tensor(iteration, type: :s32), Nx.tensor(count, type: :s32)}
    args = [step_state, metrics, stacked, log, counters]

    {:compiled, fun} = chunk_fn = compile_chunk_fn(chunk_fn, args)

    if debug? do
      Logger.debug("Axon.Loop started fused chunk of #{count} iterations")
    end

    {time, {step_state, metrics, _batches, log, _i, _iteration, _count}} =
      :timer.tc(fn -> apply(fun, args) end)

    if debug? do
      Logger.debug(
        "Axon.Loop finished fused chunk in #{Float.round(time / 1000, 1)}ms" <>
          " (#{Float.round(time / count, 1)}us/iteration)"
      )
    end

    loop_state = %{
      loop_state
      | step_state: step_state,
        metrics: metrics,
        iteration: iteration + count
    }

    {chunk_fn, loop_state, log}
  end

  # Replays the iteration events of a finished chunk on the host, in the order
  # the unfused loop fires them. Handlers see the metrics as they were at that
  # iteration, but no step state, since it only exists on the device during a
  # chunk.
  defp fire_iteration_events(loop_state, _metrics, _log, _count, %{events?: false}) do
    {:continue, loop_state}
  end

  defp fire_iteration_events(loop_state, metrics, log, count, ctx) do
    log = Nx.backend_transfer(log, Nx.BinaryBackend)
    first_iteration = loop_state.iteration - count
    step_state = loop_state.step_state
    loop_state = %{loop_state | metrics: metrics}

    result =
      Enum.reduce_while(0..(count - 1)//1, {:continue, loop_state}, fn i, {_, loop_state} ->
        iteration = first_iteration + i
        loop_state = %{loop_state | iteration: iteration, step_state: nil}

        with {:continue, loop_state} <- ctx.fire_fn.(:iteration_started, loop_state),
             loop_state = %{loop_state | metrics: metrics_at(log, metrics, i)},
             {:continue, loop_state} <- ctx.fire_fn.(:iteration_completed, loop_state) do
          {:cont, {:continue, %{loop_state | iteration: iteration + 1}}}
        else
          {status, loop_state} -> {:halt, {status, loop_state}}
        end
      end)

    {status, loop_state} = result
    {status, %{loop_state | step_state: step_state}}
  end

  defp new_log(_metrics, %{events?: false}), do: %{}

  defp new_log(metrics, ctx) do
    Composite.traverse(metrics, fn metric ->
      Nx.broadcast(
        Nx.tensor(0, type: Nx.type(metric)),
        Tuple.insert_at(Nx.shape(metric), 0, ctx.chunk)
      )
    end)
  end

  defp metrics_at(log, metrics, i) do
    leaves = Composite.flatten_list([log])

    {metrics, []} =
      Composite.traverse(metrics, leaves, fn _, [buffer | rest] -> {buffer[i], rest} end)

    metrics
  end

  # Stacks a chunk of batches into one container of tensors with the chunk as
  # their leading axis. A chunk which ends the data is padded to a full chunk,
  # so that every chunk has the same shape and the loop compiles only once.
  defp stack_batches(batches, chunk) do
    batches =
      case chunk - length(batches) do
        0 -> batches
        pad -> batches ++ List.duplicate(List.last(batches), pad)
      end

    stacked =
      batches
      |> Enum.map(&Composite.flatten_list([&1]))
      |> Enum.zip_with(&stack_leaves/1)

    {batches, []} =
      Composite.traverse(hd(batches), stacked, fn _, [leaf | rest] -> {leaf, rest} end)

    batches
  end

  defp stack_leaves(leaves) do
    Nx.stack(leaves)
  rescue
    exception in ArgumentError ->
      reraise ArgumentError,
              [
                message:
                  "cannot fuse a loop over batches of different shapes or types, all" <>
                    " batches must match because a fused loop stacks them into a single" <>
                    " tensor, got: #{Exception.message(exception)}"
              ],
              __STACKTRACE__
  end

  defp take_batches(cont, count), do: take_batches(cont, count, [])

  defp take_batches(cont, 0, acc), do: {Enum.reverse(acc), cont}

  defp take_batches(cont, count, acc) do
    case cont.({:cont, nil}) do
      {:suspended, batch, cont} -> take_batches(cont, count - 1, [batch | acc])
      {:done, _} -> {Enum.reverse(acc), nil}
      {:halted, _} -> {Enum.reverse(acc), nil}
    end
  end

  # Enumerables which hold resources release them when their suspended
  # continuation is halted, so an epoch which stops before its data is
  # exhausted must halt it.
  defp halt_enumeration(nil), do: :ok

  defp halt_enumeration(cont) do
    _ = cont.({:halt, nil})
    :ok
  end

  defp compile_chunk_fn({:non_compiled, fun, strict?, jit_opts}, args) do
    fun =
      if strict? do
        Nx.Defn.compile(fun, args, jit_opts)
      else
        Nx.Defn.jit(fun, jit_opts)
      end

    {:compiled, fun}
  end

  defp compile_chunk_fn({:compiled, _} = chunk_fn, _args), do: chunk_fn
end
