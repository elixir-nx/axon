defmodule Axon.Serving do
  use GenServer

  @doc false
  def child_spec(opts) when is_list(opts) do
    id =
      case Keyword.get(opts, :name, Axon.Serving) do
        name when is_atom(name) -> name
        {:global, name} -> name
        {:via, _module, name} -> name
      end

    %{
      id: id,
      start: {Axon.Serving, :start_link, [opts]},
    }
  end

  ## API

  @doc """
  ## Options
  """
  def start_link(opts) do
    config = %{
      batch_size: Keyword.fetch!(opts, :batch_size, 1),
      batch_timeout: Keyword.fetch!(opts, :batch_timeout, 100),
      # model: Keyword.fetch!(opts, :model)
    }

    GenServer.start_link(__MODULE__, config, opts)
  end

  def predict(server, input) do
    GenServer.call(server, {:predict, input}, :infinity)
  end

  ## Callbacks

  @impl true
  def init(config) do
    state = %{
      queue: :queue.new(),
      count: 0,
      timeout_ref: nil,
      config: config
    }

    {:ok, state}
  end

  def handle_call({:predict, input}, from, state) do
    %{queue: queue, count: count} = state
    queue = :queue.in({input, from}, queue)

    # TODO: if Nx.axis_size(input, 0) is more than batch_size,
    # we should return error.

    state =
      %{state | acc: queue, count: count + Nx.axis_size(input, 0)}
      |> maybe_start_timer()
      |> maybe_dispatch()

    {:noreply, state}
  end

  def handle_info(:timeout, state) do
    dispatch(state)
  end

  defp maybe_start_timer(%{timeout_ref: nil, count: count} = state) when count > 0 do
    ref = Process.send_after(self(), :timeout, state.config.batch_timeout)
    %{state | timeout_ref: ref}
  end

  defp maybe_start_timer(%{timeout_ref: ref} = state) when is_reference(ref) do
    state
  end

  defp maybe_dispatch(%{count: count, config: %{batch_size: batch_size} = state)
       when count >= batch_size do
    dispatch(state)
  end

  defp maybe_dispatch(state) do
    state
  end

  # TODO: Make this work with batches != 1
  defp dispatch(state) do
    %{batch_size: batch_size, queue: queue, count: count, timeout_ref: timeout_ref} = state
    cancel_timer(timeout_ref)

    {now, queue} =
      Enum.map_reduce(1..batch_size, state.queue, fn _, queue ->
        {{:value, {input, from}}, queue} = :queue.out(queue)
        1 = Nx.axis_size(input)
        {pair, queue}
      end)

    {inputs, froms} = Enum.unzip(now)

    # TODO: Do we want to start a process or not?
    inputs
    |> Nx.concatenate()
    |> state.config.model.()
    |> Nx.to_batched(batch, 1)
    |> Enum.zip_with(froms, fn tensor, from ->
      GenServer.reply(from, tensor)
    end)

    %{state | count: count - min(count, batch_size), queue: queue, timeout_ref: nil}
    |> maybe_start_timer()
    |> maybe_dispatch()
  end

  defp cancel_timer(timeout_ref) do
    Process.cancel_timer(timeout_ref)

    receive do
      :timeout -> :ok
    after
      0 -> :ok
    end
  end
end
