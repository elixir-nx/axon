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
      start: {Axon.Serving, :start_link, [opts]}
    }
  end

  ## API

  @doc """
  Starts the Axon.Serving process.

  At a minimum you must provide `:model` which is a tuple of
  `{model, params}` and `:name` which is an atom used to
  uniquely identify the model serving process.

  ## Options

    * `:model` - tuple of `{model, params}` to compile and use

    * `:shape` - input shapes for each input in the model

    * `:batch_size` - maximum batch size to forward to model

    * `:batch_timeout` - auto-batching queue timeout limit

    All other options are treated as compilation options and
    forwarded to the defn compiler.
  """
  def start_link(opts) do
    {{model, params}, opts} = Keyword.pop!(opts, :model)
    {name, opts} = Keyword.pop!(opts, :name)
    {shape, opts} = Keyword.pop(opts, :shape)
    {batch_size, opts} = Keyword.pop(opts, :batch_size, 1)
    {batch_timeout, compiler_opts} = Keyword.pop(opts, :batch_timeout, 100)

    template = template!(model, shape, batch_size)

    {_init_fun, predict_fun} = Axon.compile(model, template, %{}, compiler_opts)

    config = %{
      model: predict_fun,
      params: params,
      batch_size: batch_size,
      batch_timeout: batch_timeout
    }

    GenServer.start_link(__MODULE__, config, name: name)
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

  @impl true
  def handle_call({:predict, input}, from, state) do
    %{queue: queue, count: count} = state
    queue = :queue.in({input, from}, queue)

    # TODO: if Nx.axis_size(input, 0) is more than batch_size,
    # we should return error.
    # TODO: correctly handle counts/batch for all inputs

    [input] = Map.values(input)

    state =
      %{state | queue: queue, count: count + Nx.axis_size(input, 0)}
      |> maybe_start_timer()
      |> maybe_dispatch()

    {:noreply, state}
  end

  @impl true
  def handle_info(:timeout, state) do
    {:noreply, dispatch(state)}
  end

  defp template!(%Axon{} = model, config_shapes, batch_size) do
    model
    |> Axon.get_inputs()
    |> Map.new(fn {name, axon_shape} ->
      config_shape = config_shapes[name]

      shape =
        cond do
          config_shape != nil and same_shape?(axon_shape, config_shape) and
              valid_shape?(config_shape, batch_size) ->
            put_elem(config_shape, 0, batch_size)

          valid_shape?(axon_shape, batch_size) ->
            put_elem(axon_shape, 0, batch_size)

          true ->
            raise ArgumentError, "invalid shape for input #{name}"
        end

      # TODO: preserve input type
      {name, Nx.template(shape, :f32)}
    end)
  end

  defp same_shape?(shape1, shape2) do
    shape1
    |> Tuple.to_list()
    |> Enum.zip_with(Tuple.to_list(shape2), fn s1, s2 ->
      s1 == nil or s2 == nil or s1 == s2
    end)
    |> Enum.all?()
  end

  defp valid_shape?(shape, batch_size) do
    shape != nil and num_nil_dims(shape) <= 1 and
      (elem(shape, 0) == nil or elem(shape, 0) == batch_size)
  end

  defp num_nil_dims(shape) do
    shape |> Tuple.to_list() |> Enum.count(&(&1 == nil))
  end

  defp maybe_start_timer(%{timeout_ref: nil, count: count} = state) when count > 0 do
    ref = Process.send_after(self(), :timeout, state.config.batch_timeout)
    %{state | timeout_ref: ref}
  end

  defp maybe_start_timer(%{timeout_ref: nil} = state) do
    state
  end

  defp maybe_start_timer(%{timeout_ref: ref} = state) when is_reference(ref) do
    state
  end

  defp maybe_dispatch(%{count: count, config: %{batch_size: batch_size}} = state)
       when count >= batch_size do
    dispatch(state)
  end

  defp maybe_dispatch(state) do
    state
  end

  # TODO: Make this work with batches != 1
  defp dispatch(state) do
    %{queue: queue, count: count, timeout_ref: timeout_ref} = state
    cancel_timer(timeout_ref)

    batch_size = min(count, state.config.batch_size)

    {now, queue} =
      Enum.map_reduce(1..batch_size, queue, fn _, queue ->
        {{:value, {input, _} = pair}, queue} = :queue.out(queue)
        # TODO: handle with counts per input
        [input] = Map.values(input)
        1 = Nx.axis_size(input, 0)
        {pair, queue}
      end)

    {inputs, froms} = Enum.unzip(now)

    # TODO: Do we want to start a process or not?
    inputs
    |> concat_inputs()
    |> maybe_pad(batch_size, state.config.batch_size)
    |> then(&state.config.model.(state.config.params, &1))
    |> maybe_pad(state.config.batch_size, batch_size)
    |> Nx.to_batched(1)
    |> Enum.zip_with(froms, fn tensor, from ->
      GenServer.reply(from, tensor)
    end)

    %{state | count: count - batch_size, queue: queue, timeout_ref: nil}
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

  defp concat_inputs(input_map) do
    input_map
    |> Enum.reduce(%{}, fn batch, inputs ->
      batch
      |> Enum.reduce(inputs, fn {name, tensor}, acc ->
        Map.update(acc, name, [tensor], fn tensors -> [tensor | tensors] end)
      end)
    end)
    |> Map.new(fn {key, tensors} -> {key, Nx.concatenate(tensors, axis: 0)} end)
  end

  defp maybe_pad(input, current_batch_size, desired_batch_size)
       when current_batch_size == desired_batch_size do
    input
  end

  defp maybe_pad(%Nx.Tensor{} = tensor, current_batch_size, desired_batch_size) do
    pad_size = desired_batch_size - current_batch_size
    do_pad(tensor, pad_size)
  end

  defp maybe_pad(inputs, current_batch_size, desired_batch_size) when is_map(inputs) do
    pad_size = desired_batch_size - current_batch_size
    Map.new(inputs, fn {name, tensor} -> {name, do_pad(tensor, pad_size)} end)
  end

  defp do_pad(tensor, pad_size) do
    first_axis_pad = {0, pad_size, 0}
    rest_axes_pad = List.duplicate({0, 0, 0}, Nx.rank(tensor) - 1)
    Nx.pad(tensor, 0.0, [first_axis_pad | rest_axes_pad])
  end
end
