defmodule Axon.Serving do
  @moduledoc """
  Module for performing dynamic batch inference in production settings.

  `Axon.Serving` implements queue-based dynamic batch inference in order
  to handle concurrent inference requests between multiple processes.
  Typically, you start an `Axon.Serving` process as a part of your
  application's supervision tree:

      children = [
        {Axon.Serving,
          name: :my_model,
          model: MyApp.Model.load_model(),
          shape: MyApp.Model.input(),
          batch_size: 32,
          batch_timeout: 50,
          compiler: EXLA},
        MyApp.Repo,
        MyApp.Endpoint
      ]

      Supervisor.start_link(children, strategy: :one_for_one)

  `Axon.Serving` will compile your model with the given defn compiler
  specialized to the given input shapes. You may then invoke the model
  using `Axon.Serving.predict/2`:

      Axon.Serving.predict(:my_model, inputs)

  `Axon.Serving` will batch overlapping requests which reach the queue
  in the same window. The size of the window is given by `batch_timeout`.
  In the above example, `batch_timeout: 50` indicates the queue will wait
  50 ms to receive a total of `:batch_size` inputs. If the queue fills up
  before the timeout, `Axon.Serving` will execute the inference request and
  dispatch responses back to the respective processes.
    
  It's important to note that partial queues will always be padded to
  the given `:batch_size`. For example, if the queue has a size of 16
  when it hits timeout and the given batch size is 32, `Axon.Serving`
  will add padding to reach a batch size of 16. This is necessary to
  avoid recompilation overhead.

  If you plan on serving many models and continously reloading models
  with new versions, you should instead manage `Axon.Serving` processes
  with a `DynamicSupervisor`. Then, you can dynamically start and stop
  `Axon.Serving` processes and load/unload models in a more dynamic manner.
  """
  use GenServer
  import Axon.Shared

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

  # TODO: With pre-processing how can we determine batch size
  # TODO: Add debug to trace wait time, execution time, etc.
  # TODO: Support multiple batch sizes

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

  @doc """
  Invokes the given server's inference function with the
  given inputs.

  `input` must exactly match the form expected by `server`. It
  cannot include any additional inputs, or inputs with a differing
  shape.

  ## Examples
      
      input = %{"input" => Nx.random_uniform({1, 32})}
      result = Axon.Serving.predict(:my_model, input)
  """
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
    # TODO: correctly handle container inputs

    [first_input | _rest] = Map.values(input)

    state =
      %{state | queue: queue, count: count + Nx.axis_size(first_input, 0)}
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
    |> Enum.filter(fn {name, properties} ->
      optional = properties[:optional]
      shape = properties[:shape]

      unless optional or Map.has_key?(config_shapes, name) or shape != nil do
        raise ArgumentError, "must provide shape for required input #{name}"
      end

      # if the input is optional and we don't provide a shape, then we assume
      # that we take the optional path. if the shape is not present, but is optional
      # then we filter it out completely
      Map.has_key?(config_shapes, name) or (not optional and shape != nil)
    end)
    |> Map.new(fn {name, properties} ->
      axon_shape = properties[:shape]
      config_shape = config_shapes[name]

      # TODO: Handle inputs which are not batched across an entire batch
      # like head_mask
      shape =
        cond do
          axon_shape == nil and valid_shape?(config_shape, batch_size) ->
            put_elem(config_shape, 0, batch_size)

          config_shape != nil and same_shape?(axon_shape, config_shape) and
              valid_shape?(config_shape, batch_size) ->
            put_elem(config_shape, 0, batch_size)

          config_shape == nil and valid_shape?(axon_shape, batch_size) ->
            put_elem(axon_shape, 0, batch_size)

          true ->
            raise ArgumentError, "invalid shape for input #{name}"
        end

      # TODO: preserve input type
      {name, Nx.template(shape, :f32)}
    end)
  end

  defp same_shape?(nil, _), do: true
  defp same_shape?(_, nil), do: true

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
        [input | _rest] = Map.values(input)
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
    |> to_batched_output()
    |> Enum.reverse()
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

  defp maybe_pad(inputs, current_batch_size, desired_batch_size) do
    pad_size = desired_batch_size - current_batch_size
    # TODO: Do not pad batched outputs
    deep_new(inputs, fn tensor -> do_pad(tensor, pad_size) end)
  end

  defp do_pad(%Axon.None{} = result, _), do: result

  defp do_pad(tensor, pad_size) do
    first_axis_pad = {0, pad_size, 0}
    rest_axes_pad = List.duplicate({0, 0, 0}, Nx.rank(tensor) - 1)
    Nx.pad(tensor, 0.0, [first_axis_pad | rest_axes_pad])
  end

  defp to_batched_output(%Nx.Tensor{} = tensor), do: Nx.to_batched(tensor, 1)

  defp to_batched_output(result) do
    batch_size =
      deep_reduce(result, nil, fn
        tensor, _ -> Nx.axis_size(tensor, 0)
      end)

    for idx <- 0..(batch_size - 1) do
      deep_new(result, fn
        %Axon.None{} = out -> out
        out -> Nx.new_axis(out[[idx]], 0)
      end)
    end
  end
end
