defmodule Axon.Loop do
  @moduledoc """
  Abstraction for modeling a reduction of a dataset with an accumulated
  state for a number of epochs.

  Inspired heavily by [PyTorch Ignite](https://pytorch.org/ignite/index.html).

  The main abstraction is the `%Loop{}` struct, which controls a nested
  reduction of the form:

      Enum.reduce(1..max_epochs, state, fn epoch, state ->
        Enum.reduce(data, state, &batch_step/2)
      end)

  `data` is assumed to be an `Enumerable` or `Stream` of input data which is
  handled by a processing function, `batch_step`. The purpose of the loop
  abstraction is to take away much of the boilerplate used in solving machine
  learning tasks. Tasks such as normalizing a dataset, hyperparameter optimization,
  or training machine learning models boil down to writing one function:

      defn batch_step(batch, state) do
        # ...do something with batch...
        updated_state
      end

  For tasks such as training a neural network, `state` will encapsulate things
  such as model and optimizer state. For supervised learning tasks, `batch_step`
  might look something like:

      defn batch_step({inputs, targets}, state) do
        %{parameters: params, optimizer_state: optim_state} = state

        gradients = grad(params, objective_fn.(&1, inputs, targets))
        {updates, new_optim_state} = optimizer.(optim_state, params, gradients)

        new_params = apply_updates(params, updates)

        %{parameters: new_params, optimizer_state: optim_state}
      end

  `batch_step` takes a batch of `{input, target}` pairs and the current state,
  and updates the model parameters based on the gradients received from some arbitrary
  objective function. This function will run in a nested loop, iterating over the entire
  dataset for `N` epochs before finally returning the trained model state. By defining
  1 function, we've created a training loop that works for most machine learning models.

  In actuality, the loop abstraction accumulates a struct, `Axon.Loop.State`, which looks
  like (assuming `container` is a generic Elixir container of tensors, e.g. map, tuple, etc.):

      %State{
        epoch: integer(),
        max_epoch: integer(),
        iteration: integer(),
        max_iteration: integer(),
        metrics: map(string(), container()),
        times: map(integer(), integer()),
        step_state: container()
      }

  `batch_step` takes in the batch and the step state field and returns a `step_state`,
  which is a generic container of state accumulated at each iteration. The rest of the fields
  in the state struct are updated automatically behind the scenes.

  The loop must start from some initial step state, thus most tasks must also provide
  an additional initialization function to provide some starting point for the step
  state. For machine learning tasks, the initialization function will return things like
  initial model parameters and optimizer state.

  Typically, the final output of the loop is the accumulated final state; however, you
  may optionally apply an output transform to extract specific values at the end of the
  loop. For example, `Axon.Loop.trainer/4` by default extracts trained model state:

      output_transform = fn state ->
        state.step_state[:model_state]
      end

  ## Initialize and Step

  The core of the Axon loop are the init and step functions. The initialization is an
  arity-0 function which provides an initial step state:

      init = fn ->
        %{params: Axon.init(model)}
      end

  While the step function is the `batch_step` function mentioned earlier:

      step = fn data, state ->
        new_state = # ...do something...
        new_state
      end

  ## Metrics

  Often times you want to compute metrics associated with your training iterations.
  To accomplish this, you can attach metrics to each `Axon.Loop`. Assuming a `batch_step`
  function which looks like:

      defn batch_step({inputs, targets}, state) do
        %{parameters: params, optimizer_state: optim_state} = state

        gradients = grad(params, objective_fn.(&1, inputs, targets))
        {updates, new_optim_state} = optimizer.(optim_state, params, gradients)

        new_params = apply_updates(params, updates)

        # Shown for simplicity, you can optimize this by calculating preds
        # along with the gradient calculation
        preds = model_fn.(params, inputs)

        %{
          y_true: targets,
          y_pred: preds,
          parameters: new_params,
          optimizer_state: optim_state
        }
      end

  You can attach metrics to this by using `Axon.Loop.metric/4`:

      Axon.Loop.loop(&batch_step/2)
      |> Axon.Loop.metric("Accuracy", :accuracy, fn %{y_true: y_, y_pred: y} -> [y_, y] end)
      |> Axon.Loop.run(data)

  Because metrics work directly on `step_state`, you typically need to provide an output
  transform to indicate which values should be passed to your metric function. By default,
  Axon assumes a supervised training task with the fields `:y_true` and `:y_pred` present
  in the step state. See `Axon.Loop.metric/4` for more information.

  Metrics will be tracked in the loop state using the user-provided key. Metrics integrate
  seamlessly with the supervised metrics defined in `Axon.Metrics`. You can also use metrics
  to keep running averages of some values in the original dataset.

  ## Events and Handlers

  You can instrument several points in the loop using event handlers. By default, several events
  are fired when running a loop:

      events = [
        :started,             # After loop state initialization
        :epoch_started,       # On epoch start
        :iteration_started,   # On iteration start
        :iteration_completed, # On iteration complete
        :epoch_completed,     # On epoch complete
        :epoch_halted,        # On epoch halt, if early halted
        :halted,              # On loop halt, if early halted
        :completed            # On loop completion
      ]

  You can attach event handlers to events using `Axon.Loop.handle/4`:

      loop
      |> Axon.Loop.handle(:iteration_completed, &log_metrics/1, every: 100)
      |> Axon.Loop.run(data)

  The above will trigger `log_metrics/1` every 100 times the `:iteration_completed` event
  is fired. Event handlers must return a tuple `{status, state}`, where `status` is an
  atom with one of the following values:

      :continue   # Continue epoch, continue looping
      :halt_epoch # Halt the epoch, continue looping
      :halt_loop  # Halt looping

  And `state` is an updated `Axon.Loop.State` struct. Handler functions take as input
  the current loop state.

  It's important to note that event handlers are triggered in the order they are attached
  to the loop. If you have two handlers on the same event, they will trigger in order:

      loop
      |> Axon.Loop.handle(:epoch_completed, &normalize_state/1) # Runs first
      |> Axon.Loop.handle(:epoch_completed, &log_state/1) # Runs second

  You may provide filters to filter when event handlers trigger. See `Axon.Loop.handle/4`
  for more details on valid filters.

  ## Factories

  Axon loops are typically created from one of the factory functions provided in this
  module:

      * `Axon.Loop.loop/3` - Creates a loop from step function and optional initialization
      functions and output transform functions.

      * `Axon.Loop.trainer/3` - Creates a supervised training loop from model, loss, and
      optimizer.

      * `Axon.Loop.evaluator/2` - Creates a supervised evaluator loop from model and model
      state.

  ## Running loops

  In order to execute a loop, you should use `Axon.Loop.run/3`:

      loop
      |> Axon.Loop.run(data, epochs: 10)

  ## Resuming loops

  At times you may want to resume a loop from some previous state. You can accomplish this
  with `Axon.Loop.from_state/2`:

      loop
      |> Axon.Loop.from_state(state)
      |> Axon.Loop.run(data)
  """
  require Axon
  require Axon.Updates
  require Logger

  alias __MODULE__, as: Loop
  alias Axon.Loop.State

  @default_events [
    :started,
    :epoch_started,
    :iteration_started,
    :iteration_completed,
    :epoch_completed,
    :epoch_halted,
    :halted,
    :completed
  ]

  @default_handlers %{
    started: [],
    epoch_started: [],
    iteration_started: [],
    iteration_completed: [],
    epoch_completed: [],
    epoch_halted: [],
    halted: [],
    completed: []
  }

  @valid_axon_losses [
    :binary_cross_entropy,
    :categorical_cross_entropy,
    :categorical_hinge,
    :hinge,
    :kl_divergence,
    :log_cosh,
    :mean_absolute_error,
    :mean_squared_error,
    :poisson,
    :soft_margin
  ]

  @valid_axon_optimizers [
    :adabelief,
    :adagrad,
    :adam,
    :adamw,
    :fromage,
    :lamb,
    :noisy_sgd,
    :radam,
    :rmsprop,
    :sgd,
    :yogi
  ]

  @doc false
  @derive {Inspect, only: [:metrics, :handlers]}
  @enforce_keys [:init, :step]
  defstruct [
    :init,
    :step,
    :attached_state,
    :output_transform,
    metrics: %{},
    handlers: @default_handlers
  ]

  ## Step Factories

  @doc """
  Creates a supervised train step from a model, loss function, and
  optimizer.

  This function is intended for more fine-grained control over the loop
  creation process. It returns a tuple of `{init_fn, step_fn}` where `init_fn`
  is an initialization function which returns an initial step state and
  `step_fn` is a supervised train step constructed from `model`, `loss`,
  and `optimizer`.

  `model` must be an Axon struct, a valid defn container
  of Axon structs, or a `{init_fn, apply_fn}`-tuple where `init_fn` is
  an arity-0 function which initializes the model state and `apply_fn` is
  an arity-2 function which applies the forward pass of the model. The forward
  pass of the model must return a map with keys `:prediction` and `:state`
  representing the model's prediction and updated state for layers which
  aggregate state during training.

  `loss` must be an atom which matches a function in `Axon.Losses`, a list
  of `{loss, weight}` tuples representing a basic weighted loss function
  for multi-output models, or an arity-2 function representing a custom loss
  function.

  `optimizer` must be an atom matching the name of a valid optimizer in `Axon.Optimizers`,
  or a `{init_fn, update_fn}` tuple where `init_fn` is an arity-1 function which
  initializes the optimizer state from attached parameters and `update_fn` is an
  arity-3 function which scales gradient updates with respect to input parameters,
  optimizer state, and gradients. See `Axon.Updates` for more information on building
  optimizers.
  """
  def train_step(model, loss, optimizer) do
    {init_model_fn, forward_model_fn} = build_model_fns(model, :train)
    loss_fn = build_loss_fn(loss)
    {init_optimizer_fn, update_optimizer_fn} = build_optimizer_fns(optimizer)

    init_fn = fn ->
      model_state = init_model_fn.()
      optimizer_state = init_optimizer_fn.(model_state)

      %{
        i: Nx.tensor(0, backend: Nx.Defn.Expr),
        y_true: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        y_pred: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        loss: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        model_state: model_state,
        optimizer_state: optimizer_state
      }
    end

    objective_fn = fn state, inp, tar ->
      model_out = forward_model_fn.(state, inp)
      {model_out, loss_fn.(tar, model_out.prediction)}
    end

    step_fn = fn {inp, tar}, state ->
      %{i: i, model_state: model_state, optimizer_state: optimizer_state, loss: loss} = state

      {{model_out, batch_loss}, gradients} =
        Nx.Defn.value_and_grad(
          model_state,
          &objective_fn.(&1, inp, tar),
          fn x -> elem(x, 1) end
        )

      preds = model_out.prediction
      new_state = model_out.state

      new_loss =
        loss
        |> Nx.multiply(i)
        |> Nx.add(batch_loss)
        |> Nx.divide(Nx.add(i, 1))

      {updates, new_optimizer_state} =
        update_optimizer_fn.(gradients, optimizer_state, model_state)

      %{
        state
        | i: Nx.add(i, 1),
          y_true: tar,
          y_pred: preds,
          loss: new_loss,
          model_state: Axon.Updates.apply_updates(model_state, updates, new_state),
          optimizer_state: new_optimizer_state
      }
    end

    {init_fn, step_fn}
  end

  @doc """
  Creates a supervised evaluation step from a model and model state.

  This function is intended for more fine-grained control over the loop
  creation process. It returns a tuple of `{init_fn, step_fn}` where
  `init_fn` returns an initial step state and `step_fn` performs a
  single evaluation step.
  """
  def eval_step(model, model_state) do
    {_, forward_model_fn} = build_model_fns(model, :inference)

    init_fn = fn ->
      %{
        y_true: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        y_pred: Nx.tensor(0.0, backend: Nx.Defn.Expr)
      }
    end

    step_fn = fn {inp, tar}, _ ->
      %{
        y_true: tar,
        y_pred: forward_model_fn.(model_state, inp)
      }
    end

    {init_fn, step_fn}
  end

  ## Loop Factories

  @doc """
  Creates a loop from `step_fn`, an optional `init_fn`, and an
  optional `output_transform`.

  `step_fn` is an arity-2 function which takes a batch and state
  and returns an updated step state:

      defn batch_step(batch, step_state) do
        step_state + 1
      end

  `init_fn` by default is a function which returns an empty map. You should
  define your own if subsequent step state updates rely on an initial
  step state:

      defn init_step_state() do
        0
      end

  `step_batch/2` and `init_step_state/0` are typically called from
  within `Nx.Defn.jit/3`. While JIT-compilation will work with anonymous functions,
  `def`, and `defn`, it is recommended that you use the stricter `defn` to define
  both functions in order to avoid bugs or cryptic errors.

  `output_transform/1` applies a transformation on the final accumulated loop state.
  This is useful for extracting specific fields from a loop and piping them into
  additional functions.
  """
  def loop(step_fn, init_fn \\ fn -> %{} end, output_transform \\ & &1)
      when is_function(step_fn, 2) and is_function(init_fn, 0) and
             is_function(output_transform, 1) do
    %Loop{
      init: init_fn,
      step: step_fn,
      output_transform: output_transform
    }
  end

  @doc """
  Creates a supervised training loop from a model, loss function,
  and optimizer.

  This function is useful for training models on most standard supervised
  learning tasks. It assumes data consists of tuples of input-target pairs,
  e.g. `[{x0, y0}, {x1, y1}, ..., {xN, yN}]` where `x0` and `y0` are batched
  tensors or containers of batched tensors.

  It defines an initialization function which first initializes model state
  using the given model and then initializes optimizer state using the initial
  model state. The step function uses a differentiable objective function
  defined with respect to the model parameters, input data, and target data
  using the given loss function. It then updates model parameters using the
  given optimizer in order to minimize loss with respect to the model parameters.

  `model` must be an Axon struct, a valid defn container
  of Axon structs, or a `{init_fn, apply_fn}`-tuple where `init_fn` is
  an arity-0 function which initializes the model state and `apply_fn` is
  an arity-2 function which applies the forward pass of the model.

  `loss` must be an atom which matches a function in `Axon.Losses`, a list
  of `{loss, weight}` tuples representing a basic weighted loss function
  for multi-output models, or an arity-2 function representing a custom loss
  function.

  `optimizer` must be an atom matching the name of a valid optimizer in `Axon.Optimizers`,
  or a `{init_fn, update_fn}` tuple where `init_fn` is an arity-1 function which
  initializes the optimizer state from attached parameters and `update_fn` is an
  arity-3 function which scales gradient updates with respect to input parameters,
  optimizer state, and gradients. See `Axon.Updates` for more information on building
  optimizers.

  This function creates a step function which outputs a map consisting of the following
  fields for `step_state`:

      %{
        y_pred: tensor() | container(tensor()), # Model predictions for use in metrics
        y_true: tensor() | container(tensor()), # True labels for use in metrics
        loss: tensor(), # Running average of loss over epoch
        model_state: container(tensor()), # Model parameters and state
        optimizer_state: container(tensor()) # Optimizer state associated with each parameter
      }

  ## Examples

  ### Basic usage

      data = Stream.zip(input, target)

      model = Axon.input({nil, 32}) |> Axon.dense(1, activation: :sigmoid)

      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :adam)
      |> Axon.Loop.run(data)

  ### Customizing Optimizer

      model
      |> Axon.Loop.trainer(:binary_cross_entropy, Axon.Optimizers.adam(0.05))
      |> Axon.Loop.run(data)

  ### Custom loss

      loss_fn = fn y_true, y_pred -> Nx.cos(y_true, y_pred) end

      model
      |> Axon.Loop.trainer(loss_fn, Axon.Optimizers.rmsprop(0.01))
      |> Axon.Loop.run(data)

  ### Multiple objectives with multi-output model

      model = {Axon.input({nil, 1}), Axon.input({nil, 2})}
      loss_weights = [mean_squared_error: 0.5, mean_absolute_error: 0.5]

      model
      |> Axon.Loop.trainer(loss_weights)
      |> Axon.Loop.run(data)

  ## Options

    * `:log` - training loss and metric log interval. Set to 0 to silence
      training logs. Defaults to 50
  """
  def trainer(model, loss, optimizer, opts \\ []) do
    log_interval = opts[:log] || 50
    # Build loss now so we can use it as a metric
    loss_fn = build_loss_fn(loss)
    {init_fn, step_fn} = train_step(model, loss_fn, optimizer)
    output_transform = fn state -> state.step_state[:model_state] end

    loop =
      step_fn
      |> loop(init_fn, output_transform)
      |> metric(loss_fn, "loss")

    if log_interval > 0 do
      loop
      |> log(:iteration_completed, &supervised_log_message_fn/1, :stdio, every: log_interval)
      |> log(:epoch_completed, fn _ -> "\n" end, :stdio)
    else
      loop
    end
  end

  defp format_metric({name, val}) do
    {type, _} = val.type

    case type do
      t when t in [:s, :u] -> "#{name}: #{:io_lib.format('~.B', [Nx.to_number(val)])}"
      :f -> "#{name}: #{:io_lib.format('~.7f', [Nx.to_number(val)])}"
      :bf -> "#{name}: #{:io_lib.format('~.3f', [Nx.to_number(val)])}"
      _ -> "Unsupported type of metric"
    end
  end

  defp supervised_log_message_fn(state, log_epochs \\ true) do
    %State{metrics: metrics, epoch: epoch, iteration: iter} = state

    metrics =
      metrics
      |> Enum.map(&format_metric/1)
      |> Enum.join(" ")

    if log_epochs do
      "\rEpoch: #{Nx.to_number(epoch)}, Batch: #{Nx.to_number(iter)}, #{metrics}"
    else
      "\rBatch: #{Nx.to_number(iter)}, #{metrics}"
    end
  end

  @doc """
  Creates a supervised evaluator from a model and model state.

  An evaluator can be used for things such as testing and validation of models
  after or during training. It assumes `model` is an Axon struct, container of
  structs, or a tuple of `init` / `apply` functions. `model_state` must be a
  container usable from within `model`.

  The evaluator returns a step state of the form:

      %{
        y_true: labels,
        y_pred: predictions
      }

  Such that you can attach any number of supervised metrics to the evaluation
  loop:

      model
      |> Axon.Loop.evaluator(trained_state)
      |> Axon.Loop.metric("Accuracy", :accuracy)

  Applies an output transform which returns the map of metrics accumulated over
  the given loop.
  """
  def evaluator(model, model_state) do
    {init_fn, step_fn} = eval_step(model, model_state)
    output_transform = fn state -> state.metrics end

    loop(step_fn, init_fn, output_transform)
    |> log(:iteration_completed, &supervised_log_message_fn(&1, false), :stdio)
  end

  @doc """
  Adds a metric of the given name to the loop.

  A metric is a function which tracks or measures some value with respect
  to values in the step state. For example, when training classification
  models, it's common to track the model's accuracy during training:

      loop
      |> Axon.Loop.metric(:accuracy, "Accuracy")

  By default, metrics assume a supervised learning task and extract the fields
  `[:y_true, :y_pred]` from the step state. If you wish to work on a different
  value, you can use an output transform. An output transform is a list of keys
  to extract from the output state, or a function which returns a flattened list
  of values to pass to the given metric function. Values received from output
  transforms are passed to the given metric using:

      value = output_transform.(step_state)
      apply(metric, value)

  Thus, even if you want your metric to work on a container, your output transform
  must return a list.

  `metric` must be an atom which matches the name of a metric in `Axon.Metrics`, or
  an arbitrary function which returns a tensor or container.

  `name` must be a string or atom used to store the computed metric in the loop
  state. If names conflict, the last attached metric will take precedence:

      loop
      |> Axon.Loop.metric(:mean_squared_error, "Error") # Will be overwritten
      |> Axon.Loop.metric(:mean_absolute_error, "Error") # Will be used

  By default, metrics keep a running average of the metric calculation. You can
  override this behavior by changing `accumulate`:

      loop
      |> Axon.Loop.metric(:true_negatives, "tn", :running_sum)

  Accumulation function can be one of the accumulation combinators in Axon.Metrics
  or an arity-3 function of the form: `accumulate(acc, obs, i) :: new_acc`.
  """
  def metric(
        %Loop{metrics: metric_fns} = loop,
        metric,
        name \\ nil,
        accumulate \\ :running_average,
        transform_or_fields \\ [:y_true, :y_pred]
      ) do
    name =
      case name do
        nil ->
          if is_atom(metric) do
            Atom.to_string(metric)
          else
            raise ArgumentError, "must provide name if using a custom metric"
          end

        name ->
          name
      end

    case metric_fns do
      %{^name => _} ->
        Logger.warning(
          "Metric #{name} declared twice in loop. Original metric will be overridden."
        )

      _ ->
        :ok
    end

    metric_fn = build_metric_fn(metric, accumulate, transform_or_fields)
    # For internal use we keep the raw metric as well as the compiled metric
    # function
    %Loop{loop | metrics: Map.put(metric_fns, name, {metric_fn, metric})}
  end

  @doc """
  Adds a handler function to the loop which will be triggered on `event`
  with an optional filter.

  Events take place at different points during loop execution. The default
  events are:

      events = [
        :started,             # After loop state initialization
        :epoch_started,       # On epoch start
        :iteration_started,   # On iteration start
        :iteration_completed, # On iteration complete
        :epoch_completed,     # On epoch complete
        :epoch_halted,        # On epoch halt, if early halted
        :halted,              # On loop halt, if early halted
        :completed            # On loop completion
      ]

  Generally, event handlers are side-effecting operations which provide some
  sort of inspection into the loop's progress. It's important to note that
  if you define multiple handlers to be triggered on the same event, they
  will execute in order from when they were attached to the training
  loop:

      loop
      |> Axon.Loop.handle(:epoch_started, &normalize_step_state/1) # executes first
      |> Axon.Loop.handle(:epoch_started, &log_step_state/1) # executes second

  Thus, if you have separate handlers which alter or depend on loop state,
  you need to ensure they are ordered correctly, or combined into a single
  event handler for maximum control over execution.

  `event` must be an atom representing the event to trigger `handler` or a
  list of atoms indicating `handler` should be triggered on multiple events.
  `event` may be `:all` which indicates the handler should be triggered on
  every event during loop processing.

  `handler` must be an arity-1 function which takes as input loop state and
  returns `{status, state}`, where `status` is an atom with one of the following
  values:

      :continue   # Continue epoch, continue looping
      :halt_epoch # Halt the epoch, continue looping
      :halt_loop  # Halt looping

  `filter` is an atom representing a valid filter predicate, a keyword of
  predicate-value pairs, or a function which takes loop state and returns
  a `true`, indicating the handler should run, or `false`, indicating the
  handler should not run. Valid predicates are:

      :always # Always trigger event
      :once   # Trigger on first event firing

  Valid predicate-value pairs are:

      every: N # Trigger every `N` event
      only: N # Trigger on `N` event
  """
  # TODO(seanmor5): Custom events
  def handle(%Loop{handlers: handle_fns} = loop, event, handler, filter \\ :always) do
    filter = build_filter_fn(filter)

    handle_fns =
      case event do
        [_ | _] = events ->
          Enum.reduce(events, handle_fns, &add_event_handler(&1, &2, {handler, filter}))

        :all ->
          Enum.reduce(@default_events, handle_fns, &add_event_handler(&1, &2, {handler, filter}))

        event when is_atom(event) ->
          add_event_handler(event, handle_fns, {handler, filter})
      end

    %Loop{loop | handlers: handle_fns}
  end

  @doc """
  Adds a handler function which logs the given message produced
  by `message_fn` to the given IO device every `event` satisfying
  `filter`.

  In most cases, this is useful for inspecting the contents of
  the loop state at intermediate stages. For example, the default
  `trainer` loop factory attaches IO logging of epoch, batch, loss
  and metrics.

  It's also possible to log loop state to files by changing the
  given IO device. By default, the IO device is `:stdio`.

  `message_fn` should take the loop state and return a binary
  representing the message to be written to the IO device.
  """
  def log(%Loop{} = loop, event, message_fn, device \\ :stdio, filter \\ :always)
      when is_function(message_fn, 1) do
    log_fn = fn %State{} = state ->
      try do
        msg = message_fn.(state)
        IO.write(device, msg)
        {:continue, state}
      rescue
        error ->
          Logger.error(
            "Error on Axon.Loop.log/5 callback: " <>
              Exception.format(:error, error, __STACKTRACE__)
          )

          {:halt_loop, state}
      end
    end

    loop
    |> handle(event, log_fn, filter)
  end

  @doc """
  Adds a handler function which tests the performance of `model`
  against the given validation set.

  This handler assumes the loop state matches the state initialized
  in a supervised training loop. Typically, you'd call this immediately
  after creating a supervised training loop:

      model
      |> Axon.Loop.trainer(:mean_squared_error, :sgd)
      |> Axon.Loop.validate(model, validation_data)

  Please note that you must pass the same (or an equivalent) model
  into this method so it can be used during the validation loop. The
  metrics which are computed are those which are present BEFORE the
  validation handler was added to the loop. For the following loop:

      model
      |> Axon.Loop.trainer(:mean_squared_error, :sgd)
      |> Axon.Loop.metric(:mean_absolute_error)
      |> Axon.Loop.validate(model, validation_data)
      |> Axon.Loop.metric(:binary_cross_entropy)

  only `:mean_absolute_error` will be computed at validation time.

  The returned loop state is altered to contain validation
  metrics for use in later handlers such as early stopping and model
  checkpoints. Since the order of execution of event handlers is in
  the same order they are declared in the training loop, you MUST call
  this method before any other handler which expects or may use
  validation metrics.
  """
  def validate(%Loop{metrics: metric_fns} = loop, model, validation_data) do
    validation_loop = fn %State{metrics: metrics, step_state: step_state} = state ->
      %{model_state: model_state} = step_state

      metrics =
        model
        |> evaluator(model_state)
        |> then(
          &Enum.reduce(metric_fns, &1, fn {k, {_, v}}, loop ->
            metric(loop, v, k)
          end)
        )
        |> log(:completed, fn _ -> "\n" end)
        |> run(validation_data)
        |> Map.new(fn {k, v} ->
          {"validation_#{k}", v}
        end)
        |> Map.merge(metrics, fn _, _, v -> v end)

      {:continue, %{state | metrics: metrics}}
    end

    handle(loop, :epoch_completed, validation_loop)
  end

  @doc """
  Attaches `state` to the given loop in order to resume looping
  from a previous state.

  It's important to note that a loop's attached state takes precedence
  over defined initialization functions. Given initialization function:

      defn init_state(), do: %{foo: 1, bar: 2}

  And an attached state:

      state = %State{step_state: %{foo: 2, bar: 3}}

  `init_state/0` will never execute, and instead the initial step state
  of `%{foo: 2, bar: 3}` will be used.
  """
  def from_state(%Loop{} = loop, %State{} = state) do
    %{loop | attached_state: state}
  end

  @doc """
  Runs the given loop on data with the given options.

  `loop` must be a valid Axon.Loop struct built from one of the
  loop factories provided in this module.

  `data` must be an Enumerable or Stream which yields batches of
  data on each iteration.

  ## Options

    * `:epochs` - max epochs to run loop for. Must be non-negative integer.
      Defaults to `1`.

    * `:iterations` - max iterations to run each epoch. Must be non-negative
      integer. Defaults to `-1` or no max iterations.

    * `:jit_compile?` - whether or not to JIT compile initialization and step
      functions. JIT compilation must be used for gradient computations. Defaults
      to true.

    Additional options are forwarded to `Nx.Defn.jit` as JIT-options. If no JIT
    options are set, the default options set with `Nx.Defn.default_options` are
    used.
  """
  def run(loop, data, opts \\ []) do
    {max_epochs, opts} = Keyword.pop(opts, :epochs, 1)
    {max_iterations, opts} = Keyword.pop(opts, :iterations, -1)
    {jit_compile?, jit_opts} = Keyword.pop(opts, :jit_compile?, true)

    %Loop{
      init: init_fn,
      step: step_fn,
      handlers: handler_fns,
      metrics: metric_fns,
      attached_state: attached_state,
      output_transform: output_transform
    } = loop

    loop_state =
      init_loop_state(
        init_fn,
        attached_state,
        metric_fns,
        max_epochs,
        max_iterations,
        jit_compile?,
        jit_opts
      )

    final_metrics_map =
      for i <- 0..(max_epochs - 1) do
        {i, Map.new(metric_fns, fn {k, _} -> {k, Nx.tensor(0, backend: Nx.Defn.Expr)} end)}
      end
      |> Map.new()

    {status, final_metrics, state} =
      case fire_event(:started, handler_fns, loop_state) do
        {:halt_epoch, state} ->
          {:halted, final_metrics_map, state}

        {:halt_loop, state} ->
          {:halted, final_metrics_map, state}

        {:continue, state} ->
          Enum.reduce_while(
            0..(max_epochs - 1)//1,
            {:completed, final_metrics_map, state},
            fn epoch, {_, final_metrics_map, loop_state} ->
              case fire_event(:epoch_started, handler_fns, loop_state) do
                {:halt_epoch, state} ->
                  halt_epoch(handler_fns, state)

                {:halt_loop, state} ->
                  {:halt, {:halted, final_metrics_map, state}}

                {:continue, state} ->
                  batch_fn = build_batch_fn(step_fn, metric_fns)

                  {time, status_and_state} =
                    :timer.tc(
                      &run_epoch/6,
                      [batch_fn, handler_fns, state, data, jit_compile?, jit_opts]
                    )

                  case status_and_state do
                    {:halt_epoch, state} ->
                      halt_epoch(handler_fns, state)

                    {:halt_loop, state} ->
                      {:halt, {:halted, final_metrics_map, state}}

                    {:continue, state} ->
                      new_loop_state = put_in(state.times[epoch], time)

                      case fire_event(:epoch_completed, handler_fns, new_loop_state) do
                        {:halt_epoch, state} ->
                          halt_epoch(handler_fns, state)

                        {:halt_loop, state} ->
                          {:halt, {:halted, final_metrics_map, state}}

                        {:continue, state} ->
                          zero_metrics = Map.new(metric_fns, fn {k, _} -> {k, 0} end)

                          final_metrics_map =
                            Map.replace!(final_metrics_map, epoch, state.metrics)

                          {:cont,
                           {:completed, final_metrics_map,
                            %State{
                              state
                              | epoch: epoch + 1,
                                metrics: zero_metrics,
                                iteration: 0,
                                max_iteration: state.max_iteration
                            }}}
                      end
                  end
              end
            end
          )
      end

    {_, state} = fire_event(status, handler_fns, state)
    state = %State{state | metrics: final_metrics}

    output_transform.(state)
  end

  ## Helpers

  defp init_loop_state(
         init_fn,
         attached_state,
         metric_fns,
         max_epochs,
         max_iterations,
         jit_compile?,
         jit_opts
       ) do
    case attached_state do
      %State{} = state ->
        state

      nil ->
        metrics = Map.new(metric_fns, fn {k, _} -> {k, Nx.tensor(0)} end)
        step_state = maybe_jit(init_fn, [], jit_compile?, jit_opts)

        %State{
          epoch: 0,
          max_epoch: max_epochs,
          iteration: 0,
          max_iteration: max_iterations,
          step_state: step_state,
          metrics: metrics,
          times: %{}
        }
    end
  end

  defp run_epoch(
         batch_fn,
         handler_fns,
         loop_state,
         data,
         jit_compile?,
         jit_opts
       ) do
    Enum.reduce_while(data, {:continue, loop_state}, fn data, {_, state} ->
      case fire_event(:iteration_started, handler_fns, state) do
        {:halt_epoch, state} ->
          {:halt, {:halt_epoch, state}}

        {:halt_loop, state} ->
          {:halt, {:halt_loop, state}}

        {:continue, state} ->
          %State{
            iteration: iters,
            max_iteration: max_iters,
            step_state: step_state,
            metrics: metrics
          } = state

          {new_step_state, new_metrics} =
            maybe_jit(batch_fn, [data, iters, step_state, metrics], jit_compile?, jit_opts)

          # If the data is kept on device, it may take a while for the VM
          # to refcount and garbage collect them. So we immediately
          # deallocate. We may have the same issue with user data,
          # so we may want to provide an option for force deallocation.
          # TODO: deallocate data based on an option?
          Nx.backend_deallocate({step_state, metrics})

          state = %{state | step_state: new_step_state, metrics: new_metrics}

          case fire_event(:iteration_completed, handler_fns, state) do
            {:halt_epoch, state} ->
              {:halt, {:halt_epoch, state}}

            {:halt_loop, state} ->
              {:halt, {:halt_loop, state}}

            {:continue, state} ->
              state = %{state | iteration: iters + 1}

              if iters >= max_iters and max_iters != -1 do
                {:halt, {:continue, state}}
              else
                {:cont, {:continue, state}}
              end
          end
      end
    end)
  end

  # Adds an event handler to the map of handler funs by prepending handler
  # to the existing handler funs. Because we prepend here, we must reverse
  # handler funs in fire_event.
  # TODO(seanmor5): Custom events
  defp add_event_handler(event, handle_fns, handler) do
    Map.update!(handle_fns, event, fn event_funs -> [handler | event_funs] end)
  end

  # Fires event `event` using handler_fns associated with the event. We
  # must reverse handler funs in order to enforce order that handlers are
  # attached to the loop.
  # TODO(seanmor5): Custom events
  defp fire_event(event, handler_fns, state) do
    handler_fns[event]
    |> Enum.reverse()
    |> Enum.reduce_while({:continue, state}, fn {handler, filter}, {_, state} ->
      if filter.(state) do
        case handler.(state) do
          {:continue, %State{} = state} ->
            {:cont, {:continue, state}}

          {:halt_epoch, %State{} = state} ->
            {:halt, {:halt_epoch, state}}

          {:halt_loop, %State{} = state} ->
            {:halt, {:halt_loop, state}}

          invalid ->
            raise ArgumentError,
                  "invalid value #{inspect(invalid)} returned from event handler" <>
                    " triggered on #{inspect(event)}, event handler must return" <>
                    " a tuple of {status, state} where status is one of :halt_epoch," <>
                    " :halt_loop, or :continue and state is an updated State struct"
        end
      else
        {:cont, {:continue, state}}
      end
    end)
  end

  # Halts an epoch during looping
  defp halt_epoch(handler_fns, loop_state) do
    case fire_event(:epoch_halted, handler_fns, loop_state) do
      {:halt_epoch, state} ->
        {:cont, %State{state | epoch: state.epoch + 1, iteration: 0}}

      {:halt_loop, state} ->
        {:halt, {:halted, state}}

      {:continue, state} ->
        {:cont, state}
    end
  end

  # Builds the overall batch step function from the given
  # step function and metrics. We need to run both step and metric
  # functions from within here to ensure they can be JIT compiled
  # if that's desired
  defp build_batch_fn(step_fn, metric_fns) do
    fn data, iter, pstate, metrics ->
      new_step_state = step_fn.(data, pstate)

      new_metrics =
        metrics
        |> Enum.zip_with(metric_fns, fn {k, avg}, {k, {v, _}} ->
          # In some instances the metric is actually present in the
          # step state e.g. in a supervised training loop when we
          # are computing loss but it's already computed as a part
          # of the step state, so we need to check here
          metric = String.to_atom(k)

          case pstate do
            %{^metric => value} ->
              {k, value}

            %{} ->
              {k, v.(avg, List.wrap(new_step_state), iter)}
          end
        end)
        |> Map.new()

      {new_step_state, new_metrics}
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
      loss_name when is_atom(loss_name) and loss_name in @valid_axon_losses ->
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

  # Builds model init and forward functions from an Axon struct,
  # a tuple of Axon structs, or a tuple of init / forward
  # functions. Model functions are essentially just model
  # init / apply functions.
  # TODO(seanmor5): Update this to support any valid defn container
  defp build_model_fns(%Axon{} = model, mode) do
    Axon.compile(model, mode: mode)
  end

  defp build_model_fns({init_fn, forward_fn}, _)
       when is_function(init_fn, 0) and is_function(forward_fn, 2) do
    {init_fn, forward_fn}
  end

  defp build_model_fns(model, mode) when is_tuple(model) do
    Axon.compile(model, mode: mode)
  end

  defp build_model_fns(invalid, _) do
    raise ArgumentError,
          "Invalid model #{inspect(invalid)}, a valid model" <>
            " is an Axon struct, a container of Axon structs " <>
            " or a tuple of {init_fn, forward_fn} with signatures" <>
            " init_fn() :: model_state, forward_fn(model_state, inp) :: prediction"
  end

  # Builds optimizer init and update functions either from an atom
  # or a tuple of init / update functions. The init and update functions
  # match the signatures of those defined in Axon.Updates. If the
  # optimizer is an atom, it must match the name of a function in
  # Axon.Optimizers.
  defp build_optimizer_fns(optimizer)
       when is_atom(optimizer) and optimizer in @valid_axon_optimizers do
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
  # to extract from the step state, or a function which transforms the step
  # state before it is passed to the metric function.
  # TODO(seanmor5): Reconsider the form of output transform
  defp build_metric_fn(metric, accumulator, transform_or_fields) do
    transform_fn =
      case transform_or_fields do
        [_ | _] = fields ->
          fn output ->
            fields
            |> Enum.reduce([], fn field, acc -> [output[field] | acc] end)
            |> Enum.reverse()
          end

        field when is_atom(field) ->
          fn output ->
            output[field]
          end

        transform when is_function(transform, 1) ->
          transform

        invalid ->
          raise ArgumentError,
                "Invalid output transform #{inspect(invalid)}, a valid output" <>
                  " transform is an atom or list of atoms specifying field(s)" <>
                  " to extract from the step state, or an arity-1 function" <>
                  " applied to the step state"
      end

    metric_fn =
      case metric do
        metric when is_atom(metric) ->
          fn output ->
            output
            |> transform_fn.()
            |> then(&apply(Axon.Metrics, metric, &1))
          end

        metric_fn when is_function(metric, 2) ->
          fn output ->
            output
            |> transform_fn.()
            |> then(&apply(metric_fn, &1))

            # |> List.wrap()
          end

        invalid ->
          raise ArgumentError,
                "Invalid metric #{inspect(invalid)}, a valid metric" <>
                  " is an atom which matches the name of a function in" <>
                  " Axon.Metrics or a function which takes a transformed" <>
                  " step state and returns a value"
      end

    case accumulator do
      acc_fun when acc_fun in [:running_average, :running_sum] ->
        apply(Axon.Metrics, acc_fun, [metric_fn])

      acc_fun when is_function(acc_fun, 3) ->
        &acc_fun.(&1, metric_fn.(&2), &3)

      invalid ->
        raise ArgumentError,
              "Invalid accumulation function #{inspect(invalid)}, a valid" <>
                " accumulation function is an atom which matches the name" <>
                " of an accumulation function in Axon.Metrics, or an arity-3" <>
                " function which takes current accumulator, observation, and" <>
                " iteration and returns an updated accumulator"
    end
  end

  # Builds a filter function from an atom, keyword list, or function. A
  # valid filter is an atom which matches on of the valid predicates `:always`
  # or `:once`, a keyword which matches one of the valid predicate-value pairs
  # such as `every: N`, or a function which takes loop state and returns `true`
  # or `false`.
  #
  # TODO(seanmor5): In order to handle custom events and predicate filters,
  # we will need to track event firings in the loop state.
  defp build_filter_fn(filter) do
    case filter do
      :always ->
        fn _ -> true end

      :once ->
        fn
          %State{epoch: 0, iteration: 0} -> true
          _ -> false
        end

      [{:every, n} | _] ->
        fn %State{iteration: iter} ->
          Kernel.rem(iter, n) == 0
        end

      fun when is_function(fun, 1) ->
        fun

      invalid ->
        raise ArgumentError,
              "Invalid filter #{inspect(invalid)}, a valid filter" <>
                " is an atom which matches a valid filter predicate" <>
                " such as :always or :once, a keyword of predicate-value" <>
                " pairs such as every: N, or an arity-1 function which takes" <>
                " loop state and returns true or false"
    end
  end

  # JIT-compiles the given function if jit_compile? is true
  # otherwise just applies the function with the given arguments
  defp maybe_jit(fun, args, jit_compile?, jit_opts) do
    if jit_compile? do
      Nx.Defn.jit(fun, args, jit_opts)
    else
      apply(fun, args)
    end
  end
end
