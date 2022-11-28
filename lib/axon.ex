defmodule Axon do
  @moduledoc """
  A high-level interface for creating neural network models.

  Axon is built entirely on top of Nx numerical definitions,
  so every neural network can be JIT or AOT compiled using
  any Nx compiler, or even transformed into high-level neural
  network formats like TensorFlow Lite and
  [ONNX](https://github.com/elixir-nx/axon_onnx).

  For a more in-depth overview of Axon, refer to the [Guides](guides.html).

  ## Model Creation

  All Axon models start with an input layer, optionally specifying
  the expected shape of the input data:

      input = Axon.input("input", shape: {nil, 784})

  Notice you can specify some dimensions as `nil`, indicating
  that the dimension size will be filled in at model runtime.
  You can then compose inputs with other layers:

      model =
        input
        |> Axon.dense(128, activation: :relu)
        |> Axon.batch_norm()
        |> Axon.dropout(rate: 0.8)
        |> Axon.dense(64)
        |> Axon.tanh()
        |> Axon.dense(10)
        |> Axon.activation(:softmax)

  You can inspect the model for a nice summary:

      IO.inspect(model)

      #Axon<
        inputs: %{"input" => {nil, 784}}
        outputs: "softmax_0"
        nodes: 9
      >

  Or use the `Axon.Display` module to see more in-depth summaries:

      Axon.Display.as_table(model, Nx.template({1, 784}, :f32)) |> IO.puts

      +----------------------------------------------------------------------------------------------------------------+
      |                                                     Model                                                      |
      +=======================================+=============+==============+===================+=======================+
      | Layer                                 | Input Shape | Output Shape | Options           | Parameters            |
      +=======================================+=============+==============+===================+=======================+
      | input ( input )                       | []          | {1, 784}     | shape: {nil, 784} |                       |
      |                                       |             |              | optional: false   |                       |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+
      | dense_0 ( dense["input"] )            | [{1, 784}]  | {1, 128}     |                   | kernel: f32[784][128] |
      |                                       |             |              |                   | bias: f32[128]        |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+
      | relu_0 ( relu["dense_0"] )            | [{1, 128}]  | {1, 128}     |                   |                       |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+
      | batch_norm_0 ( batch_norm["relu_0"] ) | [{1, 128}]  | {1, 128}     | epsilon: 1.0e-5   | gamma: f32[128]       |
      |                                       |             |              | channel_index: 1  | beta: f32[128]        |
      |                                       |             |              | momentum: 0.1     | mean: f32[128]        |
      |                                       |             |              |                   | var: f32[128]         |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+
      | dropout_0 ( dropout["batch_norm_0"] ) | [{1, 128}]  | {1, 128}     | rate: 0.8         |                       |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+
      | dense_1 ( dense["dropout_0"] )        | [{1, 128}]  | {1, 64}      |                   | kernel: f32[128][64]  |
      |                                       |             |              |                   | bias: f32[64]         |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+
      | tanh_0 ( tanh["dense_1"] )            | [{1, 64}]   | {1, 64}      |                   |                       |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+
      | dense_2 ( dense["tanh_0"] )           | [{1, 64}]   | {1, 10}      |                   | kernel: f32[64][10]   |
      |                                       |             |              |                   | bias: f32[10]         |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+
      | softmax_0 ( softmax["dense_2"] )      | [{1, 10}]   | {1, 10}      |                   |                       |
      +---------------------------------------+-------------+--------------+-------------------+-----------------------+

  ### Multiple Inputs

  Creating a model with multiple inputs is as easy as declaring an
  additional input in your Axon graph. Every input layer present in
  the final Axon graph will be required to be passed as input at the
  time of model execution.

      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 1})

      # Both inputs will be used
      model1 = Axon.add(inp1, inp2)

      # Only inp2 will be used
      model2 = Axon.add(inp2, inp2)

  Axon graphs are immutable, which means composing and manipulating
  an Axon graph creates an entirely new graph. Additionally, layer
  names are lazily generated at model execution time. To avoid
  non-deterministic input orderings and names, Axon requires each
  input to have a unique binary identifier. You can then reference
  inputs by name when passing to models at execution time:

      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 1})

      model1 = Axon.add(inp1, inp2)

      {init_fn, predict_fn} = Axon.build(model1)

      params1 = init_fn.(Nx.template({1, 1}, {:f, 32}), %{})
      # Inputs are referenced by name
      predict_fn.(params1, %{"input_0" => x, "input_1" => y})

  ### Multiple Outputs

  Nx offers robust [container](https://hexdocs.pm/nx/Nx.Container.html) support
  which is extended to Axon. Axon allows you to wrap any valid Nx container
  in a layer. Containers are most commonly used to structure outputs:

      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 1})
      model = Axon.container(%{foo: inp1, bar: inp2})

  Containers can be arbitrarily nested:

      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 1})
      model = Axon.container({%{foo: {inp1, %{bar: inp2}}}})

  You can even use custom structs which implement the container protocol:

      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 1})
      model = Axon.container(%MyStruct{foo: inp1, bar: inp2})

  ### Custom Layers

  If you find that Axon's built-in layers are insufficient for your needs,
  you can create your own using the custom layer API. All of Axon's built-in
  layers (aside from special ones such as `input`, `constant`, and `container`)
  make use of this same API.

  Axon layers are really just placeholders for Nx computations with trainable
  parameters and possibly state. To define a custom layer, you just need to
  define a `defn` implementation:

      defn my_layer(x, weight, _opts \\ []) do
        Nx.atan2(x, weight)
      end

  Notice the only stipulation is that your custom layer implementation must
  accept at least 1 input and a list of options. At execution time, every
  layer will be passed a `:mode` option which can be used to control behavior
  at training and inference time.

  Inputs to your custom layer can be either Axon graph inputs or trainable
  parameters. You can pass Axon graph inputs as-is to a custom layer. To
  declare trainable parameters, use `Axon.param/3`:

      weight = Axon.param("weight", param_shape)

  To create a custom layer, you "wrap" your implementation and inputs into
  a layer using `Axon.layer`. You'll notice the API mirrors Elixir's `apply`:

      def atan2_layer(%Axon{} = input) do
        weight = Axon.param("weight", param_shape)
        Axon.layer(&my_layer/3, [input, weight])
      end

  ## Model Execution

  Under the hood, Axon models are represented as Elixir structs. You
  can initialize and apply models by building or compiling them with
  `Axon.build/2` or `Axon.compile/4` and then calling the produced
  initialization and predict functions:

      {init_fn, predict_fn} = Axon.build(model)

      init_fn.(Nx.template({1, 1}, {:f, 32}), %{})
      predict_fn.(params, inputs)

  You may either set the default JIT compiler or backend globally, or
  pass a specific compiler to `Axon.build/2`:

      EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA, mode: :train)

      init_fn.(Nx.template({1, 1}, {:f, 32}), %{})
      predict_fn.(params, inputs)

  `predict_fn` by default runs in inference mode, which performs certain
  optimizations and removes layers such as dropout layers. If constructing
  a training step using `Axon.predict/4` or `Axon.build/2`, be sure to specify
  `mode: :train`.

  ## Model Training

  Combining the Axon model creation API with the optimization and training
  APIs, you can create and train neural networks with ease:

      model =
        Axon.input("input_0", shape: {nil, 784})
        |> Axon.dense(128, activation: :relu)
        |> Axon.layer_norm()
        |> Axon.dropout()
        |> Axon.dense(10, activation: :softmax)

      IO.inspect model

      model_state =
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
        |> Axon.Loop.run(train_data, epochs: 10, compiler: EXLA)

  See `Axon.Updates` and `Axon.Loop` for a more in-depth treatment of
  model optimization and model training.
  """
  alias __MODULE__, as: Axon
  alias Axon.Parameter

  require Logger

  # Axon serialization version
  @file_version 1

  @type t :: %__MODULE__{}

  defstruct [
    :nodes,
    :output
  ]

  @doc """
  Custom Axon layer with given inputs.

  Inputs may be other Axon layers or trainable parameters created
  with `Axon.param`. At inference time, `op` will be applied with
  inputs in specified order and an additional `opts` parameter which
  specifies inference options. All options passed to layer are forwarded
  to inference function except:

    * `:name` - layer name.

    * `:op_name` - layer operation for inspection and building parameter
      map.

  Note this means your layer should not use these as input options,
  as they will always be dropped during inference compilation.

  Axon's compiler will additionally forward the following options to
  every layer at inference time:

    * `:mode` - `:inference` or `:train`. To control layer behavior
      based on inference or train time.

  `op` is a function of the form:

      fun = fn input, weight, bias, _opts ->
        input * weight + bias
      end

  """
  @doc type: :special
  def layer(op, inputs, opts \\ []) when (is_atom(op) or is_function(op)) and is_list(inputs) do
    {inputs, params, args, updated_nodes} = split_inputs(op, inputs)

    inputs = Enum.reverse(inputs)
    params = Enum.reverse(params)
    args = Enum.reverse(args)

    {name, opts} = Keyword.pop(opts, :name)
    {op_name, layer_opts} = Keyword.pop(opts, :op_name, :custom)

    {id, name} = unique_identifiers(op_name, name)

    axon_node = make_node(id, op, name, op_name, inputs, params, args, layer_opts)

    %Axon{output: id, nodes: Map.put(updated_nodes, id, axon_node)}
  end

  defp make_node(id, op, name, op_name, inputs, params, args, layer_opts) do
    {:current_stacktrace, [_process_info, _axon_layer | stacktrace]} =
      Process.info(self(), :current_stacktrace)

    %Axon.Node{
      id: id,
      name: name,
      parent: inputs,
      parameters: params,
      args: args,
      op: op,
      policy: Axon.MixedPrecision.create_policy(),
      hooks: [],
      opts: layer_opts,
      op_name: op_name,
      stacktrace: stacktrace
    }
  end

  defp split_inputs(:container, [inputs]) do
    {inputs, cache} =
      deep_map_reduce(inputs, %{}, fn %Axon{output: id, nodes: nodes}, cache ->
        {id, Map.merge(nodes, cache)}
      end)

    {[inputs], [], [:layer], cache}
  end

  defp split_inputs(_op, inputs) do
    Enum.reduce(inputs, {[], [], [], %{}}, fn
      %Axon{output: layer_input, nodes: nodes}, {layers, params, args, cache} ->
        {[layer_input | layers], params, [:layer | args], Map.merge(nodes, cache)}

      %Parameter{} = param, {layers, params, args, cache} ->
        {layers, [param | params], [:parameter | args], cache}

      invalid, _ ->
        raise ArgumentError, "invalid input given to layer: #{inspect(invalid)}"
    end)
  end

  @doc """
  Trainable Axon parameter used to create custom layers.

  Parameters are specified in usages of `Axon.layer` and will
  be automatically initialized and used in subsequent applications
  of Axon models.

  Parameters *must* be specified in order of their usage.

  ## Options

    * `:initializer` - parameter initializer. Defaults to `:glorot_uniform`.

  """
  @doc type: :special
  def param(name, shape, opts \\ [])
      when is_binary(name) and (is_tuple(shape) or is_function(shape)) do
    opts = Keyword.validate!(opts, initializer: :glorot_uniform)
    initializer = validate_initializer!(opts[:initializer])

    id = System.unique_integer([:positive, :monotonic])

    %Axon.Parameter{
      id: id,
      name: name,
      shape: shape,
      initializer: initializer
    }
  end

  @doc """
  Adds an input layer to the network.

  Input layers specify a model's inputs. Input layers are
  always the root layers of the neural network.

  You must specify the input layers name, which will be used
  to uniquely identify it in the case of multiple inputs.

  ## Options

    * `:shape` - the expected input shape, use `nil` for dimensions
      of a dynamic size.

    * `:optional` - if `true`, the input may be omitted when using
      the model. This needs to be handled in one of the subsequent
      layers. See `optional/2` for more details.

  """
  @doc type: :special
  def input(name, opts \\ [])

  def input(name, opts) when is_binary(name) and is_list(opts) do
    opts = Keyword.validate!(opts, [:shape, optional: false])
    optional = opts[:optional]

    input_shape = opts[:shape]

    output_shape = input_shape && Axon.Shape.input(input_shape)
    layer(:input, [], name: name, shape: output_shape, op_name: :input, optional: optional)
  end

  # TODO: remove on Axon v0.3

  def input(input_shape, name) when is_binary(name) do
    IO.warn(
      "Passing shape as an argument to Axon.input/2 is deprecated, pass it as an option instead"
    )

    input(name, [{:shape, input_shape}])
  end

  @deprecated "Pass the shape as an option to Axon.input/2"
  def input(input_shape, name, opts) when is_binary(name) do
    input(name, [{:shape, input_shape} | opts])
  end

  @doc """
  Wraps an Axon model in an optional node.

  By default, when an optional input is missing, all subsequent layers
  are nullified. For example, consider this model:

      values = Axon.input("values")
      mask = Axon.input("mask", optional: true)

      model =
        values
        |> Axon.dense(10)
        |> Axon.multiply(mask)
        |> Axon.dense(1)
        |> Axon.sigmoid()

  In case the mask is not provided, the input node will resolve to
  `%Axon.None{}` and so will all the layers that depend on it. By
  using `optional/2` a layer may opt-in to receive `%Axon.None{}`.
  To fix our example, we could define a custom layer to apply the
  mask only when present

      def apply_optional_mask(%Axon{} = x, %Axon{} = mask) do
        Axon.layer(
          fn x, mask, _opts ->
            case mask do
              %Axon.None{} -> x
              mask -> Nx.multiply(x, mask)
            end
          end,
          [x, Axon.optional(mask)]
        )
      end

      # ...

      model =
        values
        |> Axon.dense(10)
        |> apply_optional_mask(mask)
        |> Axon.dense(1)
        |> Axon.sigmoid()

  ## Options

    * `:name` - layer name.

  """
  @doc type: :special
  def optional(%Axon{} = x, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])
    layer(:optional, [x], name: opts[:name], op_name: :optional)
  end

  @doc """
  Adds a constant layer to the network.

  Constant layers encapsulate Nx tensors in an Axon layer for ease
  of use with other Axon layers. They can be used interchangeably
  with other Axon layers:

      inp = Axon.input("input", shape: {nil, 32})
      my_constant = Axon.constant(Nx.iota({1, 32}))
      model = Axon.add(inp, my_constant)

  Constant layers will be cast according to the mixed precision policy.
  If it's important for your constant to retain it's type during
  the computation, you will need to set the mixed precision policy to
  ignore constant layers.

  ## Options

    * `:name` - layer name.

  """
  def constant(tensor, opts \\ [])

  @doc type: :special
  def constant(%Nx.Tensor{} = tensor, opts) do
    opts = Keyword.validate!(opts, [:name])

    layer(:constant, [], name: opts[:name], value: tensor, op_name: :constant)
  end

  def constant(number, opts) when is_number(number) do
    opts = Keyword.validate!(opts, [:name])

    layer(:constant, [], name: opts[:name], value: Nx.tensor(number), op_name: :constant)
  end

  def constant(value, _) do
    raise ArgumentError,
          "value passed to constant must be an Nx tensor" <>
            " but got #{inspect(value)}, if you are passing" <>
            " a number, wrap it with a call to Nx.tensor/2"
  end

  @doc """
  Adds a container layer to the network.

  In certain cases you may want your model to have multiple
  outputs. In order to make this work, you must "join" the
  outputs into an Axon layer using this function for use in
  initialization and inference later on.

  The given container can be any valid Axon Nx container.

  ## Options

    * `:name` - layer name.

  ## Examples

      iex> inp1 = Axon.input("input_0", shape: {nil, 1})
      iex> inp2 = Axon.input("input_1", shape: {nil, 2})
      iex> model = Axon.container(%{a: inp1, b: inp2})
      iex> %{a: a, b: b} = Axon.predict(model, %{}, %{
      ...>    "input_0" => Nx.tensor([[1.0]]),
      ...>    "input_1" => Nx.tensor([[1.0, 2.0]])
      ...> })
      iex> a
      #Nx.Tensor<
        f32[1][1]
        [
          [1.0]
        ]
      >
      iex> b
      #Nx.Tensor<
        f32[1][2]
        [
          [1.0, 2.0]
        ]
      >

  """
  @doc type: :special
  def container(container, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])

    layer(:container, [container], name: opts[:name], op_name: :container)
  end

  # TODO: This should not be duplicated
  defp deep_new(%Nx.Tensor{} = x, fun), do: fun.(x)

  defp deep_new(x, fun) when is_number(x), do: fun.(x)

  defp deep_new(map, fun) do
    {cont, :ok} = Nx.Container.traverse(map, :ok, &recur_traverse(&1, &2, fun))
    cont
  end

  defp recur_traverse(item, :ok, fun) do
    case item do
      %Axon{} = t ->
        {fun.(t), :ok}

      %{axon: :axon} = t ->
        {fun.(t), :ok}

      container ->
        {deep_new(container, fun), :ok}
    end
  end

  @doc """
  Wraps an Axon model into a namespace.

  A namespace is a part of an Axon model which is meant to
  be a self-contained collection of Axon layers. Namespaces
  are guaranteed to always generate with the same internal
  layer names and can be re-used universally across models.

  Namespaces are most useful for containing large collections
  of layers and offering a straightforward means for accessing
  the parameters of individual model components. A common application
  of namespaces is to use them in with a pre-trained model for
  fine-tuning:

      {base, resnet_params} = resnet()
      base = base |> Axon.namespace("resnet")

      model = base |> Axon.dense(1)
      {init_fn, predict_fn} = Axon.build(model)

      init_fn.(Nx.template({1, 3, 224, 224}, {:f, 32}), %{"resnset" => resnet_params})

  Notice you can use `init_fn` in conjunction with namespaces
  to specify which portion of a model you'd like to initialize
  from a fixed starting point.

  Namespaces have fixed names, which means it's easy to run into namespace
  collisions. Re-using namespaces, re-using inner parts of a namespace,
  and attempting to share layers between namespaces are still sharp
  edges in namespace usage.
  """
  @doc type: :special
  def namespace(%Axon{} = axon, name) when is_binary(name) do
    layer(:namespace, [axon], name: name)
  end

  @doc """
  Adds a dense layer to the network.

  The dense layer implements:

      output = activation(dot(input, kernel) + bias)

  where `activation` is given by the `:activation` option and both
  `kernel` and `bias` are layer parameters. `units` specifies the
  number of output units.

  Compiles to `Axon.Layers.dense/4`.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`.

    * `:activation` - element-wise activation function.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`.

  """
  @doc type: :linear
  def dense(%Axon{} = x, units, opts \\ [])
      when is_integer(units) and units > 0 do
    opts =
      Keyword.validate!(opts, [
        :name,
        :activation,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true
      ])

    kernel_shape = &Axon.Shape.dense_kernel(&1, units)
    bias_shape = &Axon.Shape.dense_bias(&1, units)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], :dense}
      else
        {[x, kernel], :dense}
      end

    node = layer(op, inputs, name: opts[:name], op_name: :dense)

    if activation = opts[:activation] do
      activation(node, activation)
    else
      node
    end
  end

  @doc """
  Adds a bilinear layer to the network.

  The bilinear layer implements:

      output = activation(dot(dot(input1, kernel), input2) + bias)

  where `activation` is given by the `:activation` option and both
  `kernel` and `bias` are layer parameters. `units` specifies the
  number of output units.

  All dimensions but the last of `input1` and `input2` must match. The
  batch sizes of both inputs must also match or at least one must be `nil`.
  Inferred output batch size coerces to the strictest input batch size.

  Compiles to `Axon.Layers.bilinear/5`.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`.

    * `:activation` - element-wise activation function.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`.

  """
  @doc type: :linear
  def bilinear(
        %Axon{} = input1,
        %Axon{} = input2,
        units,
        opts \\ []
      )
      when is_integer(units) and units > 0 do
    opts =
      Keyword.validate!(opts, [
        :name,
        :activation,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true
      ])

    kernel_shape = &Axon.Shape.bilinear_kernel(&1, &2, units)
    bias_shape = &Axon.Shape.bilinear_bias(&1, &2, units)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[input1, input2, kernel, bias], :bilinear}
      else
        {[input1, input2, kernel], :bilinear}
      end

    node = layer(op, inputs, name: opts[:name], op_name: :bilinear)

    if activation = opts[:activation] do
      activation(node, activation)
    else
      node
    end
  end

  @doc """
  Adds a convolution layer to the network.

  The convolution layer implements a general dimensional
  convolutional layer - which convolves a kernel over the input
  to produce an output.

  Compiles to `Axon.Layers.conv/4`.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`

    * `:activation` - element-wise activation function.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`

    * `:kernel_size` - size of the kernel spatial dimensions. Defaults
      to `1`.

    * `:strides` - stride during convolution. Defaults to `1`.

    * `:padding` - padding to the spatial dimensions of the input.
      Defaults to `:valid`.

    * `:input_dilation` - dilation to apply to input. Defaults to `1`.

    * `:kernel_dilation` - dilation to apply to kernel. Defaults to `1`.

    * `:feature_group_size` - feature group size for convolution. Defaults
      to `1`.

    * `:channels` - channels location. One of `:first` or `:last`.
      Defaults to `:last`.

  """
  @doc type: :convolution
  def conv(%Axon{} = x, units, opts \\ [])
      when is_integer(units) and units > 0 do
    opts =
      Keyword.validate!(opts, [
        :name,
        :activation,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true,
        kernel_size: 1,
        strides: 1,
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        channels: :last,
        feature_group_size: 1
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]
    feature_group_size = opts[:feature_group_size]

    kernel_shape = &Axon.Shape.conv_kernel(&1, units, kernel_size, channels, feature_group_size)
    bias_shape = &Axon.Shape.conv_bias(&1, units, kernel_size, channels, feature_group_size)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], :conv}
      else
        {[x, kernel], :conv}
      end

    node =
      layer(op, inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        feature_group_size: feature_group_size,
        channels: channels,
        op_name: :conv
      )

    if activation = opts[:activation] do
      activation(node, activation)
    else
      node
    end
  end

  @doc """
  Adds a transposed convolution layer to the network.

  The transposed convolution layer is sometimes referred to as a
  fractionally strided convolution or (incorrectly) as a deconvolution.

  Compiles to `Axon.Layers.conv_transpose/4`.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`

    * `:activation` - element-wise activation function.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`

    * `:kernel_size` - size of the kernel spatial dimensions. Defaults
      to `1`.

    * `:strides` - stride during convolution. Defaults to `1`.

    * `:padding` - padding to the spatial dimensions of the input.
      Defaults to `:valid`.

    * `:kernel_dilation` - dilation to apply to kernel. Defaults to `1`.

    * `:channels` - channels location. One of `:first` or `:last`.
      Defaults to `:last`.

  """
  @doc type: :convolution
  def conv_transpose(%Axon{} = x, units, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        :activation,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true,
        kernel_size: 1,
        strides: 1,
        padding: :valid,
        kernel_dilation: 1,
        channels: :last
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]

    kernel_shape = &Axon.Shape.conv_kernel(&1, units, kernel_size, channels, 1)
    bias_shape = &Axon.Shape.conv_bias(&1, units, kernel_size, channels, 1)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], :conv_transpose}
      else
        {[x, kernel], :conv_transpose}
      end

    node =
      layer(op, inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        kernel_dilation: kernel_dilation,
        channels: channels,
        op_name: :conv_transpose
      )

    if activation = opts[:activation] do
      activation(node, activation)
    else
      node
    end
  end

  @doc """
  Adds a depthwise convolution layer to the network.

  The depthwise convolution layer implements a general
  dimensional depthwise convolution - which is a convolution
  where the feature group size is equal to the number of
  input channels.

  Channel multiplier grows the input channels by the given
  factor. An input factor of 1 means the output channels
  are the same as the input channels.

  Compiles to `Axon.Layers.depthwise_conv/4`.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`

    * `:activation` - element-wise activation function.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`

    * `:kernel_size` - size of the kernel spatial dimensions. Defaults
      to `1`.

    * `:strides` - stride during convolution. Defaults to `1`.

    * `:padding` - padding to the spatial dimensions of the input.
      Defaults to `:valid`.

    * `:input_dilation` - dilation to apply to input. Defaults to `1`.

    * `:kernel_dilation` - dilation to apply to kernel. Defaults to `1`.

    * `:channels` - channels location. One of `:first` or `:last`.
      Defaults to `:last`.

  """
  @doc type: :convolution
  def depthwise_conv(%Axon{} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    opts =
      Keyword.validate!(opts, [
        :name,
        :activation,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true,
        kernel_size: 1,
        strides: 1,
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        channels: :last
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]

    kernel_shape =
      &Axon.Shape.depthwise_conv_kernel(&1, channel_multiplier, kernel_size, channels)

    bias_shape = &Axon.Shape.depthwise_conv_bias(&1, channel_multiplier, kernel_size, channels)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])

        {[x, kernel, bias], :depthwise_conv}
      else
        {[x, kernel], :depthwise_conv}
      end

    node =
      layer(op, inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        channels: channels,
        op_name: :depthwise_conv
      )

    if activation = opts[:activation] do
      activation(node, activation)
    else
      node
    end
  end

  @doc """
  Adds a depthwise separable 2-dimensional convolution to the
  network.

  Depthwise separable convolutions break the kernel into kernels
  for each dimension of the input and perform a depthwise conv
  over the input with each kernel.

  Compiles to `Axon.Layers.separable_conv2d/6`.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`

    * `:activation` - element-wise activation function.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`

    * `:kernel_size` - size of the kernel spatial dimensions. Defaults
      to `1`.

    * `:strides` - stride during convolution. Defaults to `1`.

    * `:padding` - padding to the spatial dimensions of the input.
      Defaults to `:valid`.

    * `:input_dilation` - dilation to apply to input. Defaults to `1`.

    * `:kernel_dilation` - dilation to apply to kernel. Defaults to `1`.

    * `:channels` - channels location. One of `:first` or `:last`.
      Defaults to `:last`.

  """
  @doc type: :convolution
  def separable_conv2d(%Axon{} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    opts =
      Keyword.validate!(opts, [
        :name,
        :activation,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true,
        kernel_size: 1,
        strides: 1,
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        channels: :last
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]

    k1_shape =
      &Axon.Shape.separable_conv2d_kernel(
        &1,
        channel_multiplier,
        kernel_size,
        1,
        channels
      )

    k2_shape =
      &Axon.Shape.separable_conv2d_kernel(
        &1,
        channel_multiplier,
        kernel_size,
        2,
        channels
      )

    b1_shape = &Axon.Shape.separable_conv2d_bias(&1, channel_multiplier, kernel_size, channels)
    b2_shape = &Axon.Shape.separable_conv2d_bias(&1, channel_multiplier, kernel_size, channels)

    kernel_initializer = opts[:kernel_initializer]
    k1 = param("kernel_1", k1_shape, initializer: kernel_initializer)
    k2 = param("kernel_2", k2_shape, initializer: kernel_initializer)

    {inputs, op} =
      if opts[:use_bias] do
        bias_initializer = opts[:bias_initializer]
        b1 = param("bias_1", b1_shape, initializer: bias_initializer)
        b2 = param("bias_2", b2_shape, initializer: bias_initializer)
        {[x, k1, b1, k2, b2], :separable_conv2d}
      else
        {[x, k1, k2], :separable_conv2d}
      end

    node =
      layer(
        op,
        inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        channels: channels,
        op_name: :separable_conv2d
      )

    if activation = opts[:activation] do
      activation(node, activation)
    else
      node
    end
  end

  @doc """
  Adds a depthwise separable 3-dimensional convolution to the
  network.

  Depthwise separable convolutions break the kernel into kernels
  for each dimension of the input and perform a depthwise conv
  over the input with each kernel.

  Compiles to `Axon.Layers.separable_conv3d/8`.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`

    * `:activation` - element-wise activation function.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`

    * `:kernel_size` - size of the kernel spatial dimensions. Defaults
      to `1`.

    * `:strides` - stride during convolution. Defaults to `1`.

    * `:padding` - padding to the spatial dimensions of the input.
      Defaults to `:valid`.

    * `:input_dilation` - dilation to apply to input. Defaults to `1`.

    * `:kernel_dilation` - dilation to apply to kernel. Defaults to `1`.

    * `:channels` - channels location. One of `:first` or `:last`.
      Defaults to `:last`.

  """
  @doc type: :convolution
  def separable_conv3d(%Axon{} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    opts =
      Keyword.validate!(opts, [
        :name,
        :activation,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true,
        kernel_size: 1,
        strides: 1,
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        channels: :last
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]

    k1_shape =
      &Axon.Shape.separable_conv3d_kernel(
        &1,
        channel_multiplier,
        kernel_size,
        1,
        channels
      )

    k2_shape =
      &Axon.Shape.separable_conv3d_kernel(
        &1,
        channel_multiplier,
        kernel_size,
        2,
        channels
      )

    k3_shape =
      &Axon.Shape.separable_conv3d_kernel(
        &1,
        channel_multiplier,
        kernel_size,
        3,
        channels
      )

    b1_shape = &Axon.Shape.separable_conv3d_bias(&1, channel_multiplier, kernel_size, channels)

    b2_shape = &Axon.Shape.separable_conv3d_bias(&1, channel_multiplier, kernel_size, channels)

    b3_shape = &Axon.Shape.separable_conv3d_bias(&1, channel_multiplier, kernel_size, channels)

    kernel_initializer = opts[:kernel_initializer]
    k1 = param("kernel_1", k1_shape, initializer: kernel_initializer)
    k2 = param("kernel_2", k2_shape, initializer: kernel_initializer)
    k3 = param("kernel_3", k3_shape, initializer: kernel_initializer)

    {inputs, op} =
      if opts[:use_bias] do
        bias_initializer = opts[:bias_initializer]
        b1 = param("bias_1", b1_shape, initializer: bias_initializer)
        b2 = param("bias_2", b2_shape, initializer: bias_initializer)
        b3 = param("bias_3", b3_shape, initializer: bias_initializer)
        {[x, k1, b1, k2, b2, k3, b3], :separable_conv3d}
      else
        {[x, k1, k2, k3], :separable_conv3d}
      end

    node =
      layer(
        op,
        inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        channels: channels,
        op_name: :separable_conv3d
      )

    if activation = opts[:activation] do
      activation(node, activation)
    else
      node
    end
  end

  @activation_layers [
    {:celu, "Continuously-differentiable exponential linear unit", "a"},
    {:elu, "Exponential linear unit", "an"},
    {:exp, "Exponential", "an"},
    {:gelu, "Gaussian error linear unit", "a"},
    {:hard_sigmoid, "Hard sigmoid", "a"},
    {:hard_silu, "Hard sigmoid weighted linear unit", "a"},
    {:hard_tanh, "Hard hyperbolic tangent", "a"},
    {:leaky_relu, "Leaky rectified linear unit", "a"},
    {:linear, "Linear", "a"},
    {:log_sumexp, "Log-sumexp", "a"},
    {:log_sigmoid, "Log-sigmoid", "a"},
    {:log_softmax, "Log-softmax", "a"},
    {:mish, "Mish", "a"},
    {:relu, "Rectified linear unit", "a"},
    {:relu6, "Rectified linear unit 6", "a"},
    {:sigmoid, "Sigmoid", "a"},
    {:silu, "Sigmoid weighted linear unit", "a"},
    {:selu, "Scaled exponential linear unit", "a"},
    {:softmax, "Softmax", "a"},
    {:softplus, "Softplus", "a"},
    {:softsign, "Softsign", "a"},
    {:tanh, "Hyperbolic tangent", "a"}
  ]

  @doc """
  Adds an activation layer to the network.

  Activation layers are element-wise functions typically called
  after the output of another layer.

  ## Options

    * `:name` - layer name.

  """
  @doc type: :activation
  def activation(x, activation, opts \\ [])

  def activation(%Axon{} = x, activation, opts) when is_atom(activation) do
    opts = opts ++ [op_name: activation]
    layer(activation, [x], opts)
  end

  def activation(%Axon{} = x, activation, opts)
      when is_function(activation) do
    layer(activation, [x], opts)
  end

  ## Activation

  for {activation, name, a_or_an} <- @activation_layers do
    @doc """
    Adds #{a_or_an} #{name} activation layer to the network.

    See `Axon.Activations.#{Atom.to_string(activation)}/1` for more details.

    ## Options

      * `:name` - layer name.

    """
    @doc type: :activation
    def unquote(activation)(%Axon{} = x, opts \\ []) do
      activation(x, unquote(activation), opts)
    end
  end

  ## Dropout

  @dropout_layers [
    {:dropout, "Dropout", "a"},
    {:feature_alpha_dropout, "Feature alpha dropout", "a"},
    {:spatial_dropout, "Spatial dropout", "a"},
    {:alpha_dropout, "Alpha dropout", "an"}
  ]

  for {dropout, name, a_or_an} <- @dropout_layers do
    @doc """
    Adds #{a_or_an} #{name} layer to the network.

    See `Axon.Layers.#{Atom.to_string(dropout)}/2` for more details.

    ## Options

      * `:name` - layer name.

      * `:rate` - dropout rate. Defaults to `0.5`.
        Needs to be equal or greater than zero and less than one.

    """
    @doc type: :dropout
    def unquote(dropout)(%Axon{} = x, opts \\ []) do
      dropout(x, unquote(dropout), opts)
    end
  end

  defp dropout(%Axon{} = x, dropout, opts) do
    opts = Keyword.validate!(opts, [:name, rate: 0.5])

    if opts[:rate] < 0 or opts[:rate] >= 1 do
      raise ArgumentError,
            "The dropout rate needs to be >= 0 and < 1, got #{inspect(opts[:rate])}"
    end

    layer(dropout, [x],
      name: opts[:name],
      rate: opts[:rate],
      op_name: dropout
    )
  end

  ## Pooling

  @pooling_layers [
    {:max_pool, "Max pool", "a"},
    {:avg_pool, "Average pool", "an"},
    {:lp_pool, "Power average pool", "a"}
  ]

  for {pool, name, a_or_an} <- @pooling_layers do
    @doc """
    Adds #{a_or_an} #{name} layer to the network.

    See `Axon.Layers.#{Atom.to_string(pool)}/2` for more details.

    ## Options

      * `:name` - layer name.

      * `:kernel_size` - size of the kernel spatial dimensions. Defaults
        to `1`.

      * `:strides` - stride during convolution. Defaults to size of kernel.

      * `:padding` - padding to the spatial dimensions of the input.
        Defaults to `:valid`.

      * `:dilations` - window dilations. Defaults to `1`.

      * `:channels` - channels location. One of `:first` or `:last`.
        Defaults to `:last`.

    """
    @doc type: :pooling
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      pool(x, unquote(pool), opts)
    end
  end

  defp pool(%Axon{} = x, pool, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        :strides,
        kernel_size: 1,
        padding: :valid,
        channels: :last,
        dilations: 1,
        norm: 2
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    channels = opts[:channels]
    dilations = opts[:dilations]
    name = opts[:name]

    opts =
      if pool == :lp_pool do
        norm = opts[:norm]

        [
          name: name,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding,
          channels: channels,
          window_dilations: dilations,
          norm: norm,
          op_name: pool
        ]
      else
        [
          name: name,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding,
          channels: channels,
          window_dilations: dilations,
          op_name: pool
        ]
      end

    layer(pool, [x], opts)
  end

  ## Adaptive Pooling

  @adaptive_pooling_layers [
    {:adaptive_avg_pool, "Adaptive average pool", "an"},
    {:adaptive_max_pool, "Adaptive max pool", "an"},
    {:adaptive_lp_pool, "Adaptive power average pool", "an"}
  ]

  for {pool, name, a_or_an} <- @adaptive_pooling_layers do
    @doc """
    Adds #{a_or_an} #{name} layer to the network.

    See `Axon.Layers.#{Atom.to_string(pool)}/2` for more details.

    ## Options

      * `:name` - layer name.

      * `:output_size` - layer output size.

      * `:channels` - channel configuration. One of `:first` or `:last`.
        Defaults to `:last`.

    """
    @doc type: :pooling
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      adaptive_pool(x, unquote(pool), opts)
    end
  end

  defp adaptive_pool(%Axon{} = x, pool, opts) do
    opts = Keyword.validate!(opts, [:name, :output_size, channels: :last, norm: 2])

    channels = opts[:channels]
    name = opts[:name]
    output_size = opts[:output_size]

    opts =
      if pool == :adaptive_lp_pool do
        norm = opts[:norm]

        [
          name: name,
          output_size: output_size,
          norm: norm,
          channels: channels,
          op_name: pool
        ]
      else
        [
          name: name,
          output_size: output_size,
          channels: channels,
          op_name: pool
        ]
      end

    layer(pool, [x], opts)
  end

  ## Global Pooling

  @global_pooling_layers [
    {:global_avg_pool, "Global average pool"},
    {:global_max_pool, "Global max pool"},
    {:global_lp_pool, "Global LP pool"}
  ]

  for {pool, name} <- @global_pooling_layers do
    @doc """
    Adds a #{name} layer to the network.

    See `Axon.Layers.#{Atom.to_string(pool)}/2` for more details.

    Typically used to connect feature extractors such as those in convolutional
    neural networks to fully-connected models by reducing inputs along spatial
    dimensions to only feature and batch dimensions.

    ## Options

      * `:name` - layer name.

      * `:keep_axes` - option to keep reduced axes. If `true`, keeps reduced axes
        with a dimension size of 1.

      * `:channels` - channel configuration. One of `:first` or `:last`.
        Defaults to `:last`.

    """
    @doc type: :pooling
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      global_pool(x, unquote(pool), opts)
    end
  end

  defp global_pool(%Axon{} = x, pool, opts) do
    opts = Keyword.validate!(opts, [:name, keep_axes: false, channels: :last, norm: 2])

    keep_axes = opts[:keep_axes]
    name = opts[:name]
    channels = opts[:channels]

    opts =
      if pool == :global_lp_pool do
        norm = opts[:norm]

        [
          name: name,
          channels: channels,
          keep_axes: keep_axes,
          norm: norm,
          op_name: pool
        ]
      else
        [name: name, channels: channels, keep_axes: keep_axes, op_name: pool]
      end

    layer(pool, [x], opts)
  end

  ## Normalization

  @normalization_with_stats_layers [
    {:batch_norm, "Batch normalization", "a"},
    {:instance_norm, "Instance normalization", "an"}
  ]

  for {norm, name, a_or_an} <- @normalization_with_stats_layers do
    @doc """
    Adds #{a_or_an} #{name} layer to the network.

    See `Axon.Layers.#{Atom.to_string(norm)}/6` for more details.

    ## Options

      * `:name` - layer name.

      * `:gamma_initializer` - gamma parameter initializer. Defaults
        to `:glorot_uniform`.

      * `:beta_initializer` - beta parameter initializer. Defaults to
        `:zeros`.

      * `:channel_index` - input feature index used for calculating
        mean and variance. Defaults to `-1`.

      * `:epsilon` - numerical stability term.

    """
    @doc type: :normalization
    def unquote(norm)(%Axon{} = x, opts \\ []) do
      norm_with_stats(x, unquote(norm), opts)
    end
  end

  defp norm_with_stats(%Axon{} = x, norm, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        gamma_initializer: :glorot_uniform,
        beta_initializer: :zeros,
        channel_index: -1,
        epsilon: 1.0e-5,
        momentum: 0.1
      ])

    channel_index = opts[:channel_index]

    gamma_shape = &Axon.Shape.norm_param(&1, channel_index)
    beta_shape = &Axon.Shape.norm_param(&1, channel_index)
    mean_shape = &Axon.Shape.norm_param(&1, channel_index)
    var_shape = &Axon.Shape.norm_param(&1, channel_index)

    gamma = param("gamma", gamma_shape, initializer: opts[:gamma_initializer])
    beta = param("beta", beta_shape, initializer: opts[:beta_initializer])

    mean = param("mean", mean_shape, initializer: :zeros)
    var = param("var", var_shape, initializer: :ones)

    layer(
      norm,
      [x, gamma, beta, mean, var],
      name: opts[:name],
      epsilon: opts[:epsilon],
      channel_index: channel_index,
      momentum: opts[:momentum],
      op_name: norm
    )
  end

  @normalization_layers [
    {:layer_norm, "Layer normalization", "a"}
  ]

  for {norm, name, a_or_an} <- @normalization_layers do
    @doc """
    Adds #{a_or_an} #{name} layer to the network.

    See `Axon.Layers.#{Atom.to_string(norm)}/4` for more details.

    ## Options

      * `:name` - layer name.

      * `:gamma_initializer` - gamma parameter initializer. Defaults
        to `:glorot_uniform`.

      * `:beta_initializer` - beta parameter initializer. Defaults to
        `:zeros`.

      * `:channel_index` - input feature index used for calculating
        mean and variance. Defaults to `-1`.

      * `:epsilon` - numerical stability term.

    """
    @doc type: :normalization
    def unquote(norm)(%Axon{} = x, opts \\ []) do
      norm(x, unquote(norm), opts)
    end
  end

  defp norm(%Axon{} = x, norm, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        gamma_initializer: :glorot_uniform,
        beta_initializer: :zeros,
        channel_index: -1,
        epsilon: 1.0e-5
      ])

    channel_index = opts[:channel_index]

    gamma_shape = &Axon.Shape.norm_param(&1, channel_index)
    beta_shape = &Axon.Shape.norm_param(&1, channel_index)

    gamma = param("gamma", gamma_shape, initializer: opts[:gamma_initializer])
    beta = param("beta", beta_shape, initializer: opts[:beta_initializer])

    layer(norm, [x, gamma, beta],
      name: opts[:name],
      epsilon: opts[:epsilon],
      channel_index: channel_index,
      op_name: norm
    )
  end

  @doc """
  Adds a group normalization layer to the network.

  See `Axon.Layers.group_norm/4` for more details.

  ## Options

    * `:name` - layer name.

    * `:gamma_initializer` - gamma parameter initializer. Defaults
      to `:glorot_uniform`.

    * `:beta_initializer` - beta parameter initializer. Defaults to
      `:zeros`.

    * `:channel_index` - input feature index used for calculating
      mean and variance. Defaults to `-1`.

    * `:epsilon` - numerical stability term.

  """
  @doc type: :normalization
  def group_norm(%Axon{} = x, num_groups, opts \\ [])
      when is_integer(num_groups) and num_groups >= 1 do
    opts =
      Keyword.validate!(opts, [
        :name,
        gamma_initializer: :ones,
        beta_initializer: :zeros,
        channel_index: -1,
        epsilon: 1.0e-5
      ])

    channel_index = opts[:channel_index]

    gamma_shape = &Axon.Shape.norm_param(&1, channel_index)
    beta_shape = &Axon.Shape.norm_param(&1, channel_index)

    gamma = param("gamma", gamma_shape, initializer: opts[:gamma_initializer])
    beta = param("beta", beta_shape, initializer: opts[:beta_initializer])

    layer(:group_norm, [x, gamma, beta],
      name: opts[:name],
      epsilon: opts[:epsilon],
      channel_index: channel_index,
      num_groups: num_groups,
      op_name: :group_norm
    )
  end

  @doc """
  Applies the given `Nx` expression to the input.

  Nx layers are meant for quick applications of functions without
  trainable parameters. For example, they are useful for applying
  functions which apply accessors to containers:

      model = Axon.container({foo, bar})
      Axon.nx(model, &elem(&1, 0))

  ## Options

    * `:name` - layer name.

  """
  def nx(input, fun, opts \\ [])

  @doc type: :special
  def nx(%Axon{} = x, fun, opts) when is_function(fun, 1) do
    opts = Keyword.validate!(opts, [:name, :op_name])
    op_name = opts[:op_name] || :nx
    fun_with_params = fn x, _opts -> fun.(x) end
    layer(fun_with_params, [x], name: opts[:name], op_name: op_name)
  end

  @doc """
  Adds a flatten layer to the network.

  This layer will flatten all but the batch dimensions
  of the input into a single layer. Typically called to flatten
  the output of a convolution for use with a dense layer.

  ## Options

    * `:name` - layer name.

  """
  @doc type: :shape
  def flatten(%Axon{} = x, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])

    layer(:flatten, [x],
      name: opts[:name],
      op_name: :flatten
    )
  end

  @doc """
  Adds a reshape layer to the network.

  This layer implements a special case of `Nx.reshape` which accounts
  for possible batch dimensions in the input tensor. You may pass the
  magic dimension `:batch` as a placeholder for dynamic batch sizes.
  You can use `:batch` seamlessly with `:auto` dimension sizes.

  If the input is an Axon constant, the reshape behavior matches that of
  `Nx.reshape/2`.

  ## Options

    * `:name` - layer name.
  """
  @doc type: :shape
  def reshape(%Axon{} = x, new_shape, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])

    layer(:reshape, [x],
      name: opts[:name],
      shape: new_shape,
      op_name: :reshape
    )
  end

  @doc """
  Adds a transpose layer to the network.

  ## Options

    * `:name` - layer name.

  """
  @doc type: :shape
  def transpose(%Axon{} = x, permutation \\ nil, opts \\ []) do
    opts = Keyword.validate!(opts, [:name])

    layer(:transpose, [x],
      name: opts[:name],
      axes: permutation,
      op_name: :transpose
    )
  end

  @doc """
  Adds a pad layer to the network.

  This layer will pad the spatial dimensions of the input.
  Padding configuration is a list of tuples for each spatial
  dimension.

  ## Options

    * `:name` - layer name.

    * `:channels` - channel configuration. One of `:first` or
      `:last`. Defaults to `:last`.

  """
  @doc type: :shape
  def pad(%Axon{} = x, config, value \\ 0.0, opts \\ [])
      when is_list(config) and is_number(value) do
    opts = Keyword.validate!(opts, [:name, channels: :last])
    channels = opts[:channels]

    layer(:pad, [x],
      name: opts[:name],
      padding_config: config,
      value: value,
      channels: channels,
      op_name: :pad
    )
  end

  @doc """
  Adds a resize layer to the network.

  Resizing can be used for interpolation or upsampling input
  values in a neural network. For example, you can use this
  layer as an upsampling layer within a GAN.

  Resize shape must be a tuple representing the resized spatial
  dimensions of the input tensor.

  Compiles to `Axon.Layers.resize/2`.

  ## Options

    * `:name` - layer name.

    * `:method` - resize method. Defaults to `:nearest`.

    * `:channels` - channel configuration. One of `:first` or
      `:last`. Defaults to `:last`.

  """
  @doc type: :shape
  def resize(%Axon{} = x, resize_shape, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, method: :nearest, channels: :last])
    channels = opts[:channels]

    layer(:resize, [x],
      name: opts[:name],
      method: opts[:method],
      channels: channels,
      size: resize_shape,
      op_name: :resize
    )
  end

  @doc """
  Adds a concatenate layer to the network.

  This layer will concatenate inputs along the last
  dimension unless specified otherwise.

  ## Options

    * `:name` - layer name.

    * `:axis` - concatenate axis. Defaults to `-1`.

  """
  @doc type: :combinator
  def concatenate(%Axon{} = x, %Axon{} = y, opts)
      when is_list(opts) do
    opts = Keyword.validate!(opts, [:name, axis: -1])
    axis = opts[:axis]

    layer(:concatenate, [container({x, y})],
      name: opts[:name],
      axis: axis,
      op_name: :concatenate
    )
  end

  @doc type: :combinator
  def concatenate([%Axon{} | _] = inputs, opts)
      when is_list(inputs) and is_list(opts) do
    opts = Keyword.validate!(opts, [:name, axis: -1])
    axis = opts[:axis]

    layer(:concatenate, [container(List.to_tuple(inputs))],
      name: opts[:name],
      axis: axis,
      op_name: :concatenate
    )
  end

  @doc false
  def concatenate(%Axon{} = x, %Axon{} = y), do: concatenate(x, y, [])

  @doc false
  def concatenate(inputs) when is_list(inputs), do: concatenate(inputs, [])

  @element_wise_layers [:add, :subtract, :multiply]

  for op <- @element_wise_layers do
    @doc """
    Adds a #{op} layer to the network.

    This layer performs an element-wise #{Atom.to_string(op)} operation
    on input layers. All input layers must be capable of being
    broadcast together.

    If one shape has a static batch size, all other shapes must have a
    static batch size as well.

    ## Options

      * `:name` - layer name.

    """
    @doc type: :combinator
    def unquote(op)(%Axon{} = x, %Axon{} = y, opts) do
      opts = Keyword.validate!(opts, [:name])

      layer(unquote(op), [container({x, y})],
        name: opts[:name],
        op_name: unquote(op)
      )
    end

    @doc """
    Adds a #{op} layer to the network.

    This layer performs an element-wise #{Atom.to_string(op)} operation
    on all input layers. All input layers must be capable of being
    broadcast together.

    ## Options

      * `:name` - layer name.

    """
    @doc type: :combinator
    def unquote(op)(inputs, opts) when is_list(inputs) and is_list(opts) do
      opts = Keyword.validate!(opts, [:name])

      layer(unquote(op), [container(List.to_tuple(inputs))],
        name: opts[:name],
        op_name: unquote(op)
      )
    end

    @doc false
    def unquote(op)(%Axon{} = x, %Axon{} = y) do
      unquote(op)(x, y, [])
    end

    @doc false
    def unquote(op)([%Axon{} | _] = inputs), do: unquote(op)(inputs, [])
  end

  @doc """
  Adds a conditional layer which conditionally executes
  `true_graph` or `false_graph` based on the condition `cond_fn`
  at runtime.

  `cond_fn` is an arity-1 function executed on the output of the
  parent graph. It must return a boolean scalar tensor (e.g. 1 or 0).

  The shapes of `true_graph` and `false_graph` must be equal.
  """
  @doc type: :combinator
  def cond(
        %Axon{} = parent,
        cond_fn,
        %Axon{} = true_graph,
        %Axon{} = false_graph,
        opts \\ []
      )
      when is_function(cond_fn, 1) do
    opts = Keyword.validate!(opts, [:name])

    layer(:cond, [parent, true_graph, false_graph],
      name: opts[:name],
      cond: cond_fn,
      op_name: :cond
    )
  end

  @doc """
  Splits input graph into a container of `n` input graphs
  along the given axis.

  ## Options

    * `:name` - layer name.

    * `:axis` - concatenate axis. Defaults to `-1`.

  """
  @doc type: :combinator
  def split(parent, splits, opts \\ [])

  def split(%Axon{} = parent, splits, opts) when is_list(splits) do
    opts = Keyword.validate!(opts, [:name, axis: -1])
    axis = opts[:axis]

    {_, split_layers} =
      for {split, i} <- Enum.with_index(splits), reduce: {0, []} do
        {num_split, split_layers} ->
          name =
            case opts[:name] do
              names when is_list(names) ->
                Enum.at(names, i)

              name ->
                name
            end

          layer =
            layer(
              fn x, _ -> Nx.slice_along_axis(x, num_split, split, axis: axis) end,
              [parent],
              name: name,
              op_name: :split
            )

          {num_split + split, [layer | split_layers]}
      end

    split_layers |> Enum.reverse() |> List.to_tuple()
  end

  def split(%Axon{} = parent, n, opts) when is_integer(n) do
    opts = Keyword.validate!(opts, [:name, axis: -1])
    axis = opts[:axis]

    splits =
      for i <- 0..(n - 1) do
        name =
          case opts[:name] do
            names when is_list(names) ->
              Enum.at(names, i)

            name ->
              name
          end

        layer(
          &Axon.Layers.split/2,
          [parent],
          name: name,
          index: i,
          splits: n,
          axis: axis,
          op_name: :split
        )
      end

    List.to_tuple(splits)
  end

  @doc """
  See `lstm/3`.
  """
  @doc type: :recurrent
  def lstm(%Axon{} = x, units) when is_integer(units) and units > 0 do
    lstm(x, units, [])
  end

  @doc """
  Adds a long short-term memory (LSTM) layer to the network
  with a random initial hidden state.

  See `lstm/4` for more details.

  ## Additional options

    * `:recurrent_initializer` - initializer for hidden state.
      Defaults to `:orthogonal`.

  """
  @doc type: :recurrent
  def lstm(%Axon{} = x, units, opts)
      when is_integer(units) and units > 0 and is_list(opts) do
    {recurrent_initializer, opts} = Keyword.pop(opts, :recurrent_initializer, :glorot_uniform)
    c = rnn_state(x, units, :lstm, opts[:name], "c", recurrent_initializer)
    h = rnn_state(x, units, :lstm, opts[:name], "h", recurrent_initializer)
    lstm(x, {c, h}, units, opts)
  end

  def lstm(%Axon{} = x, {%Axon{}, %Axon{}} = hidden_state, units)
      when is_integer(units) and units > 0 do
    lstm(x, hidden_state, units, [])
  end

  @doc """
  Adds a long short-term memory (LSTM) layer to the network
  with the given initial hidden state.

  LSTMs apply `Axon.Layers.lstm_cell/7` over an entire input
  sequence and return:

      {output_sequence, {new_cell, new_hidden}}

  You can use the output state as the hidden state of another
  LSTM layer.

  ## Options

    * `:name` - layer name.

    * `:activation` - recurrent activation. Defaults to `:tanh`.

    * `:gate` - recurrent gate function. Defaults to `:sigmoid`.

    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`.

    * `:bias_initializer` - initializer for bias weights. Defaults to
      `:zeros`.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`.

  """
  @doc type: :recurrent
  def lstm(
        %Axon{} = x,
        {%Axon{}, %Axon{}} = hidden_state,
        units,
        opts \\ []
      )
      when is_integer(units) and units > 0 and is_list(opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        activation: :tanh,
        gate: :sigmoid,
        unroll: :dynamic,
        use_bias: true,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros
      ])

    activation = opts[:activation]
    gate = opts[:gate]
    unroll = opts[:unroll]

    input_kernel_shape = fn inp, _ -> Axon.Shape.rnn_input_kernel(inp, units, :lstm) end
    hidden_kernel_shape = fn inp, _ -> Axon.Shape.rnn_hidden_kernel(inp, units, :lstm) end
    bias_shape = fn inp, _ -> Axon.Shape.rnn_bias(inp, units, :lstm) end

    kernel_initializer = opts[:kernel_initializer]

    # Parameters
    input_kernel =
      param("input_kernel", {:tuple, List.duplicate(input_kernel_shape, 4)},
        initializer: kernel_initializer
      )

    hidden_kernel =
      param("hidden_kernel", {:tuple, List.duplicate(hidden_kernel_shape, 4)},
        initializer: kernel_initializer
      )

    hidden_state_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "lstm_#{op_counts[:lstm]}_hidden_state"
          end

        name when is_binary(name) ->
          "#{name}_hidden_state"
      end

    hidden_state = Axon.container(hidden_state, name: hidden_state_name)

    {inputs, op} =
      if opts[:use_bias] do
        bias_initializer = opts[:bias_initializer]

        bias =
          param("bias", {:tuple, List.duplicate(bias_shape, 4)}, initializer: bias_initializer)

        {[x, hidden_state, input_kernel, hidden_kernel, bias], :lstm}
      else
        {[x, hidden_state, input_kernel, hidden_kernel], &Axon.Layers.lstm/5}
      end

    output =
      layer(
        op,
        inputs,
        name: opts[:name],
        activation: activation,
        gate: gate,
        unroll: unroll,
        op_name: :lstm
      )

    new_c_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "lstm_#{op_counts[:lstm]}_c_hidden_state"
          end

        name when is_binary(name) ->
          "#{name}_c_hidden_state"
      end

    new_h_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "lstm_#{op_counts[:lstm]}_h_hidden_state"
          end

        name when is_binary(name) ->
          "#{name}_h_hidden_state"
      end

    output_sequence_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "lstm_#{op_counts[:lstm]}_output_sequence"
          end

        name when is_binary(name) ->
          "#{name}_output_sequence"
      end

    output_sequence =
      layer(fn x, _ -> elem(x, 0) end, [output],
        name: output_sequence_name,
        op_name: :elem
      )

    new_c =
      layer(fn x, _ -> elem(elem(x, 1), 0) end, [output],
        name: new_c_name,
        op_name: :elem
      )

    new_h =
      layer(fn x, _ -> elem(elem(x, 1), 1) end, [output],
        name: new_h_name,
        op_name: :elem
      )

    {output_sequence, {new_c, new_h}}
  end

  @doc """
  See `gru/3`.
  """
  @doc type: :recurrent
  def gru(%Axon{} = x, units) do
    gru(x, units, [])
  end

  @doc """
  Adds a gated recurrent unit (GRU) layer to the network with
  a random initial hidden state.

  See `gru/4` for more details.

  ## Additional options

    * `:recurrent_initializer` - initializer for hidden state.
      Defaults to `:orthogonal`.

  """
  @doc type: :recurrent
  def gru(%Axon{} = x, units, opts)
      when is_integer(units) and units > 0
      when is_list(opts) do
    {recurrent_initializer, opts} = Keyword.pop(opts, :recurrent_initializer, :glorot_uniform)
    h = rnn_state(x, units, :gru, opts[:name], "h", recurrent_initializer)
    gru(x, {h}, units, opts)
  end

  def gru(%Axon{} = x, {%Axon{}} = hidden_state, units) when is_integer(units) and units > 0 do
    gru(x, hidden_state, units, [])
  end

  @doc """
  Adds a gated recurrent unit (GRU) layer to the network with
  the given initial hidden state.

  GRUs apply `Axon.Layers.gru_cell/7` over an entire input
  sequence and return:

      {{new_hidden}, output_sequence}

  You can use the output state as the hidden state of another
  GRU layer.

  ## Options

    * `:name` - layer name.

    * `:activation` - recurrent activation. Defaults to `:tanh`.

    * `:gate` - recurrent gate function. Defaults to `:sigmoid`.

    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`.

    * `:bias_initializer` - initializer for bias weights. Defaults to
      `:zeros`.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`.

  """
  @doc type: :recurrent
  def gru(
        %Axon{} = x,
        {%Axon{}} = hidden_state,
        units,
        opts
      )
      when is_integer(units) and units > 0 and is_list(opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        activation: :tanh,
        gate: :sigmoid,
        unroll: :dynamic,
        use_bias: true,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros
      ])

    activation = opts[:activation]
    gate = opts[:gate]
    unroll = opts[:unroll]

    input_kernel_shape = fn inp, _ -> Axon.Shape.rnn_input_kernel(inp, units, :gru) end
    hidden_kernel_shape = fn inp, _ -> Axon.Shape.rnn_hidden_kernel(inp, units, :gru) end
    bias_shape = fn inp, _ -> Axon.Shape.rnn_bias(inp, units, :gru) end

    kernel_initializer = opts[:kernel_initializer]

    input_kernel =
      param("input_kernel", {:tuple, List.duplicate(input_kernel_shape, 3)},
        initializer: kernel_initializer
      )

    hidden_kernel =
      param("hidden_kernel", {:tuple, List.duplicate(hidden_kernel_shape, 3)},
        initializer: kernel_initializer
      )

    hidden_state_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "gru_#{op_counts[:gru]}_hidden_state"
          end

        name when is_binary(name) ->
          "#{name}_hidden_state"
      end

    hidden_state = Axon.container(hidden_state, name: hidden_state_name)

    inputs =
      if opts[:use_bias] do
        bias_initializer = opts[:bias_initializer]

        bias =
          param("bias", {:tuple, List.duplicate(bias_shape, 4)}, initializer: bias_initializer)

        [x, hidden_state, input_kernel, hidden_kernel, bias]
      else
        [x, hidden_state, input_kernel, hidden_kernel]
      end

    output =
      layer(
        :gru,
        inputs,
        name: opts[:name],
        activation: activation,
        gate: gate,
        unroll: unroll,
        op_name: :gru
      )

    new_h_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "gru_#{op_counts[:gru]}_hidden_state"
          end

        name when is_binary(name) ->
          "#{name}_hidden_state"
      end

    output_sequence_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "gru_#{op_counts[:gru]}_output_sequence"
          end

        name when is_binary(name) ->
          "#{name}_output_sequence"
      end

    output_sequence =
      layer(fn x, _ -> elem(x, 0) end, [output],
        name: output_sequence_name,
        op_name: :elem
      )

    new_h =
      layer(fn x, _ -> elem(elem(x, 1), 0) end, [output],
        name: new_h_name,
        op_name: :elem
      )

    {output_sequence, {new_h}}
  end

  @doc """
  See `conv_lstm/3`.
  """
  @doc type: :recurrent
  def conv_lstm(%Axon{} = x, units) when is_integer(units) and units > 0 do
    conv_lstm(x, units, [])
  end

  @doc """
  Adds a convolutional long short-term memory (LSTM) layer to the network
  with a random initial hidden state.

  See `conv_lstm/4` for more details.

  ## Additional options

    * `:recurrent_initializer` - initializer for hidden state. Defaults
      to `:orthogonal`.

  """
  @doc type: :recurrent
  def conv_lstm(%Axon{} = x, units, opts)
      when is_integer(units) and units > 0 and is_list(opts) do
    {recurrent_initializer, opts} = Keyword.pop(opts, :recurrent_initializer, :glorot_uniform)
    c = rnn_state(x, units, :conv_lstm, opts[:name], "c", recurrent_initializer)
    h = rnn_state(x, units, :conv_lstm, opts[:name], "h", recurrent_initializer)
    conv_lstm(x, {c, h}, units, opts)
  end

  def conv_lstm(%Axon{} = x, {%Axon{}, %Axon{}} = hidden_state, units)
      when is_integer(units) and units > 0 do
    conv_lstm(x, hidden_state, units, [])
  end

  @doc """
  Adds a convolutional long short-term memory (LSTM) layer to the network
  with the given initial hidden state..

  ConvLSTMs apply `Axon.Layers.conv_lstm_cell/5` over an entire input
  sequence and return:

      {{new_cell, new_hidden}, output_sequence}

  You can use the output state as the hidden state of another
  ConvLSTM layer.

  ## Options

    * `:name` - layer name.

    * `:padding` - convolutional padding. Defaults to `:same`.

    * `:kernel_size` - convolutional kernel size. Defaults to `1`.

    * `:strides` - convolutional strides. Defaults to `1`.

    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.

    * `:kernel_initializer` - initializer for kernel weights. Defaults
      to `:glorot_uniform`.

    * `:bias_initializer` - initializer for bias weights. Defaults to
      `:zeros`.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`.

  """
  @doc type: :recurrent
  def conv_lstm(
        %Axon{} = x,
        {%Axon{}, %Axon{}} = hidden_state,
        units,
        opts
      )
      when is_integer(units) and units > 0 and is_list(opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        padding: :same,
        kernel_size: 1,
        strides: 1,
        unroll: :dynamic,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true
      ])

    padding = opts[:padding]
    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    unroll = opts[:unroll]
    kernel_initializer = opts[:kernel_initializer]

    hidden_kernel_shape = fn _, {inp, _} ->
      shape = Tuple.delete_at(inp, 1)
      Axon.Shape.conv_kernel(shape, 4 * units, kernel_size, :first, 1)
    end

    input_kernel_shape = fn inp, _ ->
      shape = Tuple.delete_at(inp, 1)
      Axon.Shape.conv_kernel(shape, 4 * units, kernel_size, :first, 1)
    end

    bias_shape = fn inp, _ ->
      shape = Tuple.delete_at(inp, 1)
      Axon.Shape.conv_bias(shape, 4 * units, kernel_size, :first, 1)
    end

    wi = param("input_kernel", {:tuple, [input_kernel_shape]}, initializer: kernel_initializer)
    wh = param("hidden_kernel", {:tuple, [hidden_kernel_shape]}, initializer: kernel_initializer)

    hidden_state_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "conv_lstm_#{op_counts[:conv_lstm]}_hidden_state"
          end

        name when is_binary(name) ->
          "#{name}_hidden_state"
      end

    hidden_state = Axon.container(hidden_state, name: hidden_state_name)

    {inputs, op} =
      if opts[:use_bias] do
        bias_initializer = opts[:bias_initializer]
        b = param("bias", {:tuple, [bias_shape]}, initializer: bias_initializer)
        {[x, hidden_state, wi, wh, b], :conv_lstm}
      else
        {[x, hidden_state, wi, wh], :conv_lstm}
      end

    output =
      layer(
        op,
        inputs,
        name: opts[:name],
        conv_opts: [
          strides: strides,
          padding: padding
        ],
        unroll: unroll,
        op_name: :conv_lstm
      )

    new_c_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "conv_lstm_#{op_counts[:lstm]}_c_hidden_state"
          end

        name when is_binary(name) ->
          "#{name}_c_hidden_state"
      end

    new_h_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "conv_lstm_#{op_counts[:lstm]}_h_hidden_state"
          end

        name when is_binary(name) ->
          "#{name}_h_hidden_state"
      end

    output_sequence_name =
      case opts[:name] do
        nil ->
          fn _, op_counts ->
            "conv_lstm_#{op_counts[:lstm]}_output_sequence"
          end

        name when is_binary(name) ->
          "#{name}_output_sequence"
      end

    output_sequence =
      layer(fn x, _ -> elem(x, 0) end, [output],
        name: output_sequence_name,
        op_name: :elem
      )

    new_c =
      layer(fn x, _ -> elem(elem(x, 1), 0) end, [output],
        name: new_c_name,
        op_name: :elem
      )

    new_h =
      layer(fn x, _ -> elem(elem(x, 1), 1) end, [output],
        name: new_h_name,
        op_name: :elem
      )

    {output_sequence, {new_c, new_h}}
  end

  defp rnn_state(x, units, rnn_type, parent_name, state_name, initializer) do
    initializer = initializer || :glorot_uniform
    key = Nx.Random.key(:erlang.system_time()) |> Nx.backend_copy(Nx.Defn.Expr)

    name =
      case parent_name do
        nil ->
          fn _, op_counts ->
            "lstm_#{op_counts[rnn_type]}_#{state_name}_hidden_state"
          end

        parent_name when is_binary(parent_name) ->
          "#{parent_name}_#{state_name}_hidden_state"
      end

    fun = fn inputs, opts ->
      shape = Axon.Shape.rnn_hidden_state(Nx.shape(inputs), units, rnn_type)

      case initializer do
        fun when is_function(fun) ->
          fun.(shape)

        fun when is_atom(fun) ->
          fun = apply(Axon.Initializers, fun, [])
          {:arity, arity} = Function.info(fun, :arity)

          cond do
            arity == 2 ->
              fun.(shape, {:f, 32})

            arity == 3 ->
              fun.(shape, {:f, 32}, opts[:key])

            true ->
              raise ArgumentError, "bad arity for initializer"
          end
      end
    end

    layer(fun, [x], name: name, op_name: :recurrent_state, key: key)
  end

  @doc """
  Adds an embedding layer to the network.

  An embedding layer initializes a kernel of shape `{vocab_size, embedding_size}`
  which acts as a lookup table for sequences of discrete tokens (e.g. sentences).
  Embeddings are typically used to obtain a dense representation of a sparse input
  space.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights. Defaults
      to `:uniform`.

  """
  @doc type: :linear
  def embedding(%Axon{} = x, vocab_size, embedding_size, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, kernel_initializer: :uniform])

    kernel_shape = &Axon.Shape.embedding_kernel(&1, vocab_size, embedding_size)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    layer(:embedding, [x, kernel], name: opts[:name], op_name: :embedding)
  end

  @doc """
  Adds a bias layer to the network.

  A bias layer simply adds a trainable bias to an input.

  ## Options

    * `:name` - layer name.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`.

  """
  @doc type: :linear
  def bias(%Axon{} = x, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, bias_initializer: :zeros])

    bias_shape = fn shape -> {elem(shape, tuple_size(shape) - 1)} end
    bias = param("bias", bias_shape, initializer: opts[:bias_initializer])

    layer(:bias, [x, bias], name: opts[:name], op_name: :bias)
  end

  @doc """
  Adds a stack columns layer to the network.

  A stack columns layer is designed to be used with `Nx.LazyContainer`
  data structures like Explorer DataFrames. Given an input which is a
  DataFrame, `Axon.stack_container/2` will stack the columns in each
  row to create a single vector.

  You may optionally specify `:ignore` to ignore certain columns in
  the container.

  ## Options

    * `:name` - layer name.

    * `:ignore` - keys to ignore when stacking.
  """
  @doc type: :special
  def stack_columns(%Axon{} = x, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, ignore: []])

    layer(:stack_columns, [x], name: opts[:name], ignore: opts[:ignore], op_name: :stack_columns)
  end

  @doc """
  Freezes parameters returned from `fun` in the given model.

  `fun` takes the model's parameter list and returns the list of
  parameters it wishes to freeze. `fun` defaults to the identity
  function, freezing all of the parameters in `model`.

  Freezing parameters is useful when performing transfer learning
  to leverage features learned from another problem in a new problem.
  For example, it's common to combine the convolutional base from
  larger models trained on ImageNet with fresh fully-connected classifiers.
  The combined model is then trained on fresh data, with the convolutional
  base frozen so as not to lose information. You can see this example
  in code here:

      cnn_base = get_pretrained_cnn_base()
      model =
        cnn_base
        |> Axon.freeze()
        |> Axon.flatten()
        |> Axon.dense(1024, activation: :relu)
        |> Axon.dropout()
        |> Axon.dense(1000, activation: :softmax)

      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(0.005))
      |> Axon.Loop.run(data, epochs: 10)

  When compiled, frozen parameters are wrapped in `Nx.Defn.Kernel.stop_grad/1`,
  which zeros out the gradient with respect to the frozen parameter. Gradients
  of frozen parameters will return `0.0`, meaning they won't be changed during
  the update process.
  """
  @doc type: :model
  def freeze(%Axon{} = model, fun \\ & &1) when is_function(fun, 1) do
    parameters =
      reduce_nodes(model, MapSet.new(), fn %Axon.Node{parameters: params}, acc ->
        Enum.reduce(params, acc, fn param, acc ->
          MapSet.put(acc, param)
        end)
      end)

    parameters_to_freeze = fun.(Enum.to_list(parameters))

    map_nodes(model, fn %Axon.Node{parameters: params} = axon_node ->
      frozen_params =
        Enum.map(params, fn %{id: param_id} = v ->
          if Enum.any?(parameters_to_freeze, fn %{id: id} -> param_id == id end) do
            %{v | frozen: true}
          else
            v
          end
        end)

      %{axon_node | parameters: frozen_params}
    end)
  end

  @doc """
  Attaches a hook to the given Axon model.

  Hooks compile down to `Nx.Defn.Kernel.hook/3` and provide the same
  functionality for adding side-effecting operations to a compiled
  model. For example, you can use hooks to inspect intermediate activations,
  send data to an external service, and more.

  Hooks can be configured to be invoked on the following events:

    * `:initialize` - on model initialization.
    * `:pre_forward` - before layer forward pass is invoked.
    * `:forward` - after layer forward pass is invoked.
    * `:backward` - after layer backward pass is invoked.

  To invoke a hook on every single event, you may pass `:all` to `on:`.

      Axon.input("input", shape: {nil, 1}) |> Axon.attach_hook(&IO.inspect/1, on: :all)

  The default event is `:forward`, assuming you want a hook invoked
  on the layers forward pass.

  You may configure hooks to run in one of only training or inference
  mode using the `:mode` option. The default mode is `:both` to be invoked
  during both train and inference mode.

      Axon.input("input", shape: {nil, 1}) |> Axon.attach_hook(&IO.inspect/1, on: :forward, mode: :train)

  You can also attach multiple hooks to a single layer. Hooks are invoked in
  the order in which they are declared. If order is important, you should attach
  hooks in the order you want them to be executed:

      Axon.input("input", shape: {nil, 1})
      # I will be executed first
      |> Axon.attach_hook(&IO.inspect/1)
      # I will be executed second
      |> Axon.attach_hook(fn _ -> IO.write("HERE") end)

  Hooks are executed at their point of attachment. You must insert hooks at each point
  you want a hook to execute during model execution.

      Axon.input("input", shape: {nil, 1})
      |> Axon.attach_hook(&IO.inspect/1)
      |> Axon.relu()
      |> Axon.attach_hook(&IO.inspect/1)

  """
  @doc type: :debug
  def attach_hook(%Axon{output: id, nodes: nodes} = axon, fun, opts \\ []) do
    opts = Keyword.validate!(opts, on: :forward, mode: :both)
    on_event = opts[:on]
    mode = opts[:mode]

    updated_nodes =
      Map.update!(nodes, id, fn axon_node ->
        %{axon_node | hooks: [{on_event, mode, fun}]}
      end)

    %{axon | nodes: updated_nodes}
  end

  ## Graph Manipulation and Utilities

  # TODO: Revisit later with new decoupled structs
  # e.g. there should be a node API and graph API

  @doc """
  Returns a node's immediate parameters.

  Note this does not take into account parameters of
  parent layers - only the parameters which belong to
  the immediate layer.
  """
  @doc type: :graph
  def get_parameters(%Axon{output: id, nodes: nodes}) do
    Access.get(nodes, [id, :parameters])
  end

  @doc """
  Sets a node's immediate parameters to the given
  parameters.

  Note this does not take into account parameters of
  parent layers - only the parameters which belong to
  the immediate layer.

  The new parameters must be compatible with the layer's
  old parameters.
  """
  @doc type: :graph
  def set_parameters(%Axon{output: id, nodes: nodes} = axon, new_params) do
    # TODO: Check compatibility
    updated_nodes =
      Map.update!(nodes, id, fn axon_node ->
        %{axon_node | parameters: new_params}
      end)

    %{axon | nodes: updated_nodes}
  end

  @doc """
  Returns a node's immediate input options.

  Note that this does not take into account options of
  parent layers, only the option which belong to the
  immediate layer.
  """
  @doc type: :graph
  def get_options(%Axon{output: id, nodes: nodes}) do
    Access.get(nodes, [id, :opts])
  end

  @doc """
  Sets a node's immediate options to the given input
  options.

  Note that this does not take into account options of
  parent layers, only the option which belong to the
  immediate layer.

  New options must be compatible with the given layer
  op. Adding unsupported options to an Axon layer will
  result in an error at graph execution time.
  """
  @doc type: :graph
  def set_options(%Axon{output: id, nodes: nodes} = axon, new_opts) do
    updated_nodes =
      Map.update!(nodes, id, fn axon_node ->
        %{axon_node | opts: new_opts}
      end)

    %{axon | nodes: updated_nodes}
  end

  @doc """
  Returns information about a model's inputs.
  """
  @doc type: :graph
  def get_inputs(%Axon{} = axon) do
    reduce_nodes(axon, %{}, fn
      %Axon.Node{op: :input, name: name, opts: opts}, inputs ->
        name = name.(:input, %{})
        Map.put(inputs, name, opts[:shape])

      _, inputs ->
        inputs
    end)
  end

  @doc """
  Returns a model's output shape from the given input
  template.
  """
  @doc type: :graph
  def get_output_shape(%Axon{} = axon, inputs, opts \\ []) do
    {init_fn, forward_fn} = build(axon, opts)

    out =
      Nx.Defn.jit(
        fn inputs ->
          forward_fn.(init_fn.(inputs, %{}), inputs)
        end,
        compiler: Axon.Defn
      ).(inputs)

    safe_shape(out)
  end

  defp safe_shape(container_or_tensor) do
    case container_or_tensor do
      %Axon.None{} = none ->
        none

      %Nx.Tensor{} = tensor ->
        Nx.shape(tensor)

      container ->
        deep_new(container, &Nx.shape/1)
    end
  end

  @doc """
  Returns a map of model op counts for each unique operation
  in a model by their given `:op_name`.

  ## Examples

      iex> model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      iex> Axon.get_op_counts(model)
      %{input: 1, dense: 1}

      iex> model = Axon.input("input", shape: {nil, 1}) |> Axon.tanh() |> Axon.tanh()
      iex> Axon.get_op_counts(model)
      %{input: 1, tanh: 2}

  """
  @doc type: :graph
  def get_op_counts(%Axon{} = axon) do
    reduce_nodes(axon, %{}, fn %Axon.Node{op: op}, op_counts ->
      Map.update(op_counts, op, 1, fn x -> x + 1 end)
    end)
  end

  @doc """
  Traverses graph nodes in order, applying `fun` to each
  node exactly once to return a transformed node in its
  place(s) in the graph.

  This function maintains an internal cache which ensures
  each node is only visited and transformed exactly once.

  `fun` must accept an Axon node and return an Axon node.

  Please note that modifying node lineage (e.g. altering
  a node's parent) will result in disconnected graphs.

  ## Examples

  One common use of this function is to implement common
  instrumentation between layers without needing to build
  a new explicitly instrumented version of a model. For example,
  you can use this function to visualize intermediate activations
  of all convolutional layers in a model:

      instrumented_model = Axon.  (model, fn
        %Axon{op: :conv} = graph ->
          Axon.attach_hook(graph, &visualize_activations/1)

        graph ->
          graph
      end)

  Another use case is to replace entire classes of layers
  with another. For example, you may want to replace all
  relu layers with tanh layers:

      new_model = Axon.map_nodes(model, fn
        %Axon{op: :relu} = graph ->
          # Get nodes immediate parent
          parent = Axon.get_parent(graph)
          # Replace node with a tanh
          Axon.tanh(parent)

        graph ->
          graph
      end)

  """
  @doc type: :graph
  def map_nodes(%Axon{nodes: nodes} = axon, fun) when is_function(fun, 1) do
    updated_nodes = Map.new(nodes, fn {id, axon_node} -> {id, fun.(axon_node)} end)
    %{axon | nodes: updated_nodes}
  end

  @doc """
  Traverses graph nodes in order, applying `fun` to each
  node exactly once to return a transformed node in its
  place(s) in the graph.

  This function maintains an internal cache which ensures
  each node is only visited and transformed exactly once.

  `fun` must accept an Axon node and accumulator and return
  an updated accumulator.

  ## Examples

  Internally this function is used in several places to accumulate
  graph metadata. For example, you can use it to count the number
  of a certain type of operation in the graph:

      Axon.reduce_nodes(model, 0, fn
        %Axon.Nodes{op: :relu}, acc -> acc + 1
        _, acc -> acc
      end)

  """
  @doc type: :graph
  def reduce_nodes(%Axon{nodes: nodes}, acc, fun) when is_function(fun, 2) do
    nodes
    |> Map.values()
    |> Enum.reduce(acc, fun)
  end

  defp deep_map_reduce(leaf, acc, fun) when is_integer(leaf), do: fun.(leaf, acc)

  defp deep_map_reduce(container, acc, fun) do
    Nx.Container.traverse(container, acc, &recur_deep_map_reduce(&1, &2, fun))
  end

  defp recur_deep_map_reduce(leaf, acc, fun) do
    case leaf do
      %Axon{} = leaf ->
        fun.(leaf, acc)

      %Nx.Tensor{} = leaf ->
        fun.(leaf, acc)

      container ->
        deep_map_reduce(container, acc, fun)
    end
  end

  @doc """
  Builds the given model to `{init_fn, predict_fn}`.

  Once built, a model can be passed as argument to `Nx.Defn`.

  ## `init_fn`

  The `init_fn` receives two arguments, the input template and
  an optional map with initial parameters for layers or namespaces:

      {init_fn, predict_fn} = Axon.build(model)
      init_fn.(Nx.template({1, 1}, {:f, 32}), %{"dense_0" => dense_params})

  ## `predict_fn`

  The `predict_fn` receives two arguments, the trained parameters
  and the actual inputs:

      {_init_fn, predict_fn} = Axon.build(model, opts)
      predict_fn.(params, input)

  ## Options

    * `:mode` - one of `:inference` or `:training`. Forwarded to layers
      to control differences in compilation at training or inference time.
      Defaults to `:inference`

    * `:debug` - if `true`, will log graph traversal and generation
      metrics. Also forwarded to JIT if debug mode is available
      for your chosen compiler or backend. Defaults to `false`

  All other options are forwarded to the default JIT compiler
  or backend.
  """
  @doc type: :model
  def build(model, opts \\ []) when is_list(opts) do
    {init_fn, predict_fn} = Axon.Compiler.build(model, opts)
    opts = [on_conflict: :reuse] ++ opts
    {Nx.Defn.jit(init_fn, opts), Nx.Defn.jit(predict_fn, opts)}
  end

  @doc """
  Compiles the given model to `{init_fn, predict_fn}`.

  This function will compile a model specialized to the given
  input shapes and types. This is useful for avoiding the overhead
  of long compilations at program runtime. You must provide template
  inputs which match the expected shapes and types of inputs at
  execution time.

  This function makes use of the built-in `Nx.Defn.compile/3`. Note
  that passing inputs which differ in shape or type from the templates
  provided to this function will result in potentially expensive
  recompilation.
  """
  @doc type: :model
  def compile(model, template, init_params \\ %{}, opts \\ []) when is_list(opts) do
    {init_fn, predict_fn} = build(model, opts)
    init_params = Nx.Defn.jit_apply(init_fn, [template, init_params], opts)
    predict_compiled_fn = Nx.Defn.compile(predict_fn, [init_params, template], opts)
    {init_params, predict_compiled_fn}
  end

  @doc """
  Compiles and returns the given model's init function
  expression with the given options.

  The returned expression is an Nx expression which can be
  traversed and lowered to an IR or inspected for debugging
  purposes.

  You may optionally specify initial parameters for some layers or
  namespaces by passing a partial parameter map:

      Axon.trace_init(model, %{"dense_0" => dense_params})

  The parameter map will be merged with the initialized model
  parameters.

  ## Options

    * `:debug` - if `true`, will log graph traversal and generation
      metrics. Also forwarded to JIT if debug mode is available
      for your chosen compiler or backend. Defaults to `false`

  """
  @doc type: :debug
  def trace_init(model, template, params \\ %{}, opts \\ []) do
    {init_fn, _} = build(model, opts)
    Nx.Defn.jit(init_fn, compiler: Axon.Defn).(template, params)
  end

  @doc """
  Compiles and returns the given model's forward function
  expression with the given options.

  The returned expression is an Nx expression which can be
  traversed and lowered to an IR or inspected for debugging
  purposes.

  ## Options

    * `:mode` - one of `:inference` or `:training`. Forwarded to layers
      to control differences in compilation at training or inference time.
      Defaults to `:inference`

    * `:debug` - if `true`, will log graph traversal and generation
      metrics. Also forwarded to JIT if debug mode is available
      for your chosen compiler or backend. Defaults to `false`

  """
  @doc type: :debug
  def trace_forward(model, inputs, params, opts \\ []) when is_list(opts) do
    {_, forward_fun} = build(model, opts)
    Nx.Defn.jit(forward_fun, compiler: Axon.Defn).(params, inputs)
  end

  @doc """
  Compiles and returns the given model's backward function
  expression with respect to the given loss function.

  The returned expression is an Nx expression which can be
  traversed and lowered to an IR or inspected for debugging
  purposes.

  The given loss function must be a scalar loss function which
  expects inputs and targets with the same shapes as the model's
  output shapes as determined by the model's signature.

  ## Options

    * `:debug` - if `true`, will log graph traversal and generation
      metrics. Also forwarded to JIT if debug mode is available
      for your chosen compiler or backend. Defaults to `false`

  """
  @doc type: :debug
  def trace_backward(model, inputs, params, loss, opts \\ []) do
    {_, forward_fn} = build(model, opts)

    backward_fn = fn params, inputs, targets ->
      Nx.Defn.grad(params, fn params ->
        %{prediction: preds} = forward_fn.(params, inputs)
        loss.(targets, preds)
      end)
    end

    outputs = Nx.Defn.jit(forward_fn, compiler: Axon.Defn).(params, inputs)
    inputs = [params, inputs, outputs]

    apply(Nx.Defn.jit(backward_fn, compiler: Axon.Defn), inputs)
  end

  @doc false
  @deprecated "Use Axon.build/2 instead"
  def init(model, template, params \\ %{}, opts \\ []) when is_list(opts) do
    {init_fn, _predict_fn} = build(model, opts)
    init_fn.(template, params)
  end

  @doc """
  Compiles and runs the given Axon model with `params` on
  `input` with the given compiler options.

  ## Options

    * `:mode` - one of `:inference` or `:training`. Forwarded to layers
      to control differences in compilation at training or inference time.
      Defaults to `:inference`

    * `:debug` - if `true`, will log graph traversal and generation
      metrics. Also forwarded to JIT if debug mode is available
      for your chosen compiler or backend. Defaults to `false`

  All other options are forwarded to the default JIT compiler
  or backend.
  """
  @doc type: :model
  def predict(%Axon{} = model, params, input, opts \\ []) when is_list(opts) do
    {_init_fn, predict_fn} = build(model, opts)
    predict_fn.(params, input)
  end

  ## Inspection

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(%Axon{output: id, nodes: nodes} = axon, opts) do
      inputs =
        axon
        |> Axon.get_inputs()
        |> Enum.sort()
        |> Map.new()

      op_counts = Axon.get_op_counts(axon)
      %Axon.Node{op_name: op_name, name: name_fn} = nodes[id]
      op_counts = Map.update(op_counts, op_name, 0, fn x -> x - 1 end)
      output_name = name_fn.(op_name, op_counts)

      node_count = Enum.count(axon.nodes)

      inner =
        concat([
          line(),
          "inputs: #{inspect(inputs)}",
          line(),
          "outputs: #{inspect(output_name)}",
          line(),
          "nodes: #{inspect(node_count)}"
        ])

      force_unfit(
        concat([
          color("#Axon<", :map, opts),
          nest(inner, 2),
          line(),
          color(">", :map, opts)
        ])
      )
    end
  end

  # Serialization

  @doc """
  Serializes a model and its parameters for persisting
  models to disk or elsewhere.

  Model and parameters are serialized as a tuple, where the
  model is converted to a recursive map to ensure compatibility
  with future Axon versions and the parameters are serialized
  using `Nx.serialize/2`. There is some additional metadata included
  such as current serialization version for compatibility.

  Serialization `opts` are forwarded to `Nx.serialize/2` and
  `:erlang.term_to_binary/2` for controlling compression options.

  ## Examples

      iex> model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, kernel_initializer: :zeros, activation: :relu)
      iex> {init_fn, _} = Axon.build(model)
      iex> params = init_fn.(Nx.template({1, 2}, :f32), %{})
      iex> serialized = Axon.serialize(model, params)
      iex> {saved_model, saved_params} = Axon.deserialize(serialized)
      iex> {_, predict_fn} = Axon.build(saved_model)
      iex> predict_fn.(saved_params, Nx.tensor([[1.0, 1.0]]))
      #Nx.Tensor<
        f32[1][1]
        [
          [0.0]
        ]
      >

  """
  @doc type: :model
  def serialize(%Axon{output: id, nodes: nodes}, params, opts \\ []) do
    Logger.warning(
      "Attempting to serialize an Axon model. Serialiation is discouraged" <>
        " and will be deprecated, then removed in future releases. You should" <>
        " keep your model definitions as code and serialize your parameters using" <>
        " `Nx.serialize/2`."
    )

    nodes =
      Map.new(nodes, fn {k, %{op: op, op_name: op_name} = v} ->
        validate_serialized_op!(op_name, op)
        node_meta = Map.from_struct(v)
        {k, Map.put(node_meta, :node, :node)}
      end)

    model_meta = %{output: id, nodes: nodes, axon: :axon}
    params = Nx.serialize(params, opts)
    :erlang.term_to_binary({@file_version, model_meta, params}, opts)
  end

  # TODO: Raise on next release
  defp validate_serialized_op!(op_name, op) when is_function(op) do
    fun_info = Function.info(op)

    case fun_info[:type] do
      :local ->
        Logger.warning(
          "Attempting to serialize anonymous function in #{inspect(op_name)} layer," <>
            " this will result in errors during deserialization between" <>
            " different processes, and will be unsupported in a future" <>
            " release. You should instead use a fully-qualified MFA function" <>
            " such as &Axon.Layers.dense/3"
        )

      {:type, :external} ->
        :ok
    end
  end

  defp validate_serialized_op!(_name, op) when is_atom(op), do: :ok

  @doc """
  Deserializes serialized model and parameters into a `{model, params}`
  tuple.

  It is the opposite of `Axon.serialize/3`.

  ## Examples

      iex> model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, kernel_initializer: :zeros, activation: :relu)
      iex> {init_fn, _} = Axon.build(model)
      iex> params = init_fn.(Nx.template({1, 2}, :f32), %{})
      iex> serialized = Axon.serialize(model, params)
      iex> {saved_model, saved_params} = Axon.deserialize(serialized)
      iex> {_, predict_fn} = Axon.build(saved_model)
      iex> predict_fn.(saved_params, Nx.tensor([[1.0, 1.0]]))
      #Nx.Tensor<
        f32[1][1]
        [
          [0.0]
        ]
      >

  """
  @doc type: :model
  def deserialize(serialized, opts \\ []) do
    Logger.warning(
      "Attempting to deserialize a serialized Axon model. Deserialization" <>
        " is discouraged and will be deprecated, then removed in future" <>
        " releases. You should keep your model definitions as code and" <>
        " serialize your parameters using `Nx.serialize/2`."
    )

    {1, model_meta, serialized_params} = :erlang.binary_to_term(serialized, opts)
    %{nodes: nodes, output: id} = model_meta

    nodes =
      Map.new(nodes, fn {k, %{op_name: op_name, op: op} = v} ->
        validate_deserialized_op!(op_name, op)

        node_struct =
          v
          |> Map.delete(:node)
          |> then(&struct(Axon.Node, &1))

        {k, node_struct}
      end)

    model = %Axon{output: id, nodes: nodes}
    params = Nx.deserialize(serialized_params, opts)
    {model, params}
  end

  # TODO: Raise on next release
  defp validate_deserialized_op!(op_name, op) when is_function(op) do
    fun_info = Function.info(op)

    case fun_info[:type] do
      :local ->
        Logger.warning(
          "Attempting to deserialize anonymous function in #{inspect(op_name)} layer," <>
            " this will result in errors during deserialization between" <>
            " different processes, and will be unsupported in a future" <>
            " release"
        )

      :external ->
        unless function_exported?(fun_info[:module], fun_info[:name], fun_info[:arity]) do
          Logger.warning(
            "Attempting to deserialize model which depends on function" <>
              " #{inspect(op)} in layer #{inspect(op_name)} which does not exist in" <>
              " the current environment, check your dependencies"
          )
        end
    end
  end

  defp validate_deserialized_op!(op, _op_name) when is_atom(op), do: :ok

  ## Helpers

  @valid_initializers [:zeros, :ones, :uniform, :normal, :identity] ++
                        [:lecun_uniform, :lecun_normal, :he_uniform, :he_normal] ++
                        [:glorot_uniform, :glorot_normal, :variance_scaling]

  defp validate_initializer!(initializer)
       when is_atom(initializer) and initializer in @valid_initializers do
    apply(Axon.Initializers, initializer, [])
  end

  defp validate_initializer!(initializer) when is_function(initializer, 2) do
    initializer
  end

  defp validate_initializer!(initializer) when is_function(initializer, 3) do
    initializer
  end

  defp validate_initializer!(initializer) do
    raise ArgumentError,
          "initializer must be one of #{inspect(@valid_initializers)}," <>
            " or an arity-3 function accepting initializer shape, type, and key" <>
            " got #{inspect(initializer)}"
  end

  # Names are generated lazily at inspect, initialization, and compile
  # time, so for name we return a function which takes `op` and `op_count`
  # and returns a unique name for the given model.
  defp unique_identifiers(type, nil) do
    id = System.unique_integer([:positive, :monotonic])

    name = fn op, op_counts ->
      count = op_counts[op] || 0
      Atom.to_string(type) <> "_#{count}"
    end

    {id, name}
  end

  defp unique_identifiers(_type, name_fn) when is_function(name_fn, 2) do
    id = System.unique_integer([:positive, :monotonic])
    {id, name_fn}
  end

  defp unique_identifiers(_type, name) when is_binary(name) do
    {System.unique_integer([:positive, :monotonic]), fn _, _ -> name end}
  end

  defp unique_identifiers(_, name) do
    raise ArgumentError,
          "expected layer name to be a binary, a function or nil, " <>
            "got: #{inspect(name)}"
  end
end
