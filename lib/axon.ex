defmodule Axon do
  @moduledoc """
  A high-level interface for creating neural network models.

  Axon is built entirely on top of Nx numerical definitions,
  so every neural network can be JIT or AOT compiled using
  any Nx compiler, or even transformed into high-level neural
  network formats like TensorFlow Lite and
  [ONNX](https://github.com/elixir-nx/axon_onnx).

  ## Model Creation

  All Axon models start with an input layer, specifying the
  expected input shape of the training data:

      input = Axon.input({nil, 784}, "input")

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

      ---------------------------------------------------------------------------------------------------------
                                                        Model
      =========================================================================================================
       Layer                                   Shape        Policy              Parameters   Parameters Memory
      =========================================================================================================
       input ( input )                         {nil, 784}   p=f32 c=f32 o=f32   0            0 bytes
       dense_0 ( dense["input"] )              {nil, 128}   p=f32 c=f32 o=f32   100480       401920 bytes
       relu_0 ( relu["dense_0"] )              {nil, 128}   p=f32 c=f32 o=f32   0            0 bytes
       batch_norm_0 ( batch_norm["relu_0"] )   {nil, 128}   p=f32 c=f32 o=f32   512          2048 bytes
       dropout_0 ( dropout["batch_norm_0"] )   {nil, 128}   p=f32 c=f32 o=f32   0            0 bytes
       dense_1 ( dense["dropout_0"] )          {nil, 64}    p=f32 c=f32 o=f32   8256         33024 bytes
       tanh_0 ( tanh["dense_1"] )              {nil, 64}    p=f32 c=f32 o=f32   0            0 bytes
       dense_2 ( dense["tanh_0"] )             {nil, 10}    p=f32 c=f32 o=f32   650          2600 bytes
       softmax_0 ( softmax["dense_2"] )        {nil, 10}    p=f32 c=f32 o=f32   0            0 bytes
      ---------------------------------------------------------------------------------------------------------
      Total Parameters: 109898
      Total Parameters Memory: 439592 bytes
      Inputs: %{"input" => {nil, 784}}

  ### Multiple Inputs

  Creating a model with multiple inputs is as easy as declaring an
  additional input in your Axon graph. Every input layer present in
  the final Axon graph will be required to be passed as input at the
  time of model execution.

      inp1 = Axon.input({nil, 1}, "input_0")
      inp2 = Axon.input({nil, 1}, "input_1")

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

      inp1 = Axon.input({nil, 1}, "input_0")
      inp2 = Axon.input({nil, 1}, "input_1")

      model1 = Axon.add(inp1, inp2)
      params1 = Axon.init(model1)
      # Inputs are referenced by name
      Axon.predict(model1, params1, %{"input_0" => x, "input_1" => y})

  ### Multiple Outputs

  Nx offers robust [container](https://hexdocs.pm/nx/Nx.Container.html) support
  which is extended to Axon. Axon allows you to wrap any valid Nx container
  in a layer. Containers are most commonly used to structure outputs:

      inp1 = Axon.input({nil, 1}, "input_0")
      inp2 = Axon.input({nil, 1}, "input_1")
      model = Axon.container(%{foo: inp1, bar: inp2})

  Containers can be arbitrarily nested:

      inp1 = Axon.input({nil, 1}, "input_0")
      inp2 = Axon.input({nil, 1}, "input_1")
      model = Axon.container({%{foo: {inp1, %{bar: inp2}}}})

  You can even use custom structs which implement the container protocol:

      inp1 = Axon.input({nil, 1}, "input_0")
      inp2 = Axon.input({nil, 1}, "input_1")
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

      weight = Axon.param(input_shape, "weight")

  To create a custom layer, you "wrap" your implementation and inputs into
  a layer using `Axon.layer`. You'll notice the API mirrors Elixir's `apply`:

      def atan2_layer(%Axon{output_shape: shape} = input) do
        weight = Axon.param(input_shape, "weight")
        Axon.layer(&my_layer/3, [input, weight])
      end

  ## Model Execution

  Under the hood, Axon models are represented as Elixir structs. You
  can initialize and apply models using the macros `Axon.init/3` and
  `Axon.predict/4`:

      params = Axon.init(model, compiler: EXLA)

      Axon.predict(model, params, inputs, compiler: EXLA, mode: :train)

  It is suggested that you set compiler options globally rather than pass
  them as options to execution macros:
      
      EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

      params = Axon.init(model)
      Axon.predict(model, params, inputs, mode: :train)

  `Axon.predict/4` by default runs in inference mode, which performs certain
  optimizations and removes layers such as dropout layers. If constructing
  a training step using `Axon.predict/4`, be sure to specify `mode: :train`.

  ## Model Training

  Combining the Axon model creation API with the optimization and training
  APIs, you can create and train neural networks with ease:

      model =
        Axon.input({nil, 784}, "input_0")
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

  # Axon serialization version
  @file_version 1

  @empty_tensor Nx.tensor(-1)

  @type t :: %__MODULE__{}

  defstruct [
    :id,
    :name,
    :output_shape,
    :parent,
    :parameters,
    :args,
    :op,
    :policy,
    :hooks,
    :opts,
    :op_name
  ]

  @doc """
  Custom Axon layer with given inputs.

  Inputs may be other Axon layers or trainable parameters created
  with `Axon.param`. At inference time, `op` will be applied with
  inputs in specified order and an additional `opts` parameter which
  specifies inference options. All options passed to layer are forwarded
  to inference function except:

    * `:shape` - specify layer output shape to bypass shape inference.
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
    {inputs, params, args, input_shapes} = split_inputs(op, inputs)

    inputs = Enum.reverse(inputs)
    params = Enum.reverse(params)
    args = Enum.reverse(args)
    input_shapes = Enum.reverse(input_shapes)

    {name, opts} = Keyword.pop(opts, :name)
    {shape, opts} = Keyword.pop(opts, :shape)
    {op_name, opts} = Keyword.pop(opts, :op_name, :custom)

    {id, name} = unique_identifiers(op_name, name)

    output_shape =
      if shape do
        shape
      else
        infer_shape(input_shapes, op, opts)
      end

    %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: inputs,
      parameters: params,
      args: args,
      op: op,
      policy: Axon.MixedPrecision.create_policy(),
      hooks: [],
      opts: opts,
      op_name: op_name
    }
  end

  defp split_inputs(:container, [container] = inputs) do
    input_shapes = deep_new(container, fn %Axon{output_shape: shape} -> shape end)
    args = [:layer]
    params = []
    {inputs, params, args, [input_shapes]}
  end

  defp split_inputs(_op, inputs) do
    Enum.reduce(inputs, {[], [], [], []}, fn
      %Axon{output_shape: shape} = layer, {layers, params, args, shapes} ->
        {[layer | layers], params, [:layer | args], [shape | shapes]}

      %Parameter{shape: shape} = param, {layers, params, args, shapes} ->
        {layers, [param | params], [:parameter | args], [shape | shapes]}

      invalid, _ ->
        raise ArgumentError, "invalid input given to layer: #{inspect(invalid)}"
    end)
  end

  defp infer_shape(input_shapes, fun, opts) do
    {inputs, indices} =
      Enum.reduce(input_shapes, {[], []}, fn shape, {input_shapes, indices} ->
        {template, template_indices} = template_shape(shape)
        {[template | input_shapes], [template_indices | indices]}
      end)

    inputs = Enum.reverse(inputs)

    opts = Keyword.put(opts, :mode, :inference)

    wrapper_fun = fn tensors ->
      tensors = Tuple.to_list(tensors)
      apply(fun, tensors ++ [opts])
    end

    expr = Nx.Defn.jit(wrapper_fun, [List.to_tuple(inputs)], compiler: Axon.Defn)

    indices = Enum.map(indices, &MapSet.new/1)

    indices_that_are_1 =
      deep_new(expr, fn input ->
        indices =
          input
          |> Nx.shape()
          |> Tuple.to_list()
          |> Enum.with_index()
          |> Enum.filter(fn {x, _} -> x == 1 end)
          |> Enum.map(&elem(&1, 1))

        # This is a hack because containers don't like
        # lists as leaf values, but they do like tensors
        if indices == [] do
          @empty_tensor
        else
          Nx.tensor(indices)
        end
      end)

    deduped_nil_indices = Enum.reduce(indices, MapSet.new(), &MapSet.union/2)

    deep_merge(expr, indices_that_are_1, fn input, indices_tensor ->
      shape = Nx.shape(input)

      indices =
        if indices_tensor == @empty_tensor do
          []
        else
          Nx.to_flat_list(indices_tensor)
        end

      indices_to_make_nil = MapSet.intersection(deduped_nil_indices, MapSet.new(indices))

      Enum.reduce(indices_to_make_nil, shape, fn i, shape ->
        put_elem(shape, i, nil)
      end)
    end)
  end

  defp template_shape(shape) when is_map(shape) do
    Nx.Container.traverse(shape, [], &recur_template_shape/2)
  end

  defp template_shape(shape) do
    if tuple_size(shape) == 0 do
      {Nx.template({}, {:f, 32}), []}
    else
      first_elem = elem(shape, 0)

      if is_integer(first_elem) or is_nil(first_elem) do
        {shape, template_indices} = Axon.Shape.replace_nil(shape)
        template = Nx.template(shape, {:f, 32})
        {template, List.wrap(template_indices)}
      else
        Nx.Container.traverse(shape, [], &recur_template_shape/2)
      end
    end
  end

  defp recur_template_shape(shape, indices) do
    case shape do
      shape when is_map(shape) ->
        {template, template_indices} = template_shape(shape)
        {template, indices ++ template_indices}

      shape when is_tuple(shape) ->
        {template, template_indices} = template_shape(shape)
        {template, indices ++ template_indices}
    end
  end

  # TODO: This should not be duplicated
  def deep_merge(%Nx.Tensor{} = left, %Nx.Tensor{} = right, fun) do
    fun.(left, right)
  end

  def deep_merge(left, right, fun) do
    case Nx.Container.traverse(left, leaves(right), &recur_merge(&1, &2, fun)) do
      {merged, []} ->
        merged

      {_merged, _leftover} ->
        raise ArgumentError,
              "unable to merge arguments with incompatible" <>
                " structure"
    end
  end

  defp leaves(container) do
    container
    |> Nx.Container.reduce([], fn x, acc -> [x | acc] end)
    |> Enum.reverse()
  end

  defp recur_merge(left, [right | right_leaves], fun) do
    case {left, right} do
      {%Nx.Tensor{} = left, %Nx.Tensor{} = right} ->
        {fun.(left, right), right_leaves}

      {left, right} ->
        {deep_merge(left, right, fun), right_leaves}
    end
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
  def param(name, shape, opts \\ []) when is_binary(name) and is_tuple(shape) do
    opts = Keyword.validate!(opts, initializer: :glorot_uniform)
    initializer = opts[:initializer]
    validate_initializer!(initializer)

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

  You may optionally specify an input with a default value
  using the `:default` option. Default values can be `nil`,
  tensors, or an arity-1 function. If the default value is
  `nil` and you do not handle the possibility of missing
  values in subsequent layers, you will likely experience
  cryptic errors. Default value shape must match the expected
  `input_shape` given to model.
  """
  @doc type: :special
  def input(input_shape, name, opts \\ []) when is_binary(name) do
    opts = Keyword.validate!(opts, default: :no_default_value)
    default = validate_default_input!(opts[:default])

    output_shape = Axon.Shape.input(input_shape)
    layer(:input, [], name: name, shape: output_shape, op_name: :input, default: default)
  end

  @doc """
  Adds a constant layer to the network.

  Constant layers encapsulate Nx tensors in an Axon layer for ease
  of use with other Axon layers. They can be used interchangeably
  with other Axon layers:

      inp = Axon.input({nil, 32}, "input")
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
  def constant(%Nx.Tensor{shape: output_shape} = tensor, opts) do
    opts = Keyword.validate!(opts, [:name])

    layer(:constant, [], name: opts[:name], value: tensor, shape: output_shape, op_name: :constant)
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

      iex> inp1 = Axon.input({nil, 1}, "input_0")
      iex> inp2 = Axon.input({nil, 2}, "input_1")
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

    output_shape =
      deep_new(container, fn %Axon{output_shape: shape} ->
        shape
      end)

    layer(:container, [container], name: opts[:name], shape: output_shape, op_name: :container)
  end

  # TODO: This should not be duplicated
  defp deep_new(%Nx.Tensor{} = x, fun), do: fun.(x)

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
      Axon.init(model, %{"resnset" => resnet_params})

  Notice you can use `Axon.init` in conjunction with namespaces
  to specify which portion of a model you'd like to initialize
  from a fixed starting point.

  Namespaces have fixed names, which means it's easy to run into namespace
  collisions. Re-using namespaces, re-using inner parts of a namespace,
  and attempting to share layers between namespaces are still sharp
  edges in namespace usage.
  """
  def namespace(%Axon{output_shape: shape} = axon, name) when is_binary(name) do
    layer(:namespace, [axon], name: name, shape: shape)
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
  def dense(%Axon{output_shape: parent_shape} = x, units, opts \\ [])
      when is_integer(units) and units > 0 do
    opts =
      Keyword.validate!(opts, [
        :name,
        :activation,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros,
        use_bias: true
      ])

    kernel_shape = Axon.Shape.dense_kernel(parent_shape, units)
    bias_shape = Axon.Shape.dense_bias(parent_shape, units)
    output_shape = Axon.Shape.dense(parent_shape, units)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], :dense}
      else
        {[x, kernel], &Axon.Layers.dense(&1, &2, 0, &3)}
      end

    node = layer(op, inputs, name: opts[:name], shape: output_shape, op_name: :dense)

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
        %Axon{output_shape: parent1_shape} = input1,
        %Axon{output_shape: parent2_shape} = input2,
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

    kernel_shape = Axon.Shape.bilinear_kernel(parent1_shape, parent2_shape, units)
    bias_shape = Axon.Shape.bilinear_bias(parent1_shape, parent2_shape, units)
    output_shape = Axon.Shape.bilinear(parent1_shape, parent2_shape, units)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[input1, input2, kernel, bias], :bilinear}
      else
        {[input1, input2, kernel], &Axon.Layers.bilinear(&1, &2, &3, 0, &4)}
      end

    node = layer(op, inputs, name: opts[:name], shape: output_shape, op_name: :bilinear)

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
      Defaults to `:first`.

  """
  @doc type: :convolution
  def conv(%Axon{output_shape: parent_shape} = x, units, opts \\ [])
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
        channels: :first,
        feature_group_size: 1
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]
    feature_group_size = opts[:feature_group_size]
    inner_rank = Nx.rank(parent_shape) - 2

    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)
    strides = list_or_duplicate(:strides, strides, inner_rank)
    input_dilation = list_or_duplicate(:input_dilation, input_dilation, inner_rank)
    kernel_dilation = list_or_duplicate(:kernel_dilation, kernel_dilation, inner_rank)

    kernel_shape = Axon.Shape.conv_kernel(parent_shape, units, kernel_size, channels)
    bias_shape = Axon.Shape.conv_bias(parent_shape, units, kernel_size, channels)

    output_shape =
      Axon.Shape.conv(
        parent_shape,
        kernel_shape,
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        channels,
        feature_group_size
      )

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], :conv}
      else
        {[x, kernel], &Axon.Layers.conv(&1, &2, 0, &3)}
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
        shape: output_shape,
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
      Defaults to `:first`.

  """
  @doc type: :convolution
  def conv_transpose(%Axon{output_shape: parent_shape} = x, units, opts \\ []) do
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
        channels: :first
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]
    inner_rank = Nx.rank(parent_shape) - 2

    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)
    strides = list_or_duplicate(:strides, strides, inner_rank)
    kernel_dilation = list_or_duplicate(:kernel_dilation, kernel_dilation, inner_rank)

    kernel_shape = Axon.Shape.conv_kernel(parent_shape, units, kernel_size, channels)
    bias_shape = Axon.Shape.conv_bias(parent_shape, units, kernel_size, channels)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], :conv_transpose}
      else
        {[x, kernel], &Axon.Layers.conv_transpose(&1, &2, 0, &3)}
      end

    output_shape =
      Axon.Shape.conv_transpose(
        parent_shape,
        kernel_shape,
        strides,
        padding,
        kernel_dilation,
        channels
      )

    node =
      layer(op, inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        kernel_dilation: kernel_dilation,
        channels: channels,
        shape: output_shape,
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
      Defaults to `:first`.

  """
  @doc type: :convolution
  def depthwise_conv(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
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
        channels: :first
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]
    inner_rank = Nx.rank(parent_shape) - 2

    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)
    strides = list_or_duplicate(:strides, strides, inner_rank)
    input_dilation = list_or_duplicate(:input_dilation, input_dilation, inner_rank)
    kernel_dilation = list_or_duplicate(:kernel_dilation, kernel_dilation, inner_rank)

    kernel_shape =
      Axon.Shape.depthwise_conv_kernel(parent_shape, channel_multiplier, kernel_size, channels)

    bias_shape =
      Axon.Shape.depthwise_conv_bias(parent_shape, channel_multiplier, kernel_size, channels)

    output_shape =
      Axon.Shape.depthwise_conv(
        parent_shape,
        kernel_shape,
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        channels
      )

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = param("bias", bias_shape, initializer: opts[:bias_initializer])

        {[x, kernel, bias], :depthwise_conv}
      else
        {[x, kernel], &Axon.Layers.depthwise_conv(&1, &2, 0, &3)}
      end

    node =
      layer(op, inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        channels: channels,
        shape: output_shape,
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
      Defaults to `:first`.

  """
  @doc type: :convolution
  def separable_conv2d(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
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
        channels: :first
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]
    inner_rank = Nx.rank(parent_shape) - 2

    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)
    strides = list_or_duplicate(:strides, strides, inner_rank)
    input_dilation = list_or_duplicate(:input_dilation, input_dilation, inner_rank)
    kernel_dilation = list_or_duplicate(:kernel_dilation, kernel_dilation, inner_rank)

    k1_shape =
      Axon.Shape.separable_conv2d_kernel(
        parent_shape,
        channel_multiplier,
        kernel_size,
        1,
        channels
      )

    k2_shape =
      Axon.Shape.separable_conv2d_kernel(
        parent_shape,
        channel_multiplier,
        kernel_size,
        2,
        channels
      )

    b1_shape =
      Axon.Shape.separable_conv2d_bias(parent_shape, channel_multiplier, kernel_size, channels)

    b2_shape =
      Axon.Shape.separable_conv2d_bias(parent_shape, channel_multiplier, kernel_size, channels)

    output_shape =
      Axon.Shape.depthwise_conv(
        parent_shape,
        Axon.Shape.depthwise_conv_kernel(parent_shape, channel_multiplier, kernel_size, channels),
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        channels
      )

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
        {[x, k1, k2], &Axon.Layers.separable_conv2d(&1, &2, 0, &3, 0, &4)}
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
        shape: output_shape,
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
      Defaults to `:first`.

  """
  @doc type: :convolution
  def separable_conv3d(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
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
        channels: :first
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    input_dilation = opts[:input_dilation]
    kernel_dilation = opts[:kernel_dilation]
    channels = opts[:channels]
    inner_rank = Nx.rank(parent_shape) - 2

    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)
    strides = list_or_duplicate(:strides, strides, inner_rank)
    input_dilation = list_or_duplicate(:input_dilation, input_dilation, inner_rank)
    kernel_dilation = list_or_duplicate(:kernel_dilation, kernel_dilation, inner_rank)

    k1_shape =
      Axon.Shape.separable_conv3d_kernel(
        parent_shape,
        channel_multiplier,
        kernel_size,
        1,
        channels
      )

    k2_shape =
      Axon.Shape.separable_conv3d_kernel(
        parent_shape,
        channel_multiplier,
        kernel_size,
        2,
        channels
      )

    k3_shape =
      Axon.Shape.separable_conv3d_kernel(
        parent_shape,
        channel_multiplier,
        kernel_size,
        3,
        channels
      )

    b1_shape =
      Axon.Shape.separable_conv3d_bias(parent_shape, channel_multiplier, kernel_size, channels)

    b2_shape =
      Axon.Shape.separable_conv3d_bias(parent_shape, channel_multiplier, kernel_size, channels)

    b3_shape =
      Axon.Shape.separable_conv3d_bias(parent_shape, channel_multiplier, kernel_size, channels)

    output_shape =
      Axon.Shape.depthwise_conv(
        parent_shape,
        Axon.Shape.depthwise_conv_kernel(parent_shape, channel_multiplier, kernel_size, channels),
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        channels
      )

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
        {[x, k1, k2, k3], &Axon.Layers.separable_conv3d(&1, &2, 0, &3, 0, &4, 0, &5)}
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
        shape: output_shape,
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

  def activation(%Axon{output_shape: shape} = x, activation, opts) when is_atom(activation) do
    opts = [shape: shape, op_name: activation] ++ opts
    layer(activation, [x], opts)
  end

  def activation(%Axon{output_shape: shape} = x, activation, opts)
      when is_function(activation) do
    layer(activation, [x], [shape: shape] ++ opts)
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

    """
    @doc type: :dropout
    def unquote(dropout)(%Axon{} = x, opts \\ []) do
      dropout(x, unquote(dropout), opts)
    end
  end

  defp dropout(%Axon{output_shape: parent_shape} = x, dropout, opts) do
    opts = Keyword.validate!(opts, [:name, rate: 0.5])

    layer(dropout, [x],
      name: opts[:name],
      rate: opts[:rate],
      shape: parent_shape,
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
        Defaults to `:first`.

    """
    @doc type: :pooling
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      pool(x, unquote(pool), opts)
    end
  end

  defp pool(%Axon{output_shape: parent_shape} = x, pool, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        :strides,
        kernel_size: 1,
        padding: :valid,
        channels: :first,
        dilations: 1,
        norm: 2
      ])

    kernel_size = opts[:kernel_size]
    strides = opts[:strides]
    padding = opts[:padding]
    channels = opts[:channels]
    dilations = opts[:dilations]
    inner_rank = Nx.rank(parent_shape) - 2

    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)
    strides = if strides, do: strides, else: Tuple.to_list(kernel_size)
    strides = list_or_duplicate(:strides, strides, inner_rank)
    dilations = list_or_duplicate(:dilations, dilations, inner_rank)

    output_shape =
      Axon.Shape.pool(parent_shape, kernel_size, strides, padding, dilations, channels)

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
          shape: output_shape,
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
          shape: output_shape,
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
        Defaults to `:first`.

    """
    @doc type: :pooling
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      adaptative_pool(x, unquote(pool), opts)
    end
  end

  defp adaptative_pool(%Axon{output_shape: parent_shape} = x, pool, opts) do
    opts = Keyword.validate!(opts, [:name, :output_size, channels: :first, norm: 2])

    channels = opts[:channels]

    idx =
      if channels == :first do
        1
      else
        Nx.rank(parent_shape) - 1
      end

    output_size =
      if size = opts[:output_size] do
        size
      else
        parent_shape
        |> Tuple.delete_at(0)
        |> Tuple.delete_at(idx - 1)
      end

    inner_rank = Nx.rank(parent_shape) - 2

    output_size = tuple_or_duplicate(:output_size, output_size, inner_rank)
    output_shape = Axon.Shape.adaptive_pool(parent_shape, output_size, channels)

    name = opts[:name]

    opts =
      if pool == :adaptive_lp_pool do
        norm = opts[:norm]

        [
          name: name,
          output_size: output_size,
          norm: norm,
          channels: channels,
          shape: output_shape,
          op_name: pool
        ]
      else
        [
          name: name,
          output_size: output_size,
          channels: channels,
          shape: output_shape,
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
        Defaults to `:first`.

    """
    @doc type: :pooling
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      global_pool(x, unquote(pool), opts)
    end
  end

  defp global_pool(%Axon{output_shape: parent_shape} = x, pool, opts) do
    opts = Keyword.validate!(opts, [:name, keep_axes: false, channels: :first, norm: 2])

    keep_axes = opts[:keep_axes]
    name = opts[:name]
    channels = opts[:channels]

    output_shape = Axon.Shape.global_pool(parent_shape, keep_axes, channels)

    opts =
      if pool == :global_lp_pool do
        norm = opts[:norm]

        [
          name: name,
          channels: channels,
          keep_axes: keep_axes,
          norm: norm,
          shape: output_shape,
          op_name: pool
        ]
      else
        [name: name, channels: channels, keep_axes: keep_axes, shape: output_shape, op_name: pool]
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
        mean and variance. Defaults to `1`.

      * `:epsilon` - numerical stability term.

    """
    @doc type: :normalization
    def unquote(norm)(%Axon{} = x, opts \\ []) do
      norm_with_stats(x, unquote(norm), opts)
    end
  end

  defp norm_with_stats(%Axon{output_shape: shape} = x, norm, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        gamma_initializer: :glorot_uniform,
        beta_initializer: :zeros,
        channel_index: 1,
        epsilon: 1.0e-5,
        momentum: 0.1
      ])

    channel_index = opts[:channel_index]

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)
    mean_shape = Axon.Shape.norm_param(shape, channel_index)
    var_shape = Axon.Shape.norm_param(shape, channel_index)

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
      shape: shape,
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
        mean and variance. Defaults to `1`.

      * `:epsilon` - numerical stability term.

    """
    @doc type: :normalization
    def unquote(norm)(%Axon{} = x, opts \\ []) do
      norm(x, unquote(norm), opts)
    end
  end

  defp norm(%Axon{output_shape: shape} = x, norm, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        gamma_initializer: :glorot_uniform,
        beta_initializer: :zeros,
        channel_index: 1,
        epsilon: 1.0e-5
      ])

    channel_index = opts[:channel_index]

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)

    gamma = param("gamma", gamma_shape, initializer: opts[:gamma_initializer])
    beta = param("beta", beta_shape, initializer: opts[:beta_initializer])

    layer(norm, [x, gamma, beta],
      name: opts[:name],
      epsilon: opts[:epsilon],
      channel_index: channel_index,
      shape: shape,
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
      mean and variance. Defaults to `1`.

    * `:epsilon` - numerical stability term.

  """
  @doc type: :normalization
  def group_norm(%Axon{output_shape: shape} = x, group_size, opts \\ [])
      when is_integer(group_size) and group_size >= 1 do
    opts =
      Keyword.validate!(opts, [
        :name,
        gamma_initializer: :glorot_uniform,
        beta_initializer: :zeros,
        channel_index: 1,
        epsilon: 1.0e-5
      ])

    channel_index = opts[:channel_index]

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)

    gamma = param("gamma", gamma_shape, initializer: opts[:gamma_initializer])
    beta = param("beta", beta_shape, initializer: opts[:beta_initializer])

    layer(:group_norm, [x, gamma, beta],
      name: opts[:name],
      epsilon: opts[:epsilon],
      channel_index: channel_index,
      group_size: group_size,
      shape: shape,
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
  def nx(%Axon{output_shape: input_shape} = x, fun, opts) when is_function(fun, 1) do
    opts = Keyword.validate!(opts, [:name])
    {name, opts} = Keyword.pop(opts, :name)
    fun_with_params = fn x, _opts -> fun.(x) end
    output_shape = infer_shape([input_shape], fun_with_params, opts)
    layer(fun_with_params, [x], name: name, shape: output_shape, op_name: :nx)
  end

  @doc """
  Adds a flatten layer to the network.

  This layer will flatten all but the batch dimensions
  of the input into a single layer. Typically called to flatten
  the output of a convolution for use with a dense layer.

  ## Options

    * `:name` - layer name.

    * `:ignore_batch?` - whether to ignore batch dimension in
      transpose operation. Defaults to `true`.

  """
  @doc type: :shape
  def flatten(%Axon{op: op, output_shape: shape} = x, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, ignore_batch?: op != :constant])
    ignore_batch? = opts[:ignore_batch?]
    output_shape = Axon.Shape.flatten(shape, ignore_batch?)

    layer(:flatten, [x],
      name: opts[:name],
      ignore_batch?: ignore_batch?,
      shape: output_shape,
      op_name: :flatten
    )
  end

  @doc """
  Adds a reshape layer to the network.

  This layer implements a special case of `Nx.reshape` which accounts
  for possible batch dimensions in the input tensor. If the input contains
  batch dimensions, the reshape operation is performed on all non-batch
  dimensions of the input - preserving the original batch size.

  If the input is an Axon constant, the reshape behavior matches that of
  `Nx.reshape`.

  ## Options

    * `:name` - layer name.

    * `:ignore_batch?` - whether to ignore batch dimension in transpose
      operation. Defaults to `true`.

  """
  @doc type: :shape
  def reshape(%Axon{op: op, output_shape: shape} = x, new_shape, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, ignore_batch?: op != :constant])
    ignore_batch? = opts[:ignore_batch?]
    output_shape = Axon.Shape.reshape(shape, new_shape, ignore_batch?)

    layer(:reshape, [x],
      name: opts[:name],
      ignore_batch?: ignore_batch?,
      shape: output_shape,
      to: output_shape,
      op_name: :reshape
    )
  end

  @doc """
  Adds a transpose layer to the network.

  ## Options

    * `:name` - layer name.

    * `:ignore_batch?` - whether to ignore batch dimension in transpose
      operation. Defaults to true.

  """
  @doc type: :shape
  def transpose(%Axon{op: op, output_shape: shape} = x, permutation, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, ignore_batch?: op != :constant])
    ignore_batch? = opts[:ignore_batch?]
    output_shape = Axon.Shape.transpose(shape, permutation, ignore_batch?)

    layer(:transpose, [x],
      name: opts[:name],
      axes: permutation,
      ignore_batch?: ignore_batch?,
      shape: output_shape,
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
      `:last`. Defaults to `:first`.

  """
  @doc type: :shape
  def pad(%Axon{output_shape: shape} = x, config, value \\ 0.0, opts \\ [])
      when is_list(config) and is_number(value) do
    opts = Keyword.validate!(opts, [:name, channels: :first])
    channels = opts[:channels]
    output_shape = Axon.Shape.pad(shape, config)

    layer(:pad, [x],
      name: opts[:name],
      padding_config: config,
      value: value,
      channels: channels,
      shape: output_shape,
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
      `:last`. Defaults to `:first`.

  """
  @doc type: :shape
  def resize(%Axon{output_shape: shape} = x, resize_shape, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, method: :nearest, channels: :first])
    channels = opts[:channels]
    output_shape = Axon.Shape.resize(shape, resize_shape, channels)

    layer(:resize, [x],
      name: opts[:name],
      method: opts[:method],
      channels: channels,
      shape: output_shape,
      to: resize_shape,
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
  def concatenate(%Axon{output_shape: x_shape} = x, %Axon{output_shape: y_shape} = y, opts)
      when is_list(opts) do
    opts = Keyword.validate!(opts, [:name, axis: -1])
    axis = opts[:axis]
    output_shape = Axon.Shape.concatenate([x_shape, y_shape], axis)

    layer(:concatenate, [container({x, y})],
      name: opts[:name],
      axis: axis,
      shape: output_shape,
      op_name: :concatenate
    )
  end

  @doc type: :combinator
  def concatenate([%Axon{} | _] = inputs, opts)
      when is_list(inputs) and is_list(opts) do
    opts = Keyword.validate!(opts, [:name, axis: -1])
    axis = opts[:axis]
    input_shapes = inputs |> Enum.map(fn %Axon{output_shape: shape} -> shape end)
    output_shape = Axon.Shape.concatenate(input_shapes, axis)

    layer(:concatenate, [container(List.to_tuple(inputs))],
      name: opts[:name],
      axis: axis,
      shape: output_shape,
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
    def unquote(op)(%Axon{output_shape: lhs_shape} = x, %Axon{output_shape: rhs_shape} = y, opts) do
      opts = Keyword.validate!(opts, [:name])
      output_shape = Axon.Shape.element_wise([lhs_shape, rhs_shape])

      layer(unquote(op), [container({x, y})],
        name: opts[:name],
        shape: output_shape,
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

      shapes =
        Enum.map(inputs, fn
          %Axon{output_shape: shape} -> shape
          invalid -> raise ArgumentError, "invalid input #{inspect(invalid)}"
        end)

      output_shape = Axon.Shape.element_wise(shapes)

      layer(unquote(op), [container(List.to_tuple(inputs))],
        name: opts[:name],
        shape: output_shape,
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
        %Axon{output_shape: out_shape} = true_graph,
        %Axon{output_shape: out_shape} = false_graph,
        opts \\ []
      )
      when is_function(cond_fn, 1) do
    opts = Keyword.validate!(opts, [:name])

    layer(:cond, [parent, true_graph, false_graph],
      name: opts[:name],
      cond: cond_fn,
      shape: out_shape,
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

  def split(%Axon{output_shape: shape} = parent, n, opts) when is_integer(n) do
    opts = Keyword.validate!(opts, [:name, axis: -1])
    axis = opts[:axis]

    {slice_size, split_shape} = Axon.Shape.split(shape, n, axis)

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
          fn x, _ -> Nx.slice_along_axis(x, i * slice_size, slice_size, axis: axis) end,
          [parent],
          name: name,
          shape: split_shape,
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
      Defaults to `:glorot_uniform`.

  """
  @doc type: :recurrent
  def lstm(%Axon{output_shape: shape} = x, units, opts)
      when is_integer(units) and units > 0 and is_list(opts) do
    {recurrent_initializer, opts} = Keyword.pop(opts, :recurrent_initializer, :glorot_uniform)
    c = rnn_state(x, shape, units, :lstm, opts[:name], "c", recurrent_initializer)
    h = rnn_state(x, shape, units, :lstm, opts[:name], "h", recurrent_initializer)
    lstm(x, {c, h}, units, opts)
  end

  def lstm(%Axon{} = x, {%Axon{}, %Axon{}} = hidden_state, units)
      when is_integer(units) and units > 0 do
    lstm(x, hidden_state, units, [])
  end

  @doc """
  Adds a long short-term memory (LSTM) layer to the network
  with the given initial hidden state.

  LSTMs apply `Axon.Recurrent.lstm_cell/7` over an entire input
  sequence and return:

      {{new_cell, new_hidden}, output_sequence}

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
        %Axon{output_shape: shape} = x,
        {%Axon{output_shape: h_shape}, %Axon{output_shape: h_shape}} = hidden_state,
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

    output_shape = Axon.Shape.rnn(shape, units, :lstm)
    input_kernel_shape = Axon.Shape.rnn_input_kernel(shape, units, :lstm)
    hidden_kernel_shape = Axon.Shape.rnn_hidden_kernel(shape, units, :lstm)
    bias_shape = Axon.Shape.rnn_bias(shape, units, :lstm)

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
        {[x, hidden_state, input_kernel, hidden_kernel], &Axon.Layers.lstm(&1, &2, &3, &4, 0, &5)}
      end

    output =
      layer(
        op,
        inputs,
        name: opts[:name],
        activation: activation,
        gate: gate,
        unroll: unroll,
        shape: {{h_shape, h_shape}, output_shape},
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

    new_c =
      layer(fn x, _ -> elem(elem(x, 0), 0) end, [output],
        name: new_c_name,
        shape: h_shape,
        op_name: :elem
      )

    new_h =
      layer(fn x, _ -> elem(elem(x, 0), 1) end, [output],
        name: new_h_name,
        shape: h_shape,
        op_name: :elem
      )

    output_sequence =
      layer(fn x, _ -> elem(x, 1) end, [output],
        name: output_sequence_name,
        shape: output_shape,
        op_name: :elem
      )

    {{new_c, new_h}, output_sequence}
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
      Defaults to `:glorot_uniform`.

  """
  @doc type: :recurrent
  def gru(%Axon{output_shape: shape} = x, units, opts)
      when is_integer(units) and units > 0
      when is_list(opts) do
    {recurrent_initializer, opts} = Keyword.pop(opts, :recurrent_initializer, :glorot_uniform)
    h = rnn_state(x, shape, units, :gru, opts[:name], "h", recurrent_initializer)
    gru(x, {h}, units, opts)
  end

  def gru(%Axon{} = x, {%Axon{}} = hidden_state, units) when is_integer(units) and units > 0 do
    gru(x, hidden_state, units, [])
  end

  @doc """
  Adds a gated recurrent unit (GRU) layer to the network with
  the given initial hidden state.

  GRUs apply `Axon.Recurrent.gru_cell/7` over an entire input
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
        %Axon{output_shape: shape} = x,
        {%Axon{output_shape: h_shape}} = hidden_state,
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

    output_shape = Axon.Shape.rnn(shape, units, :gru)
    input_kernel_shape = Axon.Shape.rnn_input_kernel(shape, units, :gru)
    hidden_kernel_shape = Axon.Shape.rnn_hidden_kernel(shape, units, :gru)
    bias_shape = Axon.Shape.rnn_bias(shape, units, :gru)

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
        shape: {{h_shape}, output_shape},
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

    new_h =
      layer(fn x, _ -> elem(elem(x, 0), 0) end, [output],
        name: new_h_name,
        shape: h_shape,
        op_name: :elem
      )

    output_sequence =
      layer(fn x, _ -> elem(x, 1) end, [output],
        name: output_sequence_name,
        shape: output_shape,
        op_name: :elem
      )

    {{new_h}, output_sequence}
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
      to `:glorot_uniform`.

  """
  @doc type: :recurrent
  def conv_lstm(%Axon{output_shape: shape} = x, units, opts)
      when is_integer(units) and units > 0 and is_list(opts) do
    {recurrent_initializer, opts} = Keyword.pop(opts, :recurrent_initializer, :glorot_uniform)
    c = rnn_state(x, shape, units, :conv_lstm, opts[:name], "c", recurrent_initializer)
    h = rnn_state(x, shape, units, :conv_lstm, opts[:name], "h", recurrent_initializer)
    conv_lstm(x, {c, h}, units, opts)
  end

  def conv_lstm(%Axon{} = x, {%Axon{}, %Axon{}} = hidden_state, units)
      when is_integer(units) and units > 0 do
    conv_lstm(x, hidden_state, units, [])
  end

  @doc """
  Adds a convolutional long short-term memory (LSTM) layer to the network
  with the given initial hidden state..

  ConvLSTMs apply `Axon.Recurrent.conv_lstm_cell/5` over an entire input
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
        %Axon{output_shape: shape} = x,
        {%Axon{output_shape: h_shape}, %Axon{output_shape: h_shape}} = hidden_state,
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
    inner_rank = Nx.rank(shape) - 3
    sequence_length = elem(shape, 1)

    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)
    strides = list_or_duplicate(:strides, strides, inner_rank)
    input_dilation = List.duplicate(1, inner_rank)
    kernel_dilation = List.duplicate(1, inner_rank)

    conv_shape = Tuple.delete_at(shape, 1)
    conv_hidden_state_shape = Tuple.delete_at(h_shape, 1)

    hidden_kernel_shape =
      Axon.Shape.conv_kernel(conv_hidden_state_shape, 4 * units, kernel_size, :first)

    input_kernel_shape = Axon.Shape.conv_kernel(conv_shape, 4 * units, kernel_size, :first)
    bias_shape = Axon.Shape.conv_bias(conv_shape, 4 * units, kernel_size, :first)

    output_kernel_shape =
      Axon.Shape.conv_kernel(conv_hidden_state_shape, units, kernel_size, :first)

    output_shape =
      conv_hidden_state_shape
      |> Axon.Shape.conv(
        output_kernel_shape,
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        :first,
        1
      )
      |> Tuple.insert_at(1, sequence_length)

    kernel_initializer = opts[:kernel_initializer]

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
        {[x, hidden_state, wi, wh], &Axon.Layers.conv_lstm(&1, &2, &3, &4, {0}, &5)}
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
        shape: output_shape,
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

    new_c =
      layer(fn x, _ -> elem(elem(x, 0), 0) end, [output],
        name: new_c_name,
        shape: h_shape,
        op_name: :elem
      )

    new_h =
      layer(fn x, _ -> elem(elem(x, 0), 1) end, [output],
        name: new_h_name,
        shape: h_shape,
        op_name: :elem
      )

    output_sequence =
      layer(fn x, _ -> elem(x, 1) end, [output],
        name: output_sequence_name,
        shape: output_shape,
        op_name: :elem
      )

    {{new_c, new_h}, output_sequence}
  end

  defp rnn_state(x, shape, units, rnn_type, parent_name, state_name, initializer) do
    initializer = initializer || :glorot_uniform

    name =
      case parent_name do
        nil ->
          fn _, op_counts ->
            "lstm_#{op_counts[rnn_type]}_#{state_name}_hidden_state"
          end

        parent_name when is_binary(parent_name) ->
          "#{parent_name}_#{state_name}_hidden_state"
      end

    shape = Axon.Shape.rnn_hidden_state(shape, units, rnn_type)

    fun = fn inputs, _opts ->
      shape = put_elem(shape, 0, elem(Nx.shape(inputs), 0))

      case initializer do
        fun when is_function(fun) ->
          fun.(shape)

        fun when is_atom(fun) ->
          fun = apply(Axon.Initializers, fun, [])
          fun.(shape, {:f, 32})
      end
    end

    layer(fun, [x], name: name, op_name: :recurrent_state)
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
  def embedding(%Axon{output_shape: shape} = x, vocab_size, embedding_size, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, kernel_initializer: :uniform])

    kernel_shape = Axon.Shape.embedding_kernel(shape, vocab_size, embedding_size)
    output_shape = Axon.Shape.embedding(shape, vocab_size, embedding_size)

    kernel = param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    layer(:embedding, [x, kernel], name: opts[:name], shape: output_shape, op_name: :embedding)
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
  def bias(%Axon{output_shape: shape} = x, opts \\ []) do
    opts = Keyword.validate!(opts, [:name, bias_initializer: :zeros])

    units = elem(shape, tuple_size(shape) - 1)
    bias_shape = Axon.Shape.dense_bias(shape, units)
    bias = param("bias", bias_shape, initializer: opts[:bias_initializer])

    layer(:bias, [x, bias], name: opts[:name], shape: shape, op_name: :bias)
  end

  @doc """
  Freezes parameters returned from `fun` in the given
  model. `fun` takes the model's parameter list and returns
  the list of parameters it wishes to freeze. `fun` defaults
  to the identity function, freezing all of the parameters in
  `model`.

  Freezing parameters is useful when performing transfer learning
  to leverage features learned from another problem in a new problem.
  For example, it's common to combine the convolutional base from
  larger models trained on ImageNet with fresh fully-connected classifiers.
  The combined model is then trained on fresh data, with the convolutional
  base frozen so as not to lose information. You can see this example in code
  here:

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
  def freeze(%Axon{} = model, fun \\ & &1) when is_function(fun, 1) do
    parameters =
      tree_reduce(model, MapSet.new(), fn %Axon{parameters: params}, acc ->
        Enum.reduce(params, acc, fn param, acc ->
          MapSet.put(acc, param)
        end)
      end)

    parameters_to_freeze = fun.(Enum.to_list(parameters))

    tree_map(model, fn %Axon{parameters: params} = axon ->
      frozen_params =
        Enum.map(params, fn %{name: param_name} = v ->
          if Enum.any?(parameters_to_freeze, fn %{name: name} -> name == param_name end) do
            %{v | frozen: true}
          else
            v
          end
        end)

      %{axon | parameters: frozen_params}
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

      Axon.input({nil, 1}, "input") |> Axon.attach_hook(&IO.inspect/1, on: :all)

  The default event is `:forward`, assuming you want a hook invoked
  on the layers forward pass.

  You may configure hooks to run in one of only training or inference
  mode using the `:mode` option. The default mode is `:both` to be invoked
  during both train and inference mode.

      Axon.input({nil, 1}, "input") |> Axon.attach_hook(&IO.inspect/1, on: :forward, mode: :train)

  You can also attach multiple hooks to a single layer. Hooks are invoked in
  the order in which they are declared. If order is important, you should attach
  hooks in the order you want them to be executed:

      Axon.input({nil, 1}, "input")
      # I will be executed first
      |> Axon.attach_hook(&IO.inspect/1)
      # I will be executed second
      |> Axon.attach_hook(fn _ -> IO.write("HERE") end)

  Hooks are executed at their point of attachment. You must insert hooks at each point
  you want a hook to execute during model execution.

      Axon.input({nil, 1}, "input")
      |> Axon.attach_hook(&IO.inspect/1)
      |> Axon.relu()
      |> Axon.attach_hook(&IO.inspect/1)
  """
  def attach_hook(%Axon{hooks: hooks} = axon, fun, opts \\ []) do
    opts = Keyword.validate!(opts, on: :forward, mode: :both)
    on_event = opts[:on]
    mode = opts[:mode]

    %{axon | hooks: [{on_event, mode, fun} | hooks]}
  end

  ## Traversal

  @doc """
  Traverses a model tree applying `fun` to each layer.
  """
  def tree_map(%Axon{op: :container, parent: [container]} = axon, fun) do
    x = deep_new(container, fun)
    %{fun.(axon) | parent: [x]}
  end

  def tree_map(%Axon{parent: x} = axon, fun) when is_list(x) do
    x = Enum.map(x, &tree_map(&1, fun))
    %{fun.(axon) | parent: x}
  end

  @doc """
  Traverses a model applying `fun` with an accumulator.
  """
  def tree_reduce(%Axon{op: :container, parent: [container]} = axon, acc, fun) do
    deep_reduce(container, fun.(axon, acc), fun)
  end

  def tree_reduce(%Axon{parent: x} = axon, acc, fun) when is_list(x) do
    Enum.reduce(x, fun.(axon, acc), &tree_reduce(&1, &2, fun))
  end

  # TODO: Should not be duplicated
  def deep_reduce(map, acc, fun) do
    Nx.Container.reduce(map, acc, &recur_deep_reduce(&1, &2, fun))
  end

  defp recur_deep_reduce(value, acc, fun) do
    case value do
      %Axon{} = val ->
        fun.(val, acc)

      %Nx.Tensor{} = val ->
        fun.(val, acc)

      {:leaf, val} ->
        fun.(val, acc)

      val ->
        deep_reduce(val, acc, fun)
    end
  end

  ## Utilities

  @doc """
  Returns the model's signature as a tuple of `{input_shape, output_shape}`.

  ## Examples

      iex> model = Axon.input({nil, 32}, "input") |> Axon.dense(10)
      iex> {inp, out} = Axon.get_model_signature(model)
      iex> inp
      {nil, 32}
      iex> out
      {nil, 10}

      iex> inp1 = Axon.input({nil, 32}, "input_0")
      iex> inp2 = Axon.input({nil, 32}, "input_1")
      iex> model = Axon.concatenate(inp1, inp2)
      iex> {{inp1_shape, inp2_shape}, out} = Axon.get_model_signature(model)
      iex> inp1_shape
      {nil, 32}
      iex> inp2_shape
      {nil, 32}
      iex> out
      {nil, 64}
  """
  def get_model_signature(%Axon{output_shape: output_shape} = axon) do
    # TODO: Refactor for tuples and use `tree_*` when they support
    # tuple inputs
    input_shapes =
      tree_reduce(axon, [], fn
        %Axon{op: :input, output_shape: shape}, acc -> [shape | acc]
        _, acc -> acc
      end)

    case input_shapes do
      [input_shape] ->
        {input_shape, output_shape}

      shapes ->
        {List.to_tuple(Enum.reverse(shapes)), output_shape}
    end
  end

  @doc """
  Compiles the given model to `{init_fn, predict_fn}`.

  Once compiled, a model can be passed as argument to `Nx.Defn`.

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
  @doc type: :compilation
  def compile(model, opts \\ []) when is_list(opts) do
    {Axon.Compiler.compile_init(model, opts), Axon.Compiler.compile_predict(model, opts)}
  end

  @doc """
  Compiles and runs the given models initialization function
  with the given compiler options.

  You may optionally specify initial parameters for some layers or
  namespaces by passing a partial parameter map:

      Axon.init(model, %{"dense_0" => dense_params})

  The parameter map will be merged with the initialized model
  parameters.

  ## Options

    * `:debug` - if `true`, will log graph traversal and generation
      metrics. Also forwarded to JIT if debug mode is available
      for your chosen compiler or backend. Defaults to `false`

  All other options are forwarded to the default JIT compiler
  or backend.
  """
  @doc type: :execution
  def init(model, params \\ %{}, opts \\ []) when is_list(opts) do
    Axon.Compiler.compile_init(model, opts).(params)
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
  @doc type: :execution
  def predict(%Axon{} = model, params, input, opts \\ []) when is_list(opts) do
    Axon.Compiler.compile_predict(model, opts).(params, input)
  end

  ## Inspection

  defimpl Inspect do
    import Inspect.Algebra
    import Axon.Shared
    alias Axon.Parameter

    def inspect(axon, _opts) do
      title = "Model"
      header = ["Layer", "Shape", "Policy", "Parameters", "Parameters Memory"]
      model_info = %{num_params: 0, total_param_byte_size: 0, inputs: []}
      {_, _, cache, _, model_info} = axon_to_rows(axon, %{}, %{}, model_info)

      rows =
        cache
        |> Enum.sort()
        |> Enum.unzip()
        |> elem(1)
        |> Enum.map(&elem(&1, 0))

      rows
      |> TableRex.Table.new(header, title)
      |> TableRex.Table.render!(
        header_separator_symbol: "=",
        title_separator_symbol: "=",
        vertical_style: :off
      )
      |> then(&(&1 <> "Total Parameters: #{model_info.num_params}\n"))
      |> then(&(&1 <> "Total Parameters Memory: #{model_info.total_param_byte_size} bytes\n"))
      |> then(&(&1 <> "Inputs: #{inspect(Map.new(model_info.inputs))}\n"))
      |> string()
    end

    defp axon_to_rows(%{id: id, op_name: op_name} = graph, cache, op_counts, model_info) do
      case cache do
        %{^id => {row, name}} ->
          {row, name, cache, op_counts, model_info}

        %{} ->
          {row, name, cache, op_counts, model_info} =
            do_axon_to_rows(graph, cache, op_counts, model_info)

          cache = Map.put(cache, id, {row, name})
          op_counts = Map.update(op_counts, op_name, 1, fn x -> x + 1 end)
          {row, name, cache, op_counts, model_info}
      end
    end

    defp do_axon_to_rows(
           %Axon{
             op: :container,
             parent: [parents],
             name: name_fn,
             output_shape: shape,
             policy: policy
           },
           cache,
           op_counts,
           model_info
         ) do
      {input_names, {cache, op_counts, model_info}} =
        deep_map_reduce(parents, {cache, op_counts, model_info}, fn
          graph, {cache, op_counts, model_info} ->
            {_, name, cache, op_counts, model_info} =
              axon_to_rows(graph, cache, op_counts, model_info)

            {name, {cache, op_counts, model_info}}
        end)

      op_string = "container"

      name = name_fn.(:container, op_counts)

      row = [
        "#{name} ( #{op_string} #{inspect(input_names)} )",
        "#{inspect(shape)}",
        "#{inspect(policy)}",
        0,
        "0 bytes"
      ]

      {row, name, cache, op_counts, model_info}
    end

    defp do_axon_to_rows(
           %Axon{
             op: :namespace,
             parent: parents,
             name: name_fn,
             output_shape: shape,
             policy: policy
           },
           cache,
           op_counts,
           model_info
         ) do
      init_model_info = %{num_params: 0, total_param_byte_size: 0, inputs: []}

      {_input_names, {_cache, op_counts, namespace_model_info}} =
        Enum.map_reduce(parents, {%{}, op_counts, init_model_info}, fn
          graph, {cache, op_counts, model_info} ->
            {_, name, cache, op_counts, model_info} =
              axon_to_rows(graph, cache, op_counts, model_info)

            {name, {cache, op_counts, model_info}}
        end)

      name = name_fn.(:namespace, op_counts)

      num_params = namespace_model_info.num_params
      param_byte_size = namespace_model_info.total_param_byte_size
      inputs = namespace_model_info.inputs

      model_info =
        model_info
        |> Map.update(:num_params, 0, fn x -> x + num_params end)
        |> Map.update(:total_param_byte_size, 0, fn x -> x + param_byte_size end)
        |> Map.update(:inputs, [], fn x -> x ++ inputs end)

      row = [
        "#{name} ( #{inputs |> Map.new() |> Map.keys()} )",
        "#{inspect(shape)}",
        "#{inspect(policy)}",
        "#{num_params}",
        "#{param_byte_size} bytes"
      ]

      {row, name, cache, op_counts, model_info}
    end

    defp do_axon_to_rows(
           %Axon{
             parent: parents,
             parameters: params,
             name: name_fn,
             output_shape: shape,
             policy: %{params: {_, bitsize}} = policy,
             op_name: op_name
           },
           cache,
           op_counts,
           model_info
         ) do
      {input_names, {cache, op_counts, model_info}} =
        Enum.map_reduce(parents, {cache, op_counts, model_info}, fn
          graph, {cache, op_counts, model_info} ->
            {_, name, cache, op_counts, model_info} =
              axon_to_rows(graph, cache, op_counts, model_info)

            {name, {cache, op_counts, model_info}}
        end)

      num_params =
        Enum.reduce(params, 0, fn
          %Parameter{shape: {:tuple, shapes}}, acc ->
            Enum.reduce(shapes, acc, &(Nx.size(&1) + &2))

          %Parameter{shape: shape}, acc ->
            acc + Nx.size(shape)
        end)

      param_byte_size = num_params * div(bitsize, 8)

      op_inspect = Atom.to_string(op_name)

      inputs =
        case input_names do
          [] ->
            ""

          [_ | _] = input_names ->
            "#{inspect(input_names)}"
        end

      name = name_fn.(op_name, op_counts)

      row = [
        "#{name} ( #{op_inspect}#{inputs} )",
        "#{inspect(shape)}",
        "#{inspect(policy)}",
        "#{num_params}",
        "#{param_byte_size} bytes"
      ]

      model_info =
        model_info
        |> Map.update(:num_params, 0, &(&1 + num_params))
        |> Map.update(:total_param_byte_size, 0, &(&1 + param_byte_size))
        |> Map.update(:inputs, [], fn inputs ->
          if op_name == :input, do: [{name, shape} | inputs], else: inputs
        end)

      {row, name, cache, op_counts, model_info}
    end
  end

  ## Serialization

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

      iex> model = Axon.input({nil, 2}, "input") |> Axon.dense(1, kernel_initializer: :zeros, activation: :relu)
      iex> params = Axon.init(model)
      iex> serialized = Axon.serialize(model, params)
      iex> {saved_model, saved_params} = Axon.deserialize(serialized)
      iex> Axon.predict(saved_model, saved_params, Nx.tensor([[1.0, 1.0]]))
      #Nx.Tensor<
        f32[1][1]
        [
          [0.0]
        ]
      >
  """
  def serialize(%Axon{} = model, params, opts \\ []) do
    model_meta = axon_to_map(model)
    params = Nx.serialize(params, opts)
    :erlang.term_to_binary({@file_version, model_meta, params}, opts)
  end

  defp axon_to_map(%Axon{op: :container, parent: [parents]} = model) do
    parents = deep_new(parents, &axon_to_map/1)
    axon_map = Map.from_struct(model) |> Map.put(:axon, :axon)
    %{axon_map | parent: List.wrap(parents)}
  end

  defp axon_to_map(%Axon{parent: parents} = model) do
    parents = Enum.map(parents, &axon_to_map/1)
    axon_map = Map.from_struct(model) |> Map.put(:axon, :axon)
    %{axon_map | parent: parents}
  end

  @doc """
  Deserializes serialized model and parameters into a `{model, params}`
  tuple.

  It is the opposite of `Axon.serialize/3`.

  ## Examples

      iex> model = Axon.input({nil, 2}, "input") |> Axon.dense(1, kernel_initializer: :zeros, activation: :relu)
      iex> params = Axon.init(model)
      iex> serialized = Axon.serialize(model, params)
      iex> {saved_model, saved_params} = Axon.deserialize(serialized)
      iex> Axon.predict(saved_model, saved_params, Nx.tensor([[1.0, 1.0]]))
      #Nx.Tensor<
        f32[1][1]
        [
          [0.0]
        ]
      >
  """
  def deserialize(serialized, opts \\ []) do
    {1, model_meta, serialized_params} = :erlang.binary_to_term(serialized, [:safe | opts])
    model = map_to_axon(model_meta)
    params = Nx.deserialize(serialized_params, opts)
    {model, params}
  end

  defp map_to_axon(%{op: :container, parent: [parents]} = model) do
    parents = deep_new(parents, &map_to_axon/1)
    model = Map.drop(model, [:axon])
    model = %{model | parent: List.wrap(parents)}
    struct(__MODULE__, model)
  end

  defp map_to_axon(%{axon: :axon, parent: parents} = model) do
    parents = Enum.map(parents, &map_to_axon/1)
    model = Map.drop(model, [:axon])
    model = %{model | parent: parents}
    struct(__MODULE__, model)
  end

  ## Helpers

  @valid_initializers [:zeros, :ones, :uniform, :normal, :identity] ++
                        [:lecun_uniform, :lecun_normal, :he_uniform, :he_normal] ++
                        [:glorot_uniform, :glorot_normal, :variance_scaling]

  defp validate_initializer!(initializer)
       when is_atom(initializer) and initializer in @valid_initializers do
    :ok
  end

  defp validate_initializer!(initializer) when is_function(initializer, 2) do
    :ok
  end

  defp validate_initializer!(initializer) do
    raise ArgumentError,
          "initializer must be one of #{inspect(@valid_initializers)}," <>
            " or an arity-2 function accepting initializer shape and type" <>
            " got #{inspect(initializer)}"
  end

  defp validate_default_input!(default) do
    case default do
      nil ->
        default

      :no_default_value ->
        :no_default_value

      default when is_function(default, 1) ->
        default

      %Nx.Tensor{} = default ->
        default

      invalid ->
        raise ArgumentError,
              "default input value must be nil, tensor, or arity-1 function" <>
                " of the inputs, got #{inspect(invalid)}"
    end
  end

  defp tuple_or_duplicate(key, tuple_or_integer, rank) do
    cond do
      is_tuple(tuple_or_integer) ->
        if tuple_size(tuple_or_integer) != rank do
          raise ArgumentError,
                "expected #{inspect(key)} to be a #{rank}-element tuple, " <>
                  "got: #{inspect(tuple_or_integer)}"
        end

        tuple_or_integer

      is_integer(tuple_or_integer) ->
        Tuple.duplicate(tuple_or_integer, rank)

      true ->
        raise ArgumentError,
              "expected #{inspect(key)} to be an integer or a tuple, " <>
                "got: #{inspect(tuple_or_integer)}"
    end
  end

  defp list_or_duplicate(key, list_or_integer, rank) do
    cond do
      is_list(list_or_integer) ->
        if length(list_or_integer) != rank do
          raise ArgumentError,
                "expected #{inspect(key)} to be a #{rank}-element list, " <>
                  "got: #{inspect(list_or_integer)}"
        end

        list_or_integer

      is_integer(list_or_integer) ->
        List.duplicate(list_or_integer, rank)

      true ->
        raise ArgumentError,
              "expected #{inspect(key)} to be an integer or a list, " <>
                "got: #{inspect(list_or_integer)}"
    end
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
