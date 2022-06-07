defmodule Axon do
  @moduledoc """
  A high-level interface for creating neural network models.

  Axon is built entirely on top of Nx numerical definitions,
  so every neural network can be JIT or AOT compiled using
  any Nx compiler, or even transformed into high-level neural
  network formats like TensorFlow Lite and ONNX.

  All Axon models start with an input layer, specifying the
  expected input shape of the training data:

      input = Axon.input({nil, 784})

  Notice you can specify the batch dimension as `nil`. You can
  then compose inputs with other layers:

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

      -----------------------------------------------------
                        Model
      =====================================================
       Layer                       Shape        Parameters
      =====================================================
       input_1 (input)             {nil, 784}   0
       dense_2 (dense)             {nil, 128}   100480
       relu_3 (relu)               {nil, 128}   0
       batch_norm_4 (batch_norm)   {nil, 128}   256
       dropout_5 (dropout)         {nil, 128}   0
       dense_6 (dense)             {nil, 64}    8256
       tanh_7 (tanh)               {nil, 64}    0
       dense_8 (dense)             {nil, 10}    650
       softmax_9 (softmax)         {nil, 10}    0
      -----------------------------------------------------

  Under the hood, Axon models are represented as Elixir structs. You
  can initialize and apply models using the macros `Axon.init/2` and
  `Axon.predict/4`:

      params = Axon.init(model, compiler: EXLA)

      Axon.predict(model, params, inputs, compiler: EXLA, mode: :train)

  `Axon.predict/4` by default runs in inference mode, which performs certain
  optimizations and removes layers such as dropout layers. If constructing
  a training step using `Axon.predict/4`, be sure to specify `mode: :train`.

  Both `Axon.init/2` and `Axon.predict/4` can be used from within
  Nx defn or outside.

  Combining the Axon model creation API with the optimization and training
  APIs, you can create and train neural networks with ease:

      model =
        Axon.input({nil, 784})
        |> Axon.dense(128, activation: :relu)
        |> Axon.layer_norm()
        |> Axon.dropout()
        |> Axon.dense(10, activation: :softmax)

      IO.inspect model

      model_state =
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
        |> Axon.Loop.run(train_data, epochs: 10, compiler: EXLA)
  """
  alias __MODULE__, as: Axon
  alias Axon.Parameter

  # Axon serialization version
  @file_version 1

  @type t :: %__MODULE__{}

  @doc false
  # Note: when adding new fields, consider how they fit serialize/deserialize
  @derive {
    Nx.Container,
    containers: [],
    keep: [
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
      :meta
    ]
  }
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
    :meta
  ]

  @doc """
  Custom Axon layer with given inputs.

  Inputs may be other Axon layers or trainable parameters created
  with `Axon.param`. At inference time, `op` will be applied with
  inputs in specified order and an additional `opts` parameter which
  specifies inference options. All options passed to layer are forwarded
  to inference function except:

    * `:shape` - Specify layer output shape to bypass shape inference.
    * `:name` - Layer name.

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
    {meta, opts} = Keyword.pop(opts, :meta)

    op_name = if is_atom(op), do: op, else: :custom

    {id, name} = unique_identifiers(op_name, name)

    output_shape =
      if shape do
        shape
      else
        # TODO: This does not properly handle containers
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
      meta: meta
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
      %Axon{op: :container, parent: [parent]} = layer, {layers, params, args, shapes} ->
        {[layer | layers], params, [:layer | args], [break_shapes(parent) | shapes]}

      %Axon{output_shape: shape} = layer, {layers, params, args, shapes} ->
        {[layer | layers], params, [:layer | args], [shape | shapes]}

      %Parameter{shape: shape} = param, {layers, params, args, shapes} ->
        {layers, [param | params], [:parameter | args], [shape | shapes]}

      invalid, _ ->
        raise ArgumentError, "invalid input given to layer: #{inspect(invalid)}"
    end)
  end

  defp break_shapes(parent) do
    recur_break_shapes(parent, {})
  end

  defp recur_break_shapes(%Axon{op: :container, parent: [parent]}, acc) do
    shape = break_shapes(parent)
    Tuple.append(acc, shape)
  end

  defp recur_break_shapes(%Axon{output_shape: shape}, acc) do
    Tuple.append(acc, shape)
  end

  defp recur_break_shapes(tuple, acc) when is_tuple(tuple) do
    shape =
      tuple
      |> Tuple.to_list()
      |> Enum.reduce(acc, &recur_break_shapes/2)

    {:container, shape}
  end

  defp recur_break_shapes(map, acc) when is_map(map) do
    shape = Enum.reduce(map, acc, fn {_, v}, acc -> recur_break_shapes(v, acc) end)
    {:container, shape}
  end

  defp infer_shape(input_shapes, fun, opts) do
    {input_shapes, indices} =
      input_shapes
      |> Enum.map(&Axon.Shape.replace_nil/1)
      |> Enum.unzip()

    inputs = Enum.map(input_shapes, fn shape -> Nx.template(shape, {:f, 32}) end)

    opts = Keyword.put(opts, :mode, :inference)

    wrapper_fun = fn tensors ->
      tensors = Tuple.to_list(tensors)
      apply(fun, tensors ++ [opts])
    end

    expr = Nx.Defn.jit(wrapper_fun, [List.to_tuple(inputs)], compiler: Axon.Defn)

    indices = Enum.map(indices, &MapSet.new/1)

    indices_that_are_1 =
      expr.shape
      |> Tuple.to_list()
      |> Enum.with_index()
      |> Enum.filter(fn {x, _} -> x == 1 end)
      |> Enum.map(fn {_, i} -> i end)
      |> MapSet.new()

    indices_to_make_nil =
      case indices do
        [] ->
          []

        indices ->
          indices
          |> Enum.reduce(MapSet.new(), &MapSet.union/2)
          |> MapSet.intersection(indices_that_are_1)
          |> Enum.to_list()
      end

    Enum.reduce(indices_to_make_nil, expr.shape, fn i, shape ->
      put_elem(shape, i, nil)
    end)
  end

  @doc """
  Trainable Axon parameter used to create custom layers.

  Parameters are specified in usages of `Axon.layer` and will
  be automatically initialized and used in subsequent applications
  of Axon models.

  Parameters *must* be specified in order of their usage.

  ## Options

    * `initializer` - parameter initializer. Defaults to `:glorot_uniform`.

  """
  def param(name, shape, opts \\ []) do
    initializer = opts[:initializer] || :glorot_uniform
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

  ## Options

    * `name` - Layer name.

  """
  @doc type: :special
  def input(input_shape, opts \\ []) do
    output_shape = Axon.Shape.input(input_shape)
    layer(:input, [], name: opts[:name], shape: output_shape)
  end

  @doc """
  Adds a constant layer to the network.

  Constant layers encapsulate Nx tensors in an Axon layer for ease
  of use with other Axon layers. They can be used interchangeably
  with other Axon layers:

      inp = Axon.input({nil, 32})
      my_constant = Axon.constant(Nx.iota({1, 32}))
      model = Axon.add(inp, my_constant)

  Constant layers will be cast according to the mixed precision policy.
  If it's important for your constant to retain it's type during
  the computation, you will need to set the mixed precision policy to
  ignore constant layers.

  ## Options

    * `name` - Layer name.
  """
  def constant(tensor, opts \\ [])

  @doc type: :special
  def constant(%Nx.Tensor{shape: output_shape} = tensor, opts) do
    layer(:constant, [], name: opts[:name], value: tensor, shape: output_shape)
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

    * `:name` - Layer name.

  ## Examples

      iex> inp1 = Axon.input({nil, 1})
      iex> inp2 = Axon.input({nil, 2})
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
    output_shape =
      deep_new(container, fn %Axon{output_shape: shape} ->
        shape
      end)

    layer(:container, [container], name: opts[:name], shape: output_shape)
  end

  # TODO: This should not be duplicated
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
  Adds a dense layer to the network.

  The dense layer implements:

      output = activation(dot(input, kernel) + bias)

  where `activation` is given by the `:activation` option and both
  `kernel` and `bias` are layer parameters. `units` specifies the
  number of output units.

  Compiles to `Axon.Layers.dense/3`.

  ## Options

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.

  """
  @doc type: :linear
  def dense(%Axon{output_shape: parent_shape} = x, units, opts \\ [])
      when is_integer(units) and units > 0 do
    activation = opts[:activation]
    use_bias = Keyword.get(opts, :use_bias, true)

    kernel_shape = Axon.Shape.dense_kernel(parent_shape, units)
    bias_shape = Axon.Shape.dense_bias(parent_shape, units)
    output_shape = Axon.Shape.dense(parent_shape, units)

    kernel_initializer = opts[:kernel_initializer]

    kernel = param("kernel", kernel_shape, initializer: kernel_initializer)

    inputs =
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

        bias = param("bias", bias_shape, initializer: bias_initializer)

        [x, kernel, bias]
      else
        [x, kernel]
      end

    op = if use_bias, do: :dense, else: &Axon.Layers.dense(&1, &2, 0, &3)
    node = layer(op, inputs, name: opts[:name], shape: output_shape, meta: [op: :dense])

    if activation do
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

  Compiles to `Axon.Layers.bilinear/4`.

  ## Options

    * `name` - Layer name.
    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.

  """
  @doc type: :linear
  def bilinear(
        %Axon{output_shape: parent1_shape} = input1,
        %Axon{output_shape: parent2_shape} = input2,
        units,
        opts \\ []
      )
      when is_integer(units) and units > 0 do
    activation = opts[:activation]
    use_bias = Keyword.get(opts, :use_bias, true)

    kernel_shape = Axon.Shape.bilinear_kernel(parent1_shape, parent2_shape, units)
    bias_shape = Axon.Shape.bilinear_bias(parent1_shape, parent2_shape, units)
    output_shape = Axon.Shape.bilinear(parent1_shape, parent2_shape, units)

    kernel_initializer = opts[:kernel_initializer]

    kernel = param("kernel", kernel_shape, initializer: kernel_initializer)

    inputs =
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

        bias = param("bias", bias_shape, initializer: bias_initializer)

        [input1, input2, kernel, bias]
      else
        [input1, input2, kernel]
      end

    op = if use_bias, do: :bilinear, else: &Axon.Layers.bilinear(&1, &2, &3, 0, &4)
    node = layer(op, inputs, name: opts[:name], shape: output_shape, meta: [op: :bilinear])

    if activation do
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

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.
    * `kernel_size` - Size of the kernel spatial dimensions.
    * `strides` - Stride during convolution.
    * `padding` - Padding to the spatial dimensions of the input.
    * `input_dilation` - Dilation to apply to input.
    * `kernel_dilation` - Dilation to apply to kernel.
    * `channels` - channels location. One of `:first` or `:last`.

  """
  @doc type: :convolution
  def conv(%Axon{output_shape: parent_shape} = x, units, opts \\ [])
      when is_integer(units) and units > 0 do
    activation = opts[:activation]
    use_bias = Keyword.get(opts, :use_bias, true)
    channels = opts[:channels] || :first

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1
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
        channels
      )

    kernel_initializer = opts[:kernel_initializer]

    kernel = param("kernel", kernel_shape, initializer: kernel_initializer)

    inputs =
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

        bias = param("bias", bias_shape, initializer: bias_initializer)

        [x, kernel, bias]
      else
        [x, kernel]
      end

    op = if use_bias, do: :conv, else: &Axon.Layers.conv(&1, &2, 0, &3)

    node =
      layer(op, inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        channels: channels,
        shape: output_shape,
        meta: [op: :conv]
      )

    if activation do
      activation(node, activation)
    else
      node
    end
  end

  @doc """
  Adds a transposed convolution layer to the network.

  The tranposed convolution layer is sometimes referred to as a
  fractionally strided convolution or (incorrectly) as a deconvolution.

  Compiles to `Axon.Layers.conv_transpose/4`.

  ## Options

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.
    * `kernel_size` - Size of the kernel spatial dimensions.
    * `strides` - Stride during convolution.
    * `padding` - Padding to the spatial dimensions of the input.
    * `kernel_dilation` - Dilation to apply to kernel.
    * `channels` - channels configuration. One of `:first` or `:last`.
      Defaults to `:first`.
  """
  @doc type: :convolution
  def conv_transpose(%Axon{output_shape: parent_shape} = x, units, opts \\ []) do
    activation = opts[:activation]
    use_bias = Keyword.get(opts, :use_bias, true)
    channels = opts[:channels] || :first

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    kernel_dilation = opts[:kernel_dilation] || 1
    inner_rank = Nx.rank(parent_shape) - 2

    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)
    strides = list_or_duplicate(:strides, strides, inner_rank)
    kernel_dilation = list_or_duplicate(:kernel_dilation, kernel_dilation, inner_rank)

    kernel_shape = Axon.Shape.conv_kernel(parent_shape, units, kernel_size, channels)
    bias_shape = Axon.Shape.conv_bias(parent_shape, units, kernel_size, channels)

    kernel_initializer = opts[:kernel_initializer]

    kernel = param("kernel", kernel_shape, initializer: kernel_initializer)

    inputs =
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

        bias = param("bias", bias_shape, initializer: bias_initializer)

        [x, kernel, bias]
      else
        [x, kernel]
      end

    op = if use_bias, do: :conv_transpose, else: &Axon.Layers.conv_transpose(&1, &2, 0, &3)

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
        meta: [op: :conv_transpose]
      )

    if activation do
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

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.
    * `kernel_size` - Size of the kernel spatial dimensions.
    * `strides` - Stride during convolution.
    * `padding` - Padding to the spatial dimensions of the input.
    * `input_dilation` - Dilation to apply to input.
    * `kernel_dilation` - Dilation to apply to kernel.
    * `channels` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.
  """
  @doc type: :convolution
  def depthwise_conv(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    activation = opts[:activation]
    use_bias = Keyword.get(opts, :use_bias, true)
    channels = opts[:channels] || :first

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1
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

    kernel_initializer = opts[:kernel_initializer]

    kernel = param("kernel", kernel_shape, initializer: kernel_initializer)

    inputs =
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

        bias = param("bias", bias_shape, initializer: bias_initializer)

        [x, kernel, bias]
      else
        [x, kernel]
      end

    op = if use_bias, do: :depthwise_conv, else: &Axon.Layers.depthwise_conv(&1, &2, 0, &3)

    node =
      layer(op, inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        channels: channels,
        shape: output_shape,
        meta: [op: :depthwise_conv]
      )

    if activation do
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

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.
    * `kernel_size` - Size of the kernel spatial dimensions.
    * `strides` - Stride during convolution.
    * `padding` - Padding to the spatial dimensions of the input.
    * `input_dilation` - Dilation to apply to input.
    * `kernel_dilation` - Dilation to apply to kernel.
    * `channels` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  """
  @doc type: :convolution
  def separable_conv2d(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    activation = opts[:activation]
    use_bias = Keyword.get(opts, :use_bias, true)
    channels = opts[:channels] || :first

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1
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

    inputs =
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

        b1 = param("bias_1", b1_shape, initializer: bias_initializer)
        b2 = param("bias_2", b2_shape, initializer: bias_initializer)

        [x, k1, b1, k2, b2]
      else
        [x, k1, k2]
      end

    node =
      layer(
        :separable_conv2d,
        inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        channels: channels,
        shape: output_shape,
        meta: [op: :separable_conv2d]
      )

    if activation do
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

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.
    * `kernel_size` - Size of the kernel spatial dimensions.
    * `strides` - Stride during convolution.
    * `padding` - Padding to the spatial dimensions of the input.
    * `input_dilation` - Dilation to apply to input.
    * `kernel_dilation` - Dilation to apply to kernel.
    * `channels` - channels configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  """
  @doc type: :convolution
  def separable_conv3d(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    activation = opts[:activation]
    channels = opts[:channels] || :first
    use_bias = Keyword.get(opts, :use_bias, true)

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1
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

    inputs =
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

        b1 = param("bias_1", b1_shape, initializer: bias_initializer)
        b2 = param("bias_2", b2_shape, initializer: bias_initializer)
        b3 = param("bias_3", b3_shape, initializer: bias_initializer)

        [x, k1, b1, k2, b2, k3, b3]
      else
        [x, k1, k2, k3]
      end

    node =
      layer(
        :separable_conv3d,
        inputs,
        name: opts[:name],
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation,
        channels: channels,
        shape: output_shape,
        meta: [op: :separable_conv3d]
      )

    if activation do
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

    - `name` - Layer name.

  """
  @doc type: :activation
  def activation(x, activation, opts \\ [])

  def activation(%Axon{output_shape: shape} = x, activation, opts) when is_atom(activation) do
    opts = [shape: shape] ++ opts
    layer(activation, [x], opts)
  end

  def activation(%Axon{output_shape: shape} = x, activation, opts)
      when is_function(activation, 1) do
    layer(activation, [x], [shape: shape] ++ opts)
  end

  ## Activation

  for {activation, name, a_or_an} <- @activation_layers do
    @doc """
    Adds #{a_or_an} #{name} activation layer to the network.

    See `Axon.Activations.#{Atom.to_string(activation)}/1` for more details.

    ## Options

      - `name` - Layer name.

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

      * `:name` - Layer name.
      * `:rate` - Dropout rate.

    """
    @doc type: :dropout
    def unquote(dropout)(%Axon{} = x, opts \\ []) do
      dropout(x, unquote(dropout), opts)
    end
  end

  defp dropout(%Axon{output_shape: parent_shape} = x, dropout, opts) do
    rate = opts[:rate] || 0.5
    layer(dropout, [x], name: opts[:name], rate: rate, shape: parent_shape)
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

      * `name` - Layer name.
      * `kernel_size` - Pooling kernel size. Defaults to `1`.
      * `padding` - Padding to apply to input of pooling operation.
      * `strides` - Pooling strides. Defaults to size of kernel.
      * `dilations` - Window dilations. Defaults to `1`.
      * `channels` - channel configuration. One of `:first` or `:last`.
        Defaults to `:first`.

    """
    @doc type: :pooling
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      pool(x, unquote(pool), opts)
    end
  end

  defp pool(%Axon{output_shape: parent_shape} = x, pool, opts) do
    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides]
    padding = opts[:padding] || :valid
    channels = opts[:channels] || :first
    dilations = opts[:dilations] || 1
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
        norm = opts[:norm] || 2

        [
          name: name,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding,
          channels: channels,
          window_dilations: dilations,
          norm: norm,
          shape: output_shape
        ]
      else
        [
          name: name,
          kernel_size: kernel_size,
          strides: strides,
          padding: padding,
          channels: channels,
          window_dilations: dilations,
          shape: output_shape
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

      * `:name` - Layer name.
      * `:output_size` - Layer output size.
      * `:channels` - channel configuration. One of `:first` or `:last`.
        Defaults to `:first`.

    """
    @doc type: :pooling
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      adaptative_pool(x, unquote(pool), opts)
    end
  end

  defp adaptative_pool(%Axon{output_shape: parent_shape} = x, pool, opts) do
    channels = opts[:channels] || :first

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
        norm = opts[:norm] || 2

        [
          name: name,
          output_size: output_size,
          norm: norm,
          channels: channels,
          shape: output_shape
        ]
      else
        [name: name, output_size: output_size, channels: channels, shape: output_shape]
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

      * `:name` - Layer name.
      * `:keep_axes` - Option to keep reduced axes. If `true`, keeps reduced axes
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
    keep_axes = opts[:keep_axes]
    name = opts[:name]
    channels = opts[:channels] || :first

    output_shape = Axon.Shape.global_pool(parent_shape, keep_axes, channels)

    opts =
      if pool == :global_lp_pool do
        norm = opts[:norm] || 2
        [name: name, channels: channels, keep_axes: keep_axes, norm: norm, shape: output_shape]
      else
        [name: name, channels: channels, keep_axes: keep_axes, shape: output_shape]
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

    See `Axon.Layers.#{Atom.to_string(norm)}/4` for more details.

    ## Options

      * `:name` - Layer name.
      * `:gamma_initializer` - Gamma parameter initializer.
      * `:beta_initializer` - Beta parameter initializer.
      * `:channel_index` - Input feature index used for calculating
        mean and variance.
      * `:epsilon` - Numerical stability term.

    """
    @doc type: :normalization
    def unquote(norm)(%Axon{} = x, opts \\ []) do
      norm_with_stats(x, unquote(norm), opts)
    end
  end

  defp norm_with_stats(%Axon{output_shape: shape} = x, norm, opts) do
    channel_index = opts[:channel_index] || 1
    epsilon = opts[:epsilon] || 1.0e-5
    momentum = opts[:momentum] || 0.1

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)
    mean_shape = Axon.Shape.norm_param(shape, channel_index)
    var_shape = Axon.Shape.norm_param(shape, channel_index)

    gamma_initializer = opts[:gamma_initializer]

    gamma = param("gamma", gamma_shape, initializer: gamma_initializer)

    beta_initializer = opts[:beta_initializer] || :zeros

    beta = param("beta", beta_shape, initializer: beta_initializer)

    mean = param("mean", mean_shape, initializer: :zeros)
    var = param("var", var_shape, initializer: :ones)

    layer(
      norm,
      [x, gamma, beta, mean, var],
      name: opts[:name],
      epsilon: epsilon,
      channel_index: channel_index,
      momentum: momentum,
      shape: shape
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

      * `:name` - Layer name.
      * `:gamma_initializer` - Gamma parameter initializer.
      * `:beta_initializer` - Beta parameter initializer.
      * `:channel_index` - Input feature index used for calculating
        mean and variance.
      * `:epsilon` - Numerical stability term.

    """
    @doc type: :normalization
    def unquote(norm)(%Axon{} = x, opts \\ []) do
      norm(x, unquote(norm), opts)
    end
  end

  defp norm(%Axon{output_shape: shape} = x, norm, opts) do
    channel_index = opts[:channel_index] || 1
    epsilon = opts[:epsilon] || 1.0e-5

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)

    gamma_initializer = opts[:gamma_initializer]

    gamma = param("gamma", gamma_shape, initializer: gamma_initializer)

    beta_initializer = opts[:beta_initializer] || :zeros
    beta = param("beta", beta_shape, initializer: beta_initializer)

    layer(norm, [x, gamma, beta],
      name: opts[:name],
      epsilon: epsilon,
      channel_index: channel_index,
      shape: shape
    )
  end

  @doc """
  Adds a group normalization layer to the network.

  See `Axon.Layers.group_norm/4` for more details.

  ## Options

    * `:name` - Layer name.
    * `:gamma_initializer` - Gamma parameter initializer.
    * `:beta_initializer` - Beta parameter initializer.
    * `:channel_index` - Input feature index used for calculating
      mean and variance.
    * `:epsilon` - Numerical stability term.

  """
  @doc type: :normalization
  def group_norm(%Axon{output_shape: shape} = x, group_size, opts \\ [])
      when is_integer(group_size) and group_size >= 1 do
    channel_index = opts[:channel_index] || 1
    epsilon = opts[:epsilon] || 1.0e-5

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)

    gamma_initializer = opts[:gamma_initializer]

    gamma = param("gamma", gamma_shape, initializer: gamma_initializer)

    beta_initializer = opts[:beta_initializer] || :zeros

    beta = param("beta", beta_shape, initializer: beta_initializer)

    layer(:group_norm, [x, gamma, beta],
      name: opts[:name],
      epsilon: epsilon,
      channel_index: channel_index,
      group_size: group_size,
      shape: shape
    )
  end

  @doc """
  Applies the given `Nx` expression to the input.

  ## Options

    * `name` - Layer name.

  """
  def nx(input, fun, opts \\ [])

  @doc type: :special
  def nx(%Axon{output_shape: input_shape} = x, fun, opts) when is_function(fun, 1) do
    {name, opts} = Keyword.pop(opts, :name)
    fun_with_params = fn x, _opts -> fun.(x) end
    output_shape = infer_shape([input_shape], fun_with_params, opts)
    layer(fun_with_params, [x], name: name, shape: output_shape)
  end

  @doc """
  Adds a flatten layer to the network.

  This layer will flatten all but the batch dimensions
  of the input into a single layer. Typically called to flatten
  the output of a convolution for use with a dense layer.

  ## Options

    * `:name` - Layer name.

  """
  @doc type: :shape
  def flatten(%Axon{op: op, output_shape: shape} = x, opts \\ []) do
    ignore_batch? = Keyword.get(opts, :ignore_batch?, op != :constant)
    output_shape = Axon.Shape.flatten(shape, ignore_batch?)
    layer(:flatten, [x], name: opts[:name], ignore_batch?: ignore_batch?, shape: output_shape)
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

    * `:name` - Layer name.
  """
  @doc type: :shape
  def reshape(%Axon{op: op, output_shape: shape} = x, new_shape, opts \\ []) do
    ignore_batch? = Keyword.get(opts, :ignore_batch?, op != :constant)
    output_shape = Axon.Shape.reshape(shape, new_shape, ignore_batch?)

    layer(:reshape, [x],
      name: opts[:name],
      ignore_batch?: ignore_batch?,
      shape: output_shape,
      to: output_shape
    )
  end

  @doc """
  Adds a transpose layer to the network.

  ## Options

    * `:name` - Layer name.
    * `:ignore_batch?` - Whether to ignore batch dimension in
      transpose operation. Defaults to true.
  """
  @doc type: :shape
  def transpose(%Axon{op: op, output_shape: shape} = x, permutation, opts \\ []) do
    ignore_batch? = Keyword.get(opts, :ignore_batch?, op != :constant)
    output_shape = Axon.Shape.transpose(shape, permutation, ignore_batch?)

    layer(:transpose, [x],
      name: opts[:name],
      axes: permutation,
      ignore_batch?: ignore_batch?,
      shape: output_shape
    )
  end

  @doc """
  Adds a pad layer to the network.

  This layer will pad the spatial dimensions of the input.
  Padding configuration is a list of tuples for each spatial
  dimension.

  ## Options

    * `:name` - Layer name.
    * `:channels` - Channel configuration. One of `:first` or
      `:last`. Defaults to `:first`.
  """
  @doc type: :shape
  def pad(%Axon{output_shape: shape} = x, config, value \\ 0.0, opts \\ [])
      when is_list(config) and is_number(value) do
    channels = opts[:channels] || :first
    output_shape = Axon.Shape.pad(shape, config)

    layer(:pad, [x],
      name: opts[:name],
      padding_config: config,
      value: value,
      channels: channels,
      shape: output_shape
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

    * `:name` - Layer name.
    * `:method` - Resize method. Defaults to `:nearest`.
    * `:channels` - Channels configuration. Defaults to `:first`.
  """
  @doc type: :shape
  def resize(%Axon{output_shape: shape} = x, resize_shape, opts \\ []) do
    method = opts[:method] || :nearest
    channels = opts[:channels] || :first
    output_shape = Axon.Shape.resize(shape, resize_shape, channels)

    layer(:resize, [x],
      name: opts[:name],
      method: method,
      channels: channels,
      shape: output_shape,
      to: resize_shape
    )
  end

  @doc """
  Adds a concatenate layer to the network.

  This layer will concatenate inputs along the last
  dimension unless specified otherwise.

  ## Options

    * `:name` - Layer name.
    * `:axis` - Concatenate axis.

  """
  @doc type: :composition
  def concatenate(%Axon{output_shape: x_shape} = x, %Axon{output_shape: y_shape} = y, opts)
      when is_list(opts) do
    axis = opts[:axis] || Nx.rank(x_shape) - 1
    output_shape = Axon.Shape.concatenate([x_shape, y_shape], axis)

    layer(:concatenate, [container({x, y})], name: opts[:name], axis: axis, shape: output_shape)
  end

  @doc type: :composition
  def concatenate([%Axon{output_shape: shape} | _] = inputs, opts)
      when is_list(inputs) and is_list(opts) do
    axis = opts[:axis] || Nx.rank(shape) - 1
    input_shapes = inputs |> Enum.map(fn %Axon{output_shape: shape} -> shape end)
    output_shape = Axon.Shape.concatenate(input_shapes, axis)

    layer(:concatenate, [container(List.to_tuple(inputs))],
      name: opts[:name],
      axis: axis,
      shape: output_shape
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

      * `:name` - Layer name.

    """
    @doc type: :composition
    def unquote(op)(%Axon{output_shape: lhs_shape} = x, %Axon{output_shape: rhs_shape} = y, opts) do
      output_shape = Axon.Shape.element_wise([lhs_shape, rhs_shape])
      layer(unquote(op), [container({x, y})], name: opts[:name], shape: output_shape)
    end

    @doc """
    Adds a #{op} layer to the network.

    This layer performs an element-wise #{Atom.to_string(op)} operation
    on all input layers. All input layers must be capable of being
    broadcast together.

    ## Options

      * `:name` - Layer name.

    """
    @doc type: :composition
    def unquote(op)(inputs, opts) when is_list(inputs) and is_list(opts) do
      shapes =
        Enum.map(inputs, fn
          %Axon{output_shape: shape} -> shape
          invalid -> raise ArgumentError, "invalid input #{inspect(invalid)}"
        end)

      output_shape = Axon.Shape.element_wise(shapes)

      layer(unquote(op), [container(List.to_tuple(inputs))],
        name: opts[:name],
        shape: output_shape
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
  def cond(
        %Axon{} = parent,
        cond_fn,
        %Axon{output_shape: out_shape} = true_graph,
        %Axon{output_shape: out_shape} = false_graph,
        opts \\ []
      )
      when is_function(cond_fn, 1) do
    layer(:cond, [parent, true_graph, false_graph],
      name: opts[:name],
      cond: cond_fn,
      shape: out_shape
    )
  end

  @doc """
  Splits input graph into a container of `n` input graphs
  along the given axis.
  """
  def split(parent, splits, opts \\ [])

  def split(%Axon{} = parent, splits, opts) when is_list(splits) do
    axis = opts[:axis] || -1

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
              name: name
            )

          {num_split + split, [layer | split_layers]}
      end

    split_layers |> Enum.reverse() |> List.to_tuple()
  end

  def split(%Axon{output_shape: shape} = parent, n, opts) when is_integer(n) do
    axis = opts[:axis] || -1
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
          shape: split_shape
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

  LSTMs apply `Axon.Recurrent.lstm_cell/7` over an entire input
  sequence and return:

      {{new_cell, new_hidden}, output_sequence}

  You can use the output state as the hidden state of another
  LSTM layer.

  ## Options

    * `:activation` - recurrent activation. Defaults to `:tanh`.
    * `:gate` - recurrent gate function. Defaults to `:sigmoid`.
    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.
    * `:recurrent_initializer` - Initializer for hidden state.
  """
  @doc type: :recurrent
  def lstm(%Axon{output_shape: shape} = x, units, opts)
      when is_integer(units) and units > 0 and is_list(opts) do
    c = rnn_state(x, shape, units, :lstm, opts[:name], "c", opts[:recurrent_initializer])
    h = rnn_state(x, shape, units, :lstm, opts[:name], "h", opts[:recurrent_initializer])
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

    * `:activation` - recurrent activation. Defaults to `:tanh`.
    * `:gate` - recurrent gate function. Defaults to `:sigmoid`.
    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.

  """
  @doc type: :recurrent
  def lstm(
        %Axon{output_shape: shape} = x,
        {%Axon{output_shape: h_shape}, %Axon{output_shape: h_shape}} = hidden_state,
        units,
        opts \\ []
      )
      when is_integer(units) and units > 0 and is_list(opts) do
    activation = opts[:activation] || :tanh
    gate = opts[:gate] || :sigmoid
    unroll = opts[:unroll] || :dynamic

    use_bias = Keyword.get(opts, :use_bias, true)

    output_shape = Axon.Shape.rnn(shape, units, :lstm)
    input_kernel_shape = Axon.Shape.rnn_input_kernel(shape, units, :lstm)
    hidden_kernel_shape = Axon.Shape.rnn_hidden_kernel(shape, units, :lstm)
    bias_shape = Axon.Shape.rnn_bias(shape, units, :lstm)

    kernel_initializer = opts[:kernel_initializer] || :glorot_uniform

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

    inputs =
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

        bias =
          param("bias", {:tuple, List.duplicate(bias_shape, 4)}, initializer: bias_initializer)

        [
          x,
          hidden_state,
          input_kernel,
          hidden_kernel,
          bias
        ]
      else
        [
          x,
          hidden_state,
          input_kernel,
          hidden_kernel
        ]
      end

    output =
      layer(
        :lstm,
        inputs,
        name: opts[:name],
        activation: activation,
        gate: gate,
        unroll: unroll,
        shape: {{h_shape, h_shape}, output_shape},
        meta: [op: :lstm]
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

    new_c = layer(fn x, _ -> elem(elem(x, 0), 0) end, [output], name: new_c_name, shape: h_shape)

    new_h = layer(fn x, _ -> elem(elem(x, 0), 1) end, [output], name: new_h_name, shape: h_shape)

    output_sequence =
      layer(fn x, _ -> elem(x, 1) end, [output], name: output_sequence_name, shape: output_shape)

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

  GRUs apply `Axon.Recurrent.gru_cell/7` over an entire input
  sequence and return:

      {{new_hidden}, output_sequence}

  You can use the output state as the hidden state of another
  GRU layer.

  ## Options

    * `:activation` - recurrent activation. Defaults to `:tanh`.
    * `:gate` - recurrent gate function. Defaults to `:sigmoid`.
    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.
    * `:recurrent_initializer` - Initializer for hidden state.

  """
  @doc type: :recurrent
  def gru(%Axon{output_shape: shape} = x, units, opts)
      when is_integer(units) and units > 0
      when is_list(opts) do
    h = rnn_state(x, shape, units, :gru, opts[:name], "h", opts[:recurrent_initializer])
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

    * `:activation` - recurrent activation. Defaults to `:tanh`.
    * `:gate` - recurrent gate function. Defaults to `:sigmoid`.
    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.

  """
  @doc type: :recurrent
  def gru(
        %Axon{output_shape: shape} = x,
        {%Axon{output_shape: h_shape}} = hidden_state,
        units,
        opts
      )
      when is_integer(units) and units > 0 and is_list(opts) do
    use_bias = Keyword.get(opts, :use_bias, true)
    activation = opts[:activation] || :tanh
    gate = opts[:gate] || :sigmoid
    unroll = opts[:unroll] || :dynamic

    output_shape = Axon.Shape.rnn(shape, units, :gru)
    input_kernel_shape = Axon.Shape.rnn_input_kernel(shape, units, :gru)
    hidden_kernel_shape = Axon.Shape.rnn_hidden_kernel(shape, units, :gru)
    bias_shape = Axon.Shape.rnn_bias(shape, units, :gru)

    kernel_initializer = opts[:kernel_initializer] || :glorot_uniform

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
      if use_bias do
        bias_initializer = opts[:bias_initializer] || :zeros

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
        meta: [op: :gru]
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

    new_h = layer(fn x, _ -> elem(elem(x, 0), 0) end, [output], name: new_h_name, shape: h_shape)

    output_sequence =
      layer(fn x, _ -> elem(x, 1) end, [output], name: output_sequence_name, shape: output_shape)

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

  ConvLSTMs apply `Axon.Recurrent.conv_lstm_cell/5` over an entire input
  sequence and return:

      {{new_cell, new_hidden}, output_sequence}

  You can use the output state as the hidden state of another
  LSTM layer with the `:hidden_state` option.

  ## Options

    * `:padding` - convolutional padding. Defaults to `:same`.
    * `:kernel_size` - convolutional kernel size. Defaults to `1`.
    * `:strides` - convolutional strides. Defaults to `1`.
    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.
    * `:recurrent_initializer` - Initializer for hidden state.

  """
  @doc type: :recurrent
  def conv_lstm(%Axon{output_shape: shape} = x, units, opts)
      when is_integer(units) and units > 0 and is_list(opts) do
    c = rnn_state(x, shape, units, :conv_lstm, opts[:name], "c", opts[:recurrent_initializer])
    h = rnn_state(x, shape, units, :conv_lstm, opts[:name], "h", opts[:recurrent_initializer])
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

    * `:padding` - convolutional padding. Defaults to `:same`.
    * `:kernel_size` - convolutional kernel size. Defaults to `1`.
    * `:strides` - convolutional strides. Defaults to `1`.
    * `:unroll` - `:dynamic` (loop preserving) or `:static` (compiled)
      unrolling of RNN.

  """
  @doc type: :recurrent
  def conv_lstm(
        %Axon{output_shape: shape} = x,
        {%Axon{output_shape: h_shape}, %Axon{output_shape: h_shape}} = hidden_state,
        units,
        opts
      )
      when is_integer(units) and units > 0 and is_list(opts) do
    padding = opts[:padding] || :same
    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    unroll = opts[:unroll] || :dynamic
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
        :first
      )
      |> Tuple.insert_at(1, sequence_length)

    kernel_initializer = opts[:kernel_initializer] || :glorot_uniform
    bias_initializer = opts[:bias_initializer] || :zeros

    wi = param("input_kernel", {:tuple, [input_kernel_shape]}, initializer: kernel_initializer)
    wh = param("hidden_kernel", {:tuple, [hidden_kernel_shape]}, initializer: kernel_initializer)
    b = param("bias", {:tuple, [bias_shape]}, initializer: bias_initializer)

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

    inputs = [x, hidden_state, wi, wh, b]

    output =
      layer(
        :conv_lstm,
        inputs,
        name: opts[:name],
        conv_opts: [
          strides: strides,
          padding: padding
        ],
        unroll: unroll,
        shape: output_shape
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

    new_c = layer(fn x, _ -> elem(elem(x, 0), 0) end, [output], name: new_c_name, shape: h_shape)

    new_h = layer(fn x, _ -> elem(elem(x, 0), 1) end, [output], name: new_h_name, shape: h_shape)

    output_sequence =
      layer(fn x, _ -> elem(x, 1) end, [output], name: output_sequence_name, shape: output_shape)

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
          fun.(shape: shape)

        fun when is_atom(fun) ->
          apply(Axon.Initializers, fun, [[shape: shape]])
      end
    end

    layer(fun, [x], name: name, meta: [op: :recurrent_state])
  end

  @doc """
  Adds an embedding layer to the network.

  An embedding layer initializes a kernel of shape `{vocab_size, embedding_size}`
  which acts as a lookup table for sequences of discrete tokens (e.g. sentences).
  Embeddings are typically used to obtain a dense representation of a sparse input
  space.
  """
  @doc type: :linear
  def embedding(%Axon{output_shape: shape} = x, vocab_size, embedding_size, opts \\ []) do
    kernel_shape = Axon.Shape.embedding_kernel(shape, vocab_size, embedding_size)
    output_shape = Axon.Shape.embedding(shape, vocab_size, embedding_size)

    kernel_initializer = opts[:kernel_initializer] || :uniform
    kernel = param("kernel", kernel_shape, initializer: kernel_initializer)

    layer(:embedding, [x, kernel], name: opts[:name], shape: output_shape)
  end

  @doc """
  Adds a bias layer to the network.

  A bias layer simply adds a trainable bias to an input.
  """
  @doc type: :linear
  def bias(%Axon{output_shape: shape} = x, opts \\ []) do
    units = elem(shape, tuple_size(shape) - 1)
    bias_shape = Axon.Shape.dense_bias(shape, units)
    bias_initializer = opts[:bias_initializer] || :zeros
    bias = param("bias", bias_shape, initializer: bias_initializer)

    layer(:bias, [x, bias], name: opts[:name], shape: shape)
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

      Axon.input({nil, 1}) |> Axon.attach_hook(&IO.inspect/1, on: :all)

  The default event is `:forward`, assuming you want a hook invoked
  on the layers forward pass.

  You may configure hooks to run in one of only training or inference
  mode using the `:mode` option. The default mode is `:both` to be invoked
  during both train and inference mode.

      Axon.input({nil, 1}) |> Axon.attach_hook(&IO.inspect/1, on: :forward, mode: :train)

  You can also attach multiple hooks to a single layer. Hooks are invoked in
  the order in which they are declared. If order is important, you should attach
  hooks in the order you want them to be executed:

      Axon.input({nil, 1})
      # I will be executed first
      |> Axon.attach_hook(&IO.inspect/1)
      # I will be executed second
      |> Axon.attach_hook(fn _ -> IO.write("HERE") end)

  Hooks are executed at their point of attachment. You must insert hooks at each point
  you want a hook to execute during model execution.

      Axon.input({nil, 1})
      |> Axon.attach_hook(&IO.inspect/1)
      |> Axon.relu()
      |> Axon.attach_hook(&IO.inspect/1)
  """
  def attach_hook(%Axon{hooks: hooks} = axon, fun, opts \\ []) do
    on_event = opts[:on] || :forward
    mode = opts[:mode] || :both

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

      iex> model = Axon.input({nil, 32}) |> Axon.dense(10)
      iex> {inp, out} = Axon.get_model_signature(model)
      iex> inp
      {nil, 32}
      iex> out
      {nil, 10}

      iex> inp1 = Axon.input({nil, 32})
      iex> inp2 = Axon.input({nil, 32})
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
  """
  @doc type: :compilation
  def compile(model, opts \\ []) do
    Axon.Compiler.__compile__(model, opts)
  end

  @doc """
  Compiles and runs the given models initialization function
  with the given compiler options.
  """
  @doc type: :execution
  defmacro init(model, opts \\ []) do
    define_init(model, [], opts)
  end

  @doc """
  Compiles and runs the given Axon model with `params` on
  `input` with the given compiler options.
  """
  @doc type: :execution
  defmacro predict(model, params, input, opts \\ []) do
    define_predict(model, [params, input], opts)
  end

  ## Implementation

  defp define_init(model, args, opts \\ []) do
    quote do
      Nx.Defn.Kernel.transform(unquote(args), fn args ->
        model = unquote(model)
        opts = unquote(opts)

        Axon.Compiler.__jit_init__(model, args, opts)
      end)
    end
  end

  defp define_predict(model, args, opts \\ []) do
    quote do
      Nx.Defn.Kernel.transform(unquote(args), fn args ->
        model = unquote(model)
        opts = unquote(opts)

        Axon.Compiler.__jit_predict__(model, args, opts)
      end)
    end
  end

  ## Inspection

  defimpl Inspect do
    import Inspect.Algebra
    import Axon.Shared
    alias Axon.Parameter

    def inspect(axon, _opts) do
      title = "Model"
      header = ["Layer", "Shape", "Policy", "Parameters", "Parameters Memory"]
      {_, _, cache, _, model_info} = axon_to_rows(axon, %{}, %{}, %{})

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
      |> string()
    end

    defp axon_to_rows(%{id: id, op: op} = graph, cache, op_counts, model_info) do
      case cache do
        %{^id => {row, name}} ->
          {row, name, cache, op_counts, model_info}

        %{} ->
          {row, name, cache, op_counts, model_info} =
            do_axon_to_rows(graph, cache, op_counts, model_info)

          cache = Map.put(cache, id, {row, name})
          op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)
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
             op: op,
             parent: parents,
             parameters: params,
             name: name_fn,
             output_shape: shape,
             policy: %{params: {_, bitsize}} = policy,
             meta: meta
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

      op_inspect =
        case {op, meta[:op]} do
          {op, _} when is_atom(op) -> Atom.to_string(op)
          {_, nil} -> "custom"
          {_, layer_op} when is_atom(layer_op) -> Atom.to_string(layer_op)
          {_, layer_op} when is_binary(layer_op) -> layer_op
        end

      inputs =
        case input_names do
          [] ->
            ""

          [_ | _] = input_names ->
            "#{inspect(input_names)}"
        end

      name = name_fn.(op, op_counts)

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

      iex> model = Axon.input({nil, 2}) |> Axon.dense(1, kernel_initializer: :zeros, activation: :relu)
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

      iex> model = Axon.input({nil, 2}) |> Axon.dense(1, kernel_initializer: :zeros, activation: :relu)
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

  defp validate_initializer!(initializer) when is_function(initializer, 1) do
    :ok
  end

  defp validate_initializer!(initializer) do
    raise ArgumentError,
          "initializer must be one of #{inspect(@valid_initializers)}," <>
            " or an arity-1 function accepting initializer options" <>
            " got #{inspect(initializer)}"
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
