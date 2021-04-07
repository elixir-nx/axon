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

      IO.inspect model

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

      Axon.predict(model, params, inputs, compiler: EXLA)

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

      final_params =
        model
        |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
        |> Axon.Training.train(train_images, train_labels, epochs: 10, compiler: EXLA)
  """
  alias __MODULE__, as: Axon

  @type t :: %__MODULE__{}

  @doc false
  defstruct [:id, :name, :output_shape, :parent, :op, :params, :opts]

  @doc """
  Adds an input layer to the network.

  Input layers specify a model's inputs. Input layers are
  always the root layers of the neural network.

  ## Options

    * `name` - Layer name.

  """
  @doc type: :layer
  def input(input_shape, opts \\ []) do
    {id, name} = unique_identifiers(:input, opts[:name])
    %Axon{id: id, name: name, output_shape: input_shape, parent: nil, op: :input, params: []}
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
  @doc type: :layer
  def dense(%Axon{output_shape: parent_shape} = x, units, opts \\ [])
      when is_integer(units) and units > 0 do
    {id, name} = unique_identifiers(:dense, opts[:name])

    weight_init = opts[:kernel_initializer] || :glorot_uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]

    kernel_shape = Axon.Shape.dense_kernel(parent_shape, units)
    bias_shape = Axon.Shape.dense_bias(parent_shape, units)
    output_shape = Axon.Shape.dense(parent_shape, units)

    weight = param(name <> "_weight", kernel_shape, weight_init)
    bias = param(name <> "_bias", bias_shape, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: :dense,
      params: [bias, weight],
      opts: []
    }

    if activation do
      node
      |> activation(activation)
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

  """
  @doc type: :layer
  def conv(%Axon{output_shape: parent_shape} = x, units, opts \\ [])
      when is_integer(units) and units > 0 do
    {id, name} = unique_identifiers(:conv, opts[:name])

    kernel_init = opts[:kernel_initializer] || :glorot_uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1

    kernel_size =
      if is_tuple(kernel_size),
        do: kernel_size,
        else: Tuple.to_list(List.duplicate(kernel_size, Nx.rank(parent_shape) - 2))

    strides =
      if is_list(strides),
        do: strides,
        else: List.duplicate(strides, Nx.rank(parent_shape) - 2)

    input_dilation =
      if is_list(input_dilation),
        do: input_dilation,
        else: List.duplicate(input_dilation, Nx.rank(parent_shape) - 2)

    kernel_dilation =
      if is_list(kernel_dilation),
        do: kernel_dilation,
        else: List.duplicate(kernel_dilation, Nx.rank(parent_shape) - 2)

    kernel_shape = Axon.Shape.conv_kernel(parent_shape, units, kernel_size)
    bias_shape = Axon.Shape.conv_bias(parent_shape, units, kernel_size)

    output_shape =
      Axon.Shape.conv(
        parent_shape,
        kernel_shape,
        strides,
        padding,
        input_dilation,
        kernel_dilation
      )

    kernel = param(name <> "_kernel", kernel_shape, kernel_init)
    bias = param(name <> "_bias", bias_shape, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: :conv,
      params: [bias, kernel],
      opts: [
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation
      ]
    }

    if activation do
      node
      |> activation(activation)
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

  Compiles to `Axon.Layers.depthwise_conv`/4.

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

  """
  @doc type: :layer
  def depthwise_conv(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    {id, name} = unique_identifiers(:depthwise_conv, opts[:name])

    kernel_init = opts[:kernel_initializer] || :glorot_uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1

    kernel_size =
      if is_tuple(kernel_size),
        do: kernel_size,
        else: Tuple.to_list(List.duplicate(kernel_size, Nx.rank(parent_shape) - 2))

    strides =
      if is_list(strides),
        do: strides,
        else: List.duplicate(strides, Nx.rank(parent_shape) - 2)

    input_dilation =
      if is_list(input_dilation),
        do: input_dilation,
        else: List.duplicate(input_dilation, Nx.rank(parent_shape) - 2)

    kernel_dilation =
      if is_list(kernel_dilation),
        do: kernel_dilation,
        else: List.duplicate(kernel_dilation, Nx.rank(parent_shape) - 2)

    kernel_shape = Axon.Shape.depthwise_conv_kernel(parent_shape, channel_multiplier, kernel_size)
    bias_shape = Axon.Shape.depthwise_conv_bias(parent_shape, channel_multiplier, kernel_size)

    output_shape =
      Axon.Shape.depthwise_conv(
        parent_shape,
        kernel_shape,
        strides,
        padding,
        input_dilation,
        kernel_dilation
      )

    kernel = param(name <> "_kernel", kernel_shape, kernel_init)
    bias = param(name <> "_bias", bias_shape, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: :depthwise_conv,
      params: [bias, kernel],
      opts: [
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation
      ]
    }

    if activation do
      node
      |> activation(activation)
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

  """
  @doc type: :layer
  def separable_conv2d(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    {id, name} = unique_identifiers(:separable_conv2d, opts[:name])

    kernel_init = opts[:kernel_initializer] || :glorot_uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1

    kernel_size =
      if is_tuple(kernel_size),
        do: kernel_size,
        else: Tuple.to_list(List.duplicate(kernel_size, Nx.rank(parent_shape) - 2))

    strides =
      if is_list(strides),
        do: strides,
        else: List.duplicate(strides, Nx.rank(parent_shape) - 2)

    input_dilation =
      if is_list(input_dilation),
        do: input_dilation,
        else: List.duplicate(input_dilation, Nx.rank(parent_shape) - 2)

    kernel_dilation =
      if is_list(kernel_dilation),
        do: kernel_dilation,
        else: List.duplicate(kernel_dilation, Nx.rank(parent_shape) - 2)

    k1_shape =
      Axon.Shape.separable_conv2d_kernel(parent_shape, channel_multiplier, kernel_size, 1)

    k2_shape =
      Axon.Shape.separable_conv2d_kernel(parent_shape, channel_multiplier, kernel_size, 2)

    b1_shape = Axon.Shape.separable_conv2d_bias(parent_shape, channel_multiplier, kernel_size)
    b2_shape = Axon.Shape.separable_conv2d_bias(parent_shape, channel_multiplier, kernel_size)

    output_shape =
      Axon.Shape.depthwise_conv(
        parent_shape,
        Axon.Shape.depthwise_conv_kernel(parent_shape, channel_multiplier, kernel_size),
        strides,
        padding,
        input_dilation,
        kernel_dilation
      )

    k1 = param(name <> "_kernel_1", k1_shape, kernel_init)
    b1 = param(name <> "_bias_1", b1_shape, bias_init)
    k2 = param(name <> "_kernel_2", k2_shape, kernel_init)
    b2 = param(name <> "_bias_2", b2_shape, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: :separable_conv2d,
      params: [b1, k1, b2, k2],
      opts: [
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation
      ]
    }

    if activation do
      node
      |> activation(activation)
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

  """
  @doc type: :layer
  def separable_conv3d(%Axon{output_shape: parent_shape} = x, channel_multiplier, opts \\ [])
      when is_integer(channel_multiplier) and channel_multiplier >= 1 do
    {id, name} = unique_identifiers(:separable_conv3d, opts[:name])

    kernel_init = opts[:kernel_initializer] || :glorot_uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1

    kernel_size =
      if is_tuple(kernel_size),
        do: kernel_size,
        else: Tuple.to_list(List.duplicate(kernel_size, Nx.rank(parent_shape) - 2))

    strides =
      if is_list(strides),
        do: strides,
        else: List.duplicate(strides, Nx.rank(parent_shape) - 2)

    input_dilation =
      if is_list(input_dilation),
        do: input_dilation,
        else: List.duplicate(input_dilation, Nx.rank(parent_shape) - 2)

    kernel_dilation =
      if is_list(kernel_dilation),
        do: kernel_dilation,
        else: List.duplicate(kernel_dilation, Nx.rank(parent_shape) - 2)

    k1_shape =
      Axon.Shape.separable_conv3d_kernel(parent_shape, channel_multiplier, kernel_size, 1)

    k2_shape =
      Axon.Shape.separable_conv3d_kernel(parent_shape, channel_multiplier, kernel_size, 2)

    k3_shape =
      Axon.Shape.separable_conv3d_kernel(parent_shape, channel_multiplier, kernel_size, 3)

    b1_shape = Axon.Shape.separable_conv3d_bias(parent_shape, channel_multiplier, kernel_size)
    b2_shape = Axon.Shape.separable_conv3d_bias(parent_shape, channel_multiplier, kernel_size)
    b3_shape = Axon.Shape.separable_conv3d_bias(parent_shape, channel_multiplier, kernel_size)

    output_shape =
      Axon.Shape.depthwise_conv(
        parent_shape,
        Axon.Shape.depthwise_conv_kernel(parent_shape, channel_multiplier, kernel_size),
        strides,
        padding,
        input_dilation,
        kernel_dilation
      )

    k1 = param(name <> "_kernel_1", k1_shape, kernel_init)
    b1 = param(name <> "_bias_1", b1_shape, bias_init)
    k2 = param(name <> "_kernel_2", k2_shape, kernel_init)
    b2 = param(name <> "_bias_2", b2_shape, bias_init)
    k3 = param(name <> "_kernel_3", k3_shape, kernel_init)
    b3 = param(name <> "_bias_3", b3_shape, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: :separable_conv3d,
      params: [b1, k1, b2, k2, b3, k3],
      opts: [
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation
      ]
    }

    if activation do
      node
      |> activation(activation)
    else
      node
    end
  end

  @doc """
  Adds an activation layer to the network.

  Activation layers are element-wise functions typically called
  after the output of another layer.

  ## Options

    - `name` - Layer name.

  """
  @doc type: :activation
  def activation(%Axon{output_shape: shape} = x, activation, opts \\ [])
      when is_atom(activation) do
    id = System.unique_integer([:positive, :monotonic])
    name = opts[:name] || "#{Atom.to_string(activation)}_#{id}"
    %Axon{id: id, name: name, output_shape: shape, parent: x, op: activation, params: []}
  end

  ## Activation

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  for activation <- @activation_layers do
    @doc """
    Adds #{Atom.to_string(activation)} activation layer to the network.

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

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  for dropout <- @dropout_layers do
    @doc """
    Adds #{Atom.to_string(dropout)} layer to the network.

    See `Axon.Layers.#{Atom.to_string(dropout)}` for more details.

    ## Options

      * `:name` - Layer name.
      * `:rate` - Dropout rate.

    """
    @doc type: :layer
    def unquote(dropout)(%Axon{} = x, opts \\ []) do
      dropout(x, unquote(dropout), opts)
    end
  end

  defp dropout(%Axon{output_shape: parent_shape} = x, dropout, opts) do
    {id, name} = unique_identifiers(dropout, opts[:name])
    rate = opts[:rate] || 0.5

    %Axon{
      id: id,
      name: name,
      op: dropout,
      output_shape: parent_shape,
      parent: x,
      params: [],
      opts: [
        rate: rate
      ]
    }
  end

  ## Pooling

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  for pool <- @pooling_layers do
    @doc """
    Adds #{Atom.to_string(pool)} layer to the network.

    See `Axon.Layers.#{Atom.to_string(pool)}` for more details.

    ## Options

      * `:name` - Layer name.
      * `:kernel_size` - Pooling kernel size.
      * `:strides` - Pooling strides.

    """
    @doc type: :layer
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      pool(x, unquote(pool), opts)
    end
  end

  defp pool(%Axon{output_shape: parent_shape} = x, pool, opts) do
    {id, name} = unique_identifiers(pool, opts[:name])

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid

    kernel_size =
      if is_tuple(kernel_size),
        do: kernel_size,
        else: Tuple.to_list(List.duplicate(kernel_size, Nx.rank(parent_shape) - 2))

    strides =
      if is_list(strides),
        do: strides,
        else: List.duplicate(strides, Nx.rank(parent_shape) - 2)

    output_shape = Axon.Shape.pool(parent_shape, kernel_size, strides, padding)

    %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: pool,
      params: [],
      opts: [
        kernel_size: kernel_size,
        strides: strides,
        padding: padding
      ]
    }
  end

  ## Adaptive Pooling

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool]

  for pool <- @adaptive_pooling_layers do
    @doc """
    Adds #{Atom.to_string(pool)} layer to the network.

    See `Axon.Layers.#{Atom.to_string(pool)}` for more details.

    ## Options

      * `:name` - Layer name.
      * `:output_size` - Layer output size.

    """
    @doc type: :layer
    def unquote(pool)(%Axon{} = x, opts \\ []) do
      adaptative_pool(x, unquote(pool), opts)
    end
  end

  defp adaptative_pool(%Axon{output_shape: parent_shape} = x, pool, opts) do
    {id, name} = unique_identifiers(pool, opts[:name])

    output_size = opts[:output_size]

    output_size =
      if is_tuple(output_size),
        do: output_size,
        else: Tuple.to_list(List.duplicate(output_size, Nx.rank(parent_shape) - 2))

    output_shape = Axon.Shape.adaptive_pool(parent_shape, output_size)

    %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: pool,
      params: [],
      opts: [
        output_size: output_size
      ]
    }
  end

  ## Normalization

  @normalization_layers [:batch_norm, :layer_norm, :instance_norm]

  for norm <- @normalization_layers do
    @doc """
    Adds #{Atom.to_string(norm)} layer to the network.

    See `Axon.Layers.#{Atom.to_string(norm)}` for more details.

    ## Options

      * `:name` - Layer name.
      * `:gamma_initializer` - Gamma parameter initializer.
      * `:beta_initializer` - Beta parameter initializer.
      * `:channel_index` - Input feature index used for calculating
        mean and variance.
      * `:epsilon` - Numerical stability term.

    """
    @doc type: :layer
    def unquote(norm)(%Axon{} = x, opts \\ []) do
      norm(x, unquote(norm), opts)
    end
  end

  defp norm(%Axon{output_shape: shape} = x, norm, opts) do
    {id, name} = unique_identifiers(norm, opts[:name])

    gamma_init = opts[:gamma_initializer] || :glorot_uniform
    beta_init = opts[:beta_initializer] || :zeros

    channel_index = opts[:channel_index] || 1
    epsilon = opts[:epsilon] || 1.0e-5

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)

    gamma = param(name <> "_gamma", gamma_shape, gamma_init)
    beta = param(name <> "_beta", beta_shape, beta_init)

    %Axon{
      id: id,
      name: name,
      output_shape: shape,
      parent: x,
      op: norm,
      params: [beta, gamma],
      opts: [
        epsilon: epsilon,
        channel_index: channel_index
      ]
    }
  end

  @doc """
  Adds a group normalization layer to the network.

  See `Axon.Layers.group_norm` for more details.

  ## Options

    * `:name` - Layer name.
    * `:gamma_initializer` - Gamma parameter initializer.
    * `:beta_initializer` - Beta parameter initializer.
    * `:channel_index` - Input feature index used for calculating
      mean and variance.
    * `:epsilon` - Numerical stability term.

  """
  @doc type: :layer
  def group_norm(%Axon{output_shape: shape} = x, group_size, opts \\ [])
      when is_integer(group_size) and group_size >= 1 do
    {id, name} = unique_identifiers(:group_norm, opts[:name])

    gamma_init = opts[:gamma_initializer] || :glorot_uniform
    beta_init = opts[:beta_initializer] || :zeros

    channel_index = opts[:channel_index] || 1
    epsilon = opts[:epsilon] || 1.0e-5

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)

    gamma = param(name <> "_gamma", gamma_shape, gamma_init)
    beta = param(name <> "_beta", beta_shape, beta_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: shape,
      parent: x,
      op: :group_norm,
      params: [beta, gamma],
      opts: [
        epsilon: epsilon,
        channel_index: channel_index,
        group_size: group_size
      ]
    }

    node
  end

  @doc """
  Applies the given `Nx` expression to the input.

  ## Options

    * `name` - Layer name.

  """
  @doc type: :composition
  def nx(%Axon{output_shape: shape} = x, fun, opts \\ []) when is_function(fun, 1) do
    {id, name} = unique_identifiers(:nx, opts[:name])

    param = Nx.Defn.Expr.parameter(:nx, {:f, 32}, shape, 0)

    expr =
      if Nx.Defn.Compiler.current() do
        fun.(param)
      else
        Nx.Defn.jit(fun, [param], compiler: Axon.Defn)
      end

    node = %Axon{
      id: id,
      name: name,
      output_shape: expr.shape,
      parent: x,
      op: :nx,
      params: [],
      opts: [
        fun: fun
      ]
    }

    node
  end

  @doc """
  Adds a flatten layer to the network.

  This layer will flatten all but the batch dimensions
  of the input into a single layer. Typically called to flatten
  the output of a convolution for use with a dense layer.

  ## Options

    * `:name` - Layer name.

  """
  @doc type: :composition
  def flatten(%Axon{output_shape: shape} = x, opts \\ []) do
    {id, name} = unique_identifiers(:flatten, opts[:name])
    new_shape = Axon.Shape.flatten(shape)
    %Axon{id: id, name: name, output_shape: new_shape, parent: x, op: :flatten, params: []}
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
  def concatenate(%Axon{output_shape: x_shape} = x, %Axon{output_shape: y_shape} = y, opts) do
    {id, name} = unique_identifiers(:concatenate, opts[:name])
    axis = opts[:axis] || Nx.rank(x_shape) - 1
    output_shape = Axon.Shape.concatenate([x_shape, y_shape], axis)

    %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: [x, y],
      op: :concatenate,
      params: [],
      opts: [axis: axis]
    }
  end

  @doc type: :composition
  def concatenate([%Axon{output_shape: shape} | _] = inputs, opts) when is_list(inputs) do
    {id, name} = unique_identifiers(:concatenate, opts[:name])
    axis = opts[:axis] || Nx.rank(shape) - 1
    input_shapes = inputs |> Enum.map(fn %Axon{output_shape: shape} -> shape end)
    output_shape = Axon.Shape.concatenate(input_shapes, axis)

    %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: inputs,
      op: :concatenate,
      params: [],
      opts: [axis: axis]
    }
  end

  @element_wise_layers [:add, :subtract, :multiply]

  for op <- @element_wise_layers do
    @doc """
    Adds a #{op} layer to the network.

    This layer performs an element-wise #{Atom.to_string(op)} operation
    on input layers. All input layers must be the same shape.

    ## Options

      * `:name` - Layer name.

    """
    @doc type: :layer
    def unquote(op)(%Axon{output_shape: shape} = x, %Axon{output_shape: shape} = y) do
      {id, name} = unique_identifiers(unquote(op), nil)
      %Axon{id: id, name: name, output_shape: shape, parent: [x, y], op: unquote(op), params: []}
    end

    @doc type: :layer
    def unquote(op)([%Axon{output_shape: shape} | rest] = inputs) do
      {id, name} = unique_identifiers(unquote(op), nil)

      output_shape =
        Enum.reduce(rest, shape, fn %Axon{output_shape: shape}, acc ->
          unless shape == acc do
            raise ArgumentError, "all input shapes must match"
          end
        end)

      %Axon{
        id: id,
        name: name,
        output_shape: output_shape,
        parent: inputs,
        op: unquote(op),
        params: []
      }
    end
  end

  @doc """
  Compiles the given model to `{init_fn, predict_fn}`.
  """
  def compile(model) do
    Axon.Compiler.__compile__(model)
  end

  @doc """
  Compiles and runs the given models initialization function
  with the given compiler options.
  """
  @doc type: :execution
  defmacro init(model, opts \\ []) do
    define_init(model, :init, [], opts)
  end

  @doc """
  Compiles and runs the given Axon model with `params` on
  `input` with the given compiler options.
  """
  @doc type: :execution
  defmacro predict(model, params, input, opts \\ []) do
    define_predict(model, :predict, [params, input], opts)
  end

  ## Implementation

  defp define_init(model, caller, args, opts \\ []) do
    quote do
      Nx.Defn.Kernel.transform(unquote(args), fn args ->
        model = unquote(model)
        opts = unquote(opts)
        caller = unquote(caller)

        Axon.Compiler.__jit_init__(model, caller, args, opts)
      end)
    end
  end

  defp define_predict(model, caller, args, opts \\ []) do
    quote do
      Nx.Defn.Kernel.transform(unquote(args), fn args ->
        model = unquote(model)
        opts = unquote(opts)
        caller = unquote(caller)

        Axon.Compiler.__jit_predict__(model, caller, args, opts)
      end)
    end
  end

  ## Inspection

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(axon, _opts) do
      title = "Model"
      header = ["Layer", "Shape", "Parameters"]
      {_, cache} = axon_to_rows(axon, %{})

      rows =
        cache
        |> Enum.sort()
        |> Enum.unzip()
        |> Kernel.elem(1)

      rows
      |> TableRex.Table.new(header, title)
      |> TableRex.Table.render!(
        header_separator_symbol: "=",
        title_separator_symbol: "=",
        vertical_style: :off
      )
      |> string()
    end

    defp axon_to_rows(%{id: id} = graph, cache) do
      case cache do
        %{^id => row} ->
          {row, cache}

        %{} ->
          {row, cache} = do_axon_to_rows(graph, cache)
          cache = Map.put(cache, id, row)
          {row, cache}
      end
    end

    defp do_axon_to_rows(%Axon{op: op, parent: parents, name: name, output_shape: shape}, cache)
         when is_list(parents) do
      {names, cache} =
        Enum.map_reduce(parents, cache, fn %Axon{name: name} = graph, cache ->
          {_, cache} = axon_to_rows(graph, cache)
          {name, cache}
        end)

      row = [name <> " ( #{Atom.to_string(op)} #{inspect(names)} )", "#{inspect(shape)}", 0]

      {row, cache}
    end

    defp do_axon_to_rows(
           %Axon{op: op, params: params, parent: parent, name: name, output_shape: shape},
           cache
         ) do
      cache =
        if parent do
          {_, cache} = axon_to_rows(parent, cache)
          cache
        else
          cache
        end

      num_params =
        params
        |> Enum.reduce(0, fn %Axon.Parameter{shape: shape}, acc -> acc + Nx.size(shape) end)

      row = [name <> " ( #{Atom.to_string(op)} )", "#{inspect(shape)}", "#{num_params}"]
      {row, cache}
    end
  end

  ## Helpers

  defp unique_identifiers(type, nil) do
    id = System.unique_integer([:positive, :monotonic])
    {id, Atom.to_string(type) <> "_#{id}"}
  end

  defp unique_identifiers(_type, name), do: {System.unique_integer([:positive, :monotonic]), name}

  defp param(name, shape, initializer, _opts \\ []) do
    id = System.unique_integer([:positive, :monotonic])
    %Axon.Parameter{id: id, name: name, shape: shape, initializer: initializer}
  end
end
