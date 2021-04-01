defmodule Axon do
  @moduledoc """
  Nx-powered Neural Networks.

  Axon provides a high-level interface for creating neural
  networks in Elixir. Axon is built entirely on top of Nx
  numerical definitions, so every neural network can be
  JIT or AOT compiled using any Nx compiler, or even
  transformed into high-level neural network formats
  like TensorFlow Lite and ONNX.
  """
  alias __MODULE__, as: Axon

  @type t :: %__MODULE__{}

  defstruct [:id, :name, :output_shape, :parent, :op, :params, :opts]

  defmacro __using__(_opts) do
    quote do
      require Axon
      import Axon
      import Nx.Defn
    end
  end

  @doc """
  Adds an input layer to the network.

  Input layers specify a models inputs. Input layers are
  always the root layers of the neural network.

  ## Options

    * `name` - Layer name.

  """
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

  Compiles to `Axon.Layers.dense/4`.

  ## Options

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.

  """
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
  """
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
    k2 = param(name <> "_kernel_2", k2_shape, kernel_init)
    b1 = param(name <> "_bias_1", b1_shape, bias_init)
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
  """
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
    k2 = param(name <> "_kernel_2", k2_shape, kernel_init)
    k3 = param(name <> "_kernel_3", k3_shape, kernel_init)
    b1 = param(name <> "_bias_1", b1_shape, bias_init)
    b2 = param(name <> "_bias_2", b2_shape, bias_init)
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
  def activation(%Axon{output_shape: shape} = x, activation, opts \\ [])
      when is_atom(activation) do
    id = System.unique_integer([:positive, :monotonic])
    name = opts[:name] || "#{Atom.to_string(activation)}_#{id}"
    %Axon{id: id, name: name, output_shape: shape, parent: x, op: activation, params: []}
  end

  ## Activation

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :log_softmax, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  for activation <- @activation_layers do
    @doc """
    Adds #{Atom.to_string(activation)} activation layer to the network.

    See `Axon.Activations.#{Atom.to_string(activation)}/1` for more details.

    ## Options

      - `name` - Layer name.

    """
    def unquote(activation)(%Axon{} = x, opts \\ []) do
      activation(x, unquote(activation), opts)
    end
  end

  ## Dropout

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  for dropout <- @dropout_layers do
    def unquote(dropout)(%Axon{output_shape: parent_shape} = x, opts \\ []) do
      {id, name} = unique_identifiers(unquote(dropout), opts[:name])

      rate = opts[:rate] || 0.5

      node = %Axon{
        id: id,
        name: name,
        op: unquote(dropout),
        output_shape: parent_shape,
        parent: x,
        params: [],
        opts: [
          rate: rate
        ]
      }

      node
    end
  end

  ## Pooling

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  for pool <- @pooling_layers do
    def unquote(pool)(%Axon{output_shape: parent_shape} = x, opts \\ []) do
      {id, name} = unique_identifiers(unquote(pool), opts[:name])

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

      node = %Axon{
        id: id,
        name: name,
        output_shape: output_shape,
        parent: x,
        op: unquote(pool),
        params: [],
        opts: [
          kernel_size: kernel_size,
          strides: strides,
          padding: padding
        ]
      }

      node
    end
  end

  ## Adaptive Pooling

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool]

  for pool <- @adaptive_pooling_layers do
    def unquote(pool)(%Axon{output_shape: parent_shape} = x, opts \\ []) do
      {id, name} = unique_identifiers(unquote(pool), opts[:name])

      output_size = opts[:output_size]

      output_size =
        if is_tuple(output_size),
          do: output_size,
          else: Tuple.to_list(List.duplicate(output_size, Nx.rank(parent_shape) - 2))

      output_shape = Axon.Shape.adaptive_pool(parent_shape, output_size)

      node = %Axon{
        id: id,
        name: name,
        output_shape: output_shape,
        parent: x,
        op: unquote(pool),
        params: [],
        opts: [
          output_size: output_size
        ]
      }

      node
    end
  end

  ## Normalization

  @normalization_layers [:batch_norm, :layer_norm, :instance_norm]

  for op <- @normalization_layers do
    def unquote(op)(%Axon{output_shape: shape} = x, opts \\ []) do
      {id, name} = unique_identifiers(unquote(op), opts[:name])

      channel_index = opts[:channel_index] || 1
      epsilon = opts[:epsilon] || 1.0e-5

      gamma_shape = Axon.Shape.norm_param(shape, channel_index)
      beta_shape = Axon.Shape.norm_param(shape, channel_index)

      gamma = param(name <> "_gamma", gamma_shape, :glorot_uniform)
      beta = param(name <> "_beta", beta_shape, :glorot_uniform)

      node = %Axon{
        id: id,
        name: name,
        output_shape: shape,
        parent: x,
        op: unquote(op),
        params: [beta, gamma],
        opts: [
          epsilon: epsilon,
          channel_index: channel_index
        ]
      }

      node
    end
  end

  @doc """
  Adds a group normalization layer to the network.
  """
  def group_norm(%Axon{output_shape: shape} = x, group_size, opts \\ [])
      when is_integer(group_size) and group_size >= 1 do
    {id, name} = unique_identifiers(:group_norm, opts[:name])

    channel_index = opts[:channel_index] || 1
    epsilon = opts[:epsilon] || 1.0e-5

    gamma_shape = Axon.Shape.norm_param(shape, channel_index)
    beta_shape = Axon.Shape.norm_param(shape, channel_index)

    gamma = param(name <> "_gamma", gamma_shape, :glorot_uniform)
    beta = param(name <> "_beta", beta_shape, :glorot_uniform)

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

    * `name` - Layer name.

  """
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

    * `axis` - Concatenate axis.
  """
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
    Adds an #{op} layer to the network.
    """
    def unquote(op)(%Axon{output_shape: shape} = x, %Axon{output_shape: shape} = y, opts) do
      {id, name} = unique_identifiers(unquote(op), opts[:name])
      %Axon{id: id, name: name, output_shape: shape, parent: [x, y], op: unquote(op), params: []}
    end

    def unquote(op)([%Axon{output_shape: shape} | rest] = inputs, opts) do
      {id, name} = unique_identifiers(unquote(op), opts[:name])

      output_shape =
        rest
        |> Enum.reduce(shape, fn %Axon{output_shape: shape}, acc ->
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
  Compiles and runs the given models initialization function.
  """
  defmacro init(model, opts \\ []) do
    define_init(model, :init, [], opts)
  end

  @doc """
  Compiles and runs the given Axon model with `params` on
  `input` with the given compiler options.
  """
  defmacro predict(model, params, input, opts \\ []) do
    define_predict(model, :predict, [params, input], opts)
  end

  @doc """
  Applies updates to params.
  """
  defmacro apply_updates(params, updates) do
    quote do
      Nx.Defn.Kernel.transform({unquote(params), unquote(updates)}, fn {params, updates} ->
        params
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(updates))
        |> Enum.map(fn {x, u} -> Nx.add(x, u) end)
        |> List.to_tuple()
      end)
    end
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
      rows = axon_to_rows(axon, [])

      rows
      |> TableRex.Table.new(header, title)
      |> TableRex.Table.render!(
        header_separator_symbol: "=",
        title_separator_symbol: "=",
        vertical_style: :off
      )
      |> string()
    end

    defp axon_to_rows(%Axon{op: :input, output_shape: shape, parent: nil, name: name}, layers) do
      row = [name <> " (input)", "#{inspect(shape)}", 0]
      [row | layers]
    end

    # This is bad
    defp axon_to_rows(%Axon{op: op, parent: parents, name: name, output_shape: shape}, layers) when is_list(parents) do
      {names, rows} =
        Enum.map_reduce(parents, layers, fn %Axon{name: name} = node, acc ->
          {name, Enum.uniq(axon_to_rows(node, acc))}
        end)

      row = [name <> "( #{Atom.to_string(op)} #{inspect(names)} )", "#{inspect(shape)}", 0]

      (rows -- layers) ++ [row] ++ layers
    end

    defp axon_to_rows(
           %Axon{op: op, output_shape: shape, parent: x, name: name, params: params},
           layers
         ) do
      total_params =
        params
        |> Enum.reduce(0, fn %Axon.Parameter{shape: shape}, acc -> Nx.size(shape) + acc end)

      row = [name <> " (#{Atom.to_string(op)})", "#{inspect(shape)}", "#{total_params}"]
      axon_to_rows(x, [row | layers])
    end
  end

  ## Helpers

  defp unique_identifiers(type, nil) do
    id = System.unique_integer([:positive, :monotonic])
    {id, Atom.to_string(type) <> "_#{id}"}
  end

  defp unique_identifiers(_type, name), do: {System.unique_integer([:positive, :monotonic]), name}

  defp param(name, shape, initializer, _opts \\ []) do
    %Axon.Parameter{name: name, shape: shape, initializer: initializer}
  end
end
