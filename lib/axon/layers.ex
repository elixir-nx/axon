defmodule Axon.Layers do
  @moduledoc ~S"""
  Functional implementations of common neural network layer
  operations.

  Layers are the building blocks of neural networks. These
  functional implementations can be used to express higher-level
  constructs using fundamental building blocks. Neural network
  layers are stateful with respect to their parameters.
  These implementations do not assume the responsibility of
  managing state - instead opting to delegate this responsibility
  to the caller.

  Basic neural networks can be seen as a composition of functions:

      input
      |> dense(w1, b1)
      |> relu()
      |> dense(w2, b2)
      |> softmax()

  These kinds of models are often referred to as deep feedforward networks
  or multilayer perceptrons (MLPs) because information flows forward
  through the network with no feedback connections. Mathematically,
  a feedforward network can be represented as:

    $$f(x) = f^{(3)}(f^{(2)}(f^{(1)}(x)))$$

  You can see a similar pattern emerge if we condense the call stack
  in the previous example:

      softmax(dense(relu(dense(input, w1, b1)), w2, b2))

  The chain structure shown here is the most common structure used
  in neural networks. You can consider each function $f^{(n)}$ as a
  *layer* in the neural network - for example $f^{(2)} is the 2nd
  layer in the network. The number of function calls in the
  structure is the *depth* of the network. This is where the term
  *deep learning* comes from.

  Neural networks are often written as the mapping:

    $$y = f(x; \theta)$$

  Where $x$ is the input to the neural network and $\theta$ are the
  set of learned parameters. In Elixir, you would write this:

      y = model(input, params)

  From the previous example, `params` would represent the collection:

      {w1, b1, w2, b2}

  where `w1` and `w2` are layer *weights*, and `b1` and `b2` are layer
  *biases*.

  """

  import Nx.Defn
  import Axon.Shared

  ## Linear

  @doc ~S"""
  Functional implementation of a dense layer.

  Linear transformation of the input such that:

  $$y = xW^T + b$$

  A dense layer or fully connected layer transforms
  the input using the given weight matrix and bias
  to compute:

      Nx.dot(input, weight) + bias

  Typically, both `weight` and `bias` are learnable
  parameters trained using gradient-based optimization.

  ## Parameter Shapes

    * `input` - `{batch_size, ..., input_features}`
    * `weight` - `{input_features, output_features}`
    * `bias` - `{output_features}`

  ## Output Shape

    `{batch_size, output_features}`

  ## Examples

      iex> input = Nx.tensor([[1.0, 0.5, 1.0, 0.5], [0.0, 0.0, 0.0, 0.0]], type: {:f, 32})
      iex> weight = Nx.tensor([[0.2], [0.3], [0.5], [0.8]], type: {:f, 32})
      iex> bias = Nx.tensor([1.0], type: {:f, 32})
      iex> Axon.Layers.dense(input, weight, bias)
      #Nx.Tensor<
        f32[2][1]
        [
          [2.25],
          [1.0]
        ]
      >
  """
  @doc type: :linear
  defn dense(input, weight, bias) do
    input
    |> Nx.dot([Nx.rank(input) - 1], weight, [0])
    |> Nx.add(bias)
  end

  @doc ~S"""
  Functional implementation of a bilinear layer.

  Bilinear transformation of the input such that:

  $$y = x_1^{T}Ax_2 + b$$

  ## Parameter Shapes

    * `input1` - `{batch_size, ..., input1_features}`
    * `input2` - `{batch_size, ..., input2_features}`
    * `weight` - `{out_features, input1_features, input2_features}`

  ## Output Shape

    `{batch_size, ..., output_features}`

  ## Examples

      iex> inp1 = Nx.iota({3, 2}, type: {:f, 32})
      iex> inp2 = Nx.iota({3, 4}, type: {:f, 32})
      iex> weight = Nx.iota({1, 2, 4}, type: {:f, 32})
      iex> bias = Nx.tensor(1.0)
      iex> Axon.Layers.bilinear(inp1, inp2, weight, bias)
      #Nx.Tensor<
        f32[3][1]
        [
          [39.0],
          [455.0],
          [1319.0]
        ]
      >
  """
  @doc type: :linear
  defn bilinear(input1, input2, weight, bias) do
    inp1_axes = transform(Nx.rank(input1), fn rank -> [rank - 1] end)
    inp2_axes = transform(Nx.rank(input2), fn rank -> [rank - 1] end)

    input1
    |> Nx.dot(inp1_axes, [], weight, [1], [])
    |> Nx.dot([2], [0], input2, inp2_axes, [0])
    |> Nx.add(bias)
  end

  ## Convolutional

  @doc """
  Functional implementation of a general dimensional convolutional
  layer.

  Convolutional layers can be described as applying a convolution
  over an input signal composed of several input planes. Intuitively,
  the input kernel slides `output_channels` number of filters over
  the input tensor to extract features from the input tensor.

  Convolutional layers are most commonly used in computer vision,
  but can also be useful when working with sequences and other input signals.

  ## Parameter Shapes

    * `input` - `{batch_size, input_channels, input_spatial0, ..., input_spatialN}`
    * `weight` - `{output_channels, input_channels, kernel_spatial0, ..., kernel_spatialN}`
    * `bias` - `{output_channels}` or `{}`

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a list
      who's length matches the number of spatial dimensions in
      the input tensor. Defaults to 1.

    * `:padding` - zero padding on the input. Can be one of
      `:valid`, `:same` or a general padding configuration
      without interior padding for each spatial dimension
      of the input.

    * `:input_dilation` - input dilation factor. Equivalent
      to applying interior padding on the input. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:kernel_dilation` - kernel dilation factor. Equivalent
      to applying interior padding on the kernel. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  ## Examples

  ### One-dimensional convolution

      iex> input = Nx.tensor([[[0.1294, -0.6638, 1.0251]], [[ 0.9182,  1.1512, -1.6149]]], type: {:f, 32})
      iex> weight = Nx.tensor([[[-1.5475, 1.2425]], [[0.1871, 0.5458]], [[-0.4488,  0.8879]]], type: {:f, 32})
      iex> bias = Nx.tensor([0.7791, 0.1676, 1.5971], type: {:f, 32})
      iex> Axon.Layers.conv(input, weight, bias)
      #Nx.Tensor<
        f32[2][3][2]
        [
          [
            [-0.24591797590255737, 3.08001708984375],
            [-0.1704912781715393, 0.6029025316238403],
            [0.9496372938156128, 2.80519962310791]
          ],
          [
            [0.7885514497756958, -3.0088953971862793],
            [0.9677201509475708, -0.4984228312969208],
            [2.207162380218506, -0.3534282445907593]
          ]
        ]
      >

  ### Two-dimensional convolution

      iex> input = Nx.tensor([[[[-1.0476, -0.5041], [-0.9336, 1.5907]]]], type: {:f, 32})
      iex> weight = Nx.tensor([
      ...>  [[[0.7514, 0.7356], [1.3909,  0.6800]]],
      ...>  [[[-0.3450,  0.4551], [-0.6275, -0.9875]]],
      ...>  [[[1.8587, 0.4722], [0.6058, -1.0301]]]
      ...> ], type: {:f, 32})
      iex> bias = Nx.tensor([1.9564, 0.2822, -0.5385], type: {:f, 32})
      iex> Axon.Layers.conv(input, weight, bias)
      #Nx.Tensor<
        f32[1][3][1][1]
        [
          [
            [
              [0.5815491676330566]
            ],
            [
              [-0.5707762241363525]
            ],
            [
              [-4.927865028381348]
            ]
          ]
        ]
      >

    ### Three-dimensional convolution

    iex> input = Nx.tensor([[[[[-0.6497], [1.0939]], [[-2.5465], [0.7801]]]]], type: {:f, 32})
    iex> weight = Nx.tensor([
    ...>  [[[[ 0.7390], [-0.0927]], [[-0.8675], [-0.9209]]]],
    ...>  [[[[-0.6638], [0.4341]], [[0.6368], [1.1846]]]]
    ...> ], type: {:f, 32})
    iex> bias = Nx.tensor([-0.4101,  0.1776], type: {:f, 32})
    iex> Axon.Layers.conv(input, weight, bias)
    #Nx.Tensor<
      f32[1][2][1][1][1]
      [
        [
          [
            [
              [0.49906185269355774]
            ]
          ],
          [
            [
              [0.38622811436653137]
            ]
          ]
        ]
      ]
    >
  """
  @doc type: :convolutional
  defn conv(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: 1,
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        feature_group_size: 1,
        batch_group_size: 1,
        channels: :first
      )

    bias_reshape =
      transform({Nx.shape(bias), Nx.rank(input) - 2, opts[:channels]}, fn {bias_shape, rank,
                                                                           channels} ->
        Axon.Shape.conv_bias_reshape(bias_shape, rank, channels)
      end)

    permutations =
      transform({Nx.rank(input), opts[:channels]}, fn
        {rank, :first} ->
          Enum.to_list(0..(rank - 1))

        {rank, :last} ->
          spatial = Enum.to_list(1..(rank - 2)//1)
          [0, rank - 1 | spatial]

        {_rank, invalid} ->
          raise ArgumentError, "invalid channel configuration, #{inspect(invalid)}"
      end)

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: opts[:feature_group_size],
      batch_group_size: opts[:batch_group_size],
      input_permutation: permutations,
      output_permutation: permutations
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a general dimensional transposed
  convolutional layer.

  *Note: This layer is currently implemented as a fractionally strided
  convolution by padding the input tensor. Please open an issue if you'd
  like this behavior changed.*

  Transposed convolutions are sometimes (incorrectly) referred to as
  deconvolutions because it "reverses" the spatial dimensions
  of a normal convolution. Transposed convolutions are a form of upsampling -
  they produce larger spatial dimensions than the input tensor. They
  can be thought of as a convolution in reverse - and are sometimes
  implemented as the backward pass of a normal convolution.

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a list
      who's length matches the number of spatial dimensions in
      the input tensor. Defaults to 1.

    * `:padding` - zero padding on the input. Can be one of
      `:valid`, `:same` or a general padding configuration
      without interior padding for each spatial dimension
      of the input.

    * `:input_dilation` - input dilation factor. Equivalent
      to applying interior padding on the input. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:kernel_dilation` - kernel dilation factor. Equivalent
      to applying interior padding on the kernel. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  ## Examples

      iex> input = Nx.iota({1, 3, 3}, type: {:f, 32})
      iex> kernel = Nx.iota({6, 3, 2}, type: {:f, 32})
      iex> bias = Nx.tensor(1.0, type: {:f, 32})
      iex> Axon.Layers.conv_transpose(input, kernel, bias)
      #Nx.Tensor<
        f32[1][6][4]
        [
          [
            [40.0, 79.0, 94.0, 43.0],
            [94.0, 205.0, 256.0, 133.0],
            [148.0, 331.0, 418.0, 223.0],
            [202.0, 457.0, 580.0, 313.0],
            [256.0, 583.0, 742.0, 403.0],
            [310.0, 709.0, 904.0, 493.0]
          ]
        ]
      >

  ## References

    * [A guide to convolution arithmethic for deep learning](https://arxiv.org/abs/1603.07285v1)
    * [Deconvolutional Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """
  @doc type: :convolutional
  defn conv_transpose(input, weight, bias, opts \\ []) do
    assert_equal_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: 1,
        padding: :valid,
        kernel_dilation: 1,
        channels: :first
      )

    strides =
      transform(
        {Nx.rank(input), opts[:strides]},
        fn
          {_, [_ | _] = strides} -> strides
          {rank, strides} -> List.duplicate(strides, rank - 2)
        end
      )

    padding =
      transform(
        {Nx.shape(weight), opts[:kernel_dilation], strides, opts[:padding]},
        fn {shape, k_dilation, strides, padding} ->
          Axon.Shape.conv_transpose_padding(shape, k_dilation, strides, padding)
        end
      )

    ones = transform(Nx.rank(input), &List.duplicate(1, &1 - 2))

    conv(input, weight, bias,
      strides: ones,
      padding: padding,
      input_dilation: strides,
      kernel_dilation: opts[:kernel_dilation],
      channels: opts[:channels]
    )
  end

  @doc """
  Functional implementation of a general dimensional depthwise
  convolution.

  Depthwise convolutions apply a single convolutional filter to
  each input channel. This is done by setting `feature_group_size`
  equal to the number of input channels. This will split the
  `output_channels` into `input_channels` number of groups and
  convolve the grouped kernel channels over the corresponding input
  channel.

  ## Parameter Shapes

    * `input` - `{batch_size, input_channels, input_spatial0, ..., input_spatialN}`
    * `weight` - `{output_channels, 1, kernel_spatial0, ..., kernel_spatialN}`
    * `bias` - `{output_channels}` or `{}`

    `output_channels` must be a multiple of the input channels.

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a list
      who's length matches the number of spatial dimensions in
      the input tensor. Defaults to 1.

    * `:padding` - zero padding on the input. Can be one of
      `:valid`, `:same` or a general padding configuration
      without interior padding for each spatial dimension
      of the input.

    * `:input_dilation` - input dilation factor. Equivalent
      to applying interior padding on the input. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:kernel_dilation` - kernel dilation factor. Equivalent
      to applying interior padding on the kernel. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  """
  @doc type: :convolutional
  defn depthwise_conv(input, weight, bias, opts \\ []) do
    assert_equal_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: 1,
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        channels: :first
      )

    num_groups =
      transform({Nx.shape(input), opts[:channels]}, fn
        {shape, :first} ->
          elem(shape, 1)

        {shape, :last} ->
          elem(shape, tuple_size(shape) - 1)
      end)

    conv(input, weight, bias,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: num_groups,
      channels: opts[:channels]
    )
  end

  @doc """
  Functional implementation of a 2-dimensional separable depthwise
  convolution.

  The 2-d depthwise separable convolution performs 2 depthwise convolutions
  each over 1 spatial dimension of the input.

  ## Parameter Shapes

    * `input` - `{batch_size, input_channels, input_spatial0, ..., input_spatialN}`
    * `k1` - `{output_channels, 1, kernel_spatial0, 1}`
    * `b1` - `{output_channels}` or `{}`
    * `k2` - `{output_channels, 1, 1, kernel_spatial1}`
    * `b2` - `{output_channels}` or `{}`

    `output_channels` must be a multiple of the input channels.

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a list
      who's length matches the number of spatial dimensions in
      the input tensor. Defaults to 1.

    * `:padding` - zero padding on the input. Can be one of
      `:valid`, `:same` or a general padding configuration
      without interior padding for each spatial dimension
      of the input.

    * `:input_dilation` - input dilation factor. Equivalent
      to applying interior padding on the input. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:kernel_dilation` - kernel dilation factor. Equivalent
      to applying interior padding on the kernel. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  ## References

    * [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
  """
  @doc type: :convolutional
  defn separable_conv2d(input, k1, b1, k2, b2, opts \\ []) do
    input
    |> depthwise_conv(k1, b1, opts)
    |> depthwise_conv(k2, b2, opts)
  end

  @doc """
  Functional implementation of a 3-dimensional separable depthwise
  convolution.

  The 3-d depthwise separable convolution performs 3 depthwise convolutions
  each over 1 spatial dimension of the input.

  ## Parameter Shapes

    * `input` - `{batch_size, input_channels, input_spatial0, ..., input_spatialN}`
    * `k1` - `{output_channels, 1, kernel_spatial0, 1, 1}`
    * `b1` - `{output_channels}` or `{}`
    * `k2` - `{output_channels, 1, 1, kernel_spatial1, 1}`
    * `b2` - `{output_channels}` or `{}`
    * `k3` - `{output_channels, 1, 1, 1, 1, kernel_spatial2}`
    * `b3` - `{output_channels}` or `{}`

    `output_channels` must be a multiple of the input channels.

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a list
      who's length matches the number of spatial dimensions in
      the input tensor. Defaults to 1.

    * `:padding` - zero padding on the input. Can be one of
      `:valid`, `:same` or a general padding configuration
      without interior padding for each spatial dimension
      of the input.

    * `:input_dilation` - input dilation factor. Equivalent
      to applying interior padding on the input. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:kernel_dilation` - kernel dilation factor. Equivalent
      to applying interior padding on the kernel. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Defaults to `1` or no dilation.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  ## References

    * [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
  """
  @doc type: :convolutional
  defn separable_conv3d(input, k1, b1, k2, b2, k3, b3, opts \\ []) do
    input
    |> depthwise_conv(k1, b1, opts)
    |> depthwise_conv(k2, b2, opts)
    |> depthwise_conv(k3, b3, opts)
  end

  @doc """
  Functional implementation of a general dimensional max pooling layer.

  Pooling is applied to the spatial dimension of the input tensor.
  Max pooling returns the maximum element in each valid window of
  the input tensor. It is often used after convolutional layers
  to downsample the input even further.

  ## Options

    * `kernel_size` - window size. Rank must match spatial dimension
      of the input tensor. Required.

    * `:strides` - kernel strides. Can be a scalar or a list
      who's length matches the number of spatial dimensions in
      the input tensor. Defaults to size of kernel.

    * `:padding` - zero padding on the input. Can be one of
      `:valid`, `:same` or a general padding configuration
      without interior padding for each spatial dimension
      of the input.

    * `:window_dilations` - kernel dilation factor. Equivalent
      to applying interior padding on the kernel. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Can be scalar or list who's length matches the number of
      spatial dimensions in the input tensor. Defaults to `1` or no
      dilation.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  ## Examples

      iex> t = Nx.tensor([[
      ...> [0.051500000059604645, -0.7042999863624573, -0.32899999618530273],
      ...> [-0.37130001187324524, 1.6191999912261963, -0.11829999834299088],
      ...> [0.7099999785423279, 0.7282999753952026, -0.18639999628067017]]], type: {:f, 32})
      iex> Axon.Layers.max_pool(t, kernel_size: 2)
      #Nx.Tensor<
        f32[1][3][1]
        [
          [
            [0.051500000059604645],
            [1.6191999912261963],
            [0.7282999753952026]
          ]
        ]
      >
  """
  @doc type: :pooling
  defn max_pool(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, strides: nil, padding: :valid, window_dilations: 1, channels: :first]
      )

    window_dimensions =
      transform(
        {Nx.rank(input), opts[:kernel_size], opts[:channels]},
        fn {rank, kernel_size, channels} ->
          Axon.Shape.pool_window_size(kernel_size, rank - 2, channels)
        end
      )

    strides =
      transform(
        {Nx.rank(input), opts[:strides], window_dimensions, opts[:channels]},
        fn
          {_, nil, dims, _} -> Tuple.to_list(dims)
          {_, [_ | _] = strides, _, :first} -> [1, 1 | strides]
          {_, [_ | _] = strides, _, :last} -> [1 | strides] ++ [1]
          {rank, strides, _, :first} -> [1, 1 | List.duplicate(strides, rank - 2)]
          {rank, strides, _, :last} -> [1 | List.duplicate(strides, rank - 2)] ++ [1]
        end
      )

    padding =
      transform(
        opts[:padding],
        fn
          :same ->
            :same

          :valid ->
            :valid

          padding ->
            [{0, 0}, {0, 0} | padding]
        end
      )

    opts = transform(opts, &Keyword.delete(&1, :kernel_size))

    input
    |> Nx.window_max(window_dimensions,
      strides: strides,
      padding: padding,
      window_dilations: opts[:window_dilations]
    )
  end

  @doc """
  A general dimensional functional average pooling layer.

  Pooling is applied to the spatial dimension of the input tensor.
  Average pooling returns the average of all elements in valid
  windows in the input tensor. It is often used after convolutional
  layers to downsample the input even further.

  ## Options

    * `kernel_size` - window size. Rank must match spatial dimension
      of the input tensor. Required.

    * `:strides` - kernel strides. Can be a scalar or a list
      who's length matches the number of spatial dimensions in
      the input tensor. Defaults to 1.

    * `:padding` - zero padding on the input. Can be one of
      `:valid`, `:same` or a general padding configuration
      without interior padding for each spatial dimension
      of the input.

    * `:window_dilations` - kernel dilation factor. Equivalent
      to applying interior padding on the kernel. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Can be scalar or list who's length matches the number of
      spatial dimensions in the input tensor. Defaults to `1` or no
      dilation.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.
  """
  @doc type: :pooling
  defn avg_pool(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, strides: nil, padding: :valid, window_dilations: 1, channels: :first]
      )

    window_dimensions =
      transform(
        {Nx.rank(input), opts[:kernel_size], opts[:channels]},
        fn {rank, kernel_size, channels} ->
          Axon.Shape.pool_window_size(kernel_size, rank - 2, channels)
        end
      )

    strides =
      transform(
        {Nx.rank(input), opts[:strides], window_dimensions, opts[:channels]},
        fn
          {_, nil, dims, _} -> Tuple.to_list(dims)
          {_, [_ | _] = strides, _, :first} -> [1, 1 | strides]
          {_, [_ | _] = strides, _, :last} -> [1 | strides] ++ [1]
          {rank, strides, _, :first} -> [1, 1 | List.duplicate(strides, rank - 2)]
          {rank, strides, _, :last} -> [1 | List.duplicate(strides, rank - 2)] ++ [1]
        end
      )

    padding =
      transform(
        opts[:padding],
        fn
          :same ->
            :same

          :valid ->
            :valid

          padding ->
            [{0, 0}, {0, 0} | padding]
        end
      )

    opts = transform(opts, &Keyword.delete(&1, :kernel_size))

    input
    |> Nx.window_mean(window_dimensions,
      strides: strides,
      padding: padding,
      window_dilations: opts[:window_dilations]
    )
  end

  @doc ~S"""
  Functional implementation of a general dimensional power average
  pooling layer.

  Pooling is applied to the spatial dimension of the input tensor.
  Power average pooling computes the following function on each
  valid window of the input tensor:

  $$f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}$$

  Where $p$ is given by the keyword argument `:norm`. As $p$ approaches
  infinity, it becomes equivalent to max pooling.

  ## Options

    * `:norm` - $p$ from above equation. Defaults to 2.

    * `:kernel_size` - window size. Rank must match spatial dimension
      of the input tensor. Required.

    * `:strides` - kernel strides. Can be a scalar or a list
      who's length matches the number of spatial dimensions in
      the input tensor. Defaults to size of kernel.

    * `:padding` - zero padding on the input. Can be one of
      `:valid`, `:same` or a general padding configuration
      without interior padding for each spatial dimension
      of the input.

    * `:window_dilations` - kernel dilation factor. Equivalent
      to applying interior padding on the kernel. The amount
      of interior padding applied is given by `kernel_dilation - 1`.
      Can be scalar or list who's length matches the number of
      spatial dimensions in the input tensor. Defaults to `1` or no
      dilation.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.

  ## Examples

      iex> t = Nx.tensor([[[0.9450, 0.4684, 1.8146], [1.2663, 0.4354, -0.0781], [-0.4759, 0.3251, 0.8742]]], type: {:f, 32})
      iex> Axon.Layers.lp_pool(t, kernel_size: 2, norm: 2)
      #Nx.Tensor<
        f32[1][3][1]
        [
          [
            [1.0547149181365967],
            [1.3390626907348633],
            [0.5763426423072815]
          ]
        ]
      >
  """
  @doc type: :pooling
  defn lp_pool(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [
          :kernel_size,
          strides: nil,
          padding: :valid,
          window_dilations: 1,
          norm: 2,
          channels: :first
        ]
      )

    window_dimensions =
      transform(
        {Nx.rank(input), opts[:kernel_size], opts[:channels]},
        fn {rank, kernel_size, channels} ->
          Axon.Shape.pool_window_size(kernel_size, rank - 2, channels)
        end
      )

    strides =
      transform(
        {Nx.rank(input), opts[:strides], window_dimensions, opts[:channels]},
        fn
          {_, nil, dims, _} -> Tuple.to_list(dims)
          {_, [_ | _] = strides, _, :first} -> [1, 1 | strides]
          {_, [_ | _] = strides, _, :last} -> [1 | strides] ++ [1]
          {rank, strides, _, :first} -> [1, 1 | List.duplicate(strides, rank - 2)]
          {rank, strides, _, :last} -> [1 | List.duplicate(strides, rank - 2)] ++ [1]
        end
      )

    padding =
      transform(
        opts[:padding],
        fn
          :same ->
            :same

          :valid ->
            :valid

          padding ->
            [{0, 0}, {0, 0} | padding]
        end
      )

    norm = opts[:norm]

    input
    |> Nx.power(norm)
    |> Nx.window_sum(window_dimensions,
      strides: strides,
      padding: padding,
      window_dilations: opts[:window_dilations]
    )
    |> Nx.power(Nx.divide(Nx.tensor(1, type: Nx.type(input)), norm))
  end

  @doc """
  Functional implementation of general dimensional adaptive average
  pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size. It
  will then perform average pooling using the calculated window
  size and strides.

  Adaptive pooling can be useful when working on multiple inputs with
  different spatial input shapes. You can guarantee the output of
  an adaptive pooling operation is always the same size regardless
  of input shape.

  ## Options

    * `:output_size` - spatial output size. Must be a tuple with
      size equal to the spatial dimensions in the input tensor.
      Required.

    * `:channels ` - channel configuration. One of `:first` or `:last`.
      Defaults to `:first`.
  """
  @doc type: :pooling
  defn adaptive_avg_pool(input, opts \\ []) do
    opts = keyword!(opts, [:output_size, channels: :first])

    window_strides =
      transform(
        {Nx.shape(input), Nx.rank(input), opts[:output_size], opts[:channels]},
        fn {shape, rank, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_strides(shape, output_size, rank - 2, channels)
        end
      )

    window_dimensions =
      transform(
        {Nx.shape(input), Nx.rank(input), window_strides, opts[:output_size], opts[:channels]},
        fn {shape, rank, strides, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_size(shape, strides, output_size, rank - 2, channels)
        end
      )

    input
    |> Nx.window_mean(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of general dimensional adaptive max
  pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size. It
  will then perform max pooling using the calculated window
  size and strides.

  Adaptive pooling can be useful when working on multiple inputs with
  different spatial input shapes. You can guarantee the output of
  an adaptive pooling operation is always the same size regardless
  of input shape.

  ## Options

    * `:output_size` - spatial output size. Must be a tuple with
      size equal to the spatial dimensions in the input tensor.
      Required.
  """
  @doc type: :pooling
  defn adaptive_max_pool(input, opts \\ []) do
    opts = keyword!(opts, [:output_size, channels: :first])

    window_strides =
      transform(
        {Nx.shape(input), Nx.rank(input), opts[:output_size], opts[:channels]},
        fn {shape, rank, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_strides(shape, output_size, rank - 2, channels)
        end
      )

    window_dimensions =
      transform(
        {Nx.shape(input), Nx.rank(input), window_strides, opts[:output_size], opts[:channels]},
        fn {shape, rank, strides, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_size(shape, strides, output_size, rank - 2, channels)
        end
      )

    input
    |> Nx.window_max(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of general dimensional adaptive power
  average pooling.

  Computes:

    $$f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}$$

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size. It
  will then perform max pooling using the calculated window
  size and strides.

  Adaptive pooling can be useful when working on multiple inputs with
  different spatial input shapes. You can guarantee the output of
  an adaptive pooling operation is always the same size regardless
  of input shape.

  ## Options

    * `:norm` - $p$ from above equation. Defaults to 2.

    * `:output_size` - spatial output size. Must be a tuple with
      size equal to the spatial dimensions in the input tensor.
      Required.
  """
  @doc type: :pooling
  defn adaptive_lp_pool(input, opts \\ []) do
    opts = keyword!(opts, [:output_size, norm: 2, channels: :first])

    norm = opts[:norm]

    window_strides =
      transform(
        {Nx.shape(input), Nx.rank(input), opts[:output_size], opts[:channels]},
        fn {shape, rank, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_strides(shape, output_size, rank - 2, channels)
        end
      )

    window_dimensions =
      transform(
        {Nx.shape(input), Nx.rank(input), window_strides, opts[:output_size], opts[:channels]},
        fn {shape, rank, strides, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_size(shape, strides, output_size, rank - 2, channels)
        end
      )

    input
    |> Nx.power(norm)
    |> Nx.window_sum(window_dimensions, padding: :valid, strides: window_strides)
    |> Nx.power(Nx.divide(Nx.tensor(1, type: Nx.type(input)), norm))
  end

  ## Normalization

  @doc ~S"""
  Functional implementation of batch normalization.

  Normalizes the input by calculating mean and variance of the
  input tensor along every dimension but the given `:channel_index`,
  and then scaling according to:

  $$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$

  `gamma` and `beta` are often trainable parameters. If `training?` is
  true, this method will compute a new mean and variance, and return
  the updated `ra_mean` and `ra_var`. Otherwise, it will just compute
  batch norm from the given ra_mean and ra_var.

  ## Options

    * `:epsilon` - numerical stability term. $epsilon$ in the above
      formulation.

    * `:channel_index` - channel index used to determine reduction
      axes for mean and variance calculation.

    * `:momentum` - momentum to use for EMA update.

    * `:training?` - if true, uses training mode batch norm. Defaults to false.

  ## References

    * [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """
  @doc type: :normalization
  defn batch_norm(input, gamma, bias, ra_mean, ra_var, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-5, channel_index: 1, momentum: 0.1, training?: false)

    axes =
      transform({Nx.axes(input), opts[:channel_index]}, fn {axes, channel} ->
        Axon.Shape.batch_norm_axes(axes, channel)
      end)

    channel_index = opts[:channel_index]

    num_channels =
      transform({input, channel_index}, fn {inp, channel_idx} ->
        elem(Nx.shape(inp), channel_idx)
      end)

    {gamma, bias, ra_mean, ra_var} =
      transform(
        {gamma, bias, ra_mean, ra_var, Nx.rank(input), num_channels, channel_index},
        fn {g, b, m, v, rank, num_channels, channel_idx} ->
          new_shape =
            1
            |> List.duplicate(rank)
            |> List.to_tuple()
            |> put_elem(channel_idx, num_channels)

          {Nx.reshape(g, new_shape), Nx.reshape(b, new_shape), Nx.reshape(m, new_shape),
           Nx.reshape(v, new_shape)}
        end
      )

    transform(
      {input, gamma, bias, ra_mean, ra_var, axes, opts[:epsilon], opts[:momentum],
       opts[:training?]},
      fn
        {x, g, b, m, v, axes, eps, alpha, true} ->
          {new_mean, new_var} = mean_and_variance(x, axes: axes)
          out = normalize(x, new_mean, new_var, g, b, epsilon: eps)
          {out, update_ema(new_mean, m, alpha), update_ema(new_var, v, alpha)}

        {x, g, b, m, v, _, eps, _, _} ->
          normalize(x, m, v, g, b, epsilon: eps)
      end
    )
  end

  defnp update_ema(obs, old, momentum) do
    Nx.squeeze(momentum * old + (1 - momentum) * obs)
  end

  @doc ~S"""
  Functional implementation of layer normalization.

  Normalizes the input by calculating mean and variance of the
  input tensor along the given feature dimension `:channel_index`.

  $$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$

  `gamma` and `beta` are often trainable parameters. This method does
  not maintain an EMA of mean and variance.

  ## Options

    * `:epsilon` - numerical stability term. $epsilon$ in the above
      formulation.

    * `:channel_index` - channel index used to determine reduction
      axes for mean and variance calculation.
  """
  @doc type: :normalization
  defn layer_norm(input, gamma, bias, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-5, channel_index: 1)
    axes = opts[:channel_index]

    channel_index = opts[:channel_index]

    num_channels =
      transform({input, channel_index}, fn {inp, channel_idx} ->
        elem(Nx.shape(inp), channel_idx)
      end)

    {gamma, bias} =
      transform({gamma, bias, Nx.rank(input), num_channels, channel_index}, fn {g, b, rank,
                                                                                num_channels,
                                                                                channel_idx} ->
        new_shape =
          1
          |> List.duplicate(rank)
          |> List.to_tuple()
          |> put_elem(channel_idx, num_channels)

        {Nx.reshape(g, new_shape), Nx.reshape(b, new_shape)}
      end)

    {mean, var} = mean_and_variance(input, axes: [axes])
    normalize(input, mean, var, gamma, bias, epsilon: opts[:epsilon])
  end

  @doc """
  Functional implementation of group normalization.

  Normalizes the input by reshaping input into groups of given
  `:group_size` and then calculating the mean and variance along
  every dimension but the input batch dimension.

  $$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$

  `gamma` and `beta` are often trainable parameters. This method does
  not maintain an EMA of mean and variance.

  ## Options

    * `:group_size` - channel group size. Size of each group to split
      input channels into.

    * `:epsilon` - numerical stability term. $epsilon$ in the above
      formulation.

    * `:channel_index` - channel index used to determine reduction
      axes and group shape for mean and variance calculation.

  ## References

    * [Group Normalization](https://arxiv.org/abs/1803.08494v3)
  """
  @doc type: :normalization
  defn group_norm(input, gamma, bias, opts \\ []) do
    opts = keyword!(opts, [:group_size, epsilon: 1.0e-5, channel_index: 1])

    group_shape =
      transform({Nx.shape(input), opts[:group_size], opts[:channel_index]}, fn {shape, groups,
                                                                                channel} ->
        Axon.Shape.group_norm_shape(shape, groups, channel)
      end)

    channel_index = opts[:channel_index]

    num_channels =
      transform({input, channel_index}, fn {inp, channel_idx} ->
        elem(Nx.shape(inp), channel_idx)
      end)

    {gamma, bias} =
      transform({gamma, bias, Nx.rank(input), num_channels, channel_index}, fn {g, b, rank,
                                                                                num_channels,
                                                                                channel_idx} ->
        new_shape =
          1
          |> List.duplicate(rank)
          |> List.to_tuple()
          |> put_elem(channel_idx, num_channels)

        {Nx.reshape(g, new_shape), Nx.reshape(b, new_shape)}
      end)

    x = Nx.reshape(input, group_shape)
    axes = transform(Nx.rank(x), &Axon.Shape.group_norm_axes/1)
    {mean, var} = mean_and_variance(x, axes: axes)
    normalize(Nx.reshape(x, input), mean, var, gamma, bias, epsilon: opts[:epsilon])
  end

  @doc """
  Functional implementation of instance normalization.

  Normalizes the input by calculating mean and variance of the
  input tensor along the spatial dimensions of the input.

  $$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$

  `gamma` and `beta` are often trainable parameters. If `training?` is
  true, this method will compute a new mean and variance, and return
  the updated `ra_mean` and `ra_var`. Otherwise, it will just compute
  batch norm from the given ra_mean and ra_var.

  ## Options

    * `:epsilon` - numerical stability term. $epsilon$ in the above
      formulation.

    * `:channel_index` - channel index used to determine reduction
      axes for mean and variance calculation.

    * `:momentum` - momentum to use for EMA update.

    * `:training?` - if true, uses training mode batch norm. Defaults to false.

  ## References

    * [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022v3)
  """
  @doc type: :normalization
  defn instance_norm(input, gamma, bias, ra_mean, ra_var, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-5, channel_index: 1, momentum: 0.1, training?: false)

    axes =
      transform({Nx.axes(input), opts[:channel_index]}, fn {axes, channel} ->
        Axon.Shape.instance_norm_axes(axes, channel)
      end)

    channel_index = opts[:channel_index]

    num_channels =
      transform({input, channel_index}, fn {inp, channel_idx} ->
        elem(Nx.shape(inp), channel_idx)
      end)

    {gamma, bias, ra_mean, ra_var} =
      transform(
        {gamma, bias, ra_mean, ra_var, Nx.rank(input), num_channels, channel_index},
        fn {g, b, m, v, rank, num_channels, channel_idx} ->
          new_shape =
            1
            |> List.duplicate(rank)
            |> List.to_tuple()
            |> put_elem(channel_idx, num_channels)

          {Nx.reshape(g, new_shape), Nx.reshape(b, new_shape), Nx.reshape(m, new_shape),
           Nx.reshape(v, new_shape)}
        end
      )

    transform(
      {input, gamma, bias, ra_mean, ra_var, axes, opts[:epsilon], opts[:momentum],
       opts[:training?]},
      fn
        {x, g, b, m, v, axes, eps, alpha, true} ->
          {new_mean, new_var} = mean_and_variance(x, axes: axes)
          out = normalize(x, new_mean, new_var, g, b, epsilon: eps)
          {out, update_ema(new_mean, m, alpha), update_ema(new_var, v, alpha)}

        {x, g, b, m, v, _, eps, _, _} ->
          normalize(x, m, v, g, b, epsilon: eps)
      end
    )
  end

  ## Stochastic

  # TODO: Manage the state of these RNGs

  @doc ~S"""
  Functional implementation of a dropout layer.

  Applies a mask to some elements of the input tensor with probability
  `rate` and scales the input tensor by a factor of $\frac{1}{1 - rate}$.

  Dropout is a form of regularization that helps prevent overfitting
  by preventing models from becoming too reliant on certain connections.
  Dropout can somewhat be thought of as learning an ensemble of models
  with random connections masked.

  ## Options

    * `:rate` - dropout rate. Used to determine probability a connection
      will be dropped. Required.

    # `:noise_shape` - input noise shape. Shape of `mask` which can be useful
      for broadcasting `mask` across feature channels or other dimensions.
      Defaults to shape of input tensor.

  ## References

    * [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
  """
  @doc type: :dropout
  defn dropout(input, opts \\ []) do
    opts = keyword!(opts, [:rate, noise_shape: Nx.shape(input)])
    keep_prob = Nx.tensor(1, type: Nx.type(input)) - Nx.tensor(opts[:rate], type: Nx.type(input))
    mask = Nx.less(Nx.random_uniform(opts[:noise_shape], type: Nx.type(input)), keep_prob)

    mask =
      transform(
        {mask, Nx.shape(input)},
        fn {mask, input_shape} ->
          if Elixir.Kernel.==(Nx.shape(mask), input_shape),
            do: mask,
            else: Nx.broadcast(mask, input_shape)
        end
      )

    Nx.select(mask, input / keep_prob, Nx.tensor(0, type: Nx.type(input)))
  end

  @doc """
  Functional implementation of an n-dimensional spatial
  dropout layer.

  Applies a mask to entire feature maps instead of individual
  elements. This is done by calculating a mask shape equal to
  the spatial dimensions of the input tensor with 1 channel,
  and then broadcasting the mask across the feature dimension
  of the input tensor.

  ## Options

    * `:rate` - dropout rate. Used to determine probability a connection
      will be dropped. Required.

    # `:noise_shape` - input noise shape. Shape of `mask` which can be useful
      for broadcasting `mask` across feature channels or other dimensions.
      Defaults to shape of input tensor.

  ## References

    * [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)
  """
  @doc type: :dropout
  defn spatial_dropout(input, opts \\ []) do
    opts = keyword!(opts, rate: 0.5)
    noise_shape = transform(Nx.shape(input), &Axon.Shape.spatial_dropout_noise_shape/1)
    dropout(input, rate: opts[:rate], noise_shape: noise_shape)
  end

  @doc """
  Functional implementation of an alpha dropout layer.

  Alpha dropout is a type of dropout that forces the input
  to have zero mean and unit standard deviation. Randomly
  masks some elements and scales to enforce self-normalization.

  ## Options

    * `:rate` - dropout rate. Used to determine probability a connection
      will be dropped. Required.

    # `:noise_shape` - input noise shape. Shape of `mask` which can be useful
      for broadcasting `mask` across feature channels or other dimensions.
      Defaults to shape of input tensor.

  ## References

    * [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  """
  @doc type: :dropout
  defn alpha_dropout(input, opts \\ []) do
    opts = keyword!(opts, rate: 0.5)
    rate = opts[:rate]

    alpha = Nx.tensor(1.6732632423543772848170429916717, type: Nx.type(input))
    scale = Nx.tensor(1.0507009873554804934193349852946, type: Nx.type(input))
    alpha_p = -alpha * scale
    keep_prob = Nx.tensor(1, type: Nx.type(input)) - rate

    mask = Nx.less(Nx.random_uniform(Nx.shape(input), type: Nx.type(input)), keep_prob)

    a = Nx.rsqrt(keep_prob * Nx.power(Nx.tensor(1, type: Nx.type(input)) * alpha_p, 2))
    b = -a * alpha_p * rate

    x = Nx.select(mask, input, alpha_p)
    a * x + b
  end

  @doc """
  Functional implementation of a feature alpha dropout layer.

  Feature alpha dropout applies dropout in the same manner as
  spatial dropout; however, it also enforces self-normalization
  by masking inputs with the SELU activation function and scaling
  unmasked inputs.

  ## Options

    * `:rate` - dropout rate. Used to determine probability a connection
      will be dropped. Required.

    # `:noise_shape` - input noise shape. Shape of `mask` which can be useful
      for broadcasting `mask` across feature channels or other dimensions.
      Defaults to shape of input tensor.
  """
  @doc type: :dropout
  defn feature_alpha_dropout(input, opts \\ []) do
    opts = keyword!(opts, rate: 0.5)
    noise_shape = transform(Nx.shape(input), &Axon.Shape.spatial_dropout_noise_shape/1)
    keep_prob = 1 - opts[:rate]
    mask = Nx.less(Nx.random_uniform(noise_shape, type: Nx.type(input)), keep_prob)

    mask =
      transform(
        {mask, Nx.shape(input)},
        fn {mask, input_shape} ->
          if Elixir.Kernel.==(Nx.shape(mask), input_shape),
            do: mask,
            else: Nx.broadcast(mask, input_shape)
        end
      )

    Nx.select(mask, input / keep_prob, Nx.negate(Axon.Activations.selu(input)))
  end

  ## Global Pooling

  @doc """
  Functional implementation of global average pooling which averages across
  the spatial dimensions of the input such that the only remaining dimensions
  are the batch and feature dimensions.

  Assumes data is configured in a channels-first like format.

  ## Parameter Shapes

    * `input` - {batch_size, features, s1, ..., sN}

  ## Options

    * `:keep_axes` - option to keep reduced axes with size 1 for each reduced
      dimensions. Defaults to `false`

  ## Examples

      iex> Axon.Layers.global_avg_pool(Nx.iota({3, 2, 3}, type: {:f, 32}))
      #Nx.Tensor<
        f32[3][2]
        [
          [1.0, 4.0],
          [7.0, 10.0],
          [13.0, 16.0]
        ]
      >

      iex> Axon.Layers.global_avg_pool(Nx.iota({1, 3, 2, 2}, type: {:f, 32}), keep_axes: true)
      #Nx.Tensor<
        f32[1][3][1][1]
        [
          [
            [
              [1.5]
            ],
            [
              [5.5]
            ],
            [
              [9.5]
            ]
          ]
        ]
      >
  """
  defn global_avg_pool(input, opts \\ []) do
    opts = keyword!(opts, channels: :first, keep_axes: false)

    all_but_batch_and_feature =
      transform({Nx.rank(input), opts[:channels]}, fn
        {rank, :first} ->
          for i <- 2..(rank - 1), do: i

        {rank, :last} ->
          for i <- 1..(rank - 2), do: i
      end)

    Nx.mean(input, axes: all_but_batch_and_feature, keep_axes: opts[:keep_axes])
  end

  @doc """
  Functional implementation of global max pooling which computes maximums across
  the spatial dimensions of the input such that the only remaning dimensions are
  the batch and feature dimensions.

  Assumes data is configured in a channels-first like format.

  ## Parameter Shapes

    * `input` - {batch_size, s1, ..., sN, features}

  ## Options

    * `:keep_axes` - option to keep reduced axes with size 1 for each reduced
      dimensions. Defaults to `false`

  ## Examples

      iex> Axon.Layers.global_max_pool(Nx.iota({3, 2, 3}, type: {:f, 32}))
      #Nx.Tensor<
        f32[3][2]
        [
          [2.0, 5.0],
          [8.0, 11.0],
          [14.0, 17.0]
        ]
      >

      iex> Axon.Layers.global_max_pool(Nx.iota({1, 3, 2, 2}, type: {:f, 32}), keep_axes: true)
      #Nx.Tensor<
        f32[1][3][1][1]
        [
          [
            [
              [3.0]
            ],
            [
              [7.0]
            ],
            [
              [11.0]
            ]
          ]
        ]
      >
  """
  defn global_max_pool(x, opts \\ []) do
    opts = keyword!(opts, keep_axes: false, channels: :first)

    all_but_batch_and_feature =
      transform({Nx.rank(x), opts[:channels]}, fn
        {rank, :first} ->
          for i <- 2..(rank - 1), do: i

        {rank, :last} ->
          for i <- 1..(rank - 2), do: i
      end)

    Nx.reduce_max(x, axes: all_but_batch_and_feature, keep_axes: opts[:keep_axes])
  end

  @doc """
  Functional implementation of global LP pooling which computes the following
  function across spatial dimensions of the input:

    $$f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}$$

  Where $p$ is given by the keyword argument `:norm`. As $p$ approaches
  infinity, it becomes equivalent to max pooling.

  Assumes data is configured in a channels-first like format.

  ## Parameter Shapes

    * `input` - {batch_size, s1, ..., sN, features}

  ## Options

    * `:keep_axes` - option to keep reduced axes with size 1 for each reduced
      dimensions. Defaults to `false`
    * `:norm` - $p$ in above function. Defaults to 2

  ## Examples

      iex> Axon.Layers.global_lp_pool(Nx.iota({3, 2, 3}, type: {:f, 32}), norm: 1)
      #Nx.Tensor<
        f32[3][2]
        [
          [3.0, 12.0],
          [21.0, 30.0],
          [39.0, 48.0]
        ]
      >

      iex> Axon.Layers.global_lp_pool(Nx.iota({1, 3, 2, 2}, type: {:f, 16}), keep_axes: true)
      #Nx.Tensor<
        f16[1][3][1][1]
        [
          [
            [
              [3.7421875]
            ],
            [
              [11.2265625]
            ],
            [
              [19.125]
            ]
          ]
        ]
      >
  """
  defn global_lp_pool(x, opts \\ []) do
    opts = keyword!(opts, norm: 2, keep_axes: false, channels: :first)

    norm = opts[:norm]

    all_but_batch_and_feature =
      transform({Nx.rank(x), opts[:channels]}, fn
        {rank, :first} ->
          for i <- 2..(rank - 1), do: i

        {rank, :last} ->
          for i <- 1..(rank - 2), do: i
      end)

    x
    |> Nx.power(norm)
    |> Nx.sum(axes: all_but_batch_and_feature, keep_axes: opts[:keep_axes])
    |> Nx.power(Nx.divide(Nx.tensor(1, type: Nx.type(x)), norm))
  end

  ## Sparse

  @doc """
  Computes embedding by treating weight matrix as a lookup table
  for discrete tokens.

  `input` is a vector of discrete values, typically representing tokens
  (e.g. words, characters, etc.) from a vocabulary. `weights` is a weight
  matrix of shape `{vocab_size, embedding_size}` from which the dense
  embeddings will be drawn.

  ## Parameter Shapes

    * `input` - `{batch_size, seq_len}`
    * `weights` - `{vocab_size, embedding_size}`

  ## Examples

      iex> input = Nx.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
      iex> weights = Nx.tensor([
      ...>  [0.46299999952316284, 0.5562999844551086, 0.18170000612735748],
      ...>  [0.9801999926567078, 0.09780000150203705, 0.5333999991416931],
      ...>  [0.6980000138282776, 0.9240999817848206, 0.23479999601840973],
      ...>  [0.31929999589920044, 0.42250001430511475, 0.7865999937057495],
      ...>  [0.5519000291824341, 0.5662999749183655, 0.20559999346733093],
      ...>  [0.1898999959230423, 0.9311000108718872, 0.8356000185012817],
      ...>  [0.6383000016212463, 0.8794000148773193, 0.5282999873161316],
      ...>  [0.9523000121116638, 0.7597000002861023, 0.08250000327825546],
      ...>  [0.6622999906539917, 0.02329999953508377, 0.8205999732017517],
      ...>  [0.9855999946594238, 0.36419999599456787, 0.5372999906539917]
      ...> ])
      iex> Axon.Layers.embedding(input, weights)
      #Nx.Tensor<
        f32[2][4][3]
        [
          [
            [0.9801999926567078, 0.09780000150203705, 0.5333999991416931],
            [0.6980000138282776, 0.9240999817848206, 0.23479999601840973],
            [0.5519000291824341, 0.5662999749183655, 0.20559999346733093],
            [0.1898999959230423, 0.9311000108718872, 0.8356000185012817]
          ],
          [
            [0.5519000291824341, 0.5662999749183655, 0.20559999346733093],
            [0.31929999589920044, 0.42250001430511475, 0.7865999937057495],
            [0.6980000138282776, 0.9240999817848206, 0.23479999601840973],
            [0.9855999946594238, 0.36419999599456787, 0.5372999906539917]
          ]
        ]
      >
  """
  defn embedding(input, weights) do
    Nx.take(weights, input, axis: 0)
  end

  ## Shape

  @doc """
  Flattens input to shape of `{batch, units}` by folding outer
  dimensions.

  ## Examples

      iex> Axon.Layers.flatten(Nx.iota({1, 2, 2}, type: {:f, 32}))
      #Nx.Tensor<
        f32[1][4]
        [
          [0.0, 1.0, 2.0, 3.0]
        ]
      >
  """
  defn flatten(x) do
    new_shape = transform(Nx.shape(x), &Axon.Shape.flatten/1)
    Nx.reshape(x, new_shape)
  end

  @doc false
  # Internal helper for constructing conditional layers without
  # needing to use the if-macros in Axon.Compiler
  defn cond(cond_expr, on_true_expr, on_false_expr) do
    if cond_expr do
      on_true_expr
    else
      on_false_expr
    end
  end

  @doc """
  Resizes a batch of tensors to the given shape using one of a
  number of sampling methods.

  Requires input option `:shape` which should be a tuple specifying
  the resized spatial dimensions of the input tensor. Input tensor
  must be at least rank 3, with fixed `batch` and `channel` dimensions.
  Resizing will upsample or downsample using the given resize method.

  Supported reize methods are `:nearest`.

  ## Examples

      iex> img = Nx.iota({1, 1, 3, 3}, type: {:f, 32})
      iex> Axon.Layers.resize(img, shape: {4, 4})
      #Nx.Tensor<
        f32[1][1][4][4]
        [
          [
            [
              [0.0, 1.0, 1.0, 2.0],
              [3.0, 4.0, 4.0, 5.0],
              [3.0, 4.0, 4.0, 5.0],
              [6.0, 7.0, 7.0, 8.0]
            ]
          ]
        ]
      >

      iex> img = Nx.iota({1, 1, 3}, type: {:f, 32})
      iex> Axon.Layers.resize(img, shape: {2})
      #Nx.Tensor<
        f32[1][1][2]
        [
          [
            [0.0, 2.0]
          ]
        ]
      >

      iex> img = Nx.iota({1, 2, 2, 2, 1}, type: {:f, 32})
      iex> Axon.Layers.resize(img, shape: {1, 3, 2})
      #Nx.Tensor<
        f32[1][2][1][3][2]
        [
          [
            [
              [
                [2.0, 2.0],
                [3.0, 3.0],
                [3.0, 3.0]
              ]
            ],
            [
              [
                [6.0, 6.0],
                [7.0, 7.0],
                [7.0, 7.0]
              ]
            ]
          ]
        ]
      >

  ### Error cases

      iex> img = Nx.iota({1, 1, 3, 3}, type: {:f, 32})
      iex> Axon.Layers.resize(img, shape: {4, 4}, method: :foo)
      ** (ArgumentError) invalid resize method :foo, resize method must be one of :nearest
  """
  defn resize(input, opts \\ []) do
    opts = keyword!(opts, [:shape, method: :nearest, channels: :first])
    output_shape = opts[:shape]

    # Input must be at least rank 3
    transform(Nx.rank(input), fn rank ->
      unless rank > 3 do
        raise ArgumentError, "input rank must be at least 3, got #{inspect(rank)}"
      end
    end)

    spatial_dimensions =
      transform({Nx.rank(input), opts[:channels]}, fn
        {rank, :first} ->
          Enum.to_list(2..(rank - 1))

        {rank, :last} ->
          Enum.to_list(1..(rank - 2))
      end)

    output_shape =
      transform({input, spatial_dimensions, output_shape}, fn {input, spatial_dimensions,
                                                               output_shape} ->
        unless Nx.rank(output_shape) == Nx.rank(input) - 2 do
          raise ArgumentError,
                "invalid output shape #{inspect(output_shape)}, expected output" <>
                  " output shape to have same rank as spatial dimensions of" <>
                  " the input tensor"
        end

        for {d, i} <- Enum.with_index(spatial_dimensions), reduce: Nx.shape(input) do
          shape ->
            put_elem(shape, d, elem(output_shape, i))
        end
      end)

    transform({input, output_shape, spatial_dimensions, opts[:method]}, fn
      {img, shape, spatial_dimensions, :nearest} ->
        resize_nearest(img, shape, spatial_dimensions)

      {_, _, _, method} ->
        raise ArgumentError,
              "invalid resize method #{inspect(method)}, resize method" <>
                " must be one of :nearest"
    end)
  end

  defnp resize_nearest(input, output_shape, spatial_dimensions) do
    transform({input, output_shape, spatial_dimensions}, fn {input, output_shape,
                                                             spatial_dimensions} ->
      ones = List.duplicate(1, Nx.rank(input)) |> List.to_tuple()

      for d <- spatial_dimensions, reduce: input do
        input ->
          input_shape = Nx.shape(input)
          input_size = elem(input_shape, d)
          output_size = elem(output_shape, d)
          offset = (Nx.iota({output_size}) + 0.5) * input_size / output_size
          offset = offset |> Nx.floor() |> Nx.as_type({:s, 32})

          offset =
            offset
            |> Nx.reshape(put_elem(ones, d, output_size))
            |> Nx.broadcast(put_elem(input_shape, d, output_size))

          Nx.take_along_axis(input, offset, axis: d)
      end
    end)
  end
end
