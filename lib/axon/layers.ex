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

  where `w1` and `w2` are layer *kernels*, and `b1` and `b2` are layer
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
  the input using the given kernel matrix and bias
  to compute:

      Nx.dot(input, kernel) + bias

  Typically, both `kernel` and `bias` are learnable
  parameters trained using gradient-based optimization.

  ## Parameter Shapes

    * `input` - `{batch_size, * input_features}`
    * `kernel` - `{input_features, output_features}`
    * `bias` - `{}` or `{output_features}`

  ## Output Shape

    `{batch_size, *, output_features}`

  ## Examples

      iex> input = Nx.tensor([[1.0, 0.5, 1.0, 0.5], [0.0, 0.0, 0.0, 0.0]], type: {:f, 32})
      iex> kernel = Nx.tensor([[0.2], [0.3], [0.5], [0.8]], type: {:f, 32})
      iex> bias = Nx.tensor([1.0], type: {:f, 32})
      iex> Axon.Layers.dense(input, kernel, bias)
      #Nx.Tensor<
        f32[2][1]
        [
          [2.25],
          [1.0]
        ]
      >
  """
  @doc type: :linear
  deftransform dense(input, kernel, bias \\ 0, opts \\ []) do
    {bias, opts} =
      case bias do
        %Nx.Tensor{} = bias ->
          {bias, opts}

        bias when is_number(bias) ->
          {bias, opts}

        opts when is_list(opts) ->
          {Nx.tensor(0), opts}

        other ->
          raise ArgumentError, "invalid bias, expected a tensor, got #{inspect(other)}"
      end

    dense_impl(input, kernel, bias, opts)
  end

  defnp dense_impl(input, kernel, bias, _opts \\ []) do
    assert_min_rank!("Axon.Layers.dense", "input", input, 2)

    input
    |> Nx.dot([Nx.rank(input) - 1], kernel, [0])
    |> Nx.add(bias)
  end

  @doc ~S"""
  Functional implementation of a bilinear layer.

  Bilinear transformation of the input such that:

  $$y = x_1^{T}Ax_2 + b$$

  ## Parameter Shapes

    * `input1` - `{batch_size, ..., input1_features}`
    * `input2` - `{batch_size, ..., input2_features}`
    * `kernel` - `{out_features, input1_features, input2_features}`

  ## Output Shape

    `{batch_size, ..., output_features}`

  ## Examples

      iex> inp1 = Nx.iota({3, 2}, type: {:f, 32})
      iex> inp2 = Nx.iota({3, 4}, type: {:f, 32})
      iex> kernel = Nx.iota({1, 2, 4}, type: {:f, 32})
      iex> bias = Nx.tensor(1.0)
      iex> Axon.Layers.bilinear(inp1, inp2, kernel, bias)
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
  deftransform bilinear(input1, input2, kernel, bias \\ 0, opts \\ []) do
    {bias, opts} =
      case bias do
        %Nx.Tensor{} = bias ->
          {bias, opts}

        bias when is_number(bias) ->
          {bias, opts}

        opts when is_list(opts) ->
          {Nx.tensor(0), opts}

        other ->
          raise ArgumentError, "invalid bias, expected a tensor, got #{inspect(other)}"
      end

    bilinear_impl(input1, input2, kernel, bias, opts)
  end

  defnp bilinear_impl(input1, input2, kernel, bias, _opts \\ []) do
    assert_min_rank!("Axon.Layers.bilinear", "input1", input1, 2)
    assert_min_rank!("Axon.Layers.bilinear", "input2", input2, 2)
    assert_equal_rank!("Axon.Layers.bilinear", "input1", input1, "input2", input2)
    assert_rank!("Axon.Layers.bilinear", "kernel", kernel, 3)

    inp1_axes = transform(Nx.rank(input1), fn rank -> [rank - 1] end)
    inp2_axes = transform(Nx.rank(input2), fn rank -> [rank - 1] end)

    input1
    |> Nx.dot(inp1_axes, [], kernel, [1], [])
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
    * `kernel` - `{output_channels, input_channels, kernel_spatial0, ..., kernel_spatialN}`
    * `bias` - `{}` or `{output_channels}`

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
      iex> kernel = Nx.tensor([[[-1.5475, 1.2425]], [[0.1871, 0.5458]], [[-0.4488,  0.8879]]], type: {:f, 32})
      iex> bias = Nx.tensor([0.7791, 0.1676, 1.5971], type: {:f, 32})
      iex> Axon.Layers.conv(input, kernel, bias, channels: :first)
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
      iex> kernel = Nx.tensor([
      ...>  [[[0.7514, 0.7356], [1.3909,  0.6800]]],
      ...>  [[[-0.3450,  0.4551], [-0.6275, -0.9875]]],
      ...>  [[[1.8587, 0.4722], [0.6058, -1.0301]]]
      ...> ], type: {:f, 32})
      iex> bias = Nx.tensor([1.9564, 0.2822, -0.5385], type: {:f, 32})
      iex> Axon.Layers.conv(input, kernel, bias, channels: :first)
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
      iex> kernel = Nx.tensor([
      ...>  [[[[ 0.7390], [-0.0927]], [[-0.8675], [-0.9209]]]],
      ...>  [[[[-0.6638], [0.4341]], [[0.6368], [1.1846]]]]
      ...> ], type: {:f, 32})
      iex> bias = Nx.tensor([-0.4101,  0.1776], type: {:f, 32})
      iex> Axon.Layers.conv(input, kernel, bias, channels: :first)
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
  deftransform conv(input, kernel, bias \\ 0, opts \\ []) do
    {bias, opts} =
      case bias do
        %Nx.Tensor{} = bias ->
          {bias, opts}

        bias when is_number(bias) ->
          {bias, opts}

        opts when is_list(opts) ->
          {Nx.tensor(0), opts}

        other ->
          raise ArgumentError, "invalid bias, expected a tensor, got #{inspect(other)}"
      end

    conv_impl(input, kernel, bias, opts)
  end

  defnp conv_impl(input, kernel, bias, opts \\ []) do
    assert_min_rank!("Axon.Layers.conv", "input", input, 3)
    assert_equal_rank!("Axon.Layers.conv", "input", input, "kernel", kernel)

    opts =
      keyword!(opts,
        strides: 1,
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        feature_group_size: 1,
        batch_group_size: 1,
        channels: :last,
        mode: :inference
      )

    bias_reshape =
      transform(
        {Nx.shape(bias), Nx.rank(input) - 2, opts[:channels]},
        fn {bias_shape, rank, channels} ->
          Axon.Shape.conv_bias_reshape(bias_shape, rank, channels)
        end
      )

    {permutations, kernel_permutation} =
      transform({Nx.rank(input), opts[:channels]}, fn
        {rank, :first} ->
          perm = Enum.to_list(0..(rank - 1))
          {perm, perm}

        {rank, :last} ->
          spatial = Enum.to_list(1..(rank - 2)//1)
          perm = [0, rank - 1 | spatial]
          kernel_perm = [rank - 1, rank - 2] ++ Enum.to_list(0..(rank - 3)//1)
          {perm, kernel_perm}

        {_rank, invalid} ->
          raise ArgumentError, "invalid channel configuration, #{inspect(invalid)}"
      end)

    input
    |> Nx.conv(kernel,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: opts[:feature_group_size],
      batch_group_size: opts[:batch_group_size],
      input_permutation: permutations,
      kernel_permutation: kernel_permutation,
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
      iex> Axon.Layers.conv_transpose(input, kernel, bias, channels: :first)
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

    * [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
    * [Deconvolutional Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """
  @doc type: :convolutional
  deftransform conv_transpose(input, kernel, bias \\ 0, opts \\ []) do
    {bias, opts} =
      case bias do
        %Nx.Tensor{} = bias ->
          {bias, opts}

        bias when is_number(bias) ->
          {bias, opts}

        opts when is_list(opts) ->
          {Nx.tensor(0), opts}

        other ->
          raise ArgumentError, "invalid bias, expected a tensor, got #{inspect(other)}"
      end

    conv_transpose_impl(input, kernel, bias, opts)
  end

  defnp conv_transpose_impl(input, kernel, bias, opts \\ []) do
    assert_min_rank!("Axon.Layers.conv_transpose", "input", input, 3)
    assert_equal_rank!("Axon.Layers.conv_transpose", "input", input, "kernel", kernel)

    opts =
      keyword!(opts,
        strides: 1,
        padding: :valid,
        kernel_dilation: 1,
        channels: :last,
        mode: :inference
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
        {Nx.shape(kernel), opts[:kernel_dilation], strides, opts[:padding]},
        fn {shape, k_dilation, strides, padding} ->
          Axon.Shape.conv_transpose_padding(shape, k_dilation, strides, padding)
        end
      )

    ones = transform(Nx.rank(input), &List.duplicate(1, &1 - 2))

    conv(input, kernel, bias,
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
    * `kernel` - `{output_channels, 1, kernel_spatial0, ..., kernel_spatialN}`
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
  deftransform depthwise_conv(inputs, kernel, bias \\ 0, opts \\ []) do
    {bias, opts} =
      case bias do
        %Nx.Tensor{} = bias ->
          {bias, opts}

        bias when is_number(bias) ->
          {bias, opts}

        opts when is_list(opts) ->
          {Nx.tensor(0), opts}

        other ->
          raise ArgumentError, "invalid bias, expected a tensor, got #{inspect(other)}"
      end

    depthwise_conv_impl(inputs, kernel, bias, opts)
  end

  defnp depthwise_conv_impl(input, kernel, bias, opts \\ []) do
    assert_min_rank!("Axon.Layers.depthwise_conv", "input", input, 3)
    assert_equal_rank!("Axon.Layers.depthwise_conv", "input", input, "kernel", kernel)

    opts =
      keyword!(opts,
        strides: 1,
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        channels: :last,
        mode: :inference
      )

    num_groups =
      transform({Nx.shape(input), opts[:channels]}, fn
        {shape, :first} ->
          elem(shape, 1)

        {shape, :last} ->
          elem(shape, tuple_size(shape) - 1)
      end)

    conv(input, kernel, bias,
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
    assert_rank!("Axon.Layers.separable_conv2d", "input", input, 4)

    assert_equal_rank!("Axon.Layers.separable_conv2d", ["input", "kernel1", "kernel2"], [
      input,
      k1,
      k2
    ])

    input
    |> depthwise_conv(k1, b1, opts)
    |> depthwise_conv(k2, b2, opts)
  end

  @doc false
  defn separable_conv2d(input, k1, k2, opts \\ []) do
    separable_conv2d(input, k1, 0, k2, 0, opts)
  end

  @doc """
  Functional implementation of a 3-dimensional separable depthwise
  convolution.

  The 3-d depthwise separable convolution performs 3 depthwise convolutions
  each over 1 spatial dimension of the input.

  ## Parameter Shapes

    * `input` - `{batch_size, input_channels, input_spatial0, input_spatial1, input_spatial2}`
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
    assert_rank!("Axon.Layers.separable_conv3d", "input", input, 5)

    assert_equal_rank!(
      "Axon.Layers.separable_conv3d",
      ["input", "kernel1", "kernel2", "kernel3"],
      [input, k1, k2, k3]
    )

    input
    |> depthwise_conv(k1, b1, opts)
    |> depthwise_conv(k2, b2, opts)
    |> depthwise_conv(k3, b3, opts)
  end

  @doc false
  defn separable_conv3d(input, k1, k2, k3, opts \\ []) do
    separable_conv3d(input, k1, 0, k2, 0, k3, 0, opts)
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
      iex> Axon.Layers.max_pool(t, kernel_size: 2, channels: :first)
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
    assert_min_rank!("Axon.Layers.max_pool", "input", input, 3)

    opts =
      keyword!(
        opts,
        [
          :kernel_size,
          strides: nil,
          padding: :valid,
          window_dilations: 1,
          channels: :last,
          mode: :inference
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

    dilations =
      transform(
        {Nx.rank(input), opts[:window_dilations], opts[:channels]},
        fn
          {_, [_ | _] = dilations, :first} -> [1, 1 | dilations]
          {rank, dilations, :first} -> [1, 1 | List.duplicate(dilations, rank - 2)]
          {_, [_ | _] = dilations, :last} -> [1 | dilations] ++ [1]
          {rank, dilations, :last} -> [1 | List.duplicate(dilations, rank - 2)] ++ [1]
        end
      )

    padding =
      transform(
        {opts[:padding], opts[:channels]},
        fn
          {:same, _} -> :same
          {:valid, _} -> :valid
          {padding, :first} -> [{0, 0}, {0, 0} | padding]
          {padding, :last} -> [{0, 0} | padding] ++ [{0, 0}]
        end
      )

    input
    |> Nx.window_max(window_dimensions,
      strides: strides,
      padding: padding,
      window_dilations: dilations
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
    assert_min_rank!("Axon.Layers.avg_pool", "input", input, 3)

    opts =
      keyword!(
        opts,
        [
          :kernel_size,
          strides: nil,
          padding: :valid,
          window_dilations: 1,
          channels: :last,
          mode: :inference
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

    dilations =
      transform(
        {Nx.rank(input), opts[:window_dilations], opts[:channels]},
        fn
          {_, [_ | _] = dilations, :first} -> [1, 1 | dilations]
          {rank, dilations, :first} -> [1, 1 | List.duplicate(dilations, rank - 2)]
          {_, [_ | _] = dilations, :last} -> [1 | dilations] ++ [1]
          {rank, dilations, :last} -> [1 | List.duplicate(dilations, rank - 2)] ++ [1]
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

    input
    |> Nx.window_mean(window_dimensions,
      strides: strides,
      padding: padding,
      window_dilations: dilations
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
      iex> Axon.Layers.lp_pool(t, kernel_size: 2, norm: 2, channels: :first)
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
    assert_min_rank!("Axon.Layers.lp_pool", "input", input, 3)

    opts =
      keyword!(
        opts,
        [
          :kernel_size,
          strides: nil,
          padding: :valid,
          window_dilations: 1,
          norm: 2,
          channels: :last,
          mode: :inference
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

    dilations =
      transform(
        {Nx.rank(input), opts[:window_dilations], opts[:channels]},
        fn
          {_, [_ | _] = dilations, :first} -> [1, 1 | dilations]
          {rank, dilations, :first} -> [1, 1 | List.duplicate(dilations, rank - 2)]
          {_, [_ | _] = dilations, :last} -> [1 | dilations] ++ [1]
          {rank, dilations, :last} -> [1 | List.duplicate(dilations, rank - 2)] ++ [1]
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
      window_dilations: dilations
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
    assert_min_rank!("Axon.Layers.adaptive_avg_pool", "input", input, 3)

    opts = keyword!(opts, [:output_size, channels: :last, mode: :inference])

    output_size =
      transform({Nx.shape(input), opts[:output_size], opts[:channels]}, fn {shape, size, channels} ->
        Axon.Shape.adaptive_pool_window_size(shape, size, channels)
      end)

    window_strides =
      transform(
        {Nx.shape(input), Nx.rank(input), output_size, opts[:channels]},
        fn {shape, rank, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_strides(shape, output_size, rank - 2, channels)
        end
      )

    window_dimensions =
      transform(
        {Nx.shape(input), Nx.rank(input), window_strides, output_size, opts[:channels]},
        fn {shape, rank, strides, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_size(shape, strides, output_size, rank - 2, channels)
        end
      )

    Nx.window_mean(input, window_dimensions, padding: :valid, strides: window_strides)
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
    assert_min_rank!("Axon.Layers.adaptive_max_pool", "input", input, 3)

    opts = keyword!(opts, [:output_size, channels: :last, mode: :inference])

    output_size =
      transform({Nx.shape(input), opts[:output_size], opts[:channels]}, fn {shape, size, channels} ->
        Axon.Shape.adaptive_pool_window_size(shape, size, channels)
      end)

    window_strides =
      transform(
        {Nx.shape(input), Nx.rank(input), output_size, opts[:channels]},
        fn {shape, rank, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_strides(shape, output_size, rank - 2, channels)
        end
      )

    window_dimensions =
      transform(
        {Nx.shape(input), Nx.rank(input), window_strides, output_size, opts[:channels]},
        fn {shape, rank, strides, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_size(shape, strides, output_size, rank - 2, channels)
        end
      )

    Nx.window_max(input, window_dimensions, padding: :valid, strides: window_strides)
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
    assert_min_rank!("Axon.Layers.adaptive_lp_pool", "input", input, 3)

    opts = keyword!(opts, [:output_size, norm: 2, channels: :last, mode: :inference])

    norm = opts[:norm]

    output_size =
      transform({Nx.shape(input), opts[:output_size], opts[:channels]}, fn {shape, size, channels} ->
        Axon.Shape.adaptive_pool_window_size(shape, size, channels)
      end)

    window_strides =
      transform(
        {Nx.shape(input), Nx.rank(input), output_size, opts[:channels]},
        fn {shape, rank, output_size, channels} ->
          Axon.Shape.adaptive_pool_window_strides(shape, output_size, rank - 2, channels)
        end
      )

    window_dimensions =
      transform(
        {Nx.shape(input), Nx.rank(input), window_strides, output_size, opts[:channels]},
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
  defn batch_norm(input, gamma, beta, ra_mean, ra_var, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-5, channel_index: -1, momentum: 0.1, mode: :inference)

    training? =
      transform(opts[:mode], fn
        :inference -> false
        :train -> true
      end)

    {axes, channel_index} =
      transform({input, opts[:channel_index]}, fn {input, channel} ->
        axes = Nx.axes(input)
        axis = Nx.Shape.normalize_axis(Nx.shape(input), channel, Nx.names(input))
        {Axon.Shape.batch_norm_axes(axes, axis), axis}
      end)

    num_channels =
      transform({input, channel_index}, fn {inp, channel_idx} ->
        elem(Nx.shape(inp), channel_idx)
      end)

    {gamma, beta, ra_mean, ra_var} =
      transform(
        {gamma, beta, ra_mean, ra_var, Nx.rank(input), num_channels, channel_index},
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
      {input, gamma, beta, ra_mean, ra_var, axes, opts[:epsilon], opts[:momentum], training?},
      fn
        {x, g, b, m, v, axes, eps, alpha, true} ->
          {new_mean, new_var} = mean_and_variance(x, axes: axes)
          out = normalize(x, new_mean, new_var, g, b, epsilon: eps)
          ra_mean = update_ema(new_mean, m, alpha)
          ra_var = update_ema(new_var, v, alpha)

          %Axon.StatefulOutput{
            output: out,
            state: %{"mean" => ra_mean, "var" => ra_var}
          }

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
  defn layer_norm(input, gamma, beta, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-5, channel_index: -1, mode: :inference)
    axes = opts[:channel_index]

    channel_index = opts[:channel_index]

    num_channels =
      transform({input, channel_index}, fn {inp, channel_idx} ->
        names = List.duplicate(nil, Nx.rank(inp))
        axis = Nx.Shape.normalize_axis(Nx.shape(inp), channel_idx, names)
        elem(Nx.shape(inp), axis)
      end)

    {gamma, beta} =
      transform({gamma, beta, input, Nx.rank(input), num_channels, channel_index}, fn {g, b,
                                                                                       input,
                                                                                       rank,
                                                                                       num_channels,
                                                                                       channel_idx} ->
        names = List.duplicate(nil, rank)
        axis = Nx.Shape.normalize_axis(Nx.shape(input), channel_idx, names)

        new_shape =
          1
          |> List.duplicate(rank)
          |> List.to_tuple()
          |> put_elem(axis, num_channels)

        {Nx.reshape(g, new_shape), Nx.reshape(b, new_shape)}
      end)

    {mean, var} = mean_and_variance(input, axes: [axes])
    normalize(input, mean, var, gamma, beta, epsilon: opts[:epsilon])
  end

  @doc """
  Functional implementation of group normalization.

  Normalizes the input by reshaping input into `:num_groups`
  groups and then calculating the mean and variance along
  every dimension but the input batch dimension.

  $$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$

  `gamma` and `beta` are often trainable parameters. This method does
  not maintain an EMA of mean and variance.

  ## Options

    * `:num_groups` - Number of groups.

    * `:epsilon` - numerical stability term. $epsilon$ in the above
      formulation.

    * `:channel_index` - channel index used to determine reduction
      axes and group shape for mean and variance calculation.

  ## References

    * [Group Normalization](https://arxiv.org/abs/1803.08494v3)
  """
  @doc type: :normalization
  defn group_norm(input, gamma, beta, opts \\ []) do
    opts = keyword!(opts, [:num_groups, epsilon: 1.0e-5, channel_index: -1, mode: :inference])

    channel_axis =
      transform({Nx.shape(input), opts[:channel_index]}, fn
        {shape, channel_index} ->
          names = List.duplicate(nil, Nx.rank(shape))
          Nx.Shape.normalize_axis(shape, channel_index, names)
      end)

    group_shape =
      transform({Nx.shape(input), opts[:num_groups], channel_axis}, fn
        {shape, groups, channel_axis} ->
          Axon.Shape.group_norm_shape(shape, groups, channel_axis)
      end)

    channel_index = opts[:channel_index]

    num_channels =
      transform({input, channel_index}, fn {inp, channel_idx} ->
        names = List.duplicate(nil, Nx.rank(inp))
        axis = Nx.Shape.normalize_axis(Nx.shape(inp), channel_idx, names)
        elem(Nx.shape(inp), axis)
      end)

    {gamma, beta} =
      transform({gamma, beta, input, Nx.rank(input), num_channels, channel_index}, fn
        {g, b, inp, rank, num_channels, channel_idx} ->
          names = List.duplicate(nil, Nx.rank(inp))
          axis = Nx.Shape.normalize_axis(Nx.shape(inp), channel_idx, names)

          new_shape =
            1
            |> List.duplicate(rank)
            |> List.to_tuple()
            |> put_elem(axis, num_channels)

          {Nx.reshape(g, new_shape), Nx.reshape(b, new_shape)}
      end)

    x = Nx.reshape(input, group_shape)

    axes =
      transform({x, channel_axis}, fn {x, channel_axis} ->
        Axon.Shape.group_norm_axes(Nx.rank(x), channel_axis)
      end)

    {mean, var} = mean_and_variance(x, axes: axes)
    x = (x - mean) * Nx.rsqrt(var + opts[:epsilon])
    x = Nx.reshape(x, input)
    x * gamma + beta
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
  defn instance_norm(input, gamma, beta, ra_mean, ra_var, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-5, channel_index: -1, momentum: 0.1, mode: :inference)

    training? =
      transform(opts[:mode], fn
        :inference -> false
        :train -> true
      end)

    {axes, channel_index} =
      transform({input, opts[:channel_index]}, fn {input, channel} ->
        axes = Nx.axes(input)
        axis = Nx.Shape.normalize_axis(Nx.shape(input), channel, Nx.names(input))
        {Axon.Shape.instance_norm_axes(axes, axis), axis}
      end)

    num_channels =
      transform({input, channel_index}, fn {inp, channel_idx} ->
        elem(Nx.shape(inp), channel_idx)
      end)

    {gamma, beta, ra_mean, ra_var} =
      transform(
        {gamma, beta, ra_mean, ra_var, Nx.rank(input), num_channels, channel_index},
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
      {input, gamma, beta, ra_mean, ra_var, axes, opts[:epsilon], opts[:momentum], training?},
      fn
        {x, g, b, m, v, axes, eps, alpha, true} ->
          {new_mean, new_var} = mean_and_variance(x, axes: axes)
          out = normalize(x, new_mean, new_var, g, b, epsilon: eps)
          ra_mean = update_ema(new_mean, m, alpha)
          ra_var = update_ema(new_var, v, alpha)

          %Axon.StatefulOutput{
            output: out,
            state: %{"mean" => ra_mean, "var" => ra_var}
          }

        {x, g, b, m, v, _, eps, _, _} ->
          normalize(x, m, v, g, b, epsilon: eps)
      end
    )
  end

  ## Stochastic

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

    * `:noise_shape` - input noise shape. Shape of `mask` which can be useful
      for broadcasting `mask` across feature channels or other dimensions.
      Defaults to shape of input tensor.

  ## References

    * [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
  """
  @doc type: :dropout
  defn dropout(input, opts \\ []) do
    opts = keyword!(opts, [:key, :rate, noise_shape: Nx.shape(input), mode: :inference])
    keep_prob = Nx.tensor(1, type: Nx.type(input)) - Nx.tensor(opts[:rate], type: Nx.type(input))

    mask =
      Nx.less(
        Nx.Random.uniform_split(opts[:key], 0, 1, shape: opts[:noise_shape], type: Nx.type(input)),
        keep_prob
      )

    mask =
      transform(
        {mask, Nx.shape(input)},
        fn {mask, input_shape} ->
          if Elixir.Kernel.==(Nx.shape(mask), input_shape),
            do: mask,
            else: Nx.broadcast(mask, input_shape)
        end
      )

    out = Nx.select(mask, input / keep_prob, Nx.tensor(0, type: Nx.type(input)))

    transform({input, out, opts[:mode]}, fn
      {input, _, :inference} ->
        input

      {_, out, :train} ->
        out
    end)
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
    assert_min_rank!("Axon.Layers.spatial_dropout", "input", input, 3)

    opts = keyword!(opts, [:key, rate: 0.5, channels: :last, mode: :inference])

    noise_shape =
      transform({Nx.shape(input), opts[:channels]}, fn {shape, channels} ->
        Axon.Shape.spatial_dropout_noise_shape(shape, channels)
      end)

    dropout(input, key: opts[:key], rate: opts[:rate], noise_shape: noise_shape, mode: opts[:mode])
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
    opts = keyword!(opts, [:key, rate: 0.5, mode: :inference])
    rate = opts[:rate]

    alpha = Nx.tensor(1.6732632423543772848170429916717, type: Nx.type(input))
    scale = Nx.tensor(1.0507009873554804934193349852946, type: Nx.type(input))
    alpha_p = -alpha * scale
    keep_prob = Nx.tensor(1, type: Nx.type(input)) - rate

    mask =
      Nx.less(
        Nx.Random.uniform_split(opts[:key], 0, 1, shape: Nx.shape(input), type: Nx.type(input)),
        keep_prob
      )

    a = Nx.rsqrt(keep_prob * Nx.power(Nx.tensor(1, type: Nx.type(input)) * alpha_p, 2))
    b = -a * alpha_p * rate

    x = Nx.select(mask, input, alpha_p)
    out = a * x + b

    transform({input, out, opts[:mode]}, fn
      {input, _, :inference} ->
        input

      {_, out, :train} ->
        out
    end)
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
    assert_min_rank!("Axon.Layers.feature_alpha_dropout", "input", input, 3)

    opts = keyword!(opts, [:key, rate: 0.5, channels: :last, mode: :inference])

    noise_shape =
      transform({Nx.shape(input), opts[:channels]}, fn {shape, channels} ->
        Axon.Shape.spatial_dropout_noise_shape(shape, channels)
      end)

    keep_prob = 1 - opts[:rate]

    mask =
      Nx.less(
        Nx.Random.uniform_split(opts[:key], 0, 1, shape: noise_shape, type: Nx.type(input)),
        keep_prob
      )

    mask =
      transform(
        {mask, Nx.shape(input)},
        fn {mask, input_shape} ->
          if Elixir.Kernel.==(Nx.shape(mask), input_shape),
            do: mask,
            else: Nx.broadcast(mask, input_shape)
        end
      )

    out = Nx.select(mask, input / keep_prob, Nx.negate(Axon.Activations.selu(input)))

    transform({input, out, opts[:mode]}, fn
      {input, _, :inference} ->
        input

      {_, out, :train} ->
        out
    end)
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

      iex> Axon.Layers.global_avg_pool(Nx.iota({3, 2, 3}, type: {:f, 32}), channels: :first)
      #Nx.Tensor<
        f32[3][2]
        [
          [1.0, 4.0],
          [7.0, 10.0],
          [13.0, 16.0]
        ]
      >

      iex> Axon.Layers.global_avg_pool(Nx.iota({1, 3, 2, 2}, type: {:f, 32}), channels: :first, keep_axes: true)
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
  @doc type: :pooling
  defn global_avg_pool(input, opts \\ []) do
    assert_min_rank!("Axon.Layers.global_avg_pool", "input", input, 3)

    opts = keyword!(opts, channels: :last, keep_axes: false, mode: :inference)

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
  the spatial dimensions of the input such that the only remaining dimensions are
  the batch and feature dimensions.

  Assumes data is configured in a channels-first like format.

  ## Parameter Shapes

    * `input` - {batch_size, s1, ..., sN, features}

  ## Options

    * `:keep_axes` - option to keep reduced axes with size 1 for each reduced
      dimensions. Defaults to `false`

  ## Examples

      iex> Axon.Layers.global_max_pool(Nx.iota({3, 2, 3}, type: {:f, 32}), channels: :first)
      #Nx.Tensor<
        f32[3][2]
        [
          [2.0, 5.0],
          [8.0, 11.0],
          [14.0, 17.0]
        ]
      >

      iex> Axon.Layers.global_max_pool(Nx.iota({1, 3, 2, 2}, type: {:f, 32}), keep_axes: true, channels: :first)
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
  @doc type: :pooling
  defn global_max_pool(input, opts \\ []) do
    assert_min_rank!("Axon.Layers.global_max_pool", "input", input, 3)

    opts = keyword!(opts, keep_axes: false, channels: :last, mode: :inference)

    all_but_batch_and_feature =
      transform({Nx.rank(input), opts[:channels]}, fn
        {rank, :first} ->
          for i <- 2..(rank - 1), do: i

        {rank, :last} ->
          for i <- 1..(rank - 2), do: i
      end)

    Nx.reduce_max(input, axes: all_but_batch_and_feature, keep_axes: opts[:keep_axes])
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

      iex> Axon.Layers.global_lp_pool(Nx.iota({3, 2, 3}, type: {:f, 32}), norm: 1, channels: :first)
      #Nx.Tensor<
        f32[3][2]
        [
          [3.0, 12.0],
          [21.0, 30.0],
          [39.0, 48.0]
        ]
      >

      iex> Axon.Layers.global_lp_pool(Nx.iota({1, 3, 2, 2}, type: {:f, 16}), keep_axes: true, channels: :first)
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
  @doc type: :pooling
  defn global_lp_pool(input, opts \\ []) do
    assert_min_rank!("Axon.Layers.global_lp_pool", "input", input, 3)

    opts = keyword!(opts, norm: 2, keep_axes: false, channels: :last, mode: :inference)

    norm = opts[:norm]

    all_but_batch_and_feature =
      transform({Nx.rank(input), opts[:channels]}, fn
        {rank, :first} ->
          for i <- 2..(rank - 1), do: i

        {rank, :last} ->
          for i <- 1..(rank - 2), do: i
      end)

    input
    |> Nx.power(norm)
    |> Nx.sum(axes: all_but_batch_and_feature, keep_axes: opts[:keep_axes])
    |> Nx.power(Nx.divide(Nx.tensor(1, type: Nx.type(input)), norm))
  end

  ## Sparse

  @doc """
  Computes embedding by treating kernel matrix as a lookup table
  for discrete tokens.

  `input` is a vector of discrete values, typically representing tokens
  (e.g. words, characters, etc.) from a vocabulary. `kernel` is a kernel
  matrix of shape `{vocab_size, embedding_size}` from which the dense
  embeddings will be drawn.

  ## Parameter Shapes

    * `input` - `{batch_size, ..., seq_len}`
    * `kernel` - `{vocab_size, embedding_size}`

  ## Examples

      iex> input = Nx.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
      iex> kernels = Nx.tensor([
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
      iex> Axon.Layers.embedding(input, kernels)
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
  @doc type: :linear
  defn embedding(input, kernel, _opts \\ []) do
    assert_rank!("Axon.Layers.embedding", "kernel", kernel, 2)
    Nx.take(kernel, Nx.as_type(input, {:s, 64}), axis: 0)
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
  @doc type: :shape
  defn flatten(x, _opts \\ []) do
    new_shape = transform(Nx.shape(x), &Axon.Shape.flatten/1)

    Nx.reshape(x, new_shape)
  end

  @doc false
  # Internal version of Nx.reshape for constructing reshape layers
  # without worrying about a batch dimension
  defn reshape(x, opts \\ []) do
    opts = keyword!(opts, [:shape, mode: :inference])

    transform({opts[:shape], x}, fn {shape, x} ->
      batch_size = Nx.axis_size(x, 0)

      new_shape =
        shape
        |> Tuple.to_list()
        |> Enum.map(fn
          :batch -> batch_size
          val -> val
        end)
        |> List.to_tuple()

      Nx.reshape(x, new_shape)
    end)
  end

  @doc false
  # Internal version of Nx.pad for constructing pad layers without
  # worrying about batch or channel dimensions
  defn pad(x, opts \\ []) do
    opts = keyword!(opts, [:padding_config, :value, :channels, mode: :inference])

    config =
      transform({opts[:padding_config], opts[:channels]}, fn
        {config, :first} ->
          [{0, 0, 0}, {0, 0, 0} | Enum.map(config, fn {x, y} -> {x, y, 0} end)]

        {config, :last} ->
          [{0, 0, 0} | Enum.map(config, fn {x, y} -> {x, y, 0} end)] ++ [{0, 0, 0}]
      end)

    Nx.pad(x, Nx.as_type(opts[:value], Nx.type(x)), config)
  end

  @doc false
  # Internal version of Nx.transpose for constructing a transpose layer
  # without worrying about a batch dimension
  defn transpose(x, opts \\ []) do
    opts = keyword!(opts, [:axes, mode: :inference])

    axes =
      transform({Nx.shape(x), opts[:axes]}, fn
        {shape, nil} ->
          Nx.axes(shape) |> Enum.reverse()

        {_, axes} ->
          axes
      end)

    Nx.transpose(x, axes: axes)
  end

  @doc false
  # Internal helper for constructing conditional layers without
  # needing to use the if-macros in Axon.Compiler
  defn cond(cond_input_expr, on_true_expr, on_false_expr, opts \\ []) do
    opts = keyword!(opts, [:cond, mode: :inference])
    cond_expr = opts[:cond].(cond_input_expr)

    transform(cond_expr, fn cond_expr ->
      cond_rank = Nx.rank(cond_expr)
      cond_type = Nx.type(cond_expr)

      unless Elixir.Kernel.and(
               Elixir.Kernel.==(cond_rank, 0),
               Elixir.Kernel.==(cond_type, {:u, 8})
             ) do
        raise ArgumentError,
              "cond_fn must return a scalar-boolean tensor" <>
                " got result with rank #{inspect(cond_rank)} and" <>
                " type #{inspect(cond_type)}"
      end
    end)

    if cond_expr do
      on_true_expr
    else
      on_false_expr
    end
  end

  @doc false
  # Internal helper for constructing bias layers without
  defn bias(input, bias, _opts \\ []) do
    input + bias
  end

  @doc """
  Resizes a batch of tensors to the given shape using one of a
  number of sampling methods.

  Requires input option `:to` which should be a tuple specifying
  the resized spatial dimensions of the input tensor. Input tensor
  must be at least rank 3, with fixed `batch` and `channel` dimensions.
  Resizing will upsample or downsample using the given resize method.

  Supported resize methods are `:nearest, :linear, :bilinear, :trilinear,
  :cubic, :bicubic, :tricubic`.

  ## Examples

      iex> img = Nx.iota({1, 1, 3, 3}, type: {:f, 32})
      iex> Axon.Layers.resize(img, size: {4, 4}, channels: :first)
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

  ### Error cases

      iex> img = Nx.iota({1, 1, 3, 3}, type: {:f, 32})
      iex> Axon.Layers.resize(img, size: {4, 4}, method: :foo)
      ** (ArgumentError) expected :method to be either of :nearest, :bilinear, :bicubic, :lanczos3, :lanczos5, got: :foo
  """
  @doc type: :shape
  defn resize(input, opts \\ []) do
    assert_rank!("Axon.Layers.resize", "input", input, 4)

    opts =
      keyword!(opts, [
        :size,
        method: :nearest,
        channels: :last,
        mode: :inference
      ])

    transform({input, opts}, fn {input, opts} ->
      {spatial_axes, out_shape} =
        input
        |> spatial_axes_with_sizes(opts)
        |> Enum.reject(fn {_axis, size, out_size} -> Elixir.Kernel.==(size, out_size) end)
        |> Enum.map_reduce(Nx.shape(input), fn {axis, _size, out_size}, out_shape ->
          {axis, put_elem(out_shape, axis, out_size)}
        end)

      resized_input =
        case opts[:method] do
          :nearest ->
            resize_nearest(input, out_shape, spatial_axes)

          :bilinear ->
            resize_with_kernel(input, out_shape, spatial_axes, &fill_linear_kernel/1)

          :bicubic ->
            resize_with_kernel(input, out_shape, spatial_axes, &fill_cubic_kernel/1)

          :lanczos3 ->
            resize_with_kernel(input, out_shape, spatial_axes, &fill_lanczos_kernel(3, &1))

          :lanczos5 ->
            resize_with_kernel(input, out_shape, spatial_axes, &fill_lanczos_kernel(5, &1))

          method ->
            raise ArgumentError,
                  "expected :method to be either of :nearest, :bilinear, :bicubic, " <>
                    ":lanczos3, :lanczos5, got: #{inspect(method)}"
        end

      cast_to(resized_input, input)
    end)
  end

  defnp spatial_axes(input, opts \\ []) do
    channels = opts[:channels]

    transform({input, channels}, fn {input, channels} ->
      axes =
        case channels do
          :first -> [-2, -1]
          :last -> [-3, -2]
        end

      axes
      |> Enum.map(&Nx.axis_index(input, &1))
      |> List.to_tuple()
    end)
  end

  defnp cast_to(left, right) do
    left
    |> Nx.as_type(Nx.type(right))
    |> Nx.reshape(left, names: Nx.names(right))
  end

  defnp resize_nearest(input, out_shape, spatial_axes) do
    transform({input, out_shape, spatial_axes}, fn {input, out_shape, spatial_axes} ->
      singular_shape = List.duplicate(1, Nx.rank(input)) |> List.to_tuple()

      for axis <- spatial_axes, reduce: input do
        input ->
          input_shape = Nx.shape(input)
          input_size = elem(input_shape, axis)
          output_size = elem(out_shape, axis)
          inv_scale = input_size / output_size
          offset = (Nx.iota({output_size}) + 0.5) * inv_scale
          offset = offset |> Nx.floor() |> Nx.as_type({:s, 32})

          offset =
            offset
            |> Nx.reshape(put_elem(singular_shape, axis, output_size))
            |> Nx.broadcast(put_elem(input_shape, axis, output_size))

          Nx.take_along_axis(input, offset, axis: axis)
      end
    end)
  end

  @f32_eps :math.pow(2, -23)

  defnp resize_with_kernel(input, out_shape, spatial_axes, kernel_fun) do
    transform({input, out_shape, spatial_axes}, fn {input, out_shape, spatial_axes} ->
      for axis <- spatial_axes, reduce: input do
        input ->
          input_shape = Nx.shape(input)
          input_size = elem(input_shape, axis)
          output_size = elem(out_shape, axis)

          inv_scale = input_size / output_size
          kernel_scale = Nx.max(1, inv_scale)

          sample_f = (Nx.iota({1, output_size}) + 0.5) * inv_scale - 0.5
          x = Nx.abs(sample_f - Nx.iota({input_size, 1})) / kernel_scale
          weights = kernel_fun.(x)

          weights_sum = Nx.sum(weights, axes: [0], keep_axes: true)

          weights =
            Nx.select(Nx.abs(weights) > 1000 * @f32_eps, safe_divide(weights, weights_sum), 0)

          input = Nx.dot(input, [axis], weights, [0])
          # The transformed axis is moved to the end, so we transpose back
          reorder_axis(input, -1, axis)
      end
    end)
  end

  defnp fill_linear_kernel(x) do
    Nx.max(0, 1 - x)
  end

  defnp fill_cubic_kernel(x) do
    # See https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    out = (1.5 * x - 2.5) * x * x + 1
    out = Nx.select(x >= 1, ((-0.5 * x + 2.5) * x - 4) * x + 2, out)
    Nx.select(x >= 2, 0, out)
  end

  @pi :math.pi()

  defnp fill_lanczos_kernel(radius, x) do
    y = radius * Nx.sin(@pi * x) * Nx.sin(@pi * x / radius)
    out = Nx.select(x > 1.0e-3, safe_divide(y, @pi ** 2 * x ** 2), 1)
    Nx.select(x > radius, 0, out)
  end

  defnp safe_divide(x, y) do
    x / Nx.select(y != 0, y, 1)
  end

  defnp reorder_axis(tensor, axis, target_axis) do
    transform({tensor, axis, target_axis}, fn {tensor, axis, target_axis} ->
      axes = Nx.axes(tensor)
      {source_axis, axes} = List.pop_at(axes, axis)
      axes = List.insert_at(axes, target_axis, source_axis)
      Nx.transpose(tensor, axes: axes)
    end)
  end

  defnp spatial_axes_with_sizes(input, opts \\ []) do
    {height_axis, width_axis} = spatial_axes(input, channels: opts[:channels])
    {height, width} = size(input, channels: opts[:channels])
    {out_height, out_width} = opts[:size]
    [{height_axis, height, out_height}, {width_axis, width, out_width}]
  end

  defnp size(input, opts \\ []) do
    opts = keyword!(opts, channels: :last)
    {height_axis, width_axis} = spatial_axes(input, channels: opts[:channels])
    {Nx.axis_size(input, height_axis), Nx.axis_size(input, width_axis)}
  end

  # Private Axon.Layers implementation of activations for the compiler
  # to use when invoking activation layers.
  @activation_layers [:exp, :gelu, :hard_tanh, :linear, :log_sigmoid] ++
                       [:mish, :relu, :relu6, :sigmoid, :silu, :softplus] ++
                       [:softsign, :tanh]

  for activation <- @activation_layers do
    @doc false
    defn unquote(activation)(input, _opts \\ []) do
      transform(input, fn inp ->
        Elixir.Kernel.apply(Axon.Activations, unquote(activation), [inp])
      end)
    end
  end

  @activation_layers_with_opts [:celu, :elu, :hard_sigmoid, :hard_silu, :leaky_relu] ++
                                 [:log_sumexp, :log_softmax, :selu, :softmax]
  for activation <- @activation_layers_with_opts do
    defn unquote(activation)(input, opts \\ []) do
      transform(input, fn inp ->
        Elixir.Kernel.apply(Axon.Activations, unquote(activation), [
          inp,
          Keyword.delete(opts, :mode)
        ])
      end)
    end
  end

  # Private combinator implementations that expect variable
  # arguments
  @doc false
  @element_wise_layers [:add, :subtract, :multiply]

  for op <- @element_wise_layers do
    defn unquote(op)(inputs, _opts \\ []) do
      transform(inputs, fn inputs ->
        [first | rest] = Tuple.to_list(inputs)

        Enum.reduce(rest, first, fn next, acc ->
          apply(Nx, unquote(op), [acc, next])
        end)
      end)
    end
  end

  @doc false
  defn concatenate(inputs, opts \\ []) do
    opts = keyword!(opts, axis: -1, mode: :inference)

    transform(inputs, fn inputs ->
      inputs
      |> Tuple.to_list()
      |> Nx.concatenate(axis: opts[:axis])
    end)
  end

  ## Recurrent

  @doc """
  GRU Cell.

  When combined with `Axon.Layers.*_unroll`, implements a
  GRU-based RNN. More memory efficient than traditional LSTM.

  ## References

  * [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555v1.pdf)
  """
  defn gru_cell(
         input,
         carry,
         input_kernel,
         hidden_kernel,
         bias,
         gate_fn \\ &Axon.Activations.sigmoid/1,
         activation_fn \\ &Axon.Activations.tanh/1
       ) do
    {hidden} = carry
    {wir, wiz, win} = input_kernel
    {whr, whz, whn} = hidden_kernel
    {br, bz, bin, bhn} = bias

    r = gate_fn.(dense(input, wir, br) + dense(hidden, whr, 0))
    z = gate_fn.(dense(input, wiz, bz) + dense(hidden, whz, 0))
    n = activation_fn.(dense(input, win, bin) + r * dense(hidden, whn, bhn))

    new_h = (1.0 - z) * n + z * hidden

    {new_h, {new_h}}
  end

  @doc """
  LSTM Cell.

  When combined with `Axon.Layers.*_unroll`, implements a
  LSTM-based RNN. More memory efficient than traditional LSTM.

  ## References

  * [Long Short-Term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf)
  """
  defn lstm_cell(
         input,
         carry,
         input_kernel,
         hidden_kernel,
         bias,
         gate_fn \\ &Axon.Activations.sigmoid/1,
         activation_fn \\ &Axon.Activations.tanh/1
       ) do
    {cell, hidden} = carry
    {wii, wif, wig, wio} = input_kernel
    {whi, whf, whg, who} = hidden_kernel

    {bi, bf, bg, bo} = bias

    i = gate_fn.(dense(input, wii, bi) + dense(hidden, whi, 0))
    f = gate_fn.(dense(input, wif, bf) + dense(hidden, whf, 0))
    g = activation_fn.(dense(input, wig, bg) + dense(hidden, whg, 0))
    o = gate_fn.(dense(input, wio, bo) + dense(hidden, who, 0))

    new_c = f * cell + i * g
    new_h = o * activation_fn.(new_c)

    {new_h, {new_c, new_h}}
  end

  @doc """
  ConvLSTM Cell.

  When combined with `Axon.Layers.*_unroll`, implements a
  ConvLSTM-based RNN. More memory efficient than traditional LSTM.

  ## Options

    * `:strides` - convolution strides. Defaults to `1`.

    * `:padding` - convolution padding. Defaults to `:same`.

  ## References

    * [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)
  """
  defn conv_lstm_cell(input, carry, input_kernel, hidden_kernel, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :same)

    {ih} = input_kernel
    {hh} = hidden_kernel
    {bi} = bias

    {input, {cell, hidden}} = rank_down({input, carry})

    gates =
      Nx.add(
        conv(input, ih, bi, strides: opts[:strides], padding: opts[:padding], channels: :first),
        conv(hidden, hh, 0, strides: opts[:strides], padding: opts[:padding], channels: :first)
      )

    {i, g, f, o} = split_gates(gates)

    f = Axon.Activations.sigmoid(f + 1)
    new_c = f * cell + Axon.Activations.sigmoid(i) * Axon.Activations.tanh(g)
    new_h = Axon.Activations.sigmoid(o) * Axon.Activations.tanh(new_c)

    rank_up({new_h, {new_c, new_h}})
  end

  defnp split_gates(gates) do
    transform(gates, fn gates ->
      channels = elem(Nx.shape(gates), 1)
      split_every = div(channels, 4)

      split_dims =
        for i <- 0..3 do
          {i * split_every, split_every}
        end

      split_dims
      |> Enum.map(fn {start, len} -> Nx.slice_along_axis(gates, start, len, axis: 1) end)
      |> List.to_tuple()
    end)
  end

  defnp rank_down(rnn_data) do
    transform(rnn_data, fn {input, {cell, hidden}} ->
      [cell, hidden, input] =
        for tensor <- [cell, hidden, input] do
          Nx.squeeze(tensor, axes: [1])
        end

      {input, {cell, hidden}}
    end)
  end

  defnp rank_up(rnn_data) do
    transform(rnn_data, fn {input, {cell, hidden}} ->
      [cell, hidden, input] =
        for tensor <- [cell, hidden, input] do
          new_shape =
            Nx.shape(tensor)
            |> Tuple.insert_at(1, 1)

          Nx.reshape(tensor, new_shape)
        end

      {input, {cell, hidden}}
    end)
  end

  @doc """
  Dynamically unrolls an RNN.

  Unrolls implement a `scan` operation which applies a
  transformation on the leading axis of `input_sequence` carrying
  some state. In this instance `cell_fn` is an RNN cell function
  such as `lstm_cell` or `gru_cell`.

  This function will make use of an `defn` while-loop such and thus
  may be more efficient for long sequences.
  """
  defn dynamic_unroll(cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias) do
    time_steps = transform(Nx.shape(input_sequence), &elem(&1, 1))

    feature_dims = transform(Nx.rank(input_sequence), &List.duplicate(0, &1 - 2))

    initial_shape =
      transform({cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias}, fn
        {cell_fn, inp, carry, inp_kernel, hid_kernel, bias} ->
          seq = Nx.slice_along_axis(inp, 0, 1, axis: 1)
          {seq, _} = cell_fn.(seq, carry, inp_kernel, hid_kernel, bias)
          put_elem(Nx.shape(seq), 1, elem(Nx.shape(inp), 1))
      end)

    init_sequence = Nx.broadcast(0.0, initial_shape)
    i = Nx.tensor(0)

    {_, carry, output, _, _, _, _} =
      while {i, carry, init_sequence, input_sequence, input_kernel, recurrent_kernel, bias},
            Nx.less(i, time_steps) do
        sequence = Nx.slice_along_axis(input_sequence, i, 1, axis: 1)
        indices = transform({feature_dims, i}, fn {feature_dims, i} -> [0, i] ++ feature_dims end)
        {output, carry} = cell_fn.(sequence, carry, input_kernel, recurrent_kernel, bias)
        update_sequence = Nx.put_slice(init_sequence, indices, output)
        {i + 1, carry, update_sequence, input_sequence, input_kernel, recurrent_kernel, bias}
      end

    {output, carry}
  end

  @doc """
  Statically unrolls an RNN.

  Unrolls implement a `scan` operation which applies a
  transformation on the leading axis of `input_sequence` carrying
  some state. In this instance `cell_fn` is an RNN cell function
  such as `lstm_cell` or `gru_cell`.

  This function inlines the unrolling of the sequence such that
  the entire operation appears as a part of the compilation graph.
  This makes it suitable for shorter sequences.
  """
  defn static_unroll(cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias) do
    static_unroll_loop(cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias)
  end

  deftransformp static_unroll_loop(
                  cell_fn,
                  input_sequence,
                  carry,
                  input_kernel,
                  recurrent_kernel,
                  bias
                ) do
    time_steps = elem(Nx.shape(input_sequence), 1)

    {carry, outputs} =
      for t <- 0..(time_steps - 1), reduce: {carry, []} do
        {carry, outputs} ->
          input = Nx.slice_along_axis(input_sequence, t, 1, axis: 1)
          {output, carry} = cell_fn.(input, carry, input_kernel, recurrent_kernel, bias)
          {carry, [output | outputs]}
      end

    {Nx.concatenate(Enum.reverse(outputs), axis: 1), carry}
  end

  @recurrent_layers [lstm: {0, 0, 0, 0}, gru: {0, 0, 0, 0}, conv_lstm: {0}]

  for {rnn_op, default} <- @recurrent_layers do
    deftransform unquote(rnn_op)(
                   input,
                   hidden_state,
                   input_kernel,
                   hidden_kernel,
                   bias \\ [],
                   opts \\ []
                 ) do
      {bias, opts} =
        cond do
          is_list(bias) -> {unquote(Macro.escape(default)), bias}
          is_tuple(bias) -> {bias, opts}
          true -> raise ArgumentError, "invalid bias #{inspect(bias)}"
        end

      opts =
        Keyword.validate!(opts,
          mode: :inference,
          unroll: :static,
          activation: :sigmoid,
          gate: :tanh,
          conv_opts: []
        )

      cell_fn = get_cell_fn(unquote(rnn_op), opts[:activation], opts[:gate], opts[:conv_opts])

      case opts[:unroll] do
        :static ->
          Axon.Layers.static_unroll(
            cell_fn,
            input,
            hidden_state,
            input_kernel,
            hidden_kernel,
            bias
          )

        :dynamic ->
          Axon.Layers.dynamic_unroll(
            cell_fn,
            input,
            hidden_state,
            input_kernel,
            hidden_kernel,
            bias
          )
      end
    end
  end

  defp get_cell_fn(:lstm, activation, gate, _) do
    gate_fn = &apply(Axon.Activations, gate, [&1])
    act_fn = &apply(Axon.Activations, activation, [&1])
    &lstm_cell(&1, &2, &3, &4, &5, gate_fn, act_fn)
  end

  defp get_cell_fn(:gru, activation, gate, _) do
    gate_fn = &apply(Axon.Activations, gate, [&1])
    act_fn = &apply(Axon.Activations, activation, [&1])
    &gru_cell(&1, &2, &3, &4, &5, gate_fn, act_fn)
  end

  defp get_cell_fn(:conv_lstm, _, _, conv_opts) do
    &conv_lstm_cell(&1, &2, &3, &4, &5, conv_opts)
  end

  @doc false
  defn split(input, opts \\ []) do
    assert_min_rank!("Axon.Layers.split", "input", input, 2)
    opts = keyword!(opts, [:index, :splits, axis: -1, mode: :train])

    shape = Nx.shape(input)

    {offset, size} =
      transform(
        {shape, opts[:index], opts[:splits], opts[:axis]},
        fn {shape, idx, splits, axis} ->
          slice_size = Axon.Shape.split(shape, splits, axis)
          offset = idx * slice_size
          {offset, slice_size}
        end
      )

    Nx.slice_along_axis(input, offset, size, axis: opts[:axis])
  end

  @doc false
  defn stack_columns(inputs, opts \\ []) do
    opts = keyword!(opts, ignore: [], mode: :train)

    stack_columns_transform(inputs, opts[:ignore])
  end

  deftransformp stack_columns_transform(container, ignore) do
    container
    |> Map.from_struct()
    |> Enum.reject(fn {k, _} -> k in ignore end)
    |> Enum.reduce([], fn {_, v}, acc -> [v | acc] end)
    |> Enum.reverse()
    |> Nx.stack(axis: -1)
  end
end
