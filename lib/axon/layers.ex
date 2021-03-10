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
  parameters trained using gradient-based optimzation.

  ## Parameter Shapes

    * `input` - `{batch_size, ..., input_features}`
    * `weight` - `{input_features, output_features}`
    * `bias` - `{output_features}`

  ## Output Shape

    `{batch_size, output_features}`

  ## Examples

      iex> input = Nx.tensor([[1.0, 0.5, 1.0, 0.5], [1.0, 2.0, 1.0, 2.0]], type: {:f, 32})
      iex> weight = Nx.tensor([[0.2], [0.3], [0.5], [0.8]], type: {:f, 32})
      iex> bias = Nx.tensor([1.0], type: {:f, 32})
      iex> Axon.Layers.dense(input, weight, bias)
      #Nx.Tensor<
        f32[2][1]
        [
          [2.25],
          [3.9]
        ]
      >
  """
  @doc type: :linear
  defn dense(input, weight, bias) do
    input
    |> Nx.dot([Nx.rank(input) - 1], weight, [0])
    |> Nx.add(bias)
  end

  ## Convolutional

  @doc """
  Functional implementation of a 1-dimensional convolution.

  ## Parameter Shapes

    * `input` - `{batch_size, input_channels, input_spatial}`
    * `weight` - `{output_channels, input_channels, kernel_spatial}`
    * `bias` - `{output_channels}` or `{}`

  ## Examples

      iex> input = Nx.tensor([[[0.1294, -0.6638, 1.0251]], [[ 0.9182,  1.1512, -1.6149]]], type: {:f, 32})
      iex> weight = Nx.tensor([[[-1.5475, 1.2425]], [[0.1871, 0.5458]], [[-0.4488,  0.8879]]], type: {:f, 32})
      iex> bias = Nx.tensor([0.7791, 0.1676, 1.5971], type: {:f, 32})
      iex> Axon.Layers.conv1d(input, weight, bias)
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

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a list
      of size 1. Defaults to 1.

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

    * `:groups` - feature group count. Splits the input features
      into groups. `in_channels` must be divisible by the number
      of groups, and `out_channels` must equal `in_channels * groups`.
      Defaults to `1`.
  """
  @doc type: :convolutional
  defn conv1d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: [1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        feature_group_size: 1,
        batch_group_size: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 1))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: opts[:feature_group_size],
      batch_group_size: opts[:batch_group_size]
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 2-dimensional convolution.

  ## Parameter Shapes

    * `input` - `{batch_size, input_channels, input_height, input_width}`
    * `weight` - `{output_channels, input_channels, kernel_height, kernel_width}`
    * `bias` - `{output_channels}` or `{}`

  ## Examples

      iex> input = Nx.tensor([[[[-1.0476, -0.5041], [-0.9336, 1.5907]]]], type: {:f, 32})
      iex> weight = Nx.tensor([
      ...>  [[[0.7514, 0.7356], [1.3909,  0.6800]]],
      ...>  [[[-0.3450,  0.4551], [-0.6275, -0.9875]]],
      ...>  [[[1.8587, 0.4722], [0.6058, -1.0301]]]
      ...> ], type: {:f, 32})
      iex> bias = Nx.tensor([1.9564, 0.2822, -0.5385], type: {:f, 32})
      iex> Axon.Layers.conv2d(input, weight, bias)
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

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a tuple
      of size 2. Defaults to 1.

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

    * `:groups` - feature group count. Splits the input features
      into groups. `in_channels` must be divisible by the number
      of groups, and `out_channels` must equal `in_channels * groups`.
      Defaults to `1`.
  """
  @doc type: :convolutional
  defn conv2d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: [1, 1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        feature_group_size: 1,
        batch_group_size: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 2))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: opts[:feature_group_size],
      batch_group_size: opts[:batch_group_size]
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 3-dimensional convolution.

  ## Parameter Shapes

    * `input` - `{batch_size, input_channels, input_temporal, input_height, input_width}`
    * `weight` - `{output_channels, input_channels, kernel_temporal, kernel_height, kernel_width}`
    * `bias` - `{output_channels}` or `{}`

  ## Examples

    iex> input = Nx.tensor([[[[[-0.6497], [1.0939]], [[-2.5465], [0.7801]]]]], type: {:f, 32})
    iex> weight = Nx.tensor([
    ...>  [[[[ 0.7390], [-0.0927]], [[-0.8675], [-0.9209]]]],
    ...>  [[[[-0.6638], [0.4341]], [[0.6368], [1.1846]]]]
    ...> ], type: {:f, 32})
    iex> bias = Nx.tensor([-0.4101,  0.1776], type: {:f, 32})
    iex> Axon.Layers.conv3d(input, weight, bias)
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

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a tuple
      of size 3. Defaults to 1.

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

    * `:groups` - feature group count. Splits the input features
      into groups. `in_channels` must be divisible by the number
      of groups, and `out_channels` must equal `in_channels * groups`.
      Defaults to `1`.
  """
  @doc type: :convolutional
  defn conv3d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: [1, 1, 1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        feature_group_size: 1,
        batch_group_size: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 3))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: 1,
      batch_group_size: 1
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 1-dimensional transposed convolution.

  ## Examples

      iex> input = Nx.iota({1, 3, 3}, type: {:f, 32})
      iex> kernel = Nx.iota({6, 3, 2}, type: {:f, 32})
      iex> bias = Nx.tensor(1.0, type: {:f, 32})
      iex> Axon.Layers.conv_transpose1d(input, kernel, bias)
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
  """
  @doc type: :convolutional
  defn conv_transpose1d(input, weight, bias, opts \\ []) do
    assert_equal_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 1))

    padding =
      transform(
        {Nx.shape(weight), opts[:kernel_dilation], opts[:strides], opts[:padding]},
        &conv_transpose_padding/1
      )

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: padding,
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 2-dimensional transposed convolution.
  """
  @doc type: :convolutional
  defn conv_transpose2d(input, weight, bias, opts \\ []) do
    assert_equal_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1, 1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 2))

    padding =
      transform(
        {Nx.shape(weight), opts[:kernel_dilation], opts[:strides], opts[:padding]},
        &conv_transpose_padding/1
      )

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: padding,
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 3-dimensional transposed convolution.
  """
  @doc type: :convolutional
  defn conv_transpose3d(input, weight, bias, opts \\ []) do
    assert_equal_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1, 1, 1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 3))

    padding =
      transform(
        {Nx.shape(weight), opts[:kernel_dilation], opts[:strides], opts[:padding]},
        &conv_transpose_padding/1
      )

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: padding,
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 1-dimensional depthwise convolution.

  A depthwise convolution is essentially just a grouped convolution
  where the number of groups is equal to the number of input channels.
  """
  @doc type: :convolutional
  defn depthwise_conv1d(input, weight, bias, opts \\ []) do
    assert_equal_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1
      )

    num_groups = transform(Nx.shape(input), fn {_, channels, _} -> channels end)

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 1))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: num_groups
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 2-dimensional depthwise convolution.

  A depthwise convolution is essentially just a grouped convolution
  where the number of groups is equal to the number of input channels.
  """
  @doc type: :convolutional
  defn depthwise_conv2d(input, weight, bias, opts \\ []) do
    assert_equal_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1
      )

    num_groups = transform(Nx.shape(input), fn {_, channels, _, _} -> channels end)

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 2))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: num_groups
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 3-dimensional depthwise convolution.

  A depthwise convolution is essentially just a grouped convolution
  where the number of groups is equal to the number of input channels.
  """
  @doc type: :convolutional
  defn depthwise_conv3d(input, weight, bias, opts \\ []) do
    assert_equal_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1
      )

    num_groups = transform(Nx.shape(input), fn {_, channels, _, _, _} -> channels end)

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 3))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation],
      feature_group_size: num_groups
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 2-dimensional separable depthwise
  convolution.
  """
  @doc type: :convolutional
  defn separable_conv2d(input, k1, k2, b1, b2, opts \\ []) do
    input
    |> depthwise_conv2d(k1, b1, opts)
    |> depthwise_conv2d(k2, b2, opts)
  end

  @doc """
  Functional implementation of a 2-dimensional separable depthwise
  convolution.
  """
  @doc type: :convolutional
  defn separable_conv3d(input, k1, k2, k3, b1, b2, b3, opts \\ []) do
    input
    |> depthwise_conv3d(k1, b1, opts)
    |> depthwise_conv3d(k2, b2, opts)
    |> depthwise_conv3d(k3, b3, opts)
  end

  @doc """
  Functional implementation of 1-dimensional max pooling.

  Pooling is applied to the spatial dimension of the input tensor.

  ## Examples

      iex> t = Nx.tensor([[
      ...> [0.051500000059604645, -0.7042999863624573, -0.32899999618530273],
      ...> [-0.37130001187324524, 1.6191999912261963, -0.11829999834299088],
      ...> [0.7099999785423279, 0.7282999753952026, -0.18639999628067017]]], type: {:f, 32})
      iex> Axon.Layers.max_pool1d(t, kernel_size: 2)
      #Nx.Tensor<
        f32[1][3][2]
        [
          [
            [0.051500000059604645, -0.32899999618530273],
            [1.6191999912261963, 1.6191999912261963],
            [0.7282999753952026, 0.7282999753952026]
          ]
        ]
      >
  """
  @doc type: :pooling
  defn max_pool1d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 1))
    opts = transform(opts, &Keyword.delete(&1, :kernel_size))

    input
    |> Nx.window_max(window_dimensions, opts)
  end

  @doc """
  Functional implementation of 1-dimensional max pooling.

  Pooling is applied to the spatial dimension of the input tensor.
  """
  @doc type: :pooling
  defn max_pool2d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 2))
    opts = transform(opts, &Keyword.delete(&1, :kernel_size))

    input
    |> Nx.window_max(window_dimensions, opts)
  end

  @doc """
  Functional implementation of 1-dimensional max pooling.

  Pooling is applied to the spatial dimension of the input tensor.
  """
  @doc type: :pooling
  defn max_pool3d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 3))
    opts = transform(opts, &Keyword.delete(&1, :kernel_size))

    input
    |> Nx.window_max(window_dimensions, opts)
  end

  @doc """
  Functional implementation of 1-dimensional average pooling.

  Pooling is applied to the spatial dimension of the input tensor.
  """
  @doc type: :pooling
  defn avg_pool1d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 1))
    opts = transform(opts, &Keyword.delete(&1, :kernel_size))

    input
    |> Nx.window_mean(window_dimensions, opts)
  end

  @doc """
  Functional implementation of 1-dimensional average pooling.

  Pooling is applied to the spatial dimension of the input tensor.
  """
  @doc type: :pooling
  defn avg_pool2d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 2))
    opts = transform(opts, &Keyword.delete(&1, :kernel_size))

    input
    |> Nx.window_mean(window_dimensions, opts)
  end

  @doc """
  Functional implementation of 1-dimensional average pooling.

  Pooling is applied to the spatial dimension of the input tensor.
  """
  @doc type: :pooling
  defn avg_pool3d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 3))
    opts = transform(opts, &Keyword.delete(&1, :kernel_size))

    input
    |> Nx.window_mean(window_dimensions, opts)
  end

  @doc """
  Functional implementation of 1-dimensional power average pooling.

  Pooling is applied to the spatial dimension of the input tensor.

  ## Examples

      iex> t = Nx.tensor([[[0.9450, 0.4684, 1.8146], [1.2663, 0.4354, -0.0781], [-0.4759, 0.3251, 0.8742]]], type: {:f, 32})
      iex> Axon.Layers.lp_pool1d(t, kernel_size: 2, norm: 2)
      #Nx.Tensor<
        f32[1][3][2]
        [
          [
            [1.0547149181365967, 1.8740788698196411],
            [1.3390626907348633, 0.4423491656780243],
            [0.5763426423072815, 0.9326926469802856]
          ]
        ]
      >
  """
  @doc type: :pooling
  defn lp_pool1d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, norm: 1, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 1))
    norm = opts[:norm]

    opts =
      opts
      |> transform(&Keyword.delete(&1, :kernel_size))
      |> transform(&Keyword.delete(&1, :norm))

    input
    |> Nx.power(norm)
    |> Nx.window_sum(window_dimensions, opts)
    |> Nx.power(Nx.divide(Nx.tensor(1, type: Nx.type(input)), norm))
  end

  @doc """
  Functional implementation of 2-dimensional power average pooling.

  Pooling is applied to the spatial dimension of the input tensor.
  """
  @doc type: :pooling
  defn lp_pool2d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, norm: 1, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 2))
    norm = opts[:norm]

    opts =
      opts
      |> transform(&Keyword.delete(&1, :kernel_size))
      |> transform(&Keyword.delete(&1, :norm))

    input
    |> Nx.power(norm)
    |> Nx.window_sum(window_dimensions, opts)
    |> Nx.power(Nx.divide(1, norm))
  end

  @doc """
  Functional implementation of 1-dimensional power average pooling.

  Pooling is applied to the spatial dimension of the input tensor.
  """
  @doc type: :pooling
  defn lp_pool3d(input, opts \\ []) do
    opts =
      keyword!(
        opts,
        [:kernel_size, norm: 1, strides: 1, padding: :valid, window_dilations: 1]
      )

    window_dimensions = transform(opts[:kernel_size], &pool_window_size(&1, 1))
    norm = opts[:norm]

    opts =
      opts
      |> transform(&Keyword.delete(&1, :kernel_size))
      |> transform(&Keyword.delete(&1, :norm))

    input
    |> Nx.power(norm)
    |> Nx.window_sum(window_dimensions, opts)
    |> Nx.power(Nx.divide(1, norm))
  end

  @doc """
  Functional implementation of 1-dimensional adaptive average pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  @doc type: :pooling
  defn adaptive_avg_pool1d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides =
      transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 1))

    window_dimensions =
      transform(
        {Nx.shape(input), window_strides, opts[:output_size]},
        &adaptive_pool_window_size(&1, 1)
      )

    input
    |> Nx.window_mean(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 2-dimensional adaptive average pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  @doc type: :pooling
  defn adaptive_avg_pool2d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides =
      transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 2))

    window_dimensions =
      transform(
        {Nx.shape(input), window_strides, opts[:output_size]},
        &adaptive_pool_window_size(&1, 2)
      )

    input
    |> Nx.window_mean(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 3-dimensional adaptive average pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  @doc type: :pooling
  defn adaptive_avg_pool3d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides =
      transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 3))

    window_dimensions =
      transform(
        {Nx.shape(input), window_strides, opts[:output_size]},
        &adaptive_pool_window_size(&1, 3)
      )

    input
    |> Nx.window_mean(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 1-dimensional adaptive max pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  @doc type: :pooling
  defn adaptive_max_pool1d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides =
      transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 1))

    window_dimensions =
      transform(
        {Nx.shape(input), window_strides, opts[:output_size]},
        &adaptive_pool_window_size(&1, 1)
      )

    input
    |> Nx.window_max(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 2-dimensional adaptive max pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  @doc type: :pooling
  defn adaptive_max_pool2d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides =
      transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 2))

    window_dimensions =
      transform(
        {Nx.shape(input), window_strides, opts[:output_size]},
        &adaptive_pool_window_size(&1, 2)
      )

    input
    |> Nx.window_max(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 3-dimensional adaptive max pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  @doc type: :pooling
  defn adaptive_max_pool3d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides =
      transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 3))

    window_dimensions =
      transform(
        {Nx.shape(input), window_strides, opts[:output_size]},
        &adaptive_pool_window_size(&1, 3)
      )

    input
    |> Nx.window_max(window_dimensions, padding: :valid, strides: window_strides)
  end

  ## Normalization

  @doc """
  Functional implementation of batch normalization.

  Mean and variance need to be calculated separately because
  this implementation is stateless.
  """
  @doc type: :normalization
  defn batch_norm(input, mean, variance, gamma, bias, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-5)

    scale =
      variance
      |> Nx.add(opts[:epsilon])
      |> Nx.rsqrt()
      |> Nx.multiply(gamma)

    input
    |> Nx.subtract(mean)
    |> Nx.multiply(scale)
    |> Nx.add(bias)
  end

  @doc ~S"""
  Functional implementation of layer normalization.

  $$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$
  """
  @doc type: :normalization
  defn layer_norm(input, gamma, bias, opts \\ []) do
    opts = keyword!(opts, epsilon: Nx.tensor(1.0e-6, type: Nx.type(input)))

    mean = Nx.mean(input, axes: [-1], keep_axes: true)
    mean_of_squares = Nx.mean(Nx.power(input, 2), axes: [-1], keep_axes: true)
    var = mean_of_squares - Nx.power(mean, 2)
    mul = gamma * Nx.rsqrt(var + opts[:epsilon])

    (input - mean) * mul + bias
  end

  @doc """
  Functional implementation of group normalization.
  """
  @doc type: :normalization
  defn group_norm(input, gamma, bias, opts \\ []) do
    opts = keyword!(opts, [:group_size, epsilon: Nx.tensor(1.0e-6, type: Nx.type(input))])

    group_shape =
      transform(
        {Nx.shape(input), opts[:group_size]},
        fn {shape, group_size} ->
          channels = :erlang.element(shape, 2)
          num_groups = div(channels, group_size)

          Tuple.delete_at(shape, 1)
          |> put_elem(1, num_groups)
          |> Tuple.insert_at(2, group_size)
        end
      )

    x = Nx.reshape(input, group_shape)

    reduction_axes =
      transform(Nx.rank(x), fn rank -> for(i <- 1..(rank - 2), do: i) ++ [rank - 1] end)

    mean = Nx.mean(x, axes: reduction_axes, keep_dims: true)
    mean_of_squares = Nx.mean(Nx.power(x, 2), axes: reduction_axes, keep_dims: true)
    var = mean_of_squares - Nx.power(mean, 2)
    x = (x - mean) * Nx.rsqrt(var + opts[:epsilon])

    Nx.reshape(x, Nx.shape(input)) * gamma + bias
  end

  @doc """
  Functional implementation of instance normalization.
  """
  @doc type: :normalization
  defn instance_norm(input, mean, variance, gamma, bias, opts \\ []) do
    opts = keyword!(opts, epsilon: 1.0e-6)

    scale =
      variance
      |> Nx.add(opts[:epsilon])
      |> Nx.rsqrt()
      |> Nx.multiply(gamma)

    input
    |> Nx.subtract(mean)
    |> Nx.multiply(scale)
    |> Nx.add(bias)
  end

  ## Stochastic

  # TODO: Manage the state of these RNGs

  @doc ~S"""
  Functional implementation of a dropout layer.

  Applies a mask to some elements of the input tensor
  with probability `rate` and scales the input tensor
  by a factor of $\frac{1}{1 - rate}$.
  """
  @doc type: :dropout
  defn dropout(input, opts \\ []) do
    opts = keyword!(opts, [:rate, noise_shape: Nx.shape(input)])
    keep_prob = Nx.tensor(1, type: Nx.type(input)) - opts[:rate]
    mask = Nx.less(Nx.random_uniform(opts[:noise_shape], type: Nx.type(input)), keep_prob)
    Nx.select(mask, input / keep_prob, Nx.tensor(0, type: Nx.type(input)))
  end

  @doc """
  Functional implementation of a 1-dimensional spatial
  dropout layer.

  Applies a mask to entire 1-D feature maps instead of individual
  elements.
  """
  @doc type: :dropout
  defn spatial_dropout1d(input, opts \\ []) do
    opts = keyword!(opts, :rate)
    noise_shape = transform(Nx.shape(input), &spatial_dropout_noise_shape/1)
    dropout(input, rate: opts[:rate], noise_shape: noise_shape)
  end

  @doc """
  Functional implementation of a 2-dimensional spatial
  dropout layer.

  Applies a mask to entire 2-D feature maps instead of individual
  elements.
  """
  @doc type: :dropout
  defn spatial_dropout2d(input, opts \\ []) do
    opts = keyword!(opts, :rate)
    noise_shape = transform(Nx.shape(input), &spatial_dropout_noise_shape/1)
    dropout(input, rate: opts[:rate], noise_shape: noise_shape)
  end

  @doc """
  Functional implementation of a 3-dimensional spatial
  dropout layer.

  Applies a mask to entire 3-D feature maps instead of individual
  elements.
  """
  @doc type: :dropout
  defn spatial_dropout3d(input, opts \\ []) do
    opts = keyword!(opts, :rate)
    noise_shape = transform(Nx.shape(input), &spatial_dropout_noise_shape/1)
    dropout(input, rate: opts[:rate], noise_shape: noise_shape)
  end

  @doc """
  Functional implementation of an alpha dropout layer.
  """
  @doc type: :dropout
  defn alpha_dropout(input, rate) do
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
  """
  @doc type: :dropout
  defn feature_alpha_dropout(input, opts \\ []) do
    opts = keyword!(opts, [:rate])
    noise_shape = transform(Nx.shape(input), &spatial_dropout_noise_shape/1)
    keep_prob = 1 - opts[:rate]
    mask = Nx.less(Nx.random_uniform(noise_shape, type: Nx.type(input)), keep_prob)
    Nx.select(mask, input / keep_prob, Nx.negate(Axon.Activations.selu(input)))
  end

  ## Attention

  @doc """
  Functional implementation of dot product attention.
  """
  @doc type: :attention
  defn dot_product_attention(query, key, value, bias, opts \\ []) do
    assert_equal_rank!(key, query)
    assert_equal_rank!(key, value)
    opts = keyword!(opts, axes: [])

    depth = axis_size(query, -1)
    n = Nx.rank(key)

    axes = opts[:axes]

    batch_dims =
      transform({n, axes}, fn {n, axes} ->
        List.to_tuple(Enum.to_list(1..n) -- ([axes] ++ [n - 1]))
      end)

    qk_perm =
      transform({batch_dims, axes, n}, fn {batch_dims, axes, n} ->
        List.flatten([batch_dims, axes, n - 1])
      end)

    key = Nx.transpose(key, qk_perm)
    query = Nx.transpose(query, qk_perm)

    v_perm =
      transform({batch_dims, axes, n}, fn {batch_dims, axes, n} ->
        List.flatten([batch_dims, n - 1, axes])
      end)

    value = Nx.transpose(value, v_perm)

    query =
      query
      |> Nx.divide(Nx.sqrt(depth))

    # TODO: Add batch dims
    attn_weights =
      query
      |> Nx.dot([n - 1], key, [n - 1])
      |> Nx.add(bias)

    norm_dims =
      transform({Nx.rank(attn_weights), axes}, fn {n, axes} ->
        Enum.to_list((n - length(axes))..n)
      end)

    attn_weights =
      attn_weights
      |> logsumexp(axes: norm_dims, keep_axes: true)
      |> Nx.negate()
      |> Nx.add(attn_weights)
      |> Nx.exp()

    # TODO: Dropout

    v_contracting_dims =
      transform({Nx.rank(value), axes}, fn {n, axes} -> Enum.to_list((n - length(axes))..n) end)

    # TODO: More batch dims
    y =
      attn_weights
      |> Nx.dot(norm_dims, value, v_contracting_dims)

    perm_inv = invert_permutation(qk_perm)
    Nx.transpose(y, perm_inv)
  end

  ## Shape

  @doc """
  Flattens input to shape of `{batch, units}` by folding outer
  dimensions.
  """
  defn flatten(x) do
    new_shape = transform(Nx.shape(x),
      fn shape ->
        batch_size = elem(shape, 0)
        new_units =
          shape
          |> Tuple.delete_at(0)
          |> Nx.size()
        {batch_size, new_units}
      end
    )

    Nx.reshape(x, new_shape)
  end

  ## Helpers

  # `window_x` functions expect a window which matches the
  # rank of the input shape. For basic pooling we don't pool
  # across batch or channel dimensions, so we just specify
  # a size of `1` for each of those
  defp pool_window_size(w, spatial_rank) do
    spatial_dims =
      case w do
        x when is_integer(x) ->
          List.duplicate(x, spatial_rank)

        x when is_tuple(x) ->
          Tuple.to_list(x)

        x ->
          raise ArgumentError,
                "expected pool window to be tuple or integer" <>
                  " , got #{inspect(x)}"
      end

    List.to_tuple([1, 1 | spatial_dims])
  end

  # Adaptive pooling functions adapt the strides of the window
  # according to:
  # stride = div(input, output)
  # This preserves the size of the channel/batch dimension
  defp adaptive_pool_window_strides({{input_shape, output_spatial}}, spatial_rank) do
    input_spatial =
      input_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)
      |> Tuple.to_list()

    output_spatial =
      case output_spatial do
        x when is_integer(x) ->
          List.duplicate(x, spatial_rank)

        x when is_tuple(x) ->
          Tuple.to_list(x)

        x ->
          raise ArgumentError,
                "expected output spatial dimensions to be tuple" <>
                  " or integer, got #{inspect(x)}"
      end

    strides =
      output_spatial
      |> Tuple.to_list()
      |> Enum.zip(input_spatial)
      |> Enum.map(fn {input, output} -> div(input, output) end)

    [1, 1 | strides]
  end

  # Adaptive pooling functions adopt the size of the window
  # according to:
  # size = input_size - (output_size - 1) * stride
  # This preserves the size of the channel/batch dimension
  defp adaptive_pool_window_size({{input_shape, [_, _, stride], output_spatial}}, spatial_rank) do
    input_spatial =
      input_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)
      |> Tuple.to_list()

    output_spatial =
      case output_spatial do
        x when is_integer(x) ->
          List.duplicate(x, spatial_rank)

        x when is_tuple(x) ->
          Tuple.to_list(x)

        x ->
          raise ArgumentError,
                "expected output spatial dimensions to be tuple" <>
                  " or integer, got #{inspect(x)}"
      end

    zip_all = [input_spatial, output_spatial, stride]

    output_size =
      zip_all
      |> Enum.zip()
      |> Enum.map(fn {input, output, s} -> input - (output - 1) * s end)

    List.to_tuple([1, 1 | output_size])
  end

  # In order to effectively broadcast, we need to expand
  # the dimensions of the bias term in convolutions - if
  # the input bias shape is a vector, otherwise we'll just
  # attempt to let it broadcast itself
  defp conv_bias_reshape(input_shape, spatial_rank) do
    case input_shape do
      {} ->
        {}

      {shape} ->
        spatial_dims = List.duplicate(1, spatial_rank)
        List.to_tuple([1, shape | spatial_dims])

      shape when is_tuple(shape) ->
        shape
    end
  end

  # Spatial dropout shapes are broadcasted across feature
  # channels, so we set the channel size to 1 and preserve
  # the spatial dimensions
  defp spatial_dropout_noise_shape(input_shape) do
    :erlang.setelement(2, input_shape, 1)
  end

  # Fractionally strided convolution (transposed convolution)
  # by padding the input
  defp conv_transpose_padding({kernel_shape, kernel_dilation, strides, padding})
       when padding in [:valid, :same] do
    kernel_spatial_dims =
      kernel_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    kernel_dilation =
      if is_list(kernel_dilation),
        do: kernel_dilation,
        else: List.duplicate(kernel_dilation, tuple_size(kernel_spatial_dims))

    effective_kernel_size =
      kernel_spatial_dims
      |> Tuple.to_list()
      |> Enum.zip(kernel_dilation)
      |> Enum.map(fn {k, r} -> (k - 1) * r + 1 end)

    case padding do
      :valid ->
        effective_kernel_size
        |> Enum.zip(strides)
        |> Enum.map(fn {k, s} ->
          pad_len = k + s - 2 + max(k - s, 0)
          pad_a = k - 1
          {pad_a, pad_len - pad_a}
        end)

      :same ->
        effective_kernel_size
        |> Enum.zip(strides)
        |> Enum.map(fn {k, s} ->
          pad_len = k + s - 2

          pad_a =
            if s > k - 1 do
              k - 1
            else
              ceil(pad_len / 2)
            end

          {pad_a, pad_len - pad_a}
        end)
    end
  end

  defp conv_transpose_padding({_, _, _, padding}), do: padding
end
