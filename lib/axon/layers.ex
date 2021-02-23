defmodule Axon.Layers do
  @moduledoc """
  Functional implementations of common neural network layer
  operations.

  Layers are the building blocks of neural networks. These
  functional implementations can be used to express higher-level
  constructs using fundamental building blocks. Neural network
  layers are typically stateful with respect to their parameters.
  These implementations do not assume the responsibility of
  managing state - instead opting to delegate this responsibility
  to the caller.

  Neural networks can often be seen as a composition of functions:

      input
      |> dense(w1, b1)
      |> relu()
      |> dense(w2, b2)
      |> softmax()

  """

  import Nx.Defn
  import Axon.Shared

  ## Linear

  @doc ~S"""
  Functional implementation of a dense layer.

  Linear transformation of the input such that:

  $$y = xW^T + b$$

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
  defn conv1d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: [1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 1))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
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
  defn conv2d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: [1, 1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 2))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
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
  defn conv3d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: [1, 1, 1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv_bias_reshape(&1, 3))

    input
    |> Nx.conv(weight,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
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
  defn conv_transpose1d(input, weight, bias, opts \\ []) do
    assert_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
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
  defn conv_transpose2d(input, weight, bias, opts \\ []) do
    assert_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1, 1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
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
  defn conv_transpose3d(input, weight, bias, opts \\ []) do
    assert_rank!(input, weight)

    opts =
      keyword!(opts,
        strides: [1, 1, 1],
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
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
  defn depthwise_conv1d(input, weight, bias, opts \\ []) do
    assert_rank!(input, weight)

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
      groups: num_groups
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 2-dimensional depthwise convolution.

  A depthwise convolution is essentially just a grouped convolution
  where the number of groups is equal to the number of input channels.
  """
  defn depthwise_conv2d(input, weight, bias, opts \\ []) do
    assert_rank!(input, weight)

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
      groups: num_groups
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 3-dimensional depthwise convolution.

  A depthwise convolution is essentially just a grouped convolution
  where the number of groups is equal to the number of input channels.
  """
  defn depthwise_conv3d(input, weight, bias, opts \\ []) do
    assert_rank!(input, weight)

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
      groups: num_groups
    )
    |> Nx.add(Nx.reshape(bias, bias_reshape))
  end

  @doc """
  Functional implementation of a 2-dimensional separable depthwise
  convolution.
  """
  defn separable_conv2d(input, k1, k2, b1, b2, opts \\ []) do
    input
    |> depthwise_conv2d(k1, b1, opts)
    |> depthwise_conv2d(k2, b2, opts)
  end

  @doc """
  Functional implementation of a 2-dimensional separable depthwise
  convolution.
  """
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
  defn adaptive_avg_pool1d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides = transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 1))
    window_dimensions = transform({Nx.shape(input), window_strides, opts[:output_size]}, &adaptive_pool_window_size(&1, 1))

    input
    |> Nx.window_mean(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 2-dimensional adaptive average pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  defn adaptive_avg_pool2d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides = transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 2))
    window_dimensions = transform({Nx.shape(input), window_strides, opts[:output_size]}, &adaptive_pool_window_size(&1, 2))

    input
    |> Nx.window_mean(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 3-dimensional adaptive average pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  defn adaptive_avg_pool3d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides = transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 3))
    window_dimensions = transform({Nx.shape(input), window_strides, opts[:output_size]}, &adaptive_pool_window_size(&1, 3))

    input
    |> Nx.window_mean(window_dimensions, padding: :valid, strides: window_strides)
  end


  @doc """
  Functional implementation of 1-dimensional adaptive max pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  defn adaptive_max_pool1d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides = transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 1))
    window_dimensions = transform({Nx.shape(input), window_strides, opts[:output_size]}, &adaptive_pool_window_size(&1, 1))

    input
    |> Nx.window_max(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 2-dimensional adaptive max pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  defn adaptive_max_pool2d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides = transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 2))
    window_dimensions = transform({Nx.shape(input), window_strides, opts[:output_size]}, &adaptive_pool_window_size(&1, 2))

    input
    |> Nx.window_max(window_dimensions, padding: :valid, strides: window_strides)
  end

  @doc """
  Functional implementation of 3-dimensional adaptive max pooling.

  Adaptive pooling allows you to specify the desired output size
  of the transformed input. This will automatically adapt the
  window size and strides to obtain the desired output size.
  """
  defn adaptive_max_pool3d(input, opts \\ []) do
    opts = keyword!(opts, [:output_size])

    window_strides = transform({Nx.shape(input), opts[:output_size]}, &adaptive_pool_window_strides(&1, 3))
    window_dimensions = transform({Nx.shape(input), window_strides, opts[:output_size]}, &adaptive_pool_window_size(&1, 3))

    input
    |> Nx.window_max(window_dimensions, padding: :valid, strides: window_strides)
  end

  ## Normalization

  @doc """
  Functional implementation of batch normalization.

  Mean and variance need to be calculated separately because
  this implementation is stateless.
  """
  defn batch_norm(input, gamma, bias, mean, variance, opts \\ []) do
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

  ## Stochastic

  # TODO: Manage the state of these RNGs

  @doc ~S"""
  Functional implementation of a dropout layer.

  Applies a mask to some elements of the input tensor
  with probability `rate` and scales the input tensor
  by a factor of $\frac{1}{1 - rate}$.
  """
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
  defn spatial_dropout1d(input, opts \\ []) do
    opts = keyword!(opts, :rate)
    noise_shape = transform(Nx.shape(input), &spatial_dropout_noise_shape(&1, 1))
    dropout(input, rate: opts[:rate], noise_shape: noise_shape)
  end

  @doc """
  Functional implementation of a 2-dimensional spatial
  dropout layer.

  Applies a mask to entire 2-D feature maps instead of individual
  elements.
  """
  defn spatial_dropout2d(input, opts \\ []) do
    opts = keyword!(opts, :rate)
    noise_shape = transform(Nx.shape(input), &spatial_dropout_noise_shape(&1, 2))
    dropout(input, rate: opts[:rate], noise_shape: noise_shape)
  end

  @doc """
  Functional implementation of a 3-dimensional spatial
  dropout layer.

  Applies a mask to entire 3-D feature maps instead of individual
  elements.
  """
  defn spatial_dropout3d(input, opts \\ []) do
    opts = keyword!(opts, :rate)
    noise_shape = transform(Nx.shape(input), &spatial_dropout_noise_shape(&1, 3))
    dropout(input, rate: opts[:rate], noise_shape: noise_shape)
  end

  @doc """
  Functional implementation of an alpha dropout layer.
  """
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

  ## Attention

  ## Helpers

  # TODO: Move most of those to an `Axon.Shape` module

  # `window_x` functions expect a window which matches the
  # rank of the input shape. For basic pooling we don't pool
  # across batch or channel dimensions, so we just specify
  # a size of `1` for each of those
  defp pool_window_size({w1}, 1), do: {1, 1, w1}
  defp pool_window_size({w1, w2}, 2), do: {1, 1, w1, w2}
  defp pool_window_size({w1, w2, w3}, 3), do: {1, 1, w1, w2, w3}
  defp pool_window_size(w, 1), do: {1, 1, w}
  defp pool_window_size(w, 2), do: {1, 1, w, w}
  defp pool_window_size(w, 3), do: {1, 1, w, w, w}

  # Adaptive pooling functions adapt the strides of the window
  # according to:
  # stride = div(input, output)
  # This preserves the size of the channel/batch dimension
  defp adaptive_pool_window_strides({{_, _, input_spatial}, {output_spatial}}, 1),
    do: [1, 1, div(input_spatial, output_spatial)]
  defp adaptive_pool_window_strides({{_, _, input_spatial}, output_spatial}, 1),
    do: [1, 1, div(input_spatial, output_spatial)]

  defp adaptive_pool_window_strides({{_, _, input_height, input_width}, {output_height, output_width}}, 2),
    do: [1, 1, div(input_height, output_height), div(input_width, output_width)]
  defp adaptive_pool_window_strides({{_, _, input_height, input_width}, output_spatial}, 2),
    do: [1, 1, div(input_height, output_spatial), div(input_width, output_spatial)]

  defp adaptive_pool_window_strides({{_, _, input_height, input_width, input_temporal}, {output_height, output_width, output_temporal}}, 3),
    do: [1, 1, div(input_height, output_height), div(input_width, output_width), div(input_temporal, output_temporal)]
  defp adaptive_pool_window_strides({{_, _, input_height, input_width, input_temporal}, output_spatial}, 3),
    do: [1, 1, div(input_height, output_spatial), div(input_width, output_spatial), div(input_temporal, output_spatial)]

  # Adaptive pooling functions adopt the size of the window
  # according to:
  # size = input_size - (output_size - 1) * stride
  # This preserves the size of the channel/batch dimension
  defp adaptive_pool_window_size({{_, _, input_spatial}, [_, _, stride], {output_spatial}}, 1) do
    {1, 1, input_spatial - (output_spatial-1) * stride}
  end
  defp adaptive_pool_window_size({{_, _, input_spatial}, [_, _, stride], output_spatial}, 1) do
    {1, 1, input_spatial - (output_spatial-1) * stride}
  end

  defp adaptive_pool_window_size({{_, _, input_height, input_width}, [_, _, s1, s2], {output_height, output_width}}, 2) do
    {1, 1, input_height - (output_height-1) * s1, input_width - (output_width-1) * s2}
  end
  defp adaptive_pool_window_size({{_, _, input_height, input_width}, [_, _, s1, s2], output_spatial}, 2) do
    {1, 1, input_height - (output_spatial-1) * s1, input_width - (output_spatial-1) * s2}
  end

  defp adaptive_pool_window_size({{_, _, input_height, input_width, input_temporal}, [_, _, s1, s2, s3], {output_height, output_width, output_temporal}}, 3) do
    {1, 1, input_height - (output_height-1) * s1, input_width - (output_width-1) * s2, input_temporal - (output_temporal-1) * s3}
  end
  defp adaptive_pool_window_size({{_, _, input_height, input_width, input_temporal}, [_, _, s1, s2, s3], output_spatial}, 3) do
    {1, 1, input_height - (output_spatial-1) * s1, input_width - (output_spatial-1) * s2, input_temporal - (output_spatial-1) * s3}
  end

  # In order to effectively broadcast, we need to expand
  # the dimensions of the bias term in convolutions
  defp conv_bias_reshape({}, _), do: {}
  defp conv_bias_reshape({shape}, 1), do: {1, shape, 1}
  defp conv_bias_reshape({shape}, 2), do: {1, shape, 1, 1}
  defp conv_bias_reshape({shape}, 3), do: {1, shape, 1, 1, 1}

  # Spatial dropout shapes are broadcasted across feature
  # channels
  defp spatial_dropout_noise_shape({batch, channels, _spatial}, 1), do: {batch, channels, 1}
  defp spatial_dropout_noise_shape({batch, channels, _s1, _s2}, 2), do: {batch, channels, 1, 1}

  defp spatial_dropout_noise_shape({batch, channels, _s1, _s2, _s3}, 3),
    do: {batch, channels, 1, 1, 1}

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

    out =
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

    IO.inspect(out)
  end

  defp conv_transpose_padding({_, _, _, padding}), do: padding
end
