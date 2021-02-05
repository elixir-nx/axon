defmodule Axon.Layers do
  @moduledoc """
  Functional implementations of common neural network layers.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` backend.
  """

  import Nx.Defn

  # TODO: Configuration option
  @default_defn_compiler {Nx.Defn, max_float_type: {:f, 32}}

  ## Linear

  @doc ~S"""
  Functional implementation of a dense layer.

  Linear transformation of the input such that:

  $$y = xW^T + b$$

  ## Parameter Shapes

    * `input` - `{batch_size, ..., input_features}`
    * `weight` - `{input_features, output_features}`
    * `bias` - `{output_features}`

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

    * `:strides` - kernel strides. Can be a scalar or a tuple
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
        strides: {1},
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv1d_bias_reshape/1)

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
        strides: {1, 1},
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv2d_bias_reshape/1)

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
        strides: {1, 1, 1},
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1
      )

    bias_reshape = transform(Nx.shape(bias), &conv3d_bias_reshape/1)

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
      keyword!(opts,
        [:kernel_size,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
      keyword!(opts,
        [:kernel_size,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
      keyword!(opts,
        [:kernel_size,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
      keyword!(opts,
        [:kernel_size,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
      keyword!(opts,
        [:kernel_size,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
      keyword!(opts,
        [:kernel_size,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
      keyword!(opts,
        [:kernel_size,
        norm: 1,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
  Functional implementation of 2-dimensional power average pooling.

  Pooling is applied to the spatial dimension of the input tensor.
  """
  defn lp_pool2d(input, opts \\ []) do
    opts =
      keyword!(opts,
        [:kernel_size,
        norm: 1,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
      keyword!(opts,
        [:kernel_size,
        norm: 1,
        strides: 1,
        padding: :valid,
        window_dilations: 1]
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
  Functional implementation of a 1-dimensional transposed convolution.

  Also known as fractionally strided convolutions.

  ## Options

    * `:strides` - kernel strides. Can be a scalar or a tuple
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

    * `:output_padding` - padding configuration applied to the output
      of the transposed convolution. Must be a valid padding configuration
      as a list of `{edge_low, interior, edge_high}` for each spatial
        dimension in the output.
  """
  defn conv_transpose1d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: {1},
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1,
        output_padding: [{0, 0, 0}]
      )

    output_padding_config = transform(opts[:output_padding], &conv_transpose_padding(&1, 1))
    permutation = transform(Nx.axes(weight), &conv_transpose_permutation/1)
    transposed_kernel = Nx.transpose(weight, axes: permutation)

    input
    |> Nx.conv(transposed_kernel,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
    )
    |> Nx.pad(0, output_padding_config)
    |> Nx.add(bias)
  end

  @doc """
  Functional implementation of a 2-dimensional transposed convolution.

  Also known as fractionally strided convolutions.

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

    * `:output_padding` - padding configuration applied to the output
      of the transposed convolution. Must be a valid padding configuration
      as a list of `{edge_low, interior, edge_high}` for each spatial
      dimension in the output.
  """
  defn conv_transpose2d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: {1, 1, 1},
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1,
        output_padding: [{0, 0, 0}, {0, 0, 0}]
      )

    output_padding_config = transform(opts[:output_padding], &conv_transpose_padding(&1, 2))
    permutation = transform(Nx.axes(weight), &conv_transpose_permutation/1)
    transposed_kernel = Nx.transpose(weight, axes: permutation)

    input
    |> Nx.conv(transposed_kernel,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
    )
    |> Nx.pad(0, output_padding_config)
    |> Nx.add(bias)
  end

  @doc """
  Functional implementation of a 3-dimensional transposed convolution.

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

    * `:output_padding` - padding configuration applied to the output
      of the transposed convolution. Must be a valid padding configuration
      as a list of `{edge_low, interior, edge_high}` for each spatial
      dimension in the output.
  """
  defn conv_transpose3d(input, weight, bias, opts \\ []) do
    opts =
      keyword!(opts,
        strides: {1, 1, 1},
        padding: :valid,
        input_dilation: 1,
        kernel_dilation: 1,
        groups: 1,
        output_padding: [{0, 0, 0}, {0, 0, 0}]
      )

    output_padding_config = transform(opts[:output_padding], &conv_transpose_padding(&1, 1))
    permutation = transform(Nx.axes(weight), &conv_transpose_permutation/1)
    transposed_kernel = Nx.transpose(weight, axes: permutation)

    input
    |> Nx.conv(transposed_kernel,
      strides: opts[:strides],
      padding: opts[:padding],
      input_dilation: opts[:input_dilation],
      kernel_dilation: opts[:kernel_dilation]
    )
    |> Nx.pad(0, output_padding_config)
    |> Nx.add(bias)
  end

  ## Normalization

  @doc ~S"""
  Functional implementation of layer normalization.

  $$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$$
  """
  defn layer_norm(input, bias, gamma, opts \\ []) do
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
  defn group_norm(input, bias, gamma, opts \\ []) do
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
  defn dropout(input, rate) do
    keep_prob = Nx.tensor(1, type: Nx.type(input)) - rate
    mask = Nx.less(Nx.random_uniform(Nx.shape(input), type: Nx.type(input)), keep_prob)
    Nx.select(mask, input / keep_prob, Nx.tensor(0, type: Nx.type(input)))
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

  # Helpers

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

  # TODO: This should probably be generalized
  defp conv_transpose_padding([{_, _, _} = spatial], 1), do: [{0, 0, 0}, {0, 0, 0}, spatial]

  defp conv_transpose_padding([{_, _, _} = s1, {_, _, _} = s2], 2),
    do: [{0, 0, 0}, {0, 0, 0}, s1, s2]

  defp conv_transpose_padding([{_, _, _} = s1, {_, _, _} = s2, {_, _, _} = s3], 3),
    do: [{0, 0, 0}, {0, 0, 0}, s1, s2, s3]

  defp conv_transpose_padding(padding_config, rank),
    do:
      raise(
        ArgumentError,
        "invalid output padding configuration #{inspect(padding_config)}" <>
          " for #{rank}-d transposed convolution, you must specify the" <>
          " padding configuration for each output spatial dimension as" <>
          " a list of {edge_low, interior, edge_high} values"
      )

  defp conv_transpose_permutation([0, 1, 2]), do: [1, 0, 2]
  defp conv_transpose_permutation([0, 1, 2, 3]), do: [1, 0, 3, 2]
  defp conv_transpose_permutation([0, 1, 2, 3, 4]), do: [1, 0, 4, 3, 2]

  defp conv1d_bias_reshape({}), do: {}
  defp conv1d_bias_reshape({shape}), do: {1, shape, 1}

  defp conv2d_bias_reshape({}), do: {}
  defp conv2d_bias_reshape({shape}), do: {1, shape, 1, 1}

  defp conv3d_bias_reshape({}), do: {}
  defp conv3d_bias_reshape({shape}), do: {1, shape, 1, 1, 1}
end
