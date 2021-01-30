defmodule Axon.Layers do
  @moduledoc """
  Functional implementations of common neural network layers.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` backend.
  """

  import Nx.Defn

  @doc ~S"""
  Functional implementation of a dense layer.

  Linear transformation of the input such that:

  $$y = xW^T + b$$

  Both `input` and `weight` should be 2-dimensional tensors with
  shapes:

  ```
  input_shape = {batch_size, in_features}
  weight_shape = {in_features, out_features}
  ```

  `bias` should be a 1-dimensional tensor or a scalar. If it is
  a 1-dimensional tensor, it's size should match the last dimension
  of the input weight.

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
    transform(Nx.shape(input), &assert_rank(&1, 2))
    transform(Nx.shape(weight), &assert_rank(&1, 2))
    transform(Nx.shape(bias), &assert_rank(&1, 1))

    input
    |> Nx.dot([Nx.rank(input) - 1], weight, [0])
    |> Nx.add(bias)
  end

  @doc """
  Functional implementation of a 1-dimensional convolution.

  Both `input` and `weight` should be 3-dimensional tensors with
  shapes:

  ```
  input_shape = {batch_size, in_channels, in_spatial}
  weight_shape = {out_channels, in_channels, kernel_spatial}
  ```

  `bias` should be a 1-dimensional tensor or a scalar. If it is a 1-dimensional
  tensor, it's size should match the number of output channels from `weight`.

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

  Both `input` and `weight` should be 4-dimensional tensors with
  shapes:

  ```
  input_shape = {batch_size, in_channels, in_height, in_width}
  weight_shape = {out_channels, in_channels, kernel_height, kernel_width}
  ```

  `bias` should be a 1-dimensional tensor or a scalar. If it is a 1-dimensional
  tensor, it's size should match the number of output channels from `weight`.

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

  Both `input` and `weight` should be 5-dimensional tensors with
  shapes:

  ```
  input_shape = {batch_size, in_channels, in_temporal, in_height, in_width}
  weight_shape = {out_channels, in_channels, kernel_temporal, kernel_height, kernel_width}
  ```

  `bias` should be a 1-dimensional tensor or a scalar. If it is a 1-dimensional
  tensor, it's size should match the number of output channels from `weight`.

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
  Functional implementation of a 1-dimensional transposed convolution.

  Also known as fractionally strided convolutions.

  Both `input` and `weight` should be 3-dimensional tensors with
  shapes:

  ```
  input_shape = {batch_size, in_channels, in_spatial}
  weight_shape = {in_channels, out_channels, kernel_spatial}
  ```

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
        strides: {1, 1, 1},
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

  Both `input` and `weight` should be 4-dimensional tensors with
  shapes:

  ```
  input_shape = {batch_size, in_channels, in_height, in_width}
  weight_shape = {in_channels, out_channels, kernel_height, kernel_width}
  ```

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

  Also known as fractionally strided convolutions.

  Both `input` and `weight` should be 5-dimensional tensors with
  shapes:

  ```
  input_shape = {batch_size, in_channels, in_temporal, in_height, in_width}
  weight_shape = {in_channels, out_channels, kernel_temporal, kernel_height, kernel_width}
  ```

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

  # Helpers

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

  # TODO: Maybe it makes sense to not always use an argument
  # error but instead something generic
  defp assert_rank(shape, value) do
    rank = tuple_size(shape)

    unless rank == value,
      do:
        raise(
          ArgumentError,
          "expected input with shape #{inspect(shape)} to have rank #{value}, got #{rank}"
        )
  end
end
