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
  tensor, it's size sohuld match the number of output channels from `weight`.
  """
  defn conv1d(input, weight, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :valid, dilation: 1)
    input
    |> Nx.conv(weight, strides: opts[:strides], padding: opts[:padding], dilation: opts[:dilation])
    |> Nx.add(bias)
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
  tensor, it's size sohuld match the number of output channels from `weight`.
  """
  defn conv2d(input, weight, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :valid, dilation: 1)
    input
    |> Nx.conv(weight, strides: opts[:strides], padding: opts[:padding], dilation: opts[:dilation])
    |> Nx.add(bias)
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
  """
  defn conv3d(input, weight, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :valid, dilation: 1)
    input
    |> Nx.conv(weight, strides: opts[:strides], padding: opts[:padding], dilation: opts[:dilation])
    |> Nx.add(bias)
  end

  @doc """
  Functional implementation of a 1-dimensional transposed convolution.
  """
  defn conv_transpose1d(input, weight, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :valid, dilation: 1)
    permutation = transform(Nx.axes(weight), &conv_transpose_permutation/1)
    transposed_kernel = Nx.transpose(weight, axes: permutation)
    input
    |> Nx.conv(transposed_kernel, strides: opts[:strides], padding: opts[:padding], dilation: opts[:dilation])
    |> Nx.add(bias)
  end

  @doc """
  Functional implementation of a 2-dimensional transposed convolution.
  """
  defn conv_transpose2d(input, weight, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :valid, dilation: 1)
    permutation = transform(Nx.axes(weight), &conv_transpose_permutation/1)
    transposed_kernel = Nx.transpose(weight, axes: permutation)
    input
    |> Nx.conv(transposed_kernel, strides: opts[:strides], padding: opts[:padding], dilation: opts[:dilation])
    |> Nx.add(bias)
  end

  @doc """
  Functional implementation of a 3-dimensional transposed convolution.
  """
  defn conv_transpose3d(input, weight, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :valid, dilation: 1)
    permutation = transform(Nx.axes(weight), &conv_transpose_permutation/1)
    transposed_kernel = Nx.transpose(weight, axes: permutation)
    input
    |> Nx.conv(transposed_kernel, strides: opts[:strides], padding: opts[:padding], dilation: opts[:dilation])
    |> Nx.add(bias)
  end

  # Helpers

  # TODO: This should probably be generalized
  defp conv_transpose_permutation([0, 1, 2]), do: [1, 0, 2]
  defp conv_transpose_permutation([0, 1, 2, 3]), do: [1, 0, 3, 2]
  defp conv_transpose_permutation([0, 1, 2, 3, 4]), do: [1, 0, 4, 3, 2]

  # TODO: Maybe it makes sense to not always use an argument
  # error but instead something generic
  defp assert_rank(shape, value) do
    rank = tuple_size(shape)
    unless rank == value, do:
      raise ArgumentError, "expected input with shape #{inspect(shape)} to have rank #{value}, got #{rank}"
  end
end
