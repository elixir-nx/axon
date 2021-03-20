defmodule Axon.Shape do
  @moduledoc false

  # Collection of shape calculations for calculating the
  # output and trainable parameter shapes for high-level
  # layers.
  #
  # `nil` is often used as a stand-in for unknown batch
  # size, so each of these methods must account for that.

  ## Linear

  @doc """
  Calculates the shape of a dense kernel given the input
  shape and output units.

  ## Examples

      iex> Axon.Shape.dense_kernel({nil, 784}, 128)
      {784, 128}

      iex> Axon.Shape.dense_kernel({nil, 128}, 256)
      {128, 256}

      iex> Axon.Shape.dense_kernel({nil, 3, 256, 256}, 128)
      {256, 128}
  """
  def dense_kernel(input_shape, units) do
    {elem(input_shape, Nx.rank(input_shape) - 1), units}
  end

  @doc """
  Calculates the shape of a dense bias given the input
  shape and output units.

  ## Examples

      iex> Axon.Shape.dense_bias({nil, 784}, 128)
      {1, 128}

      iex> Axon.Shape.dense_bias({nil, 128}, 256)
      {1, 256}

      iex> Axon.Shape.dense_bias({nil, 3, 256, 256}, 128)
      {1, 128}
  """
  def dense_bias(_input_shape, units) do
    {1, units}
  end

  @doc """
  Calculates the output shape of a dense layer given the
  input shape and output units.

  ## Examples

      iex> Axon.Shape.dense({nil, 784}, 128)
      {nil, 128}

      iex> Axon.Shape.dense({nil, 256}, 512)
      {nil, 512}

      iex> Axon.Shape.dense({nil, 128}, 128)
      {nil, 128}
  """
  def dense(input_shape, units) do
    {elem(input_shape, 0), units}
  end

  ## Conv

  @doc """
  Calculates the shape of a convolution kernel given the
  input shape, output filters, and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.conv_kernel({nil, 3, 224, 224}, 32, {3, 3})
      {32, 3, 3, 3}

      iex> Axon.Shape.conv_kernel({nil, 3, 28}, 64, {2})
      {64, 3, 2}

      iex> Axon.Shape.conv_kernel({nil, 1, 32, 32, 10}, 32, {2, 1, 3})
      {32, 1, 2, 1, 3}

  ### Error cases

      iex> Axon.Shape.conv_kernel({nil, 1, 28, 28}, 32, {2})
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (2)
  """
  def conv_kernel(input_shape, output_filters, kernel_size) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    input_channels = elem(input_shape, 1)
    List.to_tuple([output_filters, input_channels | Tuple.to_list(kernel_size)])
  end

  @doc """
  Calculates the shape of a convolution bias given the
  input shape, output filters, and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.conv_bias({nil, 3, 224, 224}, 32, {3, 3})
      {1, 32, 1, 1}

      iex> Axon.Shape.conv_bias({nil, 3, 28}, 64, {2})
      {1, 64, 1}

      iex> Axon.Shape.conv_bias({nil, 1, 32, 32, 10}, 32, {2, 1, 3})
      {1, 32, 1, 1, 1}

  ### Error cases

      iex> Axon.Shape.conv_bias({nil, 1, 28, 28}, 32, {2})
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (2)
  """
  def conv_bias(input_shape, output_filters, kernel_size) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    spatial_dims = List.duplicate(1, Nx.rank(input_shape) - 2)
    List.to_tuple([1, output_filters | spatial_dims])
  end

  @doc """
  Calculates the shape after a convolution layer with
  the given parent shape, kernel shape, strides, padding,
  input dilation and kernel dilation.
  """
  def conv(parent_shape, kernel_shape, strides, padding, input_dilation, kernel_dilation) do
    permutation = [0, 1, 2, 3]
    names = List.duplicate(nil, Nx.rank(parent_shape))

    {shape, _, _} =
      Nx.Shape.conv(
        parent_shape,
        names,
        kernel_shape,
        names,
        strides,
        padding,
        1,
        1,
        input_dilation,
        kernel_dilation,
        permutation,
        permutation,
        permutation
      )

    shape
  end

  @doc """
  Calculates the output shape after a pooling operation
  with the given parent shape, kernel size, strides, and
  padding.
  """
  def pool_output(parent_shape, kernel_size, strides, padding) do
    strides =
      if is_list(strides),
        do: [1, 1 | strides],
        else: [1, 1 | List.duplicate(strides, Nx.rank(parent_shape) - 2)]

    kernel_shape =
      kernel_size
      |> Tuple.insert_at(0, 1)
      |> Tuple.insert_at(0, 1)

    padding_config = padding_config(parent_shape, kernel_shape, padding, strides)

    padding_config = [
      {0, 0, 0},
      {0, 0, 0} | Enum.map(padding_config, fn {lo, hi} -> {lo, hi, 0} end)
    ]

    padded_input_shape = Nx.Shape.pad(parent_shape, padding_config)
    shape = Nx.Shape.window(padded_input_shape, kernel_shape, strides)

    shape
  end

  @doc """
  Calculates the shape after a flatten layer, which
  flattens the non-minibatch dimensions into a single
  dimension.

  ## Examples

      iex> Axon.Shape.flatten({nil, 1, 28, 28})
      {nil, 784}

      iex> Axon.Shape.flatten({nil, 128})
      {nil, 128}

      iex> Axon.Shape.flatten({nil, 10, 10})
      {nil, 100}

  ### Error cases

      iex> Axon.Shape.flatten({nil})
      ** (ArgumentError) expected flatten input shape to have at least rank 2, got {nil} with rank 1
  """
  def flatten(shape) do
    unless Nx.rank(shape) >= 2 do
      raise ArgumentError,
            "expected flatten input shape to have at least" <>
              " rank 2, got #{inspect(shape)} with rank #{Nx.rank(shape)}"
    end

    # Account for possibly `nil` batch dimension
    out_units =
      if elem(shape, 0) do
        div(Nx.size(shape), elem(shape, 0))
      else
        Nx.size(Tuple.delete_at(shape, 0))
      end

    {elem(shape, 0), out_units}
  end

  ## Helpers

  defp padding_config(parent_shape, kernel_shape, padding, strides) do
    spatial_parent =
      parent_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    spatial_kernel =
      kernel_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(0)

    case padding do
      :valid ->
        List.duplicate({0, 0}, Nx.rank(parent_shape) - 2)

      :same ->
        Nx.Shape.calculate_padding(spatial_parent, spatial_kernel, strides)

      config when is_list(config) ->
        config

      _ ->
        raise ArgumentError,
              "invalid padding configuration, padding must be" <>
                " :valid or :same, or a padding configuration for" <>
                " the dimensions of the input tensor"
    end
  end
end
