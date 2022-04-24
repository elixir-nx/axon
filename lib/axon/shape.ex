defmodule Axon.Shape do
  @moduledoc false

  # Collection of shape calculations for calculating the
  # output and trainable parameter shapes for high-level
  # layers.
  #
  # `nil` is often used as a stand-in for unknown batch
  # size, so each of these methods must account for that.

  @doc """
  Calculates the shape of an input layer.

  ## Examples

      iex> Axon.Shape.input({nil, 784})
      {nil, 784}

      iex> Axon.Shape.input({32, 784})
      {32, 784}

      iex> Axon.Shape.input({})
      {}

      iex> Axon.Shape.input({nil})
      {nil}

      iex> Axon.Shape.input({5})
      {5}

  ### Error cases

      iex> Axon.Shape.input(5)
      ** (ArgumentError) invalid input shape 5, input shape must be a tuple with only the leading dimension as nil, if any

      iex> Axon.Shape.input({32, nil, 28, 28})
      ** (ArgumentError) invalid input shape {32, nil, 28, 28}, input shape must be a tuple with only the leading dimension as nil, if any
  """
  def input(input_shape), do: input_shape

  @doc """
  Determines if two shapes are compatible. Shapes are compatible
  if they are equal, or if all non-nil dimensions are equal.

  ## Examples

      iex> Axon.Shape.compatible?({nil, 32}, {2, 32})
      true

      iex> Axon.Shape.compatible?({1, 32}, {2, 32})
      false

      iex> Axon.Shape.compatible?({1, 3, 2}, {3, 2})
      false
  """
  def compatible?(s1, s2) do
    if Nx.rank(s1) == Nx.rank(s2) do
      s1
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(s2))
      |> Enum.reduce(true, fn {d1, d2}, acc ->
        (acc and d1 == d2) or d1 == nil or d2 == nil
      end)
    else
      false
    end
  end

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
    unless Nx.rank(input_shape) >= 2 do
      raise ArgumentError,
            "input shape must have at least rank 2, got rank" <>
              " #{Nx.rank(input_shape)}"
    end

    {elem(input_shape, Nx.rank(input_shape) - 1), units}
  end

  @doc """
  Calculates the shape of a dense bias given the input
  shape and output units.

  ## Examples

      iex> Axon.Shape.dense_bias({nil, 784}, 128)
      {128}

      iex> Axon.Shape.dense_bias({nil, 128}, 256)
      {256}

      iex> Axon.Shape.dense_bias({nil, 3, 256, 256}, 128)
      {128}
  """
  def dense_bias(input_shape, units) do
    unless Nx.rank(input_shape) >= 2 do
      raise ArgumentError,
            "input shape must have at least rank 2, got rank" <>
              " #{Nx.rank(input_shape)}"
    end

    {units}
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

  ### Errors

      iex> Axon.Shape.dense({}, 32)
      ** (ArgumentError) input shape must have at least rank 2, got rank 0

      iex> Axon.Shape.dense({1}, 32)
      ** (ArgumentError) input shape must have at least rank 2, got rank 1
  """
  def dense(input_shape, units) do
    unless Nx.rank(input_shape) >= 2 do
      raise ArgumentError,
            "input shape must have at least rank 2, got rank" <>
              " #{Nx.rank(input_shape)}"
    end

    put_elem(input_shape, Nx.rank(input_shape) - 1, units)
  end

  @doc """
  Calculates the shape of a bilinear kernel given both input
  shapes and output units.

  ## Examples

      iex> Axon.Shape.bilinear_kernel({nil, 32}, {nil, 64}, 128)
      {128, 32, 64}

      iex> Axon.Shape.bilinear_kernel({nil, 32, 64}, {nil, 16}, 32)
      {32, 64, 16}
  """
  def bilinear_kernel(parent1, parent2, units) do
    unless Nx.rank(parent1) >= 2 and Nx.rank(parent2) >= 2 do
      raise ArgumentError,
            "input shapes must both have at least rank 2" <>
              " got ranks #{Nx.rank(parent1)} and #{Nx.rank(parent2)}"
    end

    parent1_features = elem(parent1, Nx.rank(parent1) - 1)
    parent2_features = elem(parent2, Nx.rank(parent2) - 1)
    {units, parent1_features, parent2_features}
  end

  @doc """
  Calculates the shape of a bilinear bias given both input
  shapes and output units.

  ## Examples

      iex> Axon.Shape.bilinear_bias({nil, 32}, {nil, 64}, 128)
      {128}

      iex> Axon.Shape.bilinear_bias({nil, 32, 64}, {nil, 32, 16}, 32)
      {32}
  """
  def bilinear_bias(parent1, parent2, units) do
    unless Nx.rank(parent1) >= 2 and Nx.rank(parent2) >= 2 do
      raise ArgumentError,
            "input shapes must both have at least rank 2" <>
              " got ranks #{Nx.rank(parent1)} and #{Nx.rank(parent2)}"
    end

    {units}
  end

  @doc """
  Calculates the output shape of a bilinear layer given both input
  shapes and output units.

  ## Examples

      iex> Axon.Shape.bilinear({nil, 32}, {nil, 64}, 128)
      {nil, 128}

      iex> Axon.Shape.bilinear({nil, 32, 64}, {nil, 32, 16}, 32)
      {nil, 32, 32}

      iex> Axon.Shape.bilinear({nil, 32, 64}, {16, 32, 16}, 32)
      {16, 32, 32}

  ### Errors

      iex> Axon.Shape.bilinear({32, 32}, {16, 16}, 32)
      ** (ArgumentError) all input dimensions but the last must match, got 32 and 16 for shapes {32, 32} and {16, 16}

      iex> Axon.Shape.bilinear({nil, 16, 32}, {nil, 16}, 32)
      ** (ArgumentError) input ranks must match, got 3 and 2
      
      iex> Axon.Shape.bilinear({nil, 16, 32}, {}, 32)
      ** (ArgumentError) input shapes must both have at least rank 2, got ranks 3 and 0

      iex> Axon.Shape.bilinear({nil}, {12}, 32)
      ** (ArgumentError) input shapes must both have at least rank 2, got ranks 1 and 1
  """
  def bilinear(parent1, parent2, units) do
    unless Nx.rank(parent1) >= 2 and Nx.rank(parent2) >= 2 do
      raise ArgumentError,
            "input shapes must both have at least rank 2," <>
              " got ranks #{Nx.rank(parent1)} and #{Nx.rank(parent2)}"
    end

    unless Nx.rank(parent1) == Nx.rank(parent2) do
      raise ArgumentError,
            "input ranks must match, got #{inspect(Nx.rank(parent1))}" <>
              " and #{inspect(Nx.rank(parent2))}"
    end

    parent1_without_features =
      parent1
      |> Tuple.delete_at(Nx.rank(parent1) - 1)
      |> Tuple.to_list()

    parent2_without_features =
      parent2
      |> Tuple.delete_at(Nx.rank(parent2) - 1)
      |> Tuple.to_list()

    output_shape_no_features =
      parent1_without_features
      |> Enum.zip_with(parent2_without_features, fn p1, p2 ->
        unless is_nil(p1) or is_nil(p2) or p1 == p2 do
          raise ArgumentError,
                "all input dimensions but the last must match, got #{inspect(p1)}" <>
                  " and #{inspect(p2)} for shapes #{inspect(parent1)} and #{inspect(parent2)}"
        end

        if is_nil(p1) do
          p2
        else
          p1
        end
      end)
      |> List.to_tuple()

    Tuple.append(output_shape_no_features, units)
  end

  ## Sparse

  @doc """
  Calculates the shape of an embedding kernel given input shape
  vocab size and embedding size.

  ## Examples

      iex> Axon.Shape.embedding_kernel({nil, 10}, 128, 32)
      {128, 32}

      iex> Axon.Shape.embedding_kernel({nil, 32}, 10, 10)
      {10, 10}
  """
  def embedding_kernel(_input_shape, vocab_size, embedding_size) do
    {vocab_size, embedding_size}
  end

  @doc """
  Calculates the output shape of an embedding layer given input shape
  vocab size and embedding size.

  ## Examples

      iex> Axon.Shape.embedding({nil, 10}, 128, 32)
      {nil, 10, 32}

      iex> Axon.Shape.embedding({nil, 32}, 10, 10)
      {nil, 32, 10}
  """
  def embedding(input_shape, _vocab_size, embedding_size) do
    Tuple.append(input_shape, embedding_size)
  end

  ## Conv

  @doc """
  Calculates the shape of a convolution kernel given the
  input shape, output filters, and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.conv_kernel({nil, 3, 224, 224}, 32, {3, 3}, :first)
      {32, 3, 3, 3}

      iex> Axon.Shape.conv_kernel({nil, 3, 28}, 64, {2}, :first)
      {64, 3, 2}

      iex> Axon.Shape.conv_kernel({nil, 1, 32, 32, 10}, 32, {2, 1, 3}, :first)
      {32, 1, 2, 1, 3}

      iex> Axon.Shape.conv_kernel({nil, 28, 3}, 64, {2}, :last)
      {64, 3, 2}

  ### Error cases

      iex> Axon.Shape.conv_kernel({nil, 1, 28, 28}, 32, {2}, :first)
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (2)
  """
  def conv_kernel(input_shape, output_filters, kernel_size, channels) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    List.to_tuple([output_filters, input_channels | Tuple.to_list(kernel_size)])
  end

  @doc """
  Calculates the shape of a convolution bias given the
  input shape, output filters, and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.conv_bias({nil, 3, 224, 224}, 32, {3, 3}, :first)
      {32}

      iex> Axon.Shape.conv_bias({nil, 3, 28}, 64, {2}, :first)
      {64}

      iex> Axon.Shape.conv_bias({nil, 1, 32, 32, 10}, 32, {2, 1, 3}, :first)
      {32}

      iex> Axon.Shape.conv_bias({nil, 28, 3}, 64, {2}, :last)
      {64}

  ### Error cases

      iex> Axon.Shape.conv_bias({nil, 1, 28, 28}, 32, {2}, :first)
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (2)
  """
  def conv_bias(input_shape, output_filters, kernel_size, _channels) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    {output_filters}
  end

  @doc """
  Calculates the shape after a convolution layer with
  the given parent shape, kernel shape, strides, padding,
  input dilation and kernel dilation.

  ## Examples

      iex> Axon.Shape.conv({nil, 3, 224, 224}, {64, 3, 7, 7}, [3, 3], :same, [1, 1], [1, 1], :first)
      {nil, 64, 75, 75}

      iex> Axon.Shape.conv({32, 3, 32, 32}, {64, 3, 2, 2}, [1, 1], :valid, [1, 2], [1, 1], :first)
      {32, 64, 31, 62}

      iex> Axon.Shape.conv({nil, 3, 32}, {32, 3, 2}, [1], :valid, [1], [2], :first)
      {nil, 32, 30}

      iex> Axon.Shape.conv({nil, 28, 28, 3}, {64, 3, 4, 4}, [1, 1], :valid, [1, 1], [2, 2], :last)
      {nil, 22, 22, 64}
  """
  def conv(
        parent_shape,
        kernel_shape,
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        channels
      ) do
    unless Nx.rank(parent_shape) >= 3 do
      raise ArgumentError,
            "input shape must be at least rank 3," <>
              " got rank #{Nx.rank(parent_shape)}"
    end

    permutation = for i <- 0..(Nx.rank(parent_shape) - 1), do: i

    in_out_permutation =
      if channels == :first do
        permutation
      else
        rank = tuple_size(parent_shape) - 1
        spatial = Enum.to_list(1..(rank - 1)//1)
        [0, rank | spatial]
      end

    names = List.duplicate(nil, Nx.rank(parent_shape))

    # Account for possibly nil batch dimension
    input_shape =
      if elem(parent_shape, 0) do
        parent_shape
      else
        put_elem(parent_shape, 0, 1)
      end

    {shape, _, _} =
      Nx.Shape.conv(
        input_shape,
        names,
        kernel_shape,
        names,
        strides,
        padding,
        1,
        1,
        input_dilation,
        kernel_dilation,
        in_out_permutation,
        permutation,
        in_out_permutation
      )

    put_elem(shape, 0, elem(parent_shape, 0))
  end

  @doc """
  Calculates the reshape needed to broadcast convolution bias
  over the given input shape.

  In order to effectively broadcast, we need to expand
  the dimensions of the bias term in convolutions - if
  the input bias shape is a vector, otherwise we'll just
  attempt to let it broadcast itself.
  """
  def conv_bias_reshape(input_shape, spatial_rank, channels) do
    case input_shape do
      {} ->
        {}

      {shape} ->
        spatial_dims = List.duplicate(1, spatial_rank)

        if channels == :first do
          List.to_tuple([1, shape | spatial_dims])
        else
          List.to_tuple([1 | spatial_dims] ++ [shape])
        end

      shape when is_tuple(shape) ->
        shape
    end
  end

  @doc """
  Calculates the shape after a transposed convolution layer
  with the given parent shape, kernel shape, strides, padding,
  and kernel dilation.

  ## Examples

      iex> Axon.Shape.conv_transpose({nil, 3, 3}, {6, 3, 2}, [1], :valid, [1], :first)
      {nil, 6, 4}

      iex> Axon.Shape.conv_transpose({nil, 3, 3}, {6, 3, 2}, [1], :valid, [1], :last)
      {nil, 4, 6}
  """
  def conv_transpose(parent_shape, kernel_shape, strides, padding, kernel_dilation, channels) do
    unless Nx.rank(parent_shape) >= 3 do
      raise ArgumentError,
            "input shape must be at least rank 3," <>
              " got rank #{Nx.rank(parent_shape)}"
    end

    permutation = for i <- 0..(Nx.rank(parent_shape) - 1), do: i

    in_out_permutation =
      if channels == :first do
        permutation
      else
        rank = tuple_size(parent_shape) - 1
        spatial = Enum.to_list(1..(rank - 1)//1)
        [0, rank | spatial]
      end

    names = List.duplicate(nil, Nx.rank(parent_shape))
    input_dilation = strides
    one = List.duplicate(1, Nx.rank(parent_shape) - 2)
    padding = conv_transpose_padding(kernel_shape, kernel_dilation, strides, padding)

    input_shape =
      if elem(parent_shape, 0) do
        parent_shape
      else
        put_elem(parent_shape, 0, 1)
      end

    {shape, _, _} =
      Nx.Shape.conv(
        input_shape,
        names,
        kernel_shape,
        names,
        one,
        padding,
        1,
        1,
        input_dilation,
        kernel_dilation,
        in_out_permutation,
        permutation,
        in_out_permutation
      )

    put_elem(shape, 0, elem(parent_shape, 0))
  end

  @doc """
  Calculates the padding needed for a transposed convolution.
  """
  def conv_transpose_padding(kernel_shape, kernel_dilation, strides, padding)
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

  def conv_transpose_padding(_, _, _, padding), do: padding

  @doc """
  Calculates the shape of a depthwise convolution kernel given the
  input shape, output filters, and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.depthwise_conv_kernel({nil, 3, 224, 224}, 3, {3, 3}, :first)
      {9, 1, 3, 3}

      iex> Axon.Shape.depthwise_conv_kernel({nil, 3, 28}, 2, {2}, :first)
      {6, 1, 2}

      iex> Axon.Shape.depthwise_conv_kernel({nil, 1, 32, 32, 10}, 1, {2, 1, 3}, :first)
      {1, 1, 2, 1, 3}

      iex> Axon.Shape.depthwise_conv_kernel({nil, 28, 3}, 2, {2}, :last)
      {6, 1, 2}

  ### Error cases

      iex> Axon.Shape.depthwise_conv_kernel({nil, 1, 28, 28}, 32, {2}, :first)
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (2)
  """
  def depthwise_conv_kernel(input_shape, channel_multiplier, kernel_size, channels) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    List.to_tuple([input_channels * channel_multiplier, 1 | Tuple.to_list(kernel_size)])
  end

  @doc """
  Calculates the shape of a convolution bias given the
  input shape, channel multiplier, and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.depthwise_conv_bias({nil, 3, 224, 224}, 3, {3, 3}, :first)
      {9}

      iex> Axon.Shape.depthwise_conv_bias({nil, 3, 28}, 2, {2}, :first)
      {6}

      iex> Axon.Shape.depthwise_conv_bias({nil, 1, 32, 32, 10}, 1, {2, 1, 3}, :first)
      {1}

      iex> Axon.Shape.depthwise_conv_bias({nil, 28, 3}, 2, {2}, :last)
      {6}

  ### Error cases

      iex> Axon.Shape.depthwise_conv_bias({nil, 1, 28, 28}, 2, {2}, :first)
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (2)
  """
  def depthwise_conv_bias(input_shape, channel_multiplier, kernel_size, channels) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    {input_channels * channel_multiplier}
  end

  @doc """
  Calculates the shape after a depthwise convolution layer with
  the given parent shape, kernel shape, strides, padding, input
  dilation, and kernel dilation.

  ## Examples

      iex> Axon.Shape.depthwise_conv({nil, 3, 224, 224}, {9, 1, 7, 7}, [3, 3], :same, [1, 1], [1, 1], :first)
      {nil, 9, 75, 75}

      iex> Axon.Shape.depthwise_conv({32, 3, 32, 32}, {9, 1, 2, 2}, [1, 1], :valid, [1, 2], [1, 1], :first)
      {32, 9, 31, 62}

      iex> Axon.Shape.depthwise_conv({nil, 3, 32}, {9, 1, 2}, [1], :valid, [1], [2], :first)
      {nil, 9, 30}
  """
  def depthwise_conv(
        parent_shape,
        kernel_shape,
        strides,
        padding,
        input_dilation,
        kernel_dilation,
        channels
      ) do
    unless Nx.rank(parent_shape) >= 3 do
      raise ArgumentError,
            "input shape must be at least rank 3," <>
              " got rank #{Nx.rank(parent_shape)}"
    end

    permutation = for i <- 0..(Nx.rank(parent_shape) - 1), do: i

    in_out_permutation =
      if channels == :first do
        permutation
      else
        rank = tuple_size(parent_shape) - 1
        spatial = Enum.to_list(1..(rank - 1)//1)
        [0, rank | spatial]
      end

    names = List.duplicate(nil, Nx.rank(parent_shape))

    # Account for possibly nil batch dimension
    input_shape =
      if elem(parent_shape, 0) do
        parent_shape
      else
        put_elem(parent_shape, 0, 1)
      end

    input_channels =
      if channels == :first do
        elem(parent_shape, 1)
      else
        elem(parent_shape, tuple_size(parent_shape) - 1)
      end

    {shape, _, _} =
      Nx.Shape.conv(
        input_shape,
        names,
        kernel_shape,
        names,
        strides,
        padding,
        input_channels,
        1,
        input_dilation,
        kernel_dilation,
        in_out_permutation,
        permutation,
        in_out_permutation
      )

    put_elem(shape, 0, elem(parent_shape, 0))
  end

  @doc """
  Calculates the shape of a 2d depthwise separable convolution
  kernel given the input shape, channel multiplier, kernel size
  and parameter number.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.separable_conv2d_kernel({nil, 3, 32, 32}, 3, {3, 3}, 1, :first)
      {9, 1, 3, 1}

      iex> Axon.Shape.separable_conv2d_kernel({nil, 3, 32, 32}, 3, {3, 3}, 2, :first)
      {9, 1, 1, 3}

  ### Error cases

      iex> Axon.Shape.separable_conv2d_kernel({nil, 1, 28, 28}, 2, {2}, 1, :first)
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (2)

      iex> Axon.Shape.separable_conv2d_kernel({nil, 1, 28, 28}, 2, {2, 2}, 3, :first)
      ** (ArgumentError) invalid kernel number
  """
  def separable_conv2d_kernel(input_shape, channel_multiplier, kernel_size, num, channels) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    idx =
      if channels == :first do
        1
      else
        tuple_size(input_shape) - 1
      end

    cond do
      num == 1 ->
        {elem(input_shape, idx) * channel_multiplier, 1, elem(kernel_size, 0), 1}

      num == 2 ->
        {elem(input_shape, idx) * channel_multiplier, 1, 1, elem(kernel_size, 1)}

      true ->
        raise ArgumentError, "invalid kernel number"
    end
  end

  @doc """
  Calculates the shape of a depthwise separable convolution
  bias given the input shape, channel multiplier and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.separable_conv2d_bias({nil, 3, 32, 32}, 3, {3, 3}, :first)
      {9}

      iex> Axon.Shape.separable_conv2d_bias({nil, 3, 32, 32}, 4, {3, 3}, :first)
      {12}

  ### Error cases

      iex> Axon.Shape.separable_conv2d_bias({nil, 1, 28, 28}, 2, {2}, :first)
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (2)
  """
  def separable_conv2d_bias(input_shape, channel_multiplier, kernel_size, channels) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    {input_channels * channel_multiplier}
  end

  @doc """
  Calculates the shape of a 3-d depthwise separable convolution
  kernel given the input shape, channel multiplier, kernel size,
  and parameter number.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.separable_conv3d_kernel({nil, 3, 32, 32, 3}, 3, {3, 3, 3}, 1, :first)
      {9, 1, 3, 1, 1}

      iex> Axon.Shape.separable_conv3d_kernel({nil, 3, 32, 32, 3}, 4, {3, 3, 3}, 2, :first)
      {12, 1, 1, 3, 1}

      iex> Axon.Shape.separable_conv3d_kernel({nil, 3, 32, 32, 3}, 4, {3, 3, 3}, 3, :first)
      {12, 1, 1, 1, 3}

  ### Error cases

      iex> Axon.Shape.separable_conv3d_kernel({nil, 1, 28, 28, 3}, 3, {2}, 1, :first)
      ** (ArgumentError) kernel size must have same rank (1) as number of spatial dimensions in the input (3)
  """
  def separable_conv3d_kernel(input_shape, channel_multiplier, kernel_size, num, channels) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    idx =
      if channels == :first do
        1
      else
        tuple_size(input_shape) - 1
      end

    cond do
      num == 1 ->
        {elem(input_shape, idx) * channel_multiplier, 1, elem(kernel_size, 0), 1, 1}

      num == 2 ->
        {elem(input_shape, idx) * channel_multiplier, 1, 1, elem(kernel_size, 1), 1}

      num == 3 ->
        {elem(input_shape, idx) * channel_multiplier, 1, 1, 1, elem(kernel_size, 2)}
    end
  end

  @doc """
  Calculates the shape of a depthwise separable convolution
  bias.

  ## Examples

      iex> Axon.Shape.separable_conv3d_bias({nil, 3, 224, 224, 3}, 3, {3, 3, 2}, :first)
      {9}

      iex> Axon.Shape.separable_conv3d_bias({nil, 3, 32, 32, 3}, 2, {2, 3, 2}, :first)
      {6}

      iex> Axon.Shape.separable_conv3d_bias({nil, 1, 224, 224, 3}, 5, {3, 3, 1}, :first)
      {5}

  ### Error cases

      iex> Axon.Shape.separable_conv3d_bias({nil, 1, 224, 224, 3}, 2, {2, 2}, :first)
      ** (ArgumentError) kernel size must have same rank (2) as number of spatial dimensions in the input (3)
  """
  def separable_conv3d_bias(input_shape, channel_multiplier, kernel_size, channels) do
    unless Nx.rank(kernel_size) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "kernel size must have same rank (#{Nx.rank(kernel_size)})" <>
              " as number of spatial dimensions in the input (#{Nx.rank(input_shape) - 2})"
    end

    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    {input_channels * channel_multiplier}
  end

  @doc """
  Calculates the output shape after a pooling operation
  with the given parent shape, kernel size, strides, and
  padding.

  ## Examples

      iex> Axon.Shape.pool({nil, 3, 32, 32}, {2, 2}, [1, 2], :valid, [1, 1], :first)
      {nil, 3, 31, 16}

      iex> Axon.Shape.pool({32, 1, 28, 28}, {1, 2}, [1, 1], :same, [1, 1], :first)
      {32, 1, 28, 28}

      iex> Axon.Shape.pool({nil, 32, 32, 3}, {2, 2}, [1, 2], :valid, [1, 1], :last)
      {nil, 31, 16, 3}

      iex> Axon.Shape.pool({nil, 1, 28, 28}, {2, 1}, [1, 1], :valid, [2, 1], :first)
      {nil, 1, 26, 28}

      iex> Axon.Shape.pool({nil, 28, 28, 1}, {2, 1}, [1, 1], :valid, [2, 1], :last)
      {nil, 26, 28, 1}
  """
  def pool(parent_shape, kernel_size, strides, padding, dilations, channels) do
    unless Nx.rank(parent_shape) >= 3 do
      raise ArgumentError,
            "input shape must be at least rank 3," <>
              " got rank #{Nx.rank(parent_shape)}"
    end

    # Account for possibly nil batch dimension
    input_shape =
      if elem(parent_shape, 0) do
        parent_shape
      else
        put_elem(parent_shape, 0, 1)
      end

    kernel_dilation =
      if channels == :first do
        [1, 1 | dilations]
      else
        [1 | dilations] ++ [1]
      end

    padding =
      if is_list(padding),
        do: [{0, 0}, {0, 0} | padding],
        else: padding

    kernel_size =
      kernel_size
      |> Tuple.insert_at(0, 1)

    kernel_size =
      if channels == :first do
        Tuple.insert_at(kernel_size, 0, 1)
      else
        Tuple.append(kernel_size, 1)
      end

    strides =
      if channels == :first do
        [1, 1 | strides]
      else
        [1 | strides] ++ [1]
      end

    {shape, _} =
      Nx.Shape.pool(
        input_shape,
        kernel_size,
        strides,
        padding,
        kernel_dilation
      )

    put_elem(shape, 0, elem(parent_shape, 0))
  end

  @doc """
  Calculates the output shape after a global pooling operation with
  the given parent shape and option to keep axes.

  Assumes input is in a channels-first like format.

  ## Examples

      iex> Axon.Shape.global_pool({nil, 3, 2, 1, 1}, false, :first)
      {nil, 3}

      iex> Axon.Shape.global_pool({nil, 3, 1}, true, :first)
      {nil, 3, 1}

      iex> Axon.Shape.global_pool({nil, 1, 3, 3, 2, 4, 2}, true, :first)
      {nil, 1, 1, 1, 1, 1, 1}

      iex> Axon.Shape.global_pool({nil, 28, 28, 3}, true, :last)
      {nil, 1, 1, 3}

      iex> Axon.Shape.global_pool({nil, 28, 28, 3}, false, :last)
      {nil, 3}
  """
  def global_pool(parent_shape, keep_axes, channels) do
    for i <- 1..(Nx.rank(parent_shape) - 2), reduce: parent_shape do
      new_shape ->
        # Delete last element or replace last element with 1
        last_elem = tuple_size(new_shape)

        if channels == :first do
          if keep_axes do
            put_elem(new_shape, last_elem - i, 1)
          else
            Tuple.delete_at(new_shape, last_elem - 1)
          end
        else
          if keep_axes do
            put_elem(new_shape, i, 1)
          else
            Tuple.delete_at(new_shape, 1)
          end
        end
    end
  end

  @doc """
  Calculates the window size of a pooling operation based on given
  kernel size and spatial rank of the input.

  `window_x` functions expect a window which matches the
  rank of the input shape. For basic pooling we don't pool
  across batch or channel dimensions, so we just specify
  a size of `1` for each of those.
  """
  def pool_window_size(window, spatial_rank, channels) do
    spatial_dims =
      case window do
        x when is_integer(x) ->
          List.duplicate(x, spatial_rank)

        x when is_tuple(x) ->
          Tuple.to_list(x)

        x ->
          raise ArgumentError,
                "expected pool window to be tuple or integer" <>
                  " , got #{inspect(x)}"
      end

    if channels == :first do
      List.to_tuple([1, 1 | spatial_dims])
    else
      List.to_tuple([1 | spatial_dims] ++ [1])
    end
  end

  @doc """
  Calculates the output shape after an adaptive pooling operation
  with the given parent shape and output size.

  ## Examples

      iex> Axon.Shape.adaptive_pool({nil, 3, 32, 32}, {27, 27}, :first)
      {nil, 3, 27, 27}

      iex> Axon.Shape.adaptive_pool({nil, 1, 28, 28}, {25, 25}, :first)
      {nil, 1, 25, 25}

      iex> Axon.Shape.adaptive_pool({nil, 28, 28, 1}, {25, 25}, :last)
      {nil, 25, 25, 1}

  ### Error cases

      iex> Axon.Shape.adaptive_pool({nil, 1, 28, 28}, {30, 30}, :first)
      ** (ArgumentError) invalid output size for adaptive pool operation for input with shape {nil, 1, 28, 28} and output size {30, 30} each dimension of output size must be greater than or equal to spatial dimension of input
  """
  def adaptive_pool(parent_shape, output_size, channels) do
    unless Nx.rank(parent_shape) >= 3 do
      raise ArgumentError,
            "input shape must be at least rank 3," <>
              " got rank #{Nx.rank(parent_shape)}"
    end

    idx =
      if channels == :first do
        1
      else
        tuple_size(parent_shape) - 1
      end

    valid_output_size? =
      parent_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(idx - 1)
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(output_size))
      |> Enum.all?(&(elem(&1, 0) >= elem(&1, 1)))

    unless valid_output_size? do
      raise ArgumentError,
            "invalid output size for adaptive pool operation for" <>
              " input with shape #{inspect(parent_shape)} and output" <>
              " size #{inspect(output_size)} each dimension" <>
              " of output size must be greater than or equal to spatial" <>
              " dimension of input"
    end

    if channels == :first do
      List.to_tuple([elem(parent_shape, 0), elem(parent_shape, idx) | Tuple.to_list(output_size)])
    else
      List.to_tuple(
        [elem(parent_shape, 0) | Tuple.to_list(output_size)] ++ [elem(parent_shape, idx)]
      )
    end
  end

  @doc """
  Calculates strides needed for an adaptive pooling operation
  with the given input shape, output spatial shape, and spatial
  rank.

  Adaptive pooling functions adapt the strides of the window
  according to:

      stride = div(input, output)

  This preserves the size of the channel/batch dimension.
  """
  def adaptive_pool_window_strides(input_shape, output_spatial, spatial_rank, channels) do
    idx =
      if channels == :first do
        1
      else
        tuple_size(input_shape) - 1
      end

    input_spatial =
      input_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(idx - 1)
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
      input_spatial
      |> Enum.zip_with(output_spatial, &Kernel.div/2)

    if channels == :first do
      [1, 1 | strides]
    else
      [1 | strides] ++ [1]
    end
  end

  @doc """
  Calculates the window size for an adaptive pooling operation
  given input shape, strides, output spatial dimensions, and spatial
  rank.

  Adaptive pooling functions adopt the size of the window
  according to:

      size = input_size - (output_size - 1) * stride

  This preserves the size of the channel/batch dimension.
  """
  def adaptive_pool_window_size(
        input_shape,
        [_, _ | stride],
        output_spatial,
        spatial_rank,
        channels
      ) do
    idx =
      if channels == :first do
        1
      else
        tuple_size(input_shape) - 1
      end

    input_spatial =
      input_shape
      |> Tuple.delete_at(0)
      |> Tuple.delete_at(idx - 1)
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

    if channels == :first do
      List.to_tuple([1, 1 | output_size])
    else
      List.to_tuple([1 | output_size] ++ [1])
    end
  end

  @doc """
  Calculates the gamma/beta shape of a normalization layer
  given the input shape and channel index.

  ## Examples

      iex> Axon.Shape.norm_param({nil, 3, 28, 28}, 1)
      {3}

      iex> Axon.Shape.norm_param({nil, 28, 28, 3}, 3)
      {3}
  """
  def norm_param(parent_shape, channel_index) do
    {elem(parent_shape, channel_index)}
  end

  @doc """
  Calculates the reduction axes for batch normalization.
  """
  def batch_norm_axes(axes, channel_index) do
    axes
    |> Enum.filter(&(&1 != channel_index))
  end

  @doc """
  Calculates the reduction axes for instance normalization.
  """
  def instance_norm_axes(axes, channel_index) do
    reduction_axes = axes -- [0, channel_index]

    if reduction_axes == [] do
      raise ArgumentError, "rank of input shape must be at least 3"
    else
      reduction_axes
    end
  end

  @doc """
  Calculates the reduction axes for group normalization.
  """
  def group_norm_axes(rank) do
    for(i <- 1..(rank - 2), do: i) ++ [rank - 1]
  end

  @doc """
  Calculates the reshape for group normalization.
  """
  def group_norm_shape(shape, group_size, channel_index) do
    channels = :erlang.element(channel_index + 1, shape)
    num_groups = div(channels, group_size)

    Tuple.delete_at(shape, channel_index)
    |> Tuple.insert_at(channel_index, num_groups)
    |> Tuple.insert_at(channel_index + 1, group_size)
  end

  @doc """
  Calculates the shape after a flatten layer, which
  flattens the non-minibatch dimensions into a single
  dimension.

  ## Examples

      iex> Axon.Shape.flatten({nil, 1, 28, 28}, true)
      {nil, 784}

      iex> Axon.Shape.flatten({32, 128}, true)
      {32, 128}

      iex> Axon.Shape.flatten({nil, 10, 10}, true)
      {nil, 100}
  """
  def flatten(shape, ignore_batch?) do
    out_units =
      if ignore_batch? do
        Nx.size(Tuple.delete_at(shape, 0))
      else
        Nx.size(shape)
      end

    if ignore_batch? do
      {elem(shape, 0), out_units}
    else
      {out_units}
    end
  end

  @doc """
  Calculates the shape after a concatenate layer, which
  concatenates inputs along the given dimension.

  ## Examples

      iex> Axon.Shape.concatenate([{nil, 32}, {nil, 12}], 1)
      {nil, 44}

      iex> Axon.Shape.concatenate([{nil, 24, 32}, {nil, 24, 15}, {nil, 24, 10}], 2)
      {nil, 24, 57}

  ### Error cases

      iex> Axon.Shape.concatenate([{10, 32}, {5, 32}], 1)
      ** (ArgumentError) non-concat dims must be equal got 5 and 10 while concatenating on axis 1
  """
  def concatenate([s1 | _] = input_shapes, axis) do
    nil_names = for _ <- 1..length(input_shapes), do: List.duplicate(nil, Nx.rank(s1))
    {shape, _} = Nx.Shape.concatenate(input_shapes, nil_names, axis)
    shape
  end

  @doc """
  Calculates the shape after a reshape layer, which
  reshapes non-batch dimensions.

  ## Examples

      iex> Axon.Shape.reshape({nil, 8}, {4, 2}, true)
      {nil, 4, 2}

      iex> Axon.Shape.reshape({32, 8, 8}, {4, 4, 4}, true)
      {32, 4, 4, 4}

      iex> Axon.Shape.reshape({12, 2, 2}, {6, 2, 2, 2}, false)
      {6, 2, 2, 2}

  ### Error cases

      iex> Axon.Shape.reshape({nil, 4, 2}, {9}, true)
      ** (ArgumentError) new shape invalid for reshape operation, layer shape {nil, 4, 2} is incompatible with new shape {9}, new shape must have same size as non-batch dimensions of old shape
  """
  def reshape(shape, new_shape, ignore_batch?) do
    original_shape =
      if ignore_batch? do
        Tuple.delete_at(shape, 0)
      else
        shape
      end

    new_shape = Nx.Shape.reshape(shape, new_shape)

    if ignore_batch? do
      Tuple.insert_at(new_shape, 0, elem(shape, 0))
    else
      new_shape
    end
  end

  @doc """
  Calculates the shape after a transpose layer, which
  transposes non-batch dimensions.

  ## Examples

      iex> Axon.Shape.transpose({nil, 64, 10}, [1, 0], true)
      {nil, 10, 64}

      iex> Axon.Shape.transpose({nil, 3, 224, 224}, [1, 0, 2], true)
      {nil, 224, 3, 224}

      iex> Axon.Shape.transpose({1, 2, 3}, [2, 1, 0], false)
      {3, 2, 1}
  """
  def transpose(shape, permutation, ignore_batch?) do
    original_shape =
      if ignore_batch? do
        Tuple.delete_at(shape, 0)
      else
        shape
      end

    nil_names = List.duplicate(nil, Nx.rank(original_shape))
    {transposed_shape, _} = Nx.Shape.transpose(original_shape, permutation, nil_names)

    if ignore_batch? do
      Tuple.insert_at(transposed_shape, 0, elem(shape, 0))
    else
      transposed_shape
    end
  end

  @doc """
  Calculates the shape after a pad layer, which pads
  the spatial dimensions of an input.

  ## Examples

      iex> Axon.Shape.pad({nil, 3, 28, 28}, [{0, 1}, {1, 1}])
      {nil, 3, 29, 30}

      iex> Axon.Shape.pad({nil, 3, 30, 30}, [{2, -1}, {1, 1}])
      {nil, 3, 31, 32}

  ### Error cases

      iex> Axon.Shape.pad({nil, 784}, [{0, 1}])
      ** (ArgumentError) invalid padding configuration [{0, 1}], length of padding configuration must be equal to the rank of the spatial dimensions of the input
  """
  def pad(shape, config) do
    unless Nx.rank(shape) >= 1 do
      raise ArgumentError,
            "input shape must be at least rank 3," <>
              " got rank #{Nx.rank(shape)}"
    end

    unless length(config) == Nx.rank(shape) - 2 do
      raise ArgumentError,
            "invalid padding configuration #{inspect(config)}," <>
              " length of padding configuration must be equal" <>
              " to the rank of the spatial dimensions of the" <>
              " input"
    end

    inp_shape =
      if elem(shape, 0) == nil do
        put_elem(shape, 0, 1)
      else
        shape
      end

    padding_config = [{0, 0, 0}, {0, 0, 0} | Enum.map(config, fn {x, y} -> {x, y, 0} end)]

    output_shape = Nx.Shape.pad(inp_shape, padding_config)

    put_elem(output_shape, 0, elem(shape, 0))
  end

  @doc """
  Calculates the noise shape from a spatial dropout operation
  based on the input shape.

  Spatial dropout shapes are broadcasted across feature
  channels, so we set the channel size to 1 and preserve
  the spatial dimensions.

  ## Examples

      iex> Axon.Shape.spatial_dropout_noise_shape({nil, 3, 28, 28}, :first)
      {nil, 1, 28, 28}

      iex> Axon.Shape.spatial_dropout_noise_shape({nil, 28, 28, 3}, :last)
      {nil, 28, 28, 1}
  """
  def spatial_dropout_noise_shape(input_shape, channels) do
    if channels == :first do
      :erlang.setelement(2, input_shape, 1)
    else
      :erlang.setelement(tuple_size(input_shape), input_shape, 1)
    end
  end

  @doc """
  Calculates output shape of RNN.
  """
  def rnn(shape, units, type) do
    unless Nx.rank(shape) == 3 do
      raise ArgumentError,
            "#{inspect(type)} input shape must be rank 3 {batch_size, sequence_length, sequence_features}" <>
              " got #{inspect(shape)}"
    end

    {elem(shape, 0), elem(shape, 1), units}
  end

  @doc """
  Calculates the shape of RNN input kernel.
  """
  def rnn_input_kernel(shape, units, type) do
    unless Nx.rank(shape) == 3 do
      raise ArgumentError,
            "#{inspect(type)} input shape must be rank 3 {batch_size, sequence_length, sequence_features}" <>
              " got #{inspect(shape)}"
    end

    {elem(shape, 2), units}
  end

  @doc """
  Calculates the shape of RNN hidden kernel.
  """
  def rnn_hidden_kernel(shape, units, type) do
    unless Nx.rank(shape) == 3 do
      raise ArgumentError,
            "#{inspect(type)} input shape must be rank 3 {batch_size, sequence_length, sequence_features}" <>
              " got #{inspect(shape)}"
    end

    {units, units}
  end

  @doc """
  Calculates the shape of RNN bias.
  """
  def rnn_bias(shape, units, type) do
    unless Nx.rank(shape) == 3 do
      raise ArgumentError,
            "#{inspect(type)} input shape must be rank 3 {batch_size, sequence_length, sequence_features}" <>
              " got #{inspect(shape)}"
    end

    {units}
  end

  @doc """
  Calculates the shape of RNN hidden state.
  """
  def rnn_hidden_state(shape, units, :conv_lstm) do
    # input shape must be rank > 3 {batch_size, sequence_length, spacial_dimensions...}"

    shape
    |> put_elem(1, 1)
    |> put_elem(2, units)
  end

  def rnn_hidden_state(shape, units, type) do
    unless Nx.rank(shape) == 3 do
      raise ArgumentError,
            "#{inspect(type)} input shape must be rank 3 {batch_size, sequence_length, sequence_features}" <>
              " got #{inspect(shape)}"
    end

    {elem(shape, 0), 1, units}
  end

  @doc """
  Calculates the base shape and slice size of a split operation.

  ## Examples

      iex> Axon.Shape.split({nil, 1, 10}, 2, -1)
      {5, {nil, 1, 5}}

      iex> Axon.Shape.split({32, 1, 10}, 16, 0)
      {2, {2, 1, 10}}

  ### Error cases

      iex> Axon.Shape.split({nil, 1, 10}, 2, 0)
      ** (ArgumentError) cannot split along batch dimension with dynamic batch size, please provide a static (non-nil) batch size

      iex> Axon.Shape.split({nil, 5, 10}, 2, 1)
      ** (ArgumentError) unable to create 2 even splits along axis 1 of size 5
  """
  def split(shape, n, axis) do
    unless Nx.rank(shape) >= 1 do
      raise ArgumentError,
            "input shape must be at least rank 3," <>
              " got rank #{Nx.rank(shape)}"
    end

    nil_names = List.duplicate(nil, Nx.rank(shape))
    non_nil_shape = if elem(shape, 0) == nil, do: put_elem(shape, 0, 1), else: shape

    axis = Nx.Shape.normalize_axis(non_nil_shape, axis, nil_names)

    if axis == 0 and elem(shape, 0) == nil do
      raise ArgumentError,
            "cannot split along batch dimension with dynamic" <>
              " batch size, please provide a static (non-nil)" <>
              " batch size"
    end

    unless rem(elem(shape, axis), n) == 0 do
      raise ArgumentError,
            "unable to create #{n} even splits along axis #{axis}" <>
              " of size #{elem(shape, axis)}"
    end

    slice_size = div(elem(shape, axis), n)

    {slice_size, put_elem(shape, axis, slice_size)}
  end

  @doc """
  Checks if input shapes are broadcast compatible and returns
  the output shape of the element-wise operation.

  ## Examples

      iex> Axon.Shape.element_wise([{1, 128}, {128, 128}])
      {128, 128}

      iex> Axon.Shape.element_wise([{1, 32, 1}, {28, 1, 1}, {28, 1, 14}])
      {28, 32, 14}

      iex> Axon.Shape.element_wise([{nil, 32}, {nil, 32}])
      {nil, 32}

  ### Error cases

      iex> Axon.Shape.element_wise([{128, 1}, {nil, 32}])
      ** (ArgumentError) cannot broadcast tensor of dimensions {nil, 32} to {128, 1}

  """
  def element_wise([first | rest]) do
    Enum.reduce(rest, first, fn shape, target_shape ->
      lnames = List.duplicate(nil, tuple_size(shape))
      rnames = List.duplicate(nil, tuple_size(target_shape))
      # TODO(seanmor5): If this fails, I wonder if it's better to rescue
      # and re-raise with Axon specific messages?
      {out_shape, _} = Nx.Shape.binary_broadcast(shape, lnames, target_shape, rnames)
      out_shape
    end)
  end

  @doc """
  Computes the output shape after a resize layer.
  """
  def resize(input_shape, output_shape, channels) do
    unless Nx.rank(input_shape) >= 3 do
      raise ArgumentError, "input shape must be at least rank 3"
    end

    unless Nx.rank(output_shape) == Nx.rank(input_shape) - 2 do
      raise ArgumentError,
            "output shape must be equal to number of" <>
              " spatial dimensions in the input"
    end

    spatial_dimensions =
      case channels do
        :first ->
          Enum.to_list(2..(Nx.rank(input_shape) - 1))

        :last ->
          Enum.to_list(1..(Nx.rank(output_shape) - 2))

        invalid ->
          raise ArgumentError, "invalid channel configuration #{inspect(invalid)}"
      end

    for {d, i} <- Enum.with_index(spatial_dimensions), reduce: input_shape do
      shape ->
        put_elem(shape, d, elem(output_shape, i))
    end
  end
end
