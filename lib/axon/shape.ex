defmodule Axon.Shape do
  @moduledoc false

  import Nx.Defn

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
      ** (ArgumentError) invalid input shape 5, input shape must be a tuple of dimension sizes or a container of valid shapes
  """
  def input(input_shape) when is_tuple(input_shape) or is_map(input_shape) do
    cond do
      is_tuple(input_shape) and tuple_size(input_shape) == 0 ->
        input_shape

      is_tuple(input_shape) and is_dim(elem(input_shape, 0)) ->
        input_shape

      true ->
        {shapes, :ok} =
          Nx.Container.traverse(input_shape, :ok, fn shape, :ok ->
            shape = input(shape)
            {shape, :ok}
          end)

        shapes
    end
  end

  def input(shape) do
    raise ArgumentError,
          "invalid input shape #{inspect(shape)}, input shape must be" <>
            " a tuple of dimension sizes or a container of valid shapes"
  end

  defp is_dim(dim) when is_integer(dim) or is_nil(dim), do: true
  defp is_dim(_), do: false

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

  ## Conv

  @doc """
  Calculates the shape of a convolution kernel given the
  input shape, output filters, and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.conv_kernel({nil, 3, 224, 224}, 32, {3, 3}, :first, 1)
      {32, 3, 3, 3}

      iex> Axon.Shape.conv_kernel({nil, 3, 28}, 64, {2}, :first, 1)
      {64, 3, 2}

      iex> Axon.Shape.conv_kernel({nil, 1, 32, 32, 10}, 32, {2, 1, 3}, :first, 1)
      {32, 1, 2, 1, 3}

      iex> Axon.Shape.conv_kernel({nil, 28, 28, 3}, 64, {2, 2}, :last, 1)
      {2, 2, 3, 64}
  """
  def conv_kernel(input_shape, output_filters, kernel_size, channels, feature_group_size) do
    inner_rank = Nx.rank(input_shape) - 2
    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)

    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    input_channels =
      if rem(input_channels, feature_group_size) == 0 do
        div(input_channels, feature_group_size)
      else
        raise ArgumentError,
              "input channels must be evenly divisible by" <>
                " feature group size, got #{inspect(input_channels)}" <>
                " and #{inspect(feature_group_size)}"
      end

    case channels do
      :first ->
        List.to_tuple([output_filters, input_channels | Tuple.to_list(kernel_size)])

      :last ->
        List.to_tuple(Tuple.to_list(kernel_size) ++ [input_channels, output_filters])
    end
  end

  @doc """
  Calculates the shape of a convolution bias given the
  input shape, output filters, and kernel size.

  Kernel size must match the number of spatial dimensions
  in the input (input rank - 2).

  ## Examples

      iex> Axon.Shape.conv_bias({nil, 3, 224, 224}, 32, {3, 3}, :first, 1)
      {32}

      iex> Axon.Shape.conv_bias({nil, 3, 28}, 64, {2}, :first, 1)
      {64}

      iex> Axon.Shape.conv_bias({nil, 1, 32, 32, 10}, 32, {2, 1, 3}, :first, 1)
      {32}

      iex> Axon.Shape.conv_bias({nil, 28, 3}, 64, {2}, :last, 1)
      {64}
  """
  def conv_bias(_input_shape, output_filters, _kernel_size, _channels, _feature_group_size) do
    {output_filters}
  end

  @doc """
  Calculates kernel shape of a ConvLSTM Cell Kernel.
  """
  def conv_lstm_kernel(input_shape, output_filters, kernel_size, channels, feature_group_size) do
    inner_rank = Nx.rank(input_shape) - 3
    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)

    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    input_channels =
      if rem(input_channels, feature_group_size) == 0 do
        div(input_channels, feature_group_size)
      else
        raise ArgumentError,
              "input channels must be evenly divisible by" <>
                " feature group size, got #{inspect(input_channels)}" <>
                " and #{inspect(feature_group_size)}"
      end

    List.to_tuple([output_filters, input_channels | Tuple.to_list(kernel_size)])
  end

  @doc """
  Calculates the reshape needed to broadcast convolution bias
  over the given input shape.

  In order to effectively broadcast, we need to expand
  the dimensions of the bias term in convolutions - if
  the input bias shape is a vector, otherwise we'll just
  attempt to let it broadcast itself.
  """
  deftransform conv_bias_reshape(input, bias, channels) do
    bias_shape = Nx.shape(bias)
    spatial_rank = Nx.rank(input) - 2

    case bias_shape do
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
  Calculates the permutation options to pass to convolution
  based on channels configuration.

  It returns both the input/output permutation and the kernel
  permutation.
  """
  deftransform conv_permutations(input, channels) do
    rank = Nx.rank(input)

    case channels do
      :first ->
        perm = Enum.to_list(0..(rank - 1))
        {perm, perm}

      :last ->
        spatial = Enum.to_list(1..(rank - 2)//1)
        perm = [0, rank - 1 | spatial]
        kernel_perm = [rank - 1, rank - 2] ++ Enum.to_list(0..(rank - 3)//1)
        {perm, kernel_perm}

      invalid ->
        raise ArgumentError, "invalid channel configuration, #{inspect(invalid)}"
    end
  end

  @doc """
  Calculates strides for transposed convolution.
  """
  deftransform conv_transpose_strides(input, strides) do
    rank = Nx.rank(input) - 2

    case strides do
      [_ | _] = strides -> strides
      strides -> List.duplicate(strides, rank)
    end
  end

  @doc """
  Calculates the padding needed for a transposed convolution.
  """
  deftransform conv_transpose_padding(kernel, kernel_dilation, strides, padding, channels)
               when padding in [:valid, :same] do
    kernel_shape = Nx.shape(kernel)

    kernel_spatial_dims =
      case channels do
        :first ->
          kernel_shape
          |> Tuple.delete_at(0)
          |> Tuple.delete_at(0)

        :last ->
          kernel_shape
          |> Tuple.delete_at(tuple_size(kernel_shape) - 1)
          |> Tuple.delete_at(tuple_size(kernel_shape) - 2)
      end

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

  deftransform conv_transpose_padding(_, _, _, padding, _), do: padding

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

      iex> Axon.Shape.depthwise_conv_kernel({nil, 28, 28, 3}, 2, {2, 2}, :last)
      {2, 2, 1, 6}
  """
  def depthwise_conv_kernel(input_shape, channel_multiplier, kernel_size, channels) do
    inner_rank = Nx.rank(input_shape) - 2
    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)

    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    case channels do
      :first ->
        List.to_tuple([input_channels * channel_multiplier, 1 | Tuple.to_list(kernel_size)])

      :last ->
        List.to_tuple(Tuple.to_list(kernel_size) ++ [1, input_channels * channel_multiplier])
    end
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
  """
  def depthwise_conv_bias(input_shape, channel_multiplier, _kernel_size, channels) do
    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    {input_channels * channel_multiplier}
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
  """
  def separable_conv2d_kernel(input_shape, channel_multiplier, kernel_size, num, channels) do
    inner_rank = Nx.rank(input_shape) - 2
    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)

    unless Nx.rank(kernel_size) == inner_rank do
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

    case {channels, num} do
      {:first, 1} ->
        {elem(input_shape, idx) * channel_multiplier, 1, elem(kernel_size, 0), 1}

      {:first, 2} ->
        {elem(input_shape, idx) * channel_multiplier, 1, 1, elem(kernel_size, 1)}

      {:last, 1} ->
        {elem(kernel_size, 0), 1, 1, elem(input_shape, idx) * channel_multiplier}

      {:last, 2} ->
        {1, elem(kernel_size, 1), 1, elem(input_shape, idx) * channel_multiplier}
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

  """
  def separable_conv2d_bias(input_shape, channel_multiplier, _kernel_size, channels) do
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

  """
  def separable_conv3d_kernel(input_shape, channel_multiplier, kernel_size, num, channels) do
    inner_rank = Nx.rank(input_shape) - 2
    kernel_size = tuple_or_duplicate(:kernel_size, kernel_size, inner_rank)

    idx =
      if channels == :first do
        1
      else
        tuple_size(input_shape) - 1
      end

    case {channels, num} do
      {:first, 1} ->
        {elem(input_shape, idx) * channel_multiplier, 1, elem(kernel_size, 0), 1, 1}

      {:first, 2} ->
        {elem(input_shape, idx) * channel_multiplier, 1, 1, elem(kernel_size, 1), 1}

      {:first, 3} ->
        {elem(input_shape, idx) * channel_multiplier, 1, 1, 1, elem(kernel_size, 2)}

      {:last, 1} ->
        {elem(kernel_size, 0), 1, 1, 1, elem(input_shape, idx) * channel_multiplier}

      {:last, 2} ->
        {1, elem(kernel_size, 1), 1, 1, elem(input_shape, idx) * channel_multiplier}

      {:last, 3} ->
        {1, 1, elem(kernel_size, 2), 1, elem(input_shape, idx) * channel_multiplier}
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
  """
  def separable_conv3d_bias(input_shape, channel_multiplier, _kernel_size, channels) do
    input_channels =
      if channels == :first do
        elem(input_shape, 1)
      else
        elem(input_shape, tuple_size(input_shape) - 1)
      end

    {input_channels * channel_multiplier}
  end

  @doc """
  Calculates the window size of a pooling operation based on given
  kernel size and spatial rank of the input.

  `window_x` functions expect a window which matches the
  rank of the input shape. For basic pooling we don't pool
  across batch or channel dimensions, so we just specify
  a size of `1` for each of those.
  """
  deftransform pool_window_size(input, window, channels) do
    spatial_rank = Nx.rank(input) - 2

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
  Calculates the window strides of a pooling operation.
  """
  deftransform pool_window_strides(input, strides, window_dimensions, channels) do
    rank = Nx.rank(input)

    case {strides, channels} do
      {nil, _} -> Tuple.to_list(window_dimensions)
      {[_ | _] = strides, :first} -> [1, 1 | strides]
      {[_ | _] = strides, :last} -> [1 | strides] ++ [1]
      {strides, :first} -> [1, 1 | List.duplicate(strides, rank - 2)]
      {strides, :last} -> [1 | List.duplicate(strides, rank - 2)] ++ [1]
    end
  end

  @doc """
  Calculates window dilations of a pooling operation.
  """
  deftransform pool_window_dilations(input, window_dilations, channels) do
    rank = Nx.rank(input)

    case {window_dilations, channels} do
      {nil, _} -> List.duplicate(1, rank)
      {[_ | _] = dilations, :first} -> [1, 1 | dilations]
      {[_ | _] = dilations, :last} -> [1 | dilations] ++ [1]
      {dilations, :first} -> [1, 1 | List.duplicate(dilations, rank - 2)]
      {dilations, :last} -> [1 | List.duplicate(dilations, rank - 2)] ++ [1]
    end
  end

  @doc """
  Calculates padding of a pooling operation based on input padding
  and channels configuration.
  """
  deftransform pool_window_padding(padding, channels) do
    case {padding, channels} do
      {:same, _} -> :same
      {:valid, _} -> :valid
      {padding, :first} -> [{0, 0}, {0, 0} | padding]
      {padding, :last} -> [{0, 0} | padding] ++ [{0, 0}]
    end
  end

  @doc """
  Computes the adaptive pooling output size from the given parent
  shape, output shape and channels configuration.
  """
  deftransform adaptive_pool_output_size(input, nil, channels) do
    parent_shape = Nx.shape(input)

    case channels do
      :first ->
        parent_shape
        |> Tuple.delete_at(0)
        |> Tuple.delete_at(0)

      :last ->
        parent_shape
        |> Tuple.delete_at(tuple_size(parent_shape) - 1)
        |> Tuple.delete_at(0)
    end
  end

  deftransform adaptive_pool_output_size(input, output_size, _channels) do
    inner_rank = Nx.rank(input) - 2
    tuple_or_duplicate(:output_size, output_size, inner_rank)
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
  deftransform adaptive_pool_window_strides(input, output_spatial, channels) do
    input_shape = Nx.shape(input)
    spatial_rank = Nx.rank(input) - 2

    idx =
      if channels == :first do
        1
      else
        tuple_size(input_shape) - 1
      end

    input_spatial =
      input_shape
      |> Tuple.delete_at(idx)
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

    strides = Enum.zip_with(input_spatial, output_spatial, &Kernel.div/2)

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
  deftransform adaptive_pool_window_size(
                 input,
                 stride,
                 output_spatial,
                 channels
               ) do
    input_shape = Nx.shape(input)
    spatial_rank = Nx.rank(input) - 2

    strides =
      case channels do
        :first ->
          [_, _ | strides] = stride
          strides

        :last ->
          [_ | strides] = Enum.take(stride, length(stride) - 1)
          strides
      end

    idx =
      if channels == :first do
        1
      else
        tuple_size(input_shape) - 1
      end

    input_spatial =
      input_shape
      |> Tuple.delete_at(idx)
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

    zip_all = [input_spatial, output_spatial, strides]

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
    names = List.duplicate(nil, Nx.rank(parent_shape))
    axis = Nx.Shape.normalize_axis(parent_shape, channel_index, names)
    {elem(parent_shape, axis)}
  end

  @doc """
  Calculates the reduction axes for batch normalization.
  """
  deftransform batch_norm_axes(input, channel_index) do
    axis = Nx.Shape.normalize_axis(Nx.shape(input), channel_index, Nx.names(input))

    input
    |> Nx.axes()
    |> Enum.filter(&(&1 != axis))
  end

  @doc """
  Calculates the reduction axes for instance normalization.
  """
  deftransform instance_norm_axes(input, channel_index) do
    axis = Nx.Shape.normalize_axis(Nx.shape(input), channel_index, Nx.names(input))
    axes = Nx.axes(input)

    reduction_axes = axes -- [0, axis]

    if reduction_axes == [] do
      raise ArgumentError, "rank of input shape must be at least 3"
    else
      reduction_axes
    end
  end

  @doc """
  Calculates the reduction axes for group normalization.
  """
  deftransform group_norm_axes(x, channel_index) do
    Enum.to_list(1..(Nx.rank(x) - 1)) -- [channel_index]
  end

  @doc """
  Calculates the reshape for group normalization.
  """
  deftransform group_norm_shape(input, num_groups, channel_index) do
    shape = Nx.shape(input)
    channel_index = Nx.Shape.normalize_axis(Nx.shape(input), channel_index, Nx.names(input))

    channels = elem(shape, channel_index)
    group_size = div(channels, num_groups)

    shape
    |> put_elem(channel_index, num_groups)
    |> Tuple.insert_at(channel_index + 1, group_size)
  end

  @doc """
  Computes split sizes for the given splits.
  """
  deftransform split(input, index, splits, axis) do
    shape = Nx.shape(input)

    nil_names = List.duplicate(nil, Nx.rank(shape))
    axis = Nx.Shape.normalize_axis(shape, axis, nil_names)

    unless rem(elem(shape, axis), splits) == 0 do
      raise ArgumentError,
            "unable to create #{splits} even splits along axis #{axis}" <>
              " of size #{elem(shape, axis)}"
    end

    slice_size = div(elem(shape, axis), splits)

    offset = index * slice_size
    {offset, slice_size}
  end

  @doc """
  Calculates the noise shape from a spatial dropout operation
  based on the input shape.

  Spatial dropout shapes are broadcasted across feature
  channels, so we set the channel size to 1 and preserve
  the spatial dimensions.

  ## Examples

      iex> Axon.Shape.spatial_dropout_noise_shape({1, 3, 28, 28}, :first)
      {1, 1, 28, 28}

      iex> Axon.Shape.spatial_dropout_noise_shape({1, 28, 28, 3}, :last)
      {1, 28, 28, 1}
  """
  deftransform spatial_dropout_noise_shape(input, channels) do
    input_shape = Nx.shape(input)

    if channels == :first do
      :erlang.setelement(2, input_shape, 1)
    else
      :erlang.setelement(tuple_size(input_shape), input_shape, 1)
    end
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
  Returns the reduction axes for a global pooling operation
  based on the input rank and channels configuration.
  """
  deftransform global_pool_axes(input, channels) do
    rank = Nx.rank(input)

    case channels do
      :last ->
        Enum.to_list(1..(rank - 2))

      :first ->
        Enum.to_list(2..(rank - 1))
    end
  end

  defp tuple_or_duplicate(key, tuple_or_integer, rank) do
    cond do
      is_tuple(tuple_or_integer) ->
        if tuple_size(tuple_or_integer) != rank do
          raise ArgumentError,
                "expected #{inspect(key)} to be a #{rank}-element tuple, " <>
                  "got: #{inspect(tuple_or_integer)}"
        end

        tuple_or_integer

      is_integer(tuple_or_integer) ->
        Tuple.duplicate(tuple_or_integer, rank)

      true ->
        raise ArgumentError,
              "expected #{inspect(key)} to be an integer or a tuple, " <>
                "got: #{inspect(tuple_or_integer)}"
    end
  end
end
