defmodule Axon.Shape do
  @moduledoc """
  Collection of layer shape calculations for determining
  output shapes and parameter shapes.
  """

  # TODO: Clean this up along with Nx.Shape

  @doc """
  Calculates the shape after a flatten layer.
  """
  def flatten(shape) do
    out_units = div(Nx.size(shape), elem(shape, 0))
    {elem(shape, 0), out_units}
  end

  @doc """
  Calculates the shape after a convolution layer with
  the given parent shape, kernel shape, strides, padding,
  input dilation and kernel dilation.
  """
  def conv_output(parent_shape, kernel_shape, strides, padding, input_dilation, kernel_dilation) do
    padding_config = padding_config(parent_shape, kernel_shape, padding, strides)

    kernel_dilation =
      if is_list(kernel_dilation),
        do: kernel_dilation,
        else: for(_ <- 1..(Nx.rank(parent_shape) - 2), do: kernel_dilation)

    kernel_dilation_padding_config = [
      {0, 0, 0},
      {0, 0, 0} | Enum.map(kernel_dilation, &{0, 0, &1 - 1})
    ]

    dilated_kernel_shape = Nx.Shape.pad(kernel_shape, kernel_dilation_padding_config)

    input_dilation =
      if is_list(input_dilation),
        do: input_dilation,
        else: for(_ <- 1..(Nx.rank(parent_shape) - 2), do: input_dilation)

    input_dilation_padding_config = [
      {0, 0, 0},
      {0, 0, 0} | Enum.map(input_dilation, &{0, 0, &1 - 1})
    ]

    dilated_input_shape = Nx.Shape.pad(parent_shape, input_dilation_padding_config)

    nil_names = List.duplicate(nil, Nx.rank(parent_shape))

    strides =
      if is_list(strides),
        do: strides,
        else: List.duplicate(strides, Nx.rank(parent_shape) - 2)

    {shape, _} =
      Nx.Shape.conv(
        dilated_input_shape,
        nil_names,
        dilated_kernel_shape,
        nil_names,
        strides,
        1,
        padding_config
      )

    shape
  end

  @doc """
  Calculates the kernel shape for a convolution with
  the given parent shape, kernel size, and output channels.
  """
  def conv_kernel(parent_shape, output_channels, kernel_size) when is_integer(kernel_size) do
    kernel_size =
      kernel_size
      |> List.duplicate(Nx.rank(parent_shape) - 2)
      |> List.to_tuple()

    conv_kernel(parent_shape, output_channels, kernel_size)
  end

  def conv_kernel(parent_shape, output_channels, kernel_size)
      when is_tuple(parent_shape) and is_integer(output_channels) and is_tuple(kernel_size) do
    unless Nx.rank(kernel_size) == Nx.rank(parent_shape) - 2 do
      raise ArgumentError,
            "expected kernel size to match the number of spatial dimensions" <>
              " in input, got #{inspect(kernel_size)} and #{inspect(parent_shape)}"
    end

    # TODO: Conv dimension numbers
    ([output_channels, elem(parent_shape, 1)] ++ Tuple.to_list(kernel_size)) |> List.to_tuple()
  end

  @doc """
  Calculates the bias shape for a convolution with
  the given parent shape and output channels.
  """
  def conv_bias(parent_shape, output_channels) do
    spatial_dims = List.duplicate(1, Nx.rank(parent_shape) - 2)
    List.to_tuple([1, output_channels | spatial_dims])
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
