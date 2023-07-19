Mix.install([
  {:axon, "~> 0.5"},
  {:exla, "~> 0.5"},
  {:nx, "~> 0.5"},
  {:scidata, "~> 0.1"}
])

defmodule Cifar do
  require Axon

  @batch_size 32
  @image_channels 3
  @image_side_pixels 32
  @channel_value_max 255
  @label_values Enum.to_list(0..9)

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), @image_side_pixels, @image_side_pixels, @image_channels})
    |> Nx.divide(@channel_value_max)
    |> Nx.to_batched(@batch_size)
    |> Enum.split(1500)
  end

  defp transform_labels({bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(@label_values))
    |> Nx.to_batched(@batch_size)
    |> Enum.split(1500)
  end

  defp build_model(input_shape) do
    Axon.input("input", shape: input_shape)
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.flatten()
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout(rate: 0.5)
    |> Axon.dense(length(@label_values), activation: :softmax)
  end

  defp train_model(model, train_images, train_labels, epochs) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, :adam)
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), %{}, epochs: epochs, compiler: EXLA)
  end

  defp test_model(model, model_state, test_images, test_labels) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(test_images, test_labels), model_state, compiler: EXLA)
  end

  def run do
    {images, labels} = Scidata.CIFAR10.download()

    {train_images, test_images} = transform_images(images)
    {train_labels, test_labels} = transform_labels(labels)

    model = build_model({nil, @image_channels, @image_side_pixels, @image_side_pixels}) |> IO.inspect()

    IO.write("\n\n Training Model \n\n")

    model_state =
      model
      |> train_model(train_images, train_labels, 10)

    IO.write("\n\n Testing Model \n\n")

    test_model(model, model_state, test_images, test_labels)

    IO.write("\n\n")
  end
end

Cifar.run()
