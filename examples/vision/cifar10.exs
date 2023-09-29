Mix.install([
  {:axon, "~> 0.5"},
  {:exla, "~> 0.5"},
  {:nx, "~> 0.5"},
  {:scidata, "~> 0.1"}
])

defmodule Cifar do
  require Axon

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape(shape, names: [:count, :channels, :width, :height])
    # Move channels to last position to match what conv layer expects
    |> Nx.transpose(axes: [:count, :width, :height, :channels])
    |> Nx.divide(255.0)
    |> Nx.to_batched(32)
    |> Enum.split(1500)
  end

  defp transform_labels({bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched(32)
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
    |> Axon.dense(10, activation: :softmax)
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
    {{_, _, {_, channels, width, height}} = images, labels} = Scidata.CIFAR10.download()

    {train_images, test_images} = transform_images(images)
    {train_labels, test_labels} = transform_labels(labels)

    model =
      # Move channels to last position to match what conv layer expects
      {nil, width, height, channels}
      |> build_model()
      |> IO.inspect()

    IO.write("\n\n Training Model \n\n")

    model_state =
      model
      |> train_model(train_images, train_labels, 10)

    IO.write("\n\n Testing Model \n\n")

    test_model(model, model_state, test_images, test_labels)

    IO.write("\n\n")
  end
end

Nx.default_backend(EXLA.Backend)
Cifar.run()
