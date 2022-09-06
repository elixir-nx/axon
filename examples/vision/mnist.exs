Mix.install([
  {:axon, "~> 0.1.0"},
  {:exla, "~> 0.2.2"},
  {:nx, "~> 0.2.1"},
  {:scidata, "~> 0.1.6"}
])

# Configure default platform with accelerator precedence as tpu > cuda > rocm > host
EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

defmodule Mnist do
  require Axon

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
    # Test split
    |> Enum.split(1750)
  end

  defp transform_labels({bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched_list(32)
    # Test split
    |> Enum.split(1750)
  end

  defp build_model(input_shape) do
    Axon.input("input", shape: input_shape)
    |> Axon.dense(128, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(10, activation: :softmax)
  end

  defp train_model(model, train_images, train_labels, epochs) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), %{}, epochs: epochs)
  end

  defp test_model(model, model_state, test_images, test_labels) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(Stream.zip(test_images, test_labels), model_state)
  end

  def run do
    {images, labels} = Scidata.MNIST.download()

    {train_images, test_images} = transform_images(images)
    {train_labels, test_labels} = transform_labels(labels)

    model = build_model({nil, 784}) |> IO.inspect()

    IO.write("\n\n Training Model \n\n")

    model_state =
      model
      |> train_model(train_images, train_labels, 5)

    IO.write("\n\n Testing Model \n\n")

    model
    |> test_model(model_state, test_images, test_labels)

    IO.write("\n\n")
  end
end

Mnist.run()
