Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
  {:exla, github: "elixir-nx/exla", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.1"},
])

defmodule Mnist do
  require Axon

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
  end

  defp transform_labels({bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched_list(32)
  end

  defp view_images(images, {start_index, len}) do
    images
    |> hd()
    |> Nx.slice_axis(start_index, len, 0)
    |> Nx.reshape({:auto, 28, 28})
    |> Nx.to_heatmap()
    |> IO.inspect
  end

  defp build_model(input_shape) do
    Axon.input(input_shape)
    |> Axon.dense(128, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(10, activation: :softmax)
  end

  defp train_model(model, {train_images, train_labels}, epochs) do
    model
    |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005), metrics: [:accuracy])
    |> Axon.Training.train(train_images, train_labels, epochs: epochs, compiler: EXLA, log_every: 100)
  end

  def run do

    {train_images, train_labels} = Scidata.MNIST.download(transform_images: &transform_images/1, transform_labels: &transform_labels/1)

    view_images(train_images, {0, 1})

    model = build_model({nil, 784}) |> IO.inspect

    final_training_state =
      model
      |> train_model({train_images, train_labels}, 10)

    test_images = train_images |> hd() |> Nx.slice_axis(10, 3, 0)
    view_images(train_images, {10, 3})

    model
    |> Axon.predict(final_training_state[:params], test_images)
    |> Nx.argmax(axis: -1)
    |> IO.inspect
  end
end

Mnist.run()
