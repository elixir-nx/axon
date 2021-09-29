Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
  {:exla, github: "elixir-nx/exla", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.1"},
])

defmodule Cifar do
  require Axon

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 3, 32, 32})
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
    |> Nx.mean(axes: [1], keep_axes: true)
    |> Nx.to_heatmap()
    |> IO.inspect
  end

  defp build_model(input_shape) do
    Axon.input(input_shape)
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.spatial_dropout()
    |> Axon.batch_norm()
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    |> Axon.batch_norm()
    |> Axon.flatten()
    |> Axon.dense(64, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(10, activation: :softmax)
  end

  defp train_model(model, {train_images, train_labels}, epochs) do
    model
    |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.sgd(0.01), metrics: [:accuracy])
    |> Axon.Training.train(train_images, train_labels, epochs: epochs, compiler: EXLA)
    |> Nx.backend_transfer()
  end

  def run do

    {train_images, train_labels} = Scidata.CIFAR10.download(transform_images: &transform_images/1, transform_labels: &transform_labels/1)

    view_images(train_images, {0, 1})

    model = build_model({nil, 3, 32, 32})
    IO.inspect model

    final_training_state = 
      model
      |> train_model({train_images, train_labels}, 20)
      |> IO.inspect()

    test_images = train_images |> hd() |> Nx.slice_axis(10, 3, 0)
    view_images(train_images, {10, 3})

    model
    |> Axon.predict(final_training_state[:params], test_images)
    |> Nx.argmax(axis: -1)
    |> IO.inspect
  end
end

Cifar.run()
