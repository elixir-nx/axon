Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
  {:exla, github: "elixir-nx/exla", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.1"},
])

defmodule Cifar do
  require Axon

  def run do
    transform_images =
      fn {bin, type, shape} ->
        bin
        |> Nx.from_binary(type)
        |> Nx.reshape({elem(shape, 0), 3, 32, 32})
        |> Nx.divide(255.0)
        |> Nx.to_batched_list(32)
      end

    transform_labels =
      fn {bin, type, _} ->
        bin
        |> Nx.from_binary(type)
        |> Nx.new_axis(-1)
        |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
        |> Nx.to_batched_list(32)
      end

    {train_images, train_labels} = Scidata.CIFAR10.download(transform_images: transform_images, transform_labels: transform_labels)

    train_images
    |> hd()
    |> Nx.slice_axis(0, 1, 0)
    |> Nx.reshape({3, 32, 32})
    |> Nx.mean(axes: [0], keep_axes: true)
    |> Nx.to_heatmap()
    |> IO.inspect

    model =
      Axon.input({nil, 3, 32, 32})
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

    IO.inspect model

    final_training_state =
      model
      |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.sgd(0.01), metrics: [:accuracy])
      |> Axon.Training.train(train_images, train_labels, epochs: 20, compiler: EXLA)
      |> Nx.backend_transfer()
      |> IO.inspect()

    test_images = train_images |> hd() |> Nx.slice_axis(0, 3, 0)

    IO.inspect test_images |> Nx.mean(axes: [1], keep_axes: true) |> Nx.to_heatmap()

    prediction =
      model
      |> Axon.predict(final_training_state[:params], test_images)
      |> Nx.argmax(axis: -1)
      |> IO.inspect
  end
end

Cifar.run()
