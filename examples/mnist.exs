Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
  {:exla, github: "elixir-nx/exla", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.1"},
])

defmodule Mnist do
  require Axon

  def run do
    transform_images =
      fn {bin, type, shape} ->
        bin
        |> Nx.from_binary(type)
        |> Nx.reshape({elem(shape, 0), 784})
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

    {train_images, train_labels} = Scidata.MNIST.download(transform_images: transform_images, transform_labels: transform_labels)

    IO.inspect train_images |> hd() |> Nx.slice_axis(0, 1, 0) |> Nx.reshape({1, 28, 28}) |> Nx.to_heatmap()

    model =
      Axon.input({nil, 784})
      |> Axon.dense(128, activation: :relu)
      |> Axon.dropout()
      |> Axon.dense(10, activation: :softmax)

    IO.inspect model

    final_training_state =
      model
      |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005), metrics: [:accuracy])
      |> Axon.Training.train(train_images, train_labels, epochs: 10, compiler: EXLA, log_every: 100)

    test_images = train_images |> hd() |> Nx.slice_axis(10, 3, 0)

    IO.inspect test_images |> Nx.reshape({3, 28, 28}) |> Nx.to_heatmap()

    prediction =
      model
      |> Axon.predict(final_training_state[:params], test_images)
      |> Nx.argmax(axis: -1)

    prediction
  end

end

Mnist.run()
