Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
  {:exla, github: "elixir-nx/exla", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.1"},
])

defmodule Fashionmist do
  require Axon

  defmodule Autoencoder do
    defp encoder(x, latent_dim) do
      x
      |> Axon.flatten()
      |> Axon.dense(latent_dim, activation: :relu)
    end

    defp decoder(x) do
      x
      |> Axon.dense(784, activation: :sigmoid)
      |> Axon.reshape({1, 28, 28})
    end

    def build_model(input_shape, latent_dim) do
      Axon.input(input_shape)
      |> encoder(latent_dim)
      |> decoder()
    end
  end

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 1, 28, 28})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
  end

  defp train_model(model, train_images, epochs) do
    model
    |> Axon.Training.step(:mean_squared_error, Axon.Optimizers.adam(0.01), metrics: [:mean_absolute_error])
    |> Axon.Training.train(train_images, train_images, epochs: epochs, compiler: EXLA)
  end

  def run do
    {train_images, _} = Scidata.FashionMNIST.download(transform_images: &transform_images/1)

    model = Autoencoder.build_model({nil, 1, 28, 28}, 64) |> IO.inspect

    final_training_state = train_model(model, train_images, 5)

    sample_image =
      train_images
      |> hd()
      |> Nx.slice_axis(0, 1, 0)
      |> Nx.reshape({1, 1, 28, 28})

    sample_image |> Nx.to_heatmap() |> IO.inspect

    model
    |> Axon.predict(final_training_state[:params], sample_image, compiler: EXLA)
    |> Nx.to_heatmap()
    |> IO.inspect()
  end
end

Fashionmist.run()
