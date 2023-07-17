Mix.install([
  {:axon, "~> 0.5"},
  {:exla, "~> 0.5"},
  {:nx, "~> 0.5"},
  {:scidata, "~> 0.1"}
])

defmodule FashionMNIST do
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
      |> Axon.reshape({:batch, 1, 28, 28})
    end

    def build_model(input_shape, latent_dim) do
      Axon.input("input", shape: input_shape)
      |> encoder(latent_dim)
      |> decoder()
    end
  end

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 1, 28, 28})
    |> Nx.divide(Nx.Constants.max(type))
    |> Nx.to_batched(32)
  end

  defp train_model(model, train_images, epochs) do
    model
    |> Axon.Loop.trainer(:mean_squared_error, :adam)
    |> Axon.Loop.metric(:mean_absolute_error, "Error")
    |> Axon.Loop.run(Stream.zip(train_images, train_images), %{}, epochs: epochs, compiler: EXLA)
  end

  def run do
    {images, _} = Scidata.FashionMNIST.download()

    train_images = transform_images(images)

    model = Autoencoder.build_model({nil, 1, 28, 28}, 64) |> IO.inspect()

    model_state = train_model(model, train_images, 5)

    sample_image =
      train_images
      |> Enum.fetch!(0)
      |> Nx.slice_along_axis(0, 1)
      |> Nx.reshape({1, 1, 28, 28})

    sample_image |> Nx.to_heatmap() |> IO.inspect()

    model
    |> Axon.predict(model_state, sample_image, compiler: EXLA)
    |> Nx.to_heatmap()
    |> IO.inspect()
  end
end

FashionMNIST.run()
