Mix.install([
  {:axon, "~> 0.5"},
  {:exla, "~> 0.5"},
  {:nx, "~> 0.5"},
  {:scidata, "~> 0.1"}
])

defmodule FashionMNIST do
  require Axon

  @batch_size 32
  @image_channels 1
  @image_side_pixels 28

  defmodule Autoencoder do
    @image_channels 1
    @image_side_pixels 28

    defp encoder(x, latent_dim) do
      x
      |> Axon.flatten()
      |> Axon.dense(latent_dim, activation: :relu)
    end

    defp decoder(x) do
      x
      |> Axon.dense(@image_side_pixels**2, activation: :sigmoid)
      |> Axon.reshape({:batch, @image_channels, @image_side_pixels, @image_side_pixels})
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
    |> Nx.reshape({elem(shape, 0), @image_channels, @image_side_pixels, @image_side_pixels})
    |> Nx.divide(Nx.Constants.max(type))
    |> Nx.to_batched(@batch_size)
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

    model = Autoencoder.build_model({nil, @image_channels, @image_side_pixels, @image_side_pixels}, 64) |> IO.inspect()

    model_state = train_model(model, train_images, 5)

    sample_image =
      train_images
      |> Enum.fetch!(0)
      |> Nx.slice_along_axis(0, 1)
      |> Nx.reshape({1, @image_channels, @image_side_pixels, @image_side_pixels})

    sample_image |> Nx.to_heatmap() |> IO.inspect()

    model
    |> Axon.predict(model_state, sample_image, compiler: EXLA)
    |> Nx.to_heatmap()
    |> IO.inspect()
  end
end

FashionMNIST.run()
