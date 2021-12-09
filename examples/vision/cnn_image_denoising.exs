Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
  {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.0"}
])

EXLA.set_preferred_defn_options([:tpu, :cuda, :rocm, :host])

defmodule MnistDenoising do
  import Nx.Defn
  alias Axon.Loop.State

  @noise_factor 0.4
  @batch_size 32
  @epochs 25

  def run do
    {{train_images, test_images}, _} =
      Scidata.MNIST.download(transform_images: &transform_images/1)

    noisy_train_images = Stream.map(train_images, &add_noise/1)
    noisy_test_images = Stream.map(test_images, &add_noise/1)
    train_data = Stream.zip(train_images, noisy_train_images)

    # Display normal versus noisy image
    train_images |> Enum.take(1) |> hd() |> display_image()
    noisy_train_images |> Enum.take(1) |> hd() |> display_image()

    # Train with noisy images as input and train images as targets
    model = build_model({nil, 1, 28, 28})

    model_state =
      model
      |> Axon.Loop.trainer(:binary_cross_entropy, :adam)
      |> Axon.Loop.run(train_data, epochs: @epochs, compiler: EXLA)

    # Predict on batches of test images
    test = noisy_test_images |> Enum.take(1) |> hd()
    preds = Axon.predict(model, model_state, test, compiler: EXLA)

    IO.write("\n\nNoisy Image\n\n")
    test |> display_image()
    IO.write("\n\nDenoised Prediction\n\n")
    preds |> display_image()
  end

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 1, 28, 28})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(@batch_size)
    # Test split
    |> Enum.split(1750)
  end

  defnp add_noise(images) do
    @noise_factor
    |> Nx.multiply(Nx.random_normal(images))
    |> Nx.add(images)
    |> Nx.clip(0.0, 1.0)
  end

  defp display_image(images) do
    images
    |> Nx.slice_axis(0, 1, 0)
    |> Nx.reshape({1, 28, 28})
    |> Nx.to_heatmap()
    |> IO.inspect()
  end

  defp build_model(input_shape) do
    input_shape
    |> encoder()
    |> decoder()
  end

  defp encoder(input_shape) do
    input_shape
    |> Axon.input()
    |> Axon.conv(32, kernel_size: {3, 3}, padding: :same, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2}, padding: :same)
    |> Axon.conv(32, kernel_size: {3, 3}, padding: :same, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2}, padding: :same)
  end

  defp decoder(input) do
    input
    |> Axon.conv_transpose(32,
      kernel_size: {3, 3},
      strides: [2, 2],
      activation: :relu,
      padding: :same
    )
    |> Axon.conv_transpose(32,
      kernel_size: {3, 3},
      strides: [2, 2],
      activation: :relu,
      padding: :same
    )
    |> Axon.conv(1, kernel_size: {3, 3}, activation: :sigmoid, padding: :same)
  end
end

MnistDenoising.run()
