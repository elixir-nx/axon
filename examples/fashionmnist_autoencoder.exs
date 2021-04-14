defmodule Autoencoder do
  def encoder(x) do
    x
    |> Axon.dense(64, activation: :tanh)
    |> Axon.dense(3, activation: :tanh)
  end

  def decoder(x) do
    x
    |> Axon.dense(64, activation: :tanh)
    |> Axon.dense(784, activation: :tanh)
  end

  def model() do
    Axon.input({nil, 784})
    |> encoder()
    |> decoder()
  end
end

model = Autoencoder.model()

IO.inspect model

train_images = Axon.Data.MNIST.download_images()

IO.puts("Sample image:\n")

sample_image =
  train_images
  |> Enum.at(0)
  |> (&(&1[:rand.uniform(32)])).()

sample_image
|> Nx.reshape({28, 28})
|> Nx.to_heatmap()
|> IO.inspect()

IO.puts("\nTraining autoencoder...")

{final_params, _optimizer_state} =
  model
  |> Axon.Training.step(:mean_squared_error, Axon.Optimizers.adamw(0.005))
  |> Axon.Training.train(train_images, train_images, epochs: 5, compiler: EXLA)

require Axon

model
|> Axon.predict(final_params, sample_image, compiler: EXLA)
|> Nx.reshape({28, 28})
|> Nx.to_heatmap()
|> IO.inspect()
