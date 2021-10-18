Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
  {:exla, github: "elixir-nx/exla", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.1"},
])

# Configure default platform with accelerator precedence as tpu > cuda > rocm > host
case EXLA.Client.get_supported_platforms() do
  %{'TPU' => _} ->
    Application.put_env(:exla, :clients, default: [platform: :tpu])

  %{'CUDA' => _} ->
    Application.put_env(:exla, :clients, default: [platform: :cuda])

  %{'ROCM' => _} ->
    Application.put_env(:exla, :clients, default: [platform: :rocm])

  %{'Host' => _} ->
    Application.put_env(:exla, :clients, default: [platform: :host])
end

defmodule Fashionmist do
  require Axon
  alias Axon.Loop.State

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

  defp log_metrics(
         %State{epoch: epoch, iteration: iter, metrics: metrics, process_state: pstate} = state,
         mode
       ) do
    loss =
      case mode do
        :train ->
          %{loss: loss} = pstate
          "Loss: #{:io_lib.format('~.5f', [Nx.to_scalar(loss)])}"

        :test ->
          ""
      end

    metrics =
      metrics
      |> Enum.map(fn {k, v} -> "#{k}: #{:io_lib.format('~.5f', [Nx.to_scalar(v)])}" end)
      |> Enum.join(" ")

    IO.write("\rEpoch: #{Nx.to_scalar(epoch)}, Batch: #{Nx.to_scalar(iter)}, #{loss} #{metrics}")

    {:continue, state}
  end

  defp train_model(model, train_images, epochs) do
    model
    |> Axon.Loop.trainer(:mean_squared_error, :adam)
    |> Axon.Loop.metric(:mean_absolute_error, "Error")
    |> Axon.Loop.handle(:iteration_completed, &log_metrics(&1, :train), every: 50)
    |> Axon.Loop.run(Stream.zip(train_images, train_images), epochs: epochs, compiler: EXLA)
  end

  def run do
    {train_images, _} = Scidata.FashionMNIST.download(transform_images: &transform_images/1)

    model = Autoencoder.build_model({nil, 1, 28, 28}, 64) |> IO.inspect

    model_state = train_model(model, train_images, 5)

    sample_image =
      train_images
      |> hd()
      |> Nx.slice_axis(0, 1, 0)
      |> Nx.reshape({1, 1, 28, 28})

    sample_image |> Nx.to_heatmap() |> IO.inspect

    model
    |> Axon.predict(model_state, sample_image, compiler: EXLA)
    |> Nx.to_heatmap()
    |> IO.inspect()
  end
end

Fashionmist.run()
