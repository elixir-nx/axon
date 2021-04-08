defmodule FashionMNIST do
  defp unzip_cache_or_download(zip) do
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    path = Path.join("tmp/fashionmnist", zip)

    data =
      if File.exists?(path) do
        IO.puts("Using #{zip} from tmp/fashionmnist\n")
        File.read!(path)
      else
        IO.puts("Fetching #{zip} from #{base_url}\n")
        :inets.start()
        :ssl.start()

        {:ok, {_status, _response, data}} = :httpc.request(:get, {base_url ++ zip, []}, [], [])
        File.mkdir_p!("tmp/fashionmnist")
        File.write!(path, data)

        data
      end

    :zlib.gunzip(data)
  end

  def download(images) do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
      unzip_cache_or_download(images)

    train_images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows * n_cols})
      |> Nx.divide(255)
      |> Nx.to_batched_list(32)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    train_images
  end
end

defmodule Autoencoder do
  def encoder(x) do
    x
    |> Axon.dense(64, activation: :relu)
    |> Axon.dense(3, activation: :relu)
  end

  def decoder(x) do
    x
    |> Axon.dense(64, activation: :relu)
    |> Axon.dense(784, activation: :relu)
  end

  def model() do
    Axon.input({nil, 784})
    |> encoder()
    |> decoder()
  end
end

model = Autoencoder.model()

IO.inspect model

# Labels are located at train-labels-idx1-ubyte.gz
train_images = FashionMNIST.download('train-images-idx3-ubyte.gz')

IO.puts("Sample image:\n")

train_images
|> Enum.at(0)
|> (&(&1[:rand.uniform(32)])).()
|> Nx.reshape({28, 28})
|> Nx.to_heatmap()
|> IO.inspect()

IO.puts("\nTraining autoencoder...")

final_params =
  model
  |> Axon.Training.step(:mean_squared_error, Axon.Optimizers.adamw(0.005))
  |> Axon.Training.train(train_images, train_images, epochs: 5)

IO.inspect(Nx.backend_transfer(final_params))
