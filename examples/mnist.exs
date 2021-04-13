defmodule MNIST do
  defp unzip_cache_or_download(zip) do
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    path = Path.join("tmp", zip)

    data =
      if File.exists?(path) do
        IO.puts("Using #{zip} from tmp/\n")
        File.read!(path)
      else
        IO.puts("Fetching #{zip} from https://storage.googleapis.com/cvdf-datasets/mnist/\n")
        :inets.start()
        :ssl.start()

        {:ok, {_status, _response, data}} = :httpc.request(base_url ++ zip)
        File.mkdir_p!("tmp")
        File.write!(path, data)

        data
      end

    :zlib.gunzip(data)
  end

  def download(images, labels) do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
      unzip_cache_or_download(images)

    train_images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows * n_cols})
      |> Nx.divide(255)
      |> Nx.to_batched_list(32)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    <<_::32, n_labels::32, labels::binary>> = unzip_cache_or_download(labels)

    train_labels =
      labels
      |> Nx.from_binary({:u, 8})
      |> Nx.new_axis(-1)
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.to_batched_list(32)

    IO.puts("#{n_labels} labels\n")

    {train_images, train_labels}
  end
end

model =
  Axon.input({nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.layer_norm()
  |> Axon.dropout()
  |> Axon.dense(10, activation: :softmax)

IO.inspect model

{train_images, train_labels} = MNIST.download('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

{final_params, _optimizer_state} =
  model
  |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
  |> Axon.Training.train(train_images, train_labels, epochs: 10, compiler: EXLA)

IO.inspect(Nx.backend_transfer(final_params))
