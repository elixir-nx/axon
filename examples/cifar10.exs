defmodule CIFAR do
  defp unzip_cache_or_download(zip) do
    base_url = 'https://www.cs.toronto.edu/~kriz/'
    path = Path.join("tmp", zip)

    data =
      if File.exists?(path) do
        IO.puts("Using #{zip} from tmp/\n")
        File.read!(path)
      else
        IO.puts("Fetching #{zip} from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz\n")
        :inets.start()
        :ssl.start()

        {:ok, {_status, _response, data}} = :httpc.request(:get, {base_url ++ zip, []}, [], [])
        File.mkdir_p!("tmp")
        File.write!(path, data)

        data
      end

    data
  end

  def parse_images(content) do
    {imgs, labels} =
      for <<example::size(3073)-binary <- content>>, reduce: {[], []} do
        {imgs, labels} ->
          <<label::size(8)-bitstring, image::size(3072)-binary>> = example
          # In order to "stack" we need a leading batch dim,
          # we can introduce an Nx.stack for this similar to
          # torch.stack
          img =
            image
            |> Nx.from_binary({:u, 8})
            |> Nx.reshape({1, 3, 32, 32})

          label =
            label
            |> Nx.from_binary({:u, 8})

          {[img | imgs], [label | labels]}
      end

    {Nx.concatenate(imgs, axis: 0), Nx.concatenate(labels, axis: 0)}
  end

  def download(zip) do
    gz = unzip_cache_or_download(zip)

    # Extract image and labels from tarfile, ideally we can also do this lazily
    # such that we overlap training with IO - essentially have workers reading
    # batches from the files and caching them in memory while the accelerator
    # is busy doing actual training
    with {:ok, files} <- :erl_tar.extract({:binary, gz}, [:memory, :compressed]) do
      {imgs, labels} =
        files
        |> Enum.filter(fn {fname, _} -> String.match?(List.to_string(fname), ~r/data_batch/) end)
        |> Enum.map(fn {_, content} -> Task.async(fn -> parse_images(content) end) end)
        |> Enum.map(&Task.await(&1, :infinity))
        |> Enum.unzip()

      # Batch and one hot encode
      #
      # Ideally, we have something we can use as a batched stream
      # such that we can lazily query for batches and avoid loading the
      # entire dataset into memory, with this we could also apply lazy
      # transformations enabling things like data echoing, see TFDS
      imgs =
        imgs
        |> Nx.concatenate(axis: 0)
        # We shouldn't have to do this, the batch should wrap or truncate
        |> Nx.slice([0, 0, 0, 0], [49984, 3, 32, 32])
        |> Nx.divide(255)
        |> Nx.to_batched_list(32)

      labels =
        labels
        |> Nx.concatenate(axis: 0)
        |> Nx.slice([0], [49984])
        |> Nx.new_axis(-1)
        |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
        |> Nx.to_batched_list(32)

      {imgs, labels}
    end
  end
end

model =
  Axon.input({nil, 3, 32, 32})
  |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
  |> Axon.batch_norm()
  |> Axon.max_pool(kernel_size: {2, 2})
  |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
  |> Axon.spatial_dropout()
  |> Axon.batch_norm()
  |> Axon.max_pool(kernel_size: {2, 2})
  |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
  |> Axon.batch_norm()
  |> Axon.flatten()
  |> Axon.dense(64, activation: :relu)
  |> Axon.dropout()
  |> Axon.dense(10, activation: :softmax)

IO.inspect model

{train_images, train_labels} = CIFAR.download('cifar-10-binary.tar.gz')

{final_params, _optimizer_state} =
  model
  |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
  |> Axon.Training.train(train_images, train_labels, epochs: 20, compiler: EXLA)
  |> Nx.backend_transfer()
  |> IO.inspect()