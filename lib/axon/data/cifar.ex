defmodule Axon.Data.CIFAR do
  alias Data.Utils

  @default_data_path "tmp/cifar"
  @base_url 'https://www.cs.toronto.edu/~kriz/'
  @dataset_file 'cifar-10-binary.tar.gz'

  defp parse_images(content) do
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

  def download(opts \\ []) do
    batch_size = opts[:batch_size] || 32
    data_path = opts[:data_path] || @default_data_path

    gz = Utils.unzip_cache_or_download(@base_url, @dataset_file, data_path)

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
        |> Nx.to_batched_list(batch_size)

      labels =
        labels
        |> Nx.concatenate(axis: 0)
        |> Nx.slice([0], [49984])
        |> Nx.new_axis(-1)
        |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
        |> Nx.to_batched_list(batch_size)

      {imgs, labels}
    end
  end
end
