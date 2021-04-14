defmodule Axon.Data.MNIST do
  alias Axon.Data.Utils

  @default_data_path "tmp/mnist"
  @base_url 'https://storage.googleapis.com/cvdf-datasets/mnist/'
  @image_file 'train-images-idx3-ubyte.gz'
  @label_file 'train-labels-idx1-ubyte.gz'

  def download_images(opts \\ []) do
    batch_size = opts[:batch_size] || 32
    data_path = opts[:data_path] || @default_data_path

    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
      Utils.unzip_cache_or_download(@base_url, @image_file, data_path, unzip: true)

    batched_images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows * n_cols})
      |> Nx.divide(255)
      |> Nx.to_batched_list(batch_size)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    batched_images
  end

  def download_labels(opts \\ []) do
    batch_size = opts[:batch_size] || 32
    data_path = opts[:data_path] || @default_data_path

    labels =
      <<_::32, n_labels::32, labels::binary>> =
      Utils.unzip_cache_or_download(@base_url, @label_file, data_path, unzip: true)

    batched_labels =
      labels
      |> Nx.from_binary({:u, 8})
      |> Nx.slice([0], [60000])
      |> Nx.new_axis(-1)
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.to_batched_list(batch_size)

    IO.puts("#{n_labels} labels\n")

    batched_labels
  end

  def download(opts \\ []),
    do: {download_images(opts), download_labels(opts)}
end
