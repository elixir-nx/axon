defmodule MNIST do
  import Nx.Defn

  @default_defn_compiler {EXLA, run_options: [keep_on_device: true]}

  defn init_random_params do
    w1 = Axon.Initializers.lecun_normal(shape: {32, 1, 7, 7})
    b1 = Axon.Initializers.normal(shape: {1, 32, 1, 1}, scale: 1.0e-6)
    # TODO: Changing this to lecun is numerically unstable?
    w2 = Axon.Initializers.normal(shape: {64, 32, 4, 4})
    b2 = Axon.Initializers.normal(shape: {1, 64, 1, 1}, scale: 1.0e-6)
    w3 = Axon.Initializers.lecun_normal(shape: {5184, 10})
    b3 = Axon.Initializers.normal(shape: {10}, scale: 1.0e-6)
    {w1, b1, w2, b2, w3, b3}
  end

  defn predict({w1, b1, w2, b2, w3, b3}, batch) do
    batch
    |> Axon.Layers.conv2d(w1, b1, strides: [1, 1])
    |> Axon.Activations.relu()
    |> Axon.Layers.conv2d(w2, b2, strides: [1, 1])
    |> Axon.Activations.relu()
    |> Axon.Layers.max_pool2d(kernel_size: {3, 3}, strides: [2, 2])
    |> Axon.Layers.flatten()
    |> Axon.Layers.dense(w3, b3)
    |> Axon.Activations.softmax()
  end

  defn accuracy({w1, b1, w2, b2, w3, b3} = params, batch_images, batch_labels) do
    preds = predict(params, batch_images)
    Axon.Metrics.accuracy(preds, batch_labels)
  end

  defn loss({w1, b1, w2, b2, w3, b3}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2, w3, b3}, batch_images)
    -Nx.sum(Nx.mean(Nx.log(preds) * batch_labels, axes: [-1]))
  end

  defn update({w1, b1, w2, b2, w3, b3} = params, batch_images, batch_labels, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3} = grad(params, loss(params, batch_images, batch_labels))

    {
      w1 - grad_w1 * step,
      b1 - grad_b1 * step,
      w2 - grad_w2 * step,
      b2 - grad_b2 * step,
      w3 - grad_w3 * step,
      b3 - grad_b3 * step
    }
  end

  defn update_with_averages({_, _, _, _, _, _} = cur_params, imgs, tar, avg_loss, avg_accuracy, total) do
    batch_loss = loss(cur_params, imgs, tar)
    batch_accuracy = accuracy(cur_params, imgs, tar)
    avg_loss = avg_loss + batch_loss / total
    avg_accuracy = avg_accuracy + batch_accuracy / total
    {update(cur_params, imgs, tar, 0.01), avg_loss, avg_accuracy}
  end

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

        {:ok, {_status, _response, data}} = :httpc.request(:get, {base_url ++ zip, []}, [], [])
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
      |> Nx.reshape({n_images, 1, n_rows, n_cols}, names: [:batch, :channels, :height, :width])
      |> Nx.divide(255)
      |> Nx.to_batched_list(32)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    <<_::32, n_labels::32, labels::binary>> = unzip_cache_or_download(labels)

    train_labels =
      labels
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.to_batched_list(32)

    IO.puts("#{n_labels} labels\n")

    {train_images, train_labels}
  end

  def train_epoch(cur_params, imgs, labels) do
    total_batches = Enum.count(imgs)

    imgs
    |> Enum.zip(labels)
    |> Enum.reduce({cur_params, Nx.tensor(0.0), Nx.tensor(0.0)}, fn
      {imgs, tar}, {cur_params, avg_loss, avg_accuracy} ->
        update_with_averages(cur_params, imgs, tar, avg_loss, avg_accuracy, total_batches)
    end)
  end

  def train(imgs, labels, params, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: params do
      cur_params ->
        {time, {new_params, epoch_avg_loss, epoch_avg_acc}} =
          :timer.tc(__MODULE__, :train_epoch, [cur_params, imgs, labels])

        epoch_avg_loss =
          epoch_avg_loss
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        epoch_avg_acc =
          epoch_avg_acc
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        IO.puts("Epoch #{epoch} average loss: #{inspect(epoch_avg_loss)}")
        IO.puts("Epoch #{epoch} average accuracy: #{inspect(epoch_avg_acc)}")
        IO.puts("\n")
        new_params
    end
  end
end

{train_images, train_labels} =
  MNIST.download('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

IO.puts("Initializing parameters...\n")
params = MNIST.init_random_params()

IO.puts("Training MNIST for 10 epochs...\n\n")
final_params = MNIST.train(train_images, train_labels, params, epochs: 10)

IO.puts("Bring the parameters back from the device and print them")
final_params = Nx.backend_transfer(final_params)
IO.inspect(final_params)