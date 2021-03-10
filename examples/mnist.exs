defmodule MNIST do
  use Axon.Trainer

  @default_defn_compiler {EXLA, keep_on_device: true}

  defn init_random_params do
    w1 = Axon.Initializers.lecun_normal(shape: {784, 128})
    b1 = Axon.Initializers.lecun_normal(shape: {1, 128})
    w2 = Axon.Initializers.lecun_normal(shape: {128, 10})
    b2 = Axon.Initializers.lecun_normal(shape: {1, 10})
    {w1, b1, w2, b2}
  end

  defn init_adam_state do
    w1_mu = Axon.Initializers.zeros(shape: {784, 128})
    w1_nu = Axon.Initializers.zeros(shape: {784, 128})
    b1_mu = Axon.Initializers.zeros(shape: {1, 128})
    b1_nu = Axon.Initializers.zeros(shape: {1, 128})
    w2_mu = Axon.Initializers.zeros(shape: {128, 10})
    w2_nu = Axon.Initializers.zeros(shape: {128, 10})
    b2_mu = Axon.Initializers.zeros(shape: {1, 10})
    b2_nu = Axon.Initializers.zeros(shape: {1, 10})
    {w1_mu, w1_nu, b1_mu, b1_nu, w2_mu, w2_nu, b2_mu, b2_nu}
  end

  defn predict({w1, b1, w2, b2}, batch) do
    batch
    |> Axon.Layers.dense(w1, b1)
    |> Axon.Activations.relu()
    |> Axon.Layers.dense(w2, b2)
    |> Axon.Activations.softmax()
    |> Nx.log()
  end

  defn accuracy({w1, b1, w2, b2}, batch_images, batch_labels) do
    Nx.mean(
      Nx.equal(
        Nx.argmax(batch_labels, axis: 1),
        Nx.argmax(predict({w1, b1, w2, b2}, batch_images), axis: 1)
      )
    )
  end

  defn loss({w1, b1, w2, b2}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2}, batch_images)
    -Nx.mean(Nx.sum(batch_labels * preds, axes: [-1]))
  end

  @impl true
  defn update({w1, b1, w2, b2}, {w1_mu, w1_nu, b1_mu, b1_nu, w2_mu, w2_nu, b2_mu, b2_nu}, batch_images, batch_labels, step, count) do
    {grad_w1, grad_b1, grad_w2, grad_b2} =
      grad({w1, b1, w2, b2}, loss({w1, b1, w2, b2}, batch_images, batch_labels))

    {grad_w1, w1_mu, w1_nu} = Axon.Updates.scale_by_adam(grad_w1, w1_mu, w1_nu, count, b1: 0.9)
    {grad_b1, b1_mu, b1_nu} = Axon.Updates.scale_by_adam(grad_b1, b1_mu, b1_nu, count, b1: 0.9)
    {grad_w2, w2_mu, w2_nu} = Axon.Updates.scale_by_adam(grad_w2, w2_mu, w2_nu, count, b1: 0.9)
    {grad_b2, b2_mu, b2_nu} = Axon.Updates.scale_by_adam(grad_b2, b2_mu, b2_nu, count, b1: 0.9)

    {
      {
        w1 + Axon.Updates.scale(grad_w1, step: -step),
        b1 + Axon.Updates.scale(grad_b1, step: -step),
        w2 + Axon.Updates.scale(grad_w2, step: -step),
        b2 + Axon.Updates.scale(grad_b2, step: -step)
      },
      {w1_mu, w1_nu, b1_mu, b1_nu, w2_mu, w2_nu, b2_mu, b2_nu},
      count + 1
    }
  end

  defn update_with_averages({_, _, _, _} = cur_params, {_, _, _, _, _, _, _, _} = cur_opt_state, imgs, tar, avg_loss, avg_accuracy, total, count) do
    batch_loss = loss(cur_params, imgs, tar)
    batch_accuracy = accuracy(cur_params, imgs, tar)
    avg_loss = avg_loss + batch_loss / total
    avg_accuracy = avg_accuracy + batch_accuracy / total
    {update(cur_params, cur_opt_state, imgs, tar, 0.01, count), avg_loss, avg_accuracy, count + 1}
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

{train_images, train_labels} =
  MNIST.download('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

IO.puts("Initializing parameters...\n")
params = MNIST.init_random_params()
adam_state = MNIST.init_adam_state()

IO.puts("Training MNIST for 10 epochs...\n\n")
{final_params, _} = MNIST.train(params, adam_state, train_images, train_labels, epochs: 10)

IO.inspect(Nx.backend_transfer(final_params))