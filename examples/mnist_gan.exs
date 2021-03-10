defmodule MNIST do
  import Nx.Defn
  require Logger

  @default_defn_compiler {EXLA, max_float_type: {:f, 32}}

  defn init_g_random_params do
    w1 = Axon.Initializers.uniform(shape: {100, 256})
    b1 = Axon.Initializers.uniform(shape: {256})
    w2 = Axon.Initializers.uniform(shape: {256, 512})
    b2 = Axon.Initializers.uniform(shape: {512})
    w3 = Axon.Initializers.uniform(shape: {512, 1024})
    b3 = Axon.Initializers.uniform(shape: {1024})
    w4 = Axon.Initializers.uniform(shape: {1024, 784})
    b4 = Axon.Initializers.uniform(shape: {784})
    {w1, b1, w2, b2, w3, b3, w4, b4}
  end

  defn init_d_random_params do
    w1 = Axon.Initializers.uniform(shape: {784, 512})
    b1 = Axon.Initializers.uniform(shape: {512})
    w2 = Axon.Initializers.uniform(shape: {512, 256})
    b2 = Axon.Initializers.uniform(shape: {256})
    w3 = Axon.Initializers.uniform(shape: {256, 2})
    b3 = Axon.Initializers.uniform(shape: {2})
    {w1, b1, w2, b2, w3, b3}
  end

  defn g_predict({w1, b1, w2, b2, w3, b3, w4, b4}, latent) do
    batch_size = transform(Nx.shape(latent), fn {batch_size, _} -> batch_size end)
    latent
    |> Axon.Layers.dense(w1, b1)
    |> Axon.Activations.tanh()
    |> Axon.Layers.dense(w2, b2)
    |> Axon.Activations.tanh()
    |> Axon.Layers.dense(w3, b3)
    |> Axon.Activations.tanh()
    |> Axon.Layers.dense(w4, b4)
    |> Axon.Activations.tanh()
    |> Nx.reshape({batch_size, 28, 28})
  end

  defn d_predict({w1, b1, w2, b2, w3, b3}, image) do
    batch_size = transform(Nx.shape(image), fn {batch_size, _, _} -> batch_size end)
    image
    |> Nx.reshape({batch_size, 784})
    |> Axon.Layers.dense(w1, b1)
    |> Axon.Activations.tanh()
    |> Axon.Layers.dense(w2, b2)
    |> Axon.Activations.tanh()
    |> Axon.Layers.dense(w3, b3)
    |> Axon.Activations.softmax()
  end

  defn cross_entropy_loss(y_true, y_false) do
    -Nx.mean(Nx.sum(y_true * y_false, axes: [-1]))
  end

  defn d_loss({_, _, _, _, _, _} = d_params, images, targets) do
    preds = d_predict(d_params, images)
    cross_entropy_loss(preds, targets)
  end

  defn update_d({w1, b1, w2, b2, w3, b3} = d_params, images, targets, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3} =
      grad({w1, b1, w2, b2, w3, b3}, d_loss(d_params, images, targets))
    {
      w1 - grad_w1 * step,
      b1 - grad_b1 * step,
      w2 - grad_w2 * step,
      b2 - grad_b2 * step,
      w3 - grad_w3 * step,
      b3 - grad_b3 * step
    }
  end

  defn g_loss({_, _, _, _, _, _, _, _} = g_params, {_, _, _, _, _, _} = d_params, latent) do
    valid = Nx.iota({32, 2}, axis: 1)
    g_preds = g_predict(g_params, latent)
    d_loss(d_params, g_preds, valid)
  end

  defn update_g({w1, b1, w2, b2, w3, b3, w4, b4} = g_params, {_, _, _, _, _, _} = d_params, latent, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3, grad_w4, grad_b4} =
      grad({w1, b1, w2, b2, w3, b3, w4, b4}, g_loss(g_params, d_params, latent))

    {
      w1 - grad_w1 * step,
      b1 - grad_b1 * step,
      w2 - grad_w2 * step,
      b2 - grad_b2 * step,
      w3 - grad_w3 * step,
      b3 - grad_b3 * step,
      w4 - grad_w4 * step,
      b4 - grad_b4 * step
    }
  end

  defn update({gw1, gb1, gw2, gb2, gw3, gb3, gw4, gb4} = g_params, {dw1, db1, dw2, db2, dw3, db3} = d_params, images) do
    valid = Nx.iota({32, 2}, axis: 1)
    fake = Nx.iota({32, 2}, axis: 1) |> Nx.reverse()
    latent = Nx.random_normal({32, 100})

    fake_images =
      g_params
      |> g_predict(latent)

    new_d_params =
      d_params
      |> update_d(images, valid, 0.01)
      |> update_d(fake_images, fake, 0.01)

    new_g_params =
      g_params
      |> update_g(new_d_params, latent, 0.05)

    {new_g_params, new_d_params}
  end

  def train_epoch(g_params, d_params, imgs) do
    total_batches = Enum.count(imgs)
    imgs
    |> Enum.with_index()
    |> Enum.reduce({g_params, d_params}, fn
      {imgs, i}, {g_params, d_params} ->
        IO.puts("Batch: #{i+1}\n")
        {new_g, new_d} =
          update(g_params, d_params, imgs)

        if rem(i, 50) == 0 do
          latent = Nx.random_normal({1, 100})
          IO.inspect Nx.to_heatmap g_predict(g_params, latent)
        end

        {new_g, new_d}
    end)
  end

  def train(imgs, g_params, d_params, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: {g_params, d_params} do
      {g_params, d_params} ->
        {time, {new_g_params, new_d_params}} =
          :timer.tc(__MODULE__, :train_epoch, [g_params, d_params, imgs])

        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        {new_g_params, new_d_params}
    end
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

  def download(images) do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
      unzip_cache_or_download(images)

    train_images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows, n_cols})
      |> Nx.divide(255)
      |> Nx.to_batched_list(32)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    train_images
  end
end

train_images =
  MNIST.download('train-images-idx3-ubyte.gz')

IO.puts("Initializing parameters...\n")
d_params = MNIST.init_d_random_params()
g_params = MNIST.init_g_random_params()

{g_params, d_params} = MNIST.train(train_images, g_params, d_params, epochs: 10)

latent = Nx.random_uniform({1, 100})
IO.inspect Nx.to_heatmap MNIST.g_predict(g_params, latent)