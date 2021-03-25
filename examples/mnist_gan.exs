defmodule MNISTGAN do
  use Axon

  @default_defn_compiler {EXLA, keep_on_device: true}

  def generator do
    input({nil, 100})
    |> dense(256, activation: :leaky_relu)
    |> batch_norm()
    |> dense(512, activation: :leaky_relu)
    |> batch_norm()
    |> dense(1024, activation: :leaky_relu)
    |> batch_norm()
    |> dense(784, activation: :tanh)
  end

  defn init_generator, do: Axon.init(generator())

  def discriminator do
    input({nil, 28, 28})
    |> flatten()
    |> dense(512, activation: :tanh)
    |> dense(256, activation: :tanh)
    |> dense(2, activation: :softmax)
  end

  defn init_discriminator, do: Axon.init(discriminator())

  defn generate(params, latent) do
    Axon.predict(generator(), params, latent)
  end

  defn d_loss(d_params, images, targets) do
    preds = Axon.predict(discriminator(), d_params, images)
    Nx.mean(Axon.Losses.categorical_cross_entropy(preds, targets))
  end

  defn update_d(params, images, targets, step) do
    gradients = grad(params, &d_loss(&1, images, targets))

    transform({params, gradients, step},
      fn {params, gradients, step} ->
        params
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(gradients))
        |> Enum.map(fn {x, g} -> x + Axon.Updates.scale(g, -step) end)
        |> List.to_tuple()
      end)
  end

  defn g_loss(g_params, d_params, latent) do
    valid = Nx.iota({32, 2}, axis: 1, type: {:u, 8})
    g_preds = Axon.predict(generator(), g_params, latent)
    d_loss(d_params, g_preds, valid)
  end

  defn update_g(g_params, d_params, latent, step) do
    gradients = grad(g_params, &g_loss(&1, d_params, latent))

    transform({g_params, gradients, step},
      fn {params, gradients, step} ->
        params
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(gradients))
        |> Enum.map(fn {x, g} -> x + Axon.Updates.scale(g, -step) end)
        |> List.to_tuple()
      end)
  end

  defn update(g_params, d_params, images) do
    valid = Nx.iota({32, 2}, axis: 1, type: {:u, 8})
    fake = Nx.iota({32, 2}, axis: 1, type: {:u, 8}) |> Nx.reverse(axes: [1])

    latent = Nx.random_normal({32, 100})

    fake_images =
      Axon.predict(generator(), g_params, latent)
      |> Nx.reshape({32, 28, 28})

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
    imgs
    |> Enum.with_index()
    |> Enum.reduce({g_params, d_params}, fn
      {imgs, i}, {g_params, d_params} ->
        {new_g, new_d} =
          update(g_params, d_params, imgs)

        if rem(i, 50) == 0 do
          latent = Nx.random_normal({1, 100})
          IO.inspect Nx.to_heatmap generate(g_params, latent) |> Nx.reshape({1, 28, 28})
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

IO.inspect MNISTGAN.generator()
IO.inspect MNISTGAN.discriminator()

train_images = MNISTGAN.download('train-images-idx3-ubyte.gz')

IO.puts("Initializing parameters...\n")
d_params = MNISTGAN.init_discriminator()
g_params = MNISTGAN.init_generator()

{g_params, _d_params} = MNISTGAN.train(train_images, g_params, d_params, epochs: 10)

latent = Nx.random_uniform({1, 100})
IO.inspect Nx.to_heatmap MNISTGAN.generator(g_params, latent)