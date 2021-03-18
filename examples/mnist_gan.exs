defmodule MNISTGAN do
  use Axon

  @default_defn_compiler {EXLA, keep_on_device: true}

  model generator do
    input({32, 100})
    |> dense(256, activation: :tanh)
    |> batch_norm()
    |> dense(512, activation: :tanh)
    |> batch_norm()
    |> dense(1024, activation: :tanh)
    |> batch_norm()
    |> dense(784, activation: :tanh)
  end

  model discriminator do
    input({32, 28, 28})
    |> flatten()
    |> dense(512, activation: :tanh)
    |> batch_norm()
    |> dense(256, activation: :tanh)
    |> batch_norm()
    |> dense(2, activation: :log_softmax)
  end

  defn cross_entropy_loss(y_true, y_false) do
    -Nx.mean(Nx.sum(y_true * y_false, axes: [-1]))
  end

  defn d_loss({_, _, _, _, _, _, _, _, _, _} = d_params, {_, _, _, _} = d_vars, images, targets) do
    {preds, _} = discriminator(d_params, d_vars, images)
    cross_entropy_loss(preds, targets)
  end

  defn update_d({w1, b1, gamma1, beta1, w2, b2, gamma2, beta2, w3, b3} = d_params, {_, _, _, _} = d_vars, images, targets, step) do
    {grad_w1, grad_b1, grad_gamma1, grad_beta1, grad_w2, grad_b2, grad_gamma2, grad_beta2, grad_w3, grad_b3} =
      grad(d_params, &d_loss(&1, d_vars, images, targets))

    {
       w1 - grad_w1 * step,
       b1 - grad_b1 * step,
       gamma1 - grad_gamma1 * step,
       beta1 - grad_beta1 * step,
       w2 - grad_w2 * step,
       b2 - grad_b2 * step,
       gamma2 - grad_gamma2 * step,
       beta2 - grad_beta2 * step,
       w3 - grad_w3 * step,
       b3 - grad_b3 * step
     }
  end

  defn g_loss(
         {_, _, _, _, _, _, _, _, _, _, _, _, _, _} = g_params,
         {_, _, _, _, _, _} = g_vars,
         {_, _, _, _, _, _, _, _, _, _} = d_params,
         {_, _, _, _} = d_vars,
         latent
       ) do
    valid = Nx.iota({32, 2}, axis: 1)
    {g_preds, _} = generator(g_params, g_vars, latent)
    d_loss(d_params, d_vars, g_preds, valid)
  end

  defn update_g(
         {w1, b1, gamma1, beta1, w2, b2, gamma2, beta2, w3, b3, gamma3, beta3, w4, b4} = g_params,
         {_, _, _, _, _, _} = g_vars,
         {_, _, _, _, _, _, _, _, _, _} = d_params,
         {_, _, _, _} = d_vars,
         latent,
         step
       ) do
    {grad_w1, grad_b1, grad_gamma1, grad_beta1, grad_w2, grad_b2, grad_gamma2, grad_beta2, grad_w3, grad_b3, grad_gamma3, grad_beta3, grad_w4, grad_b4} =
      grad(g_params, &g_loss(&1, g_vars, d_params, d_vars, latent))

    {
       w1 - grad_w1 * step,
       b1 - grad_b1 * step,
       gamma1 - grad_gamma1 * step,
       beta1 - grad_beta1 * step,
       w2 - grad_w2 * step,
       b2 - grad_b2 * step,
       gamma2 - grad_gamma2 * step,
       beta2 - grad_beta2 * step,
       w3 - grad_w3 * step,
       b3 - grad_b3 * step,
       gamma3 - grad_gamma3 * step,
       beta3 - grad_beta3 * step,
       w4 - grad_w4 * step,
       b4 - grad_b4 * step
     }
  end

  def update(
         {_, _, _, _, _, _, _, _, _, _, _, _, _, _} = g_params,
         {_, _, _, _, _, _} = g_vars,
         {_, _, _, _, _, _, _, _, _, _} = d_params,
         {_, _, _, _} = d_vars,
         images
       ) do
    valid = Nx.iota({32, 2}, axis: 1)
    fake = Nx.iota({32, 2}, axis: 1) |> Nx.reverse()
    latent = Nx.random_normal({32, 100})

    {fake_images, new_g_vars} =
      g_params
      |> generator(g_vars, latent)

    fake_images = fake_images |> Nx.reshape({32, 28, 28})

    new_d_params = update_d(d_params, d_vars, images, valid, 0.01)
    new_d_params = update_d(new_d_params, d_vars, fake_images, fake, 0.01)

    new_g_params =
      g_params
      |> update_g(new_g_vars, new_d_params, d_vars, latent, 0.05)

    {_, new_d_vars} = discriminator(d_params, d_vars, Nx.concatenate([fake_images, images]))

    {new_g_params, new_g_vars, new_d_params, new_d_vars}
  end

  def train_epoch(g_params, g_vars, d_params, d_vars, imgs) do
    imgs
    |> Enum.with_index()
    |> Enum.reduce({g_params, g_vars, d_params, d_vars}, fn
      {imgs, i}, {g_params, g_vars, d_params, d_vars} ->
        {new_g, new_g_vars, new_d, new_d_vars} = update(g_params, g_vars, d_params, d_vars, imgs)

        if rem(i, 50) == 0 do
          latent = Nx.random_normal({1, 100})
          IO.inspect(Nx.to_heatmap(elem(generator(g_params, g_vars, latent), 0) |> Nx.reshape({1, 28, 28})))
        end

        {new_g, new_g_vars, new_d, new_d_vars}
    end)
  end

  def train(imgs, g_params, g_vars, d_params, d_vars, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: {g_params, g_vars, d_params, d_vars} do
      {g_params, g_vars, d_params, d_vars} ->
        {time, {new_g_params, new_g_vars, new_d_params, new_d_vars}} =
          :timer.tc(__MODULE__, :train_epoch, [g_params, g_vars, d_params, d_vars, imgs])

        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        {new_g_params, new_g_vars, new_d_params, new_d_vars}
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

train_images = MNISTGAN.download('train-images-idx3-ubyte.gz')

IO.puts("Initializing parameters...\n")
{d_params, d_vars} = MNISTGAN.init_discriminator()
{g_params, g_vars} = MNISTGAN.init_generator()

{g_params, g_vars, _d_params, _d_vars} = MNISTGAN.train(train_images, g_params, g_vars, d_params, d_vars, epochs: 10)

latent = Nx.random_uniform({1, 100})
IO.inspect(Nx.to_heatmap(MNISTGAN.generator(g_params, g_vars, latent)))
