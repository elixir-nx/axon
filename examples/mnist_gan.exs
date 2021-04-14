defmodule MNISTGAN do
  require Axon
  require Axon.Updates
  import Nx.Defn

  @default_defn_compiler {EXLA, keep_on_device: true}

  def generator do
    Axon.input({nil, 100})
    |> Axon.dense(256, activation: :leaky_relu)
    |> Axon.batch_norm()
    |> Axon.dense(512, activation: :leaky_relu)
    |> Axon.batch_norm()
    |> Axon.dense(1024, activation: :leaky_relu)
    |> Axon.batch_norm()
    |> Axon.dense(784, activation: :tanh)
  end

  def discriminator do
    Axon.input({nil, 28, 28})
    |> Axon.flatten()
    |> Axon.dense(512, activation: :tanh)
    |> Axon.dense(256, activation: :tanh)
    |> Axon.dense(2, activation: :softmax)
  end

  defn generate(params, latent) do
    Axon.predict(generator(), params, latent)
  end

  defn d_loss(d_params, images, targets) do
    preds = Axon.predict(discriminator(), d_params, images)
    Axon.Losses.categorical_cross_entropy(preds, targets, reduction: :mean)
  end

  defn update_d(params, d_optim_state, images, targets, update_fn) do
    gradients = grad(params, &d_loss(&1, images, targets))
    {updates, new_optim_state} = update_fn.(gradients, d_optim_state, params)
    {Axon.Updates.apply_updates(params, updates), new_optim_state}
  end

  defn g_loss(g_params, d_params, latent) do
    valid = Nx.iota({32, 2}, axis: 1, type: {:u, 8})
    g_preds = Axon.predict(generator(), g_params, latent)
    d_loss(d_params, g_preds, valid)
  end

  defn update_g(g_params, g_optim_state, d_params, update_fn, latent) do
    gradients = grad(g_params, &g_loss(&1, d_params, latent))

    {updates, new_optim_state} = update_fn.(gradients, g_optim_state, g_params)
    {Axon.Updates.apply_updates(g_params, updates), new_optim_state}
  end

  def update(g_params, g_optim_state, d_params, d_optim_state, update_fn, images) do
    valid = Nx.iota({32, 2}, axis: 1, type: {:u, 8})
    fake = Nx.iota({32, 2}, axis: 1, type: {:u, 8}) |> Nx.reverse(axes: [1])

    latent = Nx.random_normal({32, 100})

    fake_images =
      g_params
      |> generate(latent)
      |> Nx.reshape({32, 28, 28})

    {new_d_params, new_d_state} =
      d_params
      |> update_d(d_optim_state, images, valid, update_fn)

    {new_d_params, new_d_state} =
      new_d_params
      |> update_d(new_d_state, fake_images, fake, update_fn)

    {new_g_params, new_g_state} =
      g_params
      |> update_g(g_optim_state, new_d_params, update_fn, latent)

    {new_g_params, new_g_state, new_d_params, new_d_state}
  end

  def train_epoch(g_params, g_state, d_params, d_state, update_fn, imgs) do
    imgs
    |> Enum.with_index()
    |> Enum.reduce({g_params, g_state, d_params, d_state}, fn
      {imgs, i}, {g_params, g_state, d_params, d_state} ->
        {new_g, g_state, new_d, d_state} =
          update(g_params, g_state, d_params, d_state, update_fn, imgs)

        IO.write("\rBatch: #{i}")

        if rem(i, 50) == 0 do
          latent = Nx.random_normal({1, 100})
          IO.inspect Nx.to_heatmap generate(new_g, latent) |> Nx.reshape({1, 28, 28})
        end

        {new_g, g_state, new_d, d_state}
    end)
  end

  def train(imgs, g_params, g_state, d_params, d_state, update_fn, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: {g_params, g_state, d_params, d_state} do
      {g_params, g_state, d_params, d_state} ->
        {time, {new_g_params, new_g_state, new_d_params, new_d_state}} =
          :timer.tc(__MODULE__, :train_epoch, [g_params, g_state, d_params, d_state, update_fn, imgs])

        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        {new_g_params, new_g_state, new_d_params, new_d_state}
    end
  end
end

require Axon

generator = MNISTGAN.generator() |> IO.inspect
discriminator = MNISTGAN.discriminator() |> IO.inspect

train_images = Axon.Data.MNIST.download_images()

IO.puts("Initializing parameters...\n")

{init_fn, update_fn} = Axon.Optimizers.adam(0.005)

d_params = Axon.init(discriminator, compiler: EXLA)
d_state = Nx.Defn.jit(init_fn, [d_params], compiler: EXLA)
g_params = Axon.init(generator, compiler: EXLA)
g_state = Nx.Defn.jit(init_fn, [g_params], compiler: EXLA)

{g_params, _d_params} = MNISTGAN.train(train_images, g_params, g_state, d_params, d_state, update_fn, epochs: 10)

latent = Nx.random_uniform({1, 100})
IO.inspect Nx.to_heatmap MNISTGAN.generator(g_params, latent)
