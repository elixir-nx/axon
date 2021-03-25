defmodule CIFAR do
  use Axon

  @default_defn_compiler {EXLA, run_options: [keep_on_device: true]}

  def model do
    input({32, 3, 32, 32})
    |> conv(32, kernel_size: {3, 3}, activation: :relu)
    |> batch_norm()
    |> max_pool(kernel_size: {2, 2})
    |> conv(64, kernel_size: {3, 3}, activation: :relu)
    |> spatial_dropout()
    |> batch_norm()
    |> max_pool(kernel_size: {2, 2})
    |> conv(64, kernel_size: {3, 3}, activation: :relu)
    |> batch_norm()
    |> flatten()
    |> dense(64, activation: :relu)
    |> dropout()
    |> dense(10, activation: :softmax)
  end

  defn init, do: Axon.init(model())

  defn init_adam do
    w1_mu = Axon.Initializers.zeros(shape: {32, 3, 3, 3})
    w1_nu = Axon.Initializers.zeros(shape: {32, 3, 3, 3})
    b1_mu = Axon.Initializers.zeros(shape: {1, 32, 1, 1})
    b1_nu = Axon.Initializers.zeros(shape: {1, 32, 1, 1})
    w2_mu = Axon.Initializers.zeros(shape: {1, 32, 1, 1})
    w2_nu = Axon.Initializers.zeros(shape: {1, 32, 1, 1})
    b2_mu = Axon.Initializers.zeros(shape: {1, 32, 1, 1})
    b2_nu = Axon.Initializers.zeros(shape: {1, 32, 1, 1})
    w3_mu = Axon.Initializers.zeros(shape: {64, 32, 3, 3})
    w3_nu = Axon.Initializers.zeros(shape: {64, 32, 3, 3})
    b3_mu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    b3_nu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    w4_mu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    w4_nu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    b4_mu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    b4_nu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    w5_mu = Axon.Initializers.zeros(shape: {64, 64, 3, 3})
    w5_nu = Axon.Initializers.zeros(shape: {64, 64, 3, 3})
    b5_mu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    b5_nu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    w6_mu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    w6_nu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    b6_mu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    b6_nu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    w7_mu = Axon.Initializers.zeros(shape: {36864, 64})
    w7_nu = Axon.Initializers.zeros(shape: {36864, 64})
    b7_mu = Axon.Initializers.zeros(shape: {1, 64})
    b7_nu = Axon.Initializers.zeros(shape: {1, 64})
    w8_mu = Axon.Initializers.zeros(shape: {64, 10})
    w8_nu = Axon.Initializers.zeros(shape: {64, 10})
    b8_mu = Axon.Initializers.zeros(shape: {1, 10})
    b8_nu = Axon.Initializers.zeros(shape: {1, 10})

    {w1_mu, w1_nu, b1_mu, b1_nu, w2_mu, w2_nu, b2_mu, b2_nu, w3_mu, w3_nu, b3_mu, b3_nu, w4_mu, w4_nu, b4_mu, b4_nu,
      w5_mu, w5_nu, b5_mu, b5_nu, w6_mu, w6_nu, b6_mu, b6_nu, w7_mu, w7_nu, b7_mu, b7_nu, w8_mu, w8_nu, b8_mu, b8_nu}
  end

  defn accuracy({_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = params, batch_images, batch_labels) do
    preds = Axon.predict(model(), params, batch_images)
    Axon.Metrics.accuracy(preds, batch_labels)
  end

  defn loss({_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = params, batch_images, batch_labels) do
    preds = Axon.predict(model(), params, batch_images)
    Nx.mean(Axon.Losses.categorical_cross_entropy(batch_labels, preds))
  end

  defn update(
         {_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = params,
         {_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = optim_params,
         batch_images,
         batch_labels,
         step,
         count
       ) do

    gradients = grad(params, &loss(&1, batch_images, batch_labels))

    transform({gradients, params, optim_params, count, step},
      fn {updates, params, optim_params, count, step} ->
        {new_params, new_optim_params} =
          updates
            |> Tuple.to_list()
            |> Enum.zip(Tuple.to_list(params))
            |> Enum.with_index()
            |> Enum.reduce({[], []},
                fn {{update, param}, i}, {new_params, new_optim_params} ->
                  {mu, nu} = {elem(optim_params, i * 2), elem(optim_params, (i * 2) + 1)}
                  {update, new_mu, new_nu} = Axon.Updates.scale_by_adam(update, mu, nu, count)
                  {[param + Axon.Updates.scale(update, -step) | new_params], [new_nu, new_mu | new_optim_params]}
                end
              )
        {List.to_tuple(Enum.reverse(new_params)), List.to_tuple(Enum.reverse(new_optim_params))}
      end
    )
  end

  defn update_with_averages(
         {_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = cur_params,
         {_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = optim_params,
         imgs,
         tar,
         avg_loss,
         avg_accuracy,
         total,
         count
       ) do
    batch_loss = loss(cur_params, imgs, tar)
    batch_accuracy = accuracy(cur_params, imgs, tar)
    avg_loss = avg_loss + batch_loss / total
    avg_accuracy = avg_accuracy + batch_accuracy / total

    {update(cur_params, optim_params, imgs, tar, 0.001, count), avg_loss, avg_accuracy}
  end

  defn random_flip_left_right(img) do
    val = Nx.random_uniform({})
    if Nx.greater_equal(val, 0.5) do
      img
      |> Nx.reverse(axes: [-1])
    else
      img
    end
  end

  defn random_flip_up_down(img) do
    val = Nx.random_uniform({})
    if Nx.greater_equal(val, 0.5) do
      img
      |> Nx.reverse(axes: [-2])
    else
      img
    end
  end

  defn augment(imgs) do
    imgs
    |> random_flip_up_down()
    |> random_flip_left_right()
  end

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

  def train_epoch(cur_params, optim_params, imgs, labels, count) do
    total_batches = Enum.count(imgs)

    imgs
    |> Enum.zip(labels)
    |> Enum.reduce(
      {{cur_params, optim_params}, Nx.tensor(0.0), Nx.tensor(0.0), count},
      fn
        {imgs, tar}, {{cur_params, cur_optim_params}, avg_loss, avg_accuracy, count} ->
          # This augmentation maybe should be tied somewhere else?
          imgs = augment(imgs)
          {{params, optim_params}, avg_loss, avg_acc} = update_with_averages(
            cur_params,
            cur_optim_params,
            imgs,
            tar,
            avg_loss,
            avg_accuracy,
            total_batches,
            count
          )
          {{params, optim_params}, avg_loss, avg_acc, count + 1}
      end
    )
  end

  def train(imgs, labels, params, optim_params, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: {params, optim_params, 0} do
      {cur_params, cur_optim_params, count} ->
        {time, {{new_params, new_optim_params}, epoch_avg_loss, epoch_avg_acc, count}} =
          :timer.tc(__MODULE__, :train_epoch, [
            cur_params,
            cur_optim_params,
            imgs,
            labels,
            count
          ])

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
        {new_params, new_optim_params, count}
    end
  end
end

IO.inspect CIFAR.model()

{train_images, train_labels} = CIFAR.download('cifar-10-binary.tar.gz')

IO.puts("Initializing parameters...\n")
params = CIFAR.init()
optim_params = CIFAR.init_adam()

IO.puts("Training CIFAR for 10 epochs...\n\n")

{final_params, _, _} = CIFAR.train(train_images, train_labels, params, optim_params, epochs: 20)

IO.puts("Bring the parameters back from the device and print them")
final_params = Nx.backend_transfer(final_params)
IO.inspect(final_params)
