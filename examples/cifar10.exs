defmodule CIFAR do
  import Nx.Defn

  @default_defn_compiler {EXLA, run_options: [keep_on_device: true]}

  defn init_random_params() do
    # Feature extractor
    # TODO: Changing this to lecun normal makes it numerically unstable
    w1 = Axon.Initializers.normal(shape: {32, 3, 3, 3})
    b1 = Axon.Initializers.normal(shape: {1, 32, 1, 1}, scale: 1.0e-6)
    w2 = Axon.Initializers.normal(shape: {64, 32, 3, 3})
    b2 = Axon.Initializers.normal(shape: {1, 64, 1, 1}, scale: 1.0e-6)
    w3 = Axon.Initializers.normal(shape: {64, 64, 3, 3})
    b3 = Axon.Initializers.normal(shape: {1, 64, 1, 1}, scale: 1.0e-6)

    # FC Classifier
    w4 = Axon.Initializers.normal(shape: {36864, 64})
    b4 = Axon.Initializers.normal(shape: {64}, scale: 1.0e-6)
    w5 = Axon.Initializers.normal(shape: {64, 10})
    b5 = Axon.Initializers.normal(shape: {10}, scale: 1.0e-6)

    {w1, b1, w2, b2, w3, b3, w4, b4, w5, b5}
  end

  defn init_adam_state() do
    w1_mu = Axon.Initializers.zeros(shape: {32, 3, 3, 3})
    w1_nu = Axon.Initializers.zeros(shape: {32, 3, 3, 3})
    b1_mu = Axon.Initializers.zeros(shape: {1, 32, 1, 1})
    b1_nu = Axon.Initializers.zeros(shape: {1, 32, 1, 1})
    w2_mu = Axon.Initializers.zeros(shape: {64, 32, 3, 3})
    w2_nu = Axon.Initializers.zeros(shape: {64, 32, 3, 3})
    b2_mu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    b2_nu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    w3_mu = Axon.Initializers.zeros(shape: {64, 64, 3, 3})
    w3_nu = Axon.Initializers.zeros(shape: {64, 64, 3, 3})
    b3_mu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    b3_nu = Axon.Initializers.zeros(shape: {1, 64, 1, 1})
    w4_mu = Axon.Initializers.zeros(shape: {36864, 64})
    w4_nu = Axon.Initializers.zeros(shape: {36864, 64})
    b4_mu = Axon.Initializers.zeros(shape: {1, 64})
    b4_nu = Axon.Initializers.zeros(shape: {1, 64})
    w5_mu = Axon.Initializers.zeros(shape: {64, 10})
    w5_nu = Axon.Initializers.zeros(shape: {64, 10})
    b5_mu = Axon.Initializers.zeros(shape: {1, 10})
    b5_nu = Axon.Initializers.zeros(shape: {1, 10})

    {w1_mu, w1_nu, b1_mu, b1_nu, w2_mu, w2_nu, b2_mu, b2_nu, w3_mu, w3_nu, b3_mu, b3_nu, w4_mu,
     w4_nu, b4_mu, b4_nu, w5_mu, w5_nu, b5_mu, b5_nu}
  end

  defn predict({w1, b1, w2, b2, w3, b3, w4, b4, w5, b5}, batch) do
    batch
    |> Axon.Layers.conv(w1, b1, strides: [1, 1])
    |> Axon.Activations.relu()
    |> Axon.Layers.max_pool(kernel_size: {2, 2}, strides: [1, 1])
    |> Axon.Layers.conv(w2, b2, strides: [1, 1])
    |> Axon.Activations.relu()
    |> Axon.Layers.max_pool(kernel_size: {2, 2}, strides: [1, 1])
    |> Axon.Layers.conv(w3, b3, strides: [1, 1])
    |> Axon.Layers.flatten()
    |> Axon.Layers.dense(w4, b4)
    |> Axon.Activations.relu()
    |> Axon.Layers.dense(w5, b5)
    |> Axon.Activations.softmax()
  end

  defn accuracy({_, _, _, _, _, _, _, _, _, _} = params, batch_images, batch_labels) do
    preds = predict(params, batch_images)
    Axon.Metrics.accuracy(preds, batch_labels)
  end

  defn loss({w1, b1, w2, b2, w3, b3, w4, b4, w5, b5}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2, w3, b3, w4, b4, w5, b5}, batch_images)
    -Nx.sum(Nx.mean(Nx.log(preds) * batch_labels, axes: [-1]))
  end

  defn update(
         {w1, b1, w2, b2, w3, b3, w4, b4, w5, b5} = params,
         {w1_mu, w1_nu, b1_mu, b1_nu, w2_mu, w2_nu, b2_mu, b2_nu, w3_mu, w3_nu, b3_mu, b3_nu,
          w4_mu, w4_nu, b4_mu, b4_nu, w5_mu, w5_nu, b5_mu, b5_nu},
         batch_images,
         batch_labels,
         step,
         count
       ) do
    {grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3, grad_w4, grad_b4, grad_w5, grad_b5} =
      grad(params, loss(params, batch_images, batch_labels))

    {grad_w1, w1_mu, w1_nu} =
      Axon.Updates.scale_by_adam(grad_w1, w1_mu, w1_nu, count, b1: 0.9, b2: 0.999)

    {grad_b1, b1_mu, b1_nu} =
      Axon.Updates.scale_by_adam(grad_b1, b1_mu, b1_nu, count, b1: 0.9, b2: 0.999)

    {grad_w2, w2_mu, w2_nu} =
      Axon.Updates.scale_by_adam(grad_w2, w2_mu, w2_nu, count, b1: 0.9, b2: 0.999)

    {grad_b2, b2_mu, b2_nu} =
      Axon.Updates.scale_by_adam(grad_b2, b2_mu, b2_nu, count, b1: 0.9, b2: 0.999)

    {grad_w3, w3_mu, w3_nu} =
      Axon.Updates.scale_by_adam(grad_w3, w3_mu, w3_nu, count, b1: 0.9, b2: 0.999)

    {grad_b3, b3_mu, b3_nu} =
      Axon.Updates.scale_by_adam(grad_b3, b3_mu, b3_nu, count, b1: 0.9, b2: 0.999)

    {grad_w4, w4_mu, w4_nu} =
      Axon.Updates.scale_by_adam(grad_w4, w4_mu, w4_nu, count, b1: 0.9, b2: 0.999)

    {grad_b4, b4_mu, b4_nu} =
      Axon.Updates.scale_by_adam(grad_b4, b4_mu, b4_nu, count, b1: 0.9, b2: 0.999)

    {grad_w5, w5_mu, w5_nu} =
      Axon.Updates.scale_by_adam(grad_w5, w5_mu, w5_nu, count, b1: 0.9, b2: 0.999)

    {grad_b5, b5_mu, b5_nu} =
      Axon.Updates.scale_by_adam(grad_b5, b5_mu, b5_nu, count, b1: 0.9, b2: 0.999)

    {
      {
        w1 + Axon.Updates.scale(grad_w1, step: -step),
        b1 + Axon.Updates.scale(grad_b1, step: -step),
        w2 + Axon.Updates.scale(grad_w2, step: -step),
        b2 + Axon.Updates.scale(grad_b2, step: -step),
        w3 + Axon.Updates.scale(grad_w3, step: -step),
        b3 + Axon.Updates.scale(grad_b3, step: -step),
        w4 + Axon.Updates.scale(grad_w4, step: -step),
        b4 + Axon.Updates.scale(grad_b4, step: -step),
        w5 + Axon.Updates.scale(grad_w5, step: -step),
        b5 + Axon.Updates.scale(grad_b5, step: -step)
      },
      {w1_mu, w1_nu, b1_mu, b1_nu, w2_mu, w2_nu, b2_mu, b2_nu, w3_mu, w3_nu, b3_mu, b3_nu, w4_mu,
       w4_nu, b4_mu, b4_nu, w5_mu, w5_nu, b5_mu, b5_nu}
    }
  end

  defn update_with_averages(
         {_, _, _, _, _, _, _, _, _, _} = cur_params,
         {_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = optimizer_state,
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

    {update(cur_params, optimizer_state, imgs, tar, 0.01, count), avg_loss, avg_accuracy, count + 1}
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
        |> Enum.map(&Task.await(&1, 60000))
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

  def train_epoch(cur_params, cur_optimizer_state, cur_count, imgs, labels) do
    total_batches = Enum.count(imgs)

    imgs
    |> Enum.zip(labels)
    |> Enum.reduce(
      {{cur_params, cur_optimizer_state}, Nx.tensor(0.0), Nx.tensor(0.0), cur_count},
      fn
        {imgs, tar}, {{cur_params, cur_optimizer_state}, avg_loss, avg_accuracy, cur_count} ->
          # This augmentation maybe should be tied somewhere else?
          imgs = augment(imgs)
          update_with_averages(
            cur_params,
            cur_optimizer_state,
            imgs,
            tar,
            avg_loss,
            avg_accuracy,
            total_batches,
            cur_count
          )
      end
    )
  end

  def train(imgs, labels, params, optimizer_state, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: {params, optimizer_state, 1} do
      {cur_params, cur_optimizer_state, cur_count} ->
        {time, {{new_params, new_optimizer_state}, epoch_avg_loss, epoch_avg_acc, new_count}} =
          :timer.tc(__MODULE__, :train_epoch, [
            cur_params,
            cur_optimizer_state,
            cur_count,
            imgs,
            labels
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
        {new_params, new_optimizer_state, new_count}
    end
  end
end

{train_images, train_labels} = CIFAR.download('cifar-10-binary.tar.gz')

IO.puts("Initializing parameters...\n")
params = CIFAR.init_random_params()
optimizer_state = CIFAR.init_adam_state()

IO.puts("Training CIFAR for 10 epochs...\n\n")

{final_params, _, _} =
  CIFAR.train(train_images, train_labels, params, optimizer_state, epochs: 10)

IO.puts("Bring the parameters back from the device and print them")
final_params = Nx.backend_transfer(final_params)
IO.inspect(final_params)
