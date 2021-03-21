defmodule CIFAR do
  use Axon

  @default_defn_compiler {EXLA, run_options: [keep_on_device: true]}

  def model do
    input({32, 3, 32, 32})
    |> separable_conv2d(3, kernel_size: {3, 3}, activation: :relu)
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
    |> dense(10, activation: :log_softmax)
  end

  defn init, do: Axon.init(model())

  defn accuracy({_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = params, batch_images, batch_labels) do
    preds = Axon.predict(model(), params, batch_images)
    Axon.Metrics.accuracy(preds, batch_labels)
  end

  defn loss({_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = params, batch_images, batch_labels) do
    preds = Axon.predict(model(), params, batch_images)
    -Nx.sum(Nx.mean(preds * batch_labels, axes: [-1]))
  end

  defn update(
         {w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7, w8, b8, w9, b9} = params,
         batch_images,
         batch_labels,
         step
       ) do
    {grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3, grad_w4, grad_b4, grad_w5, grad_b5, grad_w6, grad_b6, grad_w7, grad_b7, grad_w8, grad_b8, grad_w9, grad_b9} =
      grad(params, &loss(&1, batch_images, batch_labels))

    {
      w1 - grad_w1 * step,
      b1 - grad_b1 * step,
      w2 - grad_w2 * step,
      b2 - grad_b2 * step,
      w3 - grad_w3 * step,
      b3 - grad_b3 * step,
      w4 - grad_w4 * step,
      b4 - grad_b4 * step,
      w5 - grad_w5 * step,
      b5 - grad_b5 * step,
      w6 - grad_w6 * step,
      b6 - grad_b6 * step,
      w7 - grad_w7 * step,
      b7 - grad_b7 * step,
      w8 - grad_w8 * step,
      b8 - grad_b8 * step,
      w9 - grad_w9 * step,
      b9 - grad_b9 * step
    }
  end

  defn update_with_averages(
         {_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _} = cur_params,
         imgs,
         tar,
         avg_loss,
         avg_accuracy,
         total
       ) do
    batch_loss = loss(cur_params, imgs, tar)
    batch_accuracy = accuracy(cur_params, imgs, tar)
    avg_loss = avg_loss + batch_loss / total
    avg_accuracy = avg_accuracy + batch_accuracy / total

    {update(cur_params, imgs, tar, 0.005), avg_loss, avg_accuracy}
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

  def train_epoch(cur_params, imgs, labels) do
    total_batches = Enum.count(imgs)

    imgs
    |> Enum.zip(labels)
    |> Enum.reduce(
      {cur_params, Nx.tensor(0.0), Nx.tensor(0.0)},
      fn
        {imgs, tar}, {cur_params, avg_loss, avg_accuracy} ->
          # This augmentation maybe should be tied somewhere else?
          imgs = augment(imgs)
          update_with_averages(
            cur_params,
            imgs,
            tar,
            avg_loss,
            avg_accuracy,
            total_batches
          )
      end
    )
  end

  def train(imgs, labels, params, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: params do
      cur_params ->
        {time, {new_params, epoch_avg_loss, epoch_avg_acc}} =
          :timer.tc(__MODULE__, :train_epoch, [
            cur_params,
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
        new_params
    end
  end
end

{train_images, train_labels} = CIFAR.download('cifar-10-binary.tar.gz')

IO.puts("Initializing parameters...\n")
params = CIFAR.init()

IO.puts("Training CIFAR for 10 epochs...\n\n")

final_params =
  CIFAR.train(train_images, train_labels, params, epochs: 20)

IO.puts("Bring the parameters back from the device and print them")
final_params = Nx.backend_transfer(final_params)
IO.inspect(final_params)
