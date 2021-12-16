# Title: KMNIST - Metric Learning

# ── Section ──

Mix.install([
  {:axon, github: "elixir-nx/axon"},
  {:exla, github: "elixir-nx/nx", sparse: "exla"},
  {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.3"}
])

{train_images, train_labels} = Scidata.KuzushijiMNIST.download()
{test_images, test_labels} = Scidata.KuzushijiMNIST.download_test()

{train_images_bin, train_images_type, train_images_shape} = train_images

train_images_tensor =
  train_images_bin
  |> Nx.from_binary(train_images_type)
  |> Nx.reshape(train_images_shape)
  |> Nx.divide(255.0)

{train_labels_bin, train_labels_type, train_labels_shape} = train_labels

train_labels_tensor =
  train_labels_bin
  |> Nx.from_binary(train_labels_type)
  |> Nx.reshape(train_labels_shape)

train_images_tensor[0] |> Nx.to_heatmap()

# Create a mapping between class labels and the images in each label. This will be used to sample and create the training dataset.

{test_images_bin, test_images_type, test_images_shape} = test_images

test_images_tensor =
  test_images_bin
  |> Nx.from_binary(test_images_type)
  |> Nx.reshape(test_images_shape)
  |> Nx.divide(255.0)

{test_labels_bin, test_labels_type, test_labels_shape} = test_labels

test_labels_tensor =
  test_labels_bin
  |> Nx.from_binary(test_labels_type)
  |> Nx.reshape(test_labels_shape)

class_idx_to_train_idxs =
  train_labels_tensor
  |> Nx.to_flat_list()
  |> Enum.with_index()
  |> Enum.reduce(%{}, fn {record, index}, acc ->
    old_records = Map.get(acc, record, [])
    Map.put(acc, record, [index | old_records])
  end)

class_idx_to_test_idxs =
  test_labels_tensor
  |> Nx.to_flat_list()
  |> Enum.with_index()
  |> Enum.reduce(%{}, fn {record, index}, acc ->
    old_records = Map.get(acc, record, [])
    Map.put(acc, record, [index | old_records])
  end)

defmodule KuzushijiMNISTMetricLearning do
  require Axon
  alias Axon.Loop.State
  import Nx.Defn

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 1, 28, 28})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
  end

  defp l2_normalize(input_tensor) do
    den =
      Nx.multiply(input_tensor, input_tensor) |> Nx.sum(axes: [-1], keep_axes: true) |> Nx.sqrt()

    Nx.divide(input_tensor, den)
  end

  def build_model(input_shape) do
    Axon.input({nil, 28, 28, 3})
    |> Axon.conv(32, kernel_size: {3, 3}, strides: 2, activation: :relu)
    |> Axon.conv(64, kernel_size: {3, 3}, strides: 2, activation: :relu)
    |> Axon.conv(128, kernel_size: {3, 3}, strides: 2, activation: :relu)
    |> Axon.global_avg_pool()
    |> Axon.dense(8)

    # |> l2_normalize()
  end

  defnp running_average(avg, obs, i) do
    avg
    |> Nx.multiply(i)
    |> Nx.add(obs)
    |> Nx.divide(Nx.add(i, 1))
  end

  defn init(model, init_optim) do
    params = Axon.init(model)

    %{
      iteration: Nx.tensor(0),
      model_state: params,
      optimizer_state: init_optim.(params),
      loss: Nx.tensor(0.0)
    }
  end

  defn batch_step(model, optim, real_images, state) do
    iter = state[:iteration]
    params = state[:model_state]
    IO.puts(iter)
    # Add code to compute cosine similarity for metric learning
  end

  defp train_loop(model) do
    {init_optim, optim} = Axon.Optimizers.adam(2.0e-3, b1: 0.5)

    step = &batch_step(model, optim, &1, &2)
    init = fn -> init(model, init_optim) end

    Axon.Loop.loop(step, init)
  end

  defp log_iteration(state) do
    %State{epoch: epoch, iteration: iter, step_state: pstate} = state

    loss = "Loss: #{:io_lib.format('~.5f', [Nx.to_scalar(pstate[:loss])])}"

    "\rEpoch: #{Nx.to_scalar(epoch)}, batch: #{Nx.to_scalar(iter)} #{loss}"
  end

  def run(train_tensor, test_tensor) do
    model = build_model({nil, 28, 28, 3})

    train_loop(model)
    |> Axon.Loop.log(:iteration_completed, &log_iteration/1, :stdio, every: 50)
    |> Axon.Loop.run(train_tensor, epochs: 10, compiler: EXLA)
  end
end

model = KuzushijiMNISTMetricLearning.run(train_images_tensor, test_images_tensor)
