Mix.install([
  {:axon, path: "."},
  {:exla, path: "../nx/exla"},
  {:nx, path: "../nx/nx", override: true},
  {:scidata, "~> 0.1.1"},
])

# Application.put_env(:exla, :clients, default: [platform: :cuda])

defmodule Mnist do
  require Axon

  alias Axon.Loop.State

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 784})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
    |> Enum.split(1750) # Test split
  end

  defp transform_labels({bin, type, _}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
    |> Nx.to_batched_list(32)
    |> Enum.split(1750) # Test split
  end

  defp build_model(input_shape) do
    Axon.input(input_shape)
    |> Axon.dense(128, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(10, activation: :softmax)
  end

  defp log_metrics(%State{epoch: epoch, iteration: iter, metrics: metrics, process_state: pstate} = state, mode) do
    loss =
      case mode do
        :train ->
          %{loss: loss} = pstate
          "Loss: #{:io_lib.format('~.5f', [Nx.to_scalar(loss)])}"

        :test ->
          ""
      end

    metrics =
      metrics
      |> Enum.map(fn {k, v} -> "#{k}: #{:io_lib.format('~.5f', [Nx.to_scalar(v)])}" end)
      |> Enum.join(" ")

    IO.write("\rEpoch: #{Nx.to_scalar(epoch)}, Batch: #{Nx.to_scalar(iter)}, #{loss} #{metrics}")

    {:continue, state}
  end

  defp train_model(model, {train_images, train_labels}, epochs) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.handle(:iteration_completed, &log_metrics(&1, :train), every: 50)
    |> Axon.Loop.run(Stream.zip(train_images, train_labels), epochs: epochs, compiler: EXLA)
  end

  defp test_model(model, model_state, test_images, test_labels) do
    model
    |> Axon.Loop.evaluator(model_state)
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.handle(:iteration_completed, &log_metrics(&1, :test), every: 50)
    |> Axon.Loop.run(Stream.zip(test_images, test_labels), compiler: EXLA)
  end

  def run do
    {images, labels} = Scidata.MNIST.download(transform_images: &transform_images/1, transform_labels: &transform_labels/1)

    {train_images, test_images} = images
    {train_labels, test_labels} = labels

    model = build_model({nil, 784}) |> IO.inspect

    IO.write("\n\n Training Model \n\n")

    model_state =
      model
      |> train_model({train_images, train_labels}, 5)

    IO.write("\n\n Testing Model \n\n")

    model
    |> test_model(model_state, test_images, test_labels)

    IO.write("\n\n")
  end
end

Mnist.run()
