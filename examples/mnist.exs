Mix.install([
  {:axon, path: "."},
  {:exla, path: "../nx/exla"},
  {:nx, path: "../nx/nx", override: true},
  {:scidata, "~> 0.1.1"},
])

defmodule Mnist do
  require Axon

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

  defp view_images(images, {start_index, len}) do
    images
    |> hd()
    |> Nx.slice_axis(start_index, len, 0)
    |> Nx.reshape({:auto, 28, 28})
    |> Nx.to_heatmap()
    |> IO.inspect
  end

  defp build_model(input_shape) do
    Axon.input(input_shape)
    |> Axon.dense(128, activation: :relu)
    |> Axon.dropout()
    |> Axon.dense(10, activation: :softmax)
  end

  defp train_model(model, {train_images, train_labels}, epochs) do
    model
    |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
    |> Axon.Training.metric(:accuracy, "Accuracy")
    |> Axon.Training.run(Stream.zip(train_images, train_labels), epochs: epochs, compiler: EXLA)
  end

  defp test_model(model, model_state, {test_images, test_labels}) do
    init_fn = fn ->
      %{
        predictions: Nx.tensor(0.0, backend: Nx.Defn.Expr),
        loss: Nx.tensor(0.0, backend: Nx.Defn.Expr),
      }
    end

    process_fn = fn _, {inp, _} ->
      preds = Axon.predict(model, model_state, inp)
      %{predictions: preds, loss: Nx.tensor(0.0, backend: Nx.Defn.Expr)}
    end

    process = %Axon.Training.Process{init: init_fn, step: process_fn}
    loop = %Axon.Training.Loop{process: process} |> Axon.Training.metric(:accuracy, "Accuracy")
    Axon.Training.run(loop, Stream.zip(test_images, test_labels), compiler: EXLA)
  end
  
  def run do
    {images, labels} = Scidata.MNIST.download(transform_images: &transform_images/1, transform_labels: &transform_labels/1)

    {train_images, test_images} = images
    {train_labels, test_labels} = labels

    model = build_model({nil, 784}) |> IO.inspect

    final_training_state =
      model
      |> train_model({train_images, train_labels}, 1)

    view_images(test_images, {0, 1})

    test_model(model, final_training_state.process_state[:model_state], {test_images, test_labels})

    first_test_image =
      test_images
      |> hd()
      |> Nx.slice_axis(0, 1, 0)

    model
    |> Axon.predict(final_training_state.process_state[:model_state], first_test_image)
    |> Nx.argmax(axis: -1)
    |> IO.inspect
  end
end

Mnist.run()
