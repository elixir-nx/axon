Mix.install([
  {:stb_image, "~> 0.5.2"},
  {:axon, "~> 0.1.0"},
  {:exla, "~> 0.2.2"},
  {:nx, "~> 0.2.1"}
])

EXLA.set_as_nx_default(
  [:tpu, :cuda, :rocm, :host],
  run_options: [keep_on_device: true]
)

defmodule HorsesOrHumans do
  alias Axon.Loop.State
  import Nx.Defn

  # Download and extract from https://laurencemoroney.com/datasets.html
  # or you can use Req to download and extract the zip file and iterate
  # over the resulting data
  @directories "examples/vision/{horses,humans}/*"

  def data() do
    Path.wildcard(@directories)
    |> Stream.chunk_every(32, 32, :discard)
    |> Task.async_stream(fn batch ->
      {inp, labels} = batch |> Enum.map(&parse_png/1) |> Enum.unzip()
      {Nx.stack(inp), Nx.stack(labels)}
    end)
    |> Stream.map(fn {:ok, {inp, labels}} -> {augment(inp), labels} end)
    |> Stream.cycle()
  end

  defnp augment(inp) do
    # Normalize
    inp = inp / 255.0

    # For now just a random flip
    if Nx.random_uniform({}) > 0.5 do
      Nx.reverse(inp, axes: [0])
    else
      Nx.reverse(inp, axes: [1])
    end
  end

  defp parse_png(filename) do
    class =
      if String.contains?(filename, "horses"),
        do: Nx.tensor([1, 0], type: {:u, 8}),
        else: Nx.tensor([0, 1], type: {:u, 8})

    {:ok, binary, shape, :u8, :rgba} = StbImage.from_file(filename)

    {StbImage.to_nx(img), class}
  end

  defp build_model(input_shape, transpose_shape) do
    Axon.input("input", shape: input_shape)
    |> Axon.transpose(transpose_shape)
    |> Axon.conv(16, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
    |> Axon.spatial_dropout(rate: 0.5)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.spatial_dropout(rate: 0.5)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
    |> Axon.max_pool(kernel_size: {2, 2})
    |> Axon.flatten()
    |> Axon.dropout(rate: 0.5)
    |> Axon.dense(512, activation: :relu)
    |> Axon.dense(2, activation: :softmax)
  end

  defp train_model(model, data, optimizer, epochs) do
    model
    |> Axon.Loop.trainer(:binary_cross_entropy, optimizer, log: 1)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(data, %{}, epochs: epochs, iterations: 100)
  end

  def run() do
    model = build_model({nil, 300, 300, 4}, [2, 0, 1]) |> IO.inspect()
    optimizer = Axon.Optimizers.adam(1.0e-4)
    centralized_optimizer = Axon.Updates.compose(Axon.Updates.centralize(), optimizer)

    data = data()

    IO.write("\n\nTraining model without gradient centralization\n\n")
    train_model(model, data, optimizer, 10)

    IO.write("\n\nTraining model with gradient centralization\n\n")
    train_model(model, data, centralized_optimizer, 10)
  end
end

HorsesOrHumans.run()
