Mix.install([
  {:flow, "~> 1.0"},
  {:pixels, "~> 0.1.0"},
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "sm-horses-humans"},
  {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
  {:nx, path: "../nx/nx", override: true}
])


defmodule HorsesOrHumans do
  alias Axon.Loop.State
  import Nx.Defn

  # Download and extract from https://laurencemoroney.com/datasets.html
  # or you can use Req to download and extract the zip file and iterate
  # over the resulting data
  @directories "examples/vision/{horses,humans}/*"

  @default_defn_compiler EXLA

  def data() do
    Path.wildcard(@directories)
    |> Flow.from_enumerable()
    |> Flow.flat_map(&parse_png/1)
    |> Stream.chunk_every(32, 32, :discard)
    |> Stream.map(fn batch ->
      {inp, labels} = Enum.unzip(batch)
      {Nx.stack(inp), Nx.stack(labels)}
    end)
    |> Stream.map(&augment/1)
  end

  defnp augment({inp, labels}) do
    # Normalize
    inp = inp / 255.0
    # For now just a random flip
    if Nx.random_uniform({}) > 0.5 do
      {Nx.reverse(inp, axes: [-1]), labels}
    else
      {Nx.reverse(inp, axes: [-2]), labels}
    end
  end

  defp parse_png(filename) do
    class =
      if String.contains?(filename, "horses"),
        do: Nx.tensor([1, 0], type: {:u, 8}),
        else: Nx.tensor([0, 1], type: {:u, 8})

    {:ok, png} = Pixels.read_file(filename)

    # Only reads RGBA :(
    tensor =
      png.data
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({4, 300, 300})

    [{tensor, class}]
  end

  defp build_model(input_shape) do
    Axon.input(input_shape)
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

  defp log_metrics(
         %State{epoch: epoch, iteration: iter, metrics: metrics, step_state: pstate} = state,
         mode
       ) do
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

  defp train_model(model, data, optimizer, epochs) do
    model
    |> Axon.Loop.trainer(:binary_cross_entropy, optimizer)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.handle(:iteration_completed, &log_metrics(&1, :train))
    |> Axon.Loop.run(data, epochs: epochs, compiler: EXLA)
  end

  def run() do
    model = build_model({nil, 4, 300, 300}) |> IO.inspect
    optimizer = Axon.Optimizers.adam(1.0e-4)
    centralized_optimizer = Axon.Updates.compose(Axon.Updates.centralize(), optimizer)

    data = data()

    IO.write("\n\nTraining model without gradient centralization\n\n")
    train_model(model, data, optimizer, 10)

    IO.write("\n\nTraining model with gradient centralization\n\n")
    train_model(model, data, optimizer, 10)
  end
end

HorsesOrHumans.run()