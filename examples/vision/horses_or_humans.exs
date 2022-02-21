Mix.install([
  {:stb_image, "~> 0.1.0"},
  {:axon, "~> 0.1.0-dev", path: "."},
  {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
])

EXLA.set_preferred_defn_options([:tpu, :cuda, :rocm, :host])
Nx.Defn.global_default_options(compiler: EXLA, run_options: [keep_on_device: true])

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

  # TODO: Fuse this with Axon network
  defnp augment(inp) do
    # Normalize
    inp = inp / 255.0

    # For now just a random flip
    if Nx.random_uniform({}) > 0.5 do
      Nx.reverse(inp, axes: [-1])
    else
      Nx.reverse(inp, axes: [-2])
    end
  end

  defp parse_png(filename) do
    class =
      if String.contains?(filename, "horses"),
        do: Nx.tensor([1, 0], type: {:u, 8}),
        else: Nx.tensor([0, 1], type: {:u, 8})

    {:ok, binary, shape, :u8, :rgba} = StbImage.from_file(filename)

    tensor =
      binary
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape(shape)

    {tensor, class}
  end

  defp build_model(input_shape, transpose_shape) do
    Axon.input(input_shape)
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

  defp log_metrics(
         %State{epoch: epoch, iteration: iter, metrics: metrics, step_state: pstate} = state,
         mode
       ) do
    loss =
      case mode do
        :train ->
          %{loss: loss} = pstate
          "Loss: #{Float.round(Nx.to_number(loss), 5)}"

        :test ->
          ""
      end

    metrics =
      metrics
      |> Enum.map(fn {k, v} -> "#{k}: #{Float.round(Nx.to_number(v), 5)}" end)
      |> Enum.join(" ")

    IO.write("\rEpoch: #{epoch}, Batch: #{iter}, #{loss} #{metrics}")

    {:continue, state}
  end

  defp train_model(model, data, optimizer, epochs) do
    model
    |> Axon.Loop.trainer(:binary_cross_entropy, optimizer)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.handle(:iteration_completed, &log_metrics(&1, :train))
    |> Axon.Loop.run(data, epochs: epochs, iterations: 100)
  end

  def run() do
    model = build_model({nil, 300, 300, 4}, [2, 0, 1]) |> IO.inspect
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