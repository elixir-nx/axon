# Benchmarks fused training loops (`fuse: true`) against the default loop,
# which dispatches one computation per batch.
#
#     MIX_ENV=test mix run bench/loop_fusion.exs
#
# EXLA is a test-only dependency of Axon, hence MIX_ENV=test.
#
# Every measurement runs the same loop for a small and a large number of epochs
# and reports the marginal cost of an epoch, so that one-off costs (building the
# model, compiling the loop) are not counted as training time.

Nx.global_default_backend(EXLA.Backend)
Logger.configure(level: :warning)

defmodule LoopFusionBench do
  @batch_size 32
  @features 128

  def models do
    [
      {"mlp-tiny", mlp([8])},
      {"mlp-small", mlp([128, 64])},
      {"mlp-large", mlp([1024, 1024])}
    ]
  end

  defp mlp(hidden) do
    input = Axon.input("input", shape: {nil, @features})

    hidden
    |> Enum.reduce(input, &Axon.dense(&2, &1, activation: :relu))
    |> Axon.dense(10, activation: :softmax)
  end

  def data(batches) do
    key = Nx.Random.key(0)
    {x, key} = Nx.Random.normal(key, shape: {batches, @batch_size, @features})
    {labels, _} = Nx.Random.randint(key, 0, 10, shape: {batches, @batch_size})
    y = Nx.equal(Nx.new_axis(labels, -1), Nx.iota({1, 1, 10})) |> Nx.as_type(:f32)

    for i <- 0..(batches - 1) do
      {Nx.backend_transfer(x[i], Nx.BinaryBackend), Nx.backend_transfer(y[i], Nx.BinaryBackend)}
    end
  end

  def train(model, data, epochs, opts) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, :sgd, log: 0, seed: 0)
    |> Axon.Loop.metric(:accuracy, "acc")
    |> Axon.Loop.run(data, Axon.ModelState.empty(), [epochs: epochs, compiler: EXLA] ++ opts)
  end

  @doc """
  Returns the microseconds an iteration costs, and the loop's result.

  The first run is thrown away so that the timed runs find their computations in
  the compiler's cache, and the fastest of several runs is reported.
  """
  def per_iteration(model, data, opts, epochs \\ 5, samples \\ 3) do
    result = train(model, data, epochs, opts)

    times =
      for _ <- 1..samples do
        {time, _} = :timer.tc(fn -> train(model, data, epochs, opts) end)
        time
      end

    {Enum.min(times) / (epochs * length(data)), result}
  end

  def same?(left, right) do
    Nx.Defn.Composite.compatible?(left, right, fn l, r ->
      Nx.all_close(l, r, atol: 1.0e-4) |> Nx.to_number() == 1
    end)
  end

  def row(cells), do: cells |> Enum.map(&pad/1) |> Enum.join() |> IO.puts()

  defp pad({text, width}) when width < 0, do: String.pad_trailing(text, -width)
  defp pad({text, width}), do: String.pad_leading(text, width)
  defp pad(text), do: pad({text, 12})

  def us(value), do: "#{Float.round(value, 1)}us"
end

alias LoopFusionBench, as: Bench

batches = 100
data = Bench.data(batches)

IO.puts("""
Axon training loop fusion
  batch size: 32, batches/epoch: #{batches}
  EXLA #{EXLA.Client.fetch!(:host).platform} client, #{System.schedulers_online()} schedulers

Time per training iteration, excluding compilation:
""")

Bench.row([{"model", -12}, "unfused", "fused", {"speedup", 10}, {"   same?", -8}])

for {name, model} <- Bench.models() do
  # warm up the caches shared by both paths
  Bench.train(model, data, 1, [])

  {unfused_us, unfused} = Bench.per_iteration(model, data, [])
  {fused_us, fused} = Bench.per_iteration(model, data, fuse: true)

  Bench.row([
    {name, -12},
    Bench.us(unfused_us),
    Bench.us(fused_us),
    {"#{Float.round(unfused_us / fused_us, 2)}x", 10},
    {"   #{Bench.same?(unfused, fused)}", -8}
  ])
end

IO.puts("\nChunk size (:fuse) sweep for mlp-small:\n")
Bench.row([{"chunk", -12}, "per iter", {"speedup", 10}])

{_, model} = Bench.models() |> Enum.at(1)
{baseline_us, _} = Bench.per_iteration(model, data, [])
Bench.row([{"unfused", -12}, Bench.us(baseline_us), {"1.0x", 10}])

for chunk <- [1, 2, 4, 8, 16, 32, 64] do
  {chunk_us, _} = Bench.per_iteration(model, data, fuse: chunk)

  Bench.row([
    {Integer.to_string(chunk), -12},
    Bench.us(chunk_us),
    {"#{Float.round(baseline_us / chunk_us, 2)}x", 10}
  ])
end
