Mix.install([
  {:axon, "~> 0.5"},
  {:exla, "~> 0.5"},
  {:nx, "~> 0.5"}
])

defmodule XOR do
  require Axon

  defp build_model(input_shape1, input_shape2) do
    inp1 = Axon.input("x1", shape: input_shape1)
    inp2 = Axon.input("x2", shape: input_shape2)

    inp1
    |> Axon.concatenate(inp2)
    |> Axon.dense(8, activation: :tanh)
    |> Axon.dense(1, activation: :sigmoid)
  end

  defp batch(key) do
    {x1, next_key} = Nx.Random.randint(key, 0, 1 + 1, shape: {32, 1})
    {x2, next_key} = Nx.Random.randint(next_key, 0, 1 + 1, shape: {32, 1})
    y = Nx.logical_xor(x1, x2)

    {{%{"x1" => x1, "x2" => x2}, y}, next_key}
  end

  defp train_model(model, data, epochs) do
    model
    |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
    |> Axon.Loop.run(data, %{}, epochs: epochs, iterations: 1000, compiler: EXLA)
  end

  def run do
    model = build_model({nil, 1}, {nil, 1})
    data = Stream.unfold(Nx.Random.key(42), &batch/1)

    model_state = train_model(model, data, 10)

    IO.inspect(
      Axon.predict(model, model_state, %{"x1" => Nx.tensor([[0]]), "x2" => Nx.tensor([[1]])})
    )
  end
end

XOR.run()
