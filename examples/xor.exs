# Normally you wouldn't do this, but this is to demonstrate
# multi input models as just using `input` many times
Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
])

defmodule XOR do
  require Axon

  defp build_model(input_shape1, input_shape2) do
    inp1 = Axon.input(input_shape1)
    inp2 = Axon.input(input_shape2)
    inp1
    |> Axon.concatenate(inp2)
    |> Axon.dense(8, activation: :tanh)
    |> Axon.dense(1, activation: :sigmoid)
  end

  defp build_data do
    for _ <- 1..1000 do
      x1 = for _ <- 1..32, do: [Enum.random(0..1)]
      x2 = for _ <- 1..32, do: [Enum.random(0..1)]
      {Nx.tensor(x1), Nx.tensor(x2)}
    end
  end

  defp train_model(model, {data, targets}, epochs) do
    model
    |> Axon.Training.step(:binary_cross_entropy, Axon.Optimizers.sgd(0.01))
    |> Axon.Training.train(data, targets, epochs: epochs)
  end

  def run do
    model = build_model({:nil, 1}, {:nil, 1})

    data = build_data()

    targets =
      for {x1, x2} <- data do
        Nx.logical_xor(x1, x2)
      end

    final_training_state = train_model(model, {data, targets}, 10)

    IO.inspect Axon.predict(model, final_training_state[:params], {Nx.tensor([[0]]), Nx.tensor([[1]])})
  end
end

XOR.run()
