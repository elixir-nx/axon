# Normally you wouldn't do this, but this is to demonstrate
# multi input models as just using `input` many times
Mix.install([
  {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
])

defmodule XOR do
  require Axon

  def run do
    inp1 = Axon.input({:nil, 1})
    inp2 = Axon.input({:nil, 1})

    model =
      inp1
      |> Axon.concatenate(inp2)
      |> Axon.dense(8, activation: :tanh)
      |> Axon.dense(1, activation: :sigmoid)

    data =
      for _ <- 1..1000 do
        x1 = for _ <- 1..32, do: [Enum.random(0..1)]
        x2 = for _ <- 1..32, do: [Enum.random(0..1)]
        {Nx.tensor(x1), Nx.tensor(x2)}
      end

    targets =
      for {x1, x2} <- data do
        Nx.logical_xor(x1, x2)
      end

    final_training_state =
      model
      |> Axon.Training.step(:binary_cross_entropy, Axon.Optimizers.sgd(0.01))
      |> Axon.Training.train(data, targets, epochs: 1)

    IO.inspect Axon.predict(model, final_training_state[:params], {Nx.tensor([[0]]), Nx.tensor([[1]])})
  end

end

XOR.run()
