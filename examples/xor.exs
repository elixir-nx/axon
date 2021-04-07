# Normally you wouldn't do this, but this is to demonstrate
# multi input models as just using `input` many times
require Axon

inp1 = Axon.input({nil, 1})
inp2 = Axon.input({nil, 1})

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

{params, _} =
  model
  |> Axon.Training.step(:binary_cross_entropy, Axon.Optimizers.sgd(0.01))
  |> Axon.Training.train(data, targets, epochs: 10, compiler: EXLA)

IO.inspect Axon.predict(model, params, {Nx.tensor([[0]]), Nx.tensor([[1]])})
