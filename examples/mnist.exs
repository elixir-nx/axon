model =
  Axon.input({nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.layer_norm()
  |> Axon.dropout()
  |> Axon.dense(10, activation: :softmax)

IO.inspect model

{train_images, train_labels} = Axon.Data.MNIST.download()

{final_params, _optimizer_state} =
  model
  |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
  |> Axon.Training.train(train_images, train_labels, epochs: 10, compiler: EXLA, log_every: 100)

IO.inspect(Nx.backend_transfer(final_params))
