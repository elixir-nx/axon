model =
  Axon.input({nil, 3, 32, 32})
  |> Axon.conv(32, kernel_size: {3, 3}, activation: :relu)
  |> Axon.batch_norm()
  |> Axon.max_pool(kernel_size: {2, 2})
  |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
  |> Axon.spatial_dropout()
  |> Axon.batch_norm()
  |> Axon.max_pool(kernel_size: {2, 2})
  |> Axon.conv(64, kernel_size: {3, 3}, activation: :relu)
  |> Axon.batch_norm()
  |> Axon.flatten()
  |> Axon.dense(64, activation: :relu)
  |> Axon.dropout()
  |> Axon.dense(10, activation: :softmax)

IO.inspect model

{train_images, train_labels} = Axon.Data.CIFAR.download()

{final_params, _optimizer_state} =
  model
  |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
  |> Axon.Training.train(train_images, train_labels, epochs: 20, compiler: EXLA)
  |> Nx.backend_transfer()
  |> IO.inspect()
