Mix.install([
  {:axon, path: "."},
  {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla", override: true},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
  {:scidata, "~> 0.1.1"}
])


to_tensor = fn {binary, type, shape} ->
  binary
  |> Nx.from_binary(type)
  |> Nx.reshape(shape)
end

{train_images, train_labels} =
  Scidata.FashionMNIST.download(
    transform_images: to_tensor,
    transform_labels: to_tensor
  )

{test_images, test_labels} =
  Scidata.FashionMNIST.download_test(
    transform_images: to_tensor,
    transform_labels: to_tensor
  )

train_images = Nx.reshape(train_images, {:auto, 28, 28})
train_labels = Nx.reshape(train_labels, {:auto, 1})
test_images = Nx.reshape(test_images, {:auto, 28, 28})
test_labels = Nx.reshape(test_labels, {:auto, 1})

:ok

# Filter tensor along the first dimension using the given binary mask
apply_mask = fn tensor, mask ->
  # Boolean indexing doesn't produce static size, so we handle it on Elixir side
  tensors = Nx.to_batched_list(tensor, 1)
  mask = Nx.to_flat_list(mask)

  tensors
  |> Enum.zip(mask)
  |> Enum.filter(fn {_, bit} -> bit == 1 end)
  |> Enum.map(&elem(&1, 0))
  |> Nx.concatenate()
end

get_mask = fn t -> Nx.logical_or(Nx.equal(t, 3), Nx.equal(t, 5)) end

train_mask = get_mask.(train_labels)
test_mask = get_mask.(test_labels)

train_images = apply_mask.(train_images, train_mask)
train_labels = apply_mask.(train_labels, train_mask)
test_images = apply_mask.(test_images, test_mask)
test_labels = apply_mask.(test_labels, test_mask)

train_labels = Nx.equal(train_labels, 5)
test_labels = Nx.equal(test_labels, 5)

# Preview images to see if filtering actually worked
Nx.to_heatmap(train_images)

IO.inspect train_labels

model =
  Axon.input({nil, 28, 28})
  |> Axon.flatten()
  |> Axon.nx(&Nx.divide(&1, 255))
  |> Axon.dense(1)
  |> Axon.sigmoid()

  data = [{train_images, train_labels}]

log_metrics = fn state ->
  IO.puts("Loss: #{Nx.to_scalar(state.step_state[:loss])}")
  {:continue, state}
end

model_state =
  model
  |> Axon.Loop.trainer(:binary_cross_entropy, :sgd)
  |> Axon.Loop.handle(:iteration_completed, log_metrics, every: 1)
  |> Axon.Loop.run(data, epochs: 10, compiler: EXLA)

{_, predict_fn} = Axon.compile(model)
 IO.inspect Nx.Defn.jit(predict_fn, [model_state, test_images], compiler: EXLA)
