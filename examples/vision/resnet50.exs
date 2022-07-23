Mix.install([
  {:axon, "~> 0.1.0"}
])

defmodule ResNet50 do
  defp conv_block(x, kernel_size, [f1, f2, f3], strides \\ [2, 2]) do
    shortcut =
      x
      |> Axon.conv(f3, kernel_size: {1, 1}, strides: strides)
      |> Axon.batch_norm()

    main =
      x
      |> Axon.conv(f1, kernel_size: {1, 1}, strides: strides)
      |> Axon.batch_norm()
      |> Axon.relu()
      |> Axon.conv(f2, kernel_size: kernel_size, padding: :same)
      |> Axon.batch_norm()
      |> Axon.relu()
      |> Axon.conv(f3, kernel_size: {1, 1})
      |> Axon.batch_norm()

    shortcut
    |> Axon.add(main)
    |> Axon.relu()
  end

  defp identity_block(%Axon{output_shape: shape} = x, kernel_size, [f1, f2]) do
    x
    |> Axon.conv(f1, kernel_size: {1, 1})
    |> Axon.batch_norm()
    |> Axon.relu()
    |> Axon.conv(f2, kernel_size: kernel_size, padding: :same)
    |> Axon.batch_norm()
    |> Axon.relu()
    |> Axon.conv(elem(shape, 1), kernel_size: {1, 1})
    |> Axon.batch_norm()
    |> Axon.add(x)
    |> Axon.relu()
  end

  def build_model(input_shape) do
    x = Axon.input("input", shape: input_shape)

    stage1 =
      x
      |> Axon.conv(64, kernel_size: {7, 7}, strides: [2, 2], padding: [{3, 3}, {3, 3}])
      |> Axon.batch_norm()
      |> Axon.relu()
      |> Axon.max_pool(kernel_size: {3, 3}, strides: [2, 2], padding: [{1, 1}, {1, 1}])

    stage2 =
      stage1
      |> conv_block({3, 3}, [64, 64, 256], [1, 1])
      |> identity_block({3, 3}, [64, 64])
      |> identity_block({3, 3}, [64, 64])

    stage3 =
      stage2
      |> conv_block({3, 3}, [128, 128, 512])
      |> identity_block({3, 3}, [128, 128])
      |> identity_block({3, 3}, [128, 128])
      |> identity_block({3, 3}, [128, 128])

    stage4 =
      stage3
      |> conv_block({3, 3}, [256, 256, 1024])
      |> identity_block({3, 3}, [256, 256])
      |> identity_block({3, 3}, [256, 256])
      |> identity_block({3, 3}, [256, 256])
      |> identity_block({3, 3}, [256, 256])
      |> identity_block({3, 3}, [256, 256])

    stage5 =
      stage4
      |> conv_block({3, 3}, [512, 512, 2048])
      |> identity_block({3, 3}, [512, 512])
      |> identity_block({3, 3}, [512, 512])
      |> Axon.avg_pool(kernel_size: {7, 7})
      |> Axon.flatten()
      |> Axon.dense(1000)

    Axon.softmax(stage5)
  end
end

IO.inspect(ResNet50.build_model({nil, 3, 224, 224}))
