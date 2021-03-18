defmodule Helpers do
  import Axon
  def conv_block(x, kernel_size, [f1, f2, f3], strides \\ [2, 2]) do
    shortcut =
      x
      |> conv(f3, kernel_size: kernel_size, strides: strides)
      # |> batch_norm()

    main =
      x
      |> conv(f1, kernel_size: {1, 1}, strides: strides)
      # |> batch_norm()
      |> relu()
      |> conv(f2, kernel_size: kernel_size, strides: strides, padding: :same)
      # |> batch_norm()
      |> relu()
      |> conv(f3, kernel_size: {1, 1})
      # |> batch_norm()

    shortcut
    |> add(main)
    |> relu()
  end

  def identity_block(%Axon{output_shape: shape} = x, kernel_size, [f1, f2]) do
    x
    |> conv(f1, kernel_size: {1, 1})
    # |> batch_norm()
    |> relu()
    |> conv(f2, kernel_size: kernel_size, padding: :same)
    # |> batch_norm()
    |> relu()
    |> conv(elem(shape, 1), kernel_size: {1, 1})
    # |> batch_norm()
    |> add(x)
    |> relu()
  end
end

defmodule ResNet50 do
  use Axon
  import Helpers

  model do
    x = input({8, 3, 224, 224})

    stage1 =
      x
      |> conv(64, kernel_size: {7, 7}, strides: [2, 2], padding: :same)
      # |> batch_norm()
      |> relu()
      |> max_pool(kernel_size: {3, 3}, strides: [2, 2])

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
      |> avg_pool(kernel_size: {7, 7})
      |> flatten()
      |> dense(1000)

    log_softmax(stage5)
  end
end