# Axon

Nx-powered Neural Networks

## Installation

In order to use `Axon`, you will need Elixir installed. Then create an Elixir project via the mix build tool:

```
$ mix new my_app
```

Then you can add `Axon` as dependency in your `mix.exs`. At the moment you will have to use a Git dependency while we work on our first release:

```elixir
def deps do
  [
    {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"}
  ]
end
```

You will typically want to include another `Nx` backend or as a dependency as well:

```elixir
def deps do
  [
    {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
    {:exla, "~> 0.1.0-dev", github: "elixir-nx/exla", branch: "main", sparse: "exla"},
    {:torchx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "torchx"},
  ]
end
```

## A Basic Neural Network

You can create neural networks inside modules with the `model` macro:

```elixir
defmodule MyNN do
  use Axon

  model do
    input({nil, 784})
    |> dense(128)
    |> relu()
    |> dense(64, activation: :tanh)
    |> dropout(rate: 0.5)
    |> dense(10)
    |> activation(:softmax)
  end
end
```

`model` will generate the numerical definitions `init_random_params/0` and `predict/2`. You can also name models for using multiple models in the same training loop:

```elixir
defmodule GAN do
  use Axon

  model generator do
    input({nil, 100})
    |> dense(128)
    |> tanh()
    |> dense(256)
    |> tanh()
    |> dense(512)
    |> tanh()
    |> dense(1024)
    |> tanh()
    |> dense(784)
    |> tanh()
  end

  model discriminator do
    input({nil, 28, 28})
    |> flatten()
    |> dense(128)
    |> relu()
    |> dense(64)
    |> relu()
    |> dense(2)
    |> softmax()
  end
end
```

In the above example, you can initialize and apply both models using: `init_generator`/`generator` and `init_discriminator`/`discriminator` respectively.