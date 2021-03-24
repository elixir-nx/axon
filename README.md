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
    {:torchx, "~> 0.1.0-dev", github: "elixir-nx/nx", branch: "main", sparse: "torchx"},
  ]
end
```

## Functional API

At the lowest level, `Axon` contains functional building blocks that implement common activation functions, layers, loss functions, parameter initializers, and more. All of these building blocks are implemented as `Nx` numerical definitions, so they can be used with any `Nx` compiler or backend and called arbitrarily from within `defn` or regular Elixir code.

For those familiar with frameworks in other ecosystems, the `Axon` functional API should feel similar to PyTorch's `torch.nn.functional` namespace.

## Creating Models

You can create models using the `Axon` API:

```elixir
model =
  input({nil, 3, 32, 32})
  |> conv(32, kernel_size: {3, 3})
  |> max_pool(kernel_size: {2, 2})
  |> flatten()
  |> dense(64, activation: :relu)
  |> dense(10, activation: :log_softmax)
```

The API builds an `Axon` struct that contains information about how to initialize, compile, and run the model. You can initialize the model anywhere using the `Axon.init/1` macro:

```elixir
model =
  input({nil, 3, 32, 32})
  |> conv(32, kernel_size: {3, 3})
  |> max_pool(kernel_size: {2, 2})
  |> flatten()
  |> dense(64, activation: :relu)
  |> dense(10, activation: :log_softmax)

params = Axon.init(model)
```

and then make predictions using `Axon.predict/2`:

```elixir
model =
  input({nil, 3, 32, 32})
  |> conv(32, kernel_size: {3, 3})
  |> max_pool(kernel_size: {2, 2})
  |> flatten()
  |> dense(64, activation: :relu)
  |> dense(10, activation: :log_softmax)

params = Axon.init(model)

Axon.predict(model, params)
```

Both macros can be used arbitrarily from within `defn` or in regular Elixir functions - and will still utilize whatever `Nx` backend or compiler you are using. You can even use regular Elixir functions to break up your model:

```elixir
defmodule Model do
  use Axon

  def residual(x) do
    x
    |> conv(32, kernel_size: {3, 3})
    |> add(x)
  end

  def model do
    input({nil, 3, 32, 32})
    |> conv(32, kernel_size: {3, 3})
    |> residual()
    |> max_pool(kernel_size: {3, 3})
    |> flatten()
    |> dense(10, activation: :log_softmax)
  end
end
```

You can also arbitrarily call `Nx` functions or predefined numerical definitions using an `nx` layer:

```elixir
defmodule Model do
  use Axon

  defn mish(x) do
    x * Nx.tanh(Axon.Activations.softplus(x))
  end

  def model do
    input({nil, 784})
    |> dense(128)
    |> nx(&mish/1)
    |> dense(64)
    |> nx(fn x -> Nx.max(x, 0) end)
    |> dense(10, activation: :log_softmax)
  end
end
```

## License

License
Copyright (c) 2021 Sean Moriarity

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.