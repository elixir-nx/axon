<h1><img src="https://github.com/elixir-nx/axon/raw/main/axon.png" alt="Axon" width="350"></h1>

Nx-powered Neural Networks for Elixir.

Axon consists of the following components:

  * Functional API – A low-level API of numerical definitions (defn) of which all other APIs build on.
  * Model Creation API – A high-level model creation API which manages model initialization and application.
  * Optimization API – An API for creating and using first-order optimization techniques.
  * Training API – An API for quickly training models.

Axon provides abstractions that enable easy integration while maintaining a level of separation between each component. You should be able to use any of the APIs without dependencies on others. By decoupling the APIs, Axon gives you full control over each aspect of creating and training a neural network.

## Overview

### Functional API

At the lowest-level, Axon consists of a number of modules with functional implementations of common methods in deep learning:

* `Axon.Activations` – Element-wise activation functions.
* `Axon.Initializers` – Model parameter initialization functions.
* `Axon.Layers` – Common deep learning layer implementations.
* `Axon.Losses` – Common loss functions.
* `Axon.Metrics` – Training metrics such as accuracy, absolute error, precision, etc.

All of the methods in the functional API are implemented as numerical definitions (`defn`). That means you can use any Nx compiler or backend to accelerate Axon. Additionally, you can arbitrarily compose methods in the Axon functional API with your own numerical definitions. Axon works entirely on Nx tensors, so any library built on top of Nx is likely to integrate well with Axon.

Because Axon’s high-level APIs build on top of the functional API, the same benefits apply. Every neural network can be JIT or AOT compiled using any Nx compiler or backend, or even transformed into high-level neural network formats like TensorFlow Lite and ONNX.

### Model Creation

An example model looks something like:

```elixir
model =
  Axon.input({nil, 784})
  |> Axon.dense(128)
  |> Axon.dense(10, activation: :softmax)
```

The model is just an Elixir struct, so serializing it to multiple formats in the future is straightforward. An added benefit is easy customization of model inspection using the Inspect protocol. The above model is printed as:

```
-----------------------------------------------
                     Model
===============================================
 Layer                 Shape        Parameters
===============================================
 input_1 (input)       {nil, 784}   0
 dense_2 (dense)       {nil, 128}   100480
 dense_3 (dense)       {nil, 10}    1290
 softmax_4 (softmax)   {nil, 10}    0
-----------------------------------------------
```

Axon provides a few conveniences for working with models. First, we chose to take the philosophy that a model’s only concerns are initialization and application. That means the model shouldn’t be concerned at all with details like training. Axon provides the macros `Axon.init/2` and `Axon.predict/4` for initializing and applying models:

```elixir
model =
  Axon.input({nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.dropout(rate: 0.5)
  |> Axon.dense(10, activation: :softmax)

params = Axon.init(model, compiler: EXLA)
Axon.predict(model, params, input, compiler: EXLA)
```

Both macros are valid inside `defn`, meaning you can easily integrate model execution with existing numerical definitions.

Axon currently has support for:

* Linear layers (dense)
* Dropout layers (dropout, feature_alpha_dropout, alpha_dropout, spatial_dropout)
* Convolutional Layers (conv, depthwise_conv, separable_conv2d, separable_conv3d)
* Normalization Layers (batch_norm, layer_norm, group_norm, instance_norm)
* Pooling Layers (max_pool, avg_pool, lp_pool, adaptive_max_pool, adaptive_avg_pool)
* Activation Layers (every function in Axon.Activations)
* Utilities/combinators (flatten, add, multiply, subtract, concatenate)

with plans to support recurrent layers, attention layers, and many more. Our goal is to maintain an API that is productive, extensible, and on par with other modern deep learning frameworks. If there is functionality you need to see that’s not included on the roadmap, feel free to open an issue.

### Optimization and training

The purpose of the training API is to provide conveniences and common routines for implementing training loops. The API is partly inspired by the excellent PyTorch Lightning library.

Currently the Axon training API consists of 2 methods: `Axon.Training.step` and `Axon.Training.train`. In practice, you can use these methods to train an Axon model like this:

```elixir
model =
  Axon.input({nil, 784})
  |> Axon.dense(128)
  |> Axon.dense(10, activation: :softmax)

trained_params =
  model
  |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
  |> Axon.Training.train(inputs, targets, epochs: 10, compiler: EXLA)
```

The step expects an optimizer as argument. The following are currently supported:

* Adabelief
* Adagrad
* Adam
* Adamw
* Fromage
* Lamb
* Noisy SGD
* Radam
* RMSProp
* SGD
* Yogi

It’s important to note that optimization API does not directly depend on Axon models. You can use the API to optimize any differentiable objective function.

Axon.Training.train implements a common training loop which initializes the training state and iterates through the training set for some given number of epochs. It returns the final training state for serialization and potential use in inference workloads. Currently, the Axon training API is rather limited; however, there are plans to extend it. In the immediate future, we plan to support:

* Validation and Testing Integration
* Logging to tools like TensorBoard
* Inclusion of Train/Validate/Test Metrics
* Per batch/epoch callbacks for things like early stopping and model checkpointing

Additionally, we would love to explore more advanced things like distributed training. We are also seeking ways to improve the performance of our training loops by running them entirely on native accelerators.

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

You'll also likely want to include an `Nx` compiler such as `EXLA` for any practical deep learning workload:

```elixir
def deps do
  [
    {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon", branch: "main"},
    {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla", override: true},
    {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
  ]
end
```

## License

Copyright (c) 2021 Sean Moriarity

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
