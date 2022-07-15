<h1><img src="https://github.com/elixir-nx/axon/raw/main/axon.png" alt="Axon" width="350"></h1>

[![Package](https://img.shields.io/badge/-Package-important)](https://hex.pm/packages/axon) [![Documentation](https://img.shields.io/badge/-Documentation-blueviolet)](https://hexdocs.pm/axon)

Nx-powered Neural Networks for Elixir.

Axon consists of the following components:

  * Functional API – A low-level API of numerical definitions (defn) of which all other APIs build on.
  * Model Creation API – A high-level model creation API which manages model initialization and application.
  * Optimization API – An API for creating and using first-order optimization techniques based on the [Optax](https://github.com/deepmind/optax) library.
  * Training API – An API for quickly training models, inspired by [PyTorch Ignite](https://pytorch.org/ignite/index.html).

Axon provides abstractions that enable easy integration while maintaining a level of separation between each component. You should be able to use any of the APIs without dependencies on others. By decoupling the APIs, Axon gives you full control over each aspect of creating and training a neural network.

## Overview

For an in-depth overview, see: [Axon: Deep Learning in Elixir](https://seanmoriarity.com/2021/04/08/axon-deep-learning-in-elixir/)

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
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128)
  |> Axon.dense(10, activation: :softmax)
```

The model is just an Elixir struct, so serializing it to multiple formats in the future is straightforward. An added benefit is easy customization of model inspection using the Inspect protocol. The above model is printed as:

```
----------------------------------------------------------------------------------------------------
                                               Model
====================================================================================================
 Layer                              Shape        Policy              Parameters   Parameters Memory
====================================================================================================
 input ( input )                    {nil, 784}   p=f32 c=f32 o=f32   0            0 bytes
 dense_0 ( dense["input"] )         {nil, 128}   p=f32 c=f32 o=f32   100480       401920 bytes
 dense_1 ( dense["dense_0"] )       {nil, 10}    p=f32 c=f32 o=f32   1290         5160 bytes
 softmax_0 ( softmax["dense_1"] )   {nil, 10}    p=f32 c=f32 o=f32   0            0 bytes
----------------------------------------------------------------------------------------------------
Total Parameters: 101770
Total Parameters Memory: 407080 bytes
Inputs: %{"input" => {nil, 784}}
```

Axon provides a few conveniences for working with models. First, we chose to take the philosophy that a model’s only concerns are initialization and application. That means the model shouldn’t be concerned at all with details like training. Axon provides the macros `Axon.init/2` and `Axon.predict/4` for initializing and applying models:

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128, activation: :relu)
  |> Axon.dropout(rate: 0.5)
  |> Axon.dense(10, activation: :softmax)

params = Axon.init(model, compiler: EXLA)
Axon.predict(model, params, input, compiler: EXLA)
```

Both macros are valid inside `defn`, meaning you can easily integrate model execution with existing numerical definitions.

Axon currently has support for:

* Linear layers (dense, bilinear, embedding)
* Dropout layers (dropout, feature_alpha_dropout, alpha_dropout, spatial_dropout)
* Convolutional Layers (conv, conv_transpose, depthwise_conv, separable_conv2d, separable_conv3d)
* Recurrent Layers (gru, lstm, conv_lstm)
* Normalization Layers (batch_norm, layer_norm, group_norm, instance_norm)
* Pooling Layers (max_pool, avg_pool, lp_pool, adaptive_max_pool, adaptive_avg_pool, adaptive_lp_pool, global_max_pool, global_avg_pool, global_lp_pool)
* Activation Layers (every function in Axon.Activations)
* Utilities/combinators (flatten, add, multiply, subtract, concatenate, pad, nx, reshape, transpose)

with plans to support recurrent layers, attention layers, and many more. Our goal is to maintain an API that is productive, extensible, and on par with other modern deep learning frameworks. If there is functionality you need to see that’s not included on the roadmap, feel free to open an issue.

### Optimization and training

The purpose of the training API is to provide conveniences and common routines for implementing training loops. The API is inspired by the excellent PyTorch Ignite library.

The general pattern for training a model is:

  1) Define model
  2) Define loop using one of the factory methods (here `Axon.Loop.trainer/3`)
  3) Instrument loop with metrics and event handlers
  4) Run loop on data

```elixir
model =
  Axon.input("input", shape: {nil, 784})
  |> Axon.dense(128)
  |> Axon.dense(10, activation: :softmax)

model_state =
  model
  |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
  |> Axon.Loop.metric(:accuracy)
  |> Axon.Loop.handle(:iteration_completed, &log_metrics/1, every: 50)
  |> Axon.Loop.run(data, epochs: 10, compiler: EXLA)
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

In the future we plan to support distributed training loops. We are also seeking ways to improve the performance of our training loops by running them entirely on native accelerators.

## Installation

In order to use `Axon`, you will need Elixir installed. Then create an Elixir project via the mix build tool:

```
$ mix new my_app
```

Then add Axon to your dependencies:

```elixir
def deps do
  [
    {:axon, "~> 0.1.0"}
  ]
end
```

You'll also likely want to include an `Nx` compiler such as `EXLA` for any practical deep learning workload:

```elixir
def deps do
  [
    {:axon, "~> 0.1.0"},
    {:exla, "~> 0.2.2"},
    {:nx, "~> 0.2.1"}
  ]
end
```

## Sponsors

<a href="https://dockyard.com"><img src="sponsors/dockyard.png" width=200 alt="DockYard"></a>

## License

Copyright (c) 2021 Sean Moriarity

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
