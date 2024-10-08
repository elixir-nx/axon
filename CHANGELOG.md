# Changelog

## v0.7.0 (2024-10-08)

### Bug Fixes

* Do not cast integers in in Axon.MixedPrecision.cast/2
* Fix iteration counts handler in loops
* Various documentation fixes and enhancements

### Enhancements

* Add support for global layer options
* Add `inspect_values` option in compiler
* Add `Axon.rewrite_nodes` API for graph manipulation
* Add simple quantization API
* Replace model parameter map with model state struct
* Use layer name as a hook name for introspection
* Use templates as parameters

## v0.6.0 (2023-08-17)

### Breaking

* Shape is now passed as an option in `Axon.input/2`.

### Deprecations

* Deprecated `Axon.Optimizers` module. Use `Polaris.Optimizers` instead.
* Deprecated `Axon.Schedules` module. Use `Polaris.Schedules` instead.
* Deprecated `Axon.Updates` module. Use `Polaris.Updates` instead.

### Enhancements

* Add `Axon.mask/2` function.
* Add `Axon.blur_pool/1` function.
* Add `Axon.MixedPrecision.cast/3` function.
* Add `Axon.Losses.huber/2` function.
* Add label smoothing functionality with `Axon.Losses.label_smoothing/1` and `Axon.Losses.apply_label_smoothing/2`.

## v0.5.1 (2023-02-17)

### Bug Fixes

* Fixed incorrect results from group normalization

## v0.5.0 (2023-02-16)

### Enhancements

* Bump Nx dependency
* Update documentation to account for channels last default
* Improve error message in compilation/build errors for models
* Remove deprecated `transform`

### Deprecations

* Deprecate `Axon.Loop.handle/4`

## v0.4.1 (2023-01-21)

### Bug Fixes

* Fixed a shape mismatch when training with certain optimizers

## v0.4.0 (2023-01-19)

### Enhancements

* Add `Axon.pop_nodes/2` for popping nodes off of a graph
* Update `Axon.freeze/2` and `Axon.unfreeze/2` for manipulating frozen portions of Axon graph
* Add `Axon.Loop.monitor/5` for firing events based on loop state criteria
* Add `Axon.Loop.kino_vega_lite_plot/4` for producing Kino plots during training
* Add `Axon.Schedules.linear_decay/1`
* Performance boosts to `Axon.Loop` which prevent compilation cache misses in most Axon training and evaluation loops
* Add global event counts for more correct filtering during Axon loops
* Use layer state to manage dropout keys, making training more deterministic when starting from the same key
* Make building Axon models fully deterministic
* Add a bidirectional combinator

### Bug Fixes

* Fix issue with namespaced stateful models not updating correctly during training
* Fix bug in `Axon.Loop.early_stop/3` which incorrectly tracked progress and would not early stop loop
* Fix bug in `Axon.Loop.reduce_lr_on_plateau/3` which incorrectly tracked progress and would not reduce learning rate
* Fix bug in `Axon.Layers.conv_transpose/4` when using channels last

## v0.3.1 (2022-12-07)

### Enhancements

* Relax Kino dependency

## v0.3.0 (2022-10-27)

### Enhancements

* Add filters and events to validation loop event handler
* Add `Axon.Losses.cosine_similarity/2`
* Add `Axon.Activations.log_sumexp/2`
* Add mermaid rendering with `Axon.Display.as_graph/2`
* Add a number of guides to documentation
* Use stateless RNGs for initialization and dropout layers
* Include stacktraces in compilation errors to improve error messages
* Update Axon data structure to reduce size of copies
* Change channels default to last for performance reasons
* Add initial implementation of Loss scaling to Axon API

### Bug fixes

* Fix issue with `Axon.get_output_shape/2` returning container shapes
* Fix implementation bugs with `Axon.group_norm/3`
* Fix issue with `Axon.Loop.from_state/2` not allowing resumption of loops from states
* Fix issue with validation loop not populating metrics correctly

## v0.2.0 (2022-08-13)

### Enhancements

* Add support for optional inputs
* Add support for non-finite float metrics
* Add support for containers as inputs
* Allow custom activation functions in `Axon.activation`
* Add utilities for debugging model execution
* Add graph manipulation API
* Add lazy model compilation in the form of `Axon.build` and `Axon.compile`. Lazy model compilation removes eager shape calculations in favor of forcing initialization functions to accept an input template. This allows the creation of more complex models without needing to know complex shape logic
* Add `top_k_categorical_accuracy` metric
* Update default model inspection
* Add `Axon.Display.as_table/2`
* Add support for stateful learning rate in optimizers
* Add `Axon.Loop.reduce_lr_on_plateau` event handler
* Update RNN output format to mirror Elixir's convention

### Bug Fixes

* Fix bug in `Axon.resize` methods. Methods other than nearest produced invalid results
* Fix bug with `feature_group_size` in `Axon.conv`. `Axon.conv` parameters did not appropriately initialize according to specified `feature_group_size`
* Fix bug in layer name serialization. `Axon.serialize` did not appropriately freeze names when serializing, which results in deserialization failures when the model is deserialized outside the same process that serialized it

### Deprecations

* Deprecate `Axon.input/3`
* Deprecate `Axon.init/4`
* Deprecate `Axon.Recurrent`

## v0.1.0 (2022-06-16)

First release.
