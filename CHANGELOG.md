# Changelog

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