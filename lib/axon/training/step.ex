defmodule Axon.Training.Step do
  @moduledoc false

  # Training step which controls the Axon training loop
  # Functions in the Training module work directly on this struct

  # Fields
  #
  # :init - Initialization of training state, training loops are modeled
  # as a reduction over the training data with the training state as the
  # accumulator. This will initialize the state of the accumulator
  #
  # :step - Training step. Performs a single step, updating the training
  # state
  #
  # :callbacks - List of training callbacks. Performed at various times
  # throughout training
  defstruct [:init, :step, :callbacks]
end
