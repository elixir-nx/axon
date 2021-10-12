defmodule Axon.Training.Process do
  @moduledoc false

  # Process function which runs iteratively within a loop,
  # reducing over data and accumulating process state. The process
  # state is initialized from `:init_fn`.

  # Fields
  #
  # :init_fn - Initialization of process state, loops are modeled
  # as a reduction over some data with the process state as the
  # accumulator. This will initialize the state of the accumulator
  #
  # :step_fn - Process function or step function. Performs processing
  # and updates of the process state.
  defstruct [:init, :step]
end
