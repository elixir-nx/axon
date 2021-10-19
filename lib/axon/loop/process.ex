defmodule Axon.Loop.Process do
  @moduledoc false

  # Process function which runs iteratively within a loop,
  # reducing over data and accumulating process state. The process
  # state is initialized from `:init` and updated at each iteration
  # with `:update`.

  # Fields
  #
  # :init - Initialization of process state, loops are modeled
  # as a reduction over some data with the process state as the
  # accumulator. This will initialize the state of the accumulator
  #
  # :update - Process function or update function. Performs processing
  # and updates of the process state.
  defstruct [:init, :update]
end
