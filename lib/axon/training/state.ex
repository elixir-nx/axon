defmodule Axon.Training.State do
  defstruct [:epoch, :iteration, :metrics, :process_state]
end
