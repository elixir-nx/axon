defmodule Axon.Training.Loop do
  @default_handlers %{
    started: [],
    epoch_started: [],
    iteration_started: [],
    iteration_completed: [],
    epoch_completed: [],
    epoch_terminated: [],
    terminated: [],
    completed: []
  }
  defstruct [:process, metrics: %{}, handlers: @default_handlers]
end
