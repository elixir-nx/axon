defmodule Axon.Loop.State do
  @moduledoc """
  Accumulated state in an Axon.Loop.

  Loop state is a struct:

      %State{
        epoch: integer(),
        max_epoch: integer(),
        iteration: integer(),
        max_iteration: integer(),
        metrics: map(string(), container()),
        times: map(integer(), integer()),
        step_state: container(),
        handler_metadata: container()
      }

  `epoch` is the current epoch, starting at 0, of the nested loop.
  Defaults to 0.

  `max_epoch` is the maximum number of epochs the loop should run
  for. Defaults to 1.

  `iteration` is the current iteration of the inner loop. In supervised
  settings, this will be the current batch. Defaults to 0.

  `max_iteration` is the maximum number of iterations the loop should
  run a given epoch for. Defaults to -1 (no max).

  `metrics` is a map of `%{"metric_name" => value}` which accumulates metrics
  over the course of loop processing. Defaults to an empty map.

  `times` is a map of `%{epoch_number => value}` which maps a given epoch
  to the processing time. Defaults to an empty map.

  `step_state` is the step state as defined by the loop's processing
  initialization and update functions. `step_state` is a required field.

  `handler_metadata` is a metadata field for storing loop handler metadata.
  For example, loop checkpoints with specific metric criteria can store
  previous best metrics in the handler meta for use between iterations.
  """
  @enforce_keys [:step_state]
  defstruct [
    :step_state,
    handler_metadata: %{},
    epoch: 0,
    max_epoch: 1,
    iteration: 0,
    max_iteration: -1,
    metrics: %{},
    times: %{}
  ]
end
