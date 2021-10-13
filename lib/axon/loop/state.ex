defmodule Axon.Loop.State do
  # TODO(seanmor5): We should not send `:times` to the device. We need
  # a way in Nx/EXLA to mark `:times` as a static property which is
  # not to be touched at JIT time.
  defstruct [:epoch, :iteration, :metrics, :times, :process_state]
end
