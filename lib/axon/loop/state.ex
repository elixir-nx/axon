defmodule Axon.Loop.State do
  @moduledoc false
  # TODO(seanmor5): We should not send `:times` to the device. We need
  # a way in Nx/EXLA to mark `:times` as a static property which is
  # not to be touched at JIT time.
  @enforce_keys [:process_state]
  defstruct [:process_state, epoch: 0, iteration: 0, metrics: %{}, times: %{}]
end
