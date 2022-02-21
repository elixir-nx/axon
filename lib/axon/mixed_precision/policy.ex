defmodule Axon.MixedPrecision.Policy do
  @moduledoc false

  # Represents a mixed precision policy for a single layer
  defstruct params: {:f, 32}, compute: {:f, 32}, output: {:f, 32}

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(policy, _opts) do
      force_unfit(
        concat([
          "p=#{Nx.Type.to_string(policy.params)} ",
          "c=#{Nx.Type.to_string(policy.compute)} ",
          "o=#{Nx.Type.to_string(policy.output)}"
        ])
      )
    end
  end
end
