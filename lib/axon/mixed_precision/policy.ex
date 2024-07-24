defmodule Axon.MixedPrecision.Policy do
  @moduledoc false

  # Represents a mixed precision policy for a single layer
  defstruct params: {:f, 32}, compute: {:f, 32}, output: {:f, 32}

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(policy, _opts) do
      policy = [
        policy.params && "p=#{Nx.Type.to_string(policy.params)}",
        policy.compute && "c=#{Nx.Type.to_string(policy.compute)}",
        policy.output && "o=#{Nx.Type.to_string(policy.output)}"
      ]

      inner =
        policy
        |> Enum.reject(&is_nil/1)
        |> Enum.intersperse(" ")

      force_unfit(
        concat(
          List.flatten([
            "#Axon.MixedPrecision.Policy<",
            inner,
            ">"
          ])
        )
      )
    end
  end
end
