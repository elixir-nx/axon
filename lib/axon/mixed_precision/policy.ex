defmodule Axon.MixedPrecision.Policy do
  @moduledoc false

  # Represents a mixed precision policy for a single layer
  defstruct params: {:f, 32}, compute: {:f, 32}, output: {:f, 32}

  defimpl Inspect do
    import Inspect.Algebra

    def inspect(policy, opts) do
      inner =
        concat([
          line(),
          color("params: ", :atom, opts),
          "#{inspect(policy.params)},",
          line(),
          color("compute: ", :atom, opts),
          "#{inspect(policy.compute)},",
          line(),
          color("output: ", :atom, opts),
          "#{inspect(policy.output)}"
        ])

      force_unfit(
        concat([
          color("#Axon.MixedPrecision.Policy<", :map, opts),
          nest(inner, 2),
          line(),
          color(">", :map, opts)
        ])
      )
    end
  end
end
