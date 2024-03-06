defmodule Axon.Schedules do
  @moduledoc false

  @deprecated "Use Polaris.Schedules.linear_decay/2 instead"
  defdelegate linear_decay(init_value, opts \\ []), to: Polaris.Schedules

  @deprecated "Use Polaris.Schedules.exponential_decay/2 instead"
  defdelegate exponential_decay(init_value, opts \\ []), to: Polaris.Schedules

  @deprecated "Use Polaris.Schedules.cosine_decay/2 instead"
  defdelegate cosine_decay(init_value, opts \\ []), to: Polaris.Schedules

  @deprecated "Use Polaris.Schedules.constant/2 instead"
  defdelegate constant(init_value, opts \\ []), to: Polaris.Schedules

  @deprecated "Use Polaris.Schedules.polynomial_decay/2 instead"
  defdelegate polynomial_decay(init_value, opts \\ []), to: Polaris.Schedules
end
