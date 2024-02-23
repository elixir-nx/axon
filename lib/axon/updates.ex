defmodule Axon.Updates do
  @moduledoc false

  import Nx.Defn

  @deprecated "Use Polaris.Updates.scale/2 instead"
  defdelegate scale(combinator \\ identity(), step_size), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_state/1 instead"
  defdelegate scale_by_state(combinator_or_step), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_state/2 instead"
  defdelegate scale_by_state(combinator, step), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_adam/1 instead"
  defdelegate scale_by_adam(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_adam/2 instead"
  defdelegate scale_by_adam(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_rss/1 instead"
  defdelegate scale_by_rss(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_rss/1 instead"
  defdelegate scale_by_rss(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_rms/1 instead"
  defdelegate scale_by_rms(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_rms/2 instead"
  defdelegate scale_by_rms(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_belief/1 instead"
  defdelegate scale_by_belief(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_belief/2 instead"
  defdelegate scale_by_belief(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_stddev/1 instead"
  defdelegate scale_by_stddev(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_stddev/2 instead"
  defdelegate scale_by_stddev(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_schedule/2 instead"
  defdelegate scale_by_schedule(combinator \\ identity(), schedule_fn), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_radam/1 instead"
  defdelegate scale_by_radam(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_radam/2 instead"
  defdelegate scale_by_radam(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.trace/1 instead"
  defdelegate trace(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.trace/2 instead"
  defdelegate trace(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.clip/1 instead"
  defdelegate clip(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.clip/2 instead"
  defdelegate clip(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.clip_by_global_norm/1 instead"
  defdelegate clip_by_global_norm(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.clip_by_global_norm/2 instead"
  defdelegate clip_by_global_norm(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.centralize/1 instead"
  defdelegate centralize(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.centralize/2 instead"
  defdelegate centralize(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.add_decayed_weights/1 instead"
  defdelegate add_decayed_weights(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.add_decayed_weights/2 instead"
  defdelegate add_decayed_weights(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_trust_ratio/1 instead"
  defdelegate scale_by_trust_ratio(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_trust_ratio/2 instead"
  defdelegate scale_by_trust_ratio(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.add_noise/1 instead"
  defdelegate add_noise(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.add_noise/2 instead"
  defdelegate add_noise(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_yogi/1 instead"
  defdelegate scale_by_yogi(combinator_or_opts \\ []), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.scale_by_yogi/2 instead"
  defdelegate scale_by_yogi(combinator, opts), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.stateless/2 instead"
  defdelegate stateless(parent_combinator \\ identity(), apply_fn), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.identity/0 instead"
  defdelegate identity(), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.identity/1 instead"
  defdelegate identity(combinator), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.compose/2 instead"
  defdelegate compose(combinator1, combinator2), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.stateful/3 instead"
  defdelegate stateful(parent_combinator \\ identity(), init_fn, apply_fn), to: Polaris.Updates

  @deprecated "Use Polaris.Updates.apply_updates/3 instead"
  defn apply_updates(params, updates, state \\ nil) do
    Polaris.Updates.apply_updates(params, updates, state)
  end
end
