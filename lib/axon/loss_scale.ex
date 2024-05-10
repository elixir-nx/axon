defmodule Axon.LossScale do
  @moduledoc """
  Implementations of loss-scalers for use in mixed precision
  training.

  Loss scaling is used to prevent underflow when using mixed
  precision during the model training process. Each loss-scale
  implementation here returns a 3-tuple of the functions:

      {init_fn, scale_fn, unscale_fn, adjust_fn} = Axon.LossScale.static(Nx.pow(2, 15))

  You can use these to scale/unscale loss and gradients as well
  as adjust the loss scale state.

  `Axon.Loop.trainer/3` builds loss-scaling in by default. You
  can reference the `Axon.Loop.train_step/3` implementation to
  see how loss-scaling is applied in practice.
  """

  @default_loss_scale 2 ** 15

  import Nx.Defn
  import Axon.Shared

  @doc """
  Implements identity loss-scale.
  """
  def identity(_opts \\ []) do
    scale_unscale_fun = fn x, _state -> x end
    adjust_fun = fn x, state -> {x, state} end
    {fn -> %{} end, scale_unscale_fun, adjust_fun}
  end

  @doc """
  Implements static loss-scale.
  """
  def static(opts \\ []) do
    opts = Keyword.validate!(opts, init_scale: @default_loss_scale)
    loss_scale = Nx.backend_copy(opts[:init_scale], Nx.BinaryBackend) |> Nx.as_type(:f32)
    {fn -> init_static(loss_scale) end, &scale_static/2, &unscale_static/2}
  end

  defnp init_static(loss_scale) do
    %{loss_scale: loss_scale}
  end

  defnp scale_static(value, %{loss_scale: loss_scale}) do
    deep_new(value, fn x -> x * loss_scale end)
  end

  defnp unscale_static(value, %{loss_scale: loss_scale} = state) do
    inv_loss_scale = 1 / loss_scale
    unscaled = deep_new(value, fn x -> x * inv_loss_scale end)
    {unscaled, state}
  end

  @doc """
  Implements dynamic loss-scale.
  """
  def dynamic(opts \\ []) do
    opts =
      Keyword.validate!(opts,
        init_scale: @default_loss_scale,
        period: 2_000,
        factor: 2,
        min_loss_scale: 1
      )

    {loss_scale, opts} = Keyword.pop(opts, :init_scale, @default_loss_scale)
    loss_scale = Nx.backend_copy(loss_scale, Nx.BinaryBackend) |> Nx.as_type(:f32)

    {
      fn -> init_dynamic(loss_scale) end,
      &scale_dynamic/2,
      &unscale_dynamic(&1, &2, opts)
    }
  end

  defnp init_dynamic(loss_scale) do
    %{
      loss_scale: loss_scale,
      counter: 0
    }
  end

  defnp scale_dynamic(value, %{loss_scale: loss_scale}) do
    deep_new(value, fn x -> x * loss_scale end)
  end

  defnp unscale_dynamic(value, %{loss_scale: loss_scale} = state, opts \\ []) do
    inv_loss_scale = 1 / loss_scale
    unscaled = deep_new(value, fn x -> x * inv_loss_scale end)
    {unscaled, adjust_dynamic(value, state, opts)}
  end

  defnp adjust_dynamic(grads, %{loss_scale: loss_scale, counter: counter}, opts \\ []) do
    opts = keyword!(opts, period: 2_000, factor: 2, min_loss_scale: 1)

    grads_are_finite =
      deep_reduce(grads, Nx.tensor(1), fn x, acc ->
        x
        |> is_finite()
        |> Nx.logical_and(acc)
      end)

    new_loss_scale =
      Nx.select(
        grads_are_finite,
        Nx.select(
          Nx.equal(counter, opts[:period] - 1),
          first_finite(loss_scale * opts[:factor], loss_scale),
          loss_scale
        ),
        Nx.max(opts[:min_loss_scale], loss_scale / opts[:factor])
      )

    new_counter = Nx.remainder(counter + 1, opts[:period]) * grads_are_finite

    %{loss_scale: new_loss_scale, counter: new_counter}
  end

  defnp is_finite(x), do: Nx.all(Nx.logical_not(Nx.is_infinity(x)))

  defnp first_finite(a, b), do: Nx.select(is_finite(a), a, b)
end
