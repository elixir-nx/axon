defmodule Axon.Constraints do
  @moduledoc """
  Parameter constraints.

  Constraints are projections applied to trainable parameters after
  each optimizer update, keeping them inside a feasible region such as
  a maximum norm or non-negativity. They are declared per parameter
  with the `:constraint` option of `Axon.param/3`, or attached to an
  existing model state with `Axon.ModelState.constrain/3`, and applied
  by `Axon.Loop.train_step/4` after every update.

  Constraints are never applied at initialization, to frozen parameters,
  or to layer state such as batch normalization statistics.

  The functions in this module return arity-1 functions which take a
  parameter tensor and return the projected tensor:

      constraint = Axon.Constraints.max_norm(max: 2.0)
      constraint.(kernel)

  Norm-based constraints compute the norm over the given `:axes` and
  rescale the vectors along the remaining axes. For a dense kernel of
  shape `{in, out}`, `axes: [0]` (the default) constrains the incoming
  weight vector of every output unit.

  You may use these functions from within `defn` or outside.
  """
  import Nx.Defn

  @epsilon 1.0e-7

  @doc """
  Constrains the norm of the parameter to be at most `:max`.

  Vectors whose norm exceeds `:max` are rescaled to have norm
  `:max`; all other vectors are left unchanged.

  ## Options

    * `:max` - maximum norm. Defaults to `2.0`.

    * `:axes` - axes to compute the norm over. Defaults to `[0]`.

  ## Examples

      iex> constraint = Axon.Constraints.max_norm(max: 1.0)
      iex> constraint.(Nx.tensor([[0.0, 0.0], [3.0, 4.0]]))
      #Nx.Tensor<
        f32[2][2]
        [
          [0.0, 0.0],
          [1.0, 1.0]
        ]
      >
  """
  def max_norm(opts \\ []) do
    fn x -> max_norm_impl(x, opts) end
  end

  defnp max_norm_impl(x, opts \\ []) do
    opts = keyword!(opts, max: 2.0, axes: [0])
    norms = norm(x, opts[:axes])
    desired = Nx.clip(norms, 0, opts[:max])
    x * (desired / (norms + @epsilon))
  end

  @doc """
  Constrains the parameter to be non-negative.

  ## Examples

      iex> constraint = Axon.Constraints.non_neg()
      iex> constraint.(Nx.tensor([[-1.0, 2.0], [3.0, -4.0]]))
      #Nx.Tensor<
        f32[2][2]
        [
          [0.0, 2.0],
          [3.0, 0.0]
        ]
      >
  """
  def non_neg() do
    &non_neg_impl/1
  end

  defnp non_neg_impl(x) do
    Nx.max(x, 0)
  end

  @doc """
  Constrains the parameter to have unit norm.

  ## Options

    * `:axes` - axes to compute the norm over. Defaults to `[0]`.

  ## Examples

      iex> constraint = Axon.Constraints.unit_norm()
      iex> constraint.(Nx.tensor([[0.0, 0.0], [2.0, 4.0]]))
      #Nx.Tensor<
        f32[2][2]
        [
          [0.0, 0.0],
          [1.0, 1.0]
        ]
      >
  """
  def unit_norm(opts \\ []) do
    fn x -> unit_norm_impl(x, opts) end
  end

  defnp unit_norm_impl(x, opts \\ []) do
    opts = keyword!(opts, axes: [0])
    x / (norm(x, opts[:axes]) + @epsilon)
  end

  @doc """
  Constrains the norm of the parameter to be between `:min` and `:max`.

  Vectors whose norm lies outside `[min, max]` are rescaled toward the
  nearest bound. `:rate` controls how strictly the constraint is
  enforced on each application: with `rate: 1.0` norms are clipped
  exactly, while smaller rates only move the norm part of the way
  toward the feasible region.

  ## Options

    * `:min` - minimum norm. Defaults to `0.0`.

    * `:max` - maximum norm. Defaults to `1.0`.

    * `:rate` - rate at which to enforce the constraint. Defaults to `1.0`.

    * `:axes` - axes to compute the norm over. Defaults to `[0]`.

  ## Examples

      iex> constraint = Axon.Constraints.min_max_norm(min: 8.0, max: 16.0)
      iex> constraint.(Nx.tensor([[4.0, 32.0]]))
      #Nx.Tensor<
        f32[1][2]
        [
          [8.0, 16.0]
        ]
      >
  """
  def min_max_norm(opts \\ []) do
    fn x -> min_max_norm_impl(x, opts) end
  end

  defnp min_max_norm_impl(x, opts \\ []) do
    opts = keyword!(opts, min: 0.0, max: 1.0, rate: 1.0, axes: [0])
    norms = norm(x, opts[:axes])

    desired =
      opts[:rate] * Nx.clip(norms, opts[:min], opts[:max]) + (1 - opts[:rate]) * norms

    x * (desired / (norms + @epsilon))
  end

  defnp norm(x, axes) do
    Nx.sqrt(Nx.sum(x * x, axes: axes, keep_axes: true))
  end
end
