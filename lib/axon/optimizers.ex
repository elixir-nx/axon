defmodule Axon.Optimizers do
  @moduledoc """
  Implementations of common gradient-based optimization algorithms.

  All of the methods in this module are written in terms of
  the update methods defined in `Axon.Updates`. Axon treats
  optimizers as the tuple:

      {init_fn, update_fn}

  where `init_fn` returns an initial optimizer state and `update_fn`
  scales input gradients. `init_fn` accepts a model's parameters
  and attaches state to each parameter. `update_fn` accepts
  gradients, optimizer state, and current model parameters and
  returns updated optimizer state and gradients.

  Custom optimizers are often created via the `Axon.Updates` API.

  ## Example

  Consider the following usage of the Adam optimizer in a basic
  update function (assuming `objective` and the `dataset` are
  defined elsewhere):

      defmodule Learning do

        import Nx.Defn

        defn init(params, init_fn) do
          init_fn.(params)
        end

        defn update(params, optimizer_state, inputs, targets, update_fn) do
          {loss, gradient} = value_and_grad(params, &objective(&1, inputs, targets))
          {scaled_updates, new_optimizer_state} = update_fn.(gradient, optimizer_state, params)
          {Axon.Updates.apply_updates(params, scaled_updates), new_optimizer_state, loss}
        end
      end

      model_params = Nx.random_uniform({784, 10})
      {init_fn, update_fn} = Axon.Optimizers.adam(0.005)

      optimizer_state =
        Learning.init(params, init_fn)

      {new_params, new_optimizer_state, loss} =
        Learning.update(params, optimizer_state, inputs, targets, update_fn)

  For a simpler approach, you can also use optimizers with the training API:

        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(0.005))
        |> Axon.Loop.run(data, epochs: 10, compiler: EXLA)

  """
  alias Axon.Updates

  @doc """
  Adabelief optimizer.

  ## Options

    * `:b1` - first moment decay. Defaults to `0.9`
    * `:b2` - second moment decay. Defaults to `0.999`
    * `:eps` - numerical stability term. Defaults to `0.0`
    * `:eps_root` - numerical stability term. Defaults to `1.0e-16`

  ## References

    * [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468)
  """
  def adabelief(learning_rate \\ 1.0e-3, opts \\ []) do
    Updates.scale_by_belief(opts)
    |> scale_by_learning_rate(learning_rate)
  end

  @doc """
  Adagrad optimizer.

  ## Options

    * `:eps` - numerical stability term. Defaults to `1.0e-7`

  ## References

    * [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  """
  def adagrad(learning_rate \\ 1.0e-3, opts \\ []) do
    Updates.scale_by_rss(opts)
    |> scale_by_learning_rate(learning_rate)
  end

  @doc """
  Adam optimizer.

  ## Options

    * `:b1` - first moment decay. Defaults to `0.9`
    * `:b2` - second moment decay. Defaults to `0.999`
    * `:eps` - numerical stability term. Defaults to `1.0e-8`
    * `:eps_root` - numerical stability term. Defaults to `1.0e-15`

  ## References

    * [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
  """
  def adam(learning_rate \\ 1.0e-3, opts \\ []) do
    Updates.scale_by_adam(opts)
    |> scale_by_learning_rate(learning_rate)
  end

  @doc """
  Adam with weight decay optimizer.

  ## Options

    * `:b1` - first moment decay. Defaults to `0.9`
    * `:b2` - second moment decay. Defaults to `0.999`
    * `:eps` - numerical stability term. Defaults to `1.0e-8`
    * `:eps_root` - numerical stability term. Defaults to `0.0`
    * `:decay` - weight decay. Defaults to `0.0`
  """
  def adamw(learning_rate \\ 1.0e-3, opts \\ []) do
    {decay, opts} = Keyword.pop(opts, :decay, 0.0)

    Updates.scale_by_adam(opts)
    |> Updates.add_decayed_weights(decay: decay)
    |> scale_by_learning_rate(learning_rate)
  end

  @doc """
  Lamb optimizer.

  ## Options

    * `:b1` - first moment decay. Defaults to `0.9`
    * `:b2` - second moment decay. Defaults to `0.999`
    * `:eps` - numerical stability term. Defaults to `1.0e-8`
    * `:eps_root` - numerical stability term. Defaults to `0.0`
    * `:decay` - weight decay. Defaults to `0.0`
    * `:min_norm` - minimum norm value. Defaults to `0.0`

  ## References

    * [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)
  """
  def lamb(learning_rate \\ 1.0e-2, opts \\ []) do
    {decay, opts} = Keyword.pop(opts, :decay, 0.0)
    {min_norm, opts} = Keyword.pop(opts, :min_norm, 0.0)

    Updates.scale_by_adam(opts)
    |> Updates.add_decayed_weights(decay: decay)
    |> Updates.scale_by_trust_ratio(min_norm: min_norm)
    |> scale_by_learning_rate(learning_rate)
  end

  @doc """
  Noisy SGD optimizer.

  ## Options

    * `:eta` - used to compute variance of noise distribution. Defaults to `0.1`
    * `:gamma` - used to compute variance of noise distribution. Defaults to `0.55`
  """
  def noisy_sgd(learning_rate \\ 1.0e-2, opts \\ []) do
    scale_by_learning_rate(learning_rate)
    |> Updates.add_noise(opts)
  end

  @doc """
  Rectified Adam optimizer.

  ## Options

    * `:b1` - first moment decay. Defaults to `0.9`
    * `:b2` - second moment decay. Defaults to `0.999`
    * `:eps` - numerical stability term. Defaults to `1.0e-8`
    * `:eps_root` - numerical stability term. Defaults to `0.0`
    * `:threshold` - threshold term. Defaults to `5.0`

  ## References

    * [On the Variance of Adaptive Learning Rate and Beyond](https://arxiv.org/pdf/1908.03265.pdf)
  """
  def radam(learning_rate \\ 1.0e-3, opts \\ []) do
    Updates.scale_by_radam(opts)
    |> scale_by_learning_rate(learning_rate)
  end

  @doc """
  RMSProp optimizer.

  ## Options

    * `:centered` - whether to scale by centered root of EMA of squares. Defaults to `false`
    * `:momentum` - momentum term. If set, uses SGD with momentum and decay set
      to value of this term.
    * `:nesterov` - whether or not to use nesterov momentum. Defaults to `false`
    * `:initial_scale` - initial value of EMA. Defaults to `0.0`
    * `:decay` - EMA decay rate. Defaults to `0.9`
    * `:eps` - numerical stability term. Defaults to `1.0e-8`
  """
  def rmsprop(learning_rate \\ 1.0e-2, opts \\ []) do
    {centered, opts} = Keyword.pop(opts, :centered, false)
    {nesterov?, opts} = Keyword.pop(opts, :nesterov, false)
    {momentum, opts} = Keyword.pop(opts, :momentum, nil)

    combinator =
      if centered do
        Updates.scale_by_stddev(opts)
      else
        Updates.scale_by_rms(opts)
      end
      |> scale_by_learning_rate(learning_rate)

    if momentum,
      do: Updates.trace(combinator, decay: momentum, nesterov: nesterov?),
      else: combinator
  end

  @doc """
  SGD optimizer.

  ## Options

    * `:momentum` - momentum term. If set, uses SGD with momentum and decay set
      to value of this term.
    * `:nesterov` - whether or not to use nesterov momentum. Defaults to `false`
  """
  def sgd(learning_rate \\ 1.0e-2, opts \\ []) do
    momentum = opts[:momentum]
    nesterov? = opts[:nesterov] || false

    if momentum do
      Updates.trace(decay: momentum, nesterov: nesterov?)
      |> scale_by_learning_rate(learning_rate)
    else
      scale_by_learning_rate(learning_rate)
    end
  end

  @doc """
  Yogi optimizer.

  ## Options

    * `:initial_accumulator_value` - initial value for first and second moment. Defaults to `0.0`
    * `:b1` - first moment decay. Defaults to `0.9`
    * `:b2` - second moment decay. Defaults to `0.999`
    * `:eps` - numerical stability term. Defaults to `1.0e-8`
    * `:eps_root` - numerical stability term. Defaults to `0.0`

  ## References

    * [Adaptive Methods for Nonconvex Optimization](https://papers.nips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf)
  """
  def yogi(learning_rate \\ 1.0e-2, opts \\ []) do
    Updates.scale_by_yogi(opts)
    |> scale_by_learning_rate(learning_rate)
  end

  ## Helpers

  defp scale_by_learning_rate(combinator \\ Updates.identity(), lr)

  defp scale_by_learning_rate(combinator, schedule) when is_function(schedule, 1) do
    Updates.scale_by_schedule(combinator, fn count -> Nx.negate(schedule.(count)) end)
  end

  defp scale_by_learning_rate(combinator, lr) do
    Updates.scale_by_state(combinator, -lr)
  end
end
