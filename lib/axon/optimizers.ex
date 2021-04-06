defmodule Axon.Optimizers do
  @moduledoc """
  Implementations of common gradient-based optimization algorithms.

  All of the methods in this module are written in terms of
  the update methods defined in Axon.Updates. Axon treates
  optimizers as the tuple:

      {init_fn, update_fn}

  where init_fn returns an initial optimizer state and update_fn
  scales input gradients. init_fn accepts a model's parameters
  to and attaches state to each parameter. update_fn accepts
  gradients, optimizer state, and current model parameters and
  returns updated optimizer state and gradients.

  As an example, consider the following usage of the Adam optimizer
  in a basic update function (assuming objective and the dataset are
  defined elsewhere):

      defmodule Learning do

        defn update(params, optimizer_state, inputs, targets, update_fn) do
          {loss, gradient} = value_and_grad(params, objective(&1, inputs, targets))
          {new_optimizer_state, scaled_updates} = update_fn.(gradient, optimizer_state, params)
          {Axon.Updates.apply_updates(params, scaled_updates), new_optimizer_state, loss}
        end

      model_params = Nx.random_uniform({784, 10})

      {init_fn, update_fn} = Axon.Optimizers.adam(0.005)
      optimizer_state = Nx.Defn.jit(init_fn, [model_params], compiler: EXLA)

      {new_params, new_optimizer_state, loss} = Learning.update(params, optimizer_state, inputs, targets, update_fn)

  For a simpler approach, you can also use optimizers with the training API:

        model
        |> Axon.Training.step(:categorical_cross_entropy, Axon.Optimizers.adam(0.005))
        |> Axon.Training.train(train_images, train_labels, epochs: 10, compiler: EXLA)

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
  def adabelief(learning_rate, opts \\ []) do
    Updates.scale_by_belief(opts)
    |> Updates.scale(-learning_rate)
  end

  @doc """
  Adagrad optimizer.
  """
  def adagrad(learning_rate, opts \\ []) do
    Updates.scale_by_rss(opts)
    |> Updates.scale(-learning_rate)
  end

  @doc """
  Adam optimizer.

  ## Options

    * `:b1` - first moment decay. Defaults to `0.9`
    * `:b2` - second moment decay. Defaults to `0.999`
    * `:eps` - numerical stability term. Defaults to `1.0e-8`
    * `:eps_root` - numerical stability term. Defaults to `0.0`

  ## References

    * [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
  """
  def adam(learning_rate, opts \\ []) do
    Updates.scale_by_adam(opts)
    |> Updates.scale(-learning_rate)
  end

  @doc """
  Adam with weight decay.
  """
  def adamw(learning_rate, opts \\ []) do
    Updates.scale_by_adam(opts)
    |> Updates.add_decayed_weights(opts)
    |> Updates.scale(-learning_rate)
  end

  @doc """
  Fromage optimizer.
  """
  def fromage(learning_rate, opts \\ []) do
    mult = Nx.divide(1, Nx.sqrt(Nx.add(1, Nx.power(learning_rate, 2))))

    Updates.scale_by_trust_ratio(opts)
    |> Updates.scale(Nx.multiply(-learning_rate, mult))
    |> Updates.add_decayed_weights(decay: Nx.subtract(mult, 1))
  end

  @doc """
  Lamb optimizer.
  """
  def lamb(learning_rate, opts \\ []) do
    Updates.scale_by_adam(opts)
    |> Updates.add_decayed_weights(opts)
    |> Updates.scale_by_trust_ratio(opts)
    |> Updates.scale(-learning_rate)
  end

  @doc """
  Noisy SGD
  """
  def noisy_sgd(learning_rate, opts \\ []) do
    Updates.scale(-learning_rate)
    |> Updates.add_noise(opts)
  end

  @doc """
  Radam optimizer.
  """
  def radam(learning_rate, opts \\ []) do
    Updates.scale_by_radam(opts)
    |> Updates.scale(-learning_rate)
  end

  @doc """
  RMSProp optimizer.
  """
  def rmsprop(learning_rate, opts \\ []) do
    centered = opts[:centered] || false
    nesterov? = opts[:nesterov] || false
    momentum = opts[:momentum]

    combinator =
      if centered do
        Updates.scale_by_stddev(opts)
      else
        Updates.scale_by_rms(opts)
      end
      |> Updates.scale(-learning_rate)

    if momentum,
      do: Updates.trace(combinator, decay: momentum, nesterov: nesterov?),
      else: combinator
  end

  @doc """
  SGD optimizer.
  """
  def sgd(learning_rate, opts \\ []) do
    momentum = opts[:momentum]
    nesterov? = opts[:nesterov] || false

    if momentum do
      Updates.trace(decay: momentum, nesterov: nesterov?)
      |> Updates.scale(-learning_rate)
    else
      Updates.scale(-learning_rate)
    end
  end

  @doc """
  Yogi optimizer.
  """
  def yogi(learning_rate, opts \\ []) do
    Updates.scale_by_yogi(opts)
    |> Updates.scale(-learning_rate)
  end
end
