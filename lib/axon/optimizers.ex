defmodule Axon.Optimizers do
  @moduledoc """
  Common optimizers.
  """
  alias Axon.Updates

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
  Adam optimizer.
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
  Adabelief optimizer.
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
  Radam optimizer.
  """
  def radam(learning_rate, opts \\ []) do
    Updates.scale_by_radam(opts)
    |> Updates.scale(-learning_rate)
  end

  @doc """
  Yogi optimizer.
  """
  def yogi(learning_rate, opts \\ []) do
    Updates.scale_by_yogi(opts)
    |> Updates.scale(-learning_rate)
  end
end
