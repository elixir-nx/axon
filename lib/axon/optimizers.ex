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
  RMSProp optimizer.
  """
  def rmsprop(learning_rate, opts \\ []) do
    centered = opts[:centered] || false
    nesterov? = opts[:nesterov] || false
    momentum = opts[:momentum]

    if centered do
      combinator =
        Updates.scale_by_stddev(opts)
        |> Updates.scale(-learning_rate)

      if momentum,
        do: combinator |> Updates.trace(decay: momentum, nesterov: nesterov?),
        else: combinator
    else
      combinator =
        Updates.scale_by_rms(opts)
        |> Updates.scale(-learning_rate)

      if momentum,
        do: combinator |> Updates.trace(decay: momentum, nesterov: nesterov?),
        else: combinator
    end
  end

  @doc """
  Radam optimizer.
  """
  def radam(learning_rate, opts \\ []) do
    Updates.scale_by_radam(opts)
    |> Updates.scale(-learning_rate)
  end
end