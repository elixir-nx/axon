defmodule Axon.Optimizers do
  @moduledoc false

  @deprecated "Use Polaris.Optimizers.adabelief/1 instead"
  def adabelief(learning_rate \\ 1.0e-3, opts \\ []) do
    Polaris.Optimizers.adabelief([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.adagrad/1 instead"
  def adagrad(learning_rate \\ 1.0e-3, opts \\ []) do
    Polaris.Optimizers.adagrad([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.adam/1 instead"
  def adam(learning_rate \\ 1.0e-3, opts \\ []) do
    Polaris.Optimizers.adam([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.adamw/1 instead"
  def adamw(learning_rate \\ 1.0e-3, opts \\ []) do
    Polaris.Optimizers.adamw([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.lamb/1 instead"
  def lamb(learning_rate \\ 1.0e-2, opts \\ []) do
    Polaris.Optimizers.lamb([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.noisy_sgd/1 instead"
  def noisy_sgd(learning_rate \\ 1.0e-2, opts \\ []) do
    Polaris.Optimizers.noisy_sgd([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.radam/1 instead"
  def radam(learning_rate \\ 1.0e-3, opts \\ []) do
    Polaris.Optimizers.radam([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.rmsprop/1 instead"
  def rmsprop(learning_rate \\ 1.0e-2, opts \\ []) do
    Polaris.Optimizers.rmsprop([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.sgd/1 instead"
  def sgd(learning_rate \\ 1.0e-2, opts \\ []) do
    Polaris.Optimizers.sgd([learning_rate: learning_rate] ++ opts)
  end

  @deprecated "Use Polaris.Optimizers.yogi/1 instead"
  def yogi(learning_rate \\ 1.0e-2, opts \\ []) do
    Polaris.Optimizers.yogi([learning_rate: learning_rate] ++ opts)
  end
end
