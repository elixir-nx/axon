defmodule Axon.Losses do
  @moduledoc """
  Collection of common loss functions.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` backend.
  """

  # TODO: Label smoothing as an option
  # TODO: Sample weighting
  # TODO: Allow use control over the reduction rather than just calling `mean`
  # TODO: Should these return loss w.r.t entire batch? Or just a sample?
  # TODO: Should allow the user to select the batch

  # TODO: binary_crossentropy/2 - requires `reduce_max`
  # TODO: categorical_hinge/2 - requires `reduce_max`
  # TODO: cosine_similarity/2 - requires `norm`

  import Nx.Defn

  @doc """
  Kullback-Leibler divergence loss function.

  ## Examples

      iex> y_true = Nx.tensor([[0, 1], [0, 0]])
      iex> y_pred = Nx.tensor([[0.6, 0.4], [0.4, 0.6]])
      iex> Axon.Losses.kl_divergence(y_true, y_pred)
      #Nx.Tensor<
        f64
        0.45814304511982884
      >
  """
  defn kl_divergence(y_true, y_pred) do
    # TODO: epsilon should be the fuzz factor
    epsilon = 1.0e-7
    y_true = Nx.clip(y_true, epsilon, 1.0)
    y_pred = Nx.clip(y_pred, epsilon, 1.0)
    Nx.mean(Nx.sum(y_true * Nx.log(y_true / y_pred), axes: [0]))
  end

  @doc """
  Logarithmic-Hyperbolic Cosine loss function.

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]])
      iex> Axon.Losses.log_cosh(y_true, y_pred)
      #Nx.Tensor<
        f64
        0.10844520762075678
      >
  """
  defn log_cosh(y_true, y_pred) do
    x = y_pred - y_true
    Nx.mean(Nx.mean(Nx.log((Nx.exp(-x) + Nx.exp(x)) / 2), axes: [0]))
  end

  @doc """
  Mean-absolute error loss function.

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]])
      iex> Axon.Losses.mean_absolute_error(y_true, y_pred)
      #Nx.Tensor<
        f64
        0.5
      >
  """
  defn mean_absolute_error(y_true, y_pred) do
    Nx.mean(Nx.abs(y_true - y_pred))
  end

  @doc """
  Mean-squared error loss function.

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]])
      iex> Axon.Losses.mean_squared_error(y_true, y_pred)
      #Nx.Tensor<
        f64
        0.5
      >
  """
  defn mean_squared_error(y_true, y_pred) do
    Nx.mean(Nx.power(y_true - y_pred, 2))
  end

  @doc """
  Poisson loss function.

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]])
      iex> Axon.Losses.poisson(y_true, y_pred)
      #Nx.Tensor<
        f64
        0.5
      >
  """
  defn poisson(y_true, y_pred) do
    # TODO: epsilon should be the fuzz factor
    epsilon = 1.0e-7
    Nx.mean(Nx.mean(y_pred - (y_true * Nx.log(y_pred + epsilon)), axes: [0]))
  end
end