defmodule Axon.Losses do
  @moduledoc """
  Collection of common loss functions.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` backend.

  Each of these functions are implemented element-wise with
  respect to each target/input pair.
  """

  # TODO: All of these need shape/type validations, probably via an
  # assert_shape transform

  # TODO: categorical_hinge/2 - requires `reduce_max`
  # TODO: cosine_similarity/2 - requires `norm`
  # TODO: ctc/2
  # TODO: neg_log_likelihood/2 - requires dynamic slicing

  import Nx.Defn

  @doc ~S"""
  Binary cross-entropy loss function.

  $$-\frac{1}{2}(\hat{y_i} \cdot \log(y_i) + (1 - \hat{y_i}) \cdot \log(1 - y_i))$$

  ## Examples

      iex> y_true = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> y_pred = Nx.tensor([[0.6811, 0.5565], [0.6551, 0.4551], [0.5422, 0.2648]])
      iex> Axon.Losses.binary_crossentropy(y_true, y_pred)
      #Nx.Tensor<
        f64[3]
        [0.8644829066163313, 0.5150601853186955, 0.4598665249291158]
      >
  """
  defn binary_crossentropy(y_true, y_pred) do
    Nx.mean(-xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred), axes: [-1])
  end

  @doc ~S"""
  Categorical cross-entropy loss function.

  $$-\sum_i^C \hat{y_i} \cdot \log(y_i)$$

  ## Examples

      iex> y_true = Nx.tensor([[0, 1, 0], [0, 0, 1]])
      iex> y_pred = Nx.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
      iex> Axon.Losses.categorical_crossentropy(y_true, y_pred)
      #Nx.Tensor<
        f64[2]
        [0.05129329438755058, 2.3025850929940455]
      >
  """
  defn categorical_crossentropy(y_true, y_pred) do
    -Nx.sum(xlogy(y_true, y_pred), axes: [-1])
  end

  @doc ~S"""
  Hinge loss function.

  $$\frac{1}{C}\max_i(1 - \hat{y_i} * y_i, 0)$$

  ## Examples

      iex> y_true = Nx.tensor([[ 1,  1, -1], [ 1,  1, -1]])
      iex> y_pred = Nx.tensor([[0.45440044, 0.31470688, 0.67920924], [0.24311459, 0.93466766, 0.10914676]])
      iex> Axon.Losses.hinge(y_true, y_pred)
      #Nx.Tensor<
        f64[2]
        [0.9700339733333333, 0.64378817]
      >
  """
  defn hinge(y_true, y_pred) do
    Nx.mean(Nx.max(1.0 - y_true * y_pred, 0.0), axes: [-1])
  end

  @doc ~S"""
  Kullback-Leibler divergence loss function.

  $$\sum_i^C \hat{y_i} \cdot \log(\frac{\hat{y_i}}{y_i})$$

  ## Examples

      iex> y_true = Nx.tensor([[0, 1], [0, 0]])
      iex> y_pred = Nx.tensor([[0.6, 0.4], [0.4, 0.6]])
      iex> Axon.Losses.kl_divergence(y_true, y_pred)
      #Nx.Tensor<
        f64[2]
        [0.9162891711471524, -3.080907494627649e-6]
      >
  """
  defn kl_divergence(y_true, y_pred) do
    # TODO: epsilon should be the fuzz factor
    epsilon = 1.0e-7
    y_true = Nx.clip(y_true, epsilon, 1.0)
    y_pred = Nx.clip(y_pred, epsilon, 1.0)
    Nx.sum(y_true * Nx.log(y_true / y_pred), axes: [-1])
  end

  @doc ~S"""
  Logarithmic-Hyperbolic Cosine loss function.

  $$\frac{1}{C} \sum_i^C (\hat{y_i} - y_i) + \log(1 + e^{-2(\hat{y_i} - y_i)}) - \log(2)$$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]])
      iex> Axon.Losses.log_cosh(y_true, y_pred)
      #Nx.Tensor<
        f64[2]
        [0.2168904152415137, 0.0]
      >
  """
  defn log_cosh(y_true, y_pred) do
    x = y_pred - y_true
    softplus_x = x + Nx.log1p(Nx.exp(-2.0 * x)) - Nx.log(2.0)
    Nx.mean(softplus_x, axes: [-1])
  end

  @doc ~S"""
  Mean-absolute error loss function.

  $$\sum_i |\hat{y_i} - y_i|$$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]])
      iex> Axon.Losses.mean_absolute_error(y_true, y_pred)
      #Nx.Tensor<
        f64[2]
        [0.5, 0.5]
      >
  """
  defn mean_absolute_error(y_true, y_pred) do
    Nx.mean(Nx.abs(y_true - y_pred), axes: [-1])
  end

  @doc ~S"""
  Mean-squared error loss function.

  $$\sum_i (\hat{y_i} - y_i)^2$$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]])
      iex> Axon.Losses.mean_squared_error(y_true, y_pred)
      #Nx.Tensor<
        f64[2]
        [0.5, 0.5]
      >
  """
  defn mean_squared_error(y_true, y_pred) do
    Nx.mean(Nx.power(y_true - y_pred, 2), axes: [-1])
  end

  @doc ~S"""
  Poisson loss function.

  $$ \frac{1}{C} \sum_i^C y_i - (\hat{y_i} \cdot \log(y_i))$$
  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]])
      iex> Axon.Losses.poisson(y_true, y_pred)
      #Nx.Tensor<
        f64[2]
        [0.9999999500000025, 0.0]
      >
  """
  defn poisson(y_true, y_pred) do
    # TODO: epsilon should be the fuzz factor
    epsilon = 1.0e-7
    Nx.mean(y_pred - y_true * Nx.log(y_pred + epsilon), axes: [-1])
  end

  # Helpers

  # TODO: Remove or simplify when there's a numerically stable log
  # function similar to this in the `Nx` API

  defnp xlogy(x, y) do
    x_ok = Nx.not_equal(x, 0.0)
    safe_x = Nx.select(x_ok, x, 1.0)
    safe_y = Nx.select(x_ok, y, 1.0)
    Nx.select(x_ok, safe_x * Nx.log(safe_y), 0.0)
  end
end
