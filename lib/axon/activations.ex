defmodule Axon.Activations do
  @moduledoc """
  Collection of common activation functions.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` backend.
  """

  import Nx.Defn

  # TODO: Nx.gelu/1 - requires erf

  @doc ~S"""
  Continuously-differentiable exponential linear unit activation.

  ## Examples

      iex> Axon.Activations.celu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-0.950212931632136, -0.8646647167633873, -0.6321205588285577, 0.0, 1.0, 2.0, 3.0]
      >
  """
  defn celu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0)
    Nx.select(Nx.greater(x, 0.0), x, opts[:alpha] * Nx.expm1(x / opts[:alpha]))
  end

  @doc """
  Exponential linear unit activation.

  ## Examples

      iex> Axon.Activations.elu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-0.950212931632136, -0.8646647167633873, -0.6321205588285577, 0.0, 1.0, 2.0, 3.0]
      >
  """
  # TODO: change alpha to keyword
  defn elu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0)
    x_hat = Nx.select(Nx.greater(x, 0.0), 0.0, x)
    Nx.select(Nx.greater(x, 0.0), x, opts[:alpha] * Nx.expm1(x_hat))
  end

  @doc ~S"""
  Exponential activation.

  $$f(x_i) = e^{x_i}$$

  ## Examples

      iex> Axon.Activations.exp(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [0.049787068367863944, 0.1353352832366127, 0.36787944117144233, 1.0, 2.718281828459045, 7.38905609893065, 20.085536923187668]
      >
  """
  defn exp(x) do
    Nx.exp(x)
  end

  @doc """
  Hard sigmoid activation.

  ## Examples

      iex> Axon.Activations.hard_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [0.0, 0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0]
      >
  """
  defn hard_sigmoid(x) do
    relu6(x + 3.0) / 6.0
  end

  @doc """
  Hard sigmoid weighted linear unit activation.

  ## Examples

      iex> Axon.Activations.hard_silu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [0.0, -0.3333333333333333, -0.3333333333333333, 0.0, 0.6666666666666666, 1.6666666666666667, 3.0]
      >
  """
  defn hard_silu(x) do
    x * hard_sigmoid(x)
  end

  @doc """
  Hard hyperbolic tangent activation.

  ## Examples

      iex> Axon.Activations.hard_tanh(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]
      >
  """
  defn hard_tanh(x) do
    Nx.select(Nx.greater(x, 1.0), 1.0, Nx.select(Nx.less(x, -1.0), -1.0, x))
  end

  @doc """
  Leaky rectified linear unit activation.

  ## Examples

      iex> Axon.Activations.leaky_relu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-0.03, -0.02, -0.01, 0.0, 1.0, 2.0, 3.0]
      >
  """
  defn leaky_relu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0e-2)
    Nx.select(Nx.greater(x, 0.0), x, x * opts[:alpha])
  end

  @doc ~S"""
  Linear activation.

  $$f(x_i) = x_i$$

  ## Examples

      iex> Axon.Activations.linear(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
      >
  """
  defn(linear(x), do: x)

  @doc """
  Log-sigmoid activation.

  ## Examples

      iex> Axon.Activations.log_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-3.048587351573742, -2.1269280110429727, -1.3132616875182228, -0.6931471805599453, -0.31326168751822286, -0.1269280110429726, -0.04858735157374196]
      >
  """
  defn(log_sigmoid(x), do: -softplus(-x))

  @doc ~S"""
  Rectified linear unit activation.

  $$f(x_i) = \max_i(x, 0)$$

  ## Examples

      iex> Axon.Activations.relu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
      >
  """
  # TODO: custom gradient
  defn relu(x) do
    Nx.max(x, 0.0)
  end

  @doc ~S"""
  Rectified linear unit 6 activation.

  $$f(x_i) = \min_i(\max_i(x, 0), 6)$$

  ## Examples

      iex> Axon.Activations.relu6(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
      >
  """
  defn relu6(x) do
    Nx.min(Nx.max(x, 0.0), 6.0)
  end

  @doc ~S"""
  Sigmoid activation.

  $$f(x_i) = \frac{1}{1 + e^{-x_i}}$$

  ## Examples

      iex> Axon.Activations.sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [0.04742587317756678, 0.11920292202211755, 0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823, 0.9525741268224334]
      >
  """
  defn(sigmoid(x), do: Nx.logistic(x))

  @doc """
  Sigmoid weighted linear unit activation.

  ## Examples

      iex> Axon.Activations.silu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-0.14227761953270035, -0.2384058440442351, -0.2689414213699951, 0.0, 0.7310585786300049, 1.7615941559557646, 2.8577223804673]
      >
  """
  defn silu(x) do
    x * sigmoid(x)
  end

  @doc """
  Softmax activation.

  ## Examples

      iex> Axon.Activations.softmax(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [0.0015683003158864725, 0.004263082250240779, 0.011588259014055805, 0.03150015390138462, 0.08562629594379711, 0.23275640430228017, 0.6326975042723549]
      >
  """
  defn softmax(x) do
    Nx.exp(x) / Nx.sum(Nx.exp(x))
  end

  @doc """
  Softplus activation.

  ## Examples

      iex> Axon.Activations.softplus(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [0.04858735157374196, 0.1269280110429726, 0.31326168751822286, 0.6931471805599453, 1.3132616875182228, 2.1269280110429727, 3.048587351573742]
      >
  """
  defn softplus(x) do
    Nx.log1p(Nx.exp(x))
  end

  @doc ~S"""
  Softsign activation.

  $$f(x_i) = \frac{x_i}{|x_i| + 1}$$

  ## Examples

      iex> Axon.Activations.softsign(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-0.75, -0.6666666666666666, -0.5, 0.0, 0.5, 0.6666666666666666, 0.75]
      >
  """
  defn softsign(x) do
    x / (Nx.abs(x) + 1)
  end

  @doc ~S"""
  Hyperbolic tangent activation.

  $$f(x_i) = \tanh(x_i)$$

  ## Examples

      iex> Axon.Activations.tanh(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-0.9950547536867305, -0.9640275800758169, -0.7615941559557649, 0.0, 0.7615941559557649, 0.9640275800758169, 0.9950547536867305]
      >
  """
  defn(tanh(x), do: Nx.tanh(x))
end
