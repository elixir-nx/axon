defmodule Axon.Activations do
  @moduledoc """
  Collection of common activation functions.

  Activation functions are element-wise, (typically) non-linear
  functions called on the output of another layer, such as
  a dense layer:

      x
      |> Axon.Layers.Dense(weight, bias)
      |> Axon.Activations.relu()

  Activation functions output the "activation" or how active
  a given layer's neurons are in learning a representation
  of the data-generating distribution.

  The choice of activation function is generally arbitrary;
  although some activations work better than others in certain
  problem domains. For example ReLU (rectified linear unit)
  activation is a widely-accepted default. You can see
  a list of activation functions and implementations
  [here](https://paperswithcode.com/methods/category/activation-functions).

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` backend.
  """

  import Nx.Defn

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
    Nx.select(Nx.greater(x, 0), x, opts[:alpha] * Nx.expm1(x / opts[:alpha]))
  end

  @doc """
  Exponential linear unit activation.

  ## Examples

      iex> Axon.Activations.elu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [-0.9502129554748535, -0.8646647334098816, -0.6321205496788025, 0.0, 1.0, 2.0, 3.0]
      >
  """
  defn elu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0)
    x_hat = Nx.select(Nx.greater(x, 0), 0, x)
    Nx.select(Nx.greater(x, 0), x, opts[:alpha] * Nx.expm1(x_hat))
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

  @doc ~S"""
  Gaussian error linear unit activation.

  $$f(x_i) = \frac{x}{2}(1 + \erf(\frac{x}{\sqrt{2}}))$$

  ## Examples

      iex> Axon.Activations.gelu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-0.00404969409489031, -0.04550026389635842, -0.15865525393145707, 0.0, 0.8413447460685429, 1.9544997361036416, 2.99595030590511]
      >
  """
  defn gelu(x) do
    sqrt2 = Nx.sqrt(2)

    x
    |> Nx.divide(sqrt2)
    |> Nx.erf()
    |> Nx.add(1)
    |> Nx.multiply(x)
    |> Nx.divide(2)
  end

  @doc """
  Hard sigmoid activation.

  ## Examples

      iex> Axon.Activations.hard_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [0.0, 0.1666666716337204, 0.3333333432674408, 0.5, 0.6666666865348816, 0.8333333134651184, 1.0]
      >
  """
  defn hard_sigmoid(x) do
    x
    |> Nx.add(3)
    |> relu6()
    |> Nx.divide(6)
  end

  @doc """
  Hard sigmoid weighted linear unit activation.

  ## Examples

      iex> Axon.Activations.hard_silu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [0.0, -0.3333333432674408, -0.3333333432674408, 0.0, 0.6666666865348816, 1.6666666269302368, 3.0]
      >
  """
  defn hard_silu(x) do
    x
    |> hard_sigmoid()
    |> Nx.multiply(x)
  end

  @doc """
  Hard hyperbolic tangent activation.

  ## Examples

      iex> Axon.Activations.hard_tanh(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]
      >
  """
  defn hard_tanh(x) do
    Nx.select(
      Nx.greater(x, 1),
      1,
      Nx.select(Nx.less(x, -1), -1, x)
    )
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
    Nx.select(Nx.greater(x, 0), x, x * opts[:alpha])
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
  defn linear(x), do: x

  @doc """
  Log-sigmoid activation.

  ## Examples

      iex> Axon.Activations.log_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f64[7]
        [-3.048587351573742, -2.1269280110429727, -1.3132616875182228, -0.6931471805599453, -0.31326168751822286, -0.1269280110429726, -0.04858735157374196]
      >
  """
  defn log_sigmoid(x), do: -softplus(-x)

  @doc ~S"""
  Rectified linear unit activation.

  $$f(x_i) = \max_i(x, 0)$$

  ## Examples

      iex> Axon.Activations.relu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
      >
  """
  defn relu(x) do
    custom_grad(
      Nx.max(x, 0),
      fn _ans, g -> Nx.select(Nx.greater(x, 0), g, Nx.broadcast(0, g)) end
    )
  end

  @doc ~S"""
  Rectified linear unit 6 activation.

  $$f(x_i) = \min_i(\max_i(x, 0), 6)$$

  ## Examples

      iex> Axon.Activations.relu6(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
      >
  """
  defn relu6(x) do
    x
    |> Nx.max(0)
    |> Nx.min(6)
  end

  @doc ~S"""
  Sigmoid activation.

  $$f(x_i) = \frac{1}{1 + e^{-x_i}}$$

  ## Examples

      iex> Axon.Activations.sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [0.04742587357759476, 0.11920291930437088, 0.2689414322376251, 0.5, 0.7310585975646973, 0.8807970881462097, 0.9525741338729858]
      >
  """
  defn sigmoid(x), do: Nx.logistic(x)

  @doc """
  Sigmoid weighted linear unit activation.

  ## Examples

      iex> Axon.Activations.silu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [-0.14227762818336487, -0.23840583860874176, -0.2689414322376251, 0.0, 0.7310585975646973, 1.7615941762924194, 2.857722282409668]
      >
  """
  defn silu(x) do
    x
    |> Nx.logistic()
    |> Nx.multiply(x)
  end

  @doc ~S"""
  Softmax activation.

  $$\frac{e^{x_i}}{\sum_i e^{x_i}}$$

  ## Examples

      iex> Axon.Activations.softmax(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [0.0015683004166930914, 0.004263082519173622, 0.011588259600102901, 0.03150015324354172, 0.08562629669904709, 0.23275642096996307, 0.6326975226402283]
      >
  """
  defn softmax(x) do
    max_val = Nx.reduce_max(x)

    stable_exp =
      x
      |> Nx.subtract(max_val)
      |> Nx.exp()

    stable_exp
    |> Nx.sum()
    |> reciprocal()
    |> Nx.multiply(stable_exp)
  end

  @doc ~S"""
  Softplus activation.

  $$\log(1 + e^x_i)$$

  ## Examples

      iex> Axon.Activations.softplus(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [0.04858734831213951, 0.12692801654338837, 0.3132616877555847, 0.6931471824645996, 1.3132617473602295, 2.1269280910491943, 3.0485873222351074]
      >
  """
  defn softplus(x) do
    stable = Nx.max(0.0, x)

    x
    |> Nx.abs()
    |> Nx.negate()
    |> Nx.exp()
    |> Nx.log1p()
    |> Nx.add(stable)
  end

  @doc ~S"""
  Softsign activation.

  $$f(x_i) = \frac{x_i}{|x_i| + 1}$$

  ## Examples

      iex> Axon.Activations.softsign(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [-0.75, -0.6666666865348816, -0.5, 0.0, 0.5, 0.6666666865348816, 0.75]
      >
  """
  defn softsign(x) do
    x
    |> Nx.abs()
    |> Nx.add(1)
    |> reciprocal()
    |> Nx.multiply(x)
  end

  @doc ~S"""
  Hyperbolic tangent activation.

  $$f(x_i) = \tanh(x_i)$$

  ## Examples

      iex> Axon.Activations.tanh(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}))
      #Nx.Tensor<
        f32[7]
        [-0.9950547814369202, -0.9640275835990906, -0.7615941762924194, 0.0, 0.7615941762924194, 0.9640275835990906, 0.9950547814369202]
      >
  """
  defn tanh(x), do: Nx.tanh(x)

  ## Helpers

  defnp reciprocal(x), do: Nx.divide(1, x)
end
