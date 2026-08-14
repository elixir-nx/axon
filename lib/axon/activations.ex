defmodule Axon.Activations do
  @moduledoc """
  Activation functions.

  Activation functions are element-wise, (typically) non-linear
  functions called on the output of another layer, such as
  a dense layer:

      x
      |> dense(weight, bias)
      |> relu()

  Activation functions output the "activation" or how active
  a given layer's neurons are in learning a representation
  of the data-generating distribution.

  Some activations are commonly used as output activations. For
  example `softmax` is often used as the output in multiclass
  classification problems because it returns a categorical
  probability distribution:

      iex> Axon.Activations.softmax(Nx.tensor([[1, 2, 3]], type: {:f, 32}))
      #Nx.Tensor<
        f32[1][3]
        [
          [0.09003057, 0.24472848, 0.66524094]
        ]
      >

  Other activations such as `tanh` or `sigmoid` are used because
  they have desirable properties, such as keeping the output
  tensor constrained within a certain range.

  Generally, the choice of activation function is arbitrary;
  although some activations work better than others in certain
  problem domains. For example ReLU (rectified linear unit)
  activation is a widely-accepted default. You can see
  a list of activation functions and implementations
  [here](https://paperswithcode.com/methods/category/activation-functions).

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn
  import Axon.Block
  import Axon.Shared

  @doc ~S"""
  Continuously-differentiable exponential linear unit activation.

  $$f(x_i) = \max(0, x_i) + \min(0, \alpha * e^{\frac{x_i}{\alpha}} - 1)$$

  ## Options

    * `alpha` - $\alpha$ in CELU formulation. Must be non-zero.
      Defaults to `1.0`

  ## Examples

      iex> Axon.Activations.celu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f32[7]
        [-0.95021296, -0.86466473, -0.63212055, 0.0, 1.0, 2.0, 3.0]
      >

      iex> Axon.Activations.celu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}))
      #Nx.Tensor<
        bf16[2][3]
        [
          [-0.63, -0.863, -0.95],
          [1.0, 2.0, 3.0]
        ]
      >

  ### Error cases

      iex> Axon.Activations.celu(Nx.tensor([0.0, 1.0, 2.0], type: {:f, 32}), alpha: 0.0)
      ** (ArgumentError) :alpha must be non-zero in CELU activation

  ## References

    * [Continuously Differentiable Exponential Linear Units](https://arxiv.org/pdf/1704.07483.pdf)

  """
  defblock CELU, celu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0)
    validate_celu_alpha!(opts[:alpha])

    Nx.select(Nx.greater(x, 0), x, opts[:alpha] * Nx.expm1(x / opts[:alpha]))
  end

  deftransformp validate_celu_alpha!(alpha) do
    if alpha == 0,
      do: raise(ArgumentError, ":alpha must be non-zero in CELU activation")
  end

  @doc ~S"""
  Exponential linear unit activation.

  Equivalent to `celu` for $\alpha = 1$

  $$f(x_i) = \begin{cases}x_i & x _i > 0 \newline \alpha * (e^{x_i} - 1) & x_i \leq 0 \\ \end{cases}$$

  ## Options

    * `alpha` - $\alpha$ in ELU formulation. Defaults to `1.0`

  ## Examples

      iex> Axon.Activations.elu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f32[7]
        [-0.95021296, -0.86466473, -0.63212055, 0.0, 1.0, 2.0, 3.0]
      >

      iex> Axon.Activations.elu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}))
      #Nx.Tensor<
        bf16[2][3]
        [
          [-0.63, -0.863, -0.95],
          [1.0, 2.0, 3.0]
        ]
      >

  ## References

    * [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

  """
  defblock ELU, elu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0)
    x_hat = Nx.select(Nx.greater(x, 0), 0, x)
    Nx.select(Nx.greater(x, 0), x, opts[:alpha] * Nx.expm1(x_hat))
  end

  @doc ~S"""
  Exponential activation.

  $$f(x_i) = e^{x_i}$$

  ## Examples

      iex> Axon.Activations.exp(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [0.049787067, 0.13533528, 0.36787945, 1.0, 2.7182817, 7.389056, 20.085537]
      >

      iex> Axon.Activations.exp(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.367, 0.135, 0.0496],
          [2.7, 7.38, 20.0]
        ]
      >

  """
  defblock exp(x) do
    Nx.exp(x)
  end

  @doc ~S"""
  Gaussian error linear unit activation.

  $$f(x_i) = \frac{x_i}{2}(1 + {erf}(\frac{x_i}{\sqrt{2}}))$$

  ## Examples

      iex> Axon.Activations.gelu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.0040496886, -0.04550028, -0.15865526, 0.0, 0.8413447, 1.9544997, 2.9959502]
      >

      iex> Axon.Activations.gelu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.16, -0.0469, -0.00586],
          [0.84, 1.95, 2.98]
        ]
      >

  ## References

    * [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)

  """
  defblock GeLU, gelu(x) do
    sqrt2 = Nx.sqrt(Nx.tensor(2, type: Nx.type(x)))

    x
    |> Nx.divide(sqrt2)
    |> Nx.erf()
    |> Nx.add(1)
    |> Nx.multiply(x)
    |> Nx.divide(2)
  end

  @doc ~S"""
  Hard sigmoid activation.

  ## Examples

      iex> Axon.Activations.hard_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8]
      >

      iex> Axon.Activations.hard_sigmoid(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.0, 0.0, 0.0],
          [0.398, 0.598, 0.797]
        ]
      >

  """
  defblock hard_sigmoid(x, opts \\ []) do
    opts = keyword!(opts, alpha: 0.2, beta: 0.2)

    x
    |> Nx.multiply(opts[:alpha])
    |> Nx.add(opts[:beta])
    |> Nx.max(0)
    |> Nx.min(1)
  end

  @doc ~S"""
  Hard sigmoid weighted linear unit activation.

  $$f(x_i) = \begin{cases} 0 & x_i \leq -3 \newline
  x & x_i \geq 3 \newline
  \frac{x_i^2}{6} + \frac{x_i}{2} & otherwise \end{cases}$$

  ## Examples

      iex> Axon.Activations.hard_silu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.0, -0.0, -0.0, 0.0, 0.4, 1.2, 2.4]
      >

      iex> Axon.Activations.hard_silu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.0, -0.0, -0.0],
          [0.398, 1.195, 2.39]
        ]
      >

  """
  defblock HardSiLU, hard_silu(x, opts \\ []) do
    x
    |> hard_sigmoid(opts)
    |> Nx.multiply(x)
  end

  @doc ~S"""
  Hard hyperbolic tangent activation.

  $$f(x_i) = \begin{cases} 1 & x > 1 \newline -1 & x < -1 \newline x & otherwise \end{cases}$$

  ## Examples

      iex> Axon.Activations.hard_tanh(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]
      >

      iex> Axon.Activations.hard_tanh(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-1.0, -1.0, -1.0],
          [1.0, 1.0, 1.0]
        ]
      >

  """
  defblock hard_tanh(x) do
    Nx.select(
      Nx.greater(x, 1),
      1,
      Nx.select(Nx.less(x, -1), -1, x)
    )
  end

  @doc ~S"""
  Leaky rectified linear unit activation.

  $$f(x_i) = \begin{cases} x & x \geq 0 \newline \alpha * x & otherwise \end{cases}$$

  ## Options

    * `:alpha` - $\alpha$ in Leaky ReLU formulation. Defaults to `1.0e-2`

  ## Examples

      iex> Axon.Activations.leaky_relu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]), alpha: 0.5)
      #Nx.Tensor<
        f32[data: 7]
        [-1.5, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0]
      >

      iex> Axon.Activations.leaky_relu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], names: [:batch, :data]), alpha: 0.5)
      #Nx.Tensor<
        f32[batch: 2][data: 3]
        [
          [-0.5, -1.0, -1.5],
          [1.0, 2.0, 3.0]
        ]
      >

  """
  defblock LeakyReLU, leaky_relu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0e-2)
    Nx.select(Nx.greater(x, 0), x, x * opts[:alpha])
  end

  @doc ~S"""
  Linear activation.

  $$f(x_i) = x_i$$

  ## Examples

      iex> Axon.Activations.linear(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
      >

      iex> Axon.Activations.linear(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-1.0, -2.0, -3.0],
          [1.0, 2.0, 3.0]
        ]
      >

  """
  defblock(linear(x), do: x)

  @doc ~S"""
  Logsumexp activation.

  $$\log(sum e^x_i)$$

  ## Examples

      iex> Axon.Activations.log_sumexp(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 1]
        [3.4577627]
      >

      iex> Axon.Activations.log_sumexp(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 1]
        [
          [-0.594],
          [3.39]
        ]
      >

  """
  defblock LogSumExp, log_sumexp(x, opts \\ []) do
    opts = keyword!(opts, axis: -1)
    axes = wrap(opts[:axis])

    # Scaling term to prevent over/underflow; max treated as constant C.
    # See also: https://github.com/google/jax/pull/2260
    max_val = Nx.reduce_max(x, axes: axes, keep_axes: true)
    max_val = stop_grad(Nx.select(Nx.is_infinity(max_val), 0, max_val))

    stable_exp =
      x
      |> Nx.subtract(max_val)
      |> Nx.exp()

    stable_exp
    |> Nx.sum(axes: axes, keep_axes: true)
    |> Nx.log()
    |> Nx.add(max_val)
  end

  @doc ~S"""
  Log-sigmoid activation.

  $$f(x_i) = \log(sigmoid(x))$$

  ## Examples

      iex> Axon.Activations.log_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}, names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-3.0485873, -2.126928, -1.3132617, -0.6931472, -0.3132617, -0.12692802, -0.04858735]
      >

      iex> Axon.Activations.log_sigmoid(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-1.31, -2.12, -3.05],
          [-0.312, -0.126, -0.0483]
        ]
      >

  """
  defblock(log_sigmoid(x), do: -softplus(-x))

  @doc """
  Log-softmax activation.

  $$f(x_i) = -\log(\sum{e^x_i})$$

  ## Examples

      iex> Axon.Activations.log_softmax(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}, names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-6.4577627, -5.4577627, -4.4577627, -3.4577627, -2.4577627, -1.4577628, -0.45776284]
      >

      iex> Axon.Activations.log_softmax(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.404, -1.4, -2.39],
          [-2.39, -1.4, -0.404]
        ]
      >
  """
  defblock LogSoftMax, log_softmax(x, opts \\ []) do
    opts = keyword!(opts, axis: -1)

    shifted = x - stop_grad(Nx.reduce_max(x, axes: [opts[:axis]], keep_axes: true))

    shifted
    |> Nx.exp()
    |> Nx.sum(axes: [opts[:axis]], keep_axes: true)
    |> Nx.log()
    |> Nx.negate()
    |> Nx.add(shifted)
  end

  @doc ~S"""
  Mish activation.

  $$f(x_i) = x_i* \tanh(\log(1 + e^x_i))$$

  ## Examples

      iex> Axon.Activations.mish(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}, names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.14564745, -0.2525015, -0.30340147, 0.0, 0.8650984, 1.943959, 2.986535]
      >

      iex> Axon.Activations.mish(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.3, -0.25, -0.144],
          [0.863, 1.94, 2.97]
        ]
      >
  """
  defblock mish(x) do
    x * tanh(softplus(x))
  end

  @doc ~S"""
  Rectified linear unit activation.

  $$f(x_i) = \max_i(x, 0)$$

  ## Examples

      iex> Axon.Activations.relu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
      >

      iex> Axon.Activations.relu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.0, 0.0, 0.0],
          [1.0, 2.0, 3.0]
        ]
      >

  """
  defblock ReLU, relu(x) do
    custom_grad(
      Nx.max(x, 0),
      [x],
      fn g -> [Nx.select(Nx.greater(x, 0), g, Nx.broadcast(0, g))] end
    )
  end

  @doc ~S"""
  Rectified linear unit 6 activation.

  $$f(x_i) = \min_i(\max_i(x, 0), 6)$$

  ## Examples

      iex> Axon.Activations.relu6(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))
      #Nx.Tensor<
        f32[7]
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
      >

      iex> Axon.Activations.relu6(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.0, 0.0, 0.0],
          [1.0, 2.0, 3.0]
        ]
      >

  ## References

    * [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861v1)

  """
  defblock ReLU6, relu6(x) do
    x
    |> Nx.max(0)
    |> Nx.min(6)
  end

  @doc ~S"""
  Sigmoid activation.

  $$f(x_i) = \frac{1}{1 + e^{-x_i}}$$

  **Implementation Note: Sigmoid logits are cached as metadata
  in the expression and can be used in calculations later on.
  For example, they are used in cross-entropy calculations for
  better stability.**

  ## Examples

      iex> Axon.Activations.sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [0.047425874, 0.11920292, 0.26894143, 0.5, 0.7310586, 0.8807971, 0.95257413]
      >

      iex> Axon.Activations.sigmoid(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.268, 0.119, 0.0474],
          [0.73, 0.88, 0.95]
        ]
      >

  """
  defn sigmoid(x) do
    # Logits metadata must wrap the block result (not sit inside it) so
    # losses can pattern-match `%{logits: _}` on the returned tensor.
    cache_logits(x, sigmoid_block(x))
  end

  defblockp Sigmoid, sigmoid_block(x) do
    Nx.sigmoid(x)
  end

  @doc ~S"""
  Sigmoid weighted linear unit activation.

  $$f(x_i) = x * sigmoid(x)$$

  ## Examples

      iex> Axon.Activations.silu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.14227763, -0.23840584, -0.26894143, 0.0, 0.7310586, 1.7615942, 2.8577223]
      >

      iex> Axon.Activations.silu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.268, -0.238, -0.142],
          [0.73, 1.76, 2.84]
        ]
      >

  ## References

    * [Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning](https://arxiv.org/abs/1702.03118v3)

  """
  defblock SiLU, silu(x) do
    x
    |> Nx.sigmoid()
    |> Nx.multiply(x)
  end

  @doc ~S"""
  Scaled exponential linear unit activation.

  $$f(x_i) = \begin{cases} \lambda x & x \geq 0 \newline
  \lambda \alpha(e^{x} - 1) & x < 0 \end{cases}$$

  $$\alpha \approx 1.6733$$
  $$\lambda \approx 1.0507$$

  ## Examples

      iex> Axon.Activations.selu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-1.6705688, -1.5201665, -1.1113307, 0.0, 1.050701, 2.101402, 3.152103]
      >

      iex> Axon.Activations.selu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-1.09, -1.5, -1.66],
          [1.05, 2.1, 3.14]
        ]
      >

  ## References

    * [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515v5)

  """
  defblock SeLU, selu(x, opts \\ []) do
    opts =
      keyword!(opts,
        alpha: 1.6732632423543772848170429916717,
        gamma: 1.0507009873554804934193349852946
      )

    opts[:gamma] * elu(x, alpha: opts[:alpha])
  end

  @doc ~S"""
  Softmax activation along an axis.

  $$\frac{e^{x_i}}{\sum_i e^{x_i}}$$

  **Implementation Note: Softmax logits are cached as metadata
  in the expression and can be used in calculations later on.
  For example, they are used in cross-entropy calculations for
  better stability.**

  ## Options

    * `:axis` - softmax axis along which to calculate distribution.
      Defaults to 1.

  ## Examples

      iex> Axon.Activations.softmax(Nx.tensor([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]], names: [:batch, :data]))
      #Nx.Tensor<
        f32[batch: 1][data: 7]
        [
          [0.0015683004, 0.0042630825, 0.01158826, 0.031500153, 0.0856263, 0.23275642, 0.6326975]
        ]
      >

      iex> Axon.Activations.softmax(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.664, 0.243, 0.0894],
          [0.0894, 0.243, 0.664]
        ]
      >

  """
  defn softmax(x, opts \\ []) do
    # Logits metadata must wrap the block result (not sit inside it) so
    # losses can pattern-match `%{logits: _}` on the returned tensor.
    cache_logits(x, softmax_block(x, opts))
  end

  defblockp SoftMax, softmax_block(x, opts \\ []) do
    opts = keyword!(opts, axis: -1)
    axes = wrap(opts[:axis])

    # Scaling term to prevent over/underflow; max treated as constant C.
    # See also: https://github.com/google/jax/pull/2260
    max_val = stop_grad(Nx.reduce_max(x, axes: axes, keep_axes: true))

    stable_exp =
      x
      |> Nx.subtract(max_val)
      |> Nx.exp()

    stable_exp
    |> Nx.sum(axes: axes, keep_axes: true)
    |> reciprocal()
    |> Nx.multiply(stable_exp)
  end

  @doc ~S"""
  Softplus activation.

  $$\log(1 + e^x_i)$$

  ## Examples

      iex> Axon.Activations.softplus(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [0.04858735, 0.12692802, 0.3132617, 0.6931472, 1.3132617, 2.126928, 3.0485873]
      >

      iex> Axon.Activations.softplus(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.312, 0.126, 0.0483],
          [1.31, 2.12, 3.05]
        ]
      >

  """
  defblock SoftPlus, softplus(x) do
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

      iex> Axon.Activations.softsign(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.75, -0.6666667, -0.5, 0.0, 0.5, 0.6666667, 0.75]
      >

      iex> Axon.Activations.softsign(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.5, -0.664, -0.75],
          [0.5, 0.664, 0.75]
        ]
      >

  """
  defblock SoftSign, softsign(x) do
    x
    |> Nx.abs()
    |> Nx.add(1)
    |> reciprocal()
    |> Nx.multiply(x)
  end

  @doc ~S"""
  Swish-gated linear unit activation.

  SwiGLU splits the input tensor along the given axis into two equal
  halves $a$ and $b$, then returns $silu(a) \odot b$. The dimension of
  the input along the given axis must be divisible by 2.

  $$f(x) = silu(a) \odot b \quad \text{where} \quad x = [a, b]$$

  ## Options

    * `:axis` - axis along which to split the input into the activation
      and gate halves. Defaults to `-1`.

  ## Examples
      
      iex> Axon.Activations.swiglu(Nx.tensor([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]], names: [:batch, :data]))
      #Nx.Tensor<
        f32[batch: 1][data: 4]
        [
          [-0.14227763, -0.47681168, -0.8068243, 0.0]
        ]
      >

  ### Error cases

      iex> Axon.Activations.swiglu(Nx.tensor([1.0, 2.0, 3.0]))
      ** (ArgumentError) axis -1 of input to swiglu must have a dimension divisible by 2, got dimension of size 3

  ## References
    * [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
  """
  defblock SwiGLU, swiglu(x, opts \\ []) do
    opts = keyword!(opts, axis: -1)
    {a, b} = split_halves(x, opts[:axis])
    silu(a) * b
  end

  deftransformp split_halves(x, axis) do
    axis_idx = Nx.axis_index(x, axis)
    size = elem(Nx.shape(x), axis_idx)

    if rem(size, 2) != 0 do
      raise ArgumentError,
            "axis #{inspect(axis)} of input to swiglu must have a dimension" <>
              " divisible by 2, got dimension of size #{size}"
    end

    half = div(size, 2)
    a = Nx.slice_along_axis(x, 0, half, axis: axis_idx)
    b = Nx.slice_along_axis(x, half, half, axis: axis_idx)
    {a, b}
  end

  @doc ~S"""
  Hyperbolic tangent activation.

  $$f(x_i) = \tanh(x_i)$$

  ## Examples

      iex> Axon.Activations.tanh(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.9950548, -0.9640276, -0.7615942, 0.0, 0.7615942, 0.9640276, 0.9950548]
      >

      iex> Axon.Activations.tanh(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.758, -0.96, -0.992],
          [0.758, 0.96, 0.992]
        ]
      >

  """
  defblock(tanh(x), do: Nx.tanh(x))

  ## Helpers

  deftransformp cache_logits(input, output) do
    Nx.Defn.Expr.metadata(output, %{logits: input})
  end

  deftransformp wrap(axis), do: List.wrap(axis)
end
