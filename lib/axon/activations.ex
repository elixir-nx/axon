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
          [0.09003057330846786, 0.2447284758090973, 0.6652409434318542]
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
        [-0.9502129554748535, -0.8646647334098816, -0.6321205496788025, 0.0, 1.0, 2.0, 3.0]
      >

      iex> Axon.Activations.celu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}))
      #Nx.Tensor<
        bf16[2][3]
        [
          [-0.62890625, -0.86328125, -0.94921875],
          [1.0, 2.0, 3.0]
        ]
      >

  ### Error cases

      iex> Axon.Activations.celu(Nx.tensor([0.0, 1.0, 2.0], type: {:f, 32}), alpha: 0.0)
      ** (ArgumentError) :alpha must be non-zero in CELU activation

  ## References

    * [Continuously Differentiable Exponential Linear Units](https://arxiv.org/pdf/1704.07483.pdf)

  """
  defn celu(x, opts \\ []) do
    opts = keyword!(opts, alpha: 1.0)

    transform(
      opts[:alpha],
      fn x ->
        if Elixir.Kernel.==(x, 0),
          do: raise(ArgumentError, ":alpha must be non-zero in CELU activation")
      end
    )

    Nx.select(Nx.greater(x, 0), x, opts[:alpha] * Nx.expm1(x / opts[:alpha]))
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
        [-0.9502129554748535, -0.8646647334098816, -0.6321205496788025, 0.0, 1.0, 2.0, 3.0]
      >

      iex> Axon.Activations.elu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}))
      #Nx.Tensor<
        bf16[2][3]
        [
          [-0.62890625, -0.86328125, -0.94921875],
          [1.0, 2.0, 3.0]
        ]
      >

  ## References

    * [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

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

      iex> Axon.Activations.exp(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [0.049787066876888275, 0.1353352814912796, 0.3678794503211975, 1.0, 2.7182817459106445, 7.389056205749512, 20.08553695678711]
      >

      iex> Axon.Activations.exp(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.3671875, 0.134765625, 0.049560546875],
          [2.703125, 7.375, 20.0]
        ]
      >

  """
  defn exp(x) do
    Nx.exp(x)
  end

  @doc ~S"""
  Gaussian error linear unit activation.

  $$f(x_i) = \frac{x_i}{2}(1 + {erf}(\frac{x_i}{\sqrt{2}}))$$

  ## Examples

      iex> Axon.Activations.gelu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.0040496885776519775, -0.04550027847290039, -0.15865525603294373, 0.0, 0.8413447141647339, 1.9544997215270996, 2.995950222015381]
      >

      iex> Axon.Activations.gelu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.16015625, -0.046875, -0.005859375],
          [0.83984375, 1.953125, 2.984375]
        ]
      >

  ## References

    * [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)

  """
  defn gelu(x) do
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
        [0.0, 0.0, 0.0, 0.20000000298023224, 0.4000000059604645, 0.6000000238418579, 0.800000011920929]
      >

      iex> Axon.Activations.hard_sigmoid(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [7.781982421875e-4, 0.0, 0.0],
          [0.3984375, 0.59765625, 0.796875]
        ]
      >

  """
  defn hard_sigmoid(x, opts \\ []) do
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
        [-0.0, -0.0, -0.0, 0.0, 0.4000000059604645, 1.2000000476837158, 2.4000000953674316]
      >

      iex> Axon.Activations.hard_silu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-7.781982421875e-4, -0.0, -0.0],
          [0.3984375, 1.1953125, 2.390625]
        ]
      >

  """
  defn hard_silu(x, opts \\ []) do
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
  defn hard_tanh(x) do
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
  defn leaky_relu(x, opts \\ []) do
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
  defn linear(x), do: x

  @doc ~S"""
  Logsumexp activation.

  $$\log(sum e^x_i)$$

  ## Examples

      iex> Axon.Activations.log_sumexp(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 1]
        [0.45776283740997314]
      >

      iex> Axon.Activations.log_sumexp(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 1]
        [
          [0.404296875],
          [0.404296875]
        ]
      >

  """
  defn log_sumexp(x, opts \\ []) do
    opts = keyword!(opts, axis: -1)
    axes = transform(opts[:axis], &List.wrap/1)

    # This is a scaling term designed to prevent over/under flow when x is very
    # large. Consider cases where the intermediate value e^x with large positive
    # x, e^x tends towards infinity or 0. This poisons the rest of the
    # calculation which would otherwise be normalized with the division by sum(e^x).
    # Thus we can scale by the max value in the tensor which guarantees all values
    # are smaller than 0.
    #
    # Given the expression is essentially:
    #
    # e^(x - C) / sum(e^(x - C))
    #
    # We are essentially treating the max value as a constant term, C. Thus there
    # is no need to differentiate through the max. See also: https://github.com/google/jax/pull/2260
    # for a note on performance.
    max_val = stop_grad(Nx.reduce_max(x, axes: axes, keep_axes: true))

    stable_exp =
      x
      |> Nx.subtract(max_val)
      |> Nx.exp()

    res =
      stable_exp
      |> Nx.sum(axes: axes, keep_axes: true)
      |> Nx.log()

    res
  end

  @doc ~S"""
  Log-sigmoid activation.

  $$f(x_i) = \log(\sigmoid(x))$$

  ## Examples

      iex> Axon.Activations.log_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}, names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-3.0485873222351074, -2.1269280910491943, -1.3132617473602295, -0.6931471824645996, -0.3132616877555847, -0.12692801654338837, -0.04858734831213951]
      >

      iex> Axon.Activations.log_sigmoid(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-1.3125, -2.125, -3.046875],
          [-0.3125, -0.1259765625, -0.04833984375]
        ]
      >

  """
  defn log_sigmoid(x), do: -softplus(-x)

  @doc """
  Log-softmax activation.

  $$f(x_i) = -\log(\sum{e^x_i})$$

  ## Examples

      iex> Axon.Activations.log_softmax(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], type: {:f, 32}, names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-6.457762718200684, -5.457762718200684, -4.457762718200684, -3.4577627182006836, -2.4577627182006836, -1.4577628374099731, -0.45776283740997314]
      >

      iex> Axon.Activations.log_softmax(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.404296875, -1.3984375, -2.390625],
          [-2.390625, -1.3984375, -0.404296875]
        ]
      >
  """
  defn log_softmax(x, opts \\ []) do
    opts = keyword!(opts, axis: -1)

    transform({x, opts}, fn {x, opts} ->
      if Elixir.Kernel.<=(Nx.rank(x), opts[:axis]) do
        raise ArgumentError, "log_softmax axis must be within rank of tensor"
      end
    end)

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
        [-0.14564745128154755, -0.2525014877319336, -0.30340147018432617, 0.0, 0.8650984168052673, 1.9439589977264404, 2.98653507232666]
      >

      iex> Axon.Activations.mish(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.30078125, -0.25, -0.1435546875],
          [0.86328125, 1.9375, 2.96875]
        ]
      >
  """
  defn mish(x) do
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
  defn relu(x) do
    custom_grad(
      Nx.max(x, 0),
      fn _ans, g -> [{x, Nx.select(Nx.greater(x, 0), g, Nx.broadcast(0, g))}] end
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
  defn relu6(x) do
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
        [0.04742587357759476, 0.11920291930437088, 0.2689414322376251, 0.5, 0.7310585975646973, 0.8807970881462097, 0.9525741338729858]
      >

      iex> Axon.Activations.sigmoid(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.267578125, 0.119140625, 0.04736328125],
          [0.73046875, 0.87890625, 0.94921875]
        ]
      >

  """
  defn sigmoid(x) do
    # Cache logits so they are available in certain calculations,
    # e.g. binary_cross_entropy and categorical_cross_entropy
    transform(Nx.sigmoid(x), &Nx.Defn.Expr.metadata(&1, %{logits: x}))
  end

  @doc ~S"""
  Sigmoid weighted linear unit activation.

  $$f(x_i) = x\sigmoid(x)$$

  ## Examples

      iex> Axon.Activations.silu(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.14227762818336487, -0.23840583860874176, -0.2689414322376251, 0.0, 0.7310585975646973, 1.7615941762924194, 2.857722282409668]
      >

      iex> Axon.Activations.silu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.267578125, -0.23828125, -0.1416015625],
          [0.73046875, 1.7578125, 2.84375]
        ]
      >

  ## References

    * [Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning](https://arxiv.org/abs/1702.03118v3)

  """
  defn silu(x) do
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
        [-1.670568823814392, -1.5201665163040161, -1.1113307476043701, 0.0, 1.0507010221481323, 2.1014020442962646, 3.1521029472351074]
      >

      iex> Axon.Activations.selu(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-1.09375, -1.5078125, -1.6640625],
          [1.046875, 2.09375, 3.140625]
        ]
      >

  ## References

    * [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515v5)

  """
  defn selu(x, opts \\ []) do
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
          [0.0015683004166930914, 0.004263082519173622, 0.011588259600102901, 0.03150015324354172, 0.08562629669904709, 0.23275642096996307, 0.6326975226402283]
        ]
      >

      iex> Axon.Activations.softmax(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.6640625, 0.2431640625, 0.08935546875],
          [0.08935546875, 0.2431640625, 0.6640625]
        ]
      >

  """
  defn softmax(x, opts \\ []) do
    opts = keyword!(opts, axis: -1)
    axes = transform(opts[:axis], &List.wrap/1)

    transform({x, axes}, fn {x, axes} ->
      Enum.each(axes, fn axis ->
        Nx.Shape.normalize_axis(Nx.shape(x), axis, Nx.names(x))
      end)
    end)

    # This is a scaling term designed to prevent over/under flow when x is very
    # large. Consider cases where the intermediate value e^x with large positive
    # x, e^x tends towards infinity or 0. This poisons the rest of the
    # calculation which would otherwise be normalized with the division by sum(e^x).
    # Thus we can scale by the max value in the tensor which guarantees all values
    # are smaller than 0.
    #
    # Given the expression is essentially:
    #
    # e^(x - C) / sum(e^(x - C))
    #
    # We are essentially treating the max value as a constant term, C. Thus there
    # is no need to differentiate through the max. See also: https://github.com/google/jax/pull/2260
    # for a note on performance.
    max_val = stop_grad(Nx.reduce_max(x, axes: axes, keep_axes: true))

    stable_exp =
      x
      |> Nx.subtract(max_val)
      |> Nx.exp()

    res =
      stable_exp
      |> Nx.sum(axes: axes, keep_axes: true)
      |> reciprocal()
      |> Nx.multiply(stable_exp)

    # Cache logits so they are available in certain calculations,
    # e.g. binary_cross_entropy and categorical_cross_entropy
    transform(res, &Nx.Defn.Expr.metadata(&1, %{logits: x}))
  end

  @doc ~S"""
  Softplus activation.

  $$\log(1 + e^x_i)$$

  ## Examples

      iex> Axon.Activations.softplus(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [0.04858734831213951, 0.12692801654338837, 0.3132616877555847, 0.6931471824645996, 1.3132617473602295, 2.1269280910491943, 3.0485873222351074]
      >

      iex> Axon.Activations.softplus(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [0.3125, 0.1259765625, 0.04833984375],
          [1.3125, 2.125, 3.046875]
        ]
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

      iex> Axon.Activations.softsign(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.75, -0.6666666865348816, -0.5, 0.0, 0.5, 0.6666666865348816, 0.75]
      >

      iex> Axon.Activations.softsign(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.5, -0.6640625, -0.75],
          [0.5, 0.6640625, 0.75]
        ]
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

      iex> Axon.Activations.tanh(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], names: [:data]))
      #Nx.Tensor<
        f32[data: 7]
        [-0.9950547814369202, -0.9640275835990906, -0.7615941762924194, 0.0, 0.7615941762924194, 0.9640275835990906, 0.9950547814369202]
      >

      iex> Axon.Activations.tanh(Nx.tensor([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], type: {:bf, 16}, names: [:batch, :data]))
      #Nx.Tensor<
        bf16[batch: 2][data: 3]
        [
          [-0.7578125, -0.9609375, -0.9921875],
          [0.7578125, 0.9609375, 0.9921875]
        ]
      >

  """
  defn tanh(x), do: Nx.tanh(x)
end
