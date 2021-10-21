defmodule Axon.Losses do
  @moduledoc """
  Loss functions.

  Loss functions evaluate predictions with respect to true
  data, often to measure the divergence between a model's
  representation of the data-generating distribution and the
  true representation of the data-generating distribution.

  Each loss function is implemented as an element-wise function
  measuring the loss with respect to the input target `y_true`
  and input prediction `y_pred`. As an example, the `mean_squared_error/2`
  loss function produces a tensor whose values are the mean squared
  error between targets and predictions:

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.mean_squared_error(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.5, 0.5]
      >

  It's common to compute the loss across an entire minibatch.
  You can easily do so by specifying a `:reduction` mode, or
  by composing one of these with an `Nx` reduction method:

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.mean_squared_error(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.5
      >

  You can even compose loss functions:

      defn my_strange_loss(y_true, y_pred) do
        y_true
        |> Axon.Losses.mean_squared_error(y_pred)
        |> Axon.Losses.binary_cross_entropy(y_pred)
        |> Nx.sum()
      end

  Or, more commonly, you can combine loss functions with penalties for
  regularization:

      defn regularized_loss(params, y_true, y_pred) do
        loss = Axon.mean_squared_error(y_true, y_pred)
        penalty = l2_penalty(params)
        Nx.sum(loss) + penalty
      end

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn
  import Axon.Shared

  @doc ~S"""
  Binary cross-entropy loss function.

  $$l_i = -\frac{1}{2}(\hat{y_i} \cdot \log(y_i) + (1 - \hat{y_i}) \cdot \log(1 - y_i))$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

    * `:negative_weights` - class weight for `0` class useful for scaling loss
      by importance of class. Defaults to `1.0`.

    * `:positive_weights` - class weight for `1` class useful for scaling loss
      by importance of class. Defaults to `1.0`.

  ## Examples

      iex> y_true = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> y_pred = Nx.tensor([[0.6811, 0.5565], [0.6551, 0.4551], [0.5422, 0.2648]])
      iex> Axon.Losses.binary_cross_entropy(y_true, y_pred)
      #Nx.Tensor<
        f32[3]
        [0.8644828796386719, 0.5150601863861084, 0.4598664939403534]
      >

      iex> y_true = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> y_pred = Nx.tensor([[0.6811, 0.5565], [0.6551, 0.4551], [0.5422, 0.2648]])
      iex> Axon.Losses.binary_cross_entropy(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.6131365299224854
      >

      iex> y_true = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> y_pred = Nx.tensor([[0.6811, 0.5565], [0.6551, 0.4551], [0.5422, 0.2648]])
      iex> Axon.Losses.binary_cross_entropy(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        1.839409589767456
      >

  """
  defn binary_cross_entropy(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, positive_weight: nil, negative_weight: nil, reduction: :none)

    # The default value of both weights mathematically is 1.0, but we've
    # initialized them to `nil` so we can match here and avoid this calculation
    # altogether if necessary. If either of them is set, then we need to set
    # both and perform this whole thing. If neither is set, we set this to
    # nil and then avoid the weighted avg later on.
    weights =
      transform({y_true, opts[:positive_weight], opts[:negative_weight]}, fn
        {_, nil, nil} ->
          nil

        {y_true, pos, nil} ->
          Nx.take(Nx.tensor([1.0, pos], backend: Nx.Defn.Expr), y_true)

        {y_true, nil, neg} ->
          Nx.take(Nx.tensor([neg, 1.0], backend: Nx.Defn.Expr), y_true)

        {y_true, pos, neg} ->
          Nx.take(Nx.tensor([neg, pos], backend: Nx.Defn.Expr), y_true)
      end)

    loss_before_avg = -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred)

    # Rather than add a redundant multiplication here if there are no weights,
    # we'll match on the weights value above.
    possibly_weighted_avg_loss =
      transform({loss_before_avg, weights}, fn
        {loss, nil} ->
          Nx.mean(loss, axes: [-1])

        {loss, weights} ->
          Nx.mean(weights * loss)
      end)

    transform(
      {opts[:reduction], possibly_weighted_avg_loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Categorical cross-entropy loss function.

  $$l_i = -\sum_i^C \hat{y_i} \cdot \log(y_i)$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

    * `:class_weights` - 1-D weights tensor corresponding to weight of each
      class useful for scaling loss according to importance of class. Tensor
      size must match number of classes in dataset. Defaults to `1.0` for all
      classes.

  ## Examples

      iex> y_true = Nx.tensor([[0, 1, 0], [0, 0, 1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
      iex> Axon.Losses.categorical_cross_entropy(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.051293306052684784, 2.3025851249694824]
      >

      iex> y_true = Nx.tensor([[0, 1, 0], [0, 0, 1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
      iex> Axon.Losses.categorical_cross_entropy(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        1.1769392490386963
      >

      iex> y_true = Nx.tensor([[0, 1, 0], [0, 0, 1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
      iex> Axon.Losses.categorical_cross_entropy(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        2.3538784980773926
      >

  """
  defn categorical_cross_entropy(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, class_weights: nil, reduction: :none)

    weights =
      transform({y_true, opts[:class_weights]}, fn
        {_, nil} ->
          nil

        {y_true, [_ | _] = class_weights} ->
          unless Elixir.Kernel.==(length(class_weights), elem(Nx.shape(y_true), 1)) do
            raise ArgumentError,
                  "expected class weights to be a 1-dimensional list" <>
                    " with size equal to the number of classes present" <>
                    " in dataset, got #{inspect(class_weights)} for data" <>
                    " with #{inspect(elem(Nx.shape(y_true), 1))} classes"
          end

          Nx.take(Nx.tensor(class_weights, backend: Nx.Defn.Expr), Nx.argmax(y_true, axis: 1))

        {_, invalid} ->
          raise ArgumentError,
                "expected class weights to be a 1-dimensional list" <>
                  " with size equal to the number of classes present" <>
                  " in dataset, got #{inspect(invalid)} for data" <>
                  " with #{inspect(elem(Nx.shape(y_true), 1))} classes"
      end)

    loss_before_avg =
      y_true
      |> xlogy(y_pred)
      |> Nx.negate()
      |> Nx.sum(axes: [-1])

    possibly_weighted_avg_loss =
      transform({weights, loss_before_avg}, fn
        {nil, loss} ->
          loss

        {weights, loss} ->
          weights * loss
      end)

    transform(
      {opts[:reduction], weights, possibly_weighted_avg_loss},
      fn
        {:mean, weights, loss} ->
          case weights do
            nil ->
              Nx.mean(loss)

            weights ->
              Nx.sum(loss) / Nx.sum(weights)
          end

        {:sum, _, loss} ->
          Nx.sum(loss)

        {:none, _, loss} ->
          loss
      end
    )
  end

  @doc ~S"""
  Categorical hinge loss function.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Examples

      iex> y_true = Nx.tensor([[1, 0, 0], [0, 0, 1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.05300799, 0.21617081, 0.68642382], [0.3754382 , 0.08494169, 0.13442067]])
      iex> Axon.Losses.categorical_hinge(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [1.6334158182144165, 1.2410175800323486]
      >

      iex> y_true = Nx.tensor([[1, 0, 0], [0, 0, 1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.05300799, 0.21617081, 0.68642382], [0.3754382 , 0.08494169, 0.13442067]])
      iex> Axon.Losses.categorical_hinge(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        1.4372167587280273
      >

      iex> y_true = Nx.tensor([[1, 0, 0], [0, 0, 1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.05300799, 0.21617081, 0.68642382], [0.3754382 , 0.08494169, 0.13442067]])
      iex> Axon.Losses.categorical_hinge(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        2.8744335174560547
      >
  """
  defn categorical_hinge(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, reduction: :none)

    loss =
      1
      |> Nx.subtract(y_true)
      |> Nx.multiply(y_pred)
      |> Nx.reduce_max(axes: [-1])
      |> Nx.subtract(Nx.sum(Nx.multiply(y_true, y_pred), axes: [-1]))
      |> Nx.add(1)
      |> Nx.max(0)

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Hinge loss function.

  $$\frac{1}{C}\max_i(1 - \hat{y_i} * y_i, 0)$$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> y_true = Nx.tensor([[ 1,  1, -1], [ 1,  1, -1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.45440044, 0.31470688, 0.67920924], [0.24311459, 0.93466766, 0.10914676]])
      iex> Axon.Losses.hinge(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.9700339436531067, 0.6437881588935852]
      >

      iex> y_true = Nx.tensor([[ 1,  1, -1], [ 1,  1, -1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.45440044, 0.31470688, 0.67920924], [0.24311459, 0.93466766, 0.10914676]])
      iex> Axon.Losses.hinge(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.806911051273346
      >

      iex> y_true = Nx.tensor([[ 1,  1, -1], [ 1,  1, -1]], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.45440044, 0.31470688, 0.67920924], [0.24311459, 0.93466766, 0.10914676]])
      iex> Axon.Losses.hinge(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        1.613822102546692
      >
  """
  defn hinge(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, reduction: :none)

    loss =
      y_true
      |> Nx.multiply(y_pred)
      |> Nx.negate()
      |> Nx.add(1)
      |> Nx.max(0)
      |> Nx.mean(axes: [-1])

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Kullback-Leibler divergence loss function.

  $$l_i = \sum_i^C \hat{y_i} \cdot \log(\frac{\hat{y_i}}{y_i})$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Examples

      iex> y_true = Nx.tensor([[0, 1], [0, 0]], type: {:u, 8})
      iex> y_pred = Nx.tensor([[0.6, 0.4], [0.4, 0.6]])
      iex> Axon.Losses.kl_divergence(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.916289210319519, -3.080907390540233e-6]
      >

      iex> y_true = Nx.tensor([[0, 1], [0, 0]], type: {:u, 8})
      iex> y_pred = Nx.tensor([[0.6, 0.4], [0.4, 0.6]])
      iex> Axon.Losses.kl_divergence(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.45814305543899536
      >

      iex> y_true = Nx.tensor([[0, 1], [0, 0]], type: {:u, 8})
      iex> y_pred = Nx.tensor([[0.6, 0.4], [0.4, 0.6]])
      iex> Axon.Losses.kl_divergence(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        0.9162861108779907
      >

  """
  defn kl_divergence(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, reduction: :none)
    epsilon = 1.0e-7
    y_true = Nx.clip(y_true, epsilon, 1)
    y_pred = Nx.clip(y_pred, epsilon, 1)

    loss =
      y_true
      |> Nx.divide(y_pred)
      |> Nx.log()
      |> Nx.multiply(y_true)
      |> Nx.sum(axes: [-1])

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Logarithmic-Hyperbolic Cosine loss function.

  $$l_i = \frac{1}{C} \sum_i^C (\hat{y_i} - y_i) + \log(1 + e^{-2(\hat{y_i} - y_i)}) - \log(2)$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]])
      iex> Axon.Losses.log_cosh(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.2168903946876526, 0.0]
      >

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]])
      iex> Axon.Losses.log_cosh(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.1084451973438263
      >

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]])
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]])
      iex> Axon.Losses.log_cosh(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        0.2168903946876526
      >
  """
  defn log_cosh(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, reduction: :none)

    x =
      y_pred
      |> Nx.subtract(y_true)

    loss =
      x
      |> Nx.multiply(-2)
      |> Nx.exp()
      |> Nx.log1p()
      |> Nx.add(x)
      |> Nx.subtract(Nx.log(2))
      |> Nx.mean(axes: [-1])

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Margin ranking loss function.

  $$l_i = \max(0, -\hat{y_i} * (y^(1)_i - y^(2)_i) + \alpha)$$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Examples

      iex> y_true = Nx.tensor([1.0, 1.0, 1.0], type: {:f, 32})
      iex> y_pred1 = Nx.tensor([0.6934, -0.7239,  1.1954], type: {:f, 32})
      iex> y_pred2 = Nx.tensor([-0.4691, 0.2670, -1.7452], type: {:f, 32})
      iex> Axon.Losses.margin_ranking(y_true, {y_pred1, y_pred2})
      #Nx.Tensor<
        f32[3]
        [0.0, 0.9909000396728516, 0.0]
      >

      iex> y_true = Nx.tensor([1.0, 1.0, 1.0], type: {:f, 32})
      iex> y_pred1 = Nx.tensor([0.6934, -0.7239,  1.1954], type: {:f, 32})
      iex> y_pred2 = Nx.tensor([-0.4691, 0.2670, -1.7452], type: {:f, 32})
      iex> Axon.Losses.margin_ranking(y_true, {y_pred1, y_pred2}, reduction: :mean)
      #Nx.Tensor<
        f32
        0.3303000032901764
      >

      iex> y_true = Nx.tensor([1.0, 1.0, 1.0], type: {:f, 32})
      iex> y_pred1 = Nx.tensor([0.6934, -0.7239,  1.1954], type: {:f, 32})
      iex> y_pred2 = Nx.tensor([-0.4691, 0.2670, -1.7452], type: {:f, 32})
      iex> Axon.Losses.margin_ranking(y_true, {y_pred1, y_pred2}, reduction: :sum)
      #Nx.Tensor<
        f32
        0.9909000396728516
      >
  """
  defn margin_ranking(y_true, {y_pred1, y_pred2}, opts \\ []) do
    assert_shape!(y_pred1, y_pred2)
    assert_shape!(y_true, y_pred1)

    opts = keyword!(opts, margin: 0.0, reduction: :none)
    margin = opts[:margin]

    loss =
      y_pred1
      |> Nx.subtract(y_pred2)
      |> Nx.multiply(Nx.negate(y_true))
      |> Nx.add(margin)
      |> Nx.max(0)

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Soft margin loss function.

  $$l_i = \sum_i \frac{\log(1 + e^{-\hat{y_i} * y_i})}{N}$$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Examples

      iex> y_true = Nx.tensor([[-1.0, 1.0,  1.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[0.2953, -0.1709, 0.9486]], type: {:f, 32})
      iex> Axon.Losses.soft_margin(y_true, y_pred)
      #Nx.Tensor<
        f32[3]
        [0.851658046245575, 0.7822436094284058, 0.3273470401763916]
      >

      iex> y_true = Nx.tensor([[-1.0, 1.0,  1.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[0.2953, -0.1709, 0.9486]], type: {:f, 32})
      iex> Axon.Losses.soft_margin(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.6537495255470276
      >

      iex> y_true = Nx.tensor([[-1.0, 1.0,  1.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[0.2953, -0.1709, 0.9486]], type: {:f, 32})
      iex> Axon.Losses.soft_margin(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        1.9612486362457275
      >
  """
  defn soft_margin(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, reduction: :none)

    loss =
      y_true
      |> Nx.negate()
      |> Nx.multiply(y_pred)
      |> Nx.exp()
      |> Nx.log1p()
      |> Nx.sum(axes: [0])

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Mean-absolute error loss function.

  $$l_i = \sum_i |\hat{y_i} - y_i|$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.mean_absolute_error(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.5, 0.5]
      >

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.mean_absolute_error(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.5
      >

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.mean_absolute_error(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        1.0
      >
  """
  defn mean_absolute_error(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, reduction: :none)

    loss =
      y_true
      |> Nx.subtract(y_pred)
      |> Nx.abs()
      |> Nx.mean(axes: [-1])

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Mean-squared error loss function.

  $$l_i = \sum_i (\hat{y_i} - y_i)^2$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.mean_squared_error(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.5, 0.5]
      >

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.mean_squared_error(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.5
      >

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.mean_squared_error(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        1.0
      >
  """
  defn mean_squared_error(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, reduction: :none)

    loss =
      y_true
      |> Nx.subtract(y_pred)
      |> Nx.power(2)
      |> Nx.mean(axes: [-1])

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  @doc ~S"""
  Poisson loss function.

  $$l_i = \frac{1}{C} \sum_i^C y_i - (\hat{y_i} \cdot \log(y_i))$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.poisson(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.9999999403953552, 0.0]
      >

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.poisson(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.4999999701976776
      >

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> Axon.Losses.poisson(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        0.9999999403953552
      >
  """
  defn poisson(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, reduction: :none)

    epsilon = 1.0e-7

    loss =
      y_pred
      |> Nx.add(epsilon)
      |> Nx.log()
      |> Nx.multiply(y_true)
      |> Nx.negate()
      |> Nx.add(y_pred)
      |> Nx.mean(axes: [-1])

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end
end
