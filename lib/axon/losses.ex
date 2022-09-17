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
  require Logger

  @doc ~S"""
  Binary cross-entropy loss function.

  $$l_i = -\frac{1}{2}(\hat{y_i} \cdot \log(y_i) + (1 - \hat{y_i}) \cdot \log(1 - y_i))$$

  Binary cross-entropy loss is most often used in binary classification problems.
  By default, it expects `y_pred` to encode probabilities from `[0.0, 1.0]`, typically
  as the output of the sigmoid function or another function which squeezes values
  between 0 and 1. You may optionally set `from_logits: true` to specify that values
  are being sent as non-normalized values (e.g. weights with possibly infinite range).
  In this case, input values will be encoded as probabilities by applying the logistic
  sigmoid function before computing loss.

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

    * `:negative_weights` - class weight for `0` class useful for scaling loss
      by importance of class. Defaults to `1.0`.

    * `:positive_weights` - class weight for `1` class useful for scaling loss
      by importance of class. Defaults to `1.0`.

    * `:from_logits` - whether `y_pred` is a logits tensor. Defaults to `false`.

  ## Examples

      iex> y_true = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> y_pred = Nx.tensor([[0.6811, 0.5565], [0.6551, 0.4551], [0.5422, 0.2648]])
      iex> Axon.Losses.binary_cross_entropy(y_true, y_pred)
      #Nx.Tensor<
        f32[3]
        [0.8644826412200928, 0.5150600075721741, 0.45986634492874146]
      >

      iex> y_true = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> y_pred = Nx.tensor([[0.6811, 0.5565], [0.6551, 0.4551], [0.5422, 0.2648]])
      iex> Axon.Losses.binary_cross_entropy(y_true, y_pred, reduction: :mean)
      #Nx.Tensor<
        f32
        0.613136351108551
      >

      iex> y_true = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> y_pred = Nx.tensor([[0.6811, 0.5565], [0.6551, 0.4551], [0.5422, 0.2648]])
      iex> Axon.Losses.binary_cross_entropy(y_true, y_pred, reduction: :sum)
      #Nx.Tensor<
        f32
        1.8394089937210083
      >

  """
  defn binary_cross_entropy(y_true, y_pred, opts \\ []) do
    assert_shape!("Axon.Losses.binary_cross_entropy", "y_true", y_true, "y_pred", y_pred)

    opts =
      keyword!(opts,
        positive_weight: nil,
        negative_weight: nil,
        reduction: :none,
        from_logits: false
      )

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

    # Merge types before computing loss to prevent under/overflow. This
    # can especially happen when targets are encoded as u8 tensors. We
    # need to do it after the weights though because weights require the
    # integer representation
    {y_true, y_pred} =
      transform({y_true, y_pred}, fn {y_true, y_pred} ->
        merged_type = Nx.Type.merge(Nx.type(y_true), Nx.type(y_pred))
        {Nx.as_type(y_true, merged_type), Nx.as_type(y_pred, merged_type)}
      end)

    loss_before_avg =
      transform({opts[:from_logits], y_true, y_pred}, fn
        {true, y_true, y_pred} ->
          logits =
            case y_pred do
              %Nx.Tensor{data: %Nx.Defn.Expr{op: :metadata, args: [_, %{logits: logits}]}} ->
                Logger.warning(
                  "Axon.Losses.binary_cross_entropy/3 received from_logits: true" <>
                    " but y_pred was produced from sigmoid or softmax activation"
                )

                logits

              _ ->
                y_pred
            end

          sigmoid_cross_entropy_from_logits(y_true, logits)

        {false, y_true, y_pred} ->
          case y_pred do
            %Nx.Tensor{data: %Nx.Defn.Expr{op: :metadata, args: [_, %{logits: logits}]}} ->
              # This is the path Keras takes when the output is a sigmoid
              # and it seems to be the more numerically stable path in those
              # cases, so we cache logits as metadata in sigmoid and then use
              # the logits to compute cross entropy here
              sigmoid_cross_entropy_from_logits(y_true, logits)

            _ ->
              # Otherwise we compute BCE with this path
              eps = 1.0e-7
              y_pred = Nx.clip(y_pred, eps, 1 - eps)

              # Compute cross entropy loss
              p = y_true * Nx.log(y_pred + eps)
              not_p = (1 - y_true) * Nx.log(1 - y_pred + eps)

              Nx.negate(p + not_p)
          end
      end)

    # Rather than add a redundant multiplication here if there are no weights,
    # we'll match on the weights value above.
    possibly_weighted_avg_loss =
      transform({loss_before_avg, weights}, fn
        {loss, nil} ->
          Nx.mean(loss, axes: [-1])

        {loss, weights} ->
          Nx.mean(weights * loss)
      end)

    reduction(possibly_weighted_avg_loss, opts[:reduction])
  end

  defnp sigmoid_cross_entropy_from_logits(y_true, y_pred) do
    log_p = Axon.Activations.log_sigmoid(y_pred)
    log_not_p = Axon.Activations.log_sigmoid(-y_pred)
    -y_true * log_p - (1 - y_true) * log_not_p
  end

  @doc ~S"""
  Categorical cross-entropy loss function.

  $$l_i = -\sum_i^C \hat{y_i} \cdot \log(y_i)$$

  Categorical cross-entropy is typically used for multi-class classifcation problems.
  By default, it expects `y_pred` to encode a probability distribution along the last
  axis. You can specify `from_logits: true` to indicate `y_pred` is a logits tensor.

      # Batch size of 3 with 3 target classes
      y_true = Nx.tensor([0, 2, 1])
      y_pred = Nx.tensor([[0.2, 0.8, 0.0], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

    * `:class_weights` - 1-D list corresponding to weight of each
      class useful for scaling loss according to importance of class. Tensor
      size must match number of classes in dataset. Defaults to `1.0` for all
      classes.

    * `:from_logits` - whether `y_pred` is a logits tensor. Defaults to `false`.

    * `:sparse` - whether `y_true` encodes a "sparse" tensor. In this case the
      inputs are integer values corresponding to the target class. Defaults to
      `false`.

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

      iex> y_true = Nx.tensor([1, 2], type: {:s, 8})
      iex> y_pred = Nx.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
      iex> Axon.Losses.categorical_cross_entropy(y_true, y_pred, reduction: :sum, sparse: true)
      #Nx.Tensor<
        f32
        2.3538784980773926
      >

  """
  defn categorical_cross_entropy(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, class_weights: nil, reduction: :none, from_logits: false, sparse: false)

    # As with binary cross entropy, we try to avoid the weights calculations
    # if they are unnecessary. We also have to do some input validation to
    # ensure the passed weights are correct for the given targets. The length
    # of the weights list must match the size of the last dimension of the targets.
    weights =
      transform({y_true, opts[:class_weights]}, fn
        {_, nil} ->
          nil

        {y_true, [_ | _] = class_weights} ->
          unless Elixir.Kernel.==(
                   length(class_weights),
                   elem(Nx.shape(y_true), Nx.rank(y_true) - 1)
                 ) do
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
      transform({opts[:from_logits], opts[:sparse], y_true, y_pred}, fn
        {true, sparse, y_true, y_pred} ->
          logits =
            case y_pred do
              %Nx.Tensor{data: %Nx.Defn.Expr{op: :metadata, args: [_, %{logits: logits}]}} ->
                Logger.warning(
                  "Axon.Losses.categorical_cross_entropy/3 received from_logits: true" <>
                    " but y_pred was produced from sigmoid or softmax activation"
                )

                logits

              _ ->
                y_pred
            end

          softmax_cross_entropy_from_logits(y_true, logits, sparse: sparse)

        {false, sparse, y_true, y_pred} ->
          case y_pred do
            %Nx.Tensor{data: %Nx.Defn.Expr{op: :metadata, args: [_, %{logits: logits}]}} ->
              softmax_cross_entropy_from_logits(y_true, logits)

            _ ->
              case sparse do
                true ->
                  # If y_true is not at least rank 2, add a new axis to select
                  # one index per value along the batch axis
                  y_true =
                    if Elixir.Kernel.<(Nx.rank(y_true), 2) do
                      Nx.new_axis(y_true, -1)
                    else
                      y_true
                    end

                  # Now we need to ensure the last axis is size 1, e.g. 1 value
                  # per index in the batch axis
                  unless Elixir.Kernel.==(elem(Nx.shape(y_true), Nx.rank(y_true) - 1), 1) do
                    raise ArgumentError,
                          "target values must have size 1 in last dimension," <>
                            " got shape #{inspect(Nx.shape(y_true))}"
                  end

                  y_pred
                  |> Nx.take_along_axis(y_true, axis: -1)
                  |> Nx.log()
                  |> Nx.negate()
                  |> Nx.sum(axes: [-1])

                false ->
                  y_true
                  |> xlogy(y_pred)
                  |> Nx.negate()
                  |> Nx.sum(axes: [-1])
              end
          end
      end)

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

  defnp softmax_cross_entropy_from_logits(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, sparse: false)

    transform({opts[:sparse], y_true, y_pred}, fn
      {true, y_true, y_pred} ->
        # If y_true is not at least rank 2, add a new axis to select
        # one index per value along the batch axis
        y_true =
          if Elixir.Kernel.<(Nx.rank(y_true), 2) do
            Nx.new_axis(y_true, -1)
          else
            y_true
          end

        # Now we need to ensure the last axis is size 1, e.g. 1 value
        # per index in the batch axis
        unless Elixir.Kernel.==(elem(Nx.shape(y_true), Nx.rank(y_true) - 1), 1) do
          raise ArgumentError,
                "target values must have size 1 in last dimension," <>
                  " got shape #{inspect(Nx.shape(y_true))}"
        end

        # Finally compute the loss of values taken from targets
        # along last axis
        -Nx.sum(
          Nx.take_along_axis(Axon.Activations.log_softmax(y_pred, axis: -1), y_true, axis: -1),
          axes: [-1]
        )

      {false, y_true, y_pred} ->
        -Nx.sum(y_true * Axon.Activations.log_softmax(y_pred, axis: -1), axes: [-1])
    end)
  end

  @doc ~S"""
  Categorical hinge loss function.

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

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

    reduction(loss, opts[:reduction])
  end

  @doc ~S"""
  Hinge loss function.

  $$\frac{1}{C}\max_i(1 - \hat{y_i} * y_i, 0)$$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

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
    opts = keyword!(opts, reduction: :none)

    loss =
      y_true
      |> Nx.multiply(y_pred)
      |> Nx.negate()
      |> Nx.add(1)
      |> Nx.max(0)
      |> Nx.mean(axes: [-1])

    reduction(loss, opts[:reduction])
  end

  @doc ~S"""
  Kullback-Leibler divergence loss function.

  $$l_i = \sum_i^C \hat{y_i} \cdot \log(\frac{\hat{y_i}}{y_i})$$

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

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

    reduction(loss, opts[:reduction])
  end

  @doc ~S"""
  Logarithmic-Hyperbolic Cosine loss function.

  $$l_i = \frac{1}{C} \sum_i^C (\hat{y_i} - y_i) + \log(1 + e^{-2(\hat{y_i} - y_i)}) - \log(2)$$

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

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

    reduction(loss, opts[:reduction])
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
    opts = keyword!(opts, margin: 0.0, reduction: :none)
    margin = opts[:margin]

    loss =
      y_pred1
      |> Nx.subtract(y_pred2)
      |> Nx.multiply(Nx.negate(y_true))
      |> Nx.add(margin)
      |> Nx.max(0)

    reduction(loss, opts[:reduction])
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

    reduction(loss, opts[:reduction])
  end

  @doc ~S"""
  Mean-absolute error loss function.

  $$l_i = \sum_i |\hat{y_i} - y_i|$$

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

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
    opts = keyword!(opts, reduction: :none)

    loss =
      y_true
      |> Nx.subtract(y_pred)
      |> Nx.abs()
      |> Nx.mean(axes: [-1])

    reduction(loss, opts[:reduction])
  end

  @doc ~S"""
  Mean-squared error loss function.

  $$l_i = \sum_i (\hat{y_i} - y_i)^2$$

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

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
    opts = keyword!(opts, reduction: :none)

    loss =
      y_true
      |> Nx.subtract(y_pred)
      |> Nx.power(2)
      |> Nx.mean(axes: [-1])

    reduction(loss, opts[:reduction])
  end

  @doc ~S"""
  Cosine Similarity error loss function.

  $$l_i = \sum_i (\hat{y_i} - y_i)^2$$

  ## Argument Shapes

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

  ## Options

    * `:reduction` - reduction mode. One of `:mean`, `:sum`, or `:none`.
      Defaults to `:none`.
    * `:axes` - Defaults to `[1]`.
    * `:eps` - Defaults to `1.0e-6`.

  ## Examples

      iex> y_pred = Nx.tensor([[1.0, 0.0], [1.0, 1.0]])
      iex> y_true = Nx.tensor([[0.0, 1.0], [1.0, 1.0]])
      iex> Axon.Losses.cosine_similarity(y_true, y_pred)
      #Nx.Tensor<
        f32[2]
        [0.0, 1.0000001192092896]
      >
  """

  defn cosine_similarity(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, axes: [1], eps: 1.0e-6, reduction: :none)
    axes = opts[:axes]
    eps = opts[:eps]

    w12 = Nx.sum(y_true * y_pred, axes: axes)
    w1 = Nx.LinAlg.norm(y_true, axes: axes)
    w2 = Nx.LinAlg.norm(y_pred, axes: axes)
    n12 = Nx.max(w1 * w2, eps)
    loss = w12 / n12

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

    * `y_true` - $(d_0, d_1, ..., d_n)$
    * `y_pred` - $(d_0, d_1, ..., d_n)$

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

    reduction(loss, opts[:reduction])
  end

  @doc """
  Connectionist Temporal Classification loss.

  ## Argument Shapes

    * `l_true` - $\(B\)$
    * `y_true` - $\(B, S\)$
    * `y_pred` - $\(B, T, D\)$

  ## Options

  * `:reduction` - reduction mode. One of `:sum` or `:none`.
    Defaults to `:none`.

  ## Description
    `l_true` contains lengths of target sequences. Nonzero positive values.
    `y_true` contains target sequences. Each value represents a class
    of element in range of available classes 0 <= y < D. Blank element
    class is included in this range, but shouldn't be presented among
    y_true values. Maximum target sequence length should be lower or equal
    to `y_pred` sequence length: S <= T.
    `y_pred` - log probabilities of classes D along the
    prediction sequence T.

  """
  defn connectionist_temporal_classification({l_true, y_true}, y_pred, opts \\ []) do
    opts = keyword!(opts, blank: 0, reduction: :none)
    eps = Nx.tensor(1.0e-7)
    b_size = elem(Nx.shape(y_true), 0)
    t_max = elem(Nx.shape(y_pred), 1) - 1
    loss = Nx.broadcast(0.0, {b_size})

    # Add padding to y_true
    y_true = Nx.pad(y_true, opts[:blank], [{0, 0, 0}, {1, 1, 1}])
    s_true = Nx.multiply(l_true, 2)

    {loss, _, _, _, _} =
      while {loss, b = 0, y_true, s_true, y_pred}, b < b_size do
        # Get boundaries for available node paths.
        st_lims = get_limits(y_true[b], s_true[b], t_max)
        # Iterate node tree backwards.
        s_pred0 = iterate_tree(y_true[b], y_pred[b], st_lims, t_max)

        {loss_b, _, _, _} =
          while {loss_b = 0.0, s = st_lims[0][0], s_pred0, st_lims}, s <= st_lims[0][1] do
            {Nx.add(loss_b, Nx.exp(s_pred0[s])), s + 1, s_pred0, st_lims}
          end

        loss_b =
          Nx.add(loss_b, eps)
          |> Nx.log()
          |> Nx.abs()

        {Nx.put_slice(loss, [b], Nx.reshape(loss_b, {1})), b + 1, y_true, s_true, y_pred}
      end

    transform(
      {opts[:reduction], loss},
      fn
        {:mean, loss} -> Nx.divide(loss, l_true) |> Nx.mean()
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end

  defnp get_limits(y_true, s_max, t_max) do
    st_max = Nx.concatenate([Nx.tensor([1]), Nx.broadcast(s_max, {t_max})])
    # Iterate target to get upper boundary values for each sequence step.
    {st_max, _, t_fin, _, _, _} =
      while {st_max, s = 1, t = 1, y_true, t_max, s_max}, t <= t_max and s <= s_max - 2 do
        s =
          cond do
            y_true[s] != y_true[s + 2] -> s + 2
            true -> s + 1
          end

        {Nx.put_slice(st_max, [t], Nx.reshape(s, {1})), s, t + 1, y_true, t_max, s_max}
      end

    st_min =
      cond do
        t_fin == t_max + 1 ->
          st_max

        true ->
          st_min = Nx.broadcast(0, {t_max + 1})

          {st_min, _, _, _} =
            while {st_min, dt = 1, st_max, t_fin}, dt <= t_fin do
              {Nx.put_slice(st_min, [t_max - dt + 1], Nx.reshape(st_max[t_fin - dt], {1})),
               dt + 1, st_max, t_fin}
            end

          st_min
      end

    Nx.stack([st_min, st_max], axis: 1)
  end

  # Get `node transition` part
  defnp get_path_prob(s, y_true, prob_prev, s_lims_prev) do
    # Iterate over all possible transition paths
    {path_prob, _, _, _, _, _} =
      while {path_prob = Nx.broadcast(0.0, {3}), s, d = 0, y_true, prob_prev, s_lims_prev},
            d <= 2 do
        path_prob =
          cond do
            s + d < s_lims_prev[0] or s + d > s_lims_prev[1] ->
              path_prob

            d == 2 and y_true[s] == y_true[s + d] ->
              path_prob

            true ->
              Nx.put_slice(path_prob, [d], Nx.reshape(Nx.exp(prob_prev[s + d]), {1}))
          end

        {path_prob, s, d + 1, y_true, prob_prev, s_lims_prev}
      end

    path_prob
  end

  # Get iteration values for acceptable nodes at a sequence step.
  defnp get_prob(prob_prev, s_lims, s_lims_prev, y_true, y_pred) do
    eps = Nx.tensor(1.0e-7)
    # Process nodes one-by-one from lower to upper bound.
    {t_prob, _, _, _, _, _} =
      while {prob_prev, s = s_lims[0], y_true, y_pred, s_lims_prev, s_lims}, s <= s_lims[1] do
        # Get `node transition` part
        path_prob =
          get_path_prob(s, y_true, prob_prev, s_lims_prev)
          |> Nx.sum()
          |> Nx.add(eps)
          |> Nx.log()

        # Add `node probability` part
        s_prob =
          Nx.add(y_pred[y_true[s]], path_prob)
          |> Nx.reshape({1})

        {Nx.put_slice(prob_prev, [s], s_prob), s + 1, y_true, y_pred, s_lims_prev, s_lims}
      end

    t_prob
  end

  defnp iterate_tree(y_true, y_pred, st_lims, t_max) do
    s_tmax_min = st_lims[t_max][0]
    s_tmax_max = st_lims[t_max][1]
    tmax_pred = y_pred[t_max]
    tmax_prob = Nx.broadcast(0.0, Nx.shape(y_true))
    # Get initial data for backwards iteration.
    {tmax_prob, _, _, _, _} =
      while {tmax_prob, s = s_tmax_min, s_tmax_max, tmax_pred, y_true}, s <= s_tmax_max do
        {Nx.put_slice(tmax_prob, [s], Nx.reshape(tmax_pred[y_true[s]], {1})), s + 1, s_tmax_max,
         tmax_pred, y_true}
      end

    # Iterate node tree backwards.
    {t0_prob, _, _, _, _} =
      while {prob = tmax_prob, t = t_max - 1, y_true, y_pred, st_lims}, t >= 0 do
        # Get iteration values for acceptable nodes at a sequence step.
        prob = get_prob(prob, st_lims[t], st_lims[t + 1], y_true, y_pred[t])
        {prob, t - 1, y_true, y_pred, st_lims}
      end

    t0_prob
  end

  defnp reduction(loss, reduction \\ :none) do
    transform(
      {reduction, loss},
      fn
        {:mean, loss} -> Nx.mean(loss)
        {:sum, loss} -> Nx.sum(loss)
        {:none, loss} -> loss
      end
    )
  end
end
