defmodule Axon.Metrics do
  @moduledoc """
  Metric functions.

  Metrics are used to measure the performance and compare
  performance of models in easy-to-understand terms. Often
  times, neural networks use surrogate loss functions such
  as negative log-likelihood to indirectly optimize a certain
  performance metric. Metrics such as accuracy, also called
  the 0-1 loss, do not have useful derivatives (e.g. they
  are information sparse), and are often intractable even
  with low input dimensions.

  Despite not being able to train specifically for certain
  metrics, it's still useful to track these metrics to
  monitor the performance of a neural network during training.
  Metrics such as accuracy provide useful feedback during
  training, whereas loss can sometimes be difficult to interpret.
    
  You can attach any of these functions as metrics within the
  `Axon.Loop` API using `Axon.Loop.metric/3`.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn

  # Standard Metrics

  @doc ~S"""
  Computes the accuracy of the given predictions.

  If the size of the last axis is 1, it performs a binary
  accuracy computation with a threshold of 0.5. Otherwise,
  computes categorical accuracy.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> Axon.Metrics.accuracy(Nx.tensor([[1], [0], [0]]), Nx.tensor([[1], [1], [1]]))
      #Nx.Tensor<
        f32
        0.3333333432674408
      >

      iex> Axon.Metrics.accuracy(Nx.tensor([[0, 1], [1, 0], [1, 0]]), Nx.tensor([[0, 1], [1, 0], [0, 1]]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

      iex> Axon.Metrics.accuracy(Nx.tensor([[0, 1, 0], [1, 0, 0]]), Nx.tensor([[0, 1, 0], [0, 1, 0]]))
      #Nx.Tensor<
        f32
        0.5
      >

  """
  defn accuracy(y_true, y_pred) do
    if elem(Nx.shape(y_pred), Nx.rank(y_pred) - 1) == 1 do
      y_pred
      |> Nx.greater(0.5)
      |> Nx.equal(y_true)
      |> Nx.mean()
    else
      y_true
      |> Nx.argmax(axis: -1)
      |> Nx.equal(Nx.argmax(y_pred, axis: -1))
      |> Nx.mean()
    end
  end

  @doc ~S"""
  Computes the precision of the given predictions with
  respect to the given targets.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

  ## Examples

      iex> Axon.Metrics.precision(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn precision(y_true, y_pred, opts \\ []) do
    true_positives = true_positives(y_true, y_pred, opts)
    false_positives = false_positives(y_true, y_pred, opts)

    true_positives
    |> Nx.divide(true_positives + false_positives + 1.0e-16)
  end

  @doc ~S"""
  Computes the recall of the given predictions with
  respect to the given targets.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

  ## Examples

      iex> Axon.Metrics.recall(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn recall(y_true, y_pred, opts \\ []) do
    true_positives = true_positives(y_true, y_pred, opts)
    false_negatives = false_negatives(y_true, y_pred, opts)

    Nx.divide(true_positives, false_negatives + true_positives + 1.0e-16)
  end

  @doc """
  Computes the number of true positive predictions with respect
  to given targets.

  ## Options

    * `:threshold` - threshold for truth value of predictions.
      Defaults to `0.5`.

  ## Examples

      iex> y_true = Nx.tensor([1, 0, 1, 1, 0, 1, 0])
      iex> y_pred = Nx.tensor([0.8, 0.6, 0.4, 0.2, 0.8, 0.2, 0.2])
      iex> Axon.Metrics.true_positives(y_true, y_pred)
      #Nx.Tensor<
        u64
        1
      >
  """
  defn true_positives(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 1))
    |> Nx.sum()
  end

  @doc """
  Computes the number of false negative predictions with respect
  to given targets.

  ## Options

    * `:threshold` - threshold for truth value of predictions.
      Defaults to `0.5`.

  ## Examples

      iex> y_true = Nx.tensor([1, 0, 1, 1, 0, 1, 0])
      iex> y_pred = Nx.tensor([0.8, 0.6, 0.4, 0.2, 0.8, 0.2, 0.2])
      iex> Axon.Metrics.false_negatives(y_true, y_pred)
      #Nx.Tensor<
        u64
        3
      >
  """
  defn false_negatives(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.not_equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 0))
    |> Nx.sum()
  end

  @doc """
  Computes the number of true negative predictions with respect
  to given targets.

  ## Options

    * `:threshold` - threshold for truth value of predictions.
      Defaults to `0.5`.

  ## Examples

      iex> y_true = Nx.tensor([1, 0, 1, 1, 0, 1, 0])
      iex> y_pred = Nx.tensor([0.8, 0.6, 0.4, 0.2, 0.8, 0.2, 0.2])
      iex> Axon.Metrics.true_negatives(y_true, y_pred)
      #Nx.Tensor<
        u64
        1
      >
  """
  defn true_negatives(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 0))
    |> Nx.sum()
  end

  @doc """
  Computes the number of false positive predictions with respect
  to given targets.

  ## Options

    * `:threshold` - threshold for truth value of predictions.
      Defaults to `0.5`.

  ## Examples

      iex> y_true = Nx.tensor([1, 0, 1, 1, 0, 1, 0])
      iex> y_pred = Nx.tensor([0.8, 0.6, 0.4, 0.2, 0.8, 0.2, 0.2])
      iex> Axon.Metrics.false_positives(y_true, y_pred)
      #Nx.Tensor<
        u64
        2
      >
  """
  defn false_positives(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.not_equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 1))
    |> Nx.sum()
  end

  @doc ~S"""
  Computes the sensitivity of the given predictions
  with respect to the given targets.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

  ## Examples

      iex> Axon.Metrics.sensitivity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn sensitivity(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, threshold: 0.5)

    recall(y_true, y_pred, opts)
  end

  @doc ~S"""
  Computes the specificity of the given predictions
  with respect to the given targets.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Options

    * `:threshold` - threshold for truth value of the predictions.
      Defaults to `0.5`

  ## Examples

      iex> Axon.Metrics.specificity(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f32
        0.0
      >

  """
  defn specificity(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds = Nx.greater(y_pred, opts[:threshold])

    true_negatives =
      thresholded_preds
      |> Nx.equal(y_true)
      |> Nx.logical_and(Nx.equal(thresholded_preds, 0))
      |> Nx.sum()

    false_positives =
      thresholded_preds
      |> Nx.not_equal(y_true)
      |> Nx.logical_and(Nx.equal(thresholded_preds, 1))
      |> Nx.sum()

    Nx.divide(true_negatives, false_positives + true_negatives + 1.0e-16)
  end

  @doc ~S"""
  Calculates the mean absolute error of predictions
  with respect to targets.

  $$l_i = \sum_i |\hat{y_i} - y_i|$$

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> y_true = Nx.tensor([[0.0, 1.0], [0.0, 0.0]], type: {:f, 32})
      iex> y_pred = Nx.tensor([[1.0, 1.0], [1.0, 0.0]], type: {:f, 32})
      iex> Axon.Metrics.mean_absolute_error(y_true, y_pred)
      #Nx.Tensor<
        f32
        0.5
      >
  """
  defn mean_absolute_error(y_true, y_pred) do
    y_true
    |> Nx.subtract(y_pred)
    |> Nx.abs()
    |> Nx.mean()
  end

  @doc ~S"""
  Computes the top-k categorical accuracy.

  ## Options

    * `k` - The k in "top-k". Defaults to 5.
    * `sparse` - If `y_true` is a sparse tensor. Defaults to `false`.


  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> Axon.Metrics.top_k_categorical_accuracy(Nx.tensor([0, 1, 0, 0, 0]), Nx.tensor([0.1, 0.4, 0.3, 0.7, 0.1]), k: 2)
      #Nx.Tensor<
        f32
        1.0
      >

      iex> Axon.Metrics.top_k_categorical_accuracy(Nx.tensor([[0, 1, 0], [1, 0, 0]]), Nx.tensor([[0.1, 0.4, 0.7], [0.1, 0.4, 0.7]]), k: 2)
      #Nx.Tensor<
        f32
        0.5
      >

      iex> Axon.Metrics.top_k_categorical_accuracy(Nx.tensor([[0], [2]]), Nx.tensor([[0.1, 0.4, 0.7], [0.1, 0.4, 0.7]]), k: 2, sparse: true)
      #Nx.Tensor<
        f32
        0.5
      >
  """
  defn top_k_categorical_accuracy(y_true, y_pred, opts \\ []) do
    opts = keyword!(opts, k: 5, sparse: false)

    y_true =
      transform(y_true, fn y_true ->
        if opts[:sparse] do
          y_true
        else
          top_k_index_transform(y_true)
        end
      end)

    cond do
      Nx.rank(y_pred) == 2 ->
        {rows, _} = Nx.shape(y_pred)

        y_pred
        |> Nx.argsort(direction: :desc, axis: -1)
        |> Nx.slice([0, 0], [rows, opts[:k]])
        |> Nx.equal(y_true)
        |> Nx.any(axes: [-1])
        |> Nx.mean()

      Nx.rank(y_pred) == 1 ->
        y_pred
        |> Nx.argsort(direction: :desc, axis: -1)
        |> Nx.slice([0], [opts[:k]])
        |> Nx.equal(y_true)
        |> Nx.any(axes: [-1])
        |> Nx.mean()

      true ->
        raise ArgumentError, "rank must be 1 or 2"
    end
  end

  defnp(top_k_index_transform(y_true), do: Nx.argmax(y_true, axis: -1, keep_axis: true))

  # Combinators

  @doc """
  Returns a function which computes a running average given current average,
  new observation, and current iteration.

  ## Examples

      iex> cur_avg = 0.5
      iex> iteration = 1
      iex> y_true = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> y_pred = Nx.tensor([[0, 1], [1, 0], [1, 0]])
      iex> avg_acc = Axon.Metrics.running_average(&Axon.Metrics.accuracy/2)
      iex> avg_acc.(cur_avg, [y_true, y_pred], iteration)
      #Nx.Tensor<
        f32
        0.75
      >
  """
  def running_average(metric) do
    &running_average_impl(&1, apply(metric, &2), &3)
  end

  defnp running_average_impl(avg, obs, i) do
    avg
    |> Nx.multiply(i)
    |> Nx.add(obs)
    |> Nx.divide(Nx.add(i, 1))
  end

  @doc """
  Returns a function which computes a running sum given current sum,
  new observation, and current iteration.

  ## Examples

      iex> cur_sum = 12
      iex> iteration = 2
      iex> y_true = Nx.tensor([0, 1, 0, 1])
      iex> y_pred = Nx.tensor([1, 1, 0, 1])
      iex> fps = Axon.Metrics.running_sum(&Axon.Metrics.false_positives/2)
      iex> fps.(cur_sum, [y_true, y_pred], iteration)
      #Nx.Tensor<
        s64
        13
      >
  """
  def running_sum(metric) do
    &running_sum_impl(&1, apply(metric, &2), &3)
  end

  defnp running_sum_impl(sum, obs, _) do
    Nx.add(sum, obs)
  end
end
