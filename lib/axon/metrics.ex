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

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """

  import Nx.Defn
  import Axon.Shared

  @doc ~S"""
  Computes the accuracy of the given predictions, assuming
  both targets and predictions are one-hot encoded.

  ## Argument Shapes

    * `y_true` - $\(d_0, d_1, ..., d_n\)$
    * `y_pred` - $\(d_0, d_1, ..., d_n\)$

  ## Examples

      iex> Axon.Metrics.accuracy(Nx.tensor([[0, 1], [1, 0], [1, 0]]), Nx.tensor([[0, 1], [1, 0], [0, 1]]))
      #Nx.Tensor<
        f32
        0.6666666865348816
      >

  """
  defn accuracy(y_true, y_pred) do
    assert_shape!(y_true, y_pred)

    y_true
    |> Nx.argmax(axis: -1)
    |> Nx.equal(Nx.argmax(y_pred, axis: -1))
    |> Nx.mean()
  end

  # defndelegate mean_squared_error(y_true, y_pred), to: Axon.Losses
  # defndelegate mean_absolute_error(y_true, y_pred), to: Axon.Losses

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
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    thresholded_preds
    |> Nx.equal(y_true)
    |> Nx.logical_and(Nx.equal(thresholded_preds, 1))
    |> Nx.sum()
    |> Nx.divide(Nx.sum(thresholded_preds) + 1.0e-16)
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
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, threshold: 0.5)

    thresholded_preds =
      y_pred
      |> Nx.greater(opts[:threshold])

    true_positives =
      thresholded_preds
      |> Nx.equal(y_true)
      |> Nx.logical_and(Nx.equal(thresholded_preds, 1))
      |> Nx.sum()

    false_negatives =
      thresholded_preds
      |> Nx.not_equal(y_true)
      |> Nx.logical_and(Nx.equal(thresholded_preds, 0))
      |> Nx.sum()

    Nx.divide(true_positives, false_negatives + true_positives + 1.0e-16)
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
    assert_shape!(y_true, y_pred)

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
    assert_shape!(y_true, y_pred)

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
    assert_shape!(y_true, y_pred)

    y_true
    |> Nx.subtract(y_pred)
    |> Nx.abs()
    |> Nx.mean()
  end
end
