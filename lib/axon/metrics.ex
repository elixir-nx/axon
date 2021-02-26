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
        f64
        0.6666666666666666
      >

  """
  defn accuracy(y_true, y_pred) do
    assert_shape!(y_true, y_pred)

    Nx.mean(
      Nx.equal(
        Nx.argmax(y_true, axis: -1),
        Nx.argmax(y_pred, axis: -1)
      )
    )
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
        f64
        0.6666666666666666
      >

  """
  defn precision(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, threshold: 0.5)
    thresholded_preds = Nx.greater(y_pred, opts[:threshold])
    positives = Nx.sum(thresholded_preds)
    true_positives = Nx.sum(Nx.logical_and(Nx.equal(y_true, thresholded_preds), positives))
    Nx.divide(true_positives, Nx.sum(positives) + 1.0e-16)
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
        f64
        0.6666666666666666
      >

  """
  defn recall(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, threshold: 0.5)
    thresholded_preds = Nx.greater(y_pred, opts[:threshold])

    true_positives =
      Nx.sum(Nx.logical_and(Nx.equal(y_true, thresholded_preds), Nx.equal(thresholded_preds, 1)))

    false_negatives =
      Nx.sum(
        Nx.logical_and(Nx.not_equal(y_true, thresholded_preds), Nx.equal(thresholded_preds, 0))
      )

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

      iex> Axon.Metrics.recall(Nx.tensor([0, 1, 1, 1]), Nx.tensor([1, 0, 1, 1]))
      #Nx.Tensor<
        f64
        0.6666666666666666
      >

  """
  defn sensitivity(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, thresold: 0.5)
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
        f64
        0.0
      >

  """
  defn specificity(y_true, y_pred, opts \\ []) do
    assert_shape!(y_true, y_pred)

    opts = keyword!(opts, threshold: 0.5)
    thresholded_preds = Nx.greater(y_pred, opts[:threshold])

    true_negatives =
      Nx.sum(Nx.logical_and(Nx.equal(y_true, thresholded_preds), Nx.equal(thresholded_preds, 0)))

    false_positives =
      Nx.sum(
        Nx.logical_and(Nx.not_equal(y_true, thresholded_preds), Nx.equal(thresholded_preds, 1))
      )

    Nx.divide(true_negatives, false_positives + true_negatives + 1.0e-16)
  end
end
