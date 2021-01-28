defmodule Axon.Layers do
  @moduledoc """
  Functional implementations of common neural network layers.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` backend.
  """

  import Nx.Defn

  @doc ~S"""
  Dense layer.

  Linear transformation of the input such that:

  $$y = xW^T + b$$

  Both `input` and `weight` should be 2-dimensional tensors with
  shapes:

  ```
  input_shape = {batch_size, in_features}
  weight_shape = {in_features, out_features}
  ```

  `bias` should be a 1 dimensional tensor or a scalar.
  """
  defn dense(input, weight, bias) do
    transform(Nx.shape(input), &assert_rank(&1, 2))
    transform(Nx.shape(weight), &assert_rank(&1, 2))
    transform(Nx.shape(bias), &assert_rank(&1, 1))
    input
    |> Nx.dot([Nx.rank(input) - 1], weight, [0])
    |> Nx.add(bias)
  end

  # Helpers

  # TODO: Maybe it makes sense to not always use an argument
  # error but instead something generic

  defp assert_rank(shape, value) do
    rank = tuple_size(shape)
    unless rank == value, do:
      raise ArgumentError, "expected input with shape #{inspect(shape)} to have rank #{value}, got #{rank}"
  end
end
