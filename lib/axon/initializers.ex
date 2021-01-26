defmodule Axon.Initializers do
  @moduledoc """
  Common parameter initializers.
  """

  # TODO: These should all be defn
  # TODO: Add random keys

  @doc """
  Initializes parameters to 0.

  ## Examples

      iex> Nx.zeros({2, 2})
      #Nx.Tensor<
        f32[2][2]
        [
          [0.0, 0.0],
          [0.0, 0.0]
        ]
      >
  """
  def zeros(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    Nx.broadcast(Nx.tensor(0, type: type), shape)
  end

  @doc """
  Initializes parameters to 1.

  ## Examples

      iex> Nx.ones({2, 2})
      #Nx.Tensor<
        f32[2][2]
        [
          [1.0, 1.0],
          [1.0, 1.0]
        ]
      >
  """
  def ones(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    Nx.broadcast(Nx.tensor(1, type: type), shape)
  end

  @doc """
  Initializes parameters with a random uniform distribution.
  """
  def uniform(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    scale = opts[:scale] || 1.0e-2
    Nx.random_uniform(shape, type: type) * scale
  end

  @doc """
  Initializes parameters with a random normal distribution.
  """
  def normal(shape, opts \\ []) do
    type = opts[:type] || {:f, 32}
    stddev = opts[:stddev] || 1.0e-2
    Nx.random_normal(shape, 0.0, stddev, type: type)
  end
end