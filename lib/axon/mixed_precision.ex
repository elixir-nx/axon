defmodule Axon.MixedPrecision do
  @moduledoc """
  Utilities for creating mixed precision policies.

  Mixed precision is useful for increasing model throughput at the possible
  price of a small dip in accuracy. When creating a mixed precision policy,
  you define the policy for `params`, `compute`, and `output`.

  The `params` policy dictates what type parameters should be stored as
  during training. The `compute` policy dictates what type should be used
  during intermediate computations in the model's forward pass. The `output`
  policy dictates what type the model should output.

  Here's an example of creating a mixed precision policy and applying it
  to a model:

      model =
        Axon.input("input", shape: {nil, 784})
        |> Axon.dense(128, activation: :relu)
        |> Axon.batch_norm()
        |> Axon.dropout(rate: 0.5)
        |> Axon.dense(64, activation: :relu)
        |> Axon.batch_norm()
        |> Axon.dropout(rate: 0.5)
        |> Axon.dense(10, activation: :softmax)

      policy = Axon.MixedPrecision.create_policy(
        params: {:f, 32},
        compute: {:f, 16},
        output: {:f, 32}
      )

      mp_model =
        model
        |> Axon.MixedPrecision.apply_policy(policy, except: [:batch_norm])

  The example above applies the mixed precision policy to every layer in
  the model except Batch Normalization layers. The policy will cast parameters
  and inputs to `{:f, 16}` for intermediate computations in the model's forward
  pass before casting the output back to `{:f, 32}`.
  """

  alias Axon.MixedPrecision.Policy

  @doc """
  Creates a mixed precision policy with the given options.

  ## Options

    * `params` - parameter precision policy. Defaults to `{:f, 32}`
    * `compute` - compute precision policy. Defaults to `{:f, 32}`
    * `output` - output precision policy. Defaults to `{:f, 32}`

  ## Examples

      iex> Axon.MixedPrecision.create_policy(params: {:f, 16}, output: {:f, 16})
      %Policy{params: {:f, 16}, compute: {:f, 32}, output: {:f, 16}}

      iex> Axon.MixedPrecision.create_policy(compute: {:bf, 16})
      %Policy{params: {:f, 32}, compute: {:bf, 16}, output: {:f, 32}}
  """
  def create_policy(opts \\ []) do
    params = opts[:params] || {:f, 32}
    compute = opts[:compute] || {:f, 32}
    output = opts[:output] || {:f, 32}

    %Policy{params: params, compute: compute, output: output}
  end

  @doc """
  Applies mixed precision policy `policy` to every layer in the
  given model which returns true for `filter`.

  `filter` may be a function or one of `:only` or `:except` - which define
  filters for specific operations in the model. You may only use one of
  `:only`, `:except`, or a function:

      # Only applies to dense layers
      Axon.MixedPrecision.apply_policy(model, policy, only: [:dense])

      # Applies to every layer but batch norm
      Axon.MixedPrecision.apply_policy(model, policy, except: [:batch_norm])

      # A more complex application using filters
      Axon.MixedPrecision.apply_policy(model, policy, fn
        %Axon{op: :dense} -> true
        %Axon{op: :batch_norm} -> false
        %Axon{op: :conv} -> false
        %Axon{op: _} -> true
      end)
  """
  def apply_policy(%Axon{} = axon, %Policy{} = policy, filter) when is_function(filter) do
    Axon.map_nodes(axon, fn layer ->
      if filter.(layer) do
        %{layer | policy: policy}
      else
        layer
      end
    end)
  end

  @doc false
  def apply_policy(axon, policy, only: only) do
    filter = fn %Axon.Node{op: op} ->
      Enum.member?(only, op)
    end

    apply_policy(axon, policy, filter)
  end

  @doc false
  def apply_policy(axon, policy, except: exceptions) do
    filter = fn %Axon.Node{op: op} ->
      not Enum.member?(exceptions, op)
    end

    apply_policy(axon, policy, filter)
  end

  @doc false
  def apply_policy(%Axon{} = axon, %Policy{} = policy) do
    apply_policy(%Axon{} = axon, %Policy{} = policy, & &1)
  end
end
