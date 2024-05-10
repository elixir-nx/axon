defmodule Axon.ModelState do
  @moduledoc """
  Model State Data Structure.

  This data structure represents all the state needed for
  a model to perform inference.
  """
  @derive {
    Nx.Container,
    keep: [:parameters, :state, :frozen_parameters], containers: [:data]
  }
  defstruct [:data, :parameters, :state, :frozen_parameters]

  alias __MODULE__

  @doc """
  Updates the given model state.
  """
  def update(
        %ModelState{
          state: state,
          frozen_parameters: frozen
        } = model_state,
        updated_parameters,
        updated_state \\ %{}
      ) do
    updated_state =
      state
      |> tree_diff(frozen)
      |> then(&tree_get(updated_state, &1))

    update_in(model_state, [Access.key!(:data)], fn data ->
      data
      |> tree_merge(updated_parameters, fn _, _, v -> v end)
      |> tree_merge(updated_state, fn _, _, v -> v end)
    end)
  end

  # TODO: Mask syntax with strings?

  @doc """
  Freezes parameters and state in the given model state
  using the given mask.

  The mask is an arity 1 function which takes the access path to the
  leaf parameter and returns `true` if the parameter should be frozen
  or `false` otherwise. With this, you can construct flexible masking
  policies:

      fn
        ["dense_" <> n, "kernel"] -> String.to_integer(n) < 3
        _ -> false
      end

  The default mask returns `true` for all paths, and is equivalent to
  freezing the entire model.
  """
  def freeze(%ModelState{data: data} = model_state, mask \\ fn _ -> true end) do
    frozen_paths =
      data
      |> get_paths()
      |> Enum.filter(mask)

    case frozen_paths do
      [] ->
        model_state

      [_ | _] = paths ->
        frozen =
          Enum.reduce(paths, %{}, fn path, acc ->
            [root | rest] = Enum.reverse(path)
            nested_put(acc, Enum.reverse(rest), root)
          end)

        %{model_state | frozen_parameters: frozen}
    end
  end

  @doc """
  Unfreezes parameters and state in the given model state
  using the given mask.

  The mask is an arity 1 function which takes the access path to the
  leaf parameter and returns `true` if the parameter should be unfrozen
  or `false` otherwise. With this, you can construct flexible masking
  policies:

      fn
        ["dense_" <> n, "kernel"] -> n < 3
        _ -> false
      end

  The default mask returns `true` for all paths, and is equivalent to
  unfreezing the entire model.
  """
  def unfreeze(
        %ModelState{data: data, frozen_parameters: frozen} = model_state,
        mask \\ fn _ -> true end
      ) do
    unfrozen_paths =
      data
      |> get_paths()
      |> Enum.filter(mask)

    case unfrozen_paths do
      [] ->
        model_state

      [_ | _] = paths ->
        unfrozen =
          Enum.reduce(paths, %{}, fn path, acc ->
            [root | rest] = Enum.reverse(path)
            nested_put(acc, Enum.reverse(rest), root)
          end)

        %{model_state | frozen_parameters: tree_diff(frozen, unfrozen)}
    end
  end

  @doc """
  Returns the trainable parameters in the given model state.
  """
  def trainable_parameters(%ModelState{
        data: data,
        parameters: parameters,
        frozen_parameters: frozen
      }) do
    parameters
    |> tree_diff(frozen)
    |> then(&tree_get(data, &1))
  end

  @doc """
  Returns the frozen parameters in the given model state.
  """
  def frozen_parameters(%ModelState{data: data, state: state, frozen_parameters: frozen}) do
    frozen
    |> tree_diff(state)
    |> then(&tree_get(data, &1))
  end

  @doc """
  Returns the trainable state in the given model state.
  """
  def trainable_state(%ModelState{data: data, state: state, frozen_parameters: frozen}) do
    state
    |> tree_diff(frozen)
    |> then(&tree_get(data, &1))
  end

  @doc """
  Returns the frozen state in the given model state.
  """
  def frozen_state(%ModelState{data: data, parameters: parameters, frozen_parameters: frozen}) do
    frozen
    |> tree_diff(parameters)
    |> then(&tree_get(data, &1))
  end

  @doc """
  Returns an empty model state.
  """
  def empty() do
    %Axon.ModelState{
      data: %{},
      parameters: %{},
      state: %{},
      frozen_parameters: %{}
    }
  end

  @doc """
  Returns a new model state struct from the given parameter
  map.
  """
  def new(data)

  def new(%Axon.ModelState{} = model_state), do: model_state

  def new(data) when is_map(data) do
    %Axon.ModelState{
      data: data,
      parameters: get_paths(data),
      state: %{},
      frozen_parameters: %{}
    }
  end

  # Helpers

  defp get_paths(map) do
    Enum.flat_map(map, fn {key, value} ->
      traverse(value, [key])
    end)
  end

  defp traverse(%Nx.Tensor{}, acc), do: [Enum.reverse(acc)]

  defp traverse(map, acc) do
    Enum.flat_map(map, fn {k, value} ->
      traverse(value, [k | acc])
    end)
  end

  defp nested_put(acc, [key], value) do
    Map.update(acc, key, [value], fn values -> [value | values] end)
  end

  defp nested_put(acc, [key | rest], value) do
    inner = nested_put(%{}, rest, value)
    Map.update(acc, key, inner, &nested_put(&1, rest, value))
  end

  defp tree_get(data, access) when is_list(access) do
    Enum.reduce(access, %{}, &Map.put(&2, &1, Map.fetch!(data, &1)))
  end

  defp tree_get(data, access) when is_map(access) do
    Enum.reduce(access, %{}, fn {key, value}, acc ->
      tree = tree_get(data[key], value)
      Map.put(acc, key, tree)
    end)
  end

  defp tree_diff(lhs, rhs) do
    Enum.reduce(lhs, %{}, fn {key, val_lhs}, acc ->
      case Map.get(rhs, key) do
        nil ->
          Map.put(acc, key, val_lhs)

        val_rhs when is_map(val_lhs) and is_map(val_rhs) ->
          updated_val = tree_diff(val_lhs, val_rhs)
          if Map.keys(updated_val) == [], do: acc, else: Map.put(acc, key, updated_val)

        val_rhs ->
          new_val = val_lhs -- val_rhs
          if new_val == [], do: acc, else: Map.put(acc, key, new_val)
      end
    end)
  end

  defp tree_merge(lhs, rhs, fun) do
    Enum.reduce(lhs, %{}, fn {key, val_lhs}, acc ->
      case Map.get(rhs, key) do
        nil ->
          Map.put(acc, key, val_lhs)

        %Nx.Tensor{} = val_rhs ->
          new_val = fun.(key, val_lhs, val_rhs)
          Map.put(acc, key, new_val)

        val_rhs when is_map(val_lhs) and is_map(val_rhs) ->
          updated_val = tree_merge(val_lhs, val_rhs, fun)
          Map.put(acc, key, updated_val)

        val_rhs ->
          new_val = fun.(key, val_lhs, val_rhs)
          Map.put(acc, key, new_val)
      end
    end)
  end
end
