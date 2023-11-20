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
          data: data,
          parameters: parameters,
          state: state,
          frozen_parameters: frozen
        },
        updated_parameters,
        updated_state
      ) do
    data
    |> Map.merge(updated_parameters)
    # TODO: Should be a deep merge
    |> Map.merge(updated_state, fn k, v1, v2 ->
      if k in frozen do
        v1
      else
        v2
      end
    end)
  end

  def freeze(%ModelState{} = model_state) do
    update_in(model_state, [Access.key!(:frozen_parameters)], fn _frozen ->
      Map.merge(model_state.parameters, model_state.state, fn _, v1, v2 ->
        v1 ++ v2
      end)
    end)
  end

  def unfreeze(%ModelState{} = model_state, layer_name) when is_binary(layer_name) do
    update_in(model_state, [Access.key!(:frozen_parameters)], fn frozen ->
      Map.delete(frozen, layer_name)
    end)
  end

  @doc """
  Merges 2 model states.
  """
  def merge(), do: :ok

  @doc """
  Returns the trainable parameters in the given model state.
  """
  # TODO: Make work with blocks and namespaces
  def trainable_parameters(%ModelState{
        data: data,
        parameters: parameters,
        frozen_parameters: frozen
      }) do
    Enum.reduce(parameters, %{}, fn {block_name, block_parameters}, acc ->
      block_params =
        Enum.reduce(block_parameters, %{}, fn param_name, params ->
          if frozen?(param_name, get_in(frozen, [block_name])) do
            params
          else
            Map.put(params, param_name, get_in(data, [block_name, param_name]))
          end
        end)

      if block_params == %{} do
        acc
      else
        Map.put(acc, block_name, block_params)
      end
    end)
  end

  @doc """
  Returns the frozen parameters in the given model state.
  """
  def frozen_parameters(%ModelState{data: data, parameters: parameters, frozen_parameters: frozen}) do
    Enum.reduce(parameters, %{}, fn {block_name, block_parameters}, acc ->
      block_params =
        Enum.reduce(block_parameters, %{}, fn param_name, params ->
          if frozen?(param_name, get_in(frozen, [block_name])) do
            Map.put(params, param_name, get_in(data, [block_name, param_name]))
          else
            params
          end
        end)

      if block_params == %{} do
        acc
      else
        Map.put(acc, block_name, block_params)
      end
    end)
  end

  @doc """
  Returns the frozen state in the given model state.
  """
  def frozen_state(%ModelState{data: data, state: state, frozen_parameters: frozen}) do
    Enum.reduce(state, %{}, fn {block_name, block_parameters}, acc ->
      block_params =
        Enum.reduce(block_parameters, %{}, fn param_name, params ->
          if frozen?(param_name, get_in(frozen, [block_name])) do
            Map.put(params, param_name, get_in(data, [block_name, param_name]))
          else
            params
          end
        end)

      if block_params == %{} do
        acc
      else
        Map.put(acc, block_name, block_params)
      end
    end)
  end

  @doc """
  Returns the trainable state in the given model state.
  """
  # TODO: Make work with blocks and namespaces
  def trainable_state(%ModelState{data: data, state: state, frozen_parameters: frozen}) do
    Enum.reduce(state, %{}, fn {block_name, block_parameters}, acc ->
      block_params =
        Enum.reduce(block_parameters, %{}, fn param_name, params ->
          if frozen?(param_name, get_in(frozen, [block_name])) do
            params
          else
            Map.put(params, param_name, get_in(data, [block_name, param_name]))
          end
        end)

      if block_params == %{} do
        acc
      else
        Map.put(acc, block_name, block_params)
      end
    end)
  end

  # Helpers

  defp frozen?(_, nil), do: false
  defp frozen?(param_name, params), do: param_name in params
end
