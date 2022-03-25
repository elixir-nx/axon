defmodule Axon.Parametrization do
  @moduledoc """
  Layer parametrization similar to PyTorch.

  Parametrization alters parameters in an Axon model according
  to the parametrization function prior to the forward pass of
  the model. Parametrizations are themselves Axon models, and
  thus can have parameters and state which will be tracked
  during the backward pass for use in model optimization.
  """

  @doc """
  Parametrizes the given layer and parameter with the given
  parametrization model.

  Name corresponds to the name of the parameter contained in
  the current layer.
  """
  def parametrize(%Axon{params: params} = axon, name, %Axon{} = parametrize) do
    :ok
  end
end
