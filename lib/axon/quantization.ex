defmodule Axon.Quantization do
  @moduledoc """
  Model quantization.

  Model quantization is a technique for reducing the memory footprint of
  a model by converting portions of a model to use quantized representations.
  Typically, these quantized representations are low-precision integers.

  This is an **experimental** API which implements weight-only quantization.
  The implementation in this module will convert dense layers in a large
  model to quantized-variants. The only supported quantization type is
  `{:s, 8}`. Axon quantization is inference-only. Training is not currently
  supported.
  """
  alias Axon.Quantization.Layers
  alias Axon.Quantization.QTensor

  @doc """
  Quantizes a model and a model state.

  Given a model and model state, this method will rewrite all
  of the dense layers in the model to perform weight-only 8-bit
  integer versions of the same operation. It will also replace values
  for all dense kernels in the given model state with quantized
  tensors.
  """
  def quantize(%Axon{} = model, %Axon.ModelState{} = model_state) do
    quantized_model = quantize_model(model)
    quantized_model_state = quantize_model_state(model, model_state)
    {quantized_model, quantized_model_state}
  end

  @doc """
  Replaces standard operations with quantized variants.

  The only supported conversion is to convert regular dense layers
  to a weight-only 8-bit integer variant. Note that this only replaces
  the properties of the model. If you have a pre-trained model state
  that you wish to quantize, refer to `Axon.Quantization.quantize_model_state/2`.

  All `:dense` layers in the model are replaced with `Axon.Quantization.weight_only_quantized_dense/3`.
  """
  def quantize_model(%Axon{} = model) do
    quantized_dense_rewriter = fn [%Axon{} = x], _output, name_fn, units, use_bias ->
      weight_only_quantized_dense(x, units,
        use_bias: use_bias,
        name: name_fn
      )
    end

    Axon.rewrite_nodes(model, fn
      %Axon.Node{op: :dense, meta: meta, name: name_fn} ->
        &quantized_dense_rewriter.(&1, &2, name_fn, meta[:units], meta[:use_bias])

      _ ->
        :skip
    end)
  end

  @doc """
  Returns a quantized model state.

  Given a model and a model state, this function will replace
  all dense layer kernels with a quantized version of the weight.

  Training is not currently supported, so all quantized layers are
  automatically frozen.
  """
  def quantize_model_state(model, model_state) do
    dense_layer_names =
      model
      |> Axon.properties()
      |> Enum.filter(fn {_, v} -> v == :dense end)
      |> Enum.map(fn {k, _} -> k end)
      |> MapSet.new()

    state =
      Enum.reduce(dense_layer_names, model_state, fn layer_name, state ->
        update_in(state, [Access.key!(:data), layer_name, "kernel"], &QTensor.from_tensor/1)
      end)

    Axon.ModelState.freeze(state, fn [name | _] ->
      MapSet.member?(dense_layer_names, name)
    end)
  end

  ## Layers

  @doc """
  Adds a weight-only quantized dense layer to the network.

  This is equivalent to a dense layer, but works on quantized
  weights for reducing model memory footprint.

  Compiles to `Axon.Quantization.Layers.weight_only_quantized_dense/3`.

  ## Options

    * `:name` - layer name.

    * `:kernel_initializer` - initializer for `kernel` weights.
      Defaults to `:glorot_uniform`.

    * `:bias_initializer` - initializer for `bias` weights. Defaults
      to `:zeros`.

    * `:use_bias` - whether the layer should add bias to the output.
      Defaults to `true`.
  """
  def weight_only_quantized_dense(x, units, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :name,
        :meta,
        use_bias: true,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros
      ])

    meta =
      opts[:meta] ||
        %{}
        |> Map.put(:units, units)
        |> Map.put(:use_bias, opts[:use_bias])

    kernel =
      Axon.param("kernel", [{:axis, -1}, units],
        initializer: fn shape, type, key ->
          fun =
            case opts[:kernel_initializer] do
              init when is_atom(init) ->
                apply(Axon.Initializers, init, [])

              fun when is_function(fun) ->
                fun
            end

          tensor =
            case fun do
              fun when is_function(fun, 2) ->
                fun.(shape, type)

              fun when is_function(fun, 3) ->
                fun.(shape, type, key)
            end

          QTensor.from_tensor(tensor)
        end
      )

    {inputs, op} =
      if opts[:use_bias] do
        bias = Axon.param("bias", [units], initializer: opts[:bias_initializer])
        {[x, kernel, bias], &Layers.weight_only_quantized_dense/4}
      else
        {[x, kernel], &Layers.weight_only_quantized_dense/3}
      end

    Axon.layer(op, inputs, name: opts[:name], meta: meta, op_name: :dense)
  end
end
