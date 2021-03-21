defmodule Axon.Compiler do
  @moduledoc false

  ## Init JIT Compilation

  # TODO: This is a pretty fragile way of enforcing parameter ordering.
  # Need a more coherent strategy

  @doc false
  def __jit_init__(%Axon{} = graph, caller, [] = args, opts) do
    {names_and_exprs, _} = to_init_expr(graph, %{}, 0)

    param_expr =
      names_and_exprs
      |> Map.values()
      |> Enum.reverse()
      |> List.to_tuple()

    if Nx.Defn.Compiler.current() do
      if opts != [] do
        raise ArgumentError,
              "cannot pass execution options to Axon.#{caller} inside defn, got: #{inspect(opts)}"
      end

      param_expr
    else
      Nx.Defn.jit(param_expr, args, opts)
    end
  end

  defp to_init_expr(%Axon{parent: nil, params: params}, names_and_exprs, counter) do
    Enum.reduce(params, {names_and_exprs, counter}, fn %Axon.Parameter{
                                                         name: name,
                                                         shape: shape,
                                                         initializer: initializer
                                                       },
                                                       {names_and_exprs, counter} ->
      {
        Map.put(
          names_and_exprs,
          "#{counter_to_name(counter)}_" <> name,
          apply(Axon.Initializers, initializer, [[shape: shape]])
        ),
        counter + 1
      }
    end)
  end

  defp to_init_expr(%Axon{parent: parent, params: params}, names_and_exprs, counter) do
    {names_and_exprs, counter} =
      Enum.reduce(params, {names_and_exprs, counter}, fn %Axon.Parameter{
                                                           name: name,
                                                           shape: shape,
                                                           initializer: initializer
                                                         },
                                                         {names_and_exprs, counter} ->
        {
          Map.put(
            names_and_exprs,
            "#{counter_to_name(counter)}_" <> name,
            apply(Axon.Initializers, initializer, [[shape: shape]])
          ),
          counter + 1
        }
      end)

    to_init_expr(parent, names_and_exprs, counter)
  end

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(%Axon{} = graph, caller, [params, input] = args, opts) do
    expr = to_predict_expr(graph, Enum.reverse(Tuple.to_list(params)), input)

    if Nx.Defn.Compiler.current() do
      if opts != [] do
        raise ArgumentError,
              "cannot pass execution options to Axon.#{caller} inside defn, got: #{inspect(opts)}"
      end

      expr
    else
      Nx.Defn.jit(expr, args, opts)
    end
  end

  ## Activation Layers

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :log_softmax, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  defp to_predict_expr(%Axon{op: op, parent: parent}, params, input)
       when op in @activation_layers do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Activations, op, [expr])
  end

  ## Linear Layers

  defp to_predict_expr(%Axon{op: :dense, parent: parent}, [b, w | params], input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :dense, [expr, w, b])
  end

  ## Pooling Layers

  @pooling_layers [:max_pool, :avg_pool, :lp_pool, :adaptive_avg_pool, :adaptive_max_pool]

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, params, input)
       when op in @pooling_layers do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, op, [expr, opts])
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, params, input)
       when op in @dropout_layers do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, op, [expr, opts])
  end

  ## Conv Layers

  @conv_layers [:conv, :depthwise_conv]

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, [b, w | params], input)
       when op in @conv_layers do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, op, [expr, w, b, opts])
  end

  defp to_predict_expr(%Axon{op: :separable_conv2d, parent: parent, opts: opts}, [b1, w1, b2, w2 | params], input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :separable_conv2d, [expr, w1, b1, w2, b2, opts])
  end

  defp to_predict_expr(%Axon{op: :separable_conv3d, parent: parent, opts: opts}, [b1, w1, b2, w2, b3, w3 | params], input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :separable_conv3d, [expr, w1, b1, w2, b2, w3, b3, opts])
  end

  ## Normalization Layers

  @normalization_layers [:batch_norm, :layer_norm, :group_norm, :instance_norm]

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, [b, w | params], input)
       when op in @normalization_layers do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, op, [expr, w, b, opts])
  end

  defp to_predict_expr(%Axon{op: :nx, parent: parent, opts: [fun: fun]}, params, input) do
    expr = to_predict_expr(parent, params, input)
    fun.(expr)
  end

  defp to_predict_expr(%Axon{op: :flatten, parent: parent}, params, input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :flatten, [expr])
  end

  defp to_predict_expr(%Axon{op: :input, parent: nil}, _params, input) do
    input
  end

  ## Helpers

  defp counter_to_name(counter) when counter >= 26 do
    [counter_to_name(div(counter, 26)) | counter_to_name(rem(counter, 26))]
  end

  defp counter_to_name(counter), do: [Enum.at(?a..?z, counter)]
end
