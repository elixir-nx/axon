defmodule Axon.Compiler do
  @moduledoc false

  ## Init JIT Compilation

  # TODO: This is a pretty fragile way of enforcing parameter ordering.
  # Need a more coherent strategy

  @doc false
  def __jit_init__(%Axon{} = graph, caller, [] = args, opts) do
    {names_and_exprs, _} = to_init_fun(graph, {%{}, 0})

    fun = fn ->
      names_and_exprs
      |> Map.values()
      |> Enum.reverse()
      |> Enum.map(& &1.())
      |> List.to_tuple()
    end

    if Nx.Defn.Compiler.current() do
      if opts != [] do
        raise ArgumentError,
              "cannot pass execution options to Axon.#{caller} inside defn, got: #{inspect(opts)}"
      end

      fun.()
    else
      Nx.Defn.jit(fun, args, opts)
    end
  end

  defp to_init_fun(%Axon{parent: parents}, acc) when is_list(parents) do
    Enum.reduce(parents, acc, fn parent, acc -> to_init_fun(parent, acc) end)
  end

  defp to_init_fun(%Axon{parent: parent, params: params}, acc) do
    acc =
      Enum.reduce(params, acc, fn
        %Axon.Parameter{} = param, {names_and_exprs, counter} ->
          %{name: name, shape: shape, initializer: initializer} = param
          key = "#{counter_to_name(counter)}_" <> name
          fun = fn -> apply(Axon.Initializers, initializer, [[shape: shape]]) end
          {Map.put(names_and_exprs, key, fun), counter + 1}
      end)

    if parent do
      to_init_fun(parent, acc)
    else
      acc
    end
  end

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(%Axon{} = graph, caller, [params, input] = args, opts) do
    {expr, _} = to_predict_expr(graph, Enum.reverse(Tuple.to_list(params)), input)

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
                       [:leaky_relu, :linear, :log_sigmoid, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  defp to_predict_expr(%Axon{op: op, parent: parent}, params, input)
       when op in @activation_layers do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Activations, op, [expr]), params}
  end

  ## Linear Layers

  defp to_predict_expr(%Axon{op: :dense, parent: parent}, [b, w | params], input) do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Layers, :dense, [expr, w, b]), params}
  end

  ## Pooling Layers

  @pooling_layers [:max_pool, :avg_pool, :lp_pool, :adaptive_avg_pool, :adaptive_max_pool]

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, params, input)
       when op in @pooling_layers do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Layers, op, [expr, opts]), params}
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, params, input)
       when op in @dropout_layers do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Layers, op, [expr, opts]), params}
  end

  ## Conv Layers

  @conv_layers [:conv, :depthwise_conv]

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, [b, w | params], input)
       when op in @conv_layers do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Layers, op, [expr, w, b, opts]), params}
  end

  defp to_predict_expr(
         %Axon{op: :separable_conv2d, parent: parent, opts: opts},
         [b1, w1, b2, w2 | params],
         input
       ) do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Layers, :separable_conv2d, [expr, w1, b1, w2, b2, opts]), params}
  end

  defp to_predict_expr(
         %Axon{op: :separable_conv3d, parent: parent, opts: opts},
         [b1, w1, b2, w2, b3, w3 | params],
         input
       ) do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Layers, :separable_conv3d, [expr, w1, b1, w2, b2, w3, b3, opts]), params}
  end

  ## Normalization Layers

  @normalization_layers [:batch_norm, :layer_norm, :group_norm, :instance_norm]

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, [b, w | params], input)
       when op in @normalization_layers do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Layers, op, [expr, w, b, opts]), params}
  end

  defp to_predict_expr(
         %Axon{op: :concatenate, parent: parents, opts: [axis: axis]},
         params,
         input
       ) do
    {exprs, params} =
      Enum.map_reduce(parents, params, fn node, params ->
        to_predict_expr(node, params, input)
      end)

    {apply(Nx, :concatenate, [exprs, [axis: axis]]), params}
  end

  ## Element-wise layers

  @element_wise_layers [:add, :subtract, :multiply]

  defp to_predict_expr(%Axon{op: op, parent: parents}, params, input)
       when op in @element_wise_layers do
    {[expr | rest], _} =
      Enum.map_reduce(parents, params, fn node, params ->
        to_predict_expr(node, params, input)
      end)

    res =
      rest
      |> Enum.reduce(expr, fn x, acc -> apply(Nx, op, [acc, x]) end)

    {res, params}
  end

  defp to_predict_expr(%Axon{op: :nx, parent: parent, opts: [fun: fun]}, params, input) do
    {expr, params} = to_predict_expr(parent, params, input)
    {fun.(expr), params}
  end

  defp to_predict_expr(%Axon{op: :flatten, parent: parent}, params, input) do
    {expr, params} = to_predict_expr(parent, params, input)
    {apply(Axon.Layers, :flatten, [expr]), params}
  end

  defp to_predict_expr(%Axon{op: :input, parent: nil}, params, input) do
    {input, params}
  end

  ## Helpers

  defp counter_to_name(counter) when counter >= 26 do
    [counter_to_name(div(counter, 26)) | counter_to_name(rem(counter, 26))]
  end

  defp counter_to_name(counter), do: [Enum.at(?a..?z, counter)]
end
