defmodule Axon.Compiler do
  @moduledoc false

  import Axon.Shared

  ## Init JIT Compilation

  @doc false
  def __compile__(graph) do
    {compile_init(graph), compile_predict(graph)}
  end

  @doc false
  def __jit_init__(graph, caller, [] = args, opts) do
    fun = compile_init(graph)
    jit_or_apply(caller, fun, args, opts)
  end

  defp compile_init(graph) when is_tuple(graph) do
    fn ->
      graph
      |> Tuple.to_list()
      |> Enum.reduce(%{}, &to_init_fun/2)
      |> Enum.sort()
      |> Enum.unzip()
      |> Kernel.elem(1)
      |> Enum.map(& &1.())
      |> List.to_tuple()
    end
  end

  defp compile_init(%Axon{} = graph) do
    fn ->
      graph
      |> to_init_fun(%{})
      |> Enum.sort()
      |> Enum.unzip()
      |> Kernel.elem(1)
      |> Enum.map(& &1.())
      |> List.to_tuple()
    end
  end

  defp to_init_fun(%Axon{parent: parents}, cache) when is_list(parents) do
    Enum.reduce(parents, cache, &to_init_fun/2)
  end

  defp to_init_fun(%Axon{parent: parent, params: params}, cache) do
    cache =
      Enum.reduce(params, cache, fn
        %{id: id} = param, cache ->
          case cache do
            %{^id => _} ->
              cache

            %{} ->
              %{id: id, shape: shape, initializer: initializer} = param
              fun = fn -> apply(Axon.Initializers, initializer, [[shape: shape]]) end
              Map.put(cache, id, fun)
          end
      end)

    if parent do
      to_init_fun(parent, cache)
    else
      cache
    end
  end

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(graph, caller, args, opts) do
    fun = compile_predict(graph)
    jit_or_apply(caller, fun, args, opts)
  end

  defp compile_predict(graph) when is_tuple(graph) do
    graph = Tuple.to_list(graph)

    {param_ids, input_ids} =
      graph
      |> Enum.reduce(
        {[], []},
        fn x, {param_ids, input_ids} ->
          get_params_and_inputs(x, param_ids, input_ids)
        end
      )

    param_map =
      param_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    input_map =
      input_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    fn params, inputs ->
      {funs, _} = Enum.map_reduce(graph, %{}, &to_predict_fun(&1, &2, param_map, input_map))

      funs
      |> Enum.reverse()
      |> Enum.map(& &1.(params, inputs))
      |> List.to_tuple()
    end
  end

  defp compile_predict(%Axon{} = graph) do
    {param_ids, input_ids} = get_params_and_inputs(graph, [], [])

    param_map =
      param_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    input_map =
      input_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    fn params, inputs ->
      {fun, _} = to_predict_fun(graph, %{}, param_map, input_map)
      fun.(params, inputs)
    end
  end

  ## Parameter Ordering

  defp get_params_and_inputs(%Axon{id: id, op: :input}, param_ids, input_ids) do
    {param_ids, [id | input_ids]}
  end

  defp get_params_and_inputs(%Axon{parent: parents}, param_ids, input_ids)
       when is_list(parents) do
    Enum.reduce(parents, {param_ids, input_ids}, fn graph, {param_ids, input_ids} ->
      get_params_and_inputs(graph, param_ids, input_ids)
    end)
  end

  defp get_params_and_inputs(%Axon{parent: parent, params: params}, param_ids, input_ids) do
    param_ids = Enum.reduce(params, param_ids, fn %{id: id}, param_ids -> [id | param_ids] end)
    get_params_and_inputs(parent, param_ids, input_ids)
  end

  defp to_predict_fun(%{id: id} = graph, cache, param_map, input_map) do
    case cache do
      %{^id => fun} ->
        {fun, cache}

      %{} ->
        {fun, cache} = recur_predict_fun(graph, cache, param_map, input_map)
        cache = Map.put(cache, id, fun)
        {fun, cache}
    end
  end

  ## Custom Layers
  defp recur_predict_fun(
         %Axon{op: op, parent: parent, params: layer_params, opts: opts},
         cache,
         param_map,
         input_map
       )
       when is_function(op) do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)
    param_indices = Enum.map(layer_params, fn %{id: id} -> param_map[id] end)

    fun = fn params, input ->
      inp_params =
        param_indices
        |> Enum.map(&elem(params, &1))

      apply(op, [fun.(params, input) | inp_params ++ opts])
    end

    {fun, cache}
  end

  ## Activation Layers

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  defp recur_predict_fun(%Axon{op: op, parent: parent}, cache, param_map, input_map)
       when op in @activation_layers do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    fun = fn params, inputs ->
      apply(Axon.Activations, op, [fun.(params, inputs)])
    end

    {fun, cache}
  end

  ## Linear Layers

  defp recur_predict_fun(
         %Axon{op: :dense, parent: parent, params: [%{id: w_id}, %{id: b_id}]},
         cache,
         param_map,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    w_idx = param_map[w_id]
    b_idx = param_map[b_id]

    fun = fn params, inputs ->
      {w, b} = {elem(params, w_idx), elem(params, b_idx)}
      apply(Axon.Layers, :dense, [fun.(params, inputs), w, b])
    end

    {fun, cache}
  end

  ## Pooling Layers

  @pooling_layers [:max_pool, :avg_pool, :lp_pool, :adaptive_avg_pool, :adaptive_max_pool]

  defp recur_predict_fun(%Axon{op: op, parent: parent, opts: opts}, cache, param_map, input_map)
       when op in @pooling_layers do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, op, [fun.(params, inputs), opts])
    end

    {fun, cache}
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp recur_predict_fun(%Axon{op: op, parent: parent, opts: opts}, cache, param_map, input_map)
       when op in @dropout_layers do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, op, [fun.(params, inputs), opts])
    end

    {fun, cache}
  end

  ## Conv Layers

  @conv_layers [:conv, :depthwise_conv]

  defp recur_predict_fun(
         %Axon{op: op, parent: parent, opts: opts, params: [%{id: k_id}, %{id: b_id}]},
         cache,
         param_map,
         input_map
       )
       when op in @conv_layers do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    k_idx = param_map[k_id]
    b_idx = param_map[b_id]

    fun = fn params, inputs ->
      {w, b} = {elem(params, k_idx), elem(params, b_idx)}
      apply(Axon.Layers, op, [fun.(params, inputs), w, b, opts])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :separable_conv2d,
           parent: parent,
           opts: opts,
           params: [%{id: k1_id}, %{id: b1_id}, %{id: k2_id}, %{id: b2_id}]
         },
         cache,
         param_map,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    k1_idx = param_map[k1_id]
    b1_idx = param_map[b1_id]
    k2_idx = param_map[k2_id]
    b2_idx = param_map[b2_id]

    fun = fn params, inputs ->
      {w1, b1, w2, b2} = {
        elem(params, k1_idx),
        elem(params, b1_idx),
        elem(params, k2_idx),
        elem(params, b2_idx)
      }

      apply(Axon.Layers, :separable_conv2d, [fun.(params, inputs), w1, b1, w2, b2, opts])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :separable_conv2d,
           parent: parent,
           opts: opts,
           params: [
             %{id: k1_id},
             %{id: b1_id},
             %{id: k2_id},
             %{id: b2_id},
             %{id: k3_id},
             %{id: b3_id}
           ]
         },
         cache,
         param_map,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    k1_idx = param_map[k1_id]
    b1_idx = param_map[b1_id]
    k2_idx = param_map[k2_id]
    b2_idx = param_map[b2_id]
    k3_idx = param_map[k3_id]
    b3_idx = param_map[b3_id]

    fun = fn params, inputs ->
      {w1, b1, w2, b2, w3, b3} = {
        elem(params, k1_idx),
        elem(params, b1_idx),
        elem(params, k2_idx),
        elem(params, b2_idx),
        elem(params, k3_idx),
        elem(params, b3_idx)
      }

      apply(Axon.Layers, :separable_conv3d, [fun.(params, inputs), w1, b1, w2, b2, w3, b3, opts])
    end

    {fun, cache}
  end

  ## Normalization Layers

  @normalization_layers [:batch_norm, :layer_norm, :group_norm, :instance_norm]

  defp recur_predict_fun(
         %Axon{op: op, parent: parent, opts: opts, params: [%{id: g_id}, %{id: b_id}]},
         cache,
         param_map,
         input_map
       )
       when op in @normalization_layers do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    g_idx = param_map[g_id]
    b_idx = param_map[b_id]

    fun = fn params, inputs ->
      {w, b} = {elem(params, g_idx), elem(params, b_idx)}
      apply(Axon.Layers, op, [fun.(params, inputs), w, b, opts])
    end

    {fun, cache}
  end

  ## Element-wise layers

  @element_wise_layers [:add, :subtract, :multiply]

  defp recur_predict_fun(%Axon{op: op, parent: parents}, cache, param_map, input_map)
       when op in @element_wise_layers do
    {[fun | funs], cache} =
      Enum.map_reduce(parents, cache, &recur_predict_fun(&1, &2, param_map, input_map))

    fun = fn params, inputs ->
      Enum.reduce(funs, fun.(params, inputs), fn next_fn, acc ->
        apply(Nx, op, [acc, next_fn.(params, inputs)])
      end)
    end

    {fun, cache}
  end

  ## Shape Layers

  defp recur_predict_fun(%Axon{op: :flatten, parent: parent}, cache, param_map, input_map) do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, :flatten, [fun.(params, inputs)])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{op: :concatenate, parent: parents, opts: [axis: axis]},
         cache,
         param_map,
         input_map
       ) do
    {funs, cache} = Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, param_map, input_map))

    fun = fn params, inputs ->
      inps = Enum.map(funs, & &1.(params, inputs))
      apply(Nx, :concatenate, [inps, [axis: axis]])
    end

    {fun, cache}
  end

  ## Special Layers

  defp recur_predict_fun(
         %Axon{op: :nx, parent: parent, opts: [fun: nx_fun]},
         cache,
         param_map,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, param_map, input_map)

    fun = fn params, inputs ->
      nx_fun.(fun.(params, inputs))
    end

    {fun, cache}
  end

  defp recur_predict_fun(%Axon{op: :input, id: id}, cache, _, input_map) do
    fun = fn _, inputs ->
      if is_tuple(inputs) do
        idx = input_map[id]
        elem(inputs, idx)
      else
        inputs
      end
    end

    {fun, cache}
  end

  ## Penalty Function Compilation

  @doc false
  def __jit_penalty__(graph, caller, args, opts) do
    fun = compile_penalty(graph)
    jit_or_apply(caller, fun, args, opts)
  end

  defp compile_penalty(graph) when is_tuple(graph) do
    graph = Tuple.to_list(graph)

    {param_ids, _} =
      Enum.reduce(graph, {[], []}, fn x, {param_ids, input_ids} ->
        get_params_and_inputs(x, param_ids, input_ids)
      end)

    param_map =
      param_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    penalties =
      graph
      |> Enum.reduce(
        %{},
        fn x, cache ->
          to_penalty_fun(x, cache, param_map)
        end
      )

    [fun | funs] = Map.values(penalties)

    fn params ->
      funs
      |> Enum.reduce(fun.(params), fn penalty, acc -> Nx.add(penalty.(params), acc) end)
    end
  end

  defp compile_penalty(%Axon{} = graph) do
    {param_ids, _} = get_params_and_inputs(graph, [], [])

    param_map =
      param_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    penalties = to_penalty_fun(graph, %{}, param_map)
    [fun | funs] = Map.values(penalties)

    fn params ->
      funs
      |> Enum.reduce(fun.(params), fn penalty, acc -> Nx.add(penalty.(params), acc) end)
    end
  end

  defp to_penalty_fun(%Axon{parent: parents}, cache, param_map) when is_list(parents) do
    Enum.reduce(parents, cache, fn graph, cache ->
      to_penalty_fun(graph, cache, param_map)
    end)
  end

  defp to_penalty_fun(%Axon{parent: parent, params: params}, cache, param_map) do
    cache =
      params
      |> Enum.reduce(cache, fn param, cache ->
        %{id: id, regularizer: regularizer} = param

        case cache do
          %{^id => _} ->
            cache

          %{} ->
            fun = fn params ->
              case regularizer do
                :none ->
                  Nx.tensor(0.0)

                regularizer when is_atom(regularizer) ->
                  idx = param_map[id]
                  apply(Axon.Regularizers, regularizer, [elem(params, idx)])

                regularizer when is_function(regularizer) ->
                  idx = param_map[id]
                  apply(regularizer, [elem(params, idx)])
              end
            end

            Map.put(cache, id, fun)
        end
      end)

    if parent do
      to_penalty_fun(parent, cache, param_map)
    else
      cache
    end
  end
end
