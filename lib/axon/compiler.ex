defmodule Axon.CompilerError do
  defexception [:exception, :graph]

  @impl true
  def message(%{graph: %Axon{op: op}, exception: exception}) do
    op_inspect =
      if is_atom(op) do
        Atom.to_string(op)
      else
        "#{inspect(op)}"
      end

    """
    error while building prediction for #{op_inspect}:

    ** (#{inspect(exception.__struct__)}) #{Exception.message(exception)}
    """
  end
end

defmodule Axon.Compiler do
  @moduledoc false
  require Logger

  import Axon.Shared

  ## Init JIT Compilation

  @doc false
  def __compile__(graph, opts) do
    mode = opts[:mode] || :inference
    {compile_init(graph), compile_predict(graph, mode)}
  end

  @doc false
  def __jit_init__(graph, [] = args, opts) do
    fun = compile_init(graph)
    Nx.Defn.jit_or_apply(fun, args, opts)
  end

  defp compile_init(%Axon{} = graph) do
    init_fn = fn ->
      {cache, _} = to_init_fun(graph, {%{}, %{}})

      cache
      |> Enum.reduce(%{}, fn {_, layer}, layers_acc ->
        Map.merge(layer, layers_acc)
      end)
    end

    fn -> Nx.Defn.jit_or_apply(init_fn, []) end
  end

  defp compile_init(graph) do
    raise ArgumentError,
          "attempting to compile initialization function from" <>
            " an unrecognized graph #{inspect(graph)}, if you" <>
            " are attempting to initialize a model with a container" <>
            " output, use `Axon.container`"
  end

  defp to_init_fun(
         %Axon{
           id: id,
           parent: parent,
           op: op,
           name: name_fn,
           params: params,
           policy: %{params: dtype},
           hooks: hooks
         },
         cache_and_counts
       ) do
    {cache, op_counts} =
      case {op, parent} do
        {:container, [parent]} ->
          deep_reduce(parent, cache_and_counts, &to_init_fun/2)

        {_, nil} ->
          cache_and_counts

        {_, parents} when is_list(parents) ->
          Enum.reduce(parents, cache_and_counts, &to_init_fun/2)

        {_, parents} when is_tuple(parents) ->
          deep_reduce(parents, cache_and_counts, &to_init_fun/2)
      end

    case cache do
      %{^id => _} ->
        {cache, op_counts}

      %{} ->
        if Enum.empty?(params) do
          {cache, op_counts}
        else
          layer_params =
            Enum.reduce(params, %{}, fn {key, param}, layer_params ->
              init_param(key, param, layer_params, dtype)
            end)

          layer_params = apply_hooks(layer_params, :initialize, nil, hooks)

          name = name_fn.(op, op_counts)
          params = %{name => layer_params}

          {
            Map.put(cache, id, params),
            Map.update(op_counts, op, 1, fn x -> x + 1 end)
          }
        end
    end
  end

  defp init_param(key, param, layer_params, dtype) do
    case param do
      %{name: name, shape: shape, initializer: initializer} ->
        fun = apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]])
        Map.put(layer_params, name, fun)

      params when is_tuple(params) ->
        params
        |> Tuple.to_list()
        |> Enum.map(fn %{shape: shape, initializer: initializer} ->
          apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]])
        end)
        |> List.to_tuple()
        |> then(&Map.put(layer_params, key, &1))
    end
  end

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(graph, args, opts) do
    {mode, opts} = Keyword.pop(opts, :mode, :inference)
    fun = compile_predict(graph, mode)
    Nx.Defn.jit_or_apply(fun, args, opts)
  end

  defp compile_predict(%Axon{} = graph, mode) do
    {root_id, {cache, _op_counts}} = to_predict_fun(graph, {%{}, %{}}, mode)

    predict_fn = fn params, inputs ->
      try do
        case mode do
          :train ->
            {pred_expr, state_expr} = cache[root_id].(params, inputs, %{}, cache)
            %{prediction: pred_expr, state: state_expr}

          :inference ->
            {pred_expr, _} = cache[root_id].(params, inputs, %{}, cache)
            pred_expr
        end
      rescue
        e -> reraise Axon.CompilerError.exception(graph: graph, exception: e), __STACKTRACE__
      end
    end

    &Nx.Defn.jit_or_apply(predict_fn, [&1, &2])
  end

  defp compile_predict(graph, _mode) do
    raise ArgumentError,
          "attempting to compile predict function from" <>
            " an unrecognized graph #{inspect(graph)}, if you" <>
            " are attempting to initialize a model with a container" <>
            " output, use `Axon.container`"
  end

  defp to_predict_fun(%{id: id} = graph, {cache, op_counts}, mode) do
    case cache do
      %{^id => _} ->
        {id, {cache, op_counts}}

      %{} ->
        try do
          recur_predict_fun(graph, {cache, op_counts}, mode)
        rescue
          e -> reraise Axon.CompilerError.exception(graph: graph, exception: e), __STACKTRACE__
        end
    end
  end

  defp call_cache(parent_id, params, inputs, state, cache) do
    key = {:cache, parent_id}

    case state do
      %{^key => expr} ->
        {expr, state}

      %{} ->
        {expr, state} = cache[parent_id].(params, inputs, state, cache)
        {expr, Map.put(state, key, expr)}
    end
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :container, parent: [parents]},
         cache_and_counts,
         mode
       ) do
    {parent_ids, {cache, op_counts}} =
      deep_map_reduce(parents, cache_and_counts, &to_predict_fun(&1, &2, mode))

    op_counts = Map.update(op_counts, :container, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache ->
      deep_map_reduce(parent_ids, state, fn parent_id, state ->
        call_cache(parent_id, params, inputs, state, cache)
      end)
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Custom Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: parents,
           params: layer_params,
           opts: opts,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when is_function(op) and is_list(parents) do
    {parent_ids, {cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, mode)
      )

    {_, opts} = Keyword.pop(opts, :layer_op)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    layer_params =
      Enum.map(layer_params, fn {k, %{name: v, frozen: frz}} ->
        {k, {v, frz}}
      end)

    fun = fn params, inputs, state, cache ->
      {res, state} =
        parent_ids
        |> Enum.map_reduce(state, fn parent_id, state ->
          call_cache(parent_id, params, inputs, state, cache)
        end)

      inp_params =
        Map.new(layer_params, fn {k, {v, frz}} ->
          {k, maybe_freeze(params[name][v], frz)}
        end)

      inputs =
        res
        |> Enum.map(&safe_as_type(&1, compute))
        |> Enum.map(&apply_hooks(&1, :pre_forward, mode, hooks))

      args =
        case opts do
          [] ->
            inputs ++ [inp_params]

          [_ | _] ->
            inputs ++ [inp_params, opts]
        end

      out =
        args
        |> then(&apply(op, &1))
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)
        |> safe_as_type(output)

      {out, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Activation Layers

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :mish, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh] ++
                       [:log_softmax]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: [parent],
           policy: %{compute: compute, output: output},
           opts: opts,
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when op in @activation_layers do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache ->
      {res, state} =
        params
        |> cache[parent_id].(inputs, state, cache)

      res =
        res
        |> safe_as_type(compute)
        |> apply_hooks(:pre_forward, mode, hooks)

      args =
        case opts do
          [] ->
            [res]

          [_ | _] ->
            [res, opts]
        end

      res =
        args
        |> then(&apply(Axon.Activations, op, &1))
        |> safe_as_type(output)
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Linear Layers

  @linear_layers [:dense, :bilinear, :conv, :depthwise_conv, :conv_transpose]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           name: name_fn,
           parent: parent,
           params: layer_params,
           policy: %{compute: compute, output: output},
           opts: opts,
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when op in @linear_layers do
    {parent_ids, {cache, op_counts}} =
      Enum.map_reduce(
        parent,
        cache_and_counts,
        &to_predict_fun(&1, &2, mode)
      )

    layer_name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    {use_bias, opts} = Keyword.pop(opts, :use_bias, true)

    %{frozen: w_frz} = layer_params["kernel"]
    %{frozen: b_frz} = if use_bias, do: layer_params["bias"], else: %{frozen: false}

    fun = fn params, inputs, state, cache ->
      {res, state} =
        parent_ids
        |> Enum.map_reduce(state, fn parent_id, state ->
          call_cache(parent_id, params, inputs, state, cache)
        end)

      inputs =
        res
        |> Enum.map(&safe_as_type(&1, compute))
        |> Enum.map(&apply_hooks(&1, :pre_forward, mode, hooks))

      w = get_param(params, layer_name, "kernel", w_frz, compute)

      b =
        if use_bias do
          get_param(params, layer_name, "bias", b_frz, compute)
        else
          Nx.tensor(0.0, type: compute)
        end

      args =
        case opts do
          [] ->
            inputs ++ [w, b]

          [_ | _] ->
            inputs ++ [w, b, opts]
        end

      res =
        args
        |> then(&apply(Axon.Layers, op, &1))
        |> safe_as_type(output)
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: :bias,
           parent: parent,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       ) do
    {parent_ids, {cache, op_counts}} =
      Enum.map_reduce(
        parent,
        cache_and_counts,
        &to_predict_fun(&1, &2, mode)
      )

    layer_name = name_fn.(:bias, op_counts)
    op_counts = Map.update(op_counts, :bias, 1, fn x -> x + 1 end)

    %{frozen: b_frz} = layer_params["bias"]

    fun = fn params, inputs, state, cache ->
      {res, state} =
        parent_ids
        |> Enum.map_reduce(state, fn parent_id, state ->
          call_cache(parent_id, params, inputs, state, cache)
        end)

      b = get_param(params, layer_name, "bias", b_frz, compute)

      inputs =
        res
        |> Enum.map(&safe_as_type(&1, compute))
        |> Enum.map(&apply_hooks(&1, :pre_forward, mode, hooks))

      args = inputs ++ [b]

      res =
        args
        |> then(&apply(Axon.Layers, :bias, &1))
        |> safe_as_type(output)
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Sparse Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: :embedding,
           parent: [parent],
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       ) do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    layer_name = name_fn.(:embedding, op_counts)
    op_counts = Map.update(op_counts, :embedding, 1, fn x -> x + 1 end)

    %{frozen: w_frz} = layer_params["kernel"]

    fun = fn params, inputs, state, cache ->
      {res, state} = call_cache(parent_id, params, inputs, state, cache)

      w = get_param(params, layer_name, "kernel", w_frz, compute)

      res =
        res
        |> apply_hooks(:pre_forward, :inference, hooks)
        |> safe_as_type({:s, 64})
        |> Axon.Layers.embedding(w)
        |> safe_as_type(output)
        |> apply_hooks(:forward, :inference, hooks)
        |> apply_hooks(:backward, :inference, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Pooling Layers

  @pooling_layers [:max_pool, :avg_pool, :adaptive_avg_pool] ++
                    [:adaptive_max_pool, :adaptive_lp_pool, :lp_pool] ++
                    [:global_lp_pool, :global_max_pool, :global_avg_pool]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: [parent],
           opts: opts,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when op in @pooling_layers do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache ->
      {res, state} = call_cache(parent_id, params, inputs, state, cache)

      res =
        res
        |> safe_as_type(compute)
        |> apply_hooks(:pre_forward, :inference, hooks)
        |> then(&apply(Axon.Layers, op, [&1, opts]))
        |> safe_as_type(output)
        |> apply_hooks(:forward, :inference, hooks)
        |> apply_hooks(:backward, :inference, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: [parent],
           opts: opts,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when op in @dropout_layers do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache ->
      {inputs, state} = call_cache(parent_id, params, inputs, state, cache)

      res =
        case mode do
          :train ->
            inputs
            |> safe_as_type(compute)
            |> apply_hooks(:pre_forward, :train, hooks)
            |> then(&apply(Axon.Layers, op, [&1, opts]))
            |> safe_as_type(output)
            |> apply_hooks(:forward, :train, hooks)
            |> apply_hooks(:backward, :train, hooks)

          :inference ->
            # Skip dropout in inference mode
            safe_as_type(inputs, output)
        end

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: :separable_conv2d,
           parent: [parent],
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       ) do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    name = name_fn.(:separable_conv2d, op_counts)
    op_counts = Map.update(op_counts, :separable_conv2d, 1, fn x -> x + 1 end)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)
    %{frozen: k1_frz} = layer_params["k1"]
    %{frozen: k2_frz} = layer_params["k2"]
    %{frozen: b1_frz} = if use_bias, do: layer_params["b1"], else: %{frozen: false}
    %{frozen: b2_frz} = if use_bias, do: layer_params["b2"], else: %{frozen: false}

    fun = fn params, inputs, state, cache ->
      {inputs, state} = call_cache(parent_id, params, inputs, state, cache)

      k1 = get_param(params, name, "kernel_1", k1_frz, compute)
      k2 = get_param(params, name, "kernel_2", k2_frz, compute)

      {b1, b2} =
        if use_bias do
          {get_param(params, name, "bias_1", b1_frz, compute),
           get_param(params, name, "bias_2", b2_frz, compute)}
        else
          {Nx.tensor(0, type: compute), Nx.tensor(0, type: compute)}
        end

      res =
        inputs
        |> safe_as_type(compute)
        |> apply_hooks(:pre_forward, mode, hooks)
        |> Axon.Layers.separable_conv2d(k1, b1, k2, b2, opts)
        |> safe_as_type(output)
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: :separable_conv3d,
           parent: [parent],
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       ) do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    name = name_fn.(:separable_conv3d, op_counts)
    op_counts = Map.update(op_counts, :separable_conv3d, 1, fn x -> x + 1 end)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    %{frozen: k1_frz} = layer_params["k1"]
    %{frozen: k2_frz} = layer_params["k2"]
    %{frozen: k3_frz} = layer_params["k3"]
    %{frozen: b1_frz} = if use_bias, do: layer_params["b1"], else: %{frozen: false}
    %{frozen: b2_frz} = if use_bias, do: layer_params["b2"], else: %{frozen: false}
    %{frozen: b3_frz} = if use_bias, do: layer_params["b3"], else: %{frozen: false}

    fun = fn params, inputs, state, cache ->
      {inputs, state} = call_cache(parent_id, params, inputs, state, cache)

      k1 = get_param(params, name, "kernel_1", k1_frz, compute)
      k2 = get_param(params, name, "kernel_2", k2_frz, compute)
      k3 = get_param(params, name, "kernel_3", k3_frz, compute)

      {b1, b2, b3} =
        if use_bias do
          {get_param(params, name, "bias_1", b1_frz, compute),
           get_param(params, name, "bias_2", b2_frz, compute),
           get_param(params, name, "bias_3", b3_frz, compute)}
        else
          {Nx.tensor(0, type: compute), Nx.tensor(0, type: compute), Nx.tensor(0, type: compute)}
        end

      res =
        inputs
        |> safe_as_type(compute)
        |> apply_hooks(:pre_forward, mode, hooks)
        |> Axon.Layers.separable_conv3d(k1, b1, k2, b2, k3, b3, opts)
        |> safe_as_type(output)
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Normalization Layers

  @normalization_with_stats [:batch_norm, :instance_norm]

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: [parent],
           opts: [epsilon: epsilon, channel_index: channel_index, momentum: momentum],
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when op in @normalization_with_stats do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    training? = mode == :train

    norm_opts = [
      epsilon: epsilon,
      channel_index: channel_index,
      momentum: momentum,
      training?: training?
    ]

    %{frozen: g_frz} = layer_params["gamma"]
    %{frozen: b_frz} = layer_params["beta"]
    %{frozen: mean_frz} = layer_params["mean"]
    %{frozen: var_frz} = layer_params["var"]

    fun = fn params, inputs, state, cache ->
      {inputs, state} = call_cache(parent_id, params, inputs, state, cache)

      g = get_param(params, name, "gamma", g_frz, compute)
      b = get_param(params, name, "beta", b_frz, compute)
      mean = get_param(params, name, "mean", mean_frz, compute)
      var = get_param(params, name, "var", var_frz, compute)

      case mode do
        :train ->
          {out, ra_mean, ra_var} =
            inputs
            |> safe_as_type(compute)
            |> apply_hooks(:pre_forward, :train, hooks)
            |> then(&apply(Axon.Layers, op, [&1, g, b, mean, var, norm_opts]))
            |> then(fn {y, m, v} -> {safe_as_type(y, output), m, v} end)
            |> apply_hooks(:forward, :train, hooks)
            |> apply_hooks(:backward, :train, hooks)

          res = safe_as_type(out, output)
          state = Map.put(state, name, %{"mean" => ra_mean, "var" => ra_var})

          {res, state}

        :inference ->
          res =
            inputs
            |> safe_as_type(compute)
            |> apply_hooks(:pre_forward, :inference, hooks)
            |> then(&apply(Axon.Layers, op, [&1, g, b, mean, var, norm_opts]))
            |> safe_as_type(output)
            |> apply_hooks(:forward, :inference, hooks)
            |> apply_hooks(:backward, :inference, hooks)

          {res, state}
      end
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  @normalization_layers [:layer_norm, :group_norm]

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: [parent],
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when op in @normalization_layers do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    %{frozen: g_frz} = layer_params["gamma"]
    %{frozen: b_frz} = layer_params["beta"]

    fun = fn params, inputs, state, cache ->
      {inputs, state} = call_cache(parent_id, params, inputs, state, cache)

      g = get_param(params, name, "gamma", g_frz, compute)
      b = get_param(params, name, "beta", b_frz, compute)

      res =
        inputs
        |> safe_as_type(compute)
        |> apply_hooks(:pre_forward, mode, hooks)
        |> then(&apply(Axon.Layers, op, [&1, g, b, opts]))
        |> safe_as_type(output)
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Recurrent Layers

  @recurrent_layers [:gru, :lstm, :conv_lstm]

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: parents,
           params: layer_params,
           policy: %{compute: compute, output: output},
           opts: opts,
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when op in @recurrent_layers do
    {[input_id, hidden_state_id], {cache, op_counts}} =
      Enum.map_reduce(parents, cache_and_counts, &to_predict_fun(&1, &2, mode))

    num_bias = if op == :conv_lstm, do: 1, else: 4

    {activation, opts} = Keyword.pop(opts, :activation)
    {gate, opts} = Keyword.pop(opts, :gate)
    {use_bias, opts} = Keyword.pop(opts, :use_bias)
    {unroll, conv_opts} = Keyword.pop(opts, :unroll)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    input_kernel = layer_params["input_kernel"]
    hidden_kernel = layer_params["hidden_kernel"]

    bias =
      if use_bias,
        do: layer_params["bias"],
        else: List.to_tuple(List.duplicate(%{frozen: false}, num_bias))

    input_kernel_frozen =
      input_kernel
      |> Tuple.to_list()
      |> Enum.map(fn %{frozen: frz} -> frz end)
      |> List.to_tuple()

    hidden_kernel_frozen =
      hidden_kernel
      |> Tuple.to_list()
      |> Enum.map(fn %{frozen: frz} -> frz end)
      |> List.to_tuple()

    bias_frozen =
      bias
      |> Tuple.to_list()
      |> Enum.map(fn %{frozen: frz} -> frz end)
      |> List.to_tuple()

    fun = fn params, inputs, state, cache ->
      {input, state} = call_cache(input_id, params, inputs, state, cache)
      {hidden_state, state} = call_cache(hidden_state_id, params, inputs, state, cache)

      input_kernel = get_param(params, name, "input_kernel", input_kernel_frozen, compute)
      hidden_kernel = get_param(params, name, "hidden_kernel", hidden_kernel_frozen, compute)

      bias =
        if use_bias do
          get_param(params, name, "bias", bias_frozen, compute)
        else
          List.duplicate(Nx.tensor(0, type: compute), num_bias)
          |> List.to_tuple()
        end

      input = safe_as_type(input, compute)
      carry = deep_new(hidden_state, &safe_as_type(&1, compute))

      # TODO: Should these be hooked together? Not at all?
      {input, carry} = apply_hooks({input, carry}, :pre_forward, mode, hooks)

      cell_fn = get_cell_fn(op, gate, activation, conv_opts)

      {carry, out} =
        case unroll do
          :static ->
            Axon.Recurrent.static_unroll(
              cell_fn,
              input,
              carry,
              input_kernel,
              hidden_kernel,
              bias
            )

          :dynamic ->
            Axon.Recurrent.dynamic_unroll(
              cell_fn,
              input,
              carry,
              input_kernel,
              hidden_kernel,
              bias
            )
        end

      res = {deep_new(carry, &safe_as_type(&1, output)), safe_as_type(out, output)}
      res = apply_hooks(res, :forward, mode, hooks)
      res = apply_hooks(res, :backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Element-wise layers

  @element_wise_layers [:add, :subtract, :multiply]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: parents,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when op in @element_wise_layers do
    {parent_ids, {cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, mode)
      )

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache ->
      {[expr | exprs], state} =
        Enum.map_reduce(parent_ids, state, fn parent_id, state ->
          call_cache(parent_id, params, inputs, state, cache)
        end)

      [expr | exprs] =
        [expr | exprs]
        |> List.to_tuple()
        |> apply_hooks(:pre_forward, mode, hooks)
        |> Tuple.to_list()

      res =
        Enum.reduce(exprs, expr, fn next_expr, acc ->
          input = safe_as_type(next_expr, compute)
          acc = safe_as_type(acc, compute)
          safe_as_type(apply(Nx, op, [acc, input]), output)
        end)

      res = apply_hooks(res, :forward, mode, hooks)
      res = apply_hooks(res, :backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Shape Layers

  @shape_layers [:resize, :flatten, :reshape, :transpose, :pad]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: [parent],
           policy: %{compute: compute, output: output},
           hooks: hooks,
           opts: opts
         },
         cache_and_counts,
         mode
       )
       when op in @shape_layers do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    opts =
      case op do
        :resize ->
          {shape, opts} = Keyword.pop(opts, :resize_shape)
          Keyword.put(opts, :shape, shape)

        :reshape ->
          {shape, opts} = Keyword.pop(opts, :reshape_shape)
          Keyword.put(opts, :shape, shape)

        _ ->
          opts
      end

    fun = fn params, inputs, state, cache ->
      {res, state} = call_cache(parent_id, params, inputs, state, cache)

      res =
        res
        |> safe_as_type(compute)
        |> apply_hooks(:pre_forward, mode, hooks)
        |> then(&apply(Axon.Layers, op, [&1, opts]))
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)
        |> safe_as_type(output)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :concatenate,
           parent: parents,
           opts: [axis: axis],
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       ) do
    {parent_ids, {cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, mode)
      )

    op_counts = Map.update(op_counts, :concatenate, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache ->
      {exprs, state} =
        Enum.map_reduce(parent_ids, state, fn parent_id, state ->
          call_cache(parent_id, params, inputs, state, cache)
        end)

      inps = Enum.map(exprs, &safe_as_type(&1, compute))

      inps =
        inps
        |> List.to_tuple()
        |> apply_hooks(:pre_forward, mode, hooks)
        |> Tuple.to_list()

      res =
        inps
        |> Nx.concatenate(axis: axis)
        |> safe_as_type(output)
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :cond,
           parent: parents,
           opts: [cond: cond_fn],
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       ) do
    {parent_ids, {cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, mode)
      )

    op_counts = Map.update(op_counts, :cond, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache ->
      {exprs, state} =
        Enum.map_reduce(parent_ids, state, fn parent_id, state ->
          call_cache(parent_id, params, inputs, state, cache)
        end)

      [cond_input_expr, true_expr, false_expr] = exprs

      cond_expr = cond_fn.(cond_input_expr)
      cond_rank = Nx.rank(cond_expr)
      cond_type = Nx.type(cond_expr)

      unless cond_rank == 0 and cond_type == {:u, 8} do
        raise Axon.CompilerError,
              "cond_fn must return a scalar-boolean tensor" <>
                " got result with rank #{inspect(cond_rank)} and" <>
                " type #{inspect(cond_type)}"
      end

      {cond_expr, on_true, on_false} =
        [cond_expr, true_expr, false_expr]
        |> List.to_tuple()
        |> apply_hooks(:pre_forward, mode, hooks)

      res =
        Axon.Layers.cond(
          Nx.all(cond_expr),
          safe_as_type(on_true, compute),
          safe_as_type(on_false, compute)
        )

      res = safe_as_type(res, output)
      res = apply_hooks(res, :forward, mode, hooks)
      res = apply_hooks(res, :backward, mode, hooks)
      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Special Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :nx,
           parent: [parent],
           opts: [fun: nx_fun],
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       ) do
    {parent_id, {cache, op_counts}} = to_predict_fun(parent, cache_and_counts, mode)

    op_counts = Map.update(op_counts, :nx, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache ->
      {res, state} = call_cache(parent_id, params, inputs, state, cache)

      res =
        res
        |> safe_as_type(compute)
        |> apply_hooks(:pre_forward, mode, hooks)
        |> nx_fun.()
        |> safe_as_type(output)
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :constant, opts: [value: tensor], policy: %{output: output}},
         {cache, op_counts},
         _
       ) do
    fun = fn _params, _inputs, state, _cache ->
      out = safe_as_type(tensor, output)
      {out, state}
    end

    op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :input, output_shape: shape, hooks: hooks, name: name_fn},
         {cache, op_counts},
         mode
       ) do
    name = name_fn.(:input, op_counts)
    op_counts = Map.update(op_counts, :input, 1, fn x -> x + 1 end)

    fun = fn _params, inputs, state, _cache ->
      res =
        case inputs do
          %Nx.Tensor{} = inputs ->
            inputs

          %{} = inputs ->
            inputs[name]

          _ ->
            raise ArgumentError,
                  "invalid input given to model, expected input" <>
                    " expected input to be a tensor or a map" <>
                    " corresponding to correct input names"
        end

      unless res do
        raise ArgumentError,
              "unable to find input #{name} for model given to predict," <>
                " you must provide an input tensor for every input" <>
                " specified in the graph"
      end

      unless Axon.Shape.compatible?(Nx.shape(res), shape) do
        raise ArgumentError,
              "invalid input shape given to model, expected input" <>
                " with shape #{inspect(shape)}, but got input with" <>
                " shape #{inspect(Nx.shape(res))}"
      end

      res =
        res
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, state}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Helpers

  defp maybe_freeze(param, true), do: Nx.Defn.Kernel.stop_grad(param)
  defp maybe_freeze(param, false), do: param

  defp apply_hooks(res, event, mode, hooks) do
    hooks
    |> Enum.reverse()
    |> Enum.reduce(res, fn {on_event, on_mode, hook_fn}, expr ->
      event? = on_event == event or on_event == :all
      mode? = on_mode == mode or on_mode == :both or mode == nil

      if event? and mode? do
        if on_event == :backward do
          Nx.Defn.Kernel.custom_grad(expr, fn _ans, g ->
            hooked_g = Nx.Defn.Kernel.hook(g, hook_fn)
            [{expr, hooked_g}]
          end)
        else
          Nx.Defn.Kernel.hook(expr, hook_fn)
        end
      else
        expr
      end
    end)
  end

  defp get_cell_fn(op, gate, activation, conv_opts) do
    case op do
      :lstm ->
        gate_fn = &apply(Axon.Activations, gate, [&1])
        activation_fn = &apply(Axon.Activations, activation, [&1])
        &Axon.Recurrent.lstm_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn)

      :gru ->
        gate_fn = &apply(Axon.Activations, gate, [&1])
        activation_fn = &apply(Axon.Activations, activation, [&1])
        &Axon.Recurrent.gru_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn)

      :conv_lstm ->
        &Axon.Recurrent.conv_lstm_cell(&1, &2, &3, &4, &5, conv_opts)
    end
  end

  defp get_param(params, layer_name, param_name, frozen?, type) do
    case params[layer_name][param_name] do
      tuple when is_tuple(tuple) ->
        tuple
        |> Tuple.to_list()
        |> Enum.zip_with(Tuple.to_list(frozen?), &maybe_freeze/2)
        |> List.to_tuple()
        |> safe_as_type(type)

      param ->
        param
        |> maybe_freeze(frozen?)
        |> safe_as_type(type)
    end
  end

  defp safe_as_type(container_or_tensor, type) do
    case container_or_tensor do
      %Nx.Tensor{} = tensor ->
        Nx.as_type(tensor, type)

      container ->
        deep_new(container, &Nx.as_type(&1, type))
    end
  end
end
