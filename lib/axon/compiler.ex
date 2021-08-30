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
      |> Map.new(fn {k, v} -> {k, v.()} end)
    end
  end

  defp compile_init(%Axon{} = graph) do
    fn ->
      graph
      |> to_init_fun(%{})
      |> Map.new(fn {k, v} -> {k, v.()} end)
    end
  end

  defp to_init_fun(graph, cache) when is_tuple(graph) do
    graph
    |> Tuple.to_list()
    |> Enum.reduce(cache, fn x, acc -> to_init_fun(x, acc) end)
  end

  defp to_init_fun(%Axon{parent: parents, params: params, policy: %{params: dtype}}, cache)
       when is_list(parents) do
    cache =
      Enum.reduce(params, cache, fn {_, %{name: name} = param}, cache ->
        case cache do
          %{^name => _} ->
            cache

          %{} ->
            %{name: name, shape: shape, initializer: initializer} = param
            fun = fn -> apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]]) end
            Map.put(cache, name, fun)
        end
      end)

    Enum.reduce(parents, cache, &to_init_fun/2)
  end

  defp to_init_fun(
         %Axon{parent: parent, params: params, opts: opts, policy: %{params: dtype}},
         cache
       ) do
    cache =
      case opts[:hidden_state] do
        state when is_tuple(state) ->
          state
          |> Tuple.to_list()
          |> Enum.reduce(cache, &to_init_fun/2)

        nil ->
          cache
      end

    cache =
      Enum.reduce(params, cache, fn
        {_, %{name: name} = param}, cache ->
          case cache do
            %{^name => _} ->
              cache

            %{} ->
              %{name: name, shape: shape, initializer: initializer} = param
              fun = fn -> apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]]) end
              Map.put(cache, name, fun)
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

  defp compile_predict(graph) do
    input_ids = get_inputs(graph, [])

    input_map =
      input_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    fn params, inputs ->
      inputs = maybe_flatten(inputs)
      {expr, _} = to_predict_fun(graph, %{}, input_map, params, inputs)

      case expr do
        [_ | _] = exprs ->
          do_recur_to_tuple(exprs, [])

        expr ->
          expr
      end
    end
  end

  defp maybe_flatten(inputs) when is_tuple(inputs) do
    inputs
    |> Tuple.to_list()
    |> do_flatten([])
    |> List.flatten()
    |> List.to_tuple()
  end

  defp maybe_flatten(inputs), do: inputs

  defp do_flatten([], acc), do: Enum.reverse(acc)

  defp do_flatten([inp | []], acc) when is_tuple(inp) do
    res = do_flatten(Tuple.to_list(inp), [])

    [res | acc]
    |> Enum.reverse()
  end

  defp do_flatten([inp | []], acc), do: Enum.reverse([inp | acc])

  defp do_flatten([inp | rest], acc) when is_tuple(inp) do
    res = do_flatten(Tuple.to_list(inp), [])
    do_flatten(rest, [res | acc])
  end

  defp do_flatten([inp | rest], acc) do
    do_flatten(rest, [inp | acc])
  end

  defp do_recur_to_tuple([res | []], acc) when is_list(res) do
    res = do_recur_to_tuple(res, [])

    [res | acc]
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp do_recur_to_tuple([res | []], acc) do
    [res | acc]
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp do_recur_to_tuple([expr | exprs], acc) when is_list(expr) do
    res = do_recur_to_tuple(expr, [])
    do_recur_to_tuple(exprs, [res | acc])
  end

  defp do_recur_to_tuple([expr | exprs], acc) do
    do_recur_to_tuple(exprs, [expr | acc])
  end

  ## Input Ordering

  defp get_inputs(graph, input_ids) when is_tuple(graph) do
    graph
    |> Tuple.to_list()
    |> Enum.reduce(input_ids, fn x, acc -> get_inputs(x, acc) end)
  end

  defp get_inputs(%Axon{op: :constant}, input_ids) do
    input_ids
  end

  defp get_inputs(%Axon{id: id, op: :input}, input_ids) do
    [id | input_ids]
  end

  defp get_inputs(%Axon{parent: parents}, input_ids)
       when is_list(parents) do
    Enum.reduce(parents, input_ids, fn graph, input_ids ->
      get_inputs(graph, input_ids)
    end)
  end

  defp get_inputs(%Axon{parent: parent, opts: opts}, input_ids) do
    input_ids =
      case opts[:hidden_state] do
        state when is_tuple(state) ->
          state
          |> Tuple.to_list()
          |> Enum.reduce(input_ids, fn graph, input_ids ->
            get_inputs(graph, input_ids)
          end)

        nil ->
          input_ids
      end

    get_inputs(parent, input_ids)
  end

  defp to_predict_fun(graph, cache, input_map, params, inputs) when is_tuple(graph) do
    graph
    |> Tuple.to_list()
    |> Enum.map_reduce(cache, fn x, acc -> to_predict_fun(x, acc, input_map, params, inputs) end)
  end

  defp to_predict_fun(%{id: id} = graph, cache, input_map, params, inputs) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        recur_predict_fun(graph, cache, input_map, params, inputs)
    end
  end

  ## Custom Layers

  defp recur_predict_fun(
         %Axon{id: id, op: op, parent: parent, params: layer_params, opts: opts},
         cache,
         input_map,
         params,
         inputs
       )
       when is_function(op) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    inp_params =
      Map.new(layer_params, fn {k, %{name: v, frozen: frz}} ->
        {k, maybe_freeze(params[v], frz)}
      end)

    res = apply(op, [res | [inp_params] ++ opts])

    {res, Map.put(cache, id, res)}
  end

  ## Activation Layers

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :mish, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: parent,
           policy: %{compute: compute, output: output},
           opts: opts
         },
         cache,
         input_map,
         params,
         inputs
       )
       when op in @activation_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    input = Nx.as_type(res, compute)

    args =
      case opts do
        [] ->
          [input]

        [_ | _] ->
          [input, opts]
      end

    res = Nx.as_type(apply(Axon.Activations, op, args), output)

    {res, Map.put(cache, id, res)}
  end

  ## Linear Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :dense,
           parent: parent,
           params: %{"kernel" => %{name: w, frozen: w_frz}} = layer_params,
           policy: %{compute: compute, output: output},
           opts: [use_bias: use_bias]
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    input = Nx.as_type(res, compute)
    w = Nx.as_type(maybe_freeze(params[w], w_frz), compute)

    b =
      if use_bias do
        %{name: b, frozen: b_frz} = layer_params["bias"]
        Nx.as_type(maybe_freeze(params[b], b_frz), compute)
      else
        Nx.tensor(0.0, type: compute)
      end

    res = Nx.as_type(apply(Axon.Layers, :dense, [input, w, b]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :bilinear,
           parent: parents,
           params: %{"kernel" => %{name: w, frozen: w_frz}} = layer_params,
           policy: %{compute: compute, output: output},
           opts: [use_bias: use_bias]
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {[res1, res2], cache} =
      Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map, params, inputs))

    input1 = Nx.as_type(res1, compute)
    input2 = Nx.as_type(res2, compute)
    w = Nx.as_type(maybe_freeze(params[w], w_frz), compute)

    b =
      if use_bias do
        %{name: b, frozen: b_frz} = layer_params["bias"]
        Nx.as_type(maybe_freeze(params[b], b_frz), compute)
      else
        Nx.tensor(0.0, type: compute)
      end

    res = Nx.as_type(apply(Axon.Layers, :bilinear, [input1, input2, w, b]), output)

    {res, Map.put(cache, id, res)}
  end

  ## Sparse Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :embedding,
           parent: parent,
           params: %{"kernel" => %{name: w, frozen: w_frz}},
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    w = Nx.as_type(maybe_freeze(params[w], w_frz), compute)
    res = Nx.as_type(apply(Axon.Layers, :embedding, [res, w]), output)

    {res, Map.put(cache, id, res)}
  end

  ## Pooling Layers

  @pooling_layers [:max_pool, :avg_pool, :adaptive_avg_pool] ++
                    [:adaptive_max_pool, :adaptive_lp_pool, :lp_pool] ++
                    [:global_lp_pool, :global_max_pool, :global_avg_pool]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: parent,
           opts: opts,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       )
       when op in @pooling_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    input = Nx.as_type(res, compute)
    res = Nx.as_type(apply(Axon.Layers, op, [input, opts]), output)

    {res, Map.put(cache, id, res)}
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: parent,
           opts: opts,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       )
       when op in @dropout_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    input = Nx.as_type(res, compute)
    res = Nx.as_type(apply(Axon.Layers, op, [input, opts]), output)

    {res, Map.put(cache, id, res)}
  end

  ## Conv Layers

  @conv_layers [:conv, :conv_transpose, :depthwise_conv]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: parent,
           opts: opts,
           params: %{"kernel" => %{name: k, frozen: k_frz}} = layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       )
       when op in @conv_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    input = Nx.as_type(res, compute)
    k = Nx.as_type(maybe_freeze(params[k], k_frz), compute)

    b =
      if use_bias do
        %{name: b, frozen: b_frz} = layer_params["bias"]
        Nx.as_type(maybe_freeze(params[b], b_frz), compute)
      else
        Nx.tensor(0, type: compute)
      end

    res = Nx.as_type(apply(Axon.Layers, op, [input, k, b, opts]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :separable_conv2d,
           parent: parent,
           opts: opts,
           params:
             %{
               "k1" => %{name: k1, frozen: k1_frz},
               "k2" => %{name: k2, frozen: k2_frz}
             } = layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    input = Nx.as_type(res, compute)
    k1 = Nx.as_type(maybe_freeze(params[k1], k1_frz), compute)
    k2 = Nx.as_type(maybe_freeze(params[k2], k2_frz), compute)

    {b1, b2} =
      if use_bias do
        %{name: b1, frozen: b1_frz} = layer_params["b1"]
        %{name: b2, frozen: b2_frz} = layer_params["b2"]
        b1 = Nx.as_type(maybe_freeze(params[b1], b1_frz), compute)
        b2 = Nx.as_type(maybe_freeze(params[b2], b2_frz), compute)
        {b1, b2}
      else
        {Nx.tensor(0, type: compute), Nx.tensor(0, type: compute)}
      end

    res = Nx.as_type(apply(Axon.Layers, :separable_conv2d, [input, k1, b1, k2, b2, opts]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :separable_conv3d,
           parent: parent,
           opts: opts,
           params:
             %{
               "k1" => %{name: k1, frozen: k1_frz},
               "k2" => %{name: k2, frozen: k2_frz},
               "k3" => %{name: k3, frozen: k3_frz}
             } = layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    input = Nx.as_type(res, compute)
    k1 = Nx.as_type(maybe_freeze(params[k1], k1_frz), compute)
    k2 = Nx.as_type(maybe_freeze(params[k2], k2_frz), compute)
    k3 = Nx.as_type(maybe_freeze(params[k3], k3_frz), compute)

    {b1, b2, b3} =
      if use_bias do
        %{name: b1, frozen: b1_frz} = layer_params["b1"]
        %{name: b2, frozen: b2_frz} = layer_params["b2"]
        %{name: b3, frozen: b3_frz} = layer_params["b3"]
        b1 = Nx.as_type(maybe_freeze(params[b1], b1_frz), compute)
        b2 = Nx.as_type(maybe_freeze(params[b2], b2_frz), compute)
        b3 = Nx.as_type(maybe_freeze(params[b3], b3_frz), compute)
        {b1, b2, b3}
      else
        {Nx.tensor(0, type: compute), Nx.tensor(0, type: compute), Nx.tensor(0, type: compute)}
      end

    res =
      apply(Axon.Layers, :separable_conv3d, [input, k1, b1, k2, b2, k3, b3, opts])
      |> Nx.as_type(output)

    {res, Map.put(cache, id, res)}
  end

  ## Normalization Layers

  @normalization_layers [:batch_norm, :layer_norm, :group_norm, :instance_norm]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: parent,
           opts: opts,
           params: %{"gamma" => %{name: g, frozen: g_frz}, "beta" => %{name: b, frozen: b_frz}},
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       )
       when op in @normalization_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    input = Nx.as_type(res, compute)
    g = Nx.as_type(maybe_freeze(params[g], g_frz), compute)
    b = Nx.as_type(maybe_freeze(params[b], b_frz), compute)
    res = Nx.as_type(apply(Axon.Layers, op, [input, g, b, opts]), output)

    {res, Map.put(cache, id, res)}
  end

  ## Recurrent Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :lstm,
           parent: parent,
           params:
             %{
               "wii" => %{name: wii, frozen: wii_frz},
               "wif" => %{name: wif, frozen: wif_frz},
               "wig" => %{name: wig, frozen: wig_frz},
               "wio" => %{name: wio, frozen: wio_frz},
               "whi" => %{name: whi, frozen: whi_frz},
               "whf" => %{name: whf, frozen: whf_frz},
               "whg" => %{name: whg, frozen: whg_frz},
               "who" => %{name: who, frozen: who_frz}
             } = layer_params,
           policy: %{compute: compute, output: output},
           opts: [
             activation: activation,
             gate: gate,
             hidden_state: hidden_state,
             hidden_state_shape: hidden_state_shape,
             recurrent_initializer: recurrent_initializer,
             unroll: unroll,
             use_bias: use_bias
           ]
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    {{h, c}, cache} =
      to_hidden_state(
        hidden_state,
        res,
        cache,
        input_map,
        params,
        inputs,
        2,
        recurrent_initializer,
        hidden_state_shape
      )

    input = Nx.as_type(res, compute)

    input_kernel = {
      Nx.as_type(maybe_freeze(params[wii], wii_frz), compute),
      Nx.as_type(maybe_freeze(params[wif], wif_frz), compute),
      Nx.as_type(maybe_freeze(params[wig], wig_frz), compute),
      Nx.as_type(maybe_freeze(params[wio], wio_frz), compute)
    }

    hidden_kernel = {
      Nx.as_type(maybe_freeze(params[whi], whi_frz), compute),
      Nx.as_type(maybe_freeze(params[whf], whf_frz), compute),
      Nx.as_type(maybe_freeze(params[whg], whg_frz), compute),
      Nx.as_type(maybe_freeze(params[who], who_frz), compute)
    }

    bias =
      if use_bias do
        %{name: bi, frozen: bi_frz} = layer_params["bi"]
        %{name: bf, frozen: bf_frz} = layer_params["bf"]
        %{name: bg, frozen: bg_frz} = layer_params["bg"]
        %{name: bo, frozen: bo_frz} = layer_params["bo"]

        {
          Nx.as_type(maybe_freeze(params[bi], bi_frz), compute),
          Nx.as_type(maybe_freeze(params[bf], bf_frz), compute),
          Nx.as_type(maybe_freeze(params[bg], bg_frz), compute),
          Nx.as_type(maybe_freeze(params[bo], bo_frz), compute)
        }
      else
        {Nx.tensor(0, type: compute), Nx.tensor(0, type: compute), Nx.tensor(0, type: compute),
         Nx.tensor(0, type: compute)}
      end

    carry = {Nx.as_type(h, compute), Nx.as_type(c, compute)}

    gate_fn = &apply(Axon.Activations, gate, [&1])
    activation_fn = &apply(Axon.Activations, activation, [&1])

    {{c1, c2}, res} =
      case unroll do
        :static ->
          Axon.Recurrent.static_unroll(
            &Axon.Recurrent.lstm_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn),
            input,
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )

        :dynamic ->
          Axon.Recurrent.dynamic_unroll(
            &Axon.Recurrent.lstm_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn),
            input,
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )
      end

    res = {{Nx.as_type(c1, output), Nx.as_type(c2, output)}, Nx.as_type(res, output)}

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :conv_lstm,
           parent: parent,
           params: %{
             "wi" => %{name: wi, frozen: wi_frz},
             "wh" => %{name: wh, frozen: wh_frz},
             "b" => %{name: b, frozen: b_frz}
           },
           policy: %{compute: compute, output: output},
           opts: [
             hidden_state: hidden_state,
             strides: strides,
             padding: padding,
             hidden_state_shape: hidden_state_shape,
             recurrent_initializer: recurrent_initializer,
             unroll: unroll
           ]
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    {{h, c}, cache} =
      to_hidden_state(
        hidden_state,
        res,
        cache,
        input_map,
        params,
        inputs,
        2,
        recurrent_initializer,
        hidden_state_shape
      )

    input = Nx.as_type(res, compute)

    input_kernel = {Nx.as_type(maybe_freeze(params[wi], wi_frz), compute)}
    hidden_kernel = {Nx.as_type(maybe_freeze(params[wh], wh_frz), compute)}
    bias = {Nx.as_type(maybe_freeze(params[b], b_frz), compute)}

    carry = {Nx.as_type(h, compute), Nx.as_type(c, compute)}

    {{c1, c2}, out} =
      case unroll do
        :static ->
          Axon.Recurrent.static_unroll(
            &Axon.Recurrent.conv_lstm_cell(&1, &2, &3, &4, &5,
              strides: strides,
              padding: padding
            ),
            input,
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )

        :dynamic ->
          Axon.Recurrent.dynamic_unroll(
            &Axon.Recurrent.conv_lstm_cell(&1, &2, &3, &4, &5,
              strides: strides,
              padding: padding
            ),
            input,
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )
      end

    res = {{Nx.as_type(c1, output), Nx.as_type(c2, output)}, Nx.as_type(out, output)}

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :gru,
           parent: parent,
           params:
             %{
               "wir" => %{name: wir, frozen: wir_frz},
               "wiz" => %{name: wiz, frozen: wiz_frz},
               "win" => %{name: win, frozen: win_frz},
               "whr" => %{name: whr, frozen: whr_frz},
               "whz" => %{name: whz, frozen: whz_frz},
               "whn" => %{name: whn, frozen: whn_frz}
             } = layer_params,
           policy: %{compute: compute, output: output},
           opts: [
             activation: activation,
             gate: gate,
             hidden_state: hidden_state,
             hidden_state_shape: hidden_state_shape,
             recurrent_initializer: recurrent_initializer,
             unroll: unroll,
             use_bias: use_bias
           ]
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    {{h}, cache} =
      to_hidden_state(
        hidden_state,
        res,
        cache,
        input_map,
        params,
        inputs,
        1,
        recurrent_initializer,
        hidden_state_shape
      )

    input = Nx.as_type(res, compute)

    input_kernel = {
      Nx.as_type(maybe_freeze(params[wir], wir_frz), compute),
      Nx.as_type(maybe_freeze(params[wiz], wiz_frz), compute),
      Nx.as_type(maybe_freeze(params[win], win_frz), compute)
    }

    hidden_kernel = {
      Nx.as_type(maybe_freeze(params[whr], whr_frz), compute),
      Nx.as_type(maybe_freeze(params[whz], whz_frz), compute),
      Nx.as_type(maybe_freeze(params[whn], whn_frz), compute)
    }

    bias =
      if use_bias do
        %{name: br, frozen: br_frz} = layer_params["br"]
        %{name: bz, frozen: bz_frz} = layer_params["bz"]
        %{name: bin, frozen: bin_frz} = layer_params["bin"]
        %{name: bhn, frozen: bhn_frz} = layer_params["bhn"]

        {
          Nx.as_type(maybe_freeze(params[br], br_frz), compute),
          Nx.as_type(maybe_freeze(params[bz], bz_frz), compute),
          Nx.as_type(maybe_freeze(params[bin], bin_frz), compute),
          Nx.as_type(maybe_freeze(params[bhn], bhn_frz), compute)
        }
      else
        {
          Nx.tensor(0, type: compute),
          Nx.tensor(0, type: compute),
          Nx.tensor(0, type: compute),
          Nx.tensor(0, type: compute)
        }
      end

    carry = {Nx.as_type(h, compute)}

    gate_fn = &apply(Axon.Activations, gate, [&1])
    activation_fn = &apply(Axon.Activations, activation, [&1])

    {{c}, out} =
      case unroll do
        :static ->
          Axon.Recurrent.static_unroll(
            &Axon.Recurrent.gru_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn),
            input,
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )

        :dynamic ->
          Axon.Recurrent.dynamic_unroll(
            &Axon.Recurrent.gru_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn),
            input,
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )
      end

    res = {{Nx.as_type(c, output)}, Nx.as_type(out, output)}

    {res, Map.put(cache, id, res)}
  end

  ## Element-wise layers

  @element_wise_layers [:add, :subtract, :multiply]

  defp recur_predict_fun(
         %Axon{id: id, op: op, parent: parents, policy: %{compute: compute, output: output}},
         cache,
         input_map,
         params,
         inputs
       )
       when op in @element_wise_layers do
    {[expr | exprs], cache} =
      Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map, params, inputs))

    res =
      Enum.reduce(exprs, expr, fn next_expr, acc ->
        input = Nx.as_type(next_expr, compute)
        acc = Nx.as_type(acc, compute)
        Nx.as_type(apply(Nx, op, [acc, input]), output)
      end)

    {res, Map.put(cache, id, res)}
  end

  ## Shape Layers

  defp recur_predict_fun(
         %Axon{id: id, op: :flatten, parent: parent, policy: %{compute: compute, output: output}},
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    input = Nx.as_type(res, compute)
    res = Nx.as_type(apply(Axon.Layers, :flatten, [input]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :reshape,
           parent: parent,
           output_shape: output_shape,
           policy: %{compute: compute, output: output},
           opts: [constant: is_constant_reshape?]
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    inp = Nx.as_type(res, compute)

    reshape_shape =
      if is_constant_reshape? do
        output_shape
      else
        put_elem(output_shape, 0, elem(Nx.shape(inp), 0))
      end

    res = Nx.as_type(apply(Nx, :reshape, [inp, reshape_shape]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :transpose,
           parent: parent,
           opts: [permutation: permutation, constant: is_constant_reshape?],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    permutation =
      if is_constant_reshape? do
        permutation
      else
        [0 | Enum.map(permutation, &(&1 + 1))]
      end

    input = Nx.as_type(res, compute)
    res = Nx.as_type(apply(Nx, :transpose, [input, [axes: permutation]]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :pad,
           parent: parent,
           opts: [padding_config: config, value: value],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    config = [{0, 0, 0}, {0, 0, 0} | Enum.map(config, fn {x, y} -> {x, y, 0} end)]
    input = Nx.as_type(res, compute)
    res = Nx.as_type(apply(Nx, :pad, [input, value, config]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :concatenate,
           parent: parents,
           opts: [axis: axis],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {exprs, cache} =
      Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map, params, inputs))

    inps = Enum.map(exprs, &Nx.as_type(&1, compute))
    res = Nx.as_type(apply(Nx, :concatenate, [inps, [axis: axis]]), output)

    {res, Map.put(cache, id, res)}
  end

  ## Special Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :nx,
           parent: parent,
           opts: [fun: nx_fun],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs)

    res = Nx.as_type(nx_fun.(Nx.as_type(res, compute)), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :constant, opts: [value: tensor], policy: %{output: output}},
         cache,
         _,
         _,
         _
       ) do
    res = Nx.as_type(tensor, output)
    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :input, output_shape: shape},
         cache,
         input_map,
         _,
         inputs
       ) do
    res =
      if is_tuple(inputs) do
        idx = input_map[id]
        elem(inputs, idx)
      else
        inputs
      end

    unless Axon.Shape.compatible?(Nx.shape(res), shape) do
      raise ArgumentError,
            "invalid input shape given to model, expected input" <>
              " with shape #{inspect(shape)}, but got input with" <>
              " shape #{inspect(Nx.shape(res))}"
    end

    {res, Map.put(cache, id, res)}
  end

  defp maybe_freeze(param, true), do: Nx.Defn.Kernel.stop_grad(param)
  defp maybe_freeze(param, false), do: param

  defp to_hidden_state(
         hidden_state,
         input,
         cache,
         input_map,
         params,
         inputs,
         num_carry,
         recurrent_initializer,
         hidden_state_shape
       ) do
    case hidden_state do
      {%Axon{} = c, %Axon{} = h} ->
        {c_res, cache} = to_predict_fun(c, cache, input_map, params, inputs)
        {h_res, cache} = to_predict_fun(h, cache, input_map, params, inputs)
        {{c_res, h_res}, cache}

      {%Axon{} = c} ->
        {h_res, cache} = to_predict_fun(c, cache, input_map, params, inputs)
        {{h_res}, cache}

      %Axon{} = x ->
        {h_res, cache} = to_predict_fun(x, cache, input_map, params, inputs)
        {h_res, cache}

      nil ->
        shape = put_elem(hidden_state_shape, 0, elem(Nx.shape(input), 0))
        # TODO: Verify this does not embed large constant into the expression
        h_res =
          for _ <- 1..num_carry,
              do: apply(Axon.Initializers, recurrent_initializer, [[shape: shape]])

        {List.to_tuple(h_res), cache}
    end
  end

  ## Penalty Function Compilation

  @doc false
  def __jit_penalty__(graph, caller, args, opts) do
    fun = compile_penalty(graph)
    jit_or_apply(caller, fun, args, opts)
  end

  defp compile_penalty(graph) when is_tuple(graph) do
    graph = Tuple.to_list(graph)

    penalties =
      graph
      |> Enum.reduce(
        %{},
        fn x, cache ->
          to_penalty_fun(x, cache)
        end
      )

    [fun | funs] = Map.values(penalties)

    fn params ->
      funs
      |> Enum.reduce(fun.(params), fn penalty, acc -> Nx.add(penalty.(params), acc) end)
    end
  end

  defp compile_penalty(%Axon{} = graph) do
    penalties = to_penalty_fun(graph, %{})
    [fun | funs] = Map.values(penalties)

    fn params ->
      funs
      |> Enum.reduce(fun.(params), fn penalty, acc -> Nx.add(penalty.(params), acc) end)
    end
  end

  defp to_penalty_fun(%Axon{parent: parents}, cache) when is_list(parents) do
    Enum.reduce(parents, cache, fn graph, cache ->
      to_penalty_fun(graph, cache)
    end)
  end

  defp to_penalty_fun(
         %Axon{parent: parent, params: params, policy: %{params: param_policy}},
         cache
       ) do
    cache =
      params
      |> Enum.reduce(cache, fn {_, param}, cache ->
        %{name: name, regularizer: regularizer} = param

        case cache do
          %{^name => _} ->
            cache

          %{} ->
            fun = fn params ->
              case regularizer do
                :none ->
                  Nx.tensor(0.0, type: param_policy)

                regularizer when is_atom(regularizer) ->
                  apply(Axon.Regularizers, regularizer, [params[name]])

                regularizer when is_function(regularizer) ->
                  apply(regularizer, [params[name]])
              end
            end

            Map.put(cache, name, fun)
        end
      end)

    if parent do
      to_penalty_fun(parent, cache)
    else
      cache
    end
  end
end
