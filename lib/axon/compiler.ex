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

  defp to_init_fun(%Axon{parent: parents}, cache) when is_list(parents) do
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
      {fun, _} = to_predict_fun(graph, %{}, input_map)

      case fun do
        [_ | _] = funs ->
          do_recur_apply(funs, params, inputs, [])

        fun when is_function(fun) ->
          fun.(params, inputs)
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

  defp do_recur_apply([fun | []], params, inputs, acc) when is_function(fun) do
    [fun.(params, inputs) | acc]
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp do_recur_apply([fun | []], params, inputs, acc) when is_list(fun) do
    res = do_recur_apply(fun, params, inputs, [])

    [res | acc]
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp do_recur_apply([fun | funs], params, inputs, acc) when is_list(fun) do
    res = do_recur_apply(fun, params, inputs, [])
    do_recur_apply(funs, params, inputs, [res | acc])
  end

  defp do_recur_apply([fun | funs], params, inputs, acc) when is_function(fun) do
    do_recur_apply(funs, params, inputs, [fun.(params, inputs) | acc])
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

  defp to_predict_fun(graph, cache, input_map) when is_tuple(graph) do
    graph
    |> Tuple.to_list()
    |> Enum.map_reduce(cache, fn x, acc -> to_predict_fun(x, acc, input_map) end)
  end

  defp to_predict_fun(%{id: id} = graph, cache, input_map) do
    case cache do
      %{^id => fun} ->
        {fun, cache}

      %{} ->
        {fun, cache} = recur_predict_fun(graph, cache, input_map)
        cache = Map.put(cache, id, fun)
        {fun, cache}
    end
  end

  ## Custom Layers

  defp recur_predict_fun(
         %Axon{op: op, parent: parent, params: layer_params, opts: opts},
         cache,
         input_map
       )
       when is_function(op) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, input ->
      inp_params =
        Map.new(layer_params, fn {k, %{name: v, frozen: frz}} ->
          {k, maybe_freeze(params[v], frz)}
        end)

      apply(op, [fun.(params, input) | [inp_params] ++ opts])
    end

    {fun, cache}
  end

  ## Activation Layers

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :mish, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  defp recur_predict_fun(
         %Axon{op: op, parent: parent, policy: %{compute: compute, output: output}, opts: opts},
         cache,
         input_map
       )
       when op in @activation_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)

      args =
        case opts do
          [] ->
            [input]

          [_ | _] ->
            [input, opts]
        end

      Nx.as_type(apply(Axon.Activations, op, args), output)
    end

    {fun, cache}
  end

  ## Linear Layers

  defp recur_predict_fun(
         %Axon{
           op: :dense,
           parent: parent,
           params: %{"kernel" => %{name: w, frozen: w_frz}} = layer_params,
           policy: %{compute: compute, output: output},
           opts: [use_bias: use_bias]
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)
      w = Nx.as_type(maybe_freeze(params[w], w_frz), compute)

      b =
        if use_bias do
          %{name: b, frozen: b_frz} = layer_params["bias"]
          Nx.as_type(maybe_freeze(params[b], b_frz), compute)
        else
          Nx.tensor(0.0, type: compute)
        end

      Nx.as_type(apply(Axon.Layers, :dense, [input, w, b]), output)
    end

    {fun, cache}
  end

  ## Sparse Layers

  defp recur_predict_fun(
         %Axon{
           op: :embedding,
           parent: parent,
           params: %{"kernel" => %{name: w, frozen: w_frz}},
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      input = fun.(params, inputs)
      w = Nx.as_type(maybe_freeze(params[w], w_frz), compute)

      Nx.as_type(apply(Axon.Layers, :embedding, [input, w]), output)
    end

    {fun, cache}
  end

  ## Pooling Layers

  @pooling_layers [
                    :max_pool,
                    :avg_pool,
                    :adaptive_avg_pool,
                    :adaptive_max_pool,
                    :adaptive_lp_pool
                  ] ++
                    [:lp_pool, :global_lp_pool, :global_max_pool, :global_avg_pool]

  defp recur_predict_fun(
         %Axon{op: op, parent: parent, opts: opts, policy: %{compute: compute, output: output}},
         cache,
         input_map
       )
       when op in @pooling_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)
      Nx.as_type(apply(Axon.Layers, op, [input, opts]), output)
    end

    {fun, cache}
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp recur_predict_fun(
         %Axon{op: op, parent: parent, opts: opts, policy: %{compute: compute, output: output}},
         cache,
         input_map
       )
       when op in @dropout_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)
      Nx.as_type(apply(Axon.Layers, op, [input, opts]), output)
    end

    {fun, cache}
  end

  ## Conv Layers

  @conv_layers [:conv, :conv_transpose, :depthwise_conv]

  defp recur_predict_fun(
         %Axon{
           op: op,
           parent: parent,
           opts: opts,
           params: %{"kernel" => %{name: k, frozen: k_frz}} = layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map
       )
       when op in @conv_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)
      k = Nx.as_type(maybe_freeze(params[k], k_frz), compute)

      b =
        if use_bias do
          %{name: b, frozen: b_frz} = layer_params["bias"]
          Nx.as_type(maybe_freeze(params[b], b_frz), compute)
        else
          Nx.tensor(0, type: compute)
        end

      Nx.as_type(apply(Axon.Layers, op, [input, k, b, opts]), output)
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
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
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)
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

      Nx.as_type(apply(Axon.Layers, :separable_conv2d, [input, k1, b1, k2, b2, opts]), output)
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
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
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)
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

      Nx.as_type(
        apply(Axon.Layers, :separable_conv3d, [input, k1, b1, k2, b2, k3, b3, opts]),
        output
      )
    end

    {fun, cache}
  end

  ## Normalization Layers

  @normalization_layers [:batch_norm, :layer_norm, :group_norm, :instance_norm]

  defp recur_predict_fun(
         %Axon{
           op: op,
           parent: parent,
           opts: opts,
           params: %{"gamma" => %{name: g, frozen: g_frz}, "beta" => %{name: b, frozen: b_frz}},
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map
       )
       when op in @normalization_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)
      g = Nx.as_type(maybe_freeze(params[g], g_frz), compute)
      b = Nx.as_type(maybe_freeze(params[b], b_frz), compute)
      Nx.as_type(apply(Axon.Layers, op, [input, g, b, opts]), output)
    end

    {fun, cache}
  end

  ## Recurrent Layers

  defp recur_predict_fun(
         %Axon{
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
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, input ->
      input = Nx.as_type(fun.(params, input), compute)

      hidden_state_fun =
        case hidden_state do
          {%Axon{} = c, %Axon{} = h} ->
            {c_fun, cache} = to_predict_fun(c, cache, input_map)
            {h_fun, _} = to_predict_fun(h, cache, input_map)

            fn params, inputs ->
              {c_fun.(params, inputs), h_fun.(params, inputs)}
            end

          %Axon{} = x ->
            {hidden_fun, _} = to_predict_fun(x, cache, input_map)
            hidden_fun

          nil ->
            shape = put_elem(hidden_state_shape, 0, elem(Nx.shape(input), 0))

            fn _, _ ->
              {
                apply(Axon.Initializers, recurrent_initializer, [[type: compute, shape: shape]]),
                apply(Axon.Initializers, recurrent_initializer, [[type: compute, shape: shape]])
              }
            end
        end

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

      {h, c} = hidden_state_fun.(params, input)
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

      {{Nx.as_type(c1, output), Nx.as_type(c2, output)}, Nx.as_type(res, output)}
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
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
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, input ->
      input = Nx.as_type(fun.(params, input), compute)

      hidden_state_fun =
        case hidden_state do
          {%Axon{} = c, %Axon{} = h} ->
            {c_fun, cache} = to_predict_fun(c, cache, input_map)
            {h_fun, _} = to_predict_fun(h, cache, input_map)

            fn params, inputs ->
              {c_fun.(params, inputs), h_fun.(params, inputs)}
            end

          %Axon{} = x ->
            {hidden_fun, _} = to_predict_fun(x, cache, input_map)
            hidden_fun

          nil ->
            shape = put_elem(hidden_state_shape, 0, elem(Nx.shape(input), 0))

            fn _, _ ->
              {
                apply(Axon.Initializers, recurrent_initializer, [[shape: shape]]),
                apply(Axon.Initializers, recurrent_initializer, [[shape: shape]])
              }
            end
        end

      input_kernel = {Nx.as_type(maybe_freeze(params[wi], wi_frz), compute)}
      hidden_kernel = {Nx.as_type(maybe_freeze(params[wh], wh_frz), compute)}
      bias = {Nx.as_type(maybe_freeze(params[b], b_frz), compute)}

      {h, c} = hidden_state_fun.(params, input)
      carry = {Nx.as_type(h, compute), Nx.as_type(c, compute)}

      case unroll do
        :static ->
          Nx.as_type(
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
            ),
            output
          )

        :dynamic ->
          Nx.as_type(
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
            ),
            output
          )
      end
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
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
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, input ->
      input = Nx.as_type(fun.(params, input), compute)

      hidden_state_fun =
        case hidden_state do
          {%Axon{} = c} ->
            {h_fun, _} = to_predict_fun(c, cache, input_map)

            fn params, inputs ->
              {h_fun.(params, inputs)}
            end

          %Axon{} = x ->
            {hidden_fun, _} = to_predict_fun(x, cache, input_map)
            hidden_fun

          nil ->
            shape = put_elem(hidden_state_shape, 0, elem(Nx.shape(input), 0))

            fn _, _ ->
              {
                apply(Axon.Initializers, recurrent_initializer, [[shape: shape]])
              }
            end
        end

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

      {h} = hidden_state_fun.(params, input)
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

      {{Nx.as_type(c, output)}, Nx.as_type(out, output)}
    end

    {fun, cache}
  end

  ## Element-wise layers

  @element_wise_layers [:add, :subtract, :multiply]

  defp recur_predict_fun(
         %Axon{op: op, parent: parents, policy: %{compute: compute, output: output}},
         cache,
         input_map
       )
       when op in @element_wise_layers do
    {[fun | funs], cache} = Enum.map_reduce(parents, cache, &recur_predict_fun(&1, &2, input_map))

    fun = fn params, inputs ->
      Enum.reduce(funs, fun.(params, inputs), fn next_fn, acc ->
        input = Nx.as_type(next_fn.(params, inputs), compute)
        acc = Nx.as_type(acc, compute)
        Nx.as_type(apply(Nx, op, [acc, input]), output)
      end)
    end

    {fun, cache}
  end

  ## Shape Layers

  defp recur_predict_fun(
         %Axon{op: :flatten, parent: parent, policy: %{compute: compute, output: output}},
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      input = Nx.as_type(fun.(params, inputs), compute)
      Nx.as_type(apply(Axon.Layers, :flatten, [input]), output)
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :reshape,
           parent: parent,
           output_shape: output_shape,
           policy: %{compute: compute, output: output},
           opts: [constant: is_constant_reshape?]
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      inp = Nx.as_type(fun.(params, inputs), compute)

      reshape_shape =
        if is_constant_reshape? do
          output_shape
        else
          put_elem(output_shape, 0, elem(Nx.shape(inp), 0))
        end

      Nx.as_type(apply(Nx, :reshape, [inp, reshape_shape]), output)
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :transpose,
           parent: parent,
           opts: [permutation: permutation, constant: is_constant_reshape?],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      permutation =
        if is_constant_reshape? do
          permutation
        else
          [0 | Enum.map(permutation, &(&1 + 1))]
        end

      input = Nx.as_type(fun.(params, inputs), compute)
      Nx.as_type(apply(Nx, :transpose, [input, [axes: permutation]]), output)
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :pad,
           parent: parent,
           opts: [padding_config: config, value: value],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      config = [{0, 0, 0}, {0, 0, 0} | Enum.map(config, fn {x, y} -> {x, y, 0} end)]
      input = Nx.as_type(fun.(params, inputs), compute)
      Nx.as_type(apply(Nx, :pad, [input, value, config]), output)
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :concatenate,
           parent: parents,
           opts: [axis: axis],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map
       ) do
    {funs, cache} = Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map))

    fun = fn params, inputs ->
      inps = Enum.map(funs, &Nx.as_type(&1.(params, inputs), compute))
      Nx.as_type(apply(Nx, :concatenate, [inps, [axis: axis]]), output)
    end

    {fun, cache}
  end

  ## Special Layers

  defp recur_predict_fun(
         %Axon{
           op: :nx,
           parent: parent,
           opts: [fun: nx_fun],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      Nx.as_type(nx_fun.(Nx.as_type(fun.(params, inputs), compute)), output)
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{op: :constant, opts: [value: tensor], policy: %{output: output}},
         cache,
         _
       ) do
    fun = fn _, _ ->
      Nx.as_type(tensor, output)
    end

    {fun, cache}
  end

  defp recur_predict_fun(%Axon{op: :input, id: id, output_shape: shape}, cache, input_map) do
    fun = fn _, inputs ->
      value =
        if is_tuple(inputs) do
          idx = input_map[id]
          elem(inputs, idx)
        else
          inputs
        end

      unless Axon.Shape.compatible?(Nx.shape(value), shape) do
        raise ArgumentError,
              "invalid input shape given to model, expected input" <>
                " with shape #{inspect(shape)}, but got input with" <>
                " shape #{inspect(Nx.shape(value))}"
      end

      value
    end

    {fun, cache}
  end

  defp maybe_freeze(param, true), do: Nx.Defn.Kernel.stop_grad(param)
  defp maybe_freeze(param, false), do: param

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
