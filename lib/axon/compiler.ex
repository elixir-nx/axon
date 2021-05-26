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

  defp to_init_fun(%Axon{parent: parents}, cache) when is_list(parents) do
    Enum.reduce(parents, cache, &to_init_fun/2)
  end

  defp to_init_fun(%Axon{parent: parent, params: params, opts: opts}, cache) do
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
              fun = fn -> apply(Axon.Initializers, initializer, [[shape: shape]]) end
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

  defp compile_predict(graph) when is_tuple(graph) do
    graph = Tuple.to_list(graph)

    input_ids =
      Enum.reduce(graph, [], fn x, input_ids ->
        get_inputs(x, input_ids)
      end)

    input_map =
      input_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    fn params, inputs ->
      {funs, _} = Enum.map_reduce(graph, %{}, &to_predict_fun(&1, &2, input_map))

      funs
      |> Enum.reverse()
      |> Enum.map(& &1.(params, inputs))
      |> List.to_tuple()
    end
  end

  defp compile_predict(%Axon{} = graph) do
    input_ids = get_inputs(graph, [])

    input_map =
      input_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    fn params, inputs ->
      {fun, _} = to_predict_fun(graph, %{}, input_map)
      fun.(params, inputs)
    end
  end

  ## Parameter Ordering

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
      inp_params = Map.new(layer_params, fn {k, %{name: v}} -> {k, params[v]} end)
      apply(op, [fun.(params, input) | [inp_params] ++ opts])
    end

    {fun, cache}
  end

  ## Activation Layers

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :mish, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  defp recur_predict_fun(%Axon{op: op, parent: parent}, cache, input_map)
       when op in @activation_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Activations, op, [fun.(params, inputs)])
    end

    {fun, cache}
  end

  ## Linear Layers

  defp recur_predict_fun(
         %Axon{
           op: :dense,
           parent: parent,
           params: %{"kernel" => %{name: w}, "bias" => %{name: b}}
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, :dense, [fun.(params, inputs), params[w], params[b]])
    end

    {fun, cache}
  end

  ## Pooling Layers

  @pooling_layers [:max_pool, :avg_pool, :adaptive_avg_pool, :adaptive_max_pool, :lp_pool]

  defp recur_predict_fun(%Axon{op: op, parent: parent, opts: opts}, cache, input_map)
       when op in @pooling_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, op, [fun.(params, inputs), opts])
    end

    {fun, cache}
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp recur_predict_fun(%Axon{op: op, parent: parent, opts: opts}, cache, input_map)
       when op in @dropout_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, op, [fun.(params, inputs), opts])
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
           params: %{"kernel" => %{name: k}, "bias" => %{name: b}}
         },
         cache,
         input_map
       )
       when op in @conv_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, op, [fun.(params, inputs), params[k], params[b], opts])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :separable_conv2d,
           parent: parent,
           opts: opts,
           params: %{
             "k1" => %{name: k1},
             "b1" => %{name: b1},
             "k2" => %{name: k2},
             "b2" => %{name: b2}
           }
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, :separable_conv2d, [
        fun.(params, inputs),
        params[k1],
        params[b1],
        params[k2],
        params[b2],
        opts
      ])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :separable_conv2d,
           parent: parent,
           opts: opts,
           params: %{
             "k1" => %{name: k1},
             "b1" => %{name: b1},
             "k2" => %{name: k2},
             "b2" => %{name: b2},
             "k3" => %{name: k3},
             "b3" => %{name: b3}
           }
         },
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, :separable_conv3d, [
        fun.(params, inputs),
        params[k1],
        params[b1],
        params[k2],
        params[b2],
        params[k3],
        params[b3],
        opts
      ])
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
           params: %{"gamma" => %{name: g}, "beta" => %{name: b}}
         },
         cache,
         input_map
       )
       when op in @normalization_layers do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, op, [fun.(params, inputs), params[g], params[b], opts])
    end

    {fun, cache}
  end

  ## Recurrent Layers

  defp recur_predict_fun(
         %Axon{
           op: :lstm,
           parent: parent,
           params: %{
             "wii" => %{name: wii},
             "wif" => %{name: wif},
             "wig" => %{name: wig},
             "wio" => %{name: wio},
             "whi" => %{name: whi},
             "whf" => %{name: whf},
             "whg" => %{name: whg},
             "who" => %{name: who},
             "bi" => %{name: bi},
             "bf" => %{name: bf},
             "bg" => %{name: bg},
             "bo" => %{name: bo}
           },
           opts: [
             activation: activation,
             gate: gate,
             hidden_state: hidden_state,
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
      input = fun.(params, input)

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

      input_kernel = {params[wii], params[wif], params[wig], params[wio]}
      hidden_kernel = {params[whi], params[whf], params[whg], params[who]}
      bias = {params[bi], params[bf], params[bg], params[bo]}

      carry = hidden_state_fun.(params, input)

      gate_fn = &apply(Axon.Activations, gate, [&1])
      activation_fn = &apply(Axon.Activations, activation, [&1])

      case unroll do
        :static ->
          Axon.Recurrent.static_unroll(
            &Axon.Recurrent.lstm_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn),
            fun.(params, input),
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )

        :dynamic ->
          Axon.Recurrent.dynamic_unroll(
            &Axon.Recurrent.lstm_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn),
            fun.(params, input),
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )
      end
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :conv_lstm,
           parent: parent,
           params: %{
             "wi" => %{name: wi},
             "wh" => %{name: wh},
             "b" => %{name: b}
           },
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
      input = fun.(params, input)

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

      input_kernel = {params[wi]}
      hidden_kernel = {params[wh]}
      bias = {params[b]}

      carry = hidden_state_fun.(params, input)

      case unroll do
        :static ->
          Axon.Recurrent.static_unroll(
            &Axon.Recurrent.conv_lstm_cell(&1, &2, &3, &4, &5, strides: strides, padding: padding),
            fun.(params, input),
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )

        :dynamic ->
          Axon.Recurrent.dynamic_unroll(
            &Axon.Recurrent.conv_lstm_cell(&1, &2, &3, &4, &5, strides: strides, padding: padding),
            fun.(params, input),
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )
      end
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{
           op: :gru,
           parent: parent,
           params: %{
             "wir" => %{name: wir},
             "wiz" => %{name: wiz},
             "win" => %{name: win},
             "whr" => %{name: whr},
             "whz" => %{name: whz},
             "whn" => %{name: whn},
             "br" => %{name: br},
             "bz" => %{name: bz},
             "bin" => %{name: bin},
             "bhn" => %{name: bhn}
           },
           opts: [
             activation: activation,
             gate: gate,
             hidden_state: hidden_state,
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
      input = fun.(params, input)

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

      input_kernel = {params[wir], params[wiz], params[win]}
      hidden_kernel = {params[whr], params[whz], params[whn]}
      bias = {params[br], params[bz], params[bin], params[bhn]}

      carry = hidden_state_fun.(params, input)

      gate_fn = &apply(Axon.Activations, gate, [&1])
      activation_fn = &apply(Axon.Activations, activation, [&1])

      case unroll do
        :static ->
          Axon.Recurrent.static_unroll(
            &Axon.Recurrent.gru_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn),
            fun.(params, input),
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )

        :dynamic ->
          Axon.Recurrent.dynamic_unroll(
            &Axon.Recurrent.gru_cell(&1, &2, &3, &4, &5, gate_fn, activation_fn),
            fun.(params, input),
            carry,
            input_kernel,
            hidden_kernel,
            bias
          )
      end
    end

    {fun, cache}
  end

  ## Element-wise layers

  @element_wise_layers [:add, :subtract, :multiply]

  defp recur_predict_fun(%Axon{op: op, parent: parents}, cache, input_map)
       when op in @element_wise_layers do
    {[fun | funs], cache} = Enum.map_reduce(parents, cache, &recur_predict_fun(&1, &2, input_map))

    fun = fn params, inputs ->
      Enum.reduce(funs, fun.(params, inputs), fn next_fn, acc ->
        apply(Nx, op, [acc, next_fn.(params, inputs)])
      end)
    end

    {fun, cache}
  end

  ## Shape Layers

  defp recur_predict_fun(%Axon{op: :flatten, parent: parent}, cache, input_map) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      apply(Axon.Layers, :flatten, [fun.(params, inputs)])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{op: :reshape, parent: parent, output_shape: output_shape},
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      inp = fun.(params, inputs)
      reshape_shape = put_elem(output_shape, 0, elem(Nx.shape(inp), 0))
      apply(Nx, :reshape, [inp, reshape_shape])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{op: :transpose, parent: parent, opts: [permutation: permutation]},
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      permutation = [0 | Enum.map(permutation, &(&1 + 1))]
      apply(Nx, :transpose, [fun.(params, inputs), [axes: permutation]])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{op: :pad, parent: parent, opts: [padding_config: config, value: value]},
         cache,
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      config = [{0, 0, 0}, {0, 0, 0} | Enum.map(config, fn {x, y} -> {x, y, 0} end)]
      apply(Nx, :pad, [fun.(params, inputs), value, config])
    end

    {fun, cache}
  end

  defp recur_predict_fun(
         %Axon{op: :concatenate, parent: parents, opts: [axis: axis]},
         cache,
         input_map
       ) do
    {funs, cache} = Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map))

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
         input_map
       ) do
    {fun, cache} = to_predict_fun(parent, cache, input_map)

    fun = fn params, inputs ->
      nx_fun.(fun.(params, inputs))
    end

    {fun, cache}
  end

  defp recur_predict_fun(%Axon{op: :input, id: id}, cache, input_map) do
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

  defp to_penalty_fun(%Axon{parent: parent, params: params}, cache) do
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
                  Nx.tensor(0.0)

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
