defmodule Axon.CompilerError do
  defexception [:exception, :graph]

  @impl true
  def message(%{graph: %Axon{op: op, name: name}, exception: exception}) do
    """
    error while building prediction for #{op} layer with name #{name}:

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
    mode = opts[:mode] || :train
    {compile_init(graph), compile_predict(graph, mode)}
  end

  @doc false
  def __jit_init__(graph, caller, [] = args, opts) do
    fun = compile_init(graph)
    jit_or_apply(caller, fun, args, opts)
  end

  defp compile_init(graph) when is_tuple(graph) do
    init_fn = fn ->
      graph
      |> Tuple.to_list()
      |> Enum.reduce(%{}, &to_init_fun/2)
      |> Map.new(fn {k, v} ->
        v = Map.new(v, fn {k_sub, v_sub} -> {k_sub, v_sub.()} end)
        {k, v}
      end)
    end

    fn -> Nx.Defn.jit_or_apply(init_fn, []) end
  end

  defp compile_init(%Axon{} = graph) do
    fn ->
      graph
      |> to_init_fun(%{})
      |> Map.new(fn {k, v} ->
        v = Map.new(v, fn {k_sub, v_sub} -> {k_sub, v_sub.()} end)
        {k, v}
      end)
    end
  end

  defp to_init_fun(graph, cache) when is_tuple(graph) do
    graph
    |> Tuple.to_list()
    |> Enum.reduce(cache, fn x, acc -> to_init_fun(x, acc) end)
  end

  defp to_init_fun(
         %Axon{parent: parents, name: name, params: params, policy: %{params: dtype}},
         cache
       )
       when is_list(parents) do
    cache =
      case cache do
        %{^name => _} ->
          cache

        %{} ->
          layer_params =
            Enum.reduce(params, %{}, fn {_, param}, layer_params ->
              %{name: name, shape: shape, initializer: initializer} = param
              fun = fn -> apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]]) end
              Map.put(layer_params, name, fun)
            end)

          if Enum.empty?(layer_params) do
            cache
          else
            Map.put(cache, name, layer_params)
          end
      end

    Enum.reduce(parents, cache, &to_init_fun/2)
  end

  defp to_init_fun(
         %Axon{parent: parent, name: name, params: params, opts: opts, policy: %{params: dtype}},
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
      case cache do
        %{^name => _} ->
          cache

        %{} ->
          layer_params =
            Enum.reduce(params, %{}, fn {_, param}, layer_params ->
              %{name: name, shape: shape, initializer: initializer} = param
              fun = fn -> apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]]) end
              Map.put(layer_params, name, fun)
            end)

          if Enum.empty?(layer_params) do
            cache
          else
            Map.put(cache, name, layer_params)
          end
      end

    if parent do
      to_init_fun(parent, cache)
    else
      cache
    end
  end

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(graph, caller, args, opts) do
    {mode, opts} = Keyword.pop(opts, :mode, :inference)
    fun = compile_predict(graph, mode)
    jit_or_apply(caller, fun, args, opts)
  end

  defp compile_predict(graph, mode) do
    input_ids = get_inputs(graph, [])

    input_map =
      input_ids
      |> Enum.uniq()
      |> Enum.sort()
      |> Enum.with_index()
      |> Enum.into(%{})

    # Warn if input map is empty
    if Enum.empty?(input_map) do
      Logger.warn(
        "You are compiling a graph with no inputs. If this was" <>
          " intentional, you can run your model's predict function" <>
          " with an empty tuple as input: predict_fn(params, {})"
      )
    end

    predict_fn = fn params, inputs ->
      inputs = maybe_flatten(inputs)
      {expr, _} = to_predict_fun(graph, %{}, input_map, params, inputs, mode)

      case expr do
        [_ | _] = exprs ->
          do_recur_to_tuple(exprs, [])

        expr ->
          expr
      end
    end

    &Nx.Defn.jit_or_apply(predict_fn, [&1, &2])
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

  defp to_predict_fun(graph, cache, input_map, params, inputs, mode) when is_tuple(graph) do
    graph
    |> Tuple.to_list()
    |> Enum.map_reduce(cache, fn x, acc ->
      to_predict_fun(x, acc, input_map, params, inputs, mode)
    end)
  end

  defp to_predict_fun(%{id: id} = graph, cache, input_map, params, inputs, mode) do
    case cache do
      %{^id => res} ->
        {res, cache}

      %{} ->
        try do
          recur_predict_fun(graph, cache, input_map, params, inputs, mode)
        rescue
          e -> reraise Axon.CompilerError.exception(graph: graph, exception: e), __STACKTRACE__
        end
    end
  end

  ## Custom Layers

  defp recur_predict_fun(
         %Axon{id: id, name: name, op: op, parent: parent, params: layer_params, opts: opts},
         cache,
         input_map,
         params,
         inputs,
         mode
       )
       when is_function(op) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    inp_params =
      Map.new(layer_params, fn {k, %{name: v, frozen: frz}} ->
        {k, maybe_freeze(params[name][v], frz)}
      end)

    res = apply(op, [res | [inp_params] ++ opts])

    {res, Map.put(cache, id, res)}
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
           parent: parent,
           policy: %{compute: compute, output: output},
           opts: opts
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       )
       when op in @activation_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

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
           name: name,
           parent: parent,
           params: layer_params,
           policy: %{compute: compute, output: output},
           opts: [use_bias: use_bias]
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    input = Nx.as_type(res, compute)
    w = layer_param(layer_params, "kernel", params[name], compute)

    b =
      if use_bias do
        layer_param(layer_params, "bias", params[name], compute)
      else
        Nx.tensor(0.0, type: compute)
      end

    res = Nx.as_type(apply(Axon.Layers, :dense, [input, w, b]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name,
           op: :bilinear,
           parent: parents,
           params: layer_params,
           policy: %{compute: compute, output: output},
           opts: [use_bias: use_bias]
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       ) do
    {[res1, res2], cache} =
      Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map, params, inputs, mode))

    input1 = Nx.as_type(res1, compute)
    input2 = Nx.as_type(res2, compute)
    w = layer_param(layer_params, "kernel", params[name], compute)

    b =
      if use_bias do
        layer_param(layer_params, "bias", params[name], compute)
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
           name: name,
           op: :embedding,
           parent: parent,
           params: layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    w = layer_param(layer_params, "kernel", params[name], compute)
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
         inputs,
         mode
       )
       when op in @pooling_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

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
         inputs,
         mode
       )
       when op in @dropout_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    input = Nx.as_type(res, compute)

    res =
      case mode do
        :train ->
          Nx.as_type(apply(Axon.Layers, op, [input, opts]), output)

        :inference ->
          # Skip dropout in inference mode
          Nx.as_type(input, output)
      end

    {res, Map.put(cache, id, res)}
  end

  ## Conv Layers

  @conv_layers [:conv, :conv_transpose, :depthwise_conv]

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name,
           op: op,
           parent: parent,
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       )
       when op in @conv_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    input = Nx.as_type(res, compute)
    k = layer_param(layer_params, "kernel", params[name], compute)

    b =
      if use_bias do
        layer_param(layer_params, "bias", params[name], compute)
      else
        Nx.tensor(0, type: compute)
      end

    res = Nx.as_type(apply(Axon.Layers, op, [input, k, b, opts]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name,
           op: :separable_conv2d,
           parent: parent,
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    input = Nx.as_type(res, compute)
    k1 = layer_param(layer_params, "k1", params[name], compute)
    k2 = layer_param(layer_params, "k2", params[name], compute)

    {b1, b2} =
      if use_bias do
        {layer_param(layer_params, "b1", params[name], compute),
         layer_param(layer_params, "b2", params[name], compute)}
      else
        {Nx.tensor(0, type: compute), Nx.tensor(0, type: compute)}
      end

    res = Nx.as_type(apply(Axon.Layers, :separable_conv2d, [input, k1, b1, k2, b2, opts]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name,
           op: :separable_conv3d,
           parent: parent,
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    input = Nx.as_type(res, compute)
    k1 = layer_param(layer_params, "k1", params[name], compute)
    k2 = layer_param(layer_params, "k2", params[name], compute)
    k3 = layer_param(layer_params, "k3", params[name], compute)

    {b1, b2, b3} =
      if use_bias do
        {layer_param(layer_params, "b1", params[name], compute),
         layer_param(layer_params, "b2", params[name], compute),
         layer_param(layer_params, "b3", params[name], compute)}
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
           name: name,
           op: op,
           parent: parent,
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       )
       when op in @normalization_layers do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    input = Nx.as_type(res, compute)
    g = layer_param(layer_params, "gamma", params[name], compute)
    b = layer_param(layer_params, "beta", params[name], compute)
    res = Nx.as_type(apply(Axon.Layers, op, [input, g, b, opts]), output)

    {res, Map.put(cache, id, res)}
  end

  ## Recurrent Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name,
           op: :lstm,
           parent: parent,
           params: layer_params,
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
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

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
        hidden_state_shape,
        mode
      )

    input = Nx.as_type(res, compute)

    input_kernel = {
      layer_param(layer_params, "wii", params[name], compute),
      layer_param(layer_params, "wif", params[name], compute),
      layer_param(layer_params, "wig", params[name], compute),
      layer_param(layer_params, "wio", params[name], compute)
    }

    hidden_kernel = {
      layer_param(layer_params, "whi", params[name], compute),
      layer_param(layer_params, "whf", params[name], compute),
      layer_param(layer_params, "whg", params[name], compute),
      layer_param(layer_params, "who", params[name], compute)
    }

    bias =
      if use_bias do
        {
          layer_param(layer_params, "bi", params[name], compute),
          layer_param(layer_params, "bf", params[name], compute),
          layer_param(layer_params, "bg", params[name], compute),
          layer_param(layer_params, "bo", params[name], compute)
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
           name: name,
           op: :conv_lstm,
           parent: parent,
           params: layer_params,
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
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

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
        hidden_state_shape,
        mode
      )

    input = Nx.as_type(res, compute)

    input_kernel = {layer_param(layer_params, "wi", params[name], compute)}
    hidden_kernel = {layer_param(layer_params, "wh", params[name], compute)}
    bias = {layer_param(layer_params, "b", params[name], compute)}

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
           name: name,
           op: :gru,
           parent: parent,
           params: layer_params,
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
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

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
        hidden_state_shape,
        mode
      )

    input = Nx.as_type(res, compute)

    input_kernel = {
      layer_param(layer_params, "wir", params[name], compute),
      layer_param(layer_params, "wiz", params[name], compute),
      layer_param(layer_params, "win", params[name], compute)
    }

    hidden_kernel = {
      layer_param(layer_params, "whr", params[name], compute),
      layer_param(layer_params, "whz", params[name], compute),
      layer_param(layer_params, "whn", params[name], compute)
    }

    bias =
      if use_bias do
        {
          layer_param(layer_params, "br", params[name], compute),
          layer_param(layer_params, "bz", params[name], compute),
          layer_param(layer_params, "bin", params[name], compute),
          layer_param(layer_params, "bhn", params[name], compute)
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
         inputs,
         mode
       )
       when op in @element_wise_layers do
    {[expr | exprs], cache} =
      Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map, params, inputs, mode))

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
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

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
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

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
           op: :resize,
           parent: parent,
           policy: %{compute: compute, output: output},
           opts: [shape: shape, method: method, channels: channels]
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    inp = Nx.as_type(res, compute)

    res =
      Nx.as_type(
        Axon.Layers.resize(inp, shape: shape, method: method, channels: channels),
        output
      )

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :transpose,
           parent: parent,
           opts: [permutation: permutation, ignore_batch?: ignore_batch?],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    permutation =
      if ignore_batch? do
        [0 | Enum.map(permutation, &(&1 + 1))]
      else
        permutation
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
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

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
         inputs,
         mode
       ) do
    {exprs, cache} =
      Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map, params, inputs, mode))

    inps = Enum.map(exprs, &Nx.as_type(&1, compute))
    res = Nx.as_type(apply(Nx, :concatenate, [inps, [axis: axis]]), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :cond,
           parent: parents,
           opts: [cond: cond_fn],
           policy: %{compute: compute, output: output}
         },
         cache,
         input_map,
         params,
         inputs,
         mode
       ) do
    {exprs, cache} =
      Enum.map_reduce(parents, cache, &to_predict_fun(&1, &2, input_map, params, inputs, mode))

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

    res =
      Axon.Layers.cond(
        Nx.all?(cond_expr),
        Nx.as_type(true_expr, compute),
        Nx.as_type(false_expr, compute)
      )

    {Nx.as_type(res, output), Map.put(cache, id, res)}
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
         inputs,
         mode
       ) do
    {res, cache} = to_predict_fun(parent, cache, input_map, params, inputs, mode)

    res = Nx.as_type(nx_fun.(Nx.as_type(res, compute)), output)

    {res, Map.put(cache, id, res)}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :constant, opts: [value: tensor], policy: %{output: output}},
         cache,
         _,
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
         inputs,
         _
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
         hidden_state_shape,
         mode
       ) do
    case hidden_state do
      {%Axon{} = c, %Axon{} = h} ->
        {c_res, cache} = to_predict_fun(c, cache, input_map, params, inputs, mode)
        {h_res, cache} = to_predict_fun(h, cache, input_map, params, inputs, mode)
        {{c_res, h_res}, cache}

      {%Axon{} = c} ->
        {h_res, cache} = to_predict_fun(c, cache, input_map, params, inputs, mode)
        {{h_res}, cache}

      %Axon{} = x ->
        {h_res, cache} = to_predict_fun(x, cache, input_map, params, inputs, mode)
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

  ## Helpers

  defp layer_param(layer_params, key, param_name, compute) do
    %{name: p, frozen: frozen} = layer_params[key]
    Nx.as_type(maybe_freeze(param_name[p], frozen), compute)
  end
end
