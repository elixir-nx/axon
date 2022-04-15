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
           parent: parents,
           op: op,
           name: name_fn,
           params: params,
           policy: %{params: dtype},
           hooks: hooks
         },
         cache_and_counts
       )
       when is_list(parents) do
    {cache, op_counts} = Enum.reduce(parents, cache_and_counts, &to_init_fun/2)

    case cache do
      %{^id => _} ->
        {cache, op_counts}

      %{} ->
        if Enum.empty?(params) do
          {cache, op_counts}
        else
          layer_params =
            Enum.reduce(params, %{}, fn {_, param}, layer_params ->
              %{name: name, shape: shape, initializer: initializer} = param
              fun = apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]])
              Map.put(layer_params, name, fun)
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

  defp to_init_fun(
         %Axon{
           id: id,
           parent: parent,
           op: op,
           name: name_fn,
           params: params,
           opts: opts,
           policy: %{params: dtype},
           hooks: hooks
         },
         cache_and_counts
       ) do
    {cache, op_counts} =
      case parent do
        %Axon{} = parent ->
          to_init_fun(parent, cache_and_counts)

        nil ->
          cache_and_counts

        parents ->
          deep_reduce(parents, cache_and_counts, &to_init_fun/2)
      end

    {cache, op_counts} =
      case opts[:hidden_state] do
        state when is_tuple(state) ->
          state
          |> Tuple.to_list()
          |> Enum.reduce({cache, op_counts}, &to_init_fun/2)

        nil ->
          {cache, op_counts}
      end

    case cache do
      %{^id => _} ->
        {cache, op_counts}

      %{} ->
        if Enum.empty?(params) do
          {cache, op_counts}
        else
          layer_params =
            Enum.reduce(params, %{}, fn {_, param}, layer_params ->
              %{name: name, shape: shape, initializer: initializer} = param
              fun = apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]])
              Map.put(layer_params, name, fun)
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

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(graph, args, opts) do
    {mode, opts} = Keyword.pop(opts, :mode, :inference)
    fun = compile_predict(graph, mode)
    Nx.Defn.jit_or_apply(fun, args, opts)
  end

  defp compile_predict(%Axon{} = graph, mode) do
    predict_fn = fn params, inputs ->
      {pred_expr, {state_expr, _, _}} =
        to_predict_fun(graph, {%{}, %{}, %{}}, params, inputs, mode)

      case mode do
        :train ->
          %{prediction: pred_expr, state: state_expr}

        :inference ->
          pred_expr
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

  defp to_predict_fun(%{id: id} = graph, {state, cache, op_counts}, params, inputs, mode) do
    case cache do
      %{^id => res} ->
        {res, {state, cache, op_counts}}

      %{} ->
        try do
          recur_predict_fun(graph, {state, cache, op_counts}, params, inputs, mode)
        rescue
          e -> reraise Axon.CompilerError.exception(graph: graph, exception: e), __STACKTRACE__
        end
    end
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :container, parent: parents},
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {exprs, {state, cache, op_counts}} =
      deep_map_reduce(parents, cache_and_counts, &to_predict_fun(&1, &2, params, inputs, mode))

    op_counts = Map.update(op_counts, :container, 1, fn x -> x + 1 end)
    cache = Map.put(cache, id, exprs)

    {exprs, {state, cache, op_counts}}
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
         params,
         inputs,
         mode
       )
       when is_function(op) and is_list(parents) do
    {exprs, {state, cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, params, inputs, mode)
      )

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    inp_params =
      Map.new(layer_params, fn {k, %{name: v, frozen: frz}} ->
        {k, maybe_freeze(params[name][v], frz)}
      end)

    param_arg =
      case inp_params do
        %{} ->
          []

        inp_params ->
          [inp_params]
      end

    res =
      exprs
      |> Enum.map(&Nx.as_type(&1, compute))
      |> then(&apply(op, &1 ++ param_arg ++ opts))
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)
      |> Nx.as_type(output)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: parent,
           params: layer_params,
           opts: opts,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       )
       when is_function(op) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    inp_params =
      Map.new(layer_params, fn {k, %{name: v, frozen: frz}} ->
        {k, maybe_freeze(params[name][v], frz)}
      end)

    param_arg =
      case inp_params do
        %{} ->
          []

        inp_params ->
          [inp_params]
      end

    input =
      case res do
        %Nx.Tensor{} = t ->
          Nx.as_type(t, compute)

        res ->
          deep_new(res, &Nx.as_type(&1, compute))
      end

    res =
      op
      |> apply([input] ++ param_arg ++ [opts])
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
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
           opts: opts,
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       )
       when op in @activation_layers do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    input =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)

    args =
      case opts do
        [] ->
          [input]

        [_ | _] ->
          [input, opts]
      end

    res =
      args
      |> then(&apply(Axon.Activations, op, &1))
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  ## Linear Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :dense,
           name: name_fn,
           parent: parent,
           params: layer_params,
           policy: %{compute: compute, output: output},
           opts: [use_bias: use_bias],
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    name = name_fn.(:dense, op_counts)

    op_counts = Map.update(op_counts, :dense, 1, fn x -> x + 1 end)

    w = layer_param(layer_params, "kernel", params[name], compute)

    b =
      if use_bias do
        layer_param(layer_params, "bias", params[name], compute)
      else
        Nx.tensor(0.0, type: compute)
      end

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Axon.Layers.dense(w, b)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: :bilinear,
           parent: parents,
           params: layer_params,
           policy: %{compute: compute, output: output},
           opts: [use_bias: use_bias],
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {[res1, res2], {state, cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, params, inputs, mode)
      )

    name = name_fn.(:bilinear, op_counts)
    op_counts = Map.update(op_counts, :bilinear, 1, fn x -> x + 1 end)

    w = layer_param(layer_params, "kernel", params[name], compute)

    b =
      if use_bias do
        layer_param(layer_params, "bias", params[name], compute)
      else
        Nx.tensor(0.0, type: compute)
      end

    input1 = Nx.as_type(res1, compute)
    input2 = Nx.as_type(res2, compute)

    {input1_hooked, input2_hooked} = apply_hooks({input1, input2}, :pre_forward, mode, hooks)

    res =
      input1_hooked
      |> Axon.Layers.bilinear(input2_hooked, w, b)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  ## Sparse Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: :embedding,
           parent: parent,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    name = name_fn.(:embedding, op_counts)
    op_counts = Map.update(op_counts, :embedding, 1, fn x -> x + 1 end)

    w = layer_param(layer_params, "kernel", params[name], compute)

    res =
      res
      |> apply_hooks(:pre_forward, :inference, hooks)
      |> Axon.Layers.embedding(w)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, :inference, hooks)
      |> apply_hooks(:backward, :inference, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
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
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       )
       when op in @pooling_layers do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, :inference, hooks)
      |> then(&apply(Axon.Layers, op, [&1, opts]))
      |> Nx.as_type(output)
      |> apply_hooks(:forward, :inference, hooks)
      |> apply_hooks(:backward, :inference, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: op,
           parent: parent,
           opts: opts,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       )
       when op in @dropout_layers do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    case mode do
      :train ->
        out =
          res.prediction
          |> Nx.as_type(compute)
          |> apply_hooks(:pre_forward, :train, hooks)
          |> then(&apply(Axon.Layers, op, [&1, opts]))
          |> Nx.as_type(output)
          |> apply_hooks(:forward, :train, hooks)
          |> apply_hooks(:backward, :train, hooks)

        res = Map.update!(res, :prediction, fn _ -> out end)
        {res, {state, Map.put(cache, id, res), op_counts}}

      :inference ->
        # Skip dropout in inference mode
        res = Nx.as_type(res, output)
        {res, {state, Map.put(cache, id, res), op_counts}}
    end
  end

  ## Conv Layers

  @conv_layers [:conv, :conv_transpose, :depthwise_conv]

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: parent,
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       )
       when op in @conv_layers do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    k = layer_param(layer_params, "kernel", params[name], compute)

    b =
      if use_bias do
        layer_param(layer_params, "bias", params[name], compute)
      else
        Nx.tensor(0, type: compute)
      end

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> then(&apply(Axon.Layers, op, [&1, k, b, opts]))
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: :separable_conv2d,
           parent: parent,
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    name = name_fn.(:separable_conv2d, op_counts)
    op_counts = Map.update(op_counts, :separable_conv2d, 1, fn x -> x + 1 end)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

    k1 = layer_param(layer_params, "k1", params[name], compute)
    k2 = layer_param(layer_params, "k2", params[name], compute)

    {b1, b2} =
      if use_bias do
        {layer_param(layer_params, "b1", params[name], compute),
         layer_param(layer_params, "b2", params[name], compute)}
      else
        {Nx.tensor(0, type: compute), Nx.tensor(0, type: compute)}
      end

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Axon.Layers.separable_conv2d(k1, b1, k2, b2, opts)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: :separable_conv3d,
           parent: parent,
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    name = name_fn.(:separable_conv3d, op_counts)
    op_counts = Map.update(op_counts, :separable_conv3d, 1, fn x -> x + 1 end)

    {use_bias, opts} = Keyword.pop!(opts, :use_bias)

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
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Axon.Layers.separable_conv3d(k1, b1, k2, b2, k3, b3, opts)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  ## Normalization Layers

  @normalization_with_stats [:batch_norm, :instance_norm]

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: parent,
           opts: [epsilon: epsilon, channel_index: channel_index, momentum: momentum],
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       )
       when op in @normalization_with_stats do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    training? = mode == :train

    norm_opts = [
      epsilon: epsilon,
      channel_index: channel_index,
      momentum: momentum,
      training?: training?
    ]

    g = layer_param(layer_params, "gamma", params[name], compute)
    b = layer_param(layer_params, "beta", params[name], compute)
    mean = layer_param(layer_params, "mean", params[name], compute)
    var = layer_param(layer_params, "var", params[name], compute)

    case mode do
      :train ->
        {out, ra_mean, ra_var} =
          res
          |> Nx.as_type(compute)
          |> apply_hooks(:pre_forward, :train, hooks)
          |> then(&apply(Axon.Layers, op, [&1, g, b, mean, var, norm_opts]))
          |> then(fn {y, m, v} -> {Nx.as_type(y, output), m, v} end)
          |> apply_hooks(:forward, :train, hooks)
          |> apply_hooks(:backward, :train, hooks)

        res = Nx.as_type(out, output)
        state = Map.put(state, name, %{"mean" => ra_mean, "var" => ra_var})

        {res, {state, Map.put(cache, id, res), op_counts}}

      :inference ->
        res =
          res
          |> Nx.as_type(compute)
          |> apply_hooks(:pre_forward, :inference, hooks)
          |> then(&apply(Axon.Layers, op, [&1, g, b, mean, var, norm_opts]))
          |> Nx.as_type(output)
          |> apply_hooks(:forward, :inference, hooks)
          |> apply_hooks(:backward, :inference, hooks)

        {res, {state, Map.put(cache, id, res), op_counts}}
    end
  end

  @normalization_layers [:layer_norm, :group_norm]

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: parent,
           opts: opts,
           params: layer_params,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       )
       when op in @normalization_layers do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    g = layer_param(layer_params, "gamma", params[name], compute)
    b = layer_param(layer_params, "beta", params[name], compute)

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> then(&apply(Axon.Layers, op, [&1, g, b, opts]))
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  ## Recurrent Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
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
           ],
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, cache_and_counts} = to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    {{h, c}, {state, cache, op_counts}} =
      to_hidden_state(
        hidden_state,
        res,
        cache_and_counts,
        params,
        inputs,
        2,
        recurrent_initializer,
        hidden_state_shape,
        mode
      )

    name = name_fn.(:lstm, op_counts)
    op_counts = Map.update(op_counts, :lstm, 1, fn x -> x + 1 end)

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

    # TODO: Should these be hooked together? Not at all?
    {input, carry} = apply_hooks({res, {h, c}}, :pre_forward, state, hooks)

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
    res = apply_hooks(res, :forward, mode, hooks)
    res = apply_hooks(res, :backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
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
           ],
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, cache_and_counts} = to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    {{h, c}, {state, cache, op_counts}} =
      to_hidden_state(
        hidden_state,
        res,
        cache_and_counts,
        params,
        inputs,
        2,
        recurrent_initializer,
        hidden_state_shape,
        mode
      )

    name = name_fn.(:conv_lstm, op_counts)
    op_counts = Map.update(op_counts, :conv_lstm, 1, fn x -> x + 1 end)

    input_kernel = {layer_param(layer_params, "wi", params[name], compute)}
    hidden_kernel = {layer_param(layer_params, "wh", params[name], compute)}
    bias = {layer_param(layer_params, "b", params[name], compute)}

    input = Nx.as_type(res, compute)
    carry = {Nx.as_type(h, compute), Nx.as_type(c, compute)}

    # TODO: Should these be hooked together? Not at all?
    {input, carry} = apply_hooks({input, carry}, :pre_forward, mode, hooks)

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
    res = apply_hooks(res, :forward, mode, hooks)
    res = apply_hooks(res, :backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
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
           ],
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, cache_and_counts} = to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    {{h}, {state, cache, op_counts}} =
      to_hidden_state(
        hidden_state,
        res,
        cache_and_counts,
        params,
        inputs,
        1,
        recurrent_initializer,
        hidden_state_shape,
        mode
      )

    name = name_fn.(:gru, op_counts)
    op_counts = Map.update(op_counts, :gru, 1, fn x -> x + 1 end)

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

    input = Nx.as_type(res, compute)
    carry = {Nx.as_type(h, compute)}

    # TODO: Should these be hooked together? Not at all?
    {input, carry} = apply_hooks({input, carry}, :pre_forward, mode, hooks)

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
    res = apply_hooks(res, :forward, mode, hooks)
    res = apply_hooks(res, :backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
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
         params,
         inputs,
         mode
       )
       when op in @element_wise_layers do
    {[expr | exprs], {state, cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, params, inputs, mode)
      )

    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    [expr | exprs] =
      [expr | exprs]
      |> List.to_tuple()
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Tuple.to_list()

    res =
      Enum.reduce(exprs, expr, fn next_expr, acc ->
        input = Nx.as_type(next_expr, compute)
        acc = Nx.as_type(acc, compute)
        Nx.as_type(apply(Nx, op, [acc, input]), output)
      end)

    res = apply_hooks(res, :forward, mode, hooks)
    res = apply_hooks(res, :backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  ## Shape Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :flatten,
           parent: parent,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, :flatten, 1, fn x -> x + 1 end)

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Axon.Layers.flatten()
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :reshape,
           parent: parent,
           output_shape: output_shape,
           policy: %{compute: compute, output: output},
           opts: [constant: is_constant_reshape?],
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, :reshape, 1, fn x -> x + 1 end)

    inp =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, :inference, hooks)

    reshape_shape =
      if is_constant_reshape? do
        output_shape
      else
        put_elem(output_shape, 0, elem(Nx.shape(inp), 0))
      end

    res =
      inp
      |> Nx.reshape(reshape_shape)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, :inference, hooks)
      |> apply_hooks(:backward, :inference, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :resize,
           parent: parent,
           policy: %{compute: compute, output: output},
           opts: opts,
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, :resize, 1, fn x -> x + 1 end)

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Axon.Layers.resize(opts)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :transpose,
           parent: parent,
           opts: [permutation: permutation, ignore_batch?: ignore_batch?],
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, :transpose, 1, fn x -> x + 1 end)

    permutation =
      if ignore_batch? do
        [0 | Enum.map(permutation, &(&1 + 1))]
      else
        permutation
      end

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Nx.transpose(axes: permutation)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :pad,
           parent: parent,
           opts: [padding_config: config, value: value],
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, :pad, 1, fn x -> x + 1 end)

    config = [{0, 0, 0}, {0, 0, 0} | Enum.map(config, fn {x, y} -> {x, y, 0} end)]

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Nx.pad(value, config)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
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
         params,
         inputs,
         mode
       ) do
    {exprs, {state, cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, params, inputs, mode)
      )

    op_counts = Map.update(op_counts, :concatenate, 1, fn x -> x + 1 end)

    inps = Enum.map(exprs, &Nx.as_type(&1, compute))

    inps =
      inps
      |> List.to_tuple()
      |> apply_hooks(:pre_forward, mode, hooks)
      |> Tuple.to_list()

    res =
      inps
      |> Nx.concatenate(axis: axis)
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
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
         params,
         inputs,
         mode
       ) do
    {exprs, {state, cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, params, inputs, mode)
      )

    op_counts = Map.update(op_counts, :cond, 1, fn x -> x + 1 end)

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
        Nx.as_type(on_true, compute),
        Nx.as_type(on_false, compute)
      )

    res = Nx.as_type(res, output)
    res = apply_hooks(res, :forward, mode, hooks)
    res = apply_hooks(res, :backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  ## Special Layers

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :nx,
           parent: parent,
           opts: [fun: nx_fun],
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         params,
         inputs,
         mode
       ) do
    {res, {state, cache, op_counts}} =
      to_predict_fun(parent, cache_and_counts, params, inputs, mode)

    op_counts = Map.update(op_counts, :nx, 1, fn x -> x + 1 end)

    res =
      res
      |> Nx.as_type(compute)
      |> apply_hooks(:pre_forward, mode, hooks)
      |> nx_fun.()
      |> Nx.as_type(output)
      |> apply_hooks(:forward, mode, hooks)
      |> apply_hooks(:backward, mode, hooks)

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :constant, opts: [value: tensor], policy: %{output: output}},
         {state, cache, op_counts},
         _,
         _,
         _
       ) do
    out = Nx.as_type(tensor, output)
    op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)

    {out, {state, Map.put(cache, id, out), op_counts}}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :input, output_shape: shape, hooks: hooks, name: name_fn},
         {state, cache, op_counts},
         _,
         inputs,
         mode
       ) do
    name = name_fn.(:input, op_counts)
    op_counts = Map.update(op_counts, :input, 1, fn x -> x + 1 end)

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

    {res, {state, Map.put(cache, id, res), op_counts}}
  end

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

  defp to_hidden_state(
         hidden_state,
         input,
         cache_and_counts,
         params,
         inputs,
         num_carry,
         recurrent_initializer,
         hidden_state_shape,
         mode
       ) do
    case hidden_state do
      {%Axon{} = c, %Axon{} = h} ->
        {c_res, cache_and_counts} = to_predict_fun(c, cache_and_counts, params, inputs, mode)

        {h_res, cache_and_counts} = to_predict_fun(h, cache_and_counts, params, inputs, mode)

        {{c_res, h_res}, cache_and_counts}

      {%Axon{} = c} ->
        {h_res, cache_and_counts} = to_predict_fun(c, cache_and_counts, params, inputs, mode)

        {{h_res}, cache_and_counts}

      %Axon{} = x ->
        {h_res, cache_and_counts} = to_predict_fun(x, cache_and_counts, params, inputs, mode)

        {h_res, cache_and_counts}

      nil ->
        shape = put_elem(hidden_state_shape, 0, elem(Nx.shape(input), 0))

        h_res =
          for _ <- 1..num_carry,
              do: apply(Axon.Initializers, recurrent_initializer, [[shape: shape]])

        res = List.to_tuple(h_res)

        res = if mode == :train, do: %{prediction: res, state: %{}}, else: res
        {res, cache_and_counts}
    end
  end

  ## Helpers

  defp layer_param(layer_params, key, param_name, compute) do
    %{name: p, frozen: frozen} = layer_params[key]
    Nx.as_type(maybe_freeze(param_name[p], frozen), compute)
  end
end
