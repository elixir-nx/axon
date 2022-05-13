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
            Enum.reduce(params, %{}, fn param, layer_params ->
              init_param(param, layer_params, dtype)
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

  defp init_param(param, layer_params, dtype) do
    case param do
      %{name: name, shape: shape, initializer: initializer} ->
        fun = apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]])
        Map.put(layer_params, name, fun)
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
      case mode do
        :train ->
          {pred_expr, {state_expr, _}} = cache[root_id].(params, inputs, %{}, cache, %{})
          %{prediction: pred_expr, state: state_expr}

        :inference ->
          {pred_expr, _} = cache[root_id].(params, inputs, %{}, cache, %{})
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

  defp to_predict_fun(%{id: id} = graph, {cache, op_counts}, mode) do
    case cache do
      %{^id => _} ->
        {id, {cache, op_counts}}

      %{} ->
        recur_predict_fun(graph, {cache, op_counts}, mode)
    end
  end

  defp call_cache(parent_id, params, inputs, state, cache, result_cache) do
    key = {:cache, parent_id}

    case result_cache do
      %{^key => {expr, state}} ->
        {expr, {state, result_cache}}

      %{} ->
        {expr, {state, result_cache}} =
          cache[parent_id].(params, inputs, state, cache, result_cache)

        {expr, {state, Map.put(result_cache, key, {expr, state})}}
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

    fun = fn params, inputs, state, cache, result_cache ->
      deep_map_reduce(parent_ids, {state, result_cache}, fn parent_id, {state, result_cache} ->
        call_cache(parent_id, params, inputs, state, cache, result_cache)
      end)
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Custom Layers

  @axon_layers [:dense, :bilinear, :conv, :depthwise_conv, :conv_transpose] ++
                   [:separable_conv2d, :separable_conv3d, :embedding, :bias] ++
                   [:max_pool, :avg_pool, :adaptive_avg_pool] ++
                   [:adaptive_max_pool, :adaptive_lp_pool, :lp_pool] ++
                   [:global_lp_pool, :global_max_pool, :global_avg_pool] ++
                   [:batch_norm, :instance_norm, :layer_norm, :group_norm] ++
                   [:resize, :flatten, :reshape, :transpose, :pad] ++
                   [:dropout, :spatial_dropout, :alpha_dropout, :feature_alpha_dropout] ++
                   [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                   [:leaky_relu, :linear, :log_sigmoid, :mish, :relu, :relu6] ++
                   [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh] ++
                   [:log_softmax] ++
                   [:cond, :add, :multiply, :subtract]

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
       when is_function(op) or op in @axon_layers and is_list(parents) do
    {parent_ids, {cache, op_counts}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_predict_fun(&1, &2, mode)
      )

    {_, opts} = Keyword.pop(opts, :layer_op)

    name = name_fn.(op, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache, result_cache ->
      {res, {state, result_cache}} =
        parent_ids
        |> Enum.map_reduce({state, result_cache}, fn parent_id, {state, result_cache} ->
          call_cache(parent_id, params, inputs, state, cache, result_cache)
        end)

      inp_params =
        Enum.map(layer_params, fn %{name: v, frozen: frz} ->
          safe_as_type(maybe_freeze(params[name][v], frz), compute)
        end)

      inputs =
        res
        |> Enum.map(&safe_as_type(&1, compute))
        |> Enum.map(&apply_hooks(&1, :pre_forward, mode, hooks))

      tensor_inputs = inputs ++ inp_params
      args = tensor_inputs ++ [Keyword.put(opts, :mode, mode)]

      out =
        args
        |> then(fn args ->
          case op do
            op when is_function(op) ->
              apply(op, args)

            op when op in @axon_layers ->
              apply(Axon.Layers, op, args)
          end
        end)

      {out, state} =
        case out do
          {out, state} ->
            new_out =
              out
              |> apply_hooks(:forward, mode, hooks)
              |> apply_hooks(:backward, mode, hooks)
              |> safe_as_type(output)
            new_state = Map.put(state, name, state)
            {new_out, new_state}

          out ->
            new_out =
              out
              |> apply_hooks(:forward, mode, hooks)
              |> apply_hooks(:backward, mode, hooks)
              |> safe_as_type(output)
            {new_out, state}
        end

      {out, {state, result_cache}}
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

    fun = fn params, inputs, state, cache, result_cache ->
      {input, {state, result_cache}} =
        call_cache(input_id, params, inputs, state, cache, result_cache)

      {hidden_state, {state, result_cache}} =
        call_cache(hidden_state_id, params, inputs, state, cache, result_cache)

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

      {res, {state, result_cache}}
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

    fun = fn params, inputs, state, cache, result_cache ->
      {exprs, {state, result_cache}} =
        Enum.map_reduce(parent_ids, {state, result_cache}, fn parent_id, {state, result_cache} ->
          call_cache(parent_id, params, inputs, state, cache, result_cache)
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

      {res, {state, result_cache}}
    end

    {id, {Map.put(cache, id, fun), op_counts}}
  end

  ## Special Layers

  defp recur_predict_fun(
         %Axon{id: id, op: :constant, opts: [value: tensor], policy: %{output: output}},
         {cache, op_counts},
         _
       ) do
    fun = fn _params, _inputs, state, _cache, result_cache ->
      out = safe_as_type(tensor, output)
      {out, {state, result_cache}}
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

    fun = fn _params, inputs, state, _cache, result_cache ->
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

      {res, {state, result_cache}}
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
