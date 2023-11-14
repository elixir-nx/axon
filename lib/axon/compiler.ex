defmodule Axon.CompileError do
  defexception [:exception, :name, :mfa, :layer_stacktrace, :compile_stacktrace]

  def message(exception) do
    {module, fun, arity} = exception.mfa
    formatted_mfa = Exception.format_mfa(module, fun, arity)
    formatted_msg = Exception.format(:error, exception.exception, exception.compile_stacktrace)

    layer_info =
      if exception.layer_stacktrace != [] do
        """

        The layer was defined at:

        #{Exception.format_stacktrace(exception.layer_stacktrace)}
        """
      else
        """

        (pass debug: true to build/compile see where the layer was defined)

        """
      end

    """
    exception found when compiling layer #{formatted_mfa} named #{exception.name}:

        #{indent(formatted_msg)}\

    #{layer_info}\

    Compiling of the model was initiated at:
    """
  end

  defp indent(msg), do: String.replace(msg, "\n", "\n    ")
end

defmodule Axon.Compiler do
  @moduledoc false
  require Logger

  import Axon.Shared
  alias Axon.StatefulOutput

  ## Init JIT Compilation

  @doc false
  def build(%Axon{output: id, nodes: nodes}, opts) do
    debug? = Keyword.get(opts, :debug, false)
    mode = Keyword.get(opts, :mode, :inference)
    seed = Keyword.get_lazy(opts, :seed, fn -> :erlang.system_time() end)
    config = %{mode: mode, debug?: debug?}

    {time, {root_id, {cache, _op_counts, _block_cache}}} =
      :timer.tc(fn ->
        to_model_funs(id, nodes, {%{}, %{}, %{}}, config)
      end)

    if debug? do
      Logger.debug("Axon finished graph traversal in #{us_to_ms(time)}ms")
    end

    predict_cache =
      Map.new(cache, fn {_, {int_id, %{predict: predict}}} -> {int_id, %{predict: predict}} end)

    predict_fun = fn params, inputs ->
      {:current_stacktrace, [_process_info, _fn | stacktrace]} =
        Process.info(self(), :current_stacktrace)

      {time, result} =
        :timer.tc(fn ->
          case mode do
            :train ->
              {pred_expr, {state_expr, _}} =
                predict_cache[root_id][:predict].(
                  params,
                  inputs,
                  %{},
                  predict_cache,
                  %{},
                  stacktrace
                )

              %{prediction: pred_expr, state: state_expr}

            :inference ->
              {pred_expr, _} =
                predict_cache[root_id][:predict].(
                  params,
                  inputs,
                  %{},
                  predict_cache,
                  %{},
                  stacktrace
                )

              pred_expr
          end
        end)

      if debug? do
        Logger.debug("Axon finished predict expression generation in #{us_to_ms(time)}ms")
      end

      with %Axon.None{} <- result do
        raise ArgumentError,
              "the compiled model will always result in %Axon.None{}." <>
                " This most likely means you specified optional output and " <>
                " did not handle the case when it is missing"
      end

      result
    end

    init_cache = Map.new(cache, fn {_, {int_id, funs}} -> {int_id, funs} end)

    init_fun = fn template, init_params ->
      {:current_stacktrace, [_process_info, _fn | stacktrace]} =
        Process.info(self(), :current_stacktrace)

      {time, params} =
        :timer.tc(fn ->
          param_keys = get_keys(nodes, seed)

          {_, {params, _}} =
            init_cache[root_id][:init].(template, init_cache, %{}, stacktrace, param_keys)

          params
        end)

      params = merge_params!(params, init_params)

      if debug? do
        Logger.debug("Axon finished init expression generation in #{us_to_ms(time)}ms")
      end

      params
    end

    {init_fun, predict_fun}
  end

  defp get_keys(nodes, seed) do
    {ids_and_data, _op_counts} =
      Enum.reduce(nodes, {[], %{}}, fn
        {_, %Axon.Node{id: id, op: op, name: name_fn, parameters: params}}, {keys, op_counts} ->
          name = name_fn.(op, op_counts)
          op_counts = Map.update(op_counts, op, 1, &(&1 + 1))

          keys =
            Enum.reduce(params, keys, fn
              %Axon.Parameter{name: param_name, initializer: fun}, keys ->
                {:arity, arity} = Function.info(fun, :arity)

                cond do
                  arity == 2 ->
                    keys

                  arity == 3 ->
                    <<data::unsigned-size(32), _rest::binary>> =
                      :erlang.md5(name <> "." <> param_name)

                    [{{id, param_name}, data} | keys]

                  true ->
                    raise ArgumentError, "bad initializer arity"
                end
            end)

          {keys, op_counts}
      end)

    {ids, data} = Enum.unzip(ids_and_data)

    case ids do
      [] ->
        %{}

      [_ | _] = ids ->
        key = Nx.Random.key(seed)

        keys_tensor =
          data
          |> Nx.tensor(type: :u32)
          |> then(&Nx.Random.fold_in(key, &1))

        {keys, _} =
          Enum.reduce(ids, {%{}, 0}, fn {layer_id, param_name}, {acc, i} ->
            key = keys_tensor[i]
            acc = Map.update(acc, layer_id, %{param_name => key}, &Map.put(&1, param_name, key))
            {acc, i + 1}
          end)

        keys
    end
  end

  defp merge_params!(params, init_params) do
    Enum.reduce(init_params, params, fn {key, value}, params ->
      case params do
        %{^key => %{} = nested} when not is_struct(nested) ->
          %{params | key => merge_params!(nested, value)}

        %{^key => template} ->
          %{params | key => merge_type(key, template, value)}

        _ ->
          Logger.warning("found unexpected key in the initial parameters map: #{inspect(key)}")
          params
      end
    end)
  end

  defp merge_type(key, template, value) do
    if Nx.type(template) != Nx.type(value) do
      Logger.warning(
        "initial type for parameter #{key} does not match policy," <>
          " consider using Axon.MixedPrecision.cast before passing" <>
          " initial state to model initialization function to avoid" <>
          " type casts"
      )
    end

    Nx.as_type(value, Nx.type(template))
  end

  def compile(graph, _opts) do
    raise ArgumentError,
          "attempting to compile model functions from" <>
            " an unrecognized graph #{inspect(graph)}, if you" <>
            " are attempting to compile a model with a container" <>
            " output, use `Axon.container`"
  end

  defp to_model_funs(id, nodes, {cache, op_counts, block_cache}, config) do
    case cache do
      %{^id => {int_id, _}} ->
        {int_id, {cache, op_counts, block_cache}}

      %{} ->
        {id, model_funs, cache, op_counts, block_cache} =
          recur_model_funs(nodes[id], nodes, {cache, op_counts, block_cache}, config)

        int_id = map_size(cache)
        {int_id, {Map.put(cache, id, {int_id, model_funs}), op_counts, block_cache}}
    end
  end

  defp call_predict_cache(parent_id, params, inputs, state, cache, result_cache, fn_stacktrace) do
    key = {:predict_cache, parent_id}

    case result_cache do
      %{^key => {expr, state}} ->
        {expr, {state, result_cache}}

      %{} ->
        {expr, {state, result_cache}} =
          cache[parent_id][:predict].(params, inputs, state, cache, result_cache, fn_stacktrace)

        {expr, {state, Map.put(result_cache, key, {expr, state})}}
    end
  end

  defp call_init_cache(parent_id, template, params, cache, result_cache, fn_stacktrace, keys) do
    key = {:init_cache, parent_id}

    {parent_shape, {parent_params, result_cache}} =
      case result_cache do
        %{^key => {parent_shape, parent_params}} ->
          {parent_shape, {parent_params, result_cache}}

        %{} ->
          {parent_shape, {parent_params, result_cache}} =
            cache[parent_id][:init].(template, cache, result_cache, fn_stacktrace, keys)

          {parent_shape,
           {parent_params, Map.put(result_cache, key, {parent_shape, parent_params})}}
      end

    {parent_shape, {Map.merge(parent_params, params), result_cache}}
  end

  # If the node is ignored for the current mode, we pass through and recur next
  defp recur_model_funs(
         %Axon.Node{id: id, mode: node_mode, parent: [parent | _]},
         nodes,
         {cache, op_counts, block_cache},
         config
       )
       when node_mode != :both and node_mode != config.mode do
    {parent_id, {cache, op_counts, block_cache}} =
      to_model_funs(parent, nodes, {cache, op_counts, block_cache}, config)

    predict_fun = fn params, inputs, state, cache, result_cache, fn_stacktrace ->
      call_predict_cache(parent_id, params, inputs, state, cache, result_cache, fn_stacktrace)
    end

    init_fun = fn template, cache, result_cache, fn_stacktrace, keys ->
      call_init_cache(parent_id, template, %{}, cache, result_cache, fn_stacktrace, keys)
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache}
  end

  defp recur_model_funs(
         %Axon.Node{id: id, op: :constant, opts: [value: tensor], policy: %{output: output}},
         _nodes,
         {cache, op_counts, block_cache},
         _
       ) do
    op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)
    tensor = Nx.backend_copy(tensor, Nx.BinaryBackend)

    predict_fun = fn _params, _inputs, state, _cache, result_cache, _fn_stacktrace ->
      out = safe_as_type(tensor, output)
      {out, {state, result_cache}}
    end

    init_fun = fn _template, _cache, result_cache, _fn_stacktrace, _keys ->
      {Nx.shape(tensor), {%{}, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache}
  end

  defp recur_model_funs(
         %Axon.Node{
           id: id,
           op: :input,
           hooks: hooks,
           name: name_fn,
           opts: [shape: _input_shape, optional: optional?]
         },
         _nodes,
         {cache, op_counts, block_cache},
         %{mode: mode}
       ) do
    name = name_fn.(:input, op_counts)
    op_counts = Map.update(op_counts, :input, 1, fn x -> x + 1 end)

    predict_fun = fn _params, inputs, state, _cache, result_cache, _fn_stacktrace ->
      value = get_input(inputs, name, optional?)

      # TODO: Add this back in
      # validate_input_shape!(value, shape)

      res =
        value
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, {state, result_cache}}
    end

    init_fun = fn template, _cache, result_cache, _fn_stacktrace, _keys ->
      input = get_input(template, name, optional?)
      {safe_shape(input), {%{}, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache}
  end

  defp recur_model_funs(
         %Axon.Node{id: id, op: :optional, parent: [parent]},
         nodes,
         {cache, op_counts, block_cache},
         config
       ) do
    {parent_id, {cache, op_counts, block_cache}} =
      to_model_funs(parent, nodes, {cache, op_counts, block_cache}, config)

    predict_fun = fn params, inputs, state, cache, result_cache, fn_stacktrace ->
      {out, {state, result_cache}} =
        call_predict_cache(parent_id, params, inputs, state, cache, result_cache, fn_stacktrace)

      out = with %Axon.None{} <- out, do: %Axon.None{__propagate__: false}

      {out, {state, result_cache}}
    end

    init_fun = fn template, cache, result_cache, fn_stacktrace, keys ->
      {out, {params, result_cache}} =
        call_init_cache(parent_id, template, %{}, cache, result_cache, fn_stacktrace, keys)

      out = with %Axon.None{} <- out, do: %Axon.None{__propagate__: false}

      {out, {params, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache}
  end

  defp recur_model_funs(
         %Axon.Node{id: id, op: :container, parent: [parents]},
         nodes,
         cache_and_counts,
         config
       ) do
    {parent_ids, {cache, op_counts, block_cache}} =
      deep_map_reduce(parents, cache_and_counts, &to_model_funs(&1, nodes, &2, config))

    op_counts = Map.update(op_counts, :container, 1, fn x -> x + 1 end)

    predict_fun = fn params, inputs, state, cache, result_cache, fn_stacktrace ->
      {input, {state, result_cache, none?}} =
        deep_map_reduce(
          parent_ids,
          {state, result_cache, false},
          fn parent_id, {state, result_cache, none?} ->
            {input, {state, result_cache}} =
              call_predict_cache(
                parent_id,
                params,
                inputs,
                state,
                cache,
                result_cache,
                fn_stacktrace
              )

            none? = none? or propagating_none?(input)
            {input, {state, result_cache, none?}}
          end
        )

      input = if none?, do: %Axon.None{}, else: input

      {input, {state, result_cache}}
    end

    init_fun = fn template, cache, result_cache, fn_stacktrace, keys ->
      {parent_shape, {parent_params, result_cache, none?}} =
        deep_map_reduce(parent_ids, {%{}, result_cache, false}, fn
          parent_id, {params, result_cache, none?} ->
            {parent_shape, {params, result_cache}} =
              call_init_cache(
                parent_id,
                template,
                params,
                cache,
                result_cache,
                fn_stacktrace,
                keys
              )

            none? = none? or propagating_none?(parent_shape)
            {parent_shape, {params, result_cache, none?}}
        end)

      parent_shape = if none?, do: %Axon.None{}, else: parent_shape

      {parent_shape, {parent_params, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache}
  end

  defp recur_model_funs(
         %Axon.Node{
           id: id,
           op: :block,
           parent: [parent],
           opts: [block_fun: block_fun, block_id: block_id],
           name: name_fn
         },
         nodes,
         cache_and_counts,
         config
       ) do
    {[parent_id], {cache, op_counts, block_cache}} =
      Enum.map_reduce(
        [parent],
        cache_and_counts,
        &to_model_funs(&1, nodes, &2, config)
      )

    {{block_init_fun, block_predict_fun}, block_name, block_cache, op_counts} =
      case block_cache do
        %{^block_id => {funs, name}} = block_cache ->
          {funs, name, block_cache, op_counts}

        %{} ->
          funs = build(block_fun.(Axon.input("subgraph")), debug?: config.debug?)
          name = name_fn.(:block, op_counts)
          op_counts = Map.update(op_counts, :block, 1, fn x -> x + 1 end)
          {funs, name, Map.put(block_cache, block_id, {funs, name}), op_counts}
      end

    predict_fun = fn params, inputs, state, cache, result_cache, fn_stacktrace ->
      # Recurse graph inputs and invoke cache to get parent results,
      # state, and result_cache and then apply dtype policy and hooks
      # to each input
      {[layer_input], {state, result_cache, none?}} =
        Enum.map_reduce(
          [parent_id],
          {state, result_cache, false},
          fn parent_id, {state, result_cache, none?} ->
            {layer_input, {state, result_cache}} =
              call_predict_cache(
                parent_id,
                params,
                inputs,
                state,
                cache,
                result_cache,
                fn_stacktrace
              )

            none? = none? or propagating_none?(layer_input)

            {layer_input, {state, result_cache, none?}}
          end
        )

      if none? do
        {%Axon.None{}, {state, result_cache}}
      else
        block_params = params[block_name] || %{}
        result = apply(block_predict_fun, [block_params, layer_input])

        {out_result, out_state} =
          case result do
            # Make sure the none is non-propagating
            %Axon.None{} -> %Axon.None{}
            %{prediction: pred_expr, state: state_expr} -> {pred_expr, state_expr}
            result -> {result, %{}}
          end

        state =
          if map_size(out_state) == 0 do
            state
          else
            Map.put(state, block_name, out_state)
          end

        {out_result, {state, result_cache}}
      end
    end

    init_fun = fn template, cache, result_cache, fn_stacktrace, keys ->
      {[parent_shape], {parent_params, result_cache, none?}} =
        Enum.map_reduce([parent_id], {%{}, result_cache, false}, fn
          parent_id, {params, result_cache, none?} ->
            {parent_shape, {params, result_cache}} =
              call_init_cache(
                parent_id,
                template,
                params,
                cache,
                result_cache,
                fn_stacktrace,
                keys
              )

            none? = none? or propagating_none?(parent_shape)
            {parent_shape, {params, result_cache, none?}}
        end)

      if none? do
        {%Axon.None{}, {parent_params, result_cache}}
      else
        template = Nx.broadcast(0.0, parent_shape)
        block_params = apply(block_init_fun, [template, %{}])

        params =
          if block_params == %{} do
            %{}
          else
            Map.put(parent_params, block_name, block_params)
          end

        {pred_expr, {_, result_cache}} =
          predict_fun.(params, template, %{}, cache, result_cache, fn_stacktrace)

        {safe_shape(pred_expr), {params, result_cache}}
      end
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache}
  end

  defp recur_model_funs(
         %Axon.Node{id: id, op: :namespace, name: name_fn, parent: [parent]},
         nodes,
         {cache, op_counts, block_cache},
         config
       ) do
    name = name_fn.(:namespace, op_counts)
    # To ensure that a namespace always has the same layer names,
    # we reset op_counts, input layers always belong to the global
    # namespace, so we include those regardless
    input_count = op_counts[:input] || 0
    namespace_op_counts = %{input: input_count}

    # All of the children of this namespace belong to it, so
    # we forward this name to the namespace, but everything after
    # it belongs to whatever namespace we're currently in
    {parent_id, {cache, namespace_op_counts, block_cache}} =
      to_model_funs(parent, nodes, {cache, namespace_op_counts, block_cache}, config)

    # Update the global op_count of input layers, since they
    # are a global operation regardless of where they are
    input_count = namespace_op_counts[:input] || 0
    op_counts = Map.put(op_counts, :input, input_count)

    # The function just returns the result of it's child,
    # or parent depending on how you view the tree
    predict_fun = fn params, inputs, state, cache, result_cache, fn_stacktrace ->
      # We're only concerned with this namespaces parameters, so we pair
      # down parameters first given the namespace
      namespace_params = params[name]

      # TODO: How should hooks be handled here?
      # TODO: I think we can actually handle parameter freezing and access
      # better here by only forwarding params[namespace] to the child function
      {out, {state, result_cache}} =
        call_predict_cache(
          parent_id,
          namespace_params,
          inputs,
          state,
          cache,
          result_cache,
          fn_stacktrace
        )

      state =
        if map_size(state) == 0 do
          state
        else
          %{name => state}
        end

      {out, {state, result_cache}}
    end

    init_fun = fn template, cache, result_cache, fn_stacktrace, keys ->
      {_parent_shape, {namespace_params, result_cache}} =
        call_init_cache(parent_id, template, %{}, cache, result_cache, fn_stacktrace, keys)

      params =
        if namespace_params == %{} do
          %{}
        else
          %{name => namespace_params}
        end

      {pred_expr, {_, result_cache}} =
        predict_fun.(params, template, %{}, cache, result_cache, fn_stacktrace)

      {safe_shape(pred_expr), {params, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache}
  end

  defp recur_model_funs(
         %Axon.Node{
           id: id,
           name: name_fn,
           op: op,
           parent: inputs,
           parameters: layer_params,
           args: args,
           opts: opts,
           policy: policy,
           hooks: hooks,
           op_name: op_name,
           stacktrace: stacktrace
         },
         nodes,
         cache_and_counts,
         %{mode: mode, debug?: debug?} = config
       )
       when (is_function(op) or is_atom(op)) and is_list(inputs) do
    # Traverse to accumulate cache and get parent_ids for
    # application within the function. We work only with
    # functions and IDs to avoid leaking entire graphs into
    # the closure
    {parent_ids, {cache, op_counts, block_cache}} =
      Enum.map_reduce(
        inputs,
        cache_and_counts,
        &to_model_funs(&1, nodes, &2, config)
      )

    # Names are computed lazily, so compute name from current
    # op and aggregate op_counts.
    name = name_fn.(op_name, op_counts)
    op_counts = Map.update(op_counts, op_name, 1, fn x -> x + 1 end)

    stacktrace = if debug?, do: stacktrace, else: []

    # Each model builds two functions: predict_fun and init_fun
    predict_fun =
      &layer_predict_fun(
        &1,
        &2,
        &3,
        &4,
        &5,
        &6,
        op,
        op_name,
        parent_ids,
        name,
        args,
        opts,
        policy,
        layer_params,
        hooks,
        mode,
        stacktrace
      )

    init_fun =
      &layer_init_fun(
        id,
        &1,
        &2,
        &3,
        &4,
        &5,
        parent_ids,
        name,
        predict_fun,
        layer_params,
        policy,
        hooks
      )

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache}
  end

  defp get_input(inputs, name, optional?) do
    res =
      case inputs do
        %Nx.Tensor{} = inputs ->
          inputs

        %{} = inputs ->
          inputs[name]

        inputs when is_tuple(inputs) ->
          inputs

        _ ->
          raise ArgumentError,
                "invalid input given to model," <>
                  " expected input to be a tensor or a map" <>
                  " corresponding to correct input names"
      end

    case {res, optional?} do
      {nil, false} ->
        raise ArgumentError,
              "unable to find input #{name} for model given to predict," <>
                " you must provide an input tensor for every required" <>
                " input specified in the graph"

      {nil, true} ->
        %Axon.None{}

      {value, _optional?} ->
        value
    end
  end

  # Sub-inference functions contain `params` - trainable parameters
  # passed to `predict`, `inputs` - inputs passed to `predict`, `state` -
  # an accumulator of layer state during the recursive building of the
  # inference function, `cache` - the built function cache for accessing
  # previous layer expressions, and `result_cache` - cached results to
  # avoid recomputing expressions in combined graphs.
  defp layer_predict_fun(
         params,
         inputs,
         state,
         cache,
         result_cache,
         fn_stacktrace,
         op,
         op_name,
         parent_ids,
         name,
         args,
         opts,
         %{output: output, compute: compute},
         layer_params,
         hooks,
         mode,
         layer_stacktrace
       ) do
    # Recurse graph inputs and invoke cache to get parent results,
    # state, and result_cache and then apply dtype policy and hooks
    # to each input
    {layer_inputs, {state, result_cache, none?}} =
      Enum.map_reduce(
        parent_ids,
        {state, result_cache, false},
        fn parent_id, {state, result_cache, none?} ->
          {layer_input, {state, result_cache}} =
            call_predict_cache(
              parent_id,
              params,
              inputs,
              state,
              cache,
              result_cache,
              fn_stacktrace
            )

          none? = none? or propagating_none?(layer_input)

          layer_input =
            layer_input
            |> safe_as_type(compute)
            |> apply_hooks(:pre_forward, mode, hooks)

          {layer_input, {state, result_cache, none?}}
        end
      )

    if none? do
      {%Axon.None{}, {state, result_cache}}
    else
      # Parameters are just accessed in the layer sub-map of the nested
      # parameter map, so we just need to extract them and then apply
      # freezing and dtype policy
      parameter_inputs =
        Enum.map(layer_params, fn %{type: type, name: v, frozen: frz} ->
          param = params[name][v]

          cond do
            param != nil ->
              safe_as_type(maybe_freeze(param, frz), compute)

            true ->
              raise ArgumentError,
                    "parameter #{inspect(v)} for layer: #{inspect(name)} in" <>
                      " was not present in the given parameter map, this can" <>
                      " happen if you are using parameters intended for another" <>
                      " model or did not initialize portions of your model with" <>
                      " Axon.init/3"
          end
        end)

      # Reorder the inputs according to the original input ordering
      # so the function is invoked correctly
      {[], [], tensor_inputs} =
        Enum.reduce(args, {layer_inputs, parameter_inputs, []}, fn
          :layer, {[layer | rest], parameters, inputs} ->
            {rest, parameters, [layer | inputs]}

          :parameter, {layer_inputs, [param | rest], inputs} ->
            {layer_inputs, rest, [param | inputs]}
        end)

      # Compute arguments to be forwarded and ensure `:mode` is included
      # for inference/training behavior dependent functions
      args = Enum.reverse(tensor_inputs, [Keyword.put(opts, :mode, mode)])

      # For built-in layers we always just apply the equivalent function
      # in Axon.Layers. The implication of this is that every function which
      # can be invoked as a layer must have a definition in Axon.Layers even
      # if there is a distinction (e.g. with activations)
      result = apply_layer(name, op, args, layer_stacktrace, fn_stacktrace, op_name)

      result =
        case result do
          # Make sure the none is non-propagating
          %Axon.None{} -> %Axon.None{}
          result -> result
        end

      # Final stage is to extract correct output form by determining if
      # the layer had stateful output, apply hooks, and cast back to policy
      # dtype for outputs
      {out, state} =
        case result do
          %StatefulOutput{output: out, state: out_state} ->
            new_out =
              out
              |> apply_hooks(:forward, mode, hooks)
              |> apply_hooks(:backward, mode, hooks)
              |> safe_as_type(output)

            new_state = Map.put(state, name, out_state)
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
  end

  defp apply_layer(name, op, args, layer_stacktrace, fn_stacktrace, op_name) do
    try do
      result =
        case op do
          op when is_function(op) ->
            apply(op, args)

          op when is_atom(op) ->
            apply(Axon.Layers, op, args)
        end

      case result do
        out when is_tuple(out) ->
          out

        %Axon.None{} = out ->
          out

        %Axon.StatefulOutput{output: out} = stateful ->
          out = Nx.Defn.Expr.metadata(Nx.Defn.Expr.tensor(out), %{axon_layer: op_name})
          %{stateful | output: out}

        %Nx.Tensor{} = out
          Nx.Defn.Expr.metadata(Nx.Defn.Expr.tensor(out), %{axon_layer: op_name})

        out ->
          out
      end
    rescue
      exception ->
        # outside_apply is the internal compiler stacktrace.
        # Print it when debugging compiler bugs.
        {inside_apply, _outside_apply} =
          Enum.split_while(__STACKTRACE__, fn {mod, fun, _arity, _info} ->
            mod != __MODULE__ and fun != :apply_layer
          end)

        mfa =
          case op do
            op when is_function(op) ->
              {:module, module} = Function.info(op, :module)
              {:name, name} = Function.info(op, :name)
              {module, name, length(args)}

            op when is_atom(op) ->
              {Axon.Layers, op, length(args)}
          end

        reraise Axon.CompileError,
                [
                  exception: exception,
                  name: name,
                  mfa: mfa,
                  layer_stacktrace: layer_stacktrace,
                  compile_stacktrace: inside_apply
                ],
                fn_stacktrace
    end
  end

  defp layer_init_fun(
         layer_id,
         template,
         cache,
         result_cache,
         fn_stacktrace,
         keys,
         parent_ids,
         name,
         predict_fun,
         parameters,
         %{params: dtype},
         hooks
       ) do
    {parent_shapes, {parent_params, result_cache, none?}} =
      Enum.map_reduce(parent_ids, {%{}, result_cache, false}, fn
        parent_id, {params, result_cache, none?} ->
          {parent_shape, {params, result_cache}} =
            call_init_cache(parent_id, template, params, cache, result_cache, fn_stacktrace, keys)

          none? = none? or propagating_none?(parent_shape)
          {parent_shape, {params, result_cache, none?}}
      end)

    if none? do
      {%Axon.None{}, {parent_params, result_cache}}
    else
      layer_params =
        Enum.reduce(parameters, %{}, fn param, layer_params ->
          init_param(layer_id, param, layer_params, parent_shapes, dtype, keys)
        end)

      layer_params = apply_hooks(layer_params, :initialize, nil, hooks)

      params =
        if layer_params == %{} do
          parent_params
        else
          Map.put(parent_params, name, layer_params)
        end

      {pred_expr, {_, result_cache}} =
        predict_fun.(params, template, %{}, cache, result_cache, fn_stacktrace)

      {safe_shape(pred_expr), {params, result_cache}}
    end
  end

  defp init_param(layer_id, param, layer_params, parent_shapes, dtype, keys) do
    %{name: name, shape: shape, initializer: initializer} = param

    params =
      case shape do
        {:tuple, params} ->
          params =
            Enum.map(params, fn shape ->
              shape = apply(shape, parent_shapes)
              apply_initializer(layer_id, initializer, name, shape, dtype, keys)
            end)

          List.to_tuple(params)

        shape ->
          shape = apply(shape, parent_shapes)
          apply_initializer(layer_id, initializer, name, shape, dtype, keys)
      end

    Map.put(layer_params, name, params)
  end

  defp apply_initializer(_layer_id, initializer, _name, shape, type, _keys)
       when is_function(initializer, 2) do
    initializer.(shape, type)
  end

  defp apply_initializer(layer_id, initializer, name, shape, type, keys)
       when is_function(initializer, 3) do
    initializer.(shape, type, keys[layer_id][name])
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
          Nx.Defn.Kernel.custom_grad(expr, [expr], fn g ->
            hooked_g = Nx.Defn.Kernel.hook(g, hook_fn)
            [hooked_g]
          end)
        else
          Nx.Defn.Kernel.hook(expr, hook_fn)
        end
      else
        expr
      end
    end)
  end

  defp safe_as_type(container_or_tensor, type) do
    case container_or_tensor do
      %Axon.None{} = none ->
        none

      %Nx.Tensor{} = tensor ->
        if not Nx.Type.integer?(Nx.type(tensor)) and not Nx.Type.integer?(type) do
          Nx.as_type(tensor, type)
        else
          tensor
        end

      container ->
        deep_new(container, fn tensor ->
          if not Nx.Type.integer?(Nx.type(tensor)) and not Nx.Type.integer?(type) do
            Nx.as_type(tensor, type)
          else
            tensor
          end
        end)
    end
  end

  defp safe_shape(container_or_tensor) do
    case container_or_tensor do
      %Axon.None{} = none ->
        none

      %Nx.Tensor{} = tensor ->
        Nx.shape(tensor)

      container ->
        deep_new(container, &Nx.shape/1)
    end
  end

  defp safe_type(container_or_tensor) do
    case container_or_tensor do
      %Axon.None{} = none ->
        none

      %Nx.Tensor{} = tensor ->
        Nx.type(tensor)

      container ->
        deep_new(container, &Nx.type/1)
    end
  end

  defp propagating_none?(%Axon.None{__propagate__: true}), do: true
  defp propagating_none?(_), do: false

  defp us_to_ms(time), do: Float.round(time / 1000, 1)
end
