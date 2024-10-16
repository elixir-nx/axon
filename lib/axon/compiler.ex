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

        (pass debug: true to build/compile to see where the layer was defined)

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

  alias Axon.StatefulOutput

  ## Init JIT Compilation

  @doc false
  def build(%Axon{output: id, nodes: nodes}, opts) do
    debug? = Keyword.get(opts, :debug, false)
    raise_on_none? = Keyword.get(opts, :raise_on_none, true)
    mode = Keyword.get(opts, :mode, :inference)
    seed = Keyword.get_lazy(opts, :seed, fn -> :erlang.system_time() end)
    print_values = Keyword.get(opts, :print_values, false)
    global_layer_options = Keyword.get(opts, :global_layer_options, [])

    config = %{
      mode: mode,
      debug?: debug?,
      global_layer_options: global_layer_options,
      print_values: print_values
    }

    {time, {root_id, {cache, _op_counts, _block_cache, model_state_meta}}} =
      :timer.tc(fn ->
        to_model_funs(
          id,
          nodes,
          {%{}, %{}, %{}, %{parameters: %{}, state: %{}, frozen_parameters: %{}}},
          config
        )
      end)

    if debug? do
      Logger.debug("Axon finished graph traversal in #{us_to_ms(time)}ms")
    end

    predict_cache =
      Map.new(cache, fn {_, {int_id, %{predict: predict}}} -> {int_id, %{predict: predict}} end)

    predict_fun = fn params, inputs ->
      # TODO: Legacy parameter map support. Remove on v1.0
      inference_params =
        case params do
          %Axon.ModelState{data: params} ->
            params

          params ->
            Logger.warning(
              "Passing a parameter map to Axon's inference methods is deprecated. Use %Axon.ModelState{} instead."
            )

            params
        end

      {:current_stacktrace, [_process_info, _fn | stacktrace]} =
        Process.info(self(), :current_stacktrace)

      {time, result} =
        :timer.tc(fn ->
          case mode do
            :train ->
              {pred_expr, {state_expr, _}} =
                predict_cache[root_id][:predict].(
                  inference_params,
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
                  inference_params,
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
        if raise_on_none? do
          raise ArgumentError,
                "the compiled model will always result in %Axon.None{}." <>
                  " This most likely means you specified optional output and " <>
                  " did not handle the case when it is missing"
        end
      end

      result
    end

    init_cache = Map.new(cache, fn {_, {int_id, funs}} -> {int_id, funs} end)

    init_fun = fn template, init_state ->
      init_model_state =
        case init_state do
          %Axon.ModelState{} = model_state ->
            model_state

          %{} = init_params ->
            Logger.warning(
              "passing parameter map to initialization is deprecated, use %Axon.ModelState{} instead"
            )

            parameters = %{}

            %Axon.ModelState{
              data: init_params,
              parameters: parameters,
              state: %{},
              frozen_parameters: %{}
            }
        end

      {:current_stacktrace, [_process_info, _fn | stacktrace]} =
        Process.info(self(), :current_stacktrace)

      {time, params} =
        :timer.tc(fn ->
          param_keys = get_keys(nodes, seed)

          {_, {params, _}} =
            init_cache[root_id][:init].(template, init_cache, %{}, stacktrace, param_keys)

          params
        end)

      out =
        params
        |> normalize_blocks(model_state_meta)
        |> merge_model_state!(init_model_state)

      if debug? do
        Logger.debug("Axon finished init expression generation in #{us_to_ms(time)}ms")
      end

      out
    end

    {init_fun, predict_fun}
  end

  defp get_keys(nodes, seed) do
    {ids_and_data, _op_counts} =
      Enum.reduce(nodes, {[], %{}}, fn
        {_, %Axon.Node{id: id, op: op, name: name_fn, parameters: params}}, {keys, op_counts} ->
          name = name_fn.(op, op_counts)
          op_counts = Map.update(op_counts, op, 1, &(&1 + 1))
          keys = get_node_keys(id, name, params, keys)
          {keys, op_counts}
      end)

    {ids, data} = Enum.unzip(ids_and_data)
    data = List.flatten(data)

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
          Enum.reduce(ids, {%{}, 0}, fn {layer_id, param}, {acc, i} ->
            {{root_name, keys}, i} = recur_slice_keys(keys_tensor, param, i)

            layer_keys =
              Map.update(acc, layer_id, %{root_name => keys}, &Map.put(&1, root_name, keys))

            {layer_keys, i}
          end)

        keys
    end
  end

  defp get_node_keys(id, parent_name, params, keys) do
    Enum.reduce(params, keys, fn param, keys ->
      case get_param_data(parent_name, param) do
        nil -> keys
        {param_name, data} -> [{{id, param_name}, data} | keys]
      end
    end)
  end

  defp get_param_data(parent_name, param) do
    case param do
      %Axon.Parameter{name: param_name, type: :map, children: inner_params} ->
        parent_name = parent_name <> "." <> param_name

        {inner_names, inner_data} =
          Enum.map(inner_params, &get_param_data(parent_name, &1))
          |> Enum.reject(&(&1 == nil))
          |> Enum.unzip()

        case inner_data do
          [] ->
            nil

          [_ | _] ->
            {{param_name, inner_names}, inner_data}
        end

      %Axon.Parameter{name: param_name, initializer: fun} ->
        {:arity, arity} = Function.info(fun, :arity)

        cond do
          arity == 2 ->
            nil

          arity == 3 ->
            <<data::unsigned-size(32), _rest::binary>> =
              :erlang.md5(parent_name <> "." <> param_name)

            {param_name, [data]}

          true ->
            raise ArgumentError, "bad initializer arity"
        end
    end
  end

  defp recur_slice_keys(keys_tensor, param, i) do
    case param do
      {composite_param_name, children} ->
        {subkeys, i} =
          Enum.reduce(children, {%{}, i}, fn child_param, {acc, i} ->
            {{root_name, keys}, i} = recur_slice_keys(keys_tensor, child_param, i)
            {Map.put(acc, root_name, keys), i}
          end)

        {{composite_param_name, subkeys}, i}

      param_name when is_binary(param_name) ->
        key = keys_tensor[i]
        {{param_name, key}, i + 1}
    end
  end

  defp merge_model_state!(state, init_state) do
    %{state | data: merge_params!(state.data, init_state.data)}
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

  defp normalize_blocks(params, %{
         state: meta_state,
         parameters: meta_params,
         frozen_parameters: frozen
       }) do
    model_state = %Axon.ModelState{
      data: %{},
      state: meta_state,
      parameters: meta_params,
      frozen_parameters: frozen
    }

    # Blocks are kinda hacky and produce a model state,
    # so we normalize them so we get proper model metadata
    # and then have just one root-level model state struct
    Enum.reduce(params, model_state, fn
      {key, %Axon.ModelState{} = model_state}, acc_model_state ->
        acc_model_state
        |> update_in([Access.key!(:parameters)], fn state ->
          if model_state.parameters == %{},
            do: state,
            else: Map.put(state, key, model_state.parameters)
        end)
        |> update_in([Access.key!(:state)], fn state ->
          if model_state.state == %{},
            do: state,
            else: Map.put(state, key, model_state.state)
        end)
        |> update_in([Access.key!(:data)], fn state ->
          Map.put(state, key, model_state.data)
        end)

      {key, data}, acc_model_state ->
        update_in(acc_model_state, [Access.key!(:data)], &Map.put(&1, key, data))
    end)
  end

  def compile(graph, _opts) do
    raise ArgumentError,
          "attempting to compile model functions from" <>
            " an unrecognized graph #{inspect(graph)}, if you" <>
            " are attempting to compile a model with a container" <>
            " output, use `Axon.container`"
  end

  defp to_model_funs(id, nodes, {cache, op_counts, block_cache, model_state_meta}, config) do
    case cache do
      %{^id => {int_id, _}} ->
        {int_id, {cache, op_counts, block_cache, model_state_meta}}

      %{} ->
        {id, model_funs, cache, op_counts, block_cache, model_state_meta} =
          recur_model_funs(
            nodes[id],
            nodes,
            {cache, op_counts, block_cache, model_state_meta},
            config
          )

        int_id = map_size(cache)

        {int_id,
         {Map.put(cache, id, {int_id, model_funs}), op_counts, block_cache, model_state_meta}}
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

    {parent_template, {parent_params, result_cache}} =
      case result_cache do
        %{^key => {parent_template, parent_params}} ->
          {parent_template, {parent_params, result_cache}}

        %{} ->
          {parent_template, {parent_params, result_cache}} =
            cache[parent_id][:init].(template, cache, result_cache, fn_stacktrace, keys)

          {parent_template,
           {parent_params, Map.put(result_cache, key, {parent_template, parent_params})}}
      end

    {parent_template, {Map.merge(parent_params, params), result_cache}}
  end

  # If the node is ignored for the current mode, we pass through and recur next
  defp recur_model_funs(
         %Axon.Node{id: id, mode: node_mode, parent: [parent | _]},
         nodes,
         {cache, op_counts, block_cache, model_state_meta},
         config
       )
       when node_mode != :both and node_mode != config.mode do
    {parent_id, {cache, op_counts, block_cache, model_state_meta}} =
      to_model_funs(parent, nodes, {cache, op_counts, block_cache, model_state_meta}, config)

    predict_fun = fn params, inputs, state, cache, result_cache, fn_stacktrace ->
      call_predict_cache(parent_id, params, inputs, state, cache, result_cache, fn_stacktrace)
    end

    init_fun = fn template, cache, result_cache, fn_stacktrace, keys ->
      call_init_cache(parent_id, template, %{}, cache, result_cache, fn_stacktrace, keys)
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache, model_state_meta}
  end

  defp recur_model_funs(
         %Axon.Node{id: id, name: name_fn, op: :constant, opts: [value: tensor], policy: policy},
         _nodes,
         {cache, op_counts, block_cache, model_state_meta},
         %{print_values: print_values}
       ) do
    name = name_fn.(:constant, op_counts)
    op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)
    tensor = Nx.backend_copy(tensor, Nx.BinaryBackend)

    predict_fun = fn _params, _inputs, state, _cache, result_cache, _fn_stacktrace ->
      out =
        tensor
        |> safe_policy_cast(policy, :output)
        |> maybe_print_values(name, print_values)

      {out, {state, result_cache}}
    end

    init_fun = fn _template, _cache, result_cache, _fn_stacktrace, _keys ->
      {Nx.shape(tensor), {%{}, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache, model_state_meta}
  end

  defp recur_model_funs(
         %Axon.Node{
           id: id,
           op: :input,
           hooks: hooks,
           name: name_fn,
           opts: [shape: _input_shape, optional: optional?]
         },
         nodes,
         {cache, op_counts, block_cache, model_state_meta},
         %{mode: mode, print_values: print_values}
       ) do
    name = name_fn.(:input, op_counts)
    op_counts = Map.update(op_counts, :input, 1, fn x -> x + 1 end)
    all_inputs = get_all_inputs(nodes)

    predict_fun = fn _params, inputs, state, _cache, result_cache, _fn_stacktrace ->
      value = get_input(all_inputs, inputs, name, optional?)

      # TODO: Add this back in
      # validate_input_shape!(value, shape)

      res =
        value
        |> apply_hooks(name, :forward, mode, hooks)
        |> apply_hooks(name, :backward, mode, hooks)
        |> maybe_print_values(name, print_values)

      {res, {state, result_cache}}
    end

    init_fun = fn template, _cache, result_cache, _fn_stacktrace, _keys ->
      input = get_input(all_inputs, template, name, optional?)
      {Nx.to_template(input), {%{}, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache, model_state_meta}
  end

  defp recur_model_funs(
         %Axon.Node{id: id, op: :optional, parent: [parent]},
         nodes,
         {cache, op_counts, block_cache, model_state_meta},
         config
       ) do
    {parent_id, {cache, op_counts, block_cache, model_state_meta}} =
      to_model_funs(parent, nodes, {cache, op_counts, block_cache, model_state_meta}, config)

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
    {id, model_funs, cache, op_counts, block_cache, model_state_meta}
  end

  defp recur_model_funs(
         %Axon.Node{
           id: id,
           op: :block,
           parent: parents,
           opts: [block_fun: block_fun, block_id: block_id],
           name: name_fn
         },
         nodes,
         cache_and_counts,
         config
       ) do
    {parent_ids, {cache, op_counts, block_cache, model_state_meta}} =
      Enum.map_reduce(
        parents,
        cache_and_counts,
        &to_model_funs(&1, nodes, &2, config)
      )

    {{block_init_fun, block_predict_fun}, block_name, block_cache, op_counts} =
      case block_cache do
        %{^block_id => {funs, name}} = block_cache ->
          {funs, name, block_cache, op_counts}

        %{} ->
          inputs = Enum.with_index(parents, fn _, i -> Axon.input("subgraph#{i}") end)
          funs = build(apply(block_fun, inputs), debug?: config.debug?)
          name = name_fn.(:block, op_counts)
          op_counts = Map.update(op_counts, :block, 1, fn x -> x + 1 end)
          {funs, name, Map.put(block_cache, block_id, {funs, name}), op_counts}
      end

    predict_fun = fn params, inputs, state, cache, result_cache, fn_stacktrace ->
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

            {layer_input, {state, result_cache, none?}}
          end
        )

      if none? do
        {%Axon.None{}, {state, result_cache}}
      else
        block_params = params[block_name] || %{}

        inputs =
          layer_inputs
          |> Enum.with_index()
          |> Map.new(fn {input, i} -> {"subgraph#{i}", input} end)

        result = apply(block_predict_fun, [Axon.ModelState.new(block_params), inputs])

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

        out_result = maybe_print_values(out_result, block_name, config.print_values)

        {out_result, {state, result_cache}}
      end
    end

    init_fun = fn template, cache, result_cache, fn_stacktrace, keys ->
      {parent_templates, {parent_params, result_cache, none?}} =
        Enum.map_reduce(parent_ids, {%{}, result_cache, false}, fn
          parent_id, {params, result_cache, none?} ->
            {parent_template, {params, result_cache}} =
              call_init_cache(
                parent_id,
                template,
                params,
                cache,
                result_cache,
                fn_stacktrace,
                keys
              )

            none? = none? or propagating_none?(parent_template)
            {parent_template, {params, result_cache, none?}}
        end)

      if none? do
        {%Axon.None{}, {parent_params, result_cache}}
      else
        templates =
          parent_templates
          |> Enum.with_index()
          |> Map.new(fn {template, i} -> {"subgraph#{i}", Nx.broadcast(0.0, template)} end)

        block_params = apply(block_init_fun, [templates, Axon.ModelState.empty()])

        params =
          if block_params == %{} do
            %{}
          else
            Map.put(parent_params, block_name, block_params)
          end

        {pred_expr, {_, result_cache}} =
          predict_fun.(params, template, %{}, cache, result_cache, fn_stacktrace)

        {Nx.to_template(pred_expr), {params, result_cache}}
      end
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache, model_state_meta}
  end

  defp recur_model_funs(
         %Axon.Node{id: id, op: :namespace, name: name_fn, parent: [parent]},
         nodes,
         {cache, op_counts, block_cache, model_state_meta},
         config
       ) do
    name = name_fn.(:namespace, op_counts)
    # To ensure that a namespace always has the same layer names,
    # we reset op_counts, input layers always belong to the global
    # namespace, so we include those regardless
    input_count = op_counts[:input] || 0
    namespace_op_counts = %{input: input_count}
    namespace_model_state_meta = %{parameters: %{}, state: %{}, frozen_parameters: %{}}

    # All of the children of this namespace belong to it, so
    # we forward this name to the namespace, but everything after
    # it belongs to whatever namespace we're currently in
    {parent_id, {cache, namespace_op_counts, block_cache, namespace_model_state_meta}} =
      to_model_funs(
        parent,
        nodes,
        {cache, namespace_op_counts, block_cache, namespace_model_state_meta},
        config
      )

    # Update the global op_count of input layers, since they
    # are a global operation regardless of where they are
    input_count = namespace_op_counts[:input] || 0
    op_counts = Map.put(op_counts, :input, input_count)

    # Update the model state meta to include the namespace model state meta
    model_state_meta =
      model_state_meta
      |> Map.update!(:parameters, &Map.put(&1, name, namespace_model_state_meta[:parameters]))
      |> Map.update!(:state, &Map.put(&1, name, namespace_model_state_meta[:state]))
      |> Map.update!(
        :frozen_parameters,
        &Map.put(&1, name, namespace_model_state_meta[:frozen_parameters])
      )

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
      {_parent_template, {namespace_params, result_cache}} =
        call_init_cache(parent_id, template, %{}, cache, result_cache, fn_stacktrace, keys)

      params =
        if namespace_params == %{} do
          %{}
        else
          %{name => namespace_params}
        end

      {pred_expr, {_, result_cache}} =
        predict_fun.(params, template, %{}, cache, result_cache, fn_stacktrace)

      {Nx.to_template(pred_expr), {params, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, model_funs, cache, op_counts, block_cache, model_state_meta}
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
           global_options: global_options,
           policy: policy,
           hooks: hooks,
           op_name: op_name,
           stacktrace: stacktrace
         },
         nodes,
         cache_and_counts,
         %{
           mode: mode,
           debug?: debug?,
           global_layer_options: global_layer_options,
           print_values: print_values
         } = config
       )
       when (is_function(op) or is_atom(op)) and is_list(inputs) do
    # Traverse to accumulate cache and get parent_ids for
    # application within the function. We work only with
    # functions and IDs to avoid leaking entire graphs into
    # the closure
    {parent_ids, {cache, op_counts, block_cache, model_state_meta}} =
      Enum.map_reduce(
        inputs,
        cache_and_counts,
        &to_model_funs(&1, nodes, &2, config)
      )

    # Names are computed lazily, so compute name from current
    # op and aggregate op_counts.
    name = name_fn.(op_name, op_counts)
    op_counts = Map.update(op_counts, op_name, 1, fn x -> x + 1 end)

    # Get parameter metadata for the layer
    model_state_meta =
      Enum.reduce(layer_params, model_state_meta, fn
        %{kind: :parameter, frozen: frozen?, name: param_name}, acc ->
          meta =
            Map.update!(acc, :parameters, fn layer_meta ->
              Map.update(layer_meta, name, [param_name], &[param_name | &1])
            end)

          if frozen? do
            Map.update!(meta, :frozen_parameters, fn layer_meta ->
              Map.update(layer_meta, name, [param_name], &[param_name | &1])
            end)
          else
            meta
          end

        %{kind: :state, name: param_name}, acc ->
          Map.update!(acc, :state, fn layer_meta ->
            Map.update(layer_meta, name, [param_name], &[param_name | &1])
          end)
      end)

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
        global_options,
        policy,
        layer_params,
        hooks,
        mode,
        global_layer_options,
        print_values,
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
    {id, model_funs, cache, op_counts, block_cache, model_state_meta}
  end

  defp get_all_inputs(nodes) do
    nodes
    |> Enum.filter(fn {_, %{op: op}} -> op == :input end)
    |> Enum.map(fn {_, %{name: name_fn}} ->
      # inputs require a name, so we can just ignore op counts
      name_fn.(:input, %{})
    end)
    |> Enum.uniq()
  end

  defp get_input(all_input_names, inputs, name, optional?) do
    res =
      case {all_input_names, inputs} do
        {[^name], %Nx.Tensor{} = inputs} ->
          inputs

        {_, %Nx.Tensor{}} ->
          raise ArgumentError,
                "ambiguous input given to the model," <>
                  " expected inputs with names #{inspect(all_input_names)}" <>
                  " but received a single tensor as input"

        {_, %{} = inputs} ->
          inputs[name]

        {[^name], inputs} when is_tuple(inputs) ->
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
         init_state,
         cache,
         result_cache,
         fn_stacktrace,
         op,
         op_name,
         parent_ids,
         name,
         args,
         opts,
         global_options,
         policy,
         layer_params,
         hooks,
         mode,
         global_layer_options,
         print_values,
         layer_stacktrace
       ) do
    # Recurse graph inputs and invoke cache to get parent results,
    # state, and result_cache and then apply dtype policy and hooks
    # to each input
    {layer_inputs, {parent_state, result_cache, none?}} =
      Enum.map_reduce(
        parent_ids,
        {%{}, result_cache, false},
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
            |> safe_policy_cast(policy, :compute)
            |> apply_hooks(name, :pre_forward, mode, hooks)

          {layer_input, {state, result_cache, none?}}
        end
      )

    # there's an issue where stateful layers can have the same input,
    # which means that if they're in the same container the "parent" state
    # will get wiped out. This happens with RNNs, so we fix that here
    state = Map.merge(init_state, parent_state)

    if none? do
      {%Axon.None{}, {state, result_cache}}
    else
      # Parameters are just accessed in the layer sub-map of the nested
      # parameter map, so we just need to extract them and then apply
      # freezing and dtype policy
      parameter_inputs =
        Enum.map(layer_params, fn %{name: v, frozen: frz} ->
          param = params[name][v]

          cond do
            param != nil ->
              safe_policy_cast(maybe_freeze(param, frz), policy, :compute)

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
      layer_opts =
        opts
        |> Keyword.merge(Keyword.take(global_layer_options, global_options))
        |> Keyword.put(:mode, mode)

      args = Enum.reverse(tensor_inputs, [layer_opts])

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
              |> apply_hooks(name, :forward, mode, hooks)
              |> apply_hooks(name, :backward, mode, hooks)
              |> safe_policy_cast(policy, :output)

            new_state = Map.put(state, name, out_state)
            {new_out, new_state}

          out ->
            new_out =
              out
              |> apply_hooks(name, :forward, mode, hooks)
              |> apply_hooks(name, :backward, mode, hooks)
              |> safe_policy_cast(policy, :output)

            {new_out, state}
        end

      out = maybe_print_values(out, name, print_values)

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

        %Nx.Tensor{data: %{op: :metadata, args: [arg, metadata]} = expr} = out ->
          %{out | data: %{expr | args: [arg, Map.put(metadata, :axon_layer, op_name)]}}

        %Nx.Tensor{} = out ->
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
    {parent_templates, {parent_params, result_cache, none?}} =
      Enum.map_reduce(parent_ids, {%{}, result_cache, false}, fn
        parent_id, {params, result_cache, none?} ->
          {parent_template, {params, result_cache}} =
            call_init_cache(parent_id, template, params, cache, result_cache, fn_stacktrace, keys)

          none? = none? or propagating_none?(parent_template)
          {parent_template, {params, result_cache, none?}}
      end)

    if none? do
      {%Axon.None{}, {parent_params, result_cache}}
    else
      layer_params =
        Enum.reduce(parameters, %{}, fn param, layer_params ->
          init_param(layer_id, param, layer_params, parent_templates, dtype, keys)
        end)

      layer_params = apply_hooks(layer_params, name, :initialize, nil, hooks)

      params =
        if layer_params == %{} do
          parent_params
        else
          Map.put(parent_params, name, layer_params)
        end

      {pred_expr, {_, result_cache}} =
        predict_fun.(params, template, %{}, cache, result_cache, fn_stacktrace)

      {Nx.to_template(pred_expr), {params, result_cache}}
    end
  end

  defp init_param(layer_id, param, layer_params, parent_templates, dtype, keys) do
    %{name: name, template: template, initializer: initializer} = param

    template =
      case template do
        template_fun when is_function(template) ->
          apply(template_fun, parent_templates)

        %Nx.Tensor{} = template ->
          template

        other ->
          raise "unsupported parameter template, template must be a template tensor or function, got #{inspect(other)}"
      end

    # TODO: We might just need to forward the template
    shape = Nx.shape(template)
    dtype = dtype || Nx.type(template)

    params = apply_initializer(layer_id, initializer, name, shape, dtype, keys)

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

  defp maybe_print_values(value, layer, true) do
    Nx.Defn.Kernel.print_value(value, label: layer)
  end

  defp maybe_print_values(value, _, _), do: value

  defp apply_hooks(res, layer_name, event, mode, hooks) do
    hooks
    |> Enum.reverse()
    |> Enum.reduce(res, fn {on_event, on_mode, hook_fn}, expr ->
      event? = on_event == event or on_event == :all
      mode? = on_mode == mode or on_mode == :both or mode == nil

      if event? and mode? do
        if on_event == :backward do
          Nx.Defn.Kernel.custom_grad(expr, [expr], fn g ->
            hooked_g = Nx.Defn.Kernel.hook(g, String.to_atom(layer_name), hook_fn)
            [hooked_g]
          end)
        else
          Nx.Defn.Kernel.hook(expr, String.to_atom(layer_name), hook_fn)
        end
      else
        expr
      end
    end)
  end

  defp safe_policy_cast(container_or_tensor, policy, variable_type) do
    case container_or_tensor do
      %Axon.None{} = none ->
        none

      container_or_tensor ->
        Axon.MixedPrecision.cast(policy, container_or_tensor, variable_type)
    end
  end

  defp propagating_none?(%Axon.None{__propagate__: true}), do: true
  defp propagating_none?(_), do: false

  defp us_to_ms(time), do: Float.round(time / 1000, 1)
end
