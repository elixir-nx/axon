defmodule Axon.Compiler do
  @moduledoc false
  require Logger

  import Axon.Shared
  alias Axon.StatefulOutput

  ## Init JIT Compilation

  @doc false
  def build(%Axon{} = graph, opts) do
    debug? = Keyword.get(opts, :debug, false)
    mode = Keyword.get(opts, :mode, :inference)
    key = Keyword.get_lazy(opts, :key, fn -> Nx.Random.key(:erlang.system_time()) end)

    {time, {root_id, {cache, _op_counts, _key}}} =
      :timer.tc(fn ->
        to_model_funs(graph, {%{}, %{}, key}, mode)
      end)

    if debug? do
      Logger.debug("Axon finished graph traversal in #{us_to_ms(time)}ms")
    end

    predict_fn = fn params, inputs ->
      {time, result} =
        :timer.tc(fn ->
          case mode do
            :train ->
              {pred_expr, {state_expr, _}} =
                cache[root_id][:predict].(params, inputs, %{}, cache, %{})

              %{prediction: pred_expr, state: state_expr}

            :inference ->
              {pred_expr, _} = cache[root_id][:predict].(params, inputs, %{}, cache, %{})
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

    init_fn = fn template, init_params ->
      {time, params} =
        :timer.tc(fn ->
          {_, {params, _}} = cache[root_id][:init].(template, cache, %{})
          params
        end)

      params = merge_params!(params, init_params)

      if debug? do
        Logger.debug("Axon finished init expression generation in #{us_to_ms(time)}ms")
      end

      params
    end

    {init_fn, predict_fn}
  end

  defp merge_params!(params, init_params) do
    Enum.reduce(init_params, params, fn {key, value}, params ->
      case params do
        %{^key => %{} = nested} when not is_struct(nested) ->
          %{params | key => merge_params!(nested, value)}

        %{^key => _} ->
          %{params | key => value}

        _ ->
          raise ArgumentError,
                "found unexpected key in the initial parameters map: #{inspect(key)}"
      end
    end)
  end

  def compile(graph, _opts) do
    raise ArgumentError,
          "attempting to compile model functions from" <>
            " an unrecognized graph #{inspect(graph)}, if you" <>
            " are attempting to compile a model with a container" <>
            " output, use `Axon.container`"
  end

  defp to_model_funs(%{id: id} = graph, {cache, op_counts, key}, mode) do
    case cache do
      %{^id => _} ->
        {id, {cache, op_counts, key}}

      %{} ->
        recur_model_funs(graph, {cache, op_counts, key}, mode)
    end
  end

  defp call_predict_cache(parent_id, params, inputs, state, cache, result_cache) do
    key = {:predict_cache, parent_id}

    case result_cache do
      %{^key => {expr, state}} ->
        {expr, {state, result_cache}}

      %{} ->
        {expr, {state, result_cache}} =
          cache[parent_id][:predict].(params, inputs, state, cache, result_cache)

        {expr, {state, Map.put(result_cache, key, {expr, state})}}
    end
  end

  defp call_init_cache(parent_id, template, params, cache, result_cache) do
    key = {:init_cache, parent_id}

    {parent_shape, {parent_params, result_cache}} =
      case result_cache do
        %{^key => {parent_shape, parent_params}} ->
          {parent_shape, {parent_params, result_cache}}

        %{} ->
          {parent_shape, {parent_params, result_cache}} =
            cache[parent_id][:init].(template, cache, result_cache)

          {parent_shape,
           {parent_params, Map.put(result_cache, key, {parent_shape, parent_params})}}
      end

    {parent_shape, {Map.merge(parent_params, params), result_cache}}
  end

  defp recur_model_funs(
         %Axon{id: id, op: :constant, opts: [value: tensor], policy: %{output: output}},
         {cache, op_counts, key},
         _
       ) do
    op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)

    tensor = Nx.backend_copy(tensor, Nx.Defn.Expr)

    predict_fun = fn _params, _inputs, state, _cache, result_cache ->
      out = safe_as_type(tensor, output)
      {out, {state, result_cache}}
    end

    init_fun = fn _template, _cache, result_cache ->
      {Nx.shape(tensor), {%{}, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}

    {id, {Map.put(cache, id, model_funs), op_counts, key}}
  end

  defp recur_model_funs(
         %Axon{
           id: id,
           op: :input,
           hooks: hooks,
           name: name_fn,
           opts: [shape: _input_shape, optional: optional?]
         },
         {cache, op_counts, key},
         mode
       ) do
    name = name_fn.(:input, op_counts)
    op_counts = Map.update(op_counts, :input, 1, fn x -> x + 1 end)

    predict_fun = fn _params, inputs, state, _cache, result_cache ->
      value = get_input(inputs, name, optional?)

      # TODO: Add this back in
      # validate_input_shape!(value, shape)

      res =
        value
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, {state, result_cache}}
    end

    init_fun = fn template, _cache, result_cache ->
      input = get_input(template, name, optional?)
      {safe_shape(input), {%{}, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}

    {id, {Map.put(cache, id, model_funs), op_counts, key}}
  end

  defp recur_model_funs(
         %Axon{id: id, op: :optional, parent: [parent]},
         {cache, op_counts, key},
         mode
       ) do
    {parent_id, {cache, op_counts, key}} = to_model_funs(parent, {cache, op_counts, key}, mode)

    predict_fun = fn params, inputs, state, cache, result_cache ->
      {out, {state, result_cache}} =
        call_predict_cache(parent_id, params, inputs, state, cache, result_cache)

      out = with %Axon.None{} <- out, do: %Axon.None{__propagate__: false}

      {out, {state, result_cache}}
    end

    init_fun = fn template, cache, result_cache ->
      {out, {params, result_cache}} =
        call_init_cache(parent_id, template, %{}, cache, result_cache)

      out = with %Axon.None{} <- out, do: %Axon.None{__propagate__: false}

      {safe_shape(out), {params, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}

    {id, {Map.put(cache, id, model_funs), op_counts, key}}
  end

  defp recur_model_funs(
         %Axon{id: id, op: :container, parent: [parents]},
         cache_counts_key,
         mode
       ) do
    {parent_ids, {cache, op_counts, key}} =
      deep_map_reduce(parents, cache_counts_key, &to_model_funs(&1, &2, mode))

    op_counts = Map.update(op_counts, :container, 1, fn x -> x + 1 end)

    predict_fun = fn params, inputs, state, cache, result_cache ->
      {input, {state, result_cache, none?}} =
        deep_map_reduce(
          parent_ids,
          {state, result_cache, false},
          fn parent_id, {state, result_cache, none?} ->
            {input, {state, result_cache}} =
              call_predict_cache(parent_id, params, inputs, state, cache, result_cache)

            none? = none? or propagating_none?(input)
            {input, {state, result_cache, none?}}
          end
        )

      input = if none?, do: %Axon.None{}, else: input

      {input, {state, result_cache}}
    end

    init_fun = fn template, cache, result_cache ->
      {parent_shape, {parent_params, result_cache, none?}} =
        deep_map_reduce(parent_ids, {%{}, result_cache, false}, fn
          parent_id, {params, result_cache, none?} ->
            {parent_shape, {params, result_cache}} =
              call_init_cache(parent_id, template, params, cache, result_cache)

            none? = none? or propagating_none?(parent_shape)
            {parent_shape, {params, result_cache, none?}}
        end)

      parent_shape = if none?, do: %Axon.None{}, else: parent_shape

      {parent_shape, {parent_params, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}

    {id, {Map.put(cache, id, model_funs), op_counts, key}}
  end

  defp recur_model_funs(
         %Axon{id: id, op: :namespace, name: name_fn, parent: [parent]},
         {cache, op_counts, key},
         mode
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
    {parent_id, {cache, namespace_op_counts}} =
      to_model_funs(parent, {cache, namespace_op_counts, key}, mode)

    # Update the global op_count of input layers, since they
    # are a global operation regardless of where they are
    input_count = namespace_op_counts[:input] || 0
    op_counts = Map.put(op_counts, :input, input_count)

    # The function just returns the result of it's child,
    # or parent depending on how you view the tree
    predict_fun = fn params, inputs, state, cache, result_cache ->
      # We're only concerned with this namespaces parameters, so we pair
      # down parameters first given the namespace
      namespace_params = params[name]

      # TODO: How should hooks be handled here?
      # TODO: I think we can actually handle parameter freezing and access
      # better here by only forwarding params[namespace] to the child function
      {out, {state, result_cache}} =
        call_predict_cache(parent_id, namespace_params, inputs, state, cache, result_cache)

      {out, {state, result_cache}}
    end

    init_fun = fn template, cache, result_cache ->
      {_parent_shape, {namespace_params, result_cache}} =
        call_init_cache(parent_id, template, %{}, cache, result_cache)

      params =
        if namespace_params == %{} do
          %{}
        else
          %{name => namespace_params}
        end

      {pred_expr, {_, result_cache}} = predict_fun.(params, template, %{}, cache, result_cache)

      {safe_shape(pred_expr), {params, result_cache}}
    end

    model_funs = %{predict: predict_fun, init: init_fun}

    # Then we return the cache, op_counts, and original namespace
    {id, {Map.put(cache, id, model_funs), op_counts, key}}
  end

  defp recur_model_funs(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: inputs,
           parameters: layer_params,
           args: args,
           opts: opts,
           policy: policy,
           hooks: hooks,
           op_name: op_name
         },
         cache_counts_key,
         mode
       )
       when (is_function(op) or is_atom(op)) and is_list(inputs) do
    # Traverse to accumulate cache and get parent_ids for
    # application within the function. We work only with
    # functions and IDs to avoid leaking entire graphs into
    # the closure
    {parent_ids, {cache, op_counts, key}} =
      Enum.map_reduce(
        inputs,
        cache_counts_key,
        &to_model_funs(&1, &2, mode)
      )

    # Names are computed lazily, so compute name from current
    # op and aggregate op_counts.
    name = name_fn.(op_name, op_counts)
    op_counts = Map.update(op_counts, op_name, 1, fn x -> x + 1 end)

    # Each model builds two functions: predict_fun and init_fun
    predict_fun =
      &layer_predict_fun(
        &1,
        &2,
        &3,
        &4,
        &5,
        op,
        parent_ids,
        name,
        args,
        opts,
        policy,
        layer_params,
        hooks,
        mode
      )

    keys = Nx.Random.split(key)
    {k1, k2} = {keys[0], keys[1]}

    init_fun =
      &layer_init_fun(
        &1,
        &2,
        &3,
        parent_ids,
        name,
        predict_fun,
        layer_params,
        policy,
        hooks,
        Nx.backend_copy(k1, Nx.Defn.Expr)
      )

    model_funs = %{predict: predict_fun, init: init_fun}
    {id, {Map.put(cache, id, model_funs), op_counts, k2}}
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
                "invalid input given to model, expected input" <>
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
         op,
         parent_ids,
         name,
         args,
         opts,
         %{output: output, compute: compute},
         layer_params,
         hooks,
         mode
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
            call_predict_cache(parent_id, params, inputs, state, cache, result_cache)

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
        Enum.map(layer_params, fn %{name: v, frozen: frz} ->
          param = params[name][v]

          if param != nil do
            safe_as_type(maybe_freeze(param, frz), compute)
          else
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
      args = Enum.reverse(tensor_inputs) ++ [Keyword.put(opts, :mode, mode)]

      # For built-in layers we always just apply the equivalent function
      # in Axon.Layers. The implication of this is that every function which
      # can be invoked as a layer must have a definition in Axon.Layers even
      # if there is a distinction (e.g. with activations)
      result =
        case op do
          op when is_function(op) ->
            apply(op, args)

          op when is_atom(op) ->
            apply(Axon.Layers, op, args)
        end

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

  defp layer_init_fun(
         template,
         cache,
         result_cache,
         parent_ids,
         name,
         predict_fun,
         parameters,
         %{params: dtype},
         hooks,
         key
       ) do
    {parent_shapes, {parent_params, result_cache, none?}} =
      Enum.map_reduce(parent_ids, {%{}, result_cache, false}, fn
        parent_id, {params, result_cache, none?} ->
          {parent_shape, {params, result_cache}} =
            call_init_cache(parent_id, template, params, cache, result_cache)

          none? = none? or propagating_none?(parent_shape)
          {parent_shape, {params, result_cache, none?}}
      end)

    if none? do
      {%Axon.None{}, {parent_params, result_cache}}
    else
      layer_params =
        Enum.reduce(parameters, %{}, fn param, layer_params ->
          init_param(param, layer_params, parent_shapes, dtype, key)
        end)

      layer_params = apply_hooks(layer_params, :initialize, nil, hooks)

      params =
        if layer_params == %{} do
          parent_params
        else
          Map.put(parent_params, name, layer_params)
        end

      {pred_expr, {_, result_cache}} = predict_fun.(params, template, %{}, cache, result_cache)

      {safe_shape(pred_expr), {params, result_cache}}
    end
  end

  defp init_param(param, layer_params, parent_shapes, dtype, key) do
    %{name: name, shape: shape, initializer: initializer} = param

    fun =
      case shape do
        {:tuple, params} ->
          params
          |> Enum.map(fn shape ->
            shape = apply(shape, parent_shapes)
            apply_initializer(initializer, shape, dtype, key)
          end)
          |> List.to_tuple()

        shape ->
          shape = apply(shape, parent_shapes)
          apply_initializer(initializer, shape, dtype, key)
      end

    Map.put(layer_params, name, fun)
  end

  defp apply_initializer(initializer, shape, type, key) when is_atom(initializer) do
    fun = apply(Axon.Initializers, initializer, [])
    fun.(shape, type, key)
  end

  defp apply_initializer(initializer, shape, type, key) when is_function(initializer, 3) do
    initializer.(shape, type, key)
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

  defp safe_as_type(container_or_tensor, type) do
    case container_or_tensor do
      %Axon.None{} = none ->
        none

      %Nx.Tensor{} = tensor ->
        Nx.as_type(tensor, type)

      container ->
        deep_new(container, &Nx.as_type(&1, type))
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

  defp propagating_none?(%Axon.None{__propagate__: true}), do: true
  defp propagating_none?(_), do: false

  defp us_to_ms(time), do: Float.round(time / 1000, 1)
end
