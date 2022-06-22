defmodule Axon.Compiler do
  @moduledoc false
  require Logger

  import Axon.Shared
  alias Axon.StatefulOutput

  ## Init JIT Compilation

  @doc false
  def compile_init(%Axon{} = graph, opts) do
    {root_id, {cache, _op_counts}} = to_init_fun(graph, {%{}, %{}})

    init_fn = fn init_params ->
      params = cache[root_id].(cache, %{})
      # TODO: Maybe merge is not best here, do we want this
      # operation to fail if structure of params do not match
      # or should it just silently continue (even though the
      # behavior is not correct)?
      Map.merge(params, init_params)
    end

    &Nx.Defn.jit_or_apply(init_fn, [&1], opts)
  end

  def compile_init(graph, _opts) do
    raise ArgumentError,
          "attempting to compile initialization function from" <>
            " an unrecognized graph #{inspect(graph)}, if you" <>
            " are attempting to initialize a model with a container" <>
            " output, use `Axon.container`"
  end

  defp to_init_fun(%{id: id} = graph, {cache, op_counts}) do
    case cache do
      %{^id => _} ->
        {id, {cache, op_counts}}

      %{} ->
        recur_init_fun(graph, {cache, op_counts})
    end
  end

  defp call_init_cache(parent_id, cache, result_cache) do
    key = {:cache, parent_id}

    case result_cache do
      %{^key => _} ->
        result_cache

      %{} ->
        cache[parent_id].(cache, result_cache)
    end
  end

  defp recur_init_fun(
         %Axon{
           id: id,
           parent: parents,
           parameters: parameters,
           op: op,
           name: name_fn,
           policy: %{params: dtype},
           hooks: hooks,
           op_name: op_name
         },
         {cache, op_counts} = cache_and_counts
       ) do
    {parent_ids, {cache, op_counts}} =
      case {op, parents} do
        {:container, [parent]} ->
          deep_map_reduce(parent, cache_and_counts, &to_init_fun/2)

        {:namespace, parents} ->
          input_count = op_counts[:input] || 0
          op_counts = %{input: input_count}

          {parent_ids, {cache, namespace_op_counts}} =
            Enum.map_reduce(parents, {cache, op_counts}, &to_init_fun/2)

          input_count = namespace_op_counts[:input] || 0
          {parent_ids, {cache, Map.put(op_counts, :input, input_count)}}

        {_, parents} when is_list(parents) ->
          Enum.map_reduce(parents, cache_and_counts, &to_init_fun/2)
      end

    name = name_fn.(op_name, op_counts)
    op_counts = Map.update(op_counts, op_name, 1, fn x -> x + 1 end)

    init_fn = fn cache, result_cache ->
      result_cache =
        case op do
          :container ->
            deep_reduce(parent_ids, result_cache, fn parent_id, result_cache ->
              call_init_cache(parent_id, cache, result_cache)
            end)

          :namespace ->
            namespace_params =
              Enum.reduce(parent_ids, %{}, fn parent_id, result_cache ->
                call_init_cache(parent_id, cache, result_cache)
              end)

            if namespace_params == %{} do
              result_cache
            else
              Map.put(result_cache, name, namespace_params)
            end

          _ ->
            Enum.reduce(parent_ids, result_cache, fn parent_id, result_cache ->
              call_init_cache(parent_id, cache, result_cache)
            end)
        end

      layer_params =
        Enum.reduce(parameters, %{}, fn param, layer_params ->
          init_param(param, layer_params, dtype)
        end)

      layer_params = apply_hooks(layer_params, :initialize, nil, hooks)

      case parameters do
        [] ->
          result_cache

        [_ | _] ->
          Map.put(result_cache, name, layer_params)
      end
    end

    {id, {Map.put(cache, id, init_fn), op_counts}}
  end

  defp init_param(param, layer_params, dtype) do
    %{name: name, shape: shape, initializer: initializer} = param

    fun =
      case shape do
        {:tuple, params} ->
          params
          |> Enum.map(fn shape ->
            apply_initializer(initializer, shape, dtype)
          end)
          |> List.to_tuple()

        shape ->
          apply_initializer(initializer, shape, dtype)
      end

    Map.put(layer_params, name, fun)
  end

  defp apply_initializer(initializer, shape, type) when is_atom(initializer) do
    fun = apply(Axon.Initializers, initializer, [])
    fun.(shape, type)
  end

  defp apply_initializer(initializer, shape, type) when is_function(initializer, 2) do
    initializer.(shape, type)
  end

  ## Model JIT Compilation

  @doc false
  def compile_predict(%Axon{} = graph, opts) do
    {mode, opts} = Keyword.pop(opts, :mode, :inference)
    {root_id, {cache, _op_counts, _namespace}} = to_predict_fun(graph, {%{}, %{}, []}, mode)

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

    &Nx.Defn.jit_or_apply(predict_fn, [&1, &2], opts)
  end

  def compile_predict(graph, _opts) do
    raise ArgumentError,
          "attempting to compile predict function from" <>
            " an unrecognized graph #{inspect(graph)}, if you" <>
            " are attempting to initialize a model with a container" <>
            " output, use `Axon.container`"
  end

  defp to_predict_fun(%{id: id} = graph, {cache, op_counts, namespace}, mode) do
    case cache do
      %{^id => _} ->
        {id, {cache, op_counts, namespace}}

      %{} ->
        recur_predict_fun(graph, {cache, op_counts, namespace}, mode)
    end
  end

  defp call_predict_cache(parent_id, params, inputs, state, cache, result_cache) do
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

  ## Special Layers

  defp recur_predict_fun(
         %Axon{id: id, op: :constant, opts: [value: tensor], policy: %{output: output}},
         {cache, op_counts, namespace},
         _
       ) do
    tensor = Nx.backend_transfer(tensor, Nx.BinaryBackend)

    fun = fn _params, _inputs, state, _cache, result_cache ->
      out = safe_as_type(tensor, output)
      {out, {state, result_cache}}
    end

    op_counts = Map.update(op_counts, :constant, 1, fn x -> x + 1 end)

    {id, {Map.put(cache, id, fun), op_counts, namespace}}
  end

  defp recur_predict_fun(
         %Axon{
           id: id,
           op: :input,
           output_shape: shape,
           hooks: hooks,
           name: name_fn,
           opts: [default: default]
         },
         {cache, op_counts, namespace},
         mode
       ) do
    name = name_fn.(:input, op_counts)
    op_counts = Map.update(op_counts, :input, 1, fn x -> x + 1 end)

    default =
      case default do
        nil ->
          nil

        :no_default_value ->
          :no_default_value

        fun when is_function(fun, 1) ->
          fun

        %Nx.Tensor{} = tensor ->
          Nx.backend_transfer(tensor, Nx.BinaryBackend)
      end

    fun = fn _params, inputs, state, _cache, result_cache ->
      res =
        case inputs do
          %Nx.Tensor{} = inputs ->
            inputs

          %{} = inputs ->
            cond do
              Map.has_key?(inputs, name) ->
                inputs[name]

              is_container_shape(shape) ->
                inputs

              true ->
                nil
            end

          inputs when is_tuple(inputs) ->
            inputs

          _ ->
            raise ArgumentError,
                  "invalid input given to model, expected input" <>
                    " expected input to be a tensor or a map" <>
                    " corresponding to correct input names"
        end

      value =
        case {res, default} do
          {nil, :no_default_value} ->
            raise ArgumentError,
                  "unable to find input #{name} for model given to predict," <>
                    " you must provide an input tensor for every input" <>
                    " specified in the graph"

          {nil, nil} ->
            Logger.debug("Input #{name} not provided, and default value is nil")
            nil

          {nil, %Nx.Tensor{} = default} ->
            default

          {nil, fun} when is_function(fun, 1) ->
            fun.(inputs)

          {value, _default} ->
            value
        end

      validate_input_shape!(value, shape)

      res =
        value
        |> apply_hooks(:forward, mode, hooks)
        |> apply_hooks(:backward, mode, hooks)

      {res, {state, result_cache}}
    end

    {id, {Map.put(cache, id, fun), op_counts, namespace}}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :container, parent: [parents]},
         cache_counts_namespace,
         mode
       ) do
    {parent_ids, {cache, op_counts, namespace}} =
      deep_map_reduce(parents, cache_counts_namespace, &to_predict_fun(&1, &2, mode))

    op_counts = Map.update(op_counts, :container, 1, fn x -> x + 1 end)

    fun = fn params, inputs, state, cache, result_cache ->
      deep_map_reduce(parent_ids, {state, result_cache}, fn parent_id, {state, result_cache} ->
        call_predict_cache(parent_id, params, inputs, state, cache, result_cache)
      end)
    end

    {id, {Map.put(cache, id, fun), op_counts, namespace}}
  end

  defp recur_predict_fun(
         %Axon{id: id, op: :namespace, name: name_fn, parent: parents},
         {cache, op_counts, namespace},
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
    {parent_ids, {cache, namespace_op_counts, _namespace}} =
      Enum.map_reduce(
        parents,
        {cache, namespace_op_counts, [name | namespace]},
        &to_predict_fun(&1, &2, mode)
      )

    # Update the global op_count of input layers, since they
    # are a global operation regardless of where they are
    input_count = namespace_op_counts[:input] || 0
    op_counts = Map.put(op_counts, :input, input_count)

    # The function just returns the result of it's child,
    # or parent depending on how you view the tree
    fun = fn params, inputs, state, cache, result_cache ->
      # TODO: How should hooks be handled here?
      # TODO: I think we can actually handle parameter freezing and access
      # better here by only forwarding params[namespace] to the child function
      {[out], {state, result_cache}} =
        Enum.map_reduce(parent_ids, {state, result_cache}, fn parent_id, {state, result_cache} ->
          call_predict_cache(parent_id, params, inputs, state, cache, result_cache)
        end)

      {out, {state, result_cache}}
    end

    # Then we return the cache, op_counts, and original namespace
    {id, {Map.put(cache, id, fun), op_counts, namespace}}
  end

  ## Every other case

  defp recur_predict_fun(
         %Axon{
           id: id,
           name: name_fn,
           op: op,
           parent: inputs,
           parameters: layer_params,
           args: args,
           opts: opts,
           policy: %{compute: compute, output: output},
           hooks: hooks,
           op_name: op_name
         },
         cache_counts_namespace,
         mode
       )
       when (is_function(op) or is_atom(op)) and is_list(inputs) do
    # Traverse to accumulate cache and get parent_ids for
    # application within the function. We work only with
    # functions and IDs to avoid leaking entire graphs into
    # the closure
    {parent_ids, {cache, op_counts, namespace}} =
      Enum.map_reduce(
        inputs,
        cache_counts_namespace,
        &to_predict_fun(&1, &2, mode)
      )

    # Names are computed lazily, so compute name from current
    # op and aggregate op_counts.
    name = name_fn.(op_name, op_counts)
    op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)

    # Sub-inference functions contain `params` - trainable parameters
    # passed to `predict`, `inputs` - inputs passed to `predict`, `state` -
    # an accumulator of layer state during the recursive building of the
    # inference function, `cache` - the built function cache for accessing
    # previous layer expressions, and `result_cache` - cached results to
    # avoid recomputing expressions in combined graphs.
    fun = fn params, inputs, state, cache, result_cache ->
      # Recurse graph inputs and invoke cache to get parent results,
      # state, and result_cache and then apply dtype policy and hooks
      # to each input
      {layer_inputs, {state, result_cache}} =
        Enum.map_reduce(parent_ids, {state, result_cache}, fn parent_id, {state, result_cache} ->
          {layer_input, {state, result_cache}} =
            call_predict_cache(parent_id, params, inputs, state, cache, result_cache)

          layer_input =
            layer_input
            |> safe_as_type(compute)
            |> apply_hooks(:pre_forward, mode, hooks)

          {layer_input, {state, result_cache}}
        end)

      # We're only concerned with this namespaces parameters, so we pair
      # down parameters first given the namespaces we've accumulated at
      # this level (if any)
      # TODO: This should be a namespace concern, not layer concern
      namespace_params =
        namespace
        |> Enum.reverse()
        |> Enum.reduce(params, fn name, params -> params[name] end)

      # Parameters are just accessed in the layer sub-map of the nested
      # parameter map, so we just need to extract them and then apply
      # freezing and dtype policy
      parameter_inputs =
        Enum.map(layer_params, fn %{name: v, frozen: frz} ->
          safe_as_type(maybe_freeze(namespace_params[name][v], frz), compute)
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

    {id, {Map.put(cache, id, fun), op_counts, namespace}}
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

  defp safe_as_type(container_or_tensor, type) do
    case container_or_tensor do
      nil ->
        nil

      %Nx.Tensor{} = tensor ->
        Nx.as_type(tensor, type)

      container ->
        deep_new(container, &Nx.as_type(&1, type))
    end
  end

  defp validate_input_shape!(nil, _), do: :ok

  defp validate_input_shape!(value, shape) do
    unless value == nil or compatible?(value, shape) do
      raise ArgumentError,
            "invalid input shape given to model, expected input" <>
              " with shape #{inspect(shape)}, but got input with" <>
              " shape #{inspect(deep_new(value, &Nx.shape/1))}"
    end
  end

  defp compatible?(%Nx.Tensor{} = value, shape),
    do: Axon.Shape.compatible?(Nx.shape(value), shape)

  defp compatible?(value, shape) do
    template_shapes = recur_shape_to_template(shape)

    merged_check =
      deep_merge(value, template_shapes, fn lhs, rhs ->
        if Axon.Shape.compatible?(Nx.shape(lhs), Nx.shape(rhs)) do
          Nx.tensor(1, type: {:u, 8})
        else
          Nx.tensor(0, type: {:u, 8})
        end
      end)

    merged_bool =
      deep_reduce(merged_check, [], fn x, acc ->
        [x == Nx.tensor(1, type: {:u, 8}) | acc]
      end)

    Enum.all?(merged_bool)
  end

  defp recur_shape_to_template(shape) do
    cond do
      is_tuple(shape) and tuple_size(shape) == 0 ->
        Nx.template(shape, {:f, 32})

      is_tuple(shape) and is_dim(elem(shape, 0)) ->
        Nx.template(shape, {:f, 32})

      true ->
        {templates, :ok} =
          Nx.Container.traverse(shape, :ok, fn shape, :ok ->
            template = recur_shape_to_template(shape)
            {template, :ok}
          end)

        templates
    end
  end

  defp is_container_shape(shape) when is_map(shape), do: true

  defp is_container_shape(shape) when is_tuple(shape) do
    if tuple_size(shape) == 0 do
      true
    else
      not is_dim(elem(shape, 0))
    end
  end

  defp is_dim(dim) when is_integer(dim) or is_nil(dim), do: true
  defp is_dim(_), do: false
end
