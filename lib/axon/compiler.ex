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
           parameters: parameters,
           op: op,
           name: name_fn,
           policy: %{params: dtype},
           hooks: hooks
         },
         cache_and_counts
       ) do
    {cache, op_counts} =
      case {op, parents} do
        {:container, [parent]} ->
          deep_reduce(parent, cache_and_counts, &to_init_fun/2)

        {_, parents} when is_list(parents) ->
          Enum.reduce(parents, cache_and_counts, &to_init_fun/2)

        {_, parents} when is_tuple(parents) ->
          deep_reduce(parents, cache_and_counts, &to_init_fun/2)
      end

    case cache do
      %{^id => _} ->
        {cache, op_counts}

      %{} ->
        if Enum.empty?(parameters) do
          {cache, op_counts}
        else
          layer_params =
            Enum.reduce(parameters, %{}, fn param, layer_params ->
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
    %{name: name, shape: shape, initializer: initializer} = param
    fun = apply(Axon.Initializers, initializer, [[type: dtype, shape: shape]])
    Map.put(layer_params, name, fun)
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
           parent: inputs,
           parameters: layer_params,
           args: args,
           opts: opts,
           policy: %{compute: compute, output: output},
           hooks: hooks
         },
         cache_and_counts,
         mode
       )
       when is_function(op) or (op in @axon_layers and is_list(inputs)) do
    # Traverse to accumulate cache and get parent_ids for
    # application within the function. We work only with
    # functions and IDs to avoid leaking entire graphs into
    # the closure
    {parent_ids, {cache, op_counts}} =
      Enum.map_reduce(
        inputs,
        cache_and_counts,
        &to_predict_fun(&1, &2, mode)
      )

    # Names are computed lazily, so compute name from current
    # op and aggregate op_counts.
    name = name_fn.(op, op_counts)
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
            call_cache(parent_id, params, inputs, state, cache, result_cache)

          layer_input =
            layer_input
            |> safe_as_type(compute)
            |> apply_hooks(:pre_forward, mode, hooks)

          {layer_input, {state, result_cache}}
        end)

      # Parameters are just accessed in the layer sub-map of the nested
      # parameter map, so we just need to extract them and then apply
      # freezing and dtype policy
      parameter_inputs =
        Enum.map(layer_params, fn %{name: v, frozen: frz} ->
          safe_as_type(maybe_freeze(params[name][v], frz), compute)
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

          op when op in @axon_layers ->
            apply(Axon.Layers, op, args)
        end

      # Final stage is to extract correct output form by determining if
      # the layer had stateful output, apply hooks, and cast back to policy
      # dtype for outputs
      # TODO: This works well enough for now, but we should consider something
      # more graceful which allows for better error checking of stateful layers
      {out, state} =
        case result do
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

  defp safe_as_type(container_or_tensor, type) do
    case container_or_tensor do
      %Nx.Tensor{} = tensor ->
        Nx.as_type(tensor, type)

      container ->
        deep_new(container, &Nx.as_type(&1, type))
    end
  end
end
