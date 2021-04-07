defmodule Axon.Compiler do
  @moduledoc false

  import Axon.Shared

  ## Init JIT Compilation

  @doc false
  def __compile__(%Axon{} = graph) do
    {compile_init(graph), compile_predict(graph)}
  end

  @doc false
  def __jit_init__(%Axon{} = graph, caller, [] = args, opts) do
    fun = compile_init(graph)
    jit_or_apply(caller, fun, args, opts)
  end

  defp compile_init(%Axon{} = graph) do
    fn ->
      graph
      |> to_init_fun(%{})
      |> Enum.sort()
      |> Enum.unzip()
      |> Kernel.elem(1)
      |> Enum.map(& &1.())
      |> List.to_tuple()
    end
  end

  defp to_init_fun(%Axon{parent: parent, params: params}, cache) do
    cache =
      Enum.reduce(params, cache, fn
        %{id: id} = param, cache ->
          case cache do
            %{^id => _} ->
              cache

            %{} ->
              %{id: id, shape: shape, initializer: initializer} = param
              fun = fn -> apply(Axon.Initializers, initializer, [[shape: shape]]) end
              Map.put(cache, id, fun)
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
  def __jit_predict__(%Axon{} = graph, caller, args, opts) do
    fun = compile_predict(graph)
    jit_or_apply(caller, fun, args, opts)
  end

  defp compile_predict(%Axon{} = graph) do
    fn params, inputs ->
      {fun, _} = to_predict_fun(graph, %{})
      fun.(params, inputs)
    end
  end

  defp to_predict_fun(%{id: id} = graph, cache) do
    case cache do
      %{^id => fun} ->
        {fun, cache}

      %{} ->
        {fun, cache} = recur_predict_fun(graph, cache)
        cache = Map.put(cache, id, fun)
        {fun, cache}
    end
  end

  ## Activation Layers

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  defp recur_predict_fun(%Axon{op: op, parent: parent}, cache)
       when op in @activation_layers do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      apply(Axon.Activations, op, [fun.(params, inputs)])
    end

    {fun, cache}
  end

  ## Linear Layers

  defp recur_predict_fun(%Axon{op: :dense, parent: parent}, cache) do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      param_size = tuple_size(params)
      {w, b} = {elem(params, param_size - 2), elem(params, param_size - 1)}

      params =
        params
        |> Tuple.delete_at(param_size - 1)
        |> Tuple.delete_at(param_size - 2)

      apply(Axon.Layers, :dense, [fun.(params, inputs), w, b])
    end

    {fun, cache}
  end

  ## Pooling Layers

  @pooling_layers [:max_pool, :avg_pool, :lp_pool, :adaptive_avg_pool, :adaptive_max_pool]

  defp recur_predict_fun(%Axon{op: op, parent: parent, opts: opts}, cache)
       when op in @pooling_layers do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      apply(Axon.Layers, op, [fun.(params, inputs), opts])
    end

    {fun, cache}
  end

  ## Dropout Layers

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  defp recur_predict_fun(%Axon{op: op, parent: parent, opts: opts}, cache)
       when op in @dropout_layers do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      apply(Axon.Layers, op, [fun.(params, inputs), opts])
    end

    {fun, cache}
  end

  ## Conv Layers

  @conv_layers [:conv, :depthwise_conv]

  defp recur_predict_fun(%Axon{op: op, parent: parent, opts: opts}, cache)
       when op in @conv_layers do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      param_size = tuple_size(params)
      {w, b} = {elem(params, param_size - 2), elem(params, param_size - 1)}

      params =
        params
        |> Tuple.delete_at(param_size - 1)
        |> Tuple.delete_at(param_size - 2)

      apply(Axon.Layers, op, [fun.(params, inputs), w, b, opts])
    end

    {fun, cache}
  end

  defp recur_predict_fun(%Axon{op: :separable_conv2d, parent: parent, opts: opts}, cache) do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      param_size = tuple_size(params)

      {w1, b1, w2, b2} = {
        elem(params, param_size - 4),
        elem(params, param_size - 3),
        elem(params, param_size - 2),
        elem(params, param_size - 1)
      }

      params =
        params
        |> Tuple.delete_at(param_size - 1)
        |> Tuple.delete_at(param_size - 2)
        |> Tuple.delete_at(param_size - 3)
        |> Tuple.delete_at(param_size - 4)

      apply(Axon.Layers, :separable_conv2d, [fun.(params, inputs), w1, b1, w2, b2, opts])
    end

    {fun, cache}
  end

  defp recur_predict_fun(%Axon{op: :separable_conv3d, parent: parent, opts: opts}, cache) do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      param_size = tuple_size(params)

      {w1, b1, w2, b2, w3, b3} = {
        elem(params, param_size - 6),
        elem(params, param_size - 5),
        elem(params, param_size - 4),
        elem(params, param_size - 3),
        elem(params, param_size - 2),
        elem(params, param_size - 1)
      }

      params =
        params
        |> Tuple.delete_at(param_size - 1)
        |> Tuple.delete_at(param_size - 2)
        |> Tuple.delete_at(param_size - 3)
        |> Tuple.delete_at(param_size - 4)

      apply(Axon.Layers, :separable_conv3d, [fun.(params, inputs), w1, b1, w2, b2, w3, b3, opts])
    end

    {fun, cache}
  end

  ## Normalization Layers

  @normalization_layers [:batch_norm, :layer_norm, :group_norm, :instance_norm]

  defp recur_predict_fun(%Axon{op: op, parent: parent, opts: opts}, cache)
       when op in @normalization_layers do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      param_size = tuple_size(params)
      {w, b} = {elem(params, param_size - 2), elem(params, param_size - 1)}

      params =
        params
        |> Tuple.delete_at(param_size - 1)
        |> Tuple.delete_at(param_size - 2)

      apply(Axon.Layers, op, [fun.(params, inputs), w, b, opts])
    end

    {fun, cache}
  end

  ## Shape Layers

  defp recur_predict_fun(%Axon{op: :flatten, parent: parent}, cache) do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      apply(Axon.Layers, :flatten, [fun.(params, inputs)])
    end

    {fun, cache}
  end

  ## Special Layers

  defp recur_predict_fun(%Axon{op: :nx, parent: parent, opts: [fun: nx_fun]}, cache) do
    {fun, cache} = to_predict_fun(parent, cache)

    fun = fn params, inputs ->
      nx_fun.(fun.(params, inputs))
    end

    {fun, cache}
  end

  defp recur_predict_fun(%Axon{op: :input}, cache) do
    fun = fn _, inputs ->
      inputs
    end

    {fun, cache}
  end
end
