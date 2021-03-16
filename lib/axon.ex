defmodule Axon do
  @moduledoc """
  Nx-powered Neural Networks.

  Axon provides a high-level interface for creating neural
  networks in Elixir. Axon is built entirely on top of Nx
  numerical definitions, so every neural network can be
  JIT or AOT compiled using any Nx compiler, or even
  transformed into high-level neural network formats
  like TensorFlow Lite and ONNX.
  """
  alias __MODULE__, as: Axon

  @type t :: %__MODULE__{}

  defstruct [:id, :name, :output_shape, :parent, :op, :params]

  @doc """
  Input node.
  """
  def input(input_shape, opts \\ []) do
    id = System.unique_integer([:positive])
    name = opts[:name] || "input_#{id}"
    %Axon{id: id, name: name, output_shape: input_shape, parent: nil, op: :input, params: []}
  end

  @doc """
  Dense node.
  """
  def dense(%Axon{output_shape: parent_shape} = x, units, opts \\ []) do
    id = System.unique_integer([:positive, :monotonic])
    name = opts[:name] || "dense_#{id}"
    weight_init = opts[:weight_initializer] || :uniform
    bias_init = opts[:bias_initializer] || :zeros
    param_shape = {elem(parent_shape, 1), units}
    shape = {elem(parent_shape, 0), units}
    weight = __param__(name <> "_weight", param_shape, weight_init)
    bias = __param__(name <> "_bias", {1, units}, bias_init)
    %Axon{id: id, name: name, output_shape: shape, parent: x, op: :dense, params: [bias, weight]}
  end

  @doc """
  Activation node.
  """
  def activation(%Axon{output_shape: shape} = x, activation, opts \\ [])
      when is_atom(activation) do
    id = System.unique_integer([:positive, :monotonic])
    name = opts[:name] || "#{Atom.to_string(activation)}_#{id}"
    %Axon{id: id, name: name, output_shape: shape, parent: x, op: activation, params: []}
  end

  @doc """
  Model parameter.
  """
  def __param__(name, shape, initializer, _opts \\ []) do
    %Axon.Parameter{name: name, shape: shape, initializer: initializer}
  end

  @doc """
  Defines a new Axon model.
  """
  defmacro model(do: block) do
    # TODO: I don't think this is the best way to determine number of params
    # to match on in predict
    {_, num_params} = Macro.prewalk(block, 0,
      fn
        {:dense, _, _} = node, acc -> {node, acc + 2}
        node, acc -> {node, acc}
      end)

    tuple_args = for _ <- 1..num_params, do: {:_, [], nil}
    predict_args = [{:=, [], [{:{}, [], tuple_args}, {:params, [line: 26], nil}]}, {:batch, [], nil}]
    predict_signature = {:predict, [], predict_args}

    quote do
      import Nx.Defn

      @axon_ast unquote(block)

      def __axon__, do: @axon_ast

      defn init_random_params() do
        transform(:ok, fn _ -> Axon.__jit_params__(__axon__()) end)
      end

      defn unquote(predict_signature) do
        transform({unquote({:params, [], nil}), unquote({:batch, [], nil})}, fn {params, input} ->
          Axon.__jit_predict__(__axon__(), params, input)
        end)
      end
    end
  end

  defmacro __using__(_opts) do
    quote do
      require Axon
      import Axon
      import Nx.Defn
    end
  end

  ## Parameter JIT Compilation

  # TODO: This is a pretty fragile way of enforcing parameter ordering.
  # Need a more coherent strategy

  @doc false
  def __jit_params__(graph) do
    {names_and_exprs, _} = to_param_expr(graph, %{}, 0)
    IO.inspect names_and_exprs
    names_and_exprs
    |> Map.values()
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp to_param_expr(%Axon{parent: nil, params: params}, names_and_exprs, counter) do
    Enum.reduce(params, {names_and_exprs, counter},
      fn %Axon.Parameter{name: name, shape: shape, initializer: initializer}, {names_and_exprs, counter} ->
        {
          Map.put(names_and_exprs, "#{counter}_" <> name, apply(Axon.Initializers, initializer, [[shape: shape]])),
          counter+1
        }
      end
    )
  end

  defp to_param_expr(%Axon{parent: parent, params: params}, names_and_exprs, counter) do
    {names_and_exprs, counter} =
      Enum.reduce(params, {names_and_exprs, counter},
        fn %Axon.Parameter{name: name, shape: shape, initializer: initializer}, {names_and_exprs, counter} ->
          {
            Map.put(names_and_exprs, "#{counter}_" <> name, apply(Axon.Initializers, initializer, [[shape: shape]])),
            counter+1
          }
        end
      )
    to_param_expr(parent, names_and_exprs, counter)
  end

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(graph, params, input) do
    to_predict_expr(graph, Enum.reverse(Tuple.to_list(params)), input)
  end

  @activation_layers [:relu, :softmax, :log_softmax]

  defp to_predict_expr(%Axon{op: op, parent: parent}, params, input) when op in @activation_layers do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Activations, op, [expr])
  end

  defp to_predict_expr(%Axon{op: :dense, parent: parent}, [b, w | params], input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :dense, [expr, w, b])
  end

  defp to_predict_expr(%Axon{op: :input, parent: nil}, _params, input) do
    input
  end
end
