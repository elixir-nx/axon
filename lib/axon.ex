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

  defstruct [:id, :name, :output_shape, :parent, :op, :params, :opts]

  @doc """
  Input node.
  """
  def input(input_shape, opts \\ []) do
    {id, name} = unique_identifiers(:input, opts[:name])
    %Axon{id: id, name: name, output_shape: input_shape, parent: nil, op: :input, params: []}
  end

  @doc """
  Dense node.
  """
  def dense(%Axon{output_shape: parent_shape} = x, units, opts \\ []) do
    {id, name} = unique_identifiers(:dense, opts[:name])

    weight_init = opts[:weight_initializer] || :uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]
    param_shape = {elem(parent_shape, 1), units}
    shape = {elem(parent_shape, 0), units}
    weight = param(name <> "_weight", param_shape, weight_init)
    bias = param(name <> "_bias", {1, units}, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: shape,
      parent: x,
      op: :dense,
      params: [bias, weight]
    }

    if activation do
      node
      |> activation(activation)
    else
      node
    end
  end

  @doc """
  Conv node.
  """
  def conv(%Axon{output_shape: parent_shape} = x, units, opts \\ []) do
    {id, name} = unique_identifiers(:conv, opts[:name])

    kernel_init = opts[:kernel_initializer] || :uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]

    kernel_size =
      opts[:kernel_size] || List.to_tuple(List.duplicate(1, Nx.rank(parent_shape) - 2))

    strides = opts[:strides] || 1
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1

    kernel_shape = Axon.Shape.conv_kernel(parent_shape, units, kernel_size)
    bias_shape = Axon.Shape.conv_bias(parent_shape, units)
    output_shape = Axon.Shape.conv_output(parent_shape, kernel_shape, strides, padding, input_dilation, kernel_dilation)

    kernel = param(name <> "_kernel", kernel_shape, kernel_init)
    bias = param(name <> "_bias", bias_shape, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: :conv,
      params: [bias, kernel],
      opts: [
        strides: strides,
        padding: padding,
        input_dilation: input_dilation,
        kernel_dilation: kernel_dilation
      ]
    }

    if activation do
      node
      |> activation(activation)
    else
      node
    end
  end

  @doc """
  Max pool node.
  """
  def max_pool(%Axon{output_shape: parent_shape} = x, opts \\ []) do
    {id, name} = unique_identifiers(:max_pool, opts[:name])

    kernel_size = opts[:kernel_size] || 1
    strides = opts[:strides] || List.duplicate(1, Nx.rank(parent_shape) - 2)
    padding = opts[:padding] || :valid

    output_shape = Axon.Shape.pool_output(parent_shape, kernel_size, strides, padding)

    node = %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: :max_pool,
      params: [],
      opts: [
        kernel_size: kernel_size,
        strides: strides,
        padding: padding
      ]
    }

    node
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

  # TODO: Wrap all Nx ops so they can be called at any point

  def reshape(%Axon{} = x, new_shape, opts \\ []) do
    id = System.unique_integer([:positive, :monotonic])
    name = opts[:name] || "reshape_#{id}"
    %Axon{id: id, name: name, output_shape: new_shape, parent: x, op: :reshape, params: []}
  end

  def flatten(%Axon{output_shape: shape} = x, opts \\ []) do
    id = System.unique_integer([:positive, :monotonic])
    name = opts[:name] || "flatten_#{id}"
    new_shape = Axon.Shape.flatten(shape)
    %Axon{id: id, name: name, output_shape: new_shape, parent: x, op: :flatten, params: []}
  end

  defp param(name, shape, initializer, _opts \\ []) do
    %Axon.Parameter{name: name, shape: shape, initializer: initializer}
  end

  @doc """
  Defines a new Axon model.
  """
  defmacro model(do: block) do
    # TODO: I don't think this is the best way to determine number of params
    # to match on in predict
    {_, num_params} =
      Macro.prewalk(block, 0, fn
        {:dense, _, _} = node, acc -> {node, acc + 2}
        {:conv, _, _} = node, acc -> {node, acc + 2}
        node, acc -> {node, acc}
      end)

    tuple_args = for _ <- 1..num_params, do: {:_, [], nil}
    predict_args = [{:=, [], [{:{}, [], tuple_args}, {:params, [], nil}]}, {:batch, [], nil}]
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

  @doc """
  Defines a new Axon model.
  """
  defmacro model(call, do: block) do
    {name, _} = Macro.decompose_call(call)

    # TODO: I don't think this is the best way to determine number of params
    # to match on in predict
    {_, num_params} =
      Macro.prewalk(block, 0, fn
        {:dense, _, _} = node, acc -> {node, acc + 2}
        {:conv, _, _} = node, acc -> {node, acc + 2}
        node, acc -> {node, acc}
      end)

    tuple_args = for _ <- 1..num_params, do: {:_, [], nil}
    predict_args = [{:=, [], [{:{}, [], tuple_args}, {:params, [], nil}]}, {:batch, [], nil}]
    predict_signature = {name, [], predict_args}

    quote do
      import Nx.Defn

      @axon_ast unquote(block)

      def unquote(axon_name(name))(), do: @axon_ast

      defn unquote(init_function_name(name))() do
        transform(:ok, fn _ -> Axon.__jit_params__(unquote(axon_name(name))()) end)
      end

      defn unquote(predict_signature) do
        transform({unquote({:params, [], nil}), unquote({:batch, [], nil})}, fn {params, input} ->
          Axon.__jit_predict__(unquote(axon_name(name))(), params, input)
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

    names_and_exprs
    |> Map.values()
    |> Enum.reverse()
    |> List.to_tuple()
  end

  defp to_param_expr(%Axon{parent: nil, params: params}, names_and_exprs, counter) do
    Enum.reduce(params, {names_and_exprs, counter}, fn %Axon.Parameter{
                                                         name: name,
                                                         shape: shape,
                                                         initializer: initializer
                                                       },
                                                       {names_and_exprs, counter} ->
      {
        Map.put(
          names_and_exprs,
          "#{counter}_" <> name,
          apply(Axon.Initializers, initializer, [[shape: shape]])
        ),
        counter + 1
      }
    end)
  end

  defp to_param_expr(%Axon{parent: parent, params: params}, names_and_exprs, counter) do
    {names_and_exprs, counter} =
      Enum.reduce(params, {names_and_exprs, counter}, fn %Axon.Parameter{
                                                           name: name,
                                                           shape: shape,
                                                           initializer: initializer
                                                         },
                                                         {names_and_exprs, counter} ->
        {
          Map.put(
            names_and_exprs,
            "#{counter}_" <> name,
            apply(Axon.Initializers, initializer, [[shape: shape]])
          ),
          counter + 1
        }
      end)

    to_param_expr(parent, names_and_exprs, counter)
  end

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(graph, params, input) do
    to_predict_expr(graph, Enum.reverse(Tuple.to_list(params)), input)
  end

  @activation_layers [:relu, :softmax, :tanh, :log_softmax]

  defp to_predict_expr(%Axon{op: op, parent: parent}, params, input)
       when op in @activation_layers do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Activations, op, [expr])
  end

  defp to_predict_expr(%Axon{op: :dense, parent: parent}, [b, w | params], input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :dense, [expr, w, b])
  end

  defp to_predict_expr(%Axon{op: :conv, parent: parent, opts: opts}, [b, w | params], input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :conv, [expr, w, b, opts])
  end

  defp to_predict_expr(%Axon{op: :max_pool, parent: parent, opts: opts}, params, input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :max_pool, [expr, opts])
  end

  defp to_predict_expr(%Axon{op: :reshape, output_shape: shape, parent: parent}, params, input) do
    expr = to_predict_expr(parent, params, input)
    apply(Nx, :reshape, [expr, shape])
  end

  defp to_predict_expr(%Axon{op: :flatten, output_shape: shape, parent: parent}, params, input) do
    expr = to_predict_expr(parent, params, input)
    apply(Axon.Layers, :flatten, [expr])
  end

  defp to_predict_expr(%Axon{op: :input, parent: nil}, _params, input) do
    input
  end

  defp init_function_name(name) do
    String.to_atom("init_" <> Atom.to_string(name))
  end

  defp axon_name(name) do
    String.to_atom("__" <> Atom.to_string(name) <> "_axon__")
  end

  ## Helpers

  defp unique_identifiers(type, nil) do
    id = System.unique_integer([:positive, :monotonic])
    {id, Atom.to_string(type) <> "_#{id}"}
  end

  defp unique_identifiers(_type, name), do: {System.unique_integer([:positive, :monotonic]), name}
end
