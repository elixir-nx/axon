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
    weight_init = opts[:weight_initializer] || :glorot_normal
    bias_init = opts[:bias_initializer] || :zeros
    param_shape = {elem(parent_shape, 1), units}
    shape = {elem(parent_shape, 0), units}
    weight = __param__(name <> "_weight", param_shape, weight_init)
    bias = __param__(name <> "_bias", {1, units}, bias_init)
    %Axon{id: id, name: name, output_shape: shape, parent: x, op: :dense, params: [weight, bias]}
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
    quote do
      import Nx.Defn

      @axon_ast unquote(block)

      def __axon__, do: @axon_ast

      defn initialize() do
        transform(:ok, fn _ -> Axon.__jit_params__(__axon__()) end)
      end
    end
  end

  ## Parameter JIT Compilation

  @doc false
  def __jit_params__(graph) do
    names_and_exprs = to_param_expr(graph, %{})
    List.to_tuple(Map.values(names_and_exprs))
  end

  defp to_param_expr(%Axon{parent: nil, params: params}, names_and_exprs) do
    Enum.reduce(params, names_and_exprs,
      fn %Axon.Parameter{name: name, shape: shape, initializer: initializer}, acc ->
        Map.put(acc, name, apply(Axon.Initializers, initializer, [[shape: shape]]))
      end
    )
  end

  defp to_param_expr(%Axon{parent: parent, params: params}, names_and_exprs) do
    names_and_exprs =
      Enum.reduce(params, names_and_exprs,
        fn %Axon.Parameter{name: name, shape: shape, initializer: initializer}, acc ->
          Map.put(acc, name, apply(Axon.Initializers, initializer, [[shape: shape]]))
        end
      )
    to_param_expr(parent, names_and_exprs)
  end
end
