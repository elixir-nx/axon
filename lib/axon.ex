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

  defstruct [:id, :name, :output_shape, :parent, :op, :params, :vars, :opts]

  # TODO: Allow arbitrary calls to Nx functions and numerical definitions inside `model`
  # TODO: Models should be easily composable inside other model blocks
  # TODO: Unify parameters and variables under `%Axon.State{}`
  # TODO: Add rest of `Axon.Layers`

  @doc """
  Adds an input layer to the network.

  Input layers specify a models inputs. For example, a model
  with two `input` layers would have the following predict function:

      defn predict(params, input_1, input_2) do
        ...
      end

  Input layers are always the root layers of the neural network.

  ## Options

    * `name` - Layer name.

  """
  def input(input_shape, opts \\ []) do
    {id, name} = unique_identifiers(:input, opts[:name])

    %Axon{
      id: id,
      name: name,
      output_shape: input_shape,
      parent: nil,
      op: :input,
      params: [],
      vars: []
    }
  end

  @doc """
  Adds a dense layer to the network.

  The dense layer implements:

      output = activation(dot(input, kernel) + bias)

  where `activation` is given by the `:activation` option and both
  `kernel` and `bias` are layer parameters. `units` specifies the
  number of output units.

  Compiles to `Axon.Layers.dense/3`.

  ## Options

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.

  """
  def dense(%Axon{output_shape: parent_shape} = x, units, opts \\ []) do
    {id, name} = unique_identifiers(:dense, opts[:name])

    weight_init = opts[:kernel_initializer] || :uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]
    param_shape = {elem(parent_shape, 1), units}
    shape = {elem(parent_shape, 0), units}
    weight = stateful(name <> "_weight", param_shape, weight_init)
    bias = stateful(name <> "_bias", {1, units}, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: shape,
      parent: x,
      op: :dense,
      params: [bias, weight],
      vars: []
    }

    if activation do
      node
      |> activation(activation)
    else
      node
    end
  end

  @doc """
  Adds a convolution layer to the network.

  The convolution layer implements a general dimensional
  convolutional layer - which convolves a kernel over the input
  to produce an output.

  Compiles to `Axon.Layers.conv/3`.

  ## Options

    * `name` - Layer name.
    * `kernel_initializer` - Initializer for `kernel` weights.
    * `bias_initializer` - Initializer for `bias` weights.
    * `activation` - Element-wise activation function.
    * `kernel_size` - Size of the kernel spatial dimensions.
    * `strides` - Stride during convolution.
    * `padding` - Padding to the spatial dimensions of the input.
    * `input_dilation` - Dilation to apply to input.
    * `kernel_dilation` - Dilation to apply to kernel.

  """
  def conv(%Axon{output_shape: parent_shape} = x, units, opts \\ []) do
    {id, name} = unique_identifiers(:conv, opts[:name])

    kernel_init = opts[:kernel_initializer] || :uniform
    bias_init = opts[:bias_initializer] || :zeros
    activation = opts[:activation]

    kernel_size =
      opts[:kernel_size] || List.to_tuple(List.duplicate(1, Nx.rank(parent_shape) - 2))

    strides = opts[:strides] || List.duplicate(1, Nx.rank(parent_shape) - 2)
    padding = opts[:padding] || :valid
    input_dilation = opts[:input_dilation] || 1
    kernel_dilation = opts[:kernel_dilation] || 1

    kernel_shape = Axon.Shape.conv_kernel(parent_shape, units, kernel_size)
    bias_shape = Axon.Shape.conv_bias(parent_shape, units)

    output_shape =
      Axon.Shape.conv_output(
        parent_shape,
        kernel_shape,
        strides,
        padding,
        input_dilation,
        kernel_dilation
      )

    kernel = stateful(name <> "_kernel", kernel_shape, kernel_init)
    bias = stateful(name <> "_bias", bias_shape, bias_init)

    node = %Axon{
      id: id,
      name: name,
      output_shape: output_shape,
      parent: x,
      op: :conv,
      params: [bias, kernel],
      vars: [],
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
  Adds an activation layer to the network.

  Activation layers are element-wise functions typically called
  after the output of another layer.

  ## Options

    - `name` - Layer name.

  """
  def activation(%Axon{output_shape: shape} = x, activation, opts \\ [])
      when is_atom(activation) do
    id = System.unique_integer([:positive, :monotonic])
    name = opts[:name] || "#{Atom.to_string(activation)}_#{id}"

    %Axon{
      id: id,
      name: name,
      output_shape: shape,
      parent: x,
      op: activation,
      params: [],
      vars: []
    }
  end

  ## Activation

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :log_softmax, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  for activation <- @activation_layers do
    @doc """
    Adds #{Atom.to_string(activation)} activation layer to the network.

    See `Axon.Activations.#{Atom.to_string(activation)}/1` for more details.

    ## Options

      - `name` - Layer name.

    """
    def unquote(activation)(%Axon{} = x, opts \\ []), do: activation(x, unquote(activation), opts)
  end

  ## Dropout

  # TODO: Need a way to turn these layers off

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  for dropout <- @dropout_layers do
    def unquote(dropout)(%Axon{output_shape: parent_shape} = x, opts \\ []) do
      {id, name} = unique_identifiers(unquote(dropout), opts[:name])

      rate = opts[:rate] || 0.5

      node = %Axon{
        id: id,
        name: name,
        op: unquote(dropout),
        output_shape: parent_shape,
        parent: x,
        params: [],
        vars: [],
        opts: [
          rate: rate
        ]
      }

      node
    end
  end

  ## Pooling

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  for pool <- @pooling_layers do
    def unquote(pool)(%Axon{output_shape: parent_shape} = x, opts \\ []) do
      {id, name} = unique_identifiers(unquote(pool), opts[:name])

      kernel_size = opts[:kernel_size] || 1
      strides = opts[:strides] || List.duplicate(1, Nx.rank(parent_shape) - 2)
      padding = opts[:padding] || :valid

      output_shape = Axon.Shape.pool_output(parent_shape, kernel_size, strides, padding)

      node = %Axon{
        id: id,
        name: name,
        output_shape: output_shape,
        parent: x,
        op: unquote(pool),
        params: [],
        vars: [],
        opts: [
          kernel_size: kernel_size,
          strides: strides,
          padding: padding
        ]
      }

      node
    end
  end

  @doc """
  Batch norm layer.
  """
  def batch_norm(%Axon{output_shape: shape} = x, opts \\ []) do
    {id, name} = unique_identifiers(:batch_norm, opts[:name])

    gamma_initializer = opts[:gamma_initializer] || :uniform
    beta_initializer = opts[:beta_initializer] || :zeros

    gamma = stateful(name <> "_gamma", {1, elem(shape, 1)}, gamma_initializer)
    beta = stateful(name <> "_beta", {1, elem(shape, 1)}, beta_initializer)

    ra_mean = stateful(name <> "_mean", {1, elem(shape, 1)}, :zeros)
    ra_var = stateful(name <> "_variance", {1, elem(shape, 1)}, :zeros)

    node = %Axon{
      id: id,
      name: name,
      output_shape: shape,
      parent: x,
      op: :batch_norm,
      params: [beta, gamma],
      vars: [ra_var, ra_mean],
      opts: []
    }

    node
  end

  @doc """
  Adds a flatten layer to the network.

  This layer will flatten all but the batch dimensions
  of the input into a single layer. Typically called to flatten
  the output of a convolution for use with a dense layer.

  ## Options

    * `name` - Layer name.

  """
  def flatten(%Axon{output_shape: shape} = x, opts \\ []) do
    {id, name} = unique_identifiers(:flatten, opts[:name])
    new_shape = Axon.Shape.flatten(shape)
    %Axon{id: id, name: name, output_shape: new_shape, parent: x, op: :flatten, params: [], vars: []}
  end

  ## Combinators

  @doc """
  Adds two layers.
  """
  def add(%Axon{output_shape: shape} = x, %Axon{} = y, opts \\ []) do
    {id, name} = unique_identifiers(:flatten, opts[:name])

    node = %Axon{
      id: id,
      name: name,
      output_shape: shape,
      parent: [x, y],
      op: :add,
      params: [],
      vars: [],
      opts: []
    }

    node
  end

  @doc """
  Defines a new Axon model.
  """
  defmacro model(do: block) do
    # TODO: I don't think this is the best way to determine number of params
    # to match on in predict
    {_, {num_params, num_vars}} =
      Macro.prewalk(block, {0, 0}, fn
        {:dense, _, _} = node, {params, vars} -> {node, {params + 2, vars}}
        {:conv, _, _} = node, {params, vars} -> {node, {params + 2, vars}}
        node, acc -> {node, acc}
      end)

    tuple_param_args = for _ <- 1..num_params, do: {:_, [], nil}
    tuple_var_args = for _ <- 1..num_vars, do: {:_, [], nil}

    predict_args = [
      {:=, [], [{:{}, [], tuple_param_args}, {:params, [], nil}]},
      {:=, [], [{:{}, [], tuple_var_args}, {:vars, [], nil}]},
      {:batch, [], nil}
    ]

    predict_signature = {:predict, [], predict_args}

    quote do
      import Nx.Defn

      @axon_ast unquote(block)

      def __axon__, do: @axon_ast

      defn init_random_params() do
        transform(:ok, fn _ -> Axon.__jit_params__(__axon__()) end)
      end

      defn unquote(predict_signature) do
        transform(
          {unquote({:params, [], nil}), unquote({:vars, [], nil}), unquote({:batch, [], nil})},
          fn {params, vars, input} ->
            Axon.__jit_predict__(__axon__(), params, vars, input)
          end
        )
      end
    end
  end

  @doc """
  Defines a new Axon model with the given name.
  """
  defmacro model(call, do: block) do
    # TODO: Call can take inputs such that we don't have
    # to specify input layers, but we lose shape information
    {name, _} = Macro.decompose_call(call)

    # TODO: I don't think this is the best way to determine number of params
    # to match on in predict
    {_, {num_params, num_vars}} =
      Macro.prewalk(block, {0, 0}, fn
        {:dense, _, _} = node, {params, vars} -> {node, {params + 2, vars}}
        {:conv, _, _} = node, {params, vars} -> {node, {params + 2, vars}}
        {:batch_norm, _, _} = node, {params, vars} -> {node, {params + 2, vars + 2}}
        node, acc -> {node, acc}
      end)

    tuple_param_args = for _ <- 1..num_params, do: {:_, [], nil}
    tuple_var_args = for _ <- 1..num_vars, do: {:_, [], nil}

    predict_args = [
      {:=, [], [{:{}, [], tuple_param_args}, {:params, [], nil}]},
      {:=, [], [{:{}, [], tuple_var_args}, {:vars, [], nil}]},
      {:batch, [], nil}
    ]

    predict_signature = {name, [], predict_args}

    quote do
      import Nx.Defn

      @axon_ast unquote(block)

      def unquote(axon_name(name))(), do: @axon_ast

      defn unquote(init_function_name(name))() do
        transform(:ok, fn _ -> Axon.__jit_params__(unquote(axon_name(name))()) end)
      end

      defn unquote(predict_signature) do
        transform(
          {unquote({:params, [], nil}), unquote({:vars, [], nil}), unquote({:batch, [], nil})},
          fn {params, vars, input} ->
            Axon.__jit_predict__(unquote(axon_name(name))(), params, vars, input)
          end
        )
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
    {param_names_and_exprs, _} = to_param_expr(graph, %{}, 1)
    {var_names_and_exprs, _} = to_var_expr(graph, %{}, 1)

    var_exprs = var_names_and_exprs |> Map.values() |> List.to_tuple()
    param_exprs = param_names_and_exprs |> Map.values() |> List.to_tuple()

    {param_exprs, var_exprs}
  end

  defp to_param_expr(%Axon{parent: nil, params: params}, names_and_exprs, counter) do
    Enum.reduce(params, {names_and_exprs, counter}, fn %Axon.State{
                                                         name: name,
                                                         shape: shape,
                                                         initializer: initializer
                                                       },
                                                       {names_and_exprs, counter} ->
      {
        Map.put(
          names_and_exprs,
          "#{counter_to_name(counter)}_" <> name,
          apply(Axon.Initializers, initializer, [[shape: shape]])
        ),
        counter + 1
      }
    end)
  end

  defp to_param_expr(%Axon{parent: parent, params: params}, names_and_exprs, counter) do
    {names_and_exprs, counter} =
      Enum.reduce(params, {names_and_exprs, counter}, fn %Axon.State{
                                                           name: name,
                                                           shape: shape,
                                                           initializer: initializer
                                                         },
                                                         {names_and_exprs, counter} ->
        {
          Map.put(
            names_and_exprs,
            "#{counter_to_name(counter)}_" <> name,
            apply(Axon.Initializers, initializer, [[shape: shape]])
          ),
          counter + 1
        }
      end)

    to_param_expr(parent, names_and_exprs, counter)
  end

  defp counter_to_name(counter) when counter >= 26 do
    [counter_to_name(div(counter, 26)) | counter_to_name(rem(counter, 26))]
  end

  defp counter_to_name(counter), do: [Enum.at(?a..?z, counter)]

  defp to_var_expr(%Axon{parent: nil, vars: vars}, names_and_exprs, counter) do
    Enum.reduce(vars, {names_and_exprs, counter}, fn %Axon.State{
                                                       name: name,
                                                       shape: shape,
                                                       initializer: initializer
                                                     },
                                                     {names_and_exprs, counter} ->
      {
        Map.put(
          names_and_exprs,
          "#{counter_to_name(counter)}_" <> name,
          apply(Axon.Initializers, initializer, [[shape: shape]])
        ),
        counter + 1
      }
    end)
  end

  defp to_var_expr(%Axon{parent: parent, vars: vars}, names_and_exprs, counter) do
    {names_and_exprs, counter} =
      Enum.reduce(vars, {names_and_exprs, counter}, fn %Axon.State{
                                                         name: name,
                                                         shape: shape,
                                                         initializer: initializer
                                                       },
                                                       {names_and_exprs, counter} ->
        {
          Map.put(
            names_and_exprs,
            "#{counter_to_name(counter)}_" <> name,
            apply(Axon.Initializers, initializer, [[shape: shape]])
          ),
          counter + 1
        }
      end)

    to_var_expr(parent, names_and_exprs, counter)
  end

  ## Model JIT Compilation

  @doc false
  def __jit_predict__(graph, params, vars, input) do
    {pred_expr, var_exprs} =
      to_predict_expr(
        graph,
        Tuple.to_list(params),
        Tuple.to_list(vars),
        input
      )

    {pred_expr, List.to_tuple(var_exprs)}
  end

  defp to_predict_expr(%Axon{op: op, parent: parent}, params, vars, input)
       when op in @activation_layers do
    {expr, update_exprs} = to_predict_expr(parent, params, vars, input)
    {apply(Axon.Activations, op, [expr]), update_exprs}
  end

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, params, vars, input)
       when op in @pooling_layers do
    {expr, update_exprs} = to_predict_expr(parent, params, vars, input)
    {apply(Axon.Layers, op, [expr, opts]), update_exprs}
  end

  defp to_predict_expr(%Axon{op: op, parent: parent, opts: opts}, params, vars, input)
       when op in @dropout_layers do
    {expr, update_exprs} = to_predict_expr(parent, params, vars, input)
    {apply(Axon.Layers, op, [expr, opts]), update_exprs}
  end

  defp to_predict_expr(%Axon{op: :dense, parent: parent}, [b, w | params], vars, input) do
    {expr, update_exprs} = to_predict_expr(parent, params, vars, input)
    {apply(Axon.Layers, :dense, [expr, w, b]), update_exprs}
  end

  defp to_predict_expr(%Axon{op: :conv, parent: parent, opts: opts}, [b, w | params], vars, input) do
    {expr, update_exprs} = to_predict_expr(parent, params, vars, input)
    {apply(Axon.Layers, :conv, [expr, w, b, opts]), update_exprs}
  end

  defp to_predict_expr(
         %Axon{op: :reshape, output_shape: shape, parent: parent},
         params,
         vars,
         input
       ) do
    {expr, update_exprs} = to_predict_expr(parent, params, vars, input)
    {apply(Nx, :reshape, [expr, shape]), update_exprs}
  end

  defp to_predict_expr(%Axon{op: :flatten, parent: parent}, params, vars, input) do
    {expr, update_exprs} = to_predict_expr(parent, params, vars, input)
    {apply(Axon.Layers, :flatten, [expr]), update_exprs}
  end

  defp to_predict_expr(
         %Axon{op: :batch_norm, parent: parent},
         [beta, gamma | params],
         [var, mean | vars],
         input
       ) do
    {expr, update_exprs} = to_predict_expr(parent, params, vars, input)
    {mean_expr, var_expr} = apply(Axon.Normalization, :batch_norm_stats, [expr, mean, var])

    {apply(Axon.Layers, :batch_norm, [expr, mean_expr, var_expr, gamma, beta]),
     [var_expr, mean_expr | update_exprs]}
  end

  defp to_predict_expr(%Axon{op: :input, parent: nil}, _params, _vars, input) do
    {input, []}
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

  defp stateful(name, shape, initializer, _opts \\ []) do
    %Axon.State{name: name, shape: shape, initializer: initializer}
  end
end
