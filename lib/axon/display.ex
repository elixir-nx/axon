defmodule Axon.Display do
  @moduledoc """
  Module for rendering various visual representations of Axon models.
  """

  import Axon.Shared
  alias Axon.Parameter

  @compile {:no_warn_undefined, TableRex.Table}

  @doc """
  Traces execution of the given Axon model with the given
  inputs, rendering the execution flow as a table.

  You must include [table_rex](https://hex.pm/packages/table_rex) as
  a dependency in your project to make use of this function.

  ## Examples

  Given an Axon model:

      model = Axon.input("input") |> Axon.dense(32)

  You can define input templates for each input:

      input = Nx.template({1, 16}, :f32)

  And then display the execution flow of the model:

      Axon.Display.as_table(model, input)
  """
  def as_table(%Axon{output: id, nodes: nodes}, input_templates) do
    assert_table_rex!("ax_table/2")

    title = "Model"
    header = ["Layer", "Input Shape", "Output Shape", "Options", "Parameters"]
    model_info = %{num_params: 0, total_param_byte_size: 0}

    {_, _, _, cache, _, model_info} =
      axon_to_rows(id, nodes, input_templates, %{}, %{}, model_info)

    rows =
      cache
      |> Enum.sort()
      |> Enum.unzip()
      |> elem(1)
      |> Enum.map(&elem(&1, 0))

    rows
    |> TableRex.Table.new(header, title)
    |> TableRex.Table.render!(
      header_separator_symbol: "=",
      title_separator_symbol: "=",
      vertical_style: :all,
      horizontal_style: :all,
      horizontal_symbol: "-",
      vertical_symbol: "|"
    )
    |> then(&(&1 <> "Total Parameters: #{model_info.num_params}\n"))
    |> then(&(&1 <> "Total Parameters Memory: #{model_info.total_param_byte_size} bytes\n"))
  end

  defp assert_table_rex!(fn_name) do
    unless Code.ensure_loaded?(TableRex) do
      raise RuntimeError, """
      #{fn_name} depends on the :table_rex package.

      You can install it by adding

          {:table_rex, "~> 3.1.1"}

      to your dependency list.
      """
    end
  end

  defp axon_to_rows(id, nodes, templates, cache, op_counts, model_info) do
    case cache do
      %{^id => {row, name, shape}} ->
        {row, name, shape, cache, op_counts, model_info}

      %{} ->
        %Axon.Node{op_name: op_name} = axon_node = nodes[id]

        {row, name, shape, cache, op_counts, model_info} =
          do_axon_to_rows(axon_node, nodes, templates, cache, op_counts, model_info)

        cache = Map.put(cache, id, {row, name, shape})
        op_counts = Map.update(op_counts, op_name, 1, fn x -> x + 1 end)
        {row, name, shape, cache, op_counts, model_info}
    end
  end

  defp do_axon_to_rows(
         %Axon.Node{
           id: id,
           op: :container,
           parent: [parents],
           name: name_fn
         },
         nodes,
         templates,
         cache,
         op_counts,
         model_info
       ) do
    {input_names, {cache, op_counts, model_info}} =
      deep_map_reduce(parents, {cache, op_counts, model_info}, fn
        parent_id, {cache, op_counts, model_info} ->
          {_, name, _shape, cache, op_counts, model_info} =
            axon_to_rows(parent_id, nodes, templates, cache, op_counts, model_info)

          {name, {cache, op_counts, model_info}}
      end)

    op_string = "container"

    name = name_fn.(:container, op_counts)
    shape = Axon.get_output_shape(%Axon{output: id, nodes: nodes}, templates)

    row = [
      "#{name} ( #{op_string} #{inspect(input_names)} )",
      "#{inspect({})}",
      "#{inspect(shape)}",
      render_options([]),
      render_parameters(%{}, [])
    ]

    {row, name, shape, cache, op_counts, model_info}
  end

  defp do_axon_to_rows(
         %Axon.Node{
           id: id,
           parent: parents,
           parameters: params,
           name: name_fn,
           opts: opts,
           policy: %{params: {_, bitsize}},
           op_name: op_name
         },
         nodes,
         templates,
         cache,
         op_counts,
         model_info
       ) do
    {input_names_and_shapes, {cache, op_counts, model_info}} =
      Enum.map_reduce(parents, {cache, op_counts, model_info}, fn
        parent_id, {cache, op_counts, model_info} ->
          {_, name, shape, cache, op_counts, model_info} =
            axon_to_rows(parent_id, nodes, templates, cache, op_counts, model_info)

          {{name, shape}, {cache, op_counts, model_info}}
      end)

    {input_names, input_shapes} = Enum.unzip(input_names_and_shapes)

    num_params =
      Enum.reduce(params, 0, fn
        %Parameter{shape: {:tuple, shapes}}, acc ->
          Enum.reduce(shapes, acc, &(Nx.size(apply(&1, input_shapes)) + &2))

        %Parameter{shape: shape_fn}, acc ->
          acc + Nx.size(apply(shape_fn, input_shapes))
      end)

    param_byte_size = num_params * div(bitsize, 8)

    op_inspect = Atom.to_string(op_name)

    inputs =
      case input_names do
        [] ->
          ""

        [_ | _] = input_names ->
          "#{inspect(input_names)}"
      end

    name = name_fn.(op_name, op_counts)
    shape = Axon.get_output_shape(%Axon{output: id, nodes: nodes}, templates)

    row = [
      "#{name} ( #{op_inspect}#{inputs} )",
      "#{inspect(input_shapes)}",
      "#{inspect(shape)}",
      render_options(opts),
      render_parameters(params, input_shapes)
    ]

    model_info =
      model_info
      |> Map.update(:num_params, 0, &(&1 + num_params))
      |> Map.update(:total_param_byte_size, 0, &(&1 + param_byte_size))
      |> Map.update(:inputs, [], fn inputs ->
        if op_name == :input, do: [{name, shape} | inputs], else: inputs
      end)

    {row, name, shape, cache, op_counts, model_info}
  end

  defp render_options(opts) do
    opts
    |> Enum.map(fn {key, val} ->
      key = Atom.to_string(key)
      "#{key}: #{inspect(val)}"
    end)
    |> Enum.join("\n")
  end

  defp render_parameters(params, input_shapes) do
    params
    |> Enum.map(fn
      %Parameter{name: name, shape: {:tuple, shape_fns}} ->
        shapes =
          shape_fns
          |> Enum.map(&apply(&1, input_shapes))
          |> Enum.map(fn shape -> "f32#{shape_string(shape)}" end)
          |> List.to_tuple()

        "#{name}: tuple#{inspect(shapes)}"

      %Parameter{name: name, shape: shape_fn} ->
        shape = apply(shape_fn, input_shapes)
        "#{name}: f32#{shape_string(shape)}"
    end)
    |> Enum.join("\n")
  end

  defp shape_string(shape) do
    shape
    |> Tuple.to_list()
    |> Enum.map(fn n -> "[#{n}]" end)
    |> Enum.join("")
  end

  @compile {:no_warn_undefined, {Kino.Mermaid, :new, 1}}

  @doc """
  Traces execution of the given Axon model with the given
  inputs, rendering the execution flow as a mermaid flowchart.

  You must include [kino](https://hex.pm/packages/kino) as
  a dependency in your project to make use of this function.

  ## Options

    * `:direction` - defines the direction of the graph visual. The
      value can either be `:top_down` or `:left_right`. Defaults to `:top_down`.

  ## Examples

  Given an Axon model:

      model = Axon.input("input") |> Axon.dense(32)

  You can define input templates for each input:

      input = Nx.template({1, 16}, :f32)

  And then display the execution flow of the model:

      Axon.Display.as_graph(model, input, direction: :top_down)
  """
  def as_graph(%Axon{output: id, nodes: nodes}, input_templates, opts \\ []) do
    assert_kino!("as_graph/3")

    direction = direction_from_opts(opts)

    {_root_node, {cache, _, edgelist}} = axon_to_edges(id, nodes, input_templates, {%{}, %{}, []})
    nodelist = Map.values(cache)

    nodes = Enum.map_join(nodelist, ";\n", &generate_mermaid_node_entry/1)
    edges = Enum.map_join(edgelist, ";\n", &generate_mermaid_edge_entry/1)

    Kino.Mermaid.new("""
    graph #{direction};
    #{nodes};
    #{edges};\
    """)
  end

  defp assert_kino!(fn_name) do
    unless Code.ensure_loaded?(Kino) do
      raise RuntimeError, """
      #{fn_name} depends on the :kino package.

      You can install it by adding

          {:kino, "~> 0.7.0"}

      to your dependency list.
      """
    end
  end

  defp axon_to_edges(id, nodes, input_templates, {cache, op_counts, edgelist}) do
    case cache do
      %{^id => entry} ->
        {entry, {cache, op_counts, edgelist}}

      %{} ->
        %Axon.Node{op_name: op} = axon_node = nodes[id]

        {entry, {cache, op_counts, edgelist}} =
          recur_axon_to_edges(axon_node, nodes, input_templates, {cache, op_counts, edgelist})

        op_counts = Map.update(op_counts, op, 1, fn x -> x + 1 end)
        {entry, {Map.put(cache, id, entry), op_counts, edgelist}}
    end
  end

  defp recur_axon_to_edges(
         %Axon.Node{id: id, op: :container, name: name_fn, parent: [parents]},
         nodes,
         templates,
         cache_counts_edgelist
       ) do
    {node_inputs, {cache, op_counts, edgelist}} =
      deep_map_reduce(parents, cache_counts_edgelist, &axon_to_edges(&1, nodes, templates, &2))

    name = name_fn.(:container, op_counts)
    node_shape = Axon.get_output_shape(%Axon{output: id, nodes: nodes}, templates)
    to_node = %{axon: :axon, id: id, op: :container, name: name, shape: node_shape}

    new_edgelist =
      deep_reduce(node_inputs, edgelist, fn from_node, acc ->
        [{from_node, to_node} | acc]
      end)

    {to_node, {cache, op_counts, new_edgelist}}
  end

  defp recur_axon_to_edges(
         %Axon.Node{id: id, op_name: op, name: name_fn, parent: parents},
         nodes,
         templates,
         cache_counts_edgelist
       ) do
    {node_inputs, {cache, op_counts, edgelist}} =
      Enum.map_reduce(parents, cache_counts_edgelist, &axon_to_edges(&1, nodes, templates, &2))

    name = name_fn.(op, op_counts)
    node_shape = Axon.get_output_shape(%Axon{output: id, nodes: nodes}, templates)
    to_node = %{axon: :axon, id: id, op: op, name: name, shape: node_shape}

    new_edgelist =
      Enum.reduce(node_inputs, edgelist, fn from_node, acc ->
        [{from_node, to_node} | acc]
      end)

    {to_node, {cache, op_counts, new_edgelist}}
  end

  defp generate_mermaid_node_entry(%{id: id, op: :input, name: name, shape: shape}) do
    ~s'#{id}[/"#{name} (:input) #{inspect(shape)}"/]'
  end

  defp generate_mermaid_node_entry(%{id: id, op: op, name: name, shape: shape}) do
    ~s'#{id}["#{name} (#{inspect(op)}) #{inspect(shape)}"]'
  end

  defp generate_mermaid_edge_entry({from_node, to_node}) do
    "#{from_node.id} --> #{to_node.id}"
  end

  defp direction_from_opts(opts) do
    opts
    |> Keyword.get(:direction, :top_down)
    |> convert_direction()
  end

  defp convert_direction(:top_down), do: "TD"
  defp convert_direction(:left_right), do: "LR"

  defp convert_direction(invalid_direction),
    do: raise(ArgumentError, "expected a valid direction, got: #{inspect(invalid_direction)}")
end
