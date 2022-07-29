defmodule Axon.Display do
  @moduledoc """
  Module for rendering various visual representations of Axon models.
  """

  import Axon.Shared
  alias Axon.Parameter

  @doc """
  Displays the given Axon model with the given input shapes
  as a table.
  """
  def as_table(%Axon{} = axon, input_templates) do
    title = "Model"
    header = ["Layer", "Input Shape", "Output Shape", "Options", "Parameters"]
    model_info = %{num_params: 0, total_param_byte_size: 0}
    {_, _, _, cache, _, model_info} = axon_to_rows(axon, input_templates, %{}, %{}, model_info)

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

  defp axon_to_rows(%{id: id, op_name: op_name} = graph, templates, cache, op_counts, model_info) do
    case cache do
      %{^id => {row, name, shape}} ->
        {row, name, shape, cache, op_counts, model_info}

      %{} ->
        {row, name, shape, cache, op_counts, model_info} =
          do_axon_to_rows(graph, templates, cache, op_counts, model_info)

        cache = Map.put(cache, id, {row, name, shape})
        op_counts = Map.update(op_counts, op_name, 1, fn x -> x + 1 end)
        {row, name, shape, cache, op_counts, model_info}
    end
  end

  defp do_axon_to_rows(
         %Axon{
           op: :container,
           parent: [parents],
           name: name_fn,
         } = model,
         templates,
         cache,
         op_counts,
         model_info
       ) do
    {input_names, {cache, op_counts, model_info}} =
      deep_map_reduce(parents, {cache, op_counts, model_info}, fn
        graph, {cache, op_counts, model_info} ->
          {_, name, _shape, cache, op_counts, model_info} =
            axon_to_rows(graph, templates, cache, op_counts, model_info)

          {name, {cache, op_counts, model_info}}
      end)

    op_string = "container"

    name = name_fn.(:container, op_counts)
    shape = Axon.get_output_shape(model, templates)

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
         %Axon{
           parent: parents,
           parameters: params,
           name: name_fn,
           opts: opts,
           policy: %{params: {_, bitsize}},
           op_name: op_name
         } = model,
         templates,
         cache,
         op_counts,
         model_info
       ) do
    {input_names_and_shapes, {cache, op_counts, model_info}} =
      Enum.map_reduce(parents, {cache, op_counts, model_info}, fn
        graph, {cache, op_counts, model_info} ->
          {_, name, shape, cache, op_counts, model_info} =
            axon_to_rows(graph, templates, cache, op_counts, model_info)

          {{name, shape}, {cache, op_counts, model_info}}
      end)

    {input_names, input_shapes} = Enum.unzip(input_names_and_shapes)

    num_params =
      Enum.reduce(params, 0, fn
        %Parameter{shape: {:tuple, shapes}}, acc ->
          Enum.reduce(shapes, acc, &Nx.size(apply(&1, input_shapes)) + &2)

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
    shape = Axon.get_output_shape(model, templates)

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
    |> Enum.map(fn %Parameter{name: name, shape: shape_fn} ->
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
end
