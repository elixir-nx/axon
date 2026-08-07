defmodule Axon.Block do
  @moduledoc """
  Defines reusable `Nx.block/4` layers via the `defblock` macro.

  `defblock` expands to two things:

    * a struct module under the **caller module**, used as the `Nx.block/4` tag
    * a `defn` whose body is wrapped in `Nx.block/4` tagged with that struct

  By default the struct module is `CallerModule.<CamelizedFunName>`:

      defblock selu(x, opts \\\\ []) do
        ...
      end

  defined in `Axon.Activations` yields `%Axon.Activations.Selu{}`.

  Pass an optional single-segment alias when you need non-default casing:

      defblock SeLU, selu(x, opts \\\\ []) do
        ...
      end

  yields `%Axon.Activations.SeLU{}`.

  Trailing keyword `opts \\\\ []` (or any list default) arguments are stored on the
  block struct as `:opts` and are **not** passed in the `Nx.block/4` args list.
  That matches current Nx: block args must be tensors (or containers of tensors);
  static options live on the struct. The block lambda still receives `opts` by
  pattern-matching the struct, so bodies can call `keyword!/2` unchanged.

  The struct is the dispatch tag for custom kernel implementations
  (for example `defimpl EXLA.CustomCall, for: Axon.Activations.Relu`).

  The module that calls `defblock` must `import Nx.Defn` so the generated
  `defn` is in scope.

  ## Examples

      defmodule MyLayers do
        import Nx.Defn
        import Axon.Block

        defblock dense(x, w, b) do
          x |> Nx.dot(w) |> Nx.add(b)
        end
      end

  This defines `MyLayers.dense/3` and the struct `%MyLayers.Dense{}`.

      defblock leaky_relu(x, opts \\\\ []) do
        opts = keyword!(opts, alpha: 1.0e-2)
        Nx.select(Nx.greater(x, 0), x, x * opts[:alpha])
      end
  """

  @doc """
  Defines a block under the caller module, camelizing the function name.
  """
  defmacro defblock(call, do: body) do
    build(__CALLER__, nil, call, body)
  end

  @doc """
  Defines a block under the caller module with an explicit module suffix.

  `suffix` must be a single-segment alias (for example `SeLU`), not a nested
  module path.
  """
  defmacro defblock(suffix, call, do: body) do
    build(__CALLER__, suffix, call, body)
  end

  defp build(env, suffix_ast, call, body) do
    {name, args} = parse_call(call, env)
    {tensor_args, opts_args} = split_opts_args(args)
    tensor_vars = Enum.map(tensor_args, &arg_var/1)
    struct_module = Module.concat(env.module, suffix_name(suffix_ast, name, env))

    {struct_def, block_struct, fun_struct} =
      case opts_args do
        [] ->
          {
            quote(do: defstruct([])),
            quote(do: %unquote(struct_module){}),
            quote(do: %unquote(struct_module){})
          }

        [opts_arg] ->
          opts_var = arg_var(opts_arg)

          {
            quote(do: defstruct(opts: [])),
            quote(do: %unquote(struct_module){opts: unquote(opts_var)}),
            quote(do: %unquote(struct_module){opts: unquote(opts_var)})
          }

        other ->
          raise CompileError,
            description:
              "defblock supports at most one trailing opts argument, got: " <>
                Macro.to_string(other),
            file: env.file,
            line: env.line
      end

    quote do
      defmodule unquote(struct_module) do
        @moduledoc false
        unquote(struct_def)
      end

      defn unquote(name)(unquote_splicing(args)) do
        Nx.block(
          unquote(block_struct),
          [unquote_splicing(tensor_vars)],
          nil,
          fn unquote(fun_struct), unquote_splicing(tensor_vars) ->
            unquote(body)
          end
        )
      end
    end
  end

  defp split_opts_args(args) do
    Enum.split_with(args, fn arg -> not opts_arg?(arg) end)
  end

  defp opts_arg?({:\\, _meta, [_var, default]}), do: is_list(default)
  defp opts_arg?(_), do: false

  defp suffix_name(nil, name, _env) do
    name |> Atom.to_string() |> Macro.camelize()
  end

  defp suffix_name({:__aliases__, _meta, [segment]}, _name, _env) when is_atom(segment) do
    Atom.to_string(segment)
  end

  defp suffix_name(other, _name, env) do
    raise CompileError,
      description:
        "defblock optional name must be a single-segment alias like `SeLU`, got: " <>
          Macro.to_string(other),
      file: env.file,
      line: env.line
  end

  defp parse_call({name, _meta, args}, _env) when is_atom(name) and is_list(args) do
    {name, args}
  end

  defp parse_call({name, _meta, nil}, _env) when is_atom(name) do
    {name, []}
  end

  defp parse_call(other, env) do
    raise CompileError,
      description:
        "defblock expects a function head like `name(args...)`, got: " <>
          Macro.to_string(other),
      file: env.file,
      line: env.line
  end

  defp arg_var({:\\, _meta, [var, _default]}), do: var
  defp arg_var(var), do: var
end
