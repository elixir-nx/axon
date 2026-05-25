defmodule Axon.Block do
  @moduledoc """
  Defines reusable `Nx.block/4` layers via the `defblock/2` macro.

  `defblock prefix, name(args...) do body end` expands to two things:

    * a struct module named `<prefix>.<CamelName>`, defined as `defstruct []`
    * a `defn` named `name` whose body is wrapped in `Nx.block/4` tagged with
      that struct

  The struct serves as the dispatch tag for custom kernel implementations.

  The module that calls `defblock` must `import Nx.Defn` so the generated
  `defn` is in scope.

  ## Examples

      defmodule MyLayers do
        import Nx.Defn
        import Axon.Block

        defblock MyBlock, dense(x, w, b) do
          x |> Nx.dot(w) |> Nx.add(b)
        end
      end

  This defines `MyLayers.dense/3` and the struct `%MyBlock.Dense{}`. External
  libraries can override the default `defn` implementation with a custom kernel:

      defimpl EXLA.CustomCall, for: MyBlock.Dense do
        ...
      end

      defimpl EMLX.Fast, for: MyBlock.Dense do
        ...
      end

  All layers and activations are implemented as `defblock`, namespaced under this
  module. Library users can use this to override default Axon implementations with
  their own custom kernels.
  """

  @doc """
  Defines a block under the given module `prefix`.
  """
  defmacro defblock(prefix, call, do: body) do
    build(__CALLER__, prefix, call, body)
  end

  defp build(env, prefix_ast, call, body) do
    {name, args} = parse_call(call, env)
    prefix = expand_prefix(prefix_ast, env)

    camelized = name |> Atom.to_string() |> Macro.camelize()
    struct_module = Module.concat(prefix, camelized)

    quote do
      defmodule unquote(struct_module) do
        @moduledoc false
        defstruct []
      end

      defn unquote(name)(unquote_splicing(args)) do
        Nx.block(
          %unquote(struct_module){},
          unquote(args),
          nil,
          fn _struct, unquote_splicing(args) ->
            unquote(body)
          end
        )
      end
    end
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

  defp expand_prefix(prefix_ast, env) do
    case Macro.expand(prefix_ast, env) do
      mod when is_atom(mod) ->
        mod

      other ->
        raise CompileError,
          description:
            "defblock expects a module prefix, got: " <> Macro.to_string(other),
          file: env.file,
          line: env.line
    end
  end
end
