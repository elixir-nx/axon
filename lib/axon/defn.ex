defmodule Axon.Defn do
  @moduledoc false

  # A defn identity compiler for yielding an expression
  # given a numerical definition
  #
  # This is useful for implementing `nx` layers because
  # we need the output shape of the given function

  @behaviour Nx.Defn.Compiler

  @impl true
  def __jit__(_key, vars, fun, _args, _opts) do
    [fun.(vars)]
  end

  @impl true
  def __stream__(_, _, _, _, _, _, _), do: raise("not implemented")

  @impl true
  def __compile__(_, _, _, _), do: raise("not implemented")
end
