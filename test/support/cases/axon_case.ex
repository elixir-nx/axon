defmodule Axon.Case do
  use ExUnit.CaseTemplate

  using do
    quote do
      import Nx.Defn
      import AxonTestUtil
    end
  end

  setup config do
    Nx.Defn.default_options(compiler: AxonTestUtil.test_compiler())
    Nx.default_backend(AxonTestUtil.test_backend())
    Process.register(self(), config.test)
    :ok
  end
end
