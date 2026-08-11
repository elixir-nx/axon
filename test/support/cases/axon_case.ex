defmodule Axon.Case do
  use ExUnit.CaseTemplate

  using do
    quote do
      import Nx.Defn
      # `assert_all_close/3` comes from `AxonTestUtil`, not `Nx.Testing`, since
      # it also recurses into tuples/maps (e.g. recurrent layer carry state).
      import Nx.Testing, only: [assert_equal: 2]
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
