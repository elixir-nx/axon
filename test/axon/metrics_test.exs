defmodule Axon.MetricsTest do
  use ExUnit.Case
  import AxonTestUtil

  setup config do
    Nx.Defn.default_options(compiler: test_compiler())
    Nx.default_backend(test_backend())
    Process.register(self(), config.test)
    :ok
  end

  # Do not doctest if USE_EXLA or USE_TORCHX is set, because
  # that will check for absolute equality and both will trigger
  # failures
  unless System.get_env("USE_EXLA") || System.get_env("USE_TORCHX") do
    doctest Axon.Metrics
  end
end
