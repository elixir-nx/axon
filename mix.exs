defmodule Axon.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/axon"
  @version "0.1.0-dev"

  def project do
    [
      app: :axon,
      version: @version,
      elixir: "~> 1.11",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
      {:ex_doc, "~> 0.23", only: :dev, runtime: false}
    ]
  end

  defp docs do
    [
      main: "Axon",
      source_ref: "v#{@version}",
      source_url: @source_url,
      groups_for_functions: [
        Linear: &(&1[:type] == :linear),
        Convolutional: &(&1[:type] == :convolutional),
        Attention: &(&1[:type] == :attention),
        Pooling: &(&1[:type] == :pooling),
        Dropout: &(&1[:type] == :dropout),
        Normalization: &(&1[:type] == :normalization)
      ],
      before_closing_body_tag: &before_closing_body_tag/1
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$']],
          displayMath: [['$$','$$']],
        },
      };
    </script>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
