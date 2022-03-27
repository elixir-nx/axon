defmodule Axon.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/axon"
  @version "0.1.0-dev"

  def project do
    [
      app: :axon,
      version: @version,
      name: "Axon",
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
      # EXLA is a test-only dependency for testing models and training
      # under JIT
      {:exla, "~> 0.1.0-dev", exla_opts()},
      # Torchx is a test-only dependency for testing models and training
      # with libtorch bindings
      {:torchx, "~> 0.1.0-dev", torchx_opts()},
      {:nx, "~> 0.1.0-dev", nx_opts()},
      {:ex_doc, "~> 0.23", only: :dev, runtime: false},
      {:table_rex, "~> 3.1.1"}
    ]
  end

  defp nx_opts do
    if path = System.get_env("AXON_NX_PATH") do
      [path: path, override: true]
    else
      [github: "elixir-nx/nx", sparse: "nx", override: true]
    end
  end

  defp exla_opts do
    if path = System.get_env("AXON_EXLA_PATH") do
      [path: path, only: :test]
    else
      [github: "elixir-nx/nx", sparse: "exla", only: :test]
    end
  end

  defp torchx_opts do
    if path = System.get_env("AXON_TORCHX_PATH") do
      [path: path, only: :test]
    else
      [github: "elixir-nx/nx", sparse: "torchx", only: :test]
    end
  end

  defp docs do
    [
      main: "Axon",
      source_ref: "v#{@version}",
      source_url: @source_url,
      groups_for_functions: [
        # Axon
        "Layers: Special": &(&1[:type] == :special),
        "Layers: Activation": &(&1[:type] == :activation),
        "Layers: Linear": &(&1[:type] == :linear),
        "Layers: Convolution": &(&1[:type] == :convolution),
        "Layers: Dropout": &(&1[:type] == :dropout),
        "Layers: Pooling": &(&1[:type] == :pooling),
        "Layers: Normalization": &(&1[:type] == :normalization),
        "Layers: Recurrent": &(&1[:type] == :recurrent),
        "Layers: Composition": &(&1[:type] == :composition),
        "Layers: Shape": &(&1[:type] == :shape),
        "Model: Compilation": &(&1[:type] == :compilation),
        "Model: Execution": &(&1[:type] == :execution),

        # Axon.Layers
        "Functions: Attention": &(&1[:type] == :attention),
        "Functions: Convolutional": &(&1[:type] == :convolutional),
        "Functions: Dropout": &(&1[:type] == :dropout),
        "Functions: Linear": &(&1[:type] == :linear),
        "Functions: Normalization": &(&1[:type] == :normalization),
        "Functions: Pooling": &(&1[:type] == :pooling)
      ],
      before_closing_body_tag: &before_closing_body_tag/1
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css" integrity="sha384-t5CR+zwDAROtph0PXGte6ia8heboACF9R5l/DiY+WZ3P2lxNgvJkQk5n7GPvLMYw" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.js" integrity="sha384-FaFLTlohFghEIZkw6VGwmf9ISTubWAVYW8tG8+w2LAIftJEULZABrF9PPFv+tVkH" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/contrib/auto-render.min.js" integrity="sha384-bHBqxz8fokvgoJ/sc17HODNxa42TlaEhB+w8ZJXTc2nZf1VgEaFZeZvT4Mznfz0v" crossorigin="anonymous"
        onload="renderMathInElement(document.body);"></script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
