defmodule Axon.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/axon"
  @version "0.2.0-dev"

  def project do
    [
      app: :axon,
      version: @version,
      name: "Axon",
      elixir: "~> 1.13",
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs(),
      description: "Create and train neural networks in Elixir",
      package: package(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # EXLA is a test-only dependency for testing models and training
      # under JIT
      {:exla, "~> 0.3.0-dev", [only: :test] ++ exla_opts()},
      {:nx, "~> 0.3.0-dev", nx_opts()},
      {:ex_doc, "~> 0.23", only: :docs},
      {:table_rex, "~> 3.1.1"}
    ]
  end

  defp package do
    [
      maintainers: ["Sean Moriarity"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp nx_opts do
    if path = System.get_env("AXON_NX_PATH") do
      [path: path, override: true]
    else
      []
    end
  end

  defp exla_opts do
    if path = System.get_env("AXON_EXLA_PATH") do
      [path: path]
    else
      []
    end
  end

  defp docs do
    [
      main: "Axon",
      source_ref: "v#{@version}",
      logo: "axon.png",
      source_url: @source_url,
      extras: [
        "notebooks/mnist.livemd",
        "notebooks/fashionmnist_autoencoder.livemd",
        "notebooks/multi_input_example.livemd"
      ],
      groups_for_extras: [
        "Axon Examples": Path.wildcard("notebooks/*.livemd")
      ],
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
        "Layers: Combinators": &(&1[:type] == :combinators),
        "Layers: Shape": &(&1[:type] == :shape),
        "Model: Execution": &(&1[:type] == :execution),

        # Axon.Layers
        "Functions: Attention": &(&1[:type] == :attention),
        "Functions: Convolutional": &(&1[:type] == :convolutional),
        "Functions: Dropout": &(&1[:type] == :dropout),
        "Functions: Linear": &(&1[:type] == :linear),
        "Functions: Normalization": &(&1[:type] == :normalization),
        "Functions: Pooling": &(&1[:type] == :pooling),
        "Functions: Shape": &(&1[:type] == :shape)
      ],
      groups_for_modules: [
        # Axon
        # Axon.MixedPrecision

        Functional: [
          Axon.Activations,
          Axon.Initalizers,
          Axon.Layers,
          Axon.Losses,
          Axon.Metrics,
          Axon.Recurrent
        ],
        Optimization: [
          Axon.Optimizers,
          Axon.Updates
        ],
        Loop: [
          Axon.Loop,
          Axon.Loop.State
        ]
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
