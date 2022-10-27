defmodule Axon.MixProject do
  use Mix.Project

  @source_url "https://github.com/elixir-nx/axon"
  @version "0.3.0"

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
      {:exla, "~> 0.4.0", [only: :test] ++ exla_opts()},
      {:torchx, "~> 0.4.0", [only: :test] ++ torchx_opts()},
      {:nx, "~> 0.4.0", nx_opts()},
      {:ex_doc, "~> 0.23", only: :docs},
      {:table_rex, "~> 3.1.1", optional: true},
      {:kino, "~> 0.7.0", optional: true}
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

  defp torchx_opts do
    if path = System.get_env("AXON_TORCHX_PATH") do
      [path: path]
    else
      []
    end
  end

  defp docs do
    [
      main: "Axon",
      source_ref: "v#{@version}",
      logo: "logo.png",
      source_url: @source_url,
      extras: [
        # Guides
        "guides/guides.md",
        "guides/model_creation/your_first_axon_model.livemd",
        "guides/model_creation/sequential_models.livemd",
        "guides/model_creation/complex_models.livemd",
        "guides/model_creation/multi_input_multi_output_models.livemd",
        "guides/model_creation/custom_layers.livemd",
        "guides/model_creation/model_hooks.livemd",
        "guides/model_execution/accelerating_axon.livemd",
        "guides/model_execution/training_and_inference_mode.livemd",
        "guides/training_and_evaluation/your_first_training_loop.livemd",
        "guides/training_and_evaluation/instrumenting_loops_with_metrics.livemd",
        "guides/training_and_evaluation/your_first_evaluation_loop.livemd",
        "guides/training_and_evaluation/using_loop_event_handlers.livemd",
        "guides/training_and_evaluation/custom_models_loss_optimizers.livemd",
        "guides/training_and_evaluation/writing_custom_metrics.livemd",
        "guides/training_and_evaluation/writing_custom_event_handlers.livemd",
        "guides/serialization/onnx_to_axon.livemd",
        # Examples
        "notebooks/basics/xor.livemd",
        "notebooks/vision/mnist.livemd",
        "notebooks/vision/horses_or_humans.livemd",
        "notebooks/text/lstm_generation.livemd",
        "notebooks/structured/credit_card_fraud.livemd",
        "notebooks/generative/mnist_autoencoder_using_kino.livemd",
        "notebooks/generative/fashionmnist_autoencoder.livemd",
        "notebooks/generative/fashionmnist_vae.livemd"
      ],
      groups_for_extras: [
        "Guides: Model Creation": Path.wildcard("guides/model_creation/*.livemd"),
        "Guides: Model Execution": Path.wildcard("guides/model_execution/*.livemd"),
        "Guides: Training and Evalutaion":
          Path.wildcard("guides/training_and_evaluation/*.livemd"),
        "Guides: Serialization": Path.wildcard("guides/serialization/*.livemd"),
        "Examples: Basics": Path.wildcard("notebooks/basics/*.livemd"),
        "Examples: Vision": Path.wildcard("notebooks/vision/*.livemd"),
        "Examples: Text": Path.wildcard("notebooks/text/*.livemd"),
        "Examples: Structured": Path.wildcard("notebooks/structured/*.livemd"),
        "Examples: Generative": Path.wildcard("notebooks/generative/*.livemd")
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
        "Layers: Combinators": &(&1[:type] == :combinator),
        "Layers: Shape": &(&1[:type] == :shape),
        Model: &(&1[:type] == :model),
        "Model: Manipulation": &(&1[:type] == :graph),
        "Model: Debugging": &(&1[:type] == :debug),

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
        Model: [
          Axon,
          Axon.MixedPrecision,
          Axon.None,
          Axon.StatefulOutput,
          Axon.Initalizers
        ],
        Summary: [
          Axon.Display
        ],
        Functional: [
          Axon.Activations,
          Axon.Initializers,
          Axon.Layers,
          Axon.Losses,
          Axon.Metrics,
          Axon.Recurrent,
          Axon.LossScale
        ],
        Optimization: [
          Axon.Optimizers,
          Axon.Updates,
          Axon.Schedules
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
    <!-- Render math with KaTeX -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css" integrity="sha384-t5CR+zwDAROtph0PXGte6ia8heboACF9R5l/DiY+WZ3P2lxNgvJkQk5n7GPvLMYw" crossorigin="anonymous">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.js" integrity="sha384-FaFLTlohFghEIZkw6VGwmf9ISTubWAVYW8tG8+w2LAIftJEULZABrF9PPFv+tVkH" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/contrib/auto-render.min.js" integrity="sha384-bHBqxz8fokvgoJ/sc17HODNxa42TlaEhB+w8ZJXTc2nZf1VgEaFZeZvT4Mznfz0v" crossorigin="anonymous"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
          ]
        });
      });
    </script>

    <!-- Render diagrams with Mermaid -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@8.13.3/dist/mermaid.min.js"></script>
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        mermaid.initialize({ startOnLoad: false });
        let id = 0;
        for (const codeEl of document.querySelectorAll("pre code.mermaid")) {
          const preEl = codeEl.parentElement;
          const graphDefinition = codeEl.textContent;
          const graphEl = document.createElement("div");
          const graphId = "mermaid-graph-" + id++;
          mermaid.render(graphId, graphDefinition, function (svgSource, bindListeners) {
            graphEl.innerHTML = svgSource;
            bindListeners && bindListeners(graphEl);
            preEl.insertAdjacentElement("afterend", graphEl);
            preEl.remove();
          });
        }
      });
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
