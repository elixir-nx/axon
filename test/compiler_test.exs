defmodule CompilerTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog
  import AxonTestUtil
  require Axon
  alias Axon.MixedPrecision, as: AMP

  setup do
    Nx.Defn.default_options(compiler: test_compiler())
    :ok
  end

  describe "input" do
    test "single input, single output" do
      model = Axon.input({nil, 1})
      input = Nx.random_uniform({1, 1}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{} = init_fn.()
      assert predict_fn.(%{}, input) == input
    end

    test "multi-input, multi-output" do
      model = {Axon.input({nil, 1}), Axon.input({nil, 2})}

      input =
        {Nx.random_uniform({1, 1}, type: {:f, 32}), Nx.random_uniform({1, 2}, type: {:f, 32})}

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{} = init_fn.()
      assert predict_fn.(%{}, input) == input
    end

    test "multi-input, multi-output nested" do
      model1 = {Axon.input({nil, 1}), {Axon.input({nil, 2})}}

      input1 =
        {Nx.random_uniform({1, 1}, type: {:f, 32}), {Nx.random_uniform({1, 2}, type: {:f, 32})}}

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{} = init_fn.()
      assert predict_fn.(%{}, input1) == input1

      model2 =
        {{Axon.input({nil, 1}), Axon.input({nil, 2})}, Axon.input({nil, 3}),
         {{{Axon.input({nil, 4}), Axon.input({nil, 5})}, Axon.input({nil, 6})}}}

      input2 =
        {{Nx.random_uniform({1, 1}, type: {:f, 32}), Nx.random_uniform({1, 2}, type: {:f, 32})},
         Nx.random_uniform({1, 3}, type: {:f, 32}),
         {{{Nx.random_uniform({1, 4}, type: {:f, 32}), Nx.random_uniform({1, 5}, type: {:f, 32})},
           Nx.random_uniform({1, 6}, type: {:f, 32})}}}

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{} = init_fn.()
      assert predict_fn.(%{}, input2) == input2
    end

    test "raises on bad input shape" do
      model = Axon.input({nil, 32})
      input = Nx.random_uniform({1, 16})
      assert {_, predict_fn} = Axon.compile(model)

      exception = assert_raise Axon.CompilerError, fn -> predict_fn.(%{}, input) end

      assert Exception.message(exception) =~
               "error while building prediction for input layer with name input_"

      assert Exception.message(exception) =~
               "** (ArgumentError) invalid input shape given to model"
    end
  end

  describe "constant" do
    @tag capture_log: true
    test "initializes with no params" do
      model = Axon.constant(Nx.tensor(1.0))

      assert {init_fn, _} = Axon.compile(model)

      assert %{} == init_fn.()
    end

    @tag capture_log: true
    test "computes forward pass with default options" do
      model = Axon.constant(Nx.tensor(1.0))

      assert {_, predict_fn} = Axon.compile(model)
      assert predict_fn.(%{}, {}) == Nx.tensor(1.0)
    end

    @tag capture_log: true
    test "computes forward pass with other layers" do
      model = Axon.add(Axon.constant(Nx.tensor(1.0)), Axon.constant(Nx.tensor(2.0)))

      assert {_, predict_fn} = Axon.compile(model)
      assert predict_fn.(%{}, {}) == Nx.tensor(3.0)
    end

    @tag capture_log: true
    test "computes forward pass with output policy" do
      model = Axon.constant(Nx.tensor(1.0))
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {_, predict_fn} = Axon.compile(mp_model)
      assert predict_fn.(%{}, {}) == Nx.tensor(1.0, type: {:bf, 16})
    end

    test "warns on constant only network" do
      model = Axon.constant(Nx.tensor(1.0))

      assert capture_log(fn ->
               Axon.compile(model)
             end) =~ "You are compiling a graph with no inputs."
    end
  end

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :mish, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  describe "activations" do
    test "initializes with no params" do
      for activation <- @activation_layers do
        model = Axon.input({nil, 32}) |> Axon.activation(activation)

        assert {init_fn, _predict_fn} = Axon.compile(model)
        assert %{} = init_fn.()
      end
    end

    test "computes forward pass with default options" do
      for activation <- @activation_layers do
        model = Axon.input({nil, 1}) |> Axon.activation(activation)
        input = Nx.random_uniform({1, 1})

        assert {_init_fn, predict_fn} = Axon.compile(model)
        assert predict_fn.(%{}, input) == apply(Axon.Activations, activation, [input])
      end
    end

    test "computes forward pass with custom options" do
      for activation <- [:celu, :elu, :leaky_relu] do
        model = Axon.input({nil, 32}) |> Axon.activation(activation, alpha: 0.8)
        input = Nx.random_uniform({1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model)

        assert predict_fn.(%{}, input) ==
                 apply(Axon.Activations, activation, [input, [alpha: 0.8]])
      end
    end

    test "computes forward pass with output policy" do
      for activation <- @activation_layers do
        model = Axon.input({nil, 1}) |> Axon.activation(activation)
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1}))) == {:bf, 16}
      end
    end
  end

  describe "dense" do
    test "initializes in default case" do
      model = Axon.input({nil, 1}) |> Axon.dense(1, name: "dense")

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 = Axon.input({nil, 1}) |> Axon.dense(1, name: "dense", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {1, 1})
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}

      model2 = Axon.input({nil, 1}) |> Axon.dense(1, name: "dense", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {1})
    end

    test "computes forward pass" do
      model = Axon.input({nil, 1}) |> Axon.dense(1, name: "dense", kernel_initializer: :identity)
      input = Nx.iota({1, 1}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input) == Axon.Layers.dense(input, kernel, bias)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1})
        |> Axon.dense(1, name: "dense")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"dense" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1})])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 2}) |> Axon.dense(1, name: "dense")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 2}) |> Axon.dense(1, name: "dense")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 2}))) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model = Axon.input({nil, 2}) |> Axon.dense(1, name: "dense", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)
      assert %{"dense" => %{"kernel" => _} = dense_params} = init_fn.()
      assert Map.has_key?(dense_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model = Axon.input({nil, 2}) |> Axon.dense(1, name: "dense", use_bias: false)
      input = Nx.random_uniform({1, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"dense" => %{"kernel" => k}} = params = init_fn.()

      assert Nx.all_close(predict_fn.(params, input), Axon.Layers.dense(input, k, Nx.tensor(0.0))) ==
               Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "bilinear" do
    test "initializes in default case" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.input({nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {1, 1, 2}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.input({nil, 2})
      model1 = Axon.bilinear(input1, input2, 1, name: "bilinear", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {1, 1, 2})
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}

      model2 = Axon.bilinear(input1, input2, 1, name: "bilinear", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {1, 1, 2}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {1})
    end

    test "computes forward pass" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.input({nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")

      input1 = Nx.iota({1, 1}, type: {:f, 32})
      input2 = Nx.iota({1, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()

      assert predict_fn.(params, {input1, input2}) ==
               Axon.Layers.bilinear(input1, input2, kernel, bias)
    end

    test "computes forward pass with constant" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.constant(Nx.iota({2, 1}))
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")

      input1 = Nx.iota({2, 1}, type: {:f, 32})
      input2 = Nx.iota({2, 1}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()

      assert predict_fn.(params, input1) == Axon.Layers.bilinear(input1, input2, kernel, bias)
    end

    test "returns zero gradient for frozen parameters" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.input({nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear") |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"bilinear" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               Nx.Defn.jit(backward, [
                 init_fn.(),
                 {Nx.random_uniform({1, 1}), Nx.random_uniform({1, 2})}
               ])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1, 2})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.input({nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.input({nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)

      assert Nx.type(
               predict_fn.(init_fn.(), {Nx.random_uniform({1, 1}), Nx.random_uniform({1, 2})})
             ) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.input({nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)
      assert %{"bilinear" => %{"kernel" => _} = bilinear_params} = init_fn.()
      assert Map.has_key?(bilinear_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      input1 = Axon.input({nil, 1})
      input2 = Axon.input({nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear", use_bias: false)

      inp1 = Nx.random_uniform({1, 1})
      inp2 = Nx.random_uniform({1, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"bilinear" => %{"kernel" => k}} = params = init_fn.()

      assert Nx.all_close(
               predict_fn.(params, {inp1, inp2}),
               Axon.Layers.bilinear(inp1, inp2, k, Nx.tensor(0.0))
             ) == Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "embedding" do
    test "initializes in default case" do
      model = Axon.input({nil, 1}) |> Axon.embedding(1, 1, name: "embedding")

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{"embedding" => %{"kernel" => kernel}} = init_fn.()
      assert Nx.shape(kernel) == {1, 1}
      assert Nx.type(kernel) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 1})
        |> Axon.embedding(1, 1, name: "embedding", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"embedding" => %{"kernel" => kernel}} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {1, 1})
    end

    test "computes forward pass" do
      model =
        Axon.input({nil, 1})
        |> Axon.embedding(1, 1, name: "embedding", kernel_initializer: :identity)

      input = Nx.tensor([[0]])

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"embedding" => %{"kernel" => kernel}} = params = init_fn.()
      assert predict_fn.(params, input) == Axon.Layers.embedding(input, kernel)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1})
        |> Axon.embedding(1, 1, name: "embedding")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"embedding" => %{"kernel" => kernel_grad}} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.tensor([[0]])])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 2}) |> Axon.embedding(1, 1, name: "embedding")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"embedding" => %{"kernel" => kernel}} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 2}) |> Axon.embedding(1, 1, name: "embedding")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.tensor([[0, 0]]))) == {:bf, 16}
    end
  end

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  describe "pooling" do
    test "initializes with no params" do
      for pool <- @pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 1, 32})])

        assert {init_fn, _predict_fn} = Axon.compile(model)
        assert %{} = init_fn.()
      end
    end

    test "computes forward pass with default options" do
      default_options = [kernel_size: 1]

      for pool <- @pooling_layers do
        model1 = apply(Axon, pool, [Axon.input({nil, 1, 32})])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        assert predict_fn.(%{}, input1) == apply(Axon.Layers, pool, [input1, default_options])

        model2 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4})])
        input2 = Nx.random_uniform({1, 1, 8, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model2)
        assert predict_fn.(%{}, input2) == apply(Axon.Layers, pool, [input2, default_options])

        model3 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4, 2})])
        input3 = Nx.random_uniform({1, 1, 8, 4, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model3)
        assert predict_fn.(%{}, input3) == apply(Axon.Layers, pool, [input3, default_options])
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @pooling_layers do
        opts1 = [kernel_size: 6]
        model1 = apply(Axon, pool, [Axon.input({nil, 1, 32}), opts1])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        assert predict_fn.(%{}, input1) == apply(Axon.Layers, pool, [input1, opts1])

        opts2 = [kernel_size: 2, strides: 2, padding: :same]
        model2 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4}), opts2])
        input2 = Nx.random_uniform({1, 1, 8, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model2)
        assert predict_fn.(%{}, input2) == apply(Axon.Layers, pool, [input2, opts2])

        opts3 = [kernel_size: {2, 1, 2}, strides: [1, 2, 1], padding: [{0, 1}, {1, 1}, {0, 2}]]
        model3 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4, 2}), opts3])
        input3 = Nx.random_uniform({1, 1, 8, 4, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model3)
        assert predict_fn.(%{}, input3) == apply(Axon.Layers, pool, [input3, opts3])
      end
    end

    test "lp_pool computes forward pass with custom norm" do
      model = Axon.input({nil, 1, 32}) |> Axon.lp_pool(norm: 3)
      input = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

      assert {_, predict_fn} = Axon.compile(model)
      assert predict_fn.(%{}, input) == Axon.Layers.lp_pool(input, kernel_size: {1}, norm: 3)
    end

    test "computes forward pass with output policy" do
      for pool <- @pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 1, 32})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
      end
    end

    test "computes forward pass with channels last" do
      for pool <- @pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 32, 1}), [channels: :last, kernel_size: {2}]])
        inp = Nx.random_uniform({1, 32, 1})

        assert {_, predict_fn} = Axon.compile(model)

        assert predict_fn.(%{}, inp) ==
                 apply(Axon.Layers, pool, [inp, [kernel_size: {2}, strides: [2], channels: :last]])
      end
    end
  end

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool, :adaptive_lp_pool]

  describe "adaptive pooling" do
    test "initializes with no params" do
      for pool <- @adaptive_pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 1, 32})])

        assert {init_fn, _predict_fn} = Axon.compile(model)
        assert %{} = init_fn.()
      end
    end

    test "computes forward pass with default options" do
      for pool <- @adaptive_pooling_layers do
        model1 = apply(Axon, pool, [Axon.input({nil, 1, 32})])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        assert predict_fn.(%{}, input1) == apply(Axon.Layers, pool, [input1, [output_size: 32]])

        model2 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4})])
        input2 = Nx.random_uniform({1, 1, 8, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model2)

        assert predict_fn.(%{}, input2) ==
                 apply(Axon.Layers, pool, [input2, [output_size: {8, 4}]])

        model3 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4, 2})])
        input3 = Nx.random_uniform({1, 1, 8, 4, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model3)

        assert predict_fn.(%{}, input3) ==
                 apply(Axon.Layers, pool, [input3, [output_size: {8, 4, 2}]])
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @adaptive_pooling_layers do
        opts1 = [output_size: 27]
        model1 = apply(Axon, pool, [Axon.input({nil, 1, 32}), opts1])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        assert predict_fn.(%{}, input1) == apply(Axon.Layers, pool, [input1, opts1])

        opts2 = [output_size: {2, 3}]
        model2 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4}), opts2])
        input2 = Nx.random_uniform({1, 1, 8, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model2)
        assert predict_fn.(%{}, input2) == apply(Axon.Layers, pool, [input2, opts2])

        opts3 = [output_size: {4, 3, 1}]
        model3 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4, 2}), opts3])
        input3 = Nx.random_uniform({1, 1, 8, 4, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model3)
        assert predict_fn.(%{}, input3) == apply(Axon.Layers, pool, [input3, opts3])
      end
    end

    test "computes forward pass with output policy" do
      for pool <- @adaptive_pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 1, 32})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
      end
    end

    test "computes forward pass with channels last" do
      for pool <- @adaptive_pooling_layers do
        model =
          apply(Axon, pool, [Axon.input({nil, 32, 1}), [channels: :last, output_size: {27}]])

        inp = Nx.random_uniform({1, 32, 1})

        assert {_, predict_fn} = Axon.compile(model)

        assert predict_fn.(%{}, inp) ==
                 apply(Axon.Layers, pool, [inp, [output_size: {27}, channels: :last]])
      end
    end
  end

  @global_pooling_layers [:global_max_pool, :global_avg_pool, :global_lp_pool]

  describe "global pooling" do
    test "initializes with no params" do
      for pool <- @global_pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 1, 32})])

        assert {init_fn, _predict_fn} = Axon.compile(model)
        assert %{} = init_fn.()
      end
    end

    test "computes forward pass with default options" do
      for pool <- @global_pooling_layers do
        model1 = apply(Axon, pool, [Axon.input({nil, 1, 4})])
        input1 = Nx.random_uniform({1, 1, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        assert predict_fn.(%{}, input1) == apply(Axon.Layers, pool, [input1])

        model2 = apply(Axon, pool, [Axon.input({nil, 1, 2, 2})])
        input2 = Nx.random_uniform({1, 1, 2, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model2)
        assert predict_fn.(%{}, input2) == apply(Axon.Layers, pool, [input2])

        model3 = apply(Axon, pool, [Axon.input({nil, 1, 2, 2, 1})])
        input3 = Nx.random_uniform({1, 1, 2, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model3)
        assert predict_fn.(%{}, input3) == apply(Axon.Layers, pool, [input3])
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @global_pooling_layers do
        opts1 = [keep_axes: true]
        model1 = apply(Axon, pool, [Axon.input({nil, 1, 2}), opts1])
        input1 = Nx.random_uniform({1, 1, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        assert predict_fn.(%{}, input1) == apply(Axon.Layers, pool, [input1, opts1])
      end
    end

    test "computes forward pass with output policy" do
      for pool <- @global_pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 1, 2})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 2}))) == {:bf, 16}
      end
    end

    test "computes forward pass with channels last" do
      for pool <- @global_pooling_layers do
        model1 = apply(Axon, pool, [Axon.input({nil, 32, 1}), [channels: :last, keep_axes: true]])

        model2 =
          apply(Axon, pool, [Axon.input({nil, 32, 1}), [channels: :last, keep_axes: false]])

        inp = Nx.random_uniform({1, 32, 1})

        assert {_, predict_fn} = Axon.compile(model1)

        assert predict_fn.(%{}, inp) ==
                 apply(Axon.Layers, pool, [inp, [keep_axes: true, channels: :last]])

        assert {_, predict_fn} = Axon.compile(model2)

        assert predict_fn.(%{}, inp) ==
                 apply(Axon.Layers, pool, [inp, [keep_axes: false, channels: :last]])
      end
    end
  end

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  describe "dropout" do
    test "initializes with no params" do
      for dropout <- @dropout_layers do
        model = apply(Axon, dropout, [Axon.input({nil, 1, 32})])

        assert {init_fn, _predict_fn} = Axon.compile(model)
        assert %{} = init_fn.()
      end
    end

    test "computes forward pass with default options" do
      for dropout <- @dropout_layers do
        model1 = apply(Axon, dropout, [Axon.input({nil, 1, 32})])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        result1 = predict_fn.(%{}, input1)

        assert Nx.shape(result1) == {1, 1, 32}
        assert Nx.type(result1) == {:f, 32}

        model2 = apply(Axon, dropout, [Axon.input({nil, 1, 8, 4})])
        input2 = Nx.random_uniform({1, 1, 8, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model2)
        result2 = predict_fn.(%{}, input2)

        assert Nx.shape(result2) == {1, 1, 8, 4}
        assert Nx.type(result2) == {:f, 32}

        model3 = apply(Axon, dropout, [Axon.input({nil, 1, 8, 4, 2})])
        input3 = Nx.random_uniform({1, 1, 8, 4, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model3)
        result3 = predict_fn.(%{}, input3)

        assert Nx.shape(result3) == {1, 1, 8, 4, 2}
        assert Nx.type(result3) == {:f, 32}
      end
    end

    test "computes forward pass with custom options" do
      for dropout <- @dropout_layers do
        opts1 = [rate: 0.25]
        model1 = apply(Axon, dropout, [Axon.input({nil, 1, 32}), opts1])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)

        result = predict_fn.(%{}, input1)

        assert Nx.shape(result) == {1, 1, 32}
        assert Nx.type(result) == {:f, 32}
      end
    end

    test "computes forward pass with output policy" do
      for dropout <- @dropout_layers do
        model = apply(Axon, dropout, [Axon.input({nil, 1, 32})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
      end
    end

    test "not present in inference mode" do
      for dropout <- @dropout_layers do
        model = apply(Axon, dropout, [Axon.input({nil, 1, 32})])
        input = Nx.random_uniform({1, 1, 32})

        assert Axon.predict(model, %{}, input) == input
      end
    end
  end

  describe "convolution" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 32, 32}) |> Axon.conv(64, name: "conv")

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {64, 3, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {64}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 3, 32, 32}) |> Axon.conv(32, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {32, 3, 1, 1})
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 32, 32}) |> Axon.conv(32, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {32, 3, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {32})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 1, 2}) |> Axon.conv(2, name: "conv")
      input1 = Nx.random_uniform({1, 1, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.conv(input1, kernel, bias)

      model2 = Axon.input({nil, 1, 2, 2}) |> Axon.conv(3, name: "conv")
      input2 = Nx.random_uniform({1, 1, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.conv(input2, kernel, bias)

      model3 = Axon.input({nil, 1, 2, 2, 2}) |> Axon.conv(4, name: "conv")
      input3 = Nx.random_uniform({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input3) == Axon.Layers.conv(input3, kernel, bias)
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, padding: :same, input_dilation: 2]
      model1 = Axon.input({nil, 1, 2}) |> Axon.conv(2, [name: "conv", kernel_size: 2] ++ opts1)
      input1 = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.conv(input1, kernel, bias, opts1)

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 = Axon.input({nil, 1, 4, 4}) |> Axon.conv(2, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 1, 4, 4})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.conv(input2, kernel, bias, opts2)

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input({nil, 1, 2, 2, 2})
        |> Axon.conv(4, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 1, 2, 2, 2})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input3) == Axon.Layers.conv(input3, kernel, bias, opts3)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 32})
        |> Axon.conv(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"conv" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 32})])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1, 1})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 32}) |> Axon.conv(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 32}) |> Axon.conv(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model = Axon.input({nil, 1, 2}) |> Axon.conv(1, name: "conv", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => _} = conv_params} = init_fn.()
      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model = Axon.input({nil, 1, 2}) |> Axon.conv(1, name: "conv", use_bias: false)
      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => k}} = params = init_fn.()
      assert predict_fn.(params, input) == Axon.Layers.conv(input, k, Nx.tensor(0))
    end

    test "computes forward pass with channels last" do
      model = Axon.input({nil, 3, 3, 6}) |> Axon.conv(2, name: "conv", channels: :last)
      input = Nx.random_uniform({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => k, "bias" => b}} = params = init_fn.()
      assert predict_fn.(params, input) == Axon.Layers.conv(input, k, b, channels: :last)
    end
  end

  describe "depthwise convolution" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 2, 2}) |> Axon.depthwise_conv(3, name: "conv")

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {9, 1, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {9}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 3, 2, 2})
        |> Axon.depthwise_conv(3, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {9, 1, 1, 1})
      assert Nx.shape(bias) == {9}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 2, 2})
        |> Axon.depthwise_conv(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {9, 1, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {9})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 1, 8}) |> Axon.depthwise_conv(3, name: "conv")
      input1 = Nx.random_uniform({1, 1, 8}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.depthwise_conv(input1, kernel, bias)

      model2 = Axon.input({nil, 1, 2, 2}) |> Axon.depthwise_conv(4, name: "conv")
      input2 = Nx.random_uniform({1, 1, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.depthwise_conv(input2, kernel, bias)

      model3 = Axon.input({nil, 1, 2, 2, 2}) |> Axon.depthwise_conv(5, name: "conv")
      input3 = Nx.random_uniform({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input3) == Axon.Layers.depthwise_conv(input3, kernel, bias)
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, padding: :same, input_dilation: 2]

      model1 =
        Axon.input({nil, 1, 8})
        |> Axon.depthwise_conv(1, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = Nx.random_uniform({1, 1, 8})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()

      assert predict_fn.(params, input1) ==
               Axon.Layers.depthwise_conv(input1, kernel, bias, opts1)

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input({nil, 1, 4, 4})
        |> Axon.depthwise_conv(8, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 1, 4, 4})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()

      assert predict_fn.(params, input2) ==
               Axon.Layers.depthwise_conv(input2, kernel, bias, opts2)

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input({nil, 1, 3, 2, 2})
        |> Axon.depthwise_conv(2, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()

      assert predict_fn.(params, input3) ==
               Axon.Layers.depthwise_conv(input3, kernel, bias, opts3)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 2})
        |> Axon.depthwise_conv(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"conv" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 2})])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1, 1})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 2}) |> Axon.depthwise_conv(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 2}) |> Axon.depthwise_conv(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 2}))) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model = Axon.input({nil, 1, 2}) |> Axon.depthwise_conv(1, name: "conv", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => _} = conv_params} = init_fn.()
      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model = Axon.input({nil, 1, 2}) |> Axon.depthwise_conv(1, name: "conv", use_bias: false)
      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => k}} = params = init_fn.()
      assert predict_fn.(params, input) == Axon.Layers.depthwise_conv(input, k, Nx.tensor(0))
    end

    test "computes forward pass with channels last" do
      model = Axon.input({nil, 3, 3, 6}) |> Axon.depthwise_conv(2, name: "conv", channels: :last)
      input = Nx.random_uniform({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => k, "bias" => b}} = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.depthwise_conv(input, k, b, channels: :last)
    end
  end

  describe "convolution transpose" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 2, 2}) |> Axon.conv_transpose(32, name: "conv")

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {32, 3, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 3, 2, 2})
        |> Axon.conv_transpose(32, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {32, 3, 1, 1})
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 32, 32})
        |> Axon.conv_transpose(32, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.shape(kernel) == {32, 3, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {32})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 1, 4}) |> Axon.conv_transpose(3, name: "conv")
      input1 = Nx.random_uniform({1, 1, 4}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.conv_transpose(input1, kernel, bias)

      model2 = Axon.input({nil, 1, 4, 4}) |> Axon.conv_transpose(4, name: "conv")
      input2 = Nx.random_uniform({1, 1, 4, 4}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.conv_transpose(input2, kernel, bias)

      model3 = Axon.input({nil, 1, 2, 2, 2}) |> Axon.conv_transpose(5, name: "conv")
      input3 = Nx.random_uniform({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()
      assert predict_fn.(params, input3) == Axon.Layers.conv_transpose(input3, kernel, bias)
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, kernel_dilation: 1]

      model1 =
        Axon.input({nil, 1, 4})
        |> Axon.conv_transpose(1, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = Nx.random_uniform({1, 1, 4})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()

      assert predict_fn.(params, input1) ==
               Axon.Layers.conv_transpose(input1, kernel, bias, opts1)

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input({nil, 1, 4, 4})
        |> Axon.conv_transpose(8, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 1, 4, 4})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()

      assert predict_fn.(params, input2) ==
               Axon.Layers.conv_transpose(input2, kernel, bias, opts2)

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input({nil, 1, 2, 2, 2})
        |> Axon.conv_transpose(2, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 1, 2, 2, 2})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.()

      assert predict_fn.(params, input3) ==
               Axon.Layers.conv_transpose(input3, kernel, bias, opts3)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 2})
        |> Axon.conv_transpose(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"conv" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 2})])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1, 1})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 2}) |> Axon.conv_transpose(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 2}) |> Axon.conv_transpose(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 2}))) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model = Axon.input({nil, 1, 2}) |> Axon.conv_transpose(1, name: "conv", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => _} = conv_params} = init_fn.()
      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model = Axon.input({nil, 1, 2}) |> Axon.conv_transpose(1, name: "conv", use_bias: false)
      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => k}} = params = init_fn.()
      assert predict_fn.(params, input) == Axon.Layers.conv_transpose(input, k, Nx.tensor(0))
    end

    test "computes forward pass with channels last" do
      model = Axon.input({nil, 3, 3, 6}) |> Axon.conv_transpose(2, name: "conv", channels: :last)
      input = Nx.random_uniform({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"conv" => %{"kernel" => k, "bias" => b}} = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.conv_transpose(input, k, b, channels: :last)
    end
  end

  describe "separable convolution 2d" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 2, 2}) |> Axon.separable_conv2d(3, name: "conv")

      assert {init_fn, _} = Axon.compile(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = init_fn.()

      assert Nx.shape(k1) == {9, 1, 1, 1}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {9, 1, 1, 1}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(b1) == {9}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 3, 2, 2})
        |> Axon.separable_conv2d(3, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model1)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = init_fn.()

      assert k1 == Axon.Initializers.zeros(shape: {9, 1, 1, 1})
      assert k2 == Axon.Initializers.zeros(shape: {9, 1, 1, 1})
      assert Nx.shape(b1) == {9}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 2, 2})
        |> Axon.separable_conv2d(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model2)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = init_fn.()

      assert b1 == Axon.Initializers.zeros(shape: {9})
      assert b2 == Axon.Initializers.zeros(shape: {9})
      assert Nx.shape(k1) == {9, 1, 1, 1}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {9, 1, 1, 1}
      assert Nx.type(k2) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model = Axon.input({nil, 3, 2, 2}) |> Axon.separable_conv2d(3, name: "conv")
      input = Nx.random_uniform({1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = params = init_fn.()

      assert predict_fn.(params, input) == Axon.Layers.separable_conv2d(input, k1, b1, k2, b2)
    end

    test "computes forward pass with custom options" do
      opts = [strides: [2, 1], input_dilation: [1, 2], kernel_dilation: 1, padding: :same]

      model =
        Axon.input({nil, 3, 3, 2})
        |> Axon.separable_conv2d(3, [name: "conv", kernel_size: {2, 2}] ++ opts)

      input = Nx.random_uniform({1, 3, 3, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv2d(input, k1, b1, k2, b2, opts)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 3, 2})
        |> Axon.separable_conv2d(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{
               "conv" => %{
                 "kernel_1" => k1_grad,
                 "bias_1" => b1_grad,
                 "kernel_2" => k2_grad,
                 "bias_2" => b2_grad
               }
             } = Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 3, 2})])

      assert k1_grad == Nx.broadcast(0.0, {1, 1, 1, 1})
      assert b1_grad == Nx.broadcast(0.0, {1})
      assert k2_grad == Nx.broadcast(0.0, {1, 1, 1, 1})
      assert b2_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 3, 2}) |> Axon.separable_conv2d(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = init_fn.()

      assert Nx.type(k1) == {:bf, 16}
      assert Nx.type(b1) == {:bf, 16}
      assert Nx.type(k2) == {:bf, 16}
      assert Nx.type(b2) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 3, 2}) |> Axon.separable_conv2d(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 3, 2}))) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input({nil, 1, 2, 2}) |> Axon.separable_conv2d(1, name: "conv", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv" => %{"kernel_1" => _, "kernel_2" => _} = conv_params} = init_fn.()
      assert Map.has_key?(conv_params, "bias_1") == false
      assert Map.has_key?(conv_params, "bias_2") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input({nil, 1, 2, 2}) |> Axon.separable_conv2d(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"conv" => %{"kernel_1" => k1, "kernel_2" => k2}} = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv2d(input, k1, Nx.tensor(0), k2, Nx.tensor(0))
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input({nil, 3, 3, 6}) |> Axon.separable_conv2d(2, name: "conv", channels: :last)

      input = Nx.random_uniform({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{"conv" => %{"kernel_1" => k1, "kernel_2" => k2, "bias_1" => b1, "bias_2" => b2}} =
               params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv2d(input, k1, b1, k2, b2, channels: :last)
    end
  end

  describe "separable convolution 3d" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 3, 2, 2}) |> Axon.separable_conv3d(3, name: "conv")

      assert {init_fn, _} = Axon.compile(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = init_fn.()

      assert Nx.shape(k1) == {9, 1, 1, 1, 1}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {9, 1, 1, 1, 1}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(k3) == {9, 1, 1, 1, 1}
      assert Nx.type(k3) == {:f, 32}
      assert Nx.shape(b1) == {9}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}
      assert Nx.shape(b3) == {9}
      assert Nx.type(b3) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 3, 3, 2, 2})
        |> Axon.separable_conv3d(3, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model1)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = init_fn.()

      assert k1 == Axon.Initializers.zeros(shape: {9, 1, 1, 1, 1})
      assert k2 == Axon.Initializers.zeros(shape: {9, 1, 1, 1, 1})
      assert k3 == Axon.Initializers.zeros(shape: {9, 1, 1, 1, 1})
      assert Nx.shape(b1) == {9}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}
      assert Nx.shape(b3) == {9}
      assert Nx.type(b3) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 3, 2, 2})
        |> Axon.separable_conv3d(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model2)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = init_fn.()

      assert b1 == Axon.Initializers.zeros(shape: {9})
      assert b2 == Axon.Initializers.zeros(shape: {9})
      assert b3 == Axon.Initializers.zeros(shape: {9})
      assert Nx.shape(k1) == {9, 1, 1, 1, 1}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {9, 1, 1, 1, 1}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(k2) == {9, 1, 1, 1, 1}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(k3) == {9, 1, 1, 1, 1}
      assert Nx.type(k3) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model = Axon.input({nil, 3, 2, 2, 2}) |> Axon.separable_conv3d(3, name: "conv")
      input = Nx.random_uniform({1, 3, 2, 2, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3)
    end

    test "computes forward pass with custom options" do
      opts = [strides: [2, 1, 1], input_dilation: [1, 2, 1], kernel_dilation: 1, padding: :same]

      model =
        Axon.input({nil, 3, 2, 3, 3})
        |> Axon.separable_conv3d(3, [name: "conv", kernel_size: {2, 2, 1}] ++ opts)

      input = Nx.random_uniform({1, 3, 2, 3, 3})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3, opts)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 3, 2, 2})
        |> Axon.separable_conv3d(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{
               "conv" => %{
                 "kernel_1" => k1_grad,
                 "bias_1" => b1_grad,
                 "kernel_2" => k2_grad,
                 "bias_2" => b2_grad,
                 "kernel_3" => k3_grad,
                 "bias_3" => b3_grad
               }
             } = Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 3, 2, 2})])

      assert k1_grad == Nx.broadcast(0.0, {1, 1, 1, 1, 1})
      assert b1_grad == Nx.broadcast(0.0, {1})
      assert k2_grad == Nx.broadcast(0.0, {1, 1, 1, 1, 1})
      assert b2_grad == Nx.broadcast(0.0, {1})
      assert k3_grad == Nx.broadcast(0.0, {1, 1, 1, 1, 1})
      assert b3_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 3, 2, 2}) |> Axon.separable_conv3d(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = init_fn.()

      assert Nx.type(k1) == {:bf, 16}
      assert Nx.type(b1) == {:bf, 16}
      assert Nx.type(k2) == {:bf, 16}
      assert Nx.type(b2) == {:bf, 16}
      assert Nx.type(k3) == {:bf, 16}
      assert Nx.type(b3) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 3, 2, 2}) |> Axon.separable_conv3d(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 3, 2, 2}))) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input({nil, 1, 3, 2, 2}) |> Axon.separable_conv3d(1, name: "conv", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)

      assert %{"conv" => %{"kernel_1" => _, "kernel_2" => _, "kernel_3" => _} = conv_params} =
               init_fn.()

      assert Map.has_key?(conv_params, "bias_1") == false
      assert Map.has_key?(conv_params, "bias_2") == false
      assert Map.has_key?(conv_params, "bias_3") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input({nil, 1, 3, 2, 2}) |> Axon.separable_conv3d(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{"conv" => %{"kernel_1" => k1, "kernel_2" => k2, "kernel_3" => k3}} =
               params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv3d(
                 input,
                 k1,
                 Nx.tensor(0),
                 k2,
                 Nx.tensor(0),
                 k3,
                 Nx.tensor(0)
               )
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input({nil, 3, 3, 3, 6}) |> Axon.separable_conv3d(2, name: "conv", channels: :last)

      input = Nx.random_uniform({1, 3, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "kernel_2" => k2,
                 "kernel_3" => k3,
                 "bias_1" => b1,
                 "bias_2" => b2,
                 "bias_3" => b3
               }
             } = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3, channels: :last)
    end
  end

  @normalization_layers [:batch_norm, :layer_norm, :instance_norm]

  describe "normalization" do
    test "initializes in default case" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input({nil, 2}), [name: "norm"]])

          assert {init_fn, _predict_fn} = Axon.compile(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
          assert Nx.shape(gamma) == {2}
          assert Nx.type(gamma) == {:f, 32}
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 = apply(Axon, norm, [Axon.input({nil, 3, 2, 2}), [name: "norm"]])

        assert {init_fn, _predict_fn} = Axon.compile(model2)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
        assert Nx.shape(gamma) == {3}
        assert Nx.type(gamma) == {:f, 32}
        assert Nx.shape(beta) == {3}
        assert Nx.type(beta) == {:f, 32}
      end
    end

    test "initializes with custom initializers" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 =
            apply(Axon, norm, [Axon.input({nil, 2}), [name: "norm", gamma_initializer: :zeros]])

          assert {init_fn, _predict_fn} = Axon.compile(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
          assert gamma == Axon.Initializers.zeros(shape: {2})
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 =
          apply(Axon, norm, [
            Axon.input({nil, 3, 2, 2}),
            [name: "norm", beta_initializer: :zeros]
          ])

        assert {init_fn, _predict_fn} = Axon.compile(model2)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
        assert Nx.shape(gamma) == {3}
        assert Nx.type(gamma) == {:f, 32}
        assert beta == Axon.Initializers.zeros(shape: {3})
      end
    end

    test "computes forward pass with default options" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input({nil, 2}), [name: "norm"]])
          input1 = Nx.random_uniform({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.compile(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.()
          assert predict_fn.(params, input1) == apply(Axon.Layers, norm, [input1, gamma, beta])
        end

        model2 = apply(Axon, norm, [Axon.input({nil, 3, 2, 2}), [name: "norm"]])
        input2 = Nx.random_uniform({1, 3, 2, 2}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.compile(model2)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.()
        assert predict_fn.(params, input2) == apply(Axon.Layers, norm, [input2, gamma, beta])
      end
    end

    test "computes forward pass with custom options" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          opts1 = [channel_index: 1, epsilon: 1.0e-3]
          model1 = apply(Axon, norm, [Axon.input({nil, 2}), [name: "norm"] ++ opts1])
          input1 = Nx.random_uniform({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.compile(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.()

          assert predict_fn.(params, input1) ==
                   apply(Axon.Layers, norm, [input1, gamma, beta, opts1])
        end

        opts2 = [channel_index: 3, epsilon: 1.0e-4]
        model2 = apply(Axon, norm, [Axon.input({nil, 2, 2, 3}), [name: "norm"] ++ opts2])
        input2 = Nx.random_uniform({1, 2, 2, 3}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.compile(model2)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.()

        assert predict_fn.(params, input2) ==
                 apply(Axon.Layers, norm, [input2, gamma, beta, opts2])
      end
    end

    test "returns zero gradient for frozen parameters" do
      for norm <- @normalization_layers do
        model =
          apply(Axon, norm, [Axon.input({nil, 1, 2}), [name: "norm"]])
          |> Axon.freeze()

        assert {init_fn, predict_fn} = Axon.compile(model)

        backward = fn params, input ->
          Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
        end

        assert %{"norm" => %{"gamma" => gamma_grad, "beta" => beta_grad}} =
                 Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 2})])

        assert gamma_grad == Nx.broadcast(0.0, {1})
        assert beta_grad == Nx.broadcast(0.0, {1})
      end
    end

    test "initializes with parameter policy" do
      for norm <- @normalization_layers do
        model = apply(Axon, norm, [Axon.input({nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(params: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, _} = Axon.compile(mp_model)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
        assert Nx.type(gamma) == {:bf, 16}
        assert Nx.type(beta) == {:bf, 16}
      end
    end

    test "computes forward pass with output policy" do
      for norm <- @normalization_layers do
        model = apply(Axon, norm, [Axon.input({nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 2}))) == {:bf, 16}
      end
    end
  end

  describe "group normalization" do
    test "initializes in default case" do
      model = Axon.input({nil, 3}) |> Axon.group_norm(3, name: "norm")

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
      assert Nx.shape(gamma) == {3}
      assert Nx.type(gamma) == {:f, 32}
      assert Nx.shape(beta) == {3}
      assert Nx.type(beta) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 = Axon.input({nil, 3}) |> Axon.group_norm(3, name: "norm", gamma_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
      assert gamma == Axon.Initializers.zeros(shape: {3})
      assert Nx.shape(beta) == {3}
      assert Nx.type(beta) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 3}) |> Axon.group_norm(3, name: "norm", beta_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
      assert beta == Axon.Initializers.zeros(shape: {3})
      assert Nx.shape(gamma) == {3}
      assert Nx.type(gamma) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 2}) |> Axon.group_norm(2, name: "norm")
      input1 = Nx.random_uniform({1, 2})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.()

      assert predict_fn.(params, input1) ==
               Axon.Layers.group_norm(input1, gamma, beta, group_size: 2)

      model2 = Axon.input({nil, 3, 2, 2}) |> Axon.group_norm(3, name: "norm")
      input2 = Nx.random_uniform({1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.()

      assert predict_fn.(params, input2) ==
               Axon.Layers.group_norm(input2, gamma, beta, group_size: 3)
    end

    test "computes forward pass with custom options" do
      opts = [epsilon: 1.0e-3, channel_index: 3]
      model = Axon.input({nil, 2, 2, 3}) |> Axon.group_norm(3, [name: "norm"] ++ opts)
      input = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.group_norm(input, gamma, beta, [group_size: 3] ++ opts)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 2})
        |> Axon.group_norm(1, name: "norm")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"norm" => %{"gamma" => gamma_grad, "beta" => beta_grad}} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 2})])

      assert gamma_grad == Nx.broadcast(0.0, {2})
      assert beta_grad == Nx.broadcast(0.0, {2})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 2}) |> Axon.group_norm(1, name: "norm")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.()
      assert Nx.type(gamma) == {:bf, 16}
      assert Nx.type(beta) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 2}) |> Axon.group_norm(1, name: "norm")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 2}))) == {:bf, 16}
    end
  end

  describe "flatten" do
    test "initializes with no params" do
      model = Axon.input({nil, 32}) |> Axon.flatten()

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{} = init_fn.()
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 32}) |> Axon.flatten()
      input1 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.compile(model1)
      assert predict_fn.(%{}, input1) == Axon.Layers.flatten(input1)

      model2 = Axon.input({nil, 3, 32, 32}) |> Axon.flatten()
      input2 = Nx.random_uniform({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.compile(model2)
      assert predict_fn.(%{}, input2) == Axon.Layers.flatten(input2)
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 32}) |> Axon.flatten()
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 32}))) == {:bf, 16}
    end
  end

  describe "transpose" do
    test "initializes with no params" do
      model = Axon.input({nil, 3, 32}) |> Axon.transpose([1, 0])

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{} = init_fn.()
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 32}) |> Axon.transpose([0])
      input1 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.compile(model1)
      assert predict_fn.(%{}, input1) == Nx.transpose(input1, axes: [0, 1])

      model2 = Axon.input({nil, 3, 32, 32}) |> Axon.transpose([1, 0, 2])
      input2 = Nx.random_uniform({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.compile(model2)
      assert predict_fn.(%{}, input2) == Nx.transpose(input2, axes: [0, 2, 1, 3])
    end

    test "computes forward pass with constant" do
      model = Axon.constant(Nx.iota({1, 2, 3})) |> Axon.transpose([2, 1, 0])

      assert {_, predict_fn} = Axon.compile(model)
      assert predict_fn.(%{}, {}) == Nx.transpose(Nx.iota({1, 2, 3}, type: {:f, 32}))
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 32}) |> Axon.transpose([0])
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 32}))) == {:bf, 16}
    end
  end

  describe "reshape" do
    test "initializes with no params" do
      model = Axon.input({nil, 1, 32}) |> Axon.reshape({16, 2})

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{} = init_fn.()
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 32}) |> Axon.reshape({16, 2})
      input1 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.compile(model1)
      assert predict_fn.(%{}, input1) == Nx.reshape(input1, {1, 16, 2})

      model2 = Axon.input({nil, 3, 32, 32}) |> Axon.reshape({3, 16, 2, 32})
      input2 = Nx.random_uniform({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.compile(model2)
      assert predict_fn.(%{}, input2) == Nx.reshape(input2, {1, 3, 16, 2, 32})
    end

    test "computes forward pass with constant input" do
      model = Axon.constant(Nx.iota({6})) |> Axon.reshape({1, 2, 3})

      assert {_, predict_fn} = Axon.compile(model)
      assert predict_fn.(%{}, {}) == Nx.tensor([[[0, 1, 2], [3, 4, 5]]], type: {:f, 32})
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 32}) |> Axon.reshape({2, 16})
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 32}))) == {:bf, 16}
    end
  end

  describe "lstm" do
    test "initializes in default case" do
      model = Axon.input({nil, 32, 10}) |> Axon.lstm(64, name: "lstm")

      assert {init_fn, _predict_fn} = Axon.compile(model)

      assert %{
               "lstm" => %{
                 "wii" => wii,
                 "wif" => wif,
                 "wig" => wig,
                 "wio" => wio,
                 "whi" => whi,
                 "whf" => whf,
                 "whg" => whg,
                 "who" => who,
                 "bi" => bi,
                 "bf" => bf,
                 "bg" => bg,
                 "bo" => bo
               }
             } = init_fn.()

      # Input kernel
      assert Nx.shape(wii) == {10, 64}
      assert Nx.type(wii) == {:f, 32}
      assert Nx.shape(wif) == {10, 64}
      assert Nx.type(wif) == {:f, 32}
      assert Nx.shape(wig) == {10, 64}
      assert Nx.type(wig) == {:f, 32}
      assert Nx.shape(wio) == {10, 64}
      assert Nx.type(wio) == {:f, 32}

      # Hidden kernel
      assert Nx.shape(whi) == {64, 64}
      assert Nx.type(whi) == {:f, 32}
      assert Nx.shape(whf) == {64, 64}
      assert Nx.type(whf) == {:f, 32}
      assert Nx.shape(whg) == {64, 64}
      assert Nx.type(whg) == {:f, 32}
      assert Nx.shape(who) == {64, 64}
      assert Nx.type(who) == {:f, 32}

      # Bias
      assert Nx.shape(bi) == {64}
      assert Nx.type(bi) == {:f, 32}
      assert Nx.shape(bf) == {64}
      assert Nx.type(bf) == {:f, 32}
      assert Nx.shape(bg) == {64}
      assert Nx.type(bg) == {:f, 32}
      assert Nx.shape(bo) == {64}
      assert Nx.type(bo) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 32, 10}) |> Axon.lstm(64, name: "lstm", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)

      assert %{
               "lstm" => %{
                 "wii" => wii,
                 "wif" => wif,
                 "wig" => wig,
                 "wio" => wio,
                 "whi" => whi,
                 "whf" => whf,
                 "whg" => whg,
                 "who" => who,
                 "bi" => bi,
                 "bf" => bf,
                 "bg" => bg,
                 "bo" => bo
               }
             } = init_fn.()

      # Input kernel
      assert wii == Axon.Initializers.zeros(shape: {10, 64})
      assert wif == Axon.Initializers.zeros(shape: {10, 64})
      assert wig == Axon.Initializers.zeros(shape: {10, 64})
      assert wio == Axon.Initializers.zeros(shape: {10, 64})

      # Hidden kernel
      assert whi == Axon.Initializers.zeros(shape: {64, 64})
      assert whf == Axon.Initializers.zeros(shape: {64, 64})
      assert whg == Axon.Initializers.zeros(shape: {64, 64})
      assert who == Axon.Initializers.zeros(shape: {64, 64})

      # Bias
      assert Nx.shape(bi) == {64}
      assert Nx.type(bi) == {:f, 32}
      assert Nx.shape(bf) == {64}
      assert Nx.type(bf) == {:f, 32}
      assert Nx.shape(bg) == {64}
      assert Nx.type(bg) == {:f, 32}
      assert Nx.shape(bo) == {64}
      assert Nx.type(bo) == {:f, 32}

      model2 = Axon.input({nil, 32, 10}) |> Axon.lstm(64, name: "lstm", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)

      assert %{
               "lstm" => %{
                 "wii" => wii,
                 "wif" => wif,
                 "wig" => wig,
                 "wio" => wio,
                 "whi" => whi,
                 "whf" => whf,
                 "whg" => whg,
                 "who" => who,
                 "bi" => bi,
                 "bf" => bf,
                 "bg" => bg,
                 "bo" => bo
               }
             } = init_fn.()

      # Input kernel
      assert Nx.shape(wii) == {10, 64}
      assert Nx.type(wii) == {:f, 32}
      assert Nx.shape(wif) == {10, 64}
      assert Nx.type(wif) == {:f, 32}
      assert Nx.shape(wig) == {10, 64}
      assert Nx.type(wig) == {:f, 32}
      assert Nx.shape(wio) == {10, 64}
      assert Nx.type(wio) == {:f, 32}

      # Hidden kernel
      assert Nx.shape(whi) == {64, 64}
      assert Nx.type(whi) == {:f, 32}
      assert Nx.shape(whf) == {64, 64}
      assert Nx.type(whf) == {:f, 32}
      assert Nx.shape(whg) == {64, 64}
      assert Nx.type(whg) == {:f, 32}
      assert Nx.shape(who) == {64, 64}
      assert Nx.type(who) == {:f, 32}

      # Bias
      assert bi == Axon.Initializers.zeros(shape: {64})
      assert bf == Axon.Initializers.zeros(shape: {64})
      assert bg == Axon.Initializers.zeros(shape: {64})
      assert bo == Axon.Initializers.zeros(shape: {64})
    end

    test "computes forward pass with default options" do
      model = Axon.input({nil, 8, 2}) |> Axon.lstm(2, name: "lstm", recurrent_initializer: :zeros)
      input = Nx.random_uniform({1, 8, 2}, type: {:f, 32})

      init_carry =
        {Axon.Initializers.zeros(shape: {1, 1, 2}), Axon.Initializers.zeros(shape: {1, 1, 2})}

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "lstm" => %{
                 "wii" => wii,
                 "wif" => wif,
                 "wig" => wig,
                 "wio" => wio,
                 "whi" => whi,
                 "whf" => whf,
                 "whg" => whg,
                 "who" => who,
                 "bi" => bi,
                 "bf" => bf,
                 "bg" => bg,
                 "bo" => bo
               }
             } = params = init_fn.()

      k = {wii, wif, wig, wio}
      h = {whi, whf, whg, who}
      b = {bi, bf, bg, bo}

      assert {{_, _} = carry, seq} = predict_fn.(params, input)

      assert {carry, seq} ==
               Axon.Recurrent.dynamic_unroll(
                 &Axon.Recurrent.lstm_cell/5,
                 input,
                 init_carry,
                 k,
                 h,
                 b
               )
    end

    test "computes forward pass with custom options" do
      model1 =
        Axon.input({nil, 8, 2})
        |> Axon.lstm(2,
          name: "lstm",
          recurrent_initializer: :zeros,
          gate: :relu,
          activation: :sigmoid
        )

      input1 = Nx.random_uniform({1, 8, 2}, type: {:f, 32})

      init_carry1 =
        {Axon.Initializers.zeros(shape: {1, 1, 2}), Axon.Initializers.zeros(shape: {1, 1, 2})}

      cell_fn1 = fn i, c, k, h, b ->
        Axon.Recurrent.lstm_cell(
          i,
          c,
          k,
          h,
          b,
          &Axon.Activations.relu/1,
          &Axon.Activations.sigmoid/1
        )
      end

      assert {init_fn, predict_fn} = Axon.compile(model1)

      assert %{
               "lstm" => %{
                 "wii" => wii,
                 "wif" => wif,
                 "wig" => wig,
                 "wio" => wio,
                 "whi" => whi,
                 "whf" => whf,
                 "whg" => whg,
                 "who" => who,
                 "bi" => bi,
                 "bf" => bf,
                 "bg" => bg,
                 "bo" => bo
               }
             } = params = init_fn.()

      k = {wii, wif, wig, wio}
      h = {whi, whf, whg, who}
      b = {bi, bf, bg, bo}

      assert {{_, _} = carry, seq} = predict_fn.(params, input1)
      assert {carry, seq} == Axon.Recurrent.dynamic_unroll(cell_fn1, input1, init_carry1, k, h, b)

      model2 =
        Axon.input({nil, 8, 2})
        |> Axon.lstm(2, name: "lstm", unroll: :static, recurrent_initializer: :zeros)

      input2 = Nx.random_uniform({1, 8, 2}, type: {:f, 32})

      init_carry2 =
        {Axon.Initializers.zeros(shape: {1, 1, 2}), Axon.Initializers.zeros(shape: {1, 1, 2})}

      cell_fn2 = &Axon.Recurrent.lstm_cell/5

      assert {init_fn, predict_fn} = Axon.compile(model2)

      assert %{
               "lstm" => %{
                 "wii" => wii,
                 "wif" => wif,
                 "wig" => wig,
                 "wio" => wio,
                 "whi" => whi,
                 "whf" => whf,
                 "whg" => whg,
                 "who" => who,
                 "bi" => bi,
                 "bf" => bf,
                 "bg" => bg,
                 "bo" => bo
               }
             } = params = init_fn.()

      k = {wii, wif, wig, wio}
      h = {whi, whf, whg, who}
      b = {bi, bf, bg, bo}

      assert {{c1, c2}, seq} = predict_fn.(params, input2)

      {{c1_static, c2_static}, seq_static} =
        Axon.Recurrent.static_unroll(cell_fn2, input2, init_carry2, k, h, b)

      assert Nx.all_close(c1, c1_static) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(c2, c2_static) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(seq, seq_static) == Nx.tensor(1, type: {:u, 8})
    end

    test "computes forward pass with hidden state" do
      seq = Axon.input({nil, 8, 2})
      {carry, _} = seq |> Axon.lstm(2, name: "encode", recurrent_initializer: :zeros)
      model = Axon.lstm(seq, 2, name: "decode", hidden_state: carry)
      input = Nx.random_uniform({1, 8, 2})

      assert {init_fn, predict_fn} = Axon.compile(model)

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        init_carry =
          {Axon.Initializers.zeros(shape: {1, 1, 2}), Axon.Initializers.zeros(shape: {1, 1, 2})}

        {carr, _} =
          Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.lstm_cell/5, inp, init_carry, ei, eh, eb)

        Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.lstm_cell/5, inp, carr, di, dh, db)
      end

      assert %{
               "encode" => %{
                 "wii" => eii,
                 "wif" => eif,
                 "wig" => eig,
                 "wio" => eio,
                 "whi" => ehi,
                 "whf" => ehf,
                 "whg" => ehg,
                 "who" => eho,
                 "bi" => ebi,
                 "bf" => ebf,
                 "bg" => ebg,
                 "bo" => ebo
               },
               "decode" => %{
                 "wii" => dii,
                 "wif" => dif,
                 "wig" => dig,
                 "wio" => dio,
                 "whi" => dhi,
                 "whf" => dhf,
                 "whg" => dhg,
                 "who" => dho,
                 "bi" => dbi,
                 "bf" => dbf,
                 "bg" => dbg,
                 "bo" => dbo
               }
             } = params = init_fn.()

      enc = {{eii, eif, eig, eio}, {ehi, ehf, ehg, eho}, {ebi, ebf, ebg, ebo}}
      dec = {{dii, dif, dig, dio}, {dhi, dhf, dhg, dho}, {dbi, dbf, dbg, dbo}}

      assert predict_fn.(params, input) == equiv_fn.(input, enc, dec)
    end

    # TODO(seanmor5): Update this with https://github.com/elixir-nx/axon/issues/90
    test "returns zero gradient for frozen parameters" do
      {_, out} =
        Axon.input({nil, 2, 1})
        |> Axon.lstm(1, name: "lstm", unroll: :static)

      model = Axon.freeze(out)

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{
               "lstm" => %{
                 "wii" => wii_grad,
                 "wif" => wif_grad,
                 "wig" => wig_grad,
                 "wio" => wio_grad,
                 "whi" => whi_grad,
                 "whf" => whf_grad,
                 "whg" => whg_grad,
                 "who" => who_grad,
                 "bi" => bi_grad,
                 "bf" => bf_grad,
                 "bg" => bg_grad,
                 "bo" => bo_grad
               }
             } = Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 2, 1})])

      assert wii_grad == Nx.broadcast(0.0, {1, 1})
      assert wif_grad == Nx.broadcast(0.0, {1, 1})
      assert wig_grad == Nx.broadcast(0.0, {1, 1})
      assert wio_grad == Nx.broadcast(0.0, {1, 1})
      assert whi_grad == Nx.broadcast(0.0, {1, 1})
      assert whf_grad == Nx.broadcast(0.0, {1, 1})
      assert whg_grad == Nx.broadcast(0.0, {1, 1})
      assert who_grad == Nx.broadcast(0.0, {1, 1})
      assert bi_grad == Nx.broadcast(0.0, {1})
      assert bf_grad == Nx.broadcast(0.0, {1})
      assert bg_grad == Nx.broadcast(0.0, {1})
      assert bo_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with use_bias false" do
      model = Axon.input({nil, 2, 1}) |> Axon.lstm(2, name: "lstm", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)

      assert %{
               "lstm" =>
                 %{
                   "wii" => _,
                   "wif" => _,
                   "wig" => _,
                   "wio" => _,
                   "whi" => _,
                   "whf" => _,
                   "whg" => _,
                   "who" => _
                 } = lstm_params
             } = init_fn.()

      assert Map.has_key?(lstm_params, "bi") == false
      assert Map.has_key?(lstm_params, "bf") == false
      assert Map.has_key?(lstm_params, "bg") == false
      assert Map.has_key?(lstm_params, "bo") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input({nil, 2, 1})
        |> Axon.lstm(2, name: "lstm", use_bias: false, recurrent_initializer: :zeros)

      input = Nx.random_uniform({1, 2, 1})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "lstm" => %{
                 "wii" => wii,
                 "wif" => wif,
                 "wig" => wig,
                 "wio" => wio,
                 "whi" => whi,
                 "whf" => whf,
                 "whg" => whg,
                 "who" => who
               }
             } = params = init_fn.()

      k = {wii, wif, wig, wio}
      h = {whi, whf, whg, who}
      b = {Nx.tensor(0), Nx.tensor(0), Nx.tensor(0), Nx.tensor(0)}
      c = {Axon.Initializers.zeros(shape: {1, 1, 2}), Axon.Initializers.zeros(shape: {1, 1, 2})}

      assert predict_fn.(params, input) ==
               Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.lstm_cell/5, input, c, k, h, b)
    end

    # TODO(seanmor5): https://github.com/elixir-nx/axon/issues/90
    # test "initializes with parameter policy" do
    # end
    # TODO(seanmor5): https://github.com/elixir-nx/axon/issues/90
    # test "computes forward pass with output policy" do
    # end
  end

  describe "convlstm" do
    test "initializes in default case" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        in_channel_n = 3,
        _width = 6,
        _heigth = 6
      }

      out_channel_n = 4
      model = Axon.input(input_shape) |> Axon.conv_lstm(out_channel_n, name: "convlstm")

      assert {init_fn, _predict_fn} = Axon.compile(model)

      assert %{
               "convlstm" => %{
                 "wi" => wi,
                 "wh" => wh,
                 "b" => b
               }
             } = init_fn.()

      # Input kernel
      assert Nx.shape(wi) == {4 * out_channel_n, in_channel_n, 1, 1}
      assert Nx.type(wi) == {:f, 32}

      # Hidden kernel
      assert Nx.shape(wh) == {4 * out_channel_n, out_channel_n, 1, 1}
      assert Nx.type(wh) == {:f, 32}

      # Bias
      assert Nx.shape(b) == {4 * out_channel_n}
      assert Nx.type(b) == {:f, 32}
    end

    test "initializes with custom initializers" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        in_channel_n = 3,
        _width = 6,
        _heigth = 6
      }

      out_channel_n = 4

      model1 =
        Axon.input(input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)

      assert %{
               "convlstm" => %{
                 "wi" => wi,
                 "wh" => wh,
                 "b" => b
               }
             } = init_fn.()

      # Input kernel
      assert wi == Axon.Initializers.zeros(shape: {4 * out_channel_n, in_channel_n, 1, 1})

      # Hidden kernel
      assert wh == Axon.Initializers.zeros(shape: {4 * out_channel_n, out_channel_n, 1, 1})

      # Bias
      assert Nx.shape(b) == {4 * out_channel_n}
      assert Nx.type(b) == {:f, 32}

      model2 =
        Axon.input(input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)

      assert %{
               "convlstm" => %{
                 "wi" => wi,
                 "wh" => wh,
                 "b" => b
               }
             } = init_fn.()

      # Input kernel
      assert Nx.shape(wi) == {4 * out_channel_n, in_channel_n, 1, 1}
      assert Nx.type(wi) == {:f, 32}

      # Hidden kernel
      assert Nx.shape(wh) == {4 * out_channel_n, out_channel_n, 1, 1}
      assert Nx.type(wh) == {:f, 32}

      # Bias
      assert b == Axon.Initializers.zeros(shape: {4 * out_channel_n})
    end

    test "computes forward pass with dynamic unroll and equal number of input and output channels" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        heigth = 6
      }

      out_channel_n = 3
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, heigth}

      model =
        Axon.input(input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", recurrent_initializer: :zeros)

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> Nx.random_uniform(type: {:f, 32})

      init_carry =
        {Axon.Initializers.zeros(shape: hidden_shape_real),
         Axon.Initializers.zeros(shape: hidden_shape_real)}

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "convlstm" => %{
                 "wi" => wi,
                 "wh" => wh,
                 "b" => b
               }
             } = params = init_fn.()

      k = {wi}
      h = {wh}
      b = {b}

      assert {{_, _} = carry, seq} = predict_fn.(params, input)

      assert {carry, seq} ==
               Axon.Recurrent.dynamic_unroll(
                 &Axon.Recurrent.conv_lstm_cell/5,
                 input,
                 init_carry,
                 k,
                 h,
                 b
               )
    end

    test "computes forward pass with static unroll and different number of input and output channels" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        heigth = 6
      }

      out_channel_n = 7
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, heigth}

      model =
        Axon.input(input_shape)
        |> Axon.conv_lstm(out_channel_n,
          name: "convlstm",
          recurrent_initializer: :zeros,
          unroll: :static
        )

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> Nx.random_uniform(type: {:f, 32})

      init_carry =
        {Axon.Initializers.zeros(shape: hidden_shape_real),
         Axon.Initializers.zeros(shape: hidden_shape_real)}

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "convlstm" => %{
                 "wi" => wi,
                 "wh" => wh,
                 "b" => b
               }
             } = params = init_fn.()

      k = {wi}
      h = {wh}
      b = {b}

      assert {{_, _} = carry, seq} = predict_fn.(params, input)

      assert {carry, seq} ==
               Axon.Recurrent.static_unroll(
                 &Axon.Recurrent.conv_lstm_cell/5,
                 input,
                 init_carry,
                 k,
                 h,
                 b
               )
    end

    # First part fails by conv_lstm_cell:
    # no support for custon gate and activation functions
    test "computes forward pass with custom options" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        heigth = 6
      }

      out_channel_n = 3
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, heigth}

      model1 =
        Axon.input(input_shape)
        |> Axon.conv_lstm(out_channel_n,
          name: "convlstm",
          recurrent_initializer: :zeros,
          gate: :relu,
          activation: :sigmoid
        )

      input1 =
        input_shape
        |> put_elem(0, batch_real)
        |> Nx.random_uniform(type: {:f, 32})

      init_carry1 =
        {Axon.Initializers.zeros(shape: hidden_shape_real),
         Axon.Initializers.zeros(shape: hidden_shape_real)}

      cell_fn1 = fn i, c, k, h, b ->
        Axon.Recurrent.conv_lstm_cell(
          i,
          c,
          k,
          h,
          b
          # &Axon.Activations.relu/1,
          # &Axon.Activations.sigmoid/1
        )
      end

      assert {init_fn, predict_fn} = Axon.compile(model1)

      assert %{
               "convlstm" => %{
                 "wi" => wi,
                 "wh" => wh,
                 "b" => b
               }
             } = params = init_fn.()

      k = {wi}
      h = {wh}
      b = {b}

      assert {{_, _} = carry, seq} = predict_fn.(params, input1)
      assert {carry, seq} == Axon.Recurrent.dynamic_unroll(cell_fn1, input1, init_carry1, k, h, b)

      model2 =
        Axon.input(input_shape)
        |> Axon.conv_lstm(out_channel_n,
          name: "convlstm",
          unroll: :static,
          recurrent_initializer: :zeros
        )

      input2 =
        input_shape
        |> put_elem(0, batch_real)
        |> Nx.random_uniform(type: {:f, 32})

      init_carry2 =
        {Axon.Initializers.zeros(shape: hidden_shape_real),
         Axon.Initializers.zeros(shape: hidden_shape_real)}

      cell_fn2 = &Axon.Recurrent.conv_lstm_cell/5

      assert {init_fn, predict_fn} = Axon.compile(model2)

      assert %{
               "convlstm" => %{
                 "wi" => wi,
                 "wh" => wh,
                 "b" => b
               }
             } = params = init_fn.()

      k = {wi}
      h = {wh}
      b = {b}

      assert {{_, _} = carry, seq} = predict_fn.(params, input2)
      assert {carry, seq} == Axon.Recurrent.static_unroll(cell_fn2, input2, init_carry2, k, h, b)
    end

    test "computes forward pass with hidden state" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        heigth = 6
      }

      out_channel_n = 3
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, heigth}
      seq = Axon.input(input_shape)

      {carry, _} =
        seq |> Axon.conv_lstm(out_channel_n, name: "encode", recurrent_initializer: :zeros)

      model = Axon.conv_lstm(seq, out_channel_n, name: "decode", hidden_state: carry)

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> Nx.random_uniform(type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        init_carry =
          {Axon.Initializers.zeros(shape: hidden_shape_real),
           Axon.Initializers.zeros(shape: hidden_shape_real)}

        {carr, _} =
          Axon.Recurrent.dynamic_unroll(
            &Axon.Recurrent.conv_lstm_cell/5,
            inp,
            init_carry,
            ei,
            eh,
            eb
          )

        Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.conv_lstm_cell/5, inp, carr, di, dh, db)
      end

      assert %{
               "encode" => %{
                 "wi" => ei,
                 "wh" => eh,
                 "b" => eb
               },
               "decode" => %{
                 "wi" => di,
                 "wh" => dh,
                 "b" => db
               }
             } = params = init_fn.()

      enc = {{ei}, {eh}, {eb}}
      dec = {{di}, {dh}, {db}}

      assert predict_fn.(params, input) == equiv_fn.(input, enc, dec)
    end

    test "returns zero gradient for frozen parameters" do
      input_shape = {
        _batch = nil,
        _sequence_length = 2,
        in_channel_n = 3,
        _width = 4,
        _heigth = 4
      }

      out_channel_n = 3
      batch_real = 1

      {_, out} =
        Axon.input(input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", unroll: :static)

      model = Axon.freeze(out)

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> Nx.random_uniform(type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{
               "convlstm" => %{
                 "wi" => wi_grad,
                 "wh" => wh_grad,
                 "b" => b_grad
               }
             } = Nx.Defn.jit(backward, [init_fn.(), input])

      assert wi_grad == Nx.broadcast(0.0, {4 * out_channel_n, in_channel_n, 1, 1})
      assert wh_grad == Nx.broadcast(0.0, {4 * out_channel_n, out_channel_n, 1, 1})
      assert b_grad == Nx.broadcast(0.0, {4 * out_channel_n})
    end

    test "computes forward pass with use_bias false" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        heigth = 6
      }

      out_channel_n = 3
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, heigth}

      model =
        Axon.input(input_shape)
        |> Axon.conv_lstm(out_channel_n,
          name: "convlstm",
          use_bias: false,
          recurrent_initializer: :zeros
        )

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> Nx.random_uniform(type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "convlstm" => %{
                 "wi" => wi,
                 "wh" => wh
               }
             } = params = init_fn.()

      k = {wi}
      h = {wh}
      b = {Nx.broadcast(0, 4 * out_channel_n)}

      c =
        {Axon.Initializers.zeros(shape: hidden_shape_real),
         Axon.Initializers.zeros(shape: hidden_shape_real)}

      assert predict_fn.(params, input) ==
               Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.conv_lstm_cell/5, input, c, k, h, b)
    end
  end

  describe "gru" do
    test "initializes in default case" do
      model = Axon.input({nil, 32, 10}) |> Axon.gru(64, name: "gru")

      assert {init_fn, _} = Axon.compile(model)

      assert %{
               "gru" => %{
                 "wir" => wir,
                 "wiz" => wiz,
                 "win" => win,
                 "whr" => whr,
                 "whz" => whz,
                 "whn" => whn,
                 "br" => br,
                 "bz" => bz,
                 "bhn" => bhn,
                 "bin" => bin
               }
             } = init_fn.()

      assert Nx.shape(wir) == {10, 64}
      assert Nx.type(wir) == {:f, 32}
      assert Nx.shape(wiz) == {10, 64}
      assert Nx.type(wiz) == {:f, 32}
      assert Nx.shape(win) == {10, 64}
      assert Nx.type(win) == {:f, 32}
      assert Nx.shape(whr) == {64, 64}
      assert Nx.type(whr) == {:f, 32}
      assert Nx.shape(whz) == {64, 64}
      assert Nx.type(whz) == {:f, 32}
      assert Nx.shape(whn) == {64, 64}
      assert Nx.type(whn) == {:f, 32}
      assert Nx.shape(br) == {64}
      assert Nx.type(br) == {:f, 32}
      assert Nx.shape(bz) == {64}
      assert Nx.type(bz) == {:f, 32}
      assert Nx.shape(bhn) == {64}
      assert Nx.type(bhn) == {:f, 32}
      assert Nx.shape(bin) == {64}
      assert Nx.type(bin) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 = Axon.input({nil, 32, 10}) |> Axon.gru(64, name: "gru", kernel_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model1)

      assert %{
               "gru" => %{
                 "wir" => wir,
                 "wiz" => wiz,
                 "win" => win,
                 "whr" => whr,
                 "whz" => whz,
                 "whn" => whn,
                 "br" => br,
                 "bz" => bz,
                 "bhn" => bhn,
                 "bin" => bin
               }
             } = init_fn.()

      assert wir == Axon.Initializers.zeros(shape: {10, 64})
      assert wiz == Axon.Initializers.zeros(shape: {10, 64})
      assert win == Axon.Initializers.zeros(shape: {10, 64})
      assert whr == Axon.Initializers.zeros(shape: {64, 64})
      assert whz == Axon.Initializers.zeros(shape: {64, 64})
      assert whn == Axon.Initializers.zeros(shape: {64, 64})
      assert Nx.shape(br) == {64}
      assert Nx.type(br) == {:f, 32}
      assert Nx.shape(bz) == {64}
      assert Nx.type(bz) == {:f, 32}
      assert Nx.shape(bhn) == {64}
      assert Nx.type(bhn) == {:f, 32}
      assert Nx.shape(bin) == {64}
      assert Nx.type(bin) == {:f, 32}

      model2 = Axon.input({nil, 32, 10}) |> Axon.gru(64, name: "gru", bias_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model2)

      assert %{
               "gru" => %{
                 "wir" => wir,
                 "wiz" => wiz,
                 "win" => win,
                 "whr" => whr,
                 "whz" => whz,
                 "whn" => whn,
                 "br" => br,
                 "bz" => bz,
                 "bhn" => bhn,
                 "bin" => bin
               }
             } = init_fn.()

      assert Nx.shape(wir) == {10, 64}
      assert Nx.type(wir) == {:f, 32}
      assert Nx.shape(wiz) == {10, 64}
      assert Nx.type(wiz) == {:f, 32}
      assert Nx.shape(win) == {10, 64}
      assert Nx.type(win) == {:f, 32}
      assert Nx.shape(whr) == {64, 64}
      assert Nx.type(whr) == {:f, 32}
      assert Nx.shape(whz) == {64, 64}
      assert Nx.type(whz) == {:f, 32}
      assert Nx.shape(whn) == {64, 64}
      assert Nx.type(whn) == {:f, 32}
      assert br == Axon.Initializers.zeros(shape: {64})
      assert bz == Axon.Initializers.zeros(shape: {64})
      assert bhn == Axon.Initializers.zeros(shape: {64})
      assert bin == Axon.Initializers.zeros(shape: {64})
    end

    test "computes forward pass with default options" do
      model = Axon.input({nil, 8, 2}) |> Axon.gru(2, name: "gru", recurrent_initializer: :zeros)
      input = Nx.random_uniform({1, 8, 2})
      carry = {Axon.Initializers.zeros(shape: {1, 1, 2})}

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "gru" => %{
                 "wir" => wir,
                 "wiz" => wiz,
                 "win" => win,
                 "whr" => whr,
                 "whz" => whz,
                 "whn" => whn,
                 "br" => br,
                 "bz" => bz,
                 "bhn" => bhn,
                 "bin" => bin
               }
             } = params = init_fn.()

      k = {wir, wiz, win}
      h = {whr, whz, whn}
      b = {br, bz, bin, bhn}

      assert predict_fn.(params, input) ==
               Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.gru_cell/5, input, carry, k, h, b)
    end

    test "computes forward pass with custom options" do
      model1 =
        Axon.input({nil, 8, 2})
        |> Axon.gru(2,
          name: "gru",
          recurrent_initializer: :zeros,
          gate: :relu,
          activation: :sigmoid
        )

      input1 = Nx.random_uniform({1, 8, 2})
      carry1 = {Axon.Initializers.zeros(shape: {1, 1, 2})}

      cell_fn1 = fn i, c, k, h, b ->
        Axon.Recurrent.gru_cell(
          i,
          c,
          k,
          h,
          b,
          &Axon.Activations.relu/1,
          &Axon.Activations.sigmoid/1
        )
      end

      assert {init_fn, predict_fn} = Axon.compile(model1)

      assert %{
               "gru" => %{
                 "wir" => wir,
                 "wiz" => wiz,
                 "win" => win,
                 "whr" => whr,
                 "whz" => whz,
                 "whn" => whn,
                 "br" => br,
                 "bz" => bz,
                 "bhn" => bhn,
                 "bin" => bin
               }
             } = params = init_fn.()

      k = {wir, wiz, win}
      h = {whr, whz, whn}
      b = {br, bz, bin, bhn}

      assert predict_fn.(params, input1) ==
               Axon.Recurrent.dynamic_unroll(cell_fn1, input1, carry1, k, h, b)

      model2 =
        Axon.input({nil, 8, 2})
        |> Axon.gru(2, name: "gru", recurrent_initializer: :zeros, unroll: :static)

      input2 = Nx.random_uniform({1, 8, 2})
      carry2 = {Axon.Initializers.zeros(shape: {1, 1, 2})}

      assert {init_fn, predict_fn} = Axon.compile(model2)

      assert %{
               "gru" => %{
                 "wir" => wir,
                 "wiz" => wiz,
                 "win" => win,
                 "whr" => whr,
                 "whz" => whz,
                 "whn" => whn,
                 "br" => br,
                 "bz" => bz,
                 "bhn" => bhn,
                 "bin" => bin
               }
             } = params = init_fn.()

      k = {wir, wiz, win}
      h = {whr, whz, whn}
      b = {br, bz, bin, bhn}

      assert predict_fn.(params, input2) ==
               Axon.Recurrent.static_unroll(&Axon.Recurrent.gru_cell/5, input2, carry2, k, h, b)
    end

    test "computes forward pass with hidden state" do
      seq = Axon.input({nil, 8, 2})
      {carry, _} = Axon.gru(seq, 2, name: "encode", recurrent_initializer: :zeros)
      model = Axon.gru(seq, 2, name: "decode", hidden_state: carry)
      input = Nx.random_uniform({1, 8, 2})
      carry = {Axon.Initializers.zeros(shape: {1, 1, 2})}

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        {carry, _} =
          Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.gru_cell/5, inp, carry, ei, eh, eb)

        Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.gru_cell/5, inp, carry, di, dh, db)
      end

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "encode" => %{
                 "wir" => eir,
                 "wiz" => eiz,
                 "win" => ein,
                 "whr" => ehr,
                 "whz" => ehz,
                 "whn" => ehn,
                 "br" => ebr,
                 "bz" => ebz,
                 "bhn" => ebhn,
                 "bin" => ebin
               },
               "decode" => %{
                 "wir" => dir,
                 "wiz" => diz,
                 "win" => din,
                 "whr" => dhr,
                 "whz" => dhz,
                 "whn" => dhn,
                 "br" => dbr,
                 "bz" => dbz,
                 "bhn" => dbhn,
                 "bin" => dbin
               }
             } = params = init_fn.()

      enc = {{eir, eiz, ein}, {ehr, ehz, ehn}, {ebr, ebz, ebin, ebhn}}
      dec = {{dir, diz, din}, {dhr, dhz, dhn}, {dbr, dbz, dbin, dbhn}}

      assert predict_fn.(params, input) == equiv_fn.(input, enc, dec)
    end

    test "initializes with use_bias false" do
      model = Axon.input({nil, 2, 1}) |> Axon.gru(2, name: "gru", use_bias: false)

      assert {init_fn, _} = Axon.compile(model)

      assert %{
               "gru" =>
                 %{
                   "wir" => _,
                   "wiz" => _,
                   "win" => _,
                   "whr" => _,
                   "whz" => _,
                   "whn" => _
                 } = gru_params
             } = init_fn.()

      assert Map.has_key?(gru_params, "br") == false
      assert Map.has_key?(gru_params, "bz") == false
      assert Map.has_key?(gru_params, "bin") == false
      assert Map.has_key?(gru_params, "bhn") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input({nil, 2, 1})
        |> Axon.gru(2, name: "gru", use_bias: false, recurrent_initializer: :zeros)

      input = Nx.random_uniform({1, 2, 1})
      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "gru" => %{
                 "wir" => wir,
                 "wiz" => wiz,
                 "win" => win,
                 "whr" => whr,
                 "whz" => whz,
                 "whn" => whn
               }
             } = params = init_fn.()

      k = {wir, wiz, win}
      h = {whr, whz, whn}
      b = {Nx.tensor(0), Nx.tensor(0), Nx.tensor(0), Nx.tensor(0)}
      c = {Axon.Initializers.zeros(shape: {1, 1, 2})}

      {{carry}, output} = predict_fn.(params, input)

      {{s_carry}, s_output} =
        Axon.Recurrent.dynamic_unroll(&Axon.Recurrent.gru_cell/5, input, c, k, h, b)

      assert Nx.all_close(carry, s_carry) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(output, s_output) == Nx.tensor(1, type: {:u, 8})
    end

    # TODO(seanmor5): https://github.com/elixir-nx/axon/issues/90
    # test ""
    # TODO(seanmor5): https://github.com/elixir-nx/axon/issues/90
    # test "returns zero gradient for frozen parameters" do
    # end
    # end
    # TODO(seanmor5): https://github.com/elixir-nx/axon/issues/90
    # test "computes forward pass with output policy" do
    # end
  end

  @binary_layers [:add, :subtract, :multiply]

  describe "binary operations" do
    test "initializes with no params" do
      for op <- @binary_layers do
        model = apply(Axon, op, [Axon.input({nil, 32}), Axon.input({nil, 32})])
        assert {init_fn, _} = Axon.compile(model)
        assert %{} == init_fn.()
      end
    end

    test "computes forward pass with default options" do
      for op <- @binary_layers do
        model1 = apply(Axon, op, [Axon.input({nil, 32}), Axon.input({nil, 32})])
        input1_1 = Nx.random_uniform({1, 32})
        input1_2 = Nx.random_uniform({1, 32})
        assert {_, predict_fn} = Axon.compile(model1)

        assert Nx.all_close(
                 predict_fn.(%{}, {input1_1, input1_2}),
                 apply(Nx, op, [input1_1, input1_2])
               ) == Nx.tensor(1, type: {:u, 8})

        model2 =
          apply(Axon, op, [[Axon.input({nil, 32}), Axon.input({nil, 32}), Axon.input({nil, 32})]])

        input2_1 = Nx.random_uniform({1, 32})
        input2_2 = Nx.random_uniform({1, 32})
        input2_3 = Nx.random_uniform({1, 32})
        assert {_, predict_fn} = Axon.compile(model2)

        assert Nx.all_close(
                 predict_fn.(%{}, {input2_1, input2_2, input2_3}),
                 apply(Nx, op, [apply(Nx, op, [input2_1, input2_2]), input2_3])
               ) == Nx.tensor(1, type: {:u, 8})
      end
    end

    test "computes forward pass with output policy" do
      for op <- @binary_layers do
        model = apply(Axon, op, [Axon.input({nil, 32}), Axon.input({nil, 32})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)
        input = {Nx.random_uniform({1, 32}), Nx.random_uniform({1, 32})}

        assert {_, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(%{}, input)) == {:bf, 16}
      end
    end

    test "computes forward pass with broadcasting" do
      inp1 = Nx.random_uniform({1, 1})
      inp2 = Nx.random_uniform({1, 2})

      for op <- @binary_layers do
        model = apply(Axon, op, [Axon.input({nil, 1}), Axon.input({nil, 2})])

        assert {_, predict_fn} = Axon.compile(model)
        assert predict_fn.(%{}, {inp1, inp2}) == apply(Nx, op, [inp1, inp2])
      end
    end
  end

  describe "concatenate" do
    test "initializes with no params" do
      model = Axon.concatenate(Axon.input({nil, 32}), Axon.input({nil, 32}))
      assert {init_fn, _} = Axon.compile(model)
      assert %{} == init_fn.()
    end

    test "computes forward pass with default options" do
      model1 = Axon.concatenate(Axon.input({nil, 32}), Axon.input({nil, 32}))
      input1_1 = Nx.random_uniform({1, 32})
      input1_2 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.compile(model1)

      assert predict_fn.(%{}, {input1_1, input1_2}) ==
               Nx.concatenate([input1_1, input1_2], axis: 1)

      model2 =
        Axon.concatenate([Axon.input({nil, 32}), Axon.input({nil, 32}), Axon.input({nil, 32})])

      input2_1 = Nx.random_uniform({1, 32})
      input2_2 = Nx.random_uniform({1, 32})
      input2_3 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.compile(model2)

      assert predict_fn.(%{}, {input2_1, input2_2, input2_3}) ==
               Nx.concatenate([input2_1, input2_2, input2_3], axis: 1)
    end

    test "computes forward pass with custom options" do
      model1 = Axon.concatenate(Axon.input({nil, 1, 32}), Axon.input({nil, 1, 32}), axis: 1)
      input1_1 = Nx.random_uniform({1, 1, 32})
      input1_2 = Nx.random_uniform({1, 1, 32})

      assert {_, predict_fn} = Axon.compile(model1)

      assert predict_fn.(%{}, {input1_1, input1_2}) ==
               Nx.concatenate([input1_1, input1_2], axis: 1)
    end

    test "computes forward pass with output policy" do
      model1 = Axon.concatenate(Axon.input({nil, 1, 32}), Axon.input({nil, 1, 32}), axis: 1)
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model1, policy)
      input1_1 = Nx.random_uniform({1, 1, 32})
      input1_2 = Nx.random_uniform({1, 1, 32})

      assert {_, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(%{}, {input1_1, input1_2})) == {:bf, 16}
    end
  end

  describe "pad" do
    test "initializes with no params" do
      model = Axon.input({nil, 3, 3}) |> Axon.pad([{1, 0}])
      assert {init_fn, _} = Axon.compile(model)
      assert %{} == init_fn.()
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 3, 3}) |> Axon.pad([{1, 0}])
      input1 = Nx.random_uniform({1, 3, 3})

      assert {_, predict_fn} = Axon.compile(model1)
      assert predict_fn.(%{}, input1) == Nx.pad(input1, 0, [{0, 0, 0}, {0, 0, 0}, {1, 0, 0}])

      model2 = Axon.input({nil, 3, 3, 3}) |> Axon.pad([{0, 1}, {0, 1}])
      input2 = Nx.random_uniform({1, 3, 3, 3})

      assert {_, predict_fn} = Axon.compile(model2)

      assert predict_fn.(%{}, input2) ==
               Nx.pad(input2, 0, [{0, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 1, 0}])

      model3 = Axon.input({nil, 3, 3, 3, 3}) |> Axon.pad([{0, 1}, {0, 1}, {1, 0}])
      input3 = Nx.random_uniform({1, 3, 3, 3, 3})

      assert {_, predict_fn} = Axon.compile(model3)

      assert predict_fn.(%{}, input3) ==
               Nx.pad(input3, 0, [{0, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}])
    end

    test "computes forward pass with custom options" do
      model = Axon.input({nil, 3, 3}) |> Axon.pad([{1, 0}], 2)
      input = Nx.random_uniform({1, 3, 3})

      assert {_, predict_fn} = Axon.compile(model)
      assert predict_fn.(%{}, input) == Nx.pad(input, 2, [{0, 0, 0}, {0, 0, 0}, {1, 0, 0}])
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 3, 3}) |> Axon.pad([{1, 0}])
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)
      input = Nx.random_uniform({1, 3, 3})

      assert {_, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(%{}, input)) == {:bf, 16}
    end
  end

  describe "nx" do
    test "computes special nx functions" do
      model = Axon.input({nil, 10}) |> Axon.nx(&Nx.sin/1)
      input = Nx.random_uniform({1, 10})

      assert {_, predict_fn} = Axon.compile(model)
      assert Nx.all_close(predict_fn.(%{}, input), Nx.sin(input)) == Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "cond" do
    test "initializes with no params" do
      inp = Axon.input({nil, 1})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.all?(x) end
      model = Axon.cond(inp, cond_fn, on_true, on_false)

      assert {init_fn, _} = Axon.compile(model)
      assert %{} == init_fn.()
    end

    test "computes forward pass with default options" do
      inp = Axon.input({nil, 2})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.all?(x) end

      input_1 = Nx.tensor([[1.0, 1.0]])
      input_2 = Nx.tensor([[0.0, 0.0]])

      model = Axon.cond(inp, cond_fn, on_true, on_false)

      assert {_, predict_fn} = Axon.compile(model)
      assert predict_fn.(%{}, input_1) == Axon.Activations.relu(input_1)
      assert predict_fn.(%{}, input_2) == Axon.Activations.sigmoid(input_2)
    end

    test "computes forward pass with output policy" do
      inp = Axon.input({nil, 1, 32})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.all?(x) end
      model1 = Axon.cond(inp, cond_fn, on_true, on_false)
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model1, policy)

      input1_1 = Nx.random_uniform({1, 1, 32})

      assert {_, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(%{}, input1_1)) == {:bf, 16}
    end

    test "raises on bad condition" do
      inp = Axon.input({nil, 1, 10})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.equal(x, 1) end

      model = Axon.cond(inp, cond_fn, on_true, on_false)

      assert_raise Axon.CompilerError, ~r/error while building prediction/, fn ->
        {_, predict_fn} = Axon.compile(model)
        predict_fn.(%{}, Nx.random_uniform({1, 1, 10}))
      end
    end
  end

  describe "split" do
    test "initializes with no parameters" do
      model = Axon.input({nil, 10}) |> Axon.split(5)

      assert {init_fn, _} = Axon.compile(model)
      assert init_fn.() == %{}
    end

    test "computes forward pass with default options" do
      model = Axon.input({nil, 10}) |> Axon.split(5)
      input = Nx.iota({1, 10}, type: {:f, 32})

      assert {_, predict_fn} = Axon.compile(model)

      assert predict_fn.(%{}, input) == {
               Nx.tensor([[0.0, 1.0]]),
               Nx.tensor([[2.0, 3.0]]),
               Nx.tensor([[4.0, 5.0]]),
               Nx.tensor([[6.0, 7.0]]),
               Nx.tensor([[8.0, 9.0]])
             }
    end
  end
end
