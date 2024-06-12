defmodule CompilerTest do
  use Axon.Case, async: true
  import ExUnit.CaptureLog

  alias Axon.MixedPrecision, as: AMP
  alias Axon.ModelState

  describe "input" do
    test "single input, single output" do
      model = Axon.input("input_0", shape: {nil, 1})
      input = random({1, 1}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
      assert_equal(predict_fn.(ModelState.empty(), input), input)
    end

    test "multi-input, map with default names" do
      model1 =
        {Axon.input("input_0", shape: {nil, 1}), Axon.input("input_1", shape: {nil, 1})}
        |> Axon.container()

      input1 = random({1, 1})
      input2 = random({1, 1})
      input = %{"input_0" => input1, "input_1" => input2}

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())

      assert_equal(
        {input1, input2},
        predict_fn.(ModelState.empty(), input)
      )
    end

    test "output map" do
      model = %{foo: Axon.input("input_0", shape: {nil, 1})} |> Axon.container()

      input = random({1, 1})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
      assert_equal(%{foo: input}, predict_fn.(ModelState.empty(), %{"input_0" => input}))
    end

    test "multi-input, multi-output, nested" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})

      model1 = {input1, {input1, {input2, {}}, input2, %{foo: input1}}} |> Axon.container()

      inp1 = random({1, 1})
      inp2 = random({1, 2})
      input = %{"input_0" => inp1, "input_1" => inp2}

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())

      assert_equal(
        {inp1, {inp1, {inp2, {}}, inp2, %{foo: inp1}}},
        predict_fn.(ModelState.empty(), input)
      )
    end

    test "multi-input, map with custom names" do
      x = Axon.input("x", shape: {nil, 1})
      y = Axon.input("y", shape: {nil, 1})
      z = Axon.input("z", shape: {nil, 1})
      model = {z, x, y} |> Axon.container()

      x_val = random({1, 1})
      y_val = random({1, 1})
      z_val = random({1, 1})
      input = %{"x" => x_val, "y" => y_val, "z" => z_val}

      assert {init_fn, predict_fn} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())

      assert_equal(
        {z_val, x_val, y_val},
        predict_fn.(ModelState.empty(), input)
      )
    end

    test "allows container inputs" do
      model = Axon.input("input_0", shape: %{foo: {nil, 1}, bar: {{nil, 2}, {nil, 3}}})

      input = %{foo: Nx.tensor([[1]]), bar: {Nx.tensor([[1, 2]]), Nx.tensor([[1, 2, 3]])}}

      assert_equal(Axon.predict(model, ModelState.empty(), %{"input_0" => input}), input)
    end

    test "allows lazy container inputs" do
      model = Axon.input("lazy_container") |> Axon.nx(fn x -> Nx.add(x.a, x.c) end)

      input = %LazyOnly{a: [[1]], b: [[2]], c: [[3]]}

      assert_equal(
        Axon.predict(model, ModelState.empty(), %{"lazy_container" => input}),
        Nx.tensor([[4]])
      )
    end

    test "raises if input not found, no default value" do
      model = Axon.input("input_0", shape: {nil, 32})
      input = random({1, 16})
      assert {_, predict_fn} = Axon.build(model)

      exception =
        assert_raise ArgumentError, fn -> predict_fn.(ModelState.empty(), %{foo: input}) end

      assert Exception.message(exception) =~
               "unable to find input"
    end

    test "raises helpful error messages" do
      input = Axon.input("input")
      x1 = Axon.dense(input, 32)
      x2 = Axon.dense(input, 64)
      model = Axon.add(x1, x2)

      {init_fn, _predict_fn} = Axon.build(model, debug: true)

      %Axon.CompileError{} =
        exception = catch_error(init_fn.(Nx.template({1, 16}, :f32), ModelState.empty()))

      message = Exception.message(exception)
      assert message =~ "exception found when compiling layer Axon.Layers.add/2 named add_0"
      assert message =~ "cannot broadcast tensor of dimensions {1, 32} to {1, 64}"
    end
  end

  describe "optional" do
    test "raises when predict compiles down to %Axon.None{}" do
      model =
        Axon.input("input_0", shape: {nil, 1}, optional: true)
        |> Axon.dense(1)

      assert_raise ArgumentError,
                   ~r/the compiled model will always result in %Axon.None{}/,
                   fn ->
                     Axon.predict(model, ModelState.empty(), %{})
                   end
    end

    test "passes optional nodes to the layer function" do
      input = Axon.input("input_0", shape: {nil, 1}, optional: true)

      model =
        Axon.layer(
          fn
            %Axon.None{}, _ -> 0
            %Nx.Tensor{}, _ -> 1
          end,
          [Axon.optional(input)]
        )

      {init_fn, predict_fn} = Axon.build(model)

      assert init_fn.(%{"input_0" => Nx.tensor([[20]])}, ModelState.empty()) == ModelState.empty()
      assert init_fn.(%{}, ModelState.empty()) == ModelState.empty()

      assert_equal(
        predict_fn.(ModelState.empty(), %{"input_0" => Nx.tensor([[20]])}),
        Nx.tensor(1)
      )

      assert_equal(
        predict_fn.(ModelState.empty(), %{}),
        Nx.tensor(0)
      )
    end

    test "propagates %Axon.None{} through subsequent layers" do
      input0 = Axon.input("input_0", shape: {nil, 1})
      input1 = Axon.input("input_1", shape: {nil, 1}, optional: true)

      sum =
        Axon.add(input0, input1)
        |> Axon.dense(1)
        |> Axon.sigmoid()

      model =
        Axon.layer(
          fn
            %Axon.None{}, _ -> Nx.tensor([0])
            %Nx.Tensor{}, _ -> Nx.tensor([1])
          end,
          [Axon.optional(sum)]
        )
        |> Axon.bias(bias_initializer: :zeros)

      {init_fn, predict_fn} = Axon.build(model)

      inputs = %{"input_0" => Nx.tensor([[20]])}

      assert %ModelState{data: params} = model_state = init_fn.(inputs, ModelState.empty())
      assert Map.keys(params) == ["bias_0"]

      assert_equal(predict_fn.(model_state, inputs), Nx.tensor([0]))

      inputs = %{"input_0" => Nx.tensor([[20]]), "input_1" => Nx.tensor([[20]])}

      assert %ModelState{data: params} = model_state = init_fn.(inputs, ModelState.empty())
      assert params |> Map.keys() |> Enum.sort() == ["bias_0", "dense_0"]

      assert_equal(predict_fn.(model_state, inputs), Nx.tensor([1]))
    end

    test "does not propagate %Axon.None{} further when returned by a layer" do
      x = Axon.input("input_0", shape: {nil, 1}, optional: true)

      x =
        Axon.layer(
          fn %Axon.None{} = none, _ -> none end,
          [Axon.optional(x)]
        )
        |> Axon.nx(fn _ -> flunk("should not evaluate") end)

      model = Axon.layer(fn _, _ -> 1 end, [Axon.optional(x)])

      assert_equal(Axon.predict(model, ModelState.empty(), %{}), Nx.tensor([1]))
    end
  end

  describe "constant" do
    test "initializes with no params" do
      model = Axon.constant(Nx.tensor(1.0))

      assert {init_fn, _} = Axon.build(model)

      assert ModelState.empty() == init_fn.(%{}, ModelState.empty())
    end

    test "computes forward pass with default options" do
      model = Axon.constant(Nx.tensor(1.0))

      assert {_, predict_fn} = Axon.build(model)
      assert_equal(predict_fn.(ModelState.empty(), {}), Nx.tensor(1.0))
    end

    test "computes forward pass with other layers" do
      model = Axon.add(Axon.constant(Nx.tensor(1.0)), Axon.constant(Nx.tensor(2.0)))

      assert {_, predict_fn} = Axon.build(model)
      assert_equal(predict_fn.(ModelState.empty(), {}), Nx.tensor(3.0))
    end

    test "computes forward pass with output policy" do
      model = Axon.constant(Nx.tensor(1.0))
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {_, predict_fn} = Axon.build(mp_model)
      assert_equal(predict_fn.(ModelState.empty(), {}), Nx.tensor(1.0, type: {:bf, 16}))
    end
  end

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :mish, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh] ++
                       [:log_softmax]

  describe "activations" do
    test "initializes with no params" do
      for activation <- @activation_layers do
        model = Axon.input("input_0", shape: {nil, 32}) |> Axon.activation(activation)
        input = random({1, 32})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert ModelState.empty() == init_fn.(input, ModelState.empty())
      end
    end

    test "computes forward pass with default options" do
      for activation <- @activation_layers do
        model = Axon.input("input_0", shape: {nil, 1}) |> Axon.activation(activation)
        input = random({1, 1})

        assert {_init_fn, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(ModelState.empty(), input),
          apply(Axon.Activations, activation, [input])
        )
      end
    end

    test "computes forward pass with custom options" do
      for activation <- [:celu, :elu, :leaky_relu] do
        model = Axon.input("input_0", shape: {nil, 32}) |> Axon.activation(activation, alpha: 0.8)
        input = random({1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(ModelState.empty(), input),
          apply(Axon.Activations, activation, [input, [alpha: 0.8]])
        )
      end
    end

    test "computes forward pass with output policy" do
      for activation <- @activation_layers do
        model = Axon.input("input_0", shape: {nil, 1}) |> Axon.activation(activation)
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 1})

        assert {init_fn, predict_fn} = Axon.build(mp_model)

        assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), random({1, 1}))) ==
                 {:bf, 16}
      end
    end
  end

  describe "bias" do
    test "initializes in default case" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.bias(name: "bias")

      input = random({1, 1})

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"bias" => %{"bias" => bias}}, parameters: %{"bias" => ["bias"]}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end
  end

  describe "dense" do
    test "initializes in default case" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(1, name: "dense")

      input = random({1, 1})

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"dense" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"dense" => ["bias", "kernel"]}
             } = init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      input = random({1, 1})

      model1 =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(1, name: "dense", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{
               data: %{"dense" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"dense" => ["bias", "kernel"]}
             } = init_fn.(input, ModelState.empty())

      assert_equal(kernel, zeros({1, 1}))
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(1, name: "dense", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)

      assert %ModelState{
               data: %{"dense" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"dense" => ["bias", "kernel"]}
             } = init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert_equal(bias, zeros({1}))
    end

    test "computes forward pass" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(1, name: "dense", kernel_initializer: :identity)

      input = Nx.iota({1, 1}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"dense" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"dense" => ["bias", "kernel"]}
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(predict_fn.(params, input), Axon.Layers.dense(input, kernel, bias))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, name: "dense")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 2})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{
               data: %{"dense" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"dense" => ["bias", "kernel"]}
             } = init_fn.(input, ModelState.empty())

      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, name: "dense")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, name: "dense", use_bias: false)

      input = random({1, 2})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{"dense" => %{"kernel" => _} = dense_params},
               parameters: %{"dense" => ["kernel"]}
             } = init_fn.(input, ModelState.empty())

      assert Map.has_key?(dense_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, name: "dense", use_bias: false)

      input = random({1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"dense" => %{"kernel" => k}},
               parameters: %{"dense" => ["kernel"]}
             } = params = init_fn.(input, ModelState.empty())

      assert_all_close(predict_fn.(params, input), Axon.Layers.dense(input, k, Nx.tensor(0.0)))
    end
  end

  describe "bilinear" do
    test "initializes in default case" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")

      inputs = %{"input_0" => random({1, 1}), "input_1" => random({1, 2})}

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"bilinear" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"bilinear" => ["bias", "kernel"]}
             } = init_fn.(inputs, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1, 2}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model1 = Axon.bilinear(input1, input2, 1, name: "bilinear", kernel_initializer: :zeros)

      inputs = %{"input_0" => random({1, 1}), "input_1" => random({1, 2})}

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{
               data: %{"bilinear" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"bilinear" => ["bias", "kernel"]}
             } = init_fn.(inputs, ModelState.empty())

      assert_equal(kernel, zeros({1, 2}))
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}

      model2 = Axon.bilinear(input1, input2, 1, name: "bilinear", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)

      assert %ModelState{
               data: %{"bilinear" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"bilinear" => ["bias", "kernel"]}
             } = init_fn.(inputs, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1, 2}
      assert Nx.type(kernel) == {:f, 32}
      assert_equal(bias, zeros({1}))
    end

    test "computes forward pass" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")

      input1 = Nx.iota({1, 1}, type: {:f, 32})
      input2 = Nx.iota({1, 2}, type: {:f, 32})

      inputs = %{"input_0" => input1, "input_1" => input2}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"bilinear" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"bilinear" => ["bias", "kernel"]}
             } =
               params = init_fn.(inputs, ModelState.empty())

      assert_equal(
        predict_fn.(params, inputs),
        Axon.Layers.bilinear(input1, input2, kernel, bias)
      )
    end

    test "computes forward pass with constant" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.constant(Nx.iota({2, 1}))
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")

      input1 = Nx.iota({2, 1}, type: {:f, 32})
      input2 = Nx.iota({2, 1}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"bilinear" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"bilinear" => ["bias", "kernel"]}
             } =
               params = init_fn.(input1, ModelState.empty())

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.bilinear(input1, input2, kernel, bias)
      )
    end

    test "initializes with parameter policy" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = %{"input_0" => random({1, 1}), "input_1" => random({1, 2})}

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{
               data: %{"bilinear" => %{"kernel" => kernel, "bias" => bias}},
               parameters: %{"bilinear" => ["bias", "kernel"]}
             } = init_fn.(input, ModelState.empty())

      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = %{"input_0" => random({1, 1}), "input_1" => random({1, 2})}

      assert {init_fn, predict_fn} = Axon.build(mp_model)

      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear", use_bias: false)

      input = %{"input_0" => random({1, 1}), "input_1" => random({1, 2})}

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{"bilinear" => %{"kernel" => _} = bilinear_params},
               parameters: %{"bilinear" => ["kernel"]}
             } = init_fn.(input, ModelState.empty())

      assert Map.has_key?(bilinear_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear", use_bias: false)

      inp1 = random({1, 1})
      inp2 = random({1, 2})

      input = %{"input_0" => inp1, "input_1" => inp2}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"bilinear" => %{"kernel" => k}},
               parameters: %{"bilinear" => ["kernel"]}
             } = params = init_fn.(input, ModelState.empty())

      assert_all_close(
        predict_fn.(params, input),
        Axon.Layers.bilinear(inp1, inp2, k, Nx.tensor(0.0))
      )
    end
  end

  describe "embedding" do
    test "initializes in default case" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.embedding(1, 1, name: "embedding")

      input = random({1, 1}) |> Nx.as_type(:s64)

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"embedding" => %{"kernel" => kernel}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1}
      assert Nx.type(kernel) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.embedding(1, 1, name: "embedding", kernel_initializer: :zeros)

      input = random({1, 1}) |> Nx.as_type(:s64)

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"embedding" => %{"kernel" => kernel}}} =
               init_fn.(input, ModelState.empty())

      assert_equal(kernel, zeros({1, 1}))
    end

    test "computes forward pass" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.embedding(1, 1, name: "embedding", kernel_initializer: :identity)

      input = Nx.tensor([[0]])

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"embedding" => %{"kernel" => kernel}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(predict_fn.(params, input), Axon.Layers.embedding(input, kernel))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.embedding(1, 1, name: "embedding")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.tensor([[0, 1]])

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{data: %{"embedding" => %{"kernel" => kernel}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.type(kernel) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.embedding(1, 1, name: "embedding")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.tensor([[0]])

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end
  end

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  describe "pooling" do
    test "initializes with no params" do
      for pool <- @pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])

        input = random({1, 32, 1})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert ModelState.empty() == init_fn.(input, ModelState.empty())
      end
    end

    test "computes forward pass with default options" do
      default_options = [kernel_size: 1]

      for pool <- @pooling_layers do
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])
        input1 = random({1, 32, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(ModelState.empty(), input1),
          apply(Axon.Layers, pool, [input1, default_options])
        )

        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 1})])
        input2 = random({1, 8, 4, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)

        assert_equal(
          predict_fn.(ModelState.empty(), input2),
          apply(Axon.Layers, pool, [input2, default_options])
        )

        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 2, 1})])
        input3 = random({1, 8, 4, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)

        assert_equal(
          predict_fn.(ModelState.empty(), input3),
          apply(Axon.Layers, pool, [input3, default_options])
        )
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @pooling_layers do
        opts1 = [kernel_size: 6]
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1}), opts1])
        input1 = random({1, 32, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(ModelState.empty(), input1),
          apply(Axon.Layers, pool, [input1, opts1])
        )

        opts2 = [kernel_size: 2, strides: 2, padding: :same]
        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 1}), opts2])
        input2 = random({1, 8, 4, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)

        assert_equal(
          predict_fn.(ModelState.empty(), input2),
          apply(Axon.Layers, pool, [input2, opts2])
        )

        opts3 = [kernel_size: {2, 1, 2}, strides: [1, 2, 1], padding: [{0, 1}, {1, 1}, {0, 2}]]
        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 2, 1}), opts3])
        input3 = random({1, 8, 4, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)

        assert_equal(
          predict_fn.(ModelState.empty(), input3),
          apply(Axon.Layers, pool, [input3, opts3])
        )
      end
    end

    test "lp_pool computes forward pass with custom norm" do
      model = Axon.input("input", shape: {nil, 32, 1}) |> Axon.lp_pool(norm: 3)
      input = random({1, 32, 1}, type: {:f, 32})

      assert {_, predict_fn} = Axon.build(model)

      assert_equal(
        predict_fn.(ModelState.empty(), input),
        Axon.Layers.lp_pool(input, kernel_size: {1}, norm: 3)
      )
    end

    test "computes forward pass with output policy" do
      for pool <- @pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 32, 1})

        assert {init_fn, predict_fn} = Axon.build(mp_model)

        assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), random({1, 32, 1}))) ==
                 {:bf, 16}
      end
    end

    test "computes forward pass with channels last" do
      for pool <- @pooling_layers do
        model =
          apply(Axon, pool, [
            Axon.input("input", shape: {nil, 32, 1}),
            [channels: :last, kernel_size: {2}]
          ])

        inp = random({1, 32, 1})

        assert {_, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(ModelState.empty(), inp),
          apply(Axon.Layers, pool, [inp, [kernel_size: {2}, strides: [2], channels: :last]])
        )
      end
    end
  end

  describe "blur_pool" do
    test "initializes with no params" do
      model = apply(Axon, :blur_pool, [Axon.input("input", shape: {nil, 32, 32, 1})])

      input = random({1, 32, 32, 1})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
    end

    test "computes forward pass with default options" do
      model2 = apply(Axon, :blur_pool, [Axon.input("input", shape: {nil, 8, 4, 1})])
      input2 = random({1, 8, 4, 1}, type: {:f, 32})

      assert {_, predict_fn} = Axon.build(model2)

      assert_equal(
        predict_fn.(ModelState.empty(), input2),
        apply(Axon.Layers, :blur_pool, [input2])
      )
    end

    test "computes forward pass with output policy" do
      model = apply(Axon, :blur_pool, [Axon.input("input", shape: {nil, 32, 32, 1})])
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 32, 32, 1})

      assert {init_fn, predict_fn} = Axon.build(mp_model)

      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), random({1, 32, 32, 1}))) ==
               {:bf, 16}
    end

    test "computes forward pass with channels last" do
      model =
        apply(Axon, :blur_pool, [
          Axon.input("input", shape: {nil, 32, 32, 1}),
          [channels: :last]
        ])

      inp = random({1, 32, 32, 1})

      assert {_, predict_fn} = Axon.build(model)

      assert_equal(
        predict_fn.(ModelState.empty(), inp),
        apply(Axon.Layers, :blur_pool, [inp, [channels: :last]])
      )
    end
  end

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool, :adaptive_lp_pool]

  describe "adaptive pooling" do
    test "initializes with no params" do
      for pool <- @adaptive_pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])

        input = random({1, 32, 1})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert ModelState.empty() == init_fn.(input, ModelState.empty())
      end
    end

    test "computes forward pass with default options" do
      for pool <- @adaptive_pooling_layers do
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])
        input1 = random({1, 32, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(ModelState.empty(), input1),
          apply(Axon.Layers, pool, [input1, [output_size: 32]])
        )

        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 1})])
        input2 = random({1, 8, 4, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)

        assert_equal(
          predict_fn.(ModelState.empty(), input2),
          apply(Axon.Layers, pool, [input2, [output_size: {8, 4}]])
        )

        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 2, 1})])
        input3 = random({1, 8, 4, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)

        assert_equal(
          predict_fn.(ModelState.empty(), input3),
          apply(Axon.Layers, pool, [input3, [output_size: {8, 4, 2}]])
        )
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @adaptive_pooling_layers do
        opts1 = [output_size: 27]
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1}), opts1])
        input1 = random({1, 32, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(ModelState.empty(), input1),
          apply(Axon.Layers, pool, [input1, opts1])
        )

        opts2 = [output_size: {2, 3}]
        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 1}), opts2])
        input2 = random({1, 8, 4, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)

        assert_equal(
          predict_fn.(ModelState.empty(), input2),
          apply(Axon.Layers, pool, [input2, opts2])
        )

        opts3 = [output_size: {4, 3, 1}]
        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 2, 1}), opts3])
        input3 = random({1, 8, 4, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)

        assert_equal(
          predict_fn.(ModelState.empty(), input3),
          apply(Axon.Layers, pool, [input3, opts3])
        )
      end
    end

    test "computes forward pass with output policy" do
      for pool <- @adaptive_pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 32, 1})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
      end
    end

    test "computes forward pass with channels last" do
      for pool <- @adaptive_pooling_layers do
        model =
          apply(Axon, pool, [
            Axon.input("input", shape: {nil, 32, 1}),
            [channels: :last, output_size: {27}]
          ])

        inp = random({1, 32, 1})

        assert {_, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(ModelState.empty(), inp),
          apply(Axon.Layers, pool, [inp, [output_size: {27}, channels: :last]])
        )
      end
    end
  end

  @global_pooling_layers [:global_max_pool, :global_avg_pool, :global_lp_pool]

  describe "global pooling" do
    test "initializes with no params" do
      for pool <- @global_pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 32})])

        input = random({1, 1, 32})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert ModelState.empty() == init_fn.(input, ModelState.empty())
      end
    end

    test "computes forward pass with default options" do
      for pool <- @global_pooling_layers do
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 4})])
        input1 = random({1, 1, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)
        assert_equal(predict_fn.(ModelState.empty(), input1), apply(Axon.Layers, pool, [input1]))

        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 2, 2})])
        input2 = random({1, 1, 2, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)
        assert_equal(predict_fn.(ModelState.empty(), input2), apply(Axon.Layers, pool, [input2]))

        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 2, 2, 1})])
        input3 = random({1, 1, 2, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)
        assert_equal(predict_fn.(ModelState.empty(), input3), apply(Axon.Layers, pool, [input3]))
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @global_pooling_layers do
        opts1 = [keep_axes: true]
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 2}), opts1])
        input1 = random({1, 1, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(ModelState.empty(), input1),
          apply(Axon.Layers, pool, [input1, opts1])
        )
      end
    end

    test "computes forward pass with output policy" do
      for pool <- @global_pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 2})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 1, 2})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
      end
    end

    test "computes forward pass with channels last" do
      for pool <- @global_pooling_layers do
        model1 =
          apply(Axon, pool, [
            Axon.input("input", shape: {nil, 32, 1}),
            [channels: :last, keep_axes: true]
          ])

        model2 =
          apply(Axon, pool, [
            Axon.input("input", shape: {nil, 32, 1}),
            [channels: :last, keep_axes: false]
          ])

        inp = random({1, 32, 1})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(ModelState.empty(), inp),
          apply(Axon.Layers, pool, [inp, [keep_axes: true, channels: :last]])
        )

        assert {_, predict_fn} = Axon.build(model2)

        assert_equal(
          predict_fn.(ModelState.empty(), inp),
          apply(Axon.Layers, pool, [inp, [keep_axes: false, channels: :last]])
        )
      end
    end
  end

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  describe "dropout" do
    test "initializes with key" do
      for dropout <- @dropout_layers do
        model =
          apply(Axon, dropout, [
            Axon.input("input", shape: {nil, 1, 32}),
            [name: "dropout", seed: 0]
          ])

        input = random({1, 1, 32})

        assert {init_fn, _predict_fn} = Axon.build(model, mode: :train)

        assert %ModelState{data: %{"dropout" => %{"key" => key}}, state: %{"dropout" => ["key"]}} =
                 init_fn.(input, ModelState.empty())

        assert_equal(key, Nx.Random.key(0))
      end
    end

    test "same key results in same mask" do
      for dropout <- @dropout_layers do
        model =
          apply(Axon, dropout, [
            Axon.input("input", shape: {nil, 1, 32}),
            [name: "dropout", seed: 0]
          ])

        input = random({1, 1, 32})

        assert {init_fn, predict_fn} = Axon.build(model, mode: :train)

        params = init_fn.(input, ModelState.empty())
        result1 = predict_fn.(params, input)
        result2 = predict_fn.(params, input)

        assert_equal(result1, result2)
      end
    end

    test "does not return same mask with updated key in training mode" do
      for dropout <- @dropout_layers do
        model =
          apply(Axon, dropout, [
            Axon.input("input", shape: {nil, 32, 32}),
            [rate: 0.5, name: "dropout", seed: 0]
          ])

        input = random({1, 16, 32})

        assert {init_fn, predict_fn} = Axon.build(model, mode: :train)

        params = init_fn.(input, ModelState.empty())
        %{prediction: result1, state: new_state} = predict_fn.(params, input)
        %{prediction: result2} = predict_fn.(ModelState.update(params, %{}, new_state), input)

        assert_not_equal(result1, result2)
      end
    end

    test "computes forward pass with default options" do
      for dropout <- @dropout_layers do
        model1 = apply(Axon, dropout, [Axon.input("input", shape: {nil, 32, 32})])
        input1 = random({1, 32, 32}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model1, mode: :train)
        %{prediction: result1} = predict_fn.(init_fn.(input1, ModelState.empty()), input1)

        assert Nx.shape(result1) == {1, 32, 32}
        assert Nx.type(result1) == {:f, 32}
        assert_not_equal(result1, input1)

        model2 = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 8, 4})])
        input2 = random({1, 1, 8, 4}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2, mode: :train)
        %{prediction: result2} = predict_fn.(init_fn.(input2, ModelState.empty()), input2)

        assert Nx.shape(result2) == {1, 1, 8, 4}
        assert Nx.type(result2) == {:f, 32}
        assert_not_equal(result2, input2)

        model3 = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 8, 4, 2})])
        input3 = random({1, 1, 8, 4, 2}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model3, mode: :train)
        %{prediction: result3} = predict_fn.(init_fn.(input3, ModelState.empty()), input3)

        assert Nx.shape(result3) == {1, 1, 8, 4, 2}
        assert Nx.type(result3) == {:f, 32}
        assert_not_equal(result3, input3)
      end
    end

    test "computes forward pass with custom options" do
      for dropout <- @dropout_layers do
        opts1 = [rate: 0.5]
        model1 = apply(Axon, dropout, [Axon.input("input", shape: {nil, 32, 128}), opts1])
        input1 = random({1, 32, 128}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model1, mode: :train)

        %{prediction: result} = predict_fn.(init_fn.(input1, ModelState.empty()), input1)

        assert Nx.shape(result) == {1, 32, 128}
        assert Nx.type(result) == {:f, 32}
        assert_not_equal(result, input1)
      end
    end

    test "computes forward pass with output policy" do
      for dropout <- @dropout_layers do
        model = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 32})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 1, 32})

        assert {init_fn, predict_fn} = Axon.build(mp_model, mode: :train)

        assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input).prediction) ==
                 {:bf, 16}
      end
    end

    test "not present in inference mode" do
      for dropout <- @dropout_layers do
        model = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 32})])
        input = random({1, 1, 32})

        {init_fn, predict_fn} = Axon.build(model)
        assert_equal(predict_fn.(init_fn.(input, ModelState.empty()), input), input)
      end
    end

    test "initializes correctly when node appears with and without dropout" do
      for dropout <- @dropout_layers do
        input = Axon.input("input", shape: {nil, 1, 32})
        model = Axon.add([input, apply(Axon, dropout, [input])])
        input = random({1, 1, 32})

        {init_fn, _predict_fn} = Axon.build(model)
        assert ModelState.empty() == init_fn.(input, ModelState.empty())
      end
    end
  end

  describe "convolution" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 32, 32, 3}) |> Axon.conv(64, name: "conv")

      input = Nx.template({1, 32, 32, 3}, {:f, 32})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1, 3, 64}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {64}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input", shape: {nil, 32, 32, 3})
        |> Axon.conv(32, name: "conv", kernel_initializer: :zeros)

      input = Nx.template({1, 32, 32, 3}, {:f, 32})

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert_equal(kernel, zeros({1, 1, 3, 32}))
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 3, 32, 32})
        |> Axon.conv(32, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1, 3, 32}
      assert Nx.type(kernel) == {:f, 32}
      assert_equal(bias, zeros({32}))
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv(2, name: "conv")
      input1 = random({1, 1, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input1, ModelState.empty())

      assert_equal(predict_fn.(params, input1), Axon.Layers.conv(input1, kernel, bias))

      model2 = Axon.input("input", shape: {nil, 1, 2, 2}) |> Axon.conv(3, name: "conv")
      input2 = random({1, 1, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input2, ModelState.empty())

      assert_equal(predict_fn.(params, input2), Axon.Layers.conv(input2, kernel, bias))

      model3 = Axon.input("input", shape: {nil, 1, 2, 2, 2}) |> Axon.conv(4, name: "conv")
      input3 = random({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model3)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input3, ModelState.empty())

      assert_equal(predict_fn.(params, input3), Axon.Layers.conv(input3, kernel, bias))
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, padding: :same, input_dilation: 2]

      model1 =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.conv(2, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = random({1, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input1, ModelState.empty())

      assert_equal(predict_fn.(params, input1), Axon.Layers.conv(input1, kernel, bias, opts1))

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input("input", shape: {nil, 4, 4, 1})
        |> Axon.conv(2, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = random({1, 4, 4, 1})

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input2, ModelState.empty())

      assert_equal(predict_fn.(params, input2), Axon.Layers.conv(input2, kernel, bias, opts2))

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input("input", shape: {nil, 2, 2, 2, 1})
        |> Axon.conv(4, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = random({1, 2, 2, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model3)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input3, ModelState.empty())

      assert_equal(predict_fn.(params, input3), Axon.Layers.conv(input3, kernel, bias, opts3))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 1, 32}) |> Axon.conv(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.template({1, 1, 32}, {:f, 32})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 32}) |> Axon.conv(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 32})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv(1, name: "conv", use_bias: false)

      input = random({1, 1, 2})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => _} = conv_params}} =
               init_fn.(input, ModelState.empty())

      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv(1, name: "conv", use_bias: false)

      input = random({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => k}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(predict_fn.(params, input), Axon.Layers.conv(input, k, Nx.tensor(0)))
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 6}) |> Axon.conv(2, name: "conv", channels: :last)

      input = random({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => k, "bias" => b}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(predict_fn.(params, input), Axon.Layers.conv(input, k, b, channels: :last))
    end
  end

  describe "depthwise convolution" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.depthwise_conv(3, name: "conv")

      input = random({1, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1, 1, 9}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {9}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input", shape: {nil, 2, 2, 3})
        |> Axon.depthwise_conv(3, name: "conv", kernel_initializer: :zeros)

      input = random({1, 2, 2, 3})

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert_equal(kernel, zeros({1, 1, 1, 9}))
      assert Nx.shape(bias) == {9}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 2, 2, 3})
        |> Axon.depthwise_conv(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1, 1, 9}
      assert Nx.type(kernel) == {:f, 32}
      assert_equal(bias, zeros({9}))
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 1, 8}) |> Axon.depthwise_conv(3, name: "conv")
      input1 = random({1, 1, 8}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input1, ModelState.empty())

      assert_equal(predict_fn.(params, input1), Axon.Layers.depthwise_conv(input1, kernel, bias))

      model2 = Axon.input("input", shape: {nil, 1, 2, 2}) |> Axon.depthwise_conv(4, name: "conv")
      input2 = random({1, 1, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input2, ModelState.empty())

      assert_equal(predict_fn.(params, input2), Axon.Layers.depthwise_conv(input2, kernel, bias))

      model3 =
        Axon.input("input", shape: {nil, 1, 2, 2, 2}) |> Axon.depthwise_conv(5, name: "conv")

      input3 = random({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model3)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input3, ModelState.empty())

      assert_equal(predict_fn.(params, input3), Axon.Layers.depthwise_conv(input3, kernel, bias))
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, padding: :same, input_dilation: 2]

      model1 =
        Axon.input("input", shape: {nil, 8, 1})
        |> Axon.depthwise_conv(1, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = random({1, 8, 1})

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input1, ModelState.empty())

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.depthwise_conv(input1, kernel, bias, opts1)
      )

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input("input", shape: {nil, 4, 4, 1})
        |> Axon.depthwise_conv(8, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = random({1, 4, 4, 1})

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input2, ModelState.empty())

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.depthwise_conv(input2, kernel, bias, opts2)
      )

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input("input", shape: {nil, 3, 2, 2, 1})
        |> Axon.depthwise_conv(2, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = random({1, 3, 2, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model3)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input3, ModelState.empty())

      assert_equal(
        predict_fn.(params, input3),
        Axon.Layers.depthwise_conv(input3, kernel, bias, opts3)
      )
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 1, 2}) |> Axon.depthwise_conv(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 2})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 2}) |> Axon.depthwise_conv(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.depthwise_conv(1, name: "conv", use_bias: false)

      input = random({1, 1, 2})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => _} = conv_params}} =
               init_fn.(input, ModelState.empty())

      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.depthwise_conv(1, name: "conv", use_bias: false)

      input = random({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => k}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(predict_fn.(params, input), Axon.Layers.depthwise_conv(input, k, Nx.tensor(0)))
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 6})
        |> Axon.depthwise_conv(2, name: "conv", channels: :last)

      input = random({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => k, "bias" => b}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.depthwise_conv(input, k, b, channels: :last)
      )
    end
  end

  describe "convolution transpose" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.conv_transpose(32, name: "conv")

      input = random({1, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1, 3, 32}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input", shape: {nil, 2, 2, 3})
        |> Axon.conv_transpose(32, name: "conv", kernel_initializer: :zeros)

      input = random({1, 2, 2, 3})

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert_equal(kernel, zeros({1, 1, 3, 32}))
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 32, 32, 3})
        |> Axon.conv_transpose(32, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(kernel) == {1, 1, 3, 32}
      assert Nx.type(kernel) == {:f, 32}
      assert_equal(bias, zeros({32}))
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 1, 4}) |> Axon.conv_transpose(3, name: "conv")
      input1 = random({1, 1, 4}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input1, ModelState.empty())

      assert_equal(predict_fn.(params, input1), Axon.Layers.conv_transpose(input1, kernel, bias))

      model2 = Axon.input("input", shape: {nil, 1, 4, 4}) |> Axon.conv_transpose(4, name: "conv")
      input2 = random({1, 1, 4, 4}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input2, ModelState.empty())

      assert_equal(predict_fn.(params, input2), Axon.Layers.conv_transpose(input2, kernel, bias))

      model3 =
        Axon.input("input", shape: {nil, 1, 2, 2, 2}) |> Axon.conv_transpose(5, name: "conv")

      input3 = random({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model3)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input3, ModelState.empty())

      assert_equal(predict_fn.(params, input3), Axon.Layers.conv_transpose(input3, kernel, bias))
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, kernel_dilation: 1]

      model1 =
        Axon.input("input", shape: {nil, 4, 1})
        |> Axon.conv_transpose(1, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = random({1, 4, 1})

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input1, ModelState.empty())

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.conv_transpose(input1, kernel, bias, opts1)
      )

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input("input", shape: {nil, 4, 4, 1})
        |> Axon.conv_transpose(8, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = random({1, 4, 4, 1})

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input2, ModelState.empty())

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.conv_transpose(input2, kernel, bias, opts2)
      )

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input("input", shape: {nil, 2, 2, 2, 1})
        |> Axon.conv_transpose(2, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = random({1, 2, 2, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model3)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(input3, ModelState.empty())

      assert_equal(
        predict_fn.(params, input3),
        Axon.Layers.conv_transpose(input3, kernel, bias, opts3)
      )
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv_transpose(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 2})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{data: %{"conv" => %{"kernel" => kernel, "bias" => bias}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv_transpose(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.conv_transpose(1, name: "conv", use_bias: false)

      input = random({1, 1, 2})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => _} = conv_params}} =
               init_fn.(input, ModelState.empty())

      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.conv_transpose(1, name: "conv", use_bias: false)

      input = random({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => k}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(predict_fn.(params, input), Axon.Layers.conv_transpose(input, k, Nx.tensor(0)))
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 6})
        |> Axon.conv_transpose(2, name: "conv", channels: :last)

      input = random({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel" => k, "bias" => b}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.conv_transpose(input, k, b, channels: :last)
      )
    end
  end

  describe "separable convolution 2d" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.separable_conv2d(3, name: "conv")

      input = random({1, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert Nx.shape(k1) == {1, 1, 1, 9}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {1, 1, 1, 9}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(b1) == {9}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input", shape: {nil, 2, 2, 3})
        |> Axon.separable_conv2d(3, name: "conv", kernel_initializer: :zeros)

      input = random({1, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model1)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert_equal(k1, zeros({1, 1, 1, 9}))
      assert_equal(k2, zeros({1, 1, 1, 9}))
      assert Nx.shape(b1) == {9}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 2, 2, 3})
        |> Axon.separable_conv2d(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _} = Axon.build(model2)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert_equal(b1, zeros({9}))
      assert_equal(b2, zeros({9}))
      assert Nx.shape(k1) == {1, 1, 1, 9}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {1, 1, 1, 9}
      assert Nx.type(k2) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model = Axon.input("input", shape: {nil, 3, 2, 2}) |> Axon.separable_conv2d(3, name: "conv")
      input = random({1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv2d(input, k1, b1, k2, b2)
      )
    end

    test "computes forward pass with custom options" do
      opts = [strides: [2, 1], input_dilation: [1, 2], kernel_dilation: 1, padding: :same]

      model =
        Axon.input("input", shape: {nil, 3, 3, 2})
        |> Axon.separable_conv2d(3, [name: "conv", kernel_size: {2, 2}] ++ opts)

      input = random({1, 3, 3, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv2d(input, k1, b1, k2, b2, opts)
      )
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 1, 3, 2}) |> Axon.separable_conv2d(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 3, 2})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert Nx.type(k1) == {:bf, 16}
      assert Nx.type(b1) == {:bf, 16}
      assert Nx.type(k2) == {:bf, 16}
      assert Nx.type(b2) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 3, 2}) |> Axon.separable_conv2d(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 3, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2, 2})
        |> Axon.separable_conv2d(1, name: "conv", use_bias: false)

      input = random({1, 1, 2, 2})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel_1" => _, "kernel_2" => _} = conv_params}} =
               init_fn.(input, ModelState.empty())

      assert Map.has_key?(conv_params, "bias_1") == false
      assert Map.has_key?(conv_params, "bias_2") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2, 2})
        |> Axon.separable_conv2d(1, name: "conv", use_bias: false)

      input = random({1, 1, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"conv" => %{"kernel_1" => k1, "kernel_2" => k2}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv2d(input, k1, Nx.tensor(0), k2, Nx.tensor(0))
      )
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 6})
        |> Axon.separable_conv2d(2, name: "conv", channels: :last)

      input = random({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{"kernel_1" => k1, "kernel_2" => k2, "bias_1" => b1, "bias_2" => b2}
               }
             } =
               params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv2d(input, k1, b1, k2, b2, channels: :last)
      )
    end
  end

  describe "separable convolution 3d" do
    test "initializes in default case" do
      model =
        Axon.input("input", shape: {nil, 3, 2, 2, 3}) |> Axon.separable_conv3d(3, name: "conv")

      input = random({1, 3, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2,
                   "kernel_3" => k3,
                   "bias_3" => b3
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert Nx.shape(k1) == {1, 1, 1, 1, 9}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {1, 1, 1, 1, 9}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(k3) == {1, 1, 1, 1, 9}
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
        Axon.input("input", shape: {nil, 3, 3, 2, 2})
        |> Axon.separable_conv3d(3, name: "conv", kernel_initializer: :zeros)

      input = random({1, 3, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model1)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2,
                   "kernel_3" => k3,
                   "bias_3" => b3
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert_equal(k1, zeros({1, 1, 1, 1, 9}))
      assert_equal(k2, zeros({1, 1, 1, 1, 9}))
      assert_equal(k3, zeros({1, 1, 1, 1, 9}))
      assert Nx.shape(b1) == {9}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}
      assert Nx.shape(b3) == {9}
      assert Nx.type(b3) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 3, 2, 2, 3})
        |> Axon.separable_conv3d(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _} = Axon.build(model2)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2,
                   "kernel_3" => k3,
                   "bias_3" => b3
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert_equal(b1, zeros({9}))
      assert_equal(b2, zeros({9}))
      assert_equal(b3, zeros({9}))
      assert Nx.shape(k1) == {1, 1, 1, 1, 9}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {1, 1, 1, 1, 9}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(k2) == {1, 1, 1, 1, 9}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(k3) == {1, 1, 1, 1, 9}
      assert Nx.type(k3) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model =
        Axon.input("input", shape: {nil, 3, 2, 2, 2}) |> Axon.separable_conv3d(3, name: "conv")

      input = random({1, 3, 2, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2,
                   "kernel_3" => k3,
                   "bias_3" => b3
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3)
      )
    end

    test "computes forward pass with custom options" do
      opts = [strides: [2, 1, 1], input_dilation: [1, 2, 1], kernel_dilation: 1, padding: :same]

      model =
        Axon.input("input", shape: {nil, 3, 2, 3, 3})
        |> Axon.separable_conv3d(3, [name: "conv", kernel_size: {2, 2, 1}] ++ opts)

      input = random({1, 3, 2, 3, 3})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2,
                   "kernel_3" => k3,
                   "bias_3" => b3
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3, opts)
      )
    end

    test "initializes with parameter policy" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2, 2}) |> Axon.separable_conv3d(1, name: "conv")

      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 3, 2, 2})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "bias_1" => b1,
                   "kernel_2" => k2,
                   "bias_2" => b2,
                   "kernel_3" => k3,
                   "bias_3" => b3
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert Nx.type(k1) == {:bf, 16}
      assert Nx.type(b1) == {:bf, 16}
      assert Nx.type(k2) == {:bf, 16}
      assert Nx.type(b2) == {:bf, 16}
      assert Nx.type(k3) == {:bf, 16}
      assert Nx.type(b3) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2, 2}) |> Axon.separable_conv3d(1, name: "conv")

      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2, 2})
        |> Axon.separable_conv3d(1, name: "conv", use_bias: false)

      input = random({1, 1, 3, 2, 2})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{"kernel_1" => _, "kernel_2" => _, "kernel_3" => _} = conv_params
               }
             } =
               init_fn.(input, ModelState.empty())

      assert Map.has_key?(conv_params, "bias_1") == false
      assert Map.has_key?(conv_params, "bias_2") == false
      assert Map.has_key?(conv_params, "bias_3") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2, 2})
        |> Axon.separable_conv3d(1, name: "conv", use_bias: false)

      input = random({1, 1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"conv" => %{"kernel_1" => k1, "kernel_2" => k2, "kernel_3" => k3}}
             } =
               params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv3d(
          input,
          k1,
          Nx.tensor(0),
          k2,
          Nx.tensor(0),
          k3,
          Nx.tensor(0)
        )
      )
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 3, 6})
        |> Axon.separable_conv3d(2, name: "conv", channels: :last)

      input = random({1, 3, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "conv" => %{
                   "kernel_1" => k1,
                   "kernel_2" => k2,
                   "kernel_3" => k3,
                   "bias_1" => b1,
                   "bias_2" => b2,
                   "bias_3" => b3
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3, channels: :last)
      )
    end
  end

  @normalization_with_stats_layers [:batch_norm, :instance_norm]

  describe "normalization with stats" do
    test "initializes in default case" do
      for norm <- @normalization_with_stats_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"]])

          input = random({1, 2})

          assert {init_fn, _predict_fn} = Axon.build(model1)

          assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                   init_fn.(input, ModelState.empty())

          assert Nx.shape(gamma) == {2}
          assert Nx.type(gamma) == {:f, 32}
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2, 2, 3}), [name: "norm"]])

        input = random({1, 2, 2, 3})

        assert {init_fn, _predict_fn} = Axon.build(model2)

        assert %ModelState{
                 data: %{
                   "norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}
                 }
               } =
                 init_fn.(input, ModelState.empty())

        assert Nx.shape(gamma) == {3}
        assert Nx.type(gamma) == {:f, 32}
        assert Nx.shape(beta) == {3}
        assert Nx.type(beta) == {:f, 32}
        assert Nx.shape(mean) == {3}
        assert Nx.type(mean) == {:f, 32}
        assert Nx.shape(var) == {3}
        assert Nx.type(var) == {:f, 32}
      end
    end

    test "initializes with custom initializers" do
      for norm <- @normalization_with_stats_layers do
        if norm != :instance_norm do
          model1 =
            apply(Axon, norm, [
              Axon.input("input", shape: {nil, 2}),
              [name: "norm", gamma_initializer: :zeros]
            ])

          input = random({1, 2})

          assert {init_fn, _predict_fn} = Axon.build(model1)

          assert %ModelState{
                   data: %{
                     "norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}
                   }
                 } =
                   init_fn.(input, ModelState.empty())

          assert_equal(gamma, zeros({2}))
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
          assert Nx.shape(mean) == {2}
          assert Nx.type(mean) == {:f, 32}
          assert Nx.shape(var) == {2}
          assert Nx.type(var) == {:f, 32}
        end

        model2 =
          apply(Axon, norm, [
            Axon.input("input", shape: {nil, 2, 2, 3}),
            [name: "norm", beta_initializer: :zeros]
          ])

        input = random({1, 2, 2, 3})

        assert {init_fn, _predict_fn} = Axon.build(model2)

        assert %ModelState{
                 data: %{
                   "norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}
                 }
               } =
                 init_fn.(input, ModelState.empty())

        assert Nx.shape(gamma) == {3}
        assert Nx.type(gamma) == {:f, 32}
        assert_equal(beta, zeros({3}))
        assert Nx.shape(mean) == {3}
        assert Nx.type(mean) == {:f, 32}
        assert Nx.shape(var) == {3}
        assert Nx.type(var) == {:f, 32}
      end
    end

    test "computes forward pass with default options" do
      for norm <- @normalization_with_stats_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"]])
          input1 = random({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.build(model1)

          assert %ModelState{
                   data: %{
                     "norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}
                   }
                 } =
                   params = init_fn.(input1, ModelState.empty())

          assert_equal(
            predict_fn.(params, input1),
            apply(Axon.Layers, norm, [input1, gamma, beta, mean, var])
          )
        end

        model2 = apply(Axon, norm, [Axon.input("input", shape: {nil, 3, 2, 2}), [name: "norm"]])
        input2 = random({1, 3, 2, 2}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2)

        assert %ModelState{
                 data: %{
                   "norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}
                 }
               } =
                 params = init_fn.(input2, ModelState.empty())

        assert_equal(
          predict_fn.(params, input2),
          apply(Axon.Layers, norm, [input2, gamma, beta, mean, var])
        )
      end
    end

    test "computes forward pass with custom options" do
      for norm <- @normalization_with_stats_layers do
        if norm != :instance_norm do
          opts1 = [channel_index: 1, epsilon: 1.0e-3]

          model1 =
            apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"] ++ opts1])

          input1 = random({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.build(model1)

          assert %ModelState{
                   data: %{
                     "norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}
                   }
                 } =
                   params = init_fn.(input1, ModelState.empty())

          assert_equal(
            predict_fn.(params, input1),
            apply(Axon.Layers, norm, [input1, gamma, beta, mean, var, opts1])
          )
        end

        opts2 = [channel_index: 3, epsilon: 1.0e-4]

        model2 =
          apply(Axon, norm, [Axon.input("input", shape: {nil, 2, 2, 3}), [name: "norm"] ++ opts2])

        input2 = random({1, 2, 2, 3}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2)

        assert %ModelState{
                 data: %{
                   "norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}
                 }
               } =
                 params = init_fn.(input2, ModelState.empty())

        assert_equal(
          predict_fn.(params, input2),
          apply(Axon.Layers, norm, [input2, gamma, beta, mean, var, opts2])
        )
      end
    end

    test "initializes with parameter policy" do
      for norm <- @normalization_with_stats_layers do
        model = apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(params: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 1, 2})

        assert {init_fn, _} = Axon.build(mp_model)

        assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                 init_fn.(input, ModelState.empty())

        assert Nx.type(gamma) == {:bf, 16}
        assert Nx.type(beta) == {:bf, 16}
      end
    end

    test "computes forward pass with output policy" do
      for norm <- @normalization_with_stats_layers do
        model = apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 1, 2})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
      end
    end
  end

  @normalization_layers [:layer_norm]

  describe "normalization" do
    test "initializes in default case" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"]])

          input = random({1, 2})

          assert {init_fn, _predict_fn} = Axon.build(model1)

          assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                   init_fn.(input, ModelState.empty())

          assert Nx.shape(gamma) == {2}
          assert Nx.type(gamma) == {:f, 32}
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2, 2, 3}), [name: "norm"]])

        input = random({1, 2, 2, 3})

        assert {init_fn, _predict_fn} = Axon.build(model2)

        assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                 init_fn.(input, ModelState.empty())

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
            apply(Axon, norm, [
              Axon.input("input", shape: {nil, 2}),
              [name: "norm", gamma_initializer: :zeros]
            ])

          input = random({1, 2})

          assert {init_fn, _predict_fn} = Axon.build(model1)

          assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                   init_fn.(input, ModelState.empty())

          assert_equal(gamma, zeros({2}))
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 =
          apply(Axon, norm, [
            Axon.input("input", shape: {nil, 2, 2, 3}),
            [name: "norm", beta_initializer: :zeros]
          ])

        input = random({1, 2, 2, 3})

        assert {init_fn, _predict_fn} = Axon.build(model2)

        assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                 init_fn.(input, ModelState.empty())

        assert Nx.shape(gamma) == {3}
        assert Nx.type(gamma) == {:f, 32}
        assert_equal(beta, zeros({3}))
      end
    end

    test "computes forward pass with default options" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"]])
          input1 = random({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.build(model1)

          assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                   params = init_fn.(input1, ModelState.empty())

          assert_equal(
            predict_fn.(params, input1),
            apply(Axon.Layers, norm, [input1, gamma, beta])
          )
        end

        model2 = apply(Axon, norm, [Axon.input("input", shape: {nil, 3, 2, 2}), [name: "norm"]])
        input2 = random({1, 3, 2, 2}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2)

        assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                 params = init_fn.(input2, ModelState.empty())

        assert_equal(predict_fn.(params, input2), apply(Axon.Layers, norm, [input2, gamma, beta]))
      end
    end

    test "computes forward pass with custom options" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          opts1 = [channel_index: 1, epsilon: 1.0e-3]

          model1 =
            apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"] ++ opts1])

          input1 = random({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.build(model1)

          assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                   params = init_fn.(input1, ModelState.empty())

          assert_equal(
            predict_fn.(params, input1),
            apply(Axon.Layers, norm, [input1, gamma, beta, opts1])
          )
        end

        opts2 = [channel_index: 3, epsilon: 1.0e-4]

        model2 =
          apply(Axon, norm, [Axon.input("input", shape: {nil, 2, 2, 3}), [name: "norm"] ++ opts2])

        input2 = random({1, 2, 2, 3}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2)

        assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                 params = init_fn.(input2, ModelState.empty())

        assert_equal(
          predict_fn.(params, input2),
          apply(Axon.Layers, norm, [input2, gamma, beta, opts2])
        )
      end
    end

    test "initializes with parameter policy" do
      for norm <- @normalization_layers do
        model = apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(params: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 1, 2})

        assert {init_fn, _} = Axon.build(mp_model)

        assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
                 init_fn.(input, ModelState.empty())

        assert Nx.type(gamma) == {:bf, 16}
        assert Nx.type(beta) == {:bf, 16}
      end
    end

    test "computes forward pass with output policy" do
      for norm <- @normalization_layers do
        model = apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = random({1, 1, 2})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
      end
    end
  end

  describe "group normalization" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 3}) |> Axon.group_norm(3, name: "norm")

      input = random({1, 3})

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.shape(gamma) == {3}
      assert Nx.type(gamma) == {:f, 32}
      assert Nx.shape(beta) == {3}
      assert Nx.type(beta) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input", shape: {nil, 3})
        |> Axon.group_norm(3, name: "norm", gamma_initializer: :zeros)

      input = random({1, 3})

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
               init_fn.(input, ModelState.empty())

      assert_equal(gamma, zeros({3}))
      assert Nx.shape(beta) == {3}
      assert Nx.type(beta) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 3, 3})
        |> Axon.group_norm(3, name: "norm", beta_initializer: :zeros)

      input = random({1, 3, 3})

      assert {init_fn, _predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
               init_fn.(input, ModelState.empty())

      assert_equal(beta, zeros({3}))
      assert Nx.shape(gamma) == {3}
      assert Nx.type(gamma) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 2}) |> Axon.group_norm(2, name: "norm")
      input1 = random({1, 2})

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
               params = init_fn.(input1, ModelState.empty())

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.group_norm(input1, gamma, beta, num_groups: 2)
      )

      model2 = Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.group_norm(3, name: "norm")
      input2 = random({1, 2, 2, 3})

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
               params = init_fn.(input2, ModelState.empty())

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.group_norm(input2, gamma, beta, num_groups: 3)
      )
    end

    test "computes forward pass with custom options" do
      opts = [epsilon: 1.0e-3, channel_index: 3]

      model =
        Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.group_norm(3, [name: "norm"] ++ opts)

      input = random({1, 2, 2, 3})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
               params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.group_norm(input, gamma, beta, [num_groups: 3] ++ opts)
      )
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.group_norm(1, name: "norm")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 2})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %ModelState{data: %{"norm" => %{"gamma" => gamma, "beta" => beta}}} =
               init_fn.(input, ModelState.empty())

      assert Nx.type(gamma) == {:bf, 16}
      assert Nx.type(beta) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.group_norm(1, name: "norm")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 3})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end
  end

  describe "flatten" do
    test "initializes with no params" do
      model = Axon.input("input_0", shape: {nil, 32}) |> Axon.flatten()

      input = random({1, 32})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input_0", shape: {nil, 32}) |> Axon.flatten()
      input1 = random({1, 32})

      assert {_, predict_fn} = Axon.build(model1)
      assert_equal(predict_fn.(ModelState.empty(), input1), Axon.Layers.flatten(input1))

      model2 = Axon.input("input", shape: {nil, 3, 32, 32}) |> Axon.flatten()
      input2 = random({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.build(model2)
      assert_equal(predict_fn.(ModelState.empty(), input2), Axon.Layers.flatten(input2))
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input_0", shape: {nil, 32}) |> Axon.flatten()
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 3})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end
  end

  describe "transpose" do
    test "initializes with no params" do
      model = Axon.input("input", shape: {nil, 3, 32}) |> Axon.transpose()

      input = random({1, 3, 32})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input_0", shape: {nil, 32}) |> Axon.transpose([0, 1])
      input1 = random({1, 32})

      assert {_, predict_fn} = Axon.build(model1)
      assert_equal(predict_fn.(ModelState.empty(), input1), Nx.transpose(input1, axes: [0, 1]))

      model2 = Axon.input("input", shape: {nil, 3, 32, 32}) |> Axon.transpose([0, 2, 1, 3])
      input2 = random({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.build(model2)

      assert_equal(
        predict_fn.(ModelState.empty(), input2),
        Nx.transpose(input2, axes: [0, 2, 1, 3])
      )
    end

    test "computes forward pass with constant" do
      model = Axon.constant(Nx.iota({1, 2, 3})) |> Axon.transpose([2, 1, 0])

      assert {_, predict_fn} = Axon.build(model)

      assert_equal(
        predict_fn.(ModelState.empty(), {}),
        Nx.transpose(Nx.iota({1, 2, 3}, type: {:f, 32}))
      )
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input_0", shape: {nil, 32}) |> Axon.transpose([0, 1])
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 32})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end
  end

  describe "reshape" do
    test "initializes with no params" do
      model = Axon.input("input", shape: {nil, 1, 32}) |> Axon.reshape({16, 2})

      input = random({1, 1, 32})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input_0", shape: {nil, 32}) |> Axon.reshape({16, 2})
      input1 = random({1, 32})

      assert {_, predict_fn} = Axon.build(model1)
      assert_equal(predict_fn.(ModelState.empty(), input1), Nx.reshape(input1, {1, 16, 2}))

      model2 = Axon.input("input", shape: {nil, 3, 32, 32}) |> Axon.reshape({3, 16, 2, 32})
      input2 = random({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.build(model2)
      assert_equal(predict_fn.(ModelState.empty(), input2), Nx.reshape(input2, {1, 3, 16, 2, 32}))
    end

    test "computes forward pass with constant input" do
      model = Axon.constant(Nx.iota({6})) |> Axon.reshape({1, 2, 3})

      assert {_, predict_fn} = Axon.build(model)

      assert_equal(
        predict_fn.(ModelState.empty(), {}),
        Nx.tensor([[[0, 1, 2], [3, 4, 5]]], type: {:f, 32})
      )
    end

    test "computes forward pass with magic :batch and :auto" do
      model = Axon.input("input") |> Axon.reshape({:batch, 3, :auto})

      assert {_, predict_fn} = Axon.build(model)

      input = random({2, 4, 6})
      assert_equal(predict_fn.(ModelState.empty(), input), Nx.reshape(input, {2, 3, 8}))
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input_0", shape: {nil, 32}) |> Axon.reshape({2, 16})
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 32})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end
  end

  describe "resize" do
    test "initializes with no params" do
      model = Axon.input("input", shape: {nil, 1, 3, 3}) |> Axon.resize({4, 4})

      input = random({1, 1, 3, 3})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 1, 3, 3}) |> Axon.resize({4, 4})
      input1 = random({1, 1, 3, 3})

      assert {_, predict_fn} = Axon.build(model1)

      assert_equal(
        predict_fn.(ModelState.empty(), input1),
        Axon.Layers.resize(input1, size: {4, 4})
      )
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 3, 3}) |> Axon.resize({4, 4})
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = random({1, 1, 3, 3})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, ModelState.empty()), input)) == {:bf, 16}
    end
  end

  describe "lstm" do
    test "initializes in default case" do
      model =
        Axon.input("input", shape: {nil, 32, 10})
        |> Axon.lstm(64, name: "lstm")
        |> Axon.container()

      input = random({1, 32, 10})

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "lstm" => %{
                   "input_kernel" => %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio},
                   "hidden_kernel" => %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who},
                   "bias" => %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}
                 }
               }
             } = init_fn.(input, ModelState.empty())

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
        Axon.input("input", shape: {nil, 32, 10})
        |> Axon.lstm(64, name: "lstm", kernel_initializer: :zeros)
        |> Axon.container()

      input = random({1, 32, 10})

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{
               data: %{
                 "lstm" => %{
                   "input_kernel" => %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio},
                   "hidden_kernel" => %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who},
                   "bias" => %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}
                 }
               }
             } = init_fn.(input, ModelState.empty())

      # Input kernel
      assert_equal(wii, zeros({10, 64}))
      assert_equal(wif, zeros({10, 64}))
      assert_equal(wig, zeros({10, 64}))
      assert_equal(wio, zeros({10, 64}))

      # Hidden kernel
      assert_equal(whi, zeros({64, 64}))
      assert_equal(whf, zeros({64, 64}))
      assert_equal(whg, zeros({64, 64}))
      assert_equal(who, zeros({64, 64}))

      # Bias
      assert Nx.shape(bi) == {64}
      assert Nx.type(bi) == {:f, 32}
      assert Nx.shape(bf) == {64}
      assert Nx.type(bf) == {:f, 32}
      assert Nx.shape(bg) == {64}
      assert Nx.type(bg) == {:f, 32}
      assert Nx.shape(bo) == {64}
      assert Nx.type(bo) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 32, 10})
        |> Axon.lstm(64, name: "lstm", bias_initializer: :zeros)
        |> Axon.container()

      assert {init_fn, _predict_fn} = Axon.build(model2)

      assert %ModelState{
               data: %{
                 "lstm" => %{
                   "input_kernel" => %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio},
                   "hidden_kernel" => %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who},
                   "bias" => %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}
                 }
               }
             } = init_fn.(input, ModelState.empty())

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
      assert_equal(bi, zeros({64}))
      assert_equal(bf, zeros({64}))
      assert_equal(bg, zeros({64}))
      assert_equal(bo, zeros({64}))
    end

    test "computes forward pass with default options" do
      model =
        Axon.input("input", shape: {nil, 8, 2})
        |> Axon.lstm(2, name: "lstm", recurrent_initializer: :zeros)
        |> Axon.container()

      input = random({1, 8, 2}, type: {:f, 32})

      init_carry = {zeros({1, 2}), zeros({1, 2})}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "lstm" => %{
                   "input_kernel" => %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio},
                   "hidden_kernel" => %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who},
                   "bias" => %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      k = %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio}
      h = %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who}
      b = %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(
          &Axon.Layers.lstm_cell/6,
          input,
          init_carry,
          Nx.tensor(0),
          k,
          h,
          b
        )
      )
    end

    test "computes forward pass with custom options" do
      model1 =
        Axon.input("input", shape: {nil, 8, 2})
        |> Axon.lstm(2,
          name: "lstm",
          recurrent_initializer: :zeros,
          gate: :relu,
          activation: :sigmoid
        )
        |> Axon.container()

      input1 = random({1, 8, 2}, type: {:f, 32})

      init_carry1 = {zeros({1, 2}), zeros({1, 2})}

      cell_fn1 = fn i, c, mask, k, h, b ->
        Axon.Layers.lstm_cell(
          i,
          c,
          mask,
          k,
          h,
          b,
          &Axon.Activations.relu/1,
          &Axon.Activations.sigmoid/1
        )
      end

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{
               data: %{
                 "lstm" => %{
                   "input_kernel" => %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio},
                   "hidden_kernel" => %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who},
                   "bias" => %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}
                 }
               }
             } = params = init_fn.(input1, ModelState.empty())

      k = %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio}
      h = %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who}
      b = %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}

      assert_all_close(
        predict_fn.(params, input1),
        Axon.Layers.dynamic_unroll(cell_fn1, input1, init_carry1, Nx.tensor(0), k, h, b)
      )

      model2 =
        Axon.input("input", shape: {nil, 8, 2})
        |> Axon.lstm(2, name: "lstm", unroll: :static, recurrent_initializer: :zeros)
        |> Axon.container()

      input2 = random({1, 8, 2}, type: {:f, 32})

      init_carry2 = {zeros({1, 2}), zeros({1, 2})}

      cell_fn2 = &Axon.Layers.lstm_cell/6

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{
               data: %{
                 "lstm" => %{
                   "input_kernel" => %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio},
                   "hidden_kernel" => %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who},
                   "bias" => %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}
                 }
               }
             } = params = init_fn.(input2, ModelState.empty())

      k = %{"wii" => wii, "wif" => wif, "wig" => wig, "wio" => wio}
      h = %{"whi" => whi, "whf" => whf, "whg" => whg, "who" => who}
      b = %{"bi" => bi, "bf" => bf, "bg" => bg, "bo" => bo}

      assert_all_close(
        predict_fn.(params, input2),
        Axon.Layers.static_unroll(cell_fn2, input2, init_carry2, Nx.tensor(0), k, h, b)
      )
    end

    test "computes forward pass with hidden state" do
      seq = Axon.input("input", shape: {nil, 8, 2})
      {_, carry} = seq |> Axon.lstm(2, name: "encode", recurrent_initializer: :zeros)
      model = Axon.lstm(seq, carry, 2, name: "decode") |> Axon.container()
      input = random({1, 8, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        init_carry = {zeros({1, 2}), zeros({1, 2})}

        {_, carry} =
          Axon.Layers.dynamic_unroll(
            &Axon.Layers.lstm_cell/6,
            inp,
            init_carry,
            Nx.tensor(0),
            ei,
            eh,
            eb
          )

        Axon.Layers.dynamic_unroll(&Axon.Layers.lstm_cell/6, inp, carry, Nx.tensor(0), di, dh, db)
      end

      assert %ModelState{
               data: %{
                 "encode" => %{
                   "input_kernel" => ek,
                   "hidden_kernel" => eh,
                   "bias" => eb
                 },
                 "decode" => %{
                   "input_kernel" => dk,
                   "hidden_kernel" => dh,
                   "bias" => db
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      enc = {ek, eh, eb}
      dec = {dk, dh, db}

      assert_equal(predict_fn.(params, input), equiv_fn.(input, enc, dec))
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.lstm(2, name: "lstm", use_bias: false)
        |> Axon.container()

      input = random({1, 2, 1})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "lstm" =>
                   %{
                     "input_kernel" => %{"wii" => _, "wif" => _, "wig" => _, "wio" => _},
                     "hidden_kernel" => %{"whi" => _, "whf" => _, "whg" => _, "who" => _}
                   } = lstm_params
               }
             } = init_fn.(input, ModelState.empty())

      assert Map.has_key?(lstm_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.lstm(2, name: "lstm", use_bias: false, recurrent_initializer: :zeros)
        |> Axon.container()

      input = random({1, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "lstm" => %{
                   "input_kernel" => k,
                   "hidden_kernel" => h
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      b = %{
        "bi" => Nx.tensor(0),
        "bf" => Nx.tensor(0),
        "bg" => Nx.tensor(0),
        "bo" => Nx.tensor(0)
      }

      c = {zeros({1, 2}), zeros({1, 2})}

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(&Axon.Layers.lstm_cell/6, input, c, Nx.tensor(0), k, h, b)
      )
    end

    test "mask actually works" do
      sequence = Axon.input("review")
      mask = Axon.mask(sequence, 0)
      embedded = sequence |> Axon.embedding(2048, 64)
      {rnn_sequence, _state} = Axon.lstm(embedded, 64, mask: mask)

      {init_fn, predict_fn} = Axon.build(rnn_sequence)
      params = init_fn.(Nx.template({64, 64}, :s64), ModelState.empty())

      input = Nx.tensor([[1, 2, 3, 4]])
      padded = Nx.pad(input, 0, [{0, 0, 0}, {0, 60, 0}])
      out = predict_fn.(params, padded)

      last_token = out[[.., 3, ..]]

      for i <- 4..63 do
        # all eos tokens will be ignored so we just propagate the value
        # to the next token and thus these should all be the same as the
        # last non eos token
        assert_equal(last_token, out[[.., i, ..]])
      end
    end
  end

  describe "conv_lstm" do
    test "initializes in default case" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        in_channel_n = 3,
        _width = 6,
        _heigth = 6
      }

      out_channel_n = 4

      model =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm")
        |> Axon.container()

      input = random({1, 10, 3, 6, 6})

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "convlstm" => %{
                   "input_kernel" => wi,
                   "hidden_kernel" => wh,
                   "bias" => b
                 }
               }
             } = init_fn.(input, ModelState.empty())

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

      input = random({1, 10, 3, 6, 6})

      out_channel_n = 4

      model1 =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", kernel_initializer: :zeros)
        |> Axon.container()

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %ModelState{
               data: %{
                 "convlstm" => %{
                   "input_kernel" => wi,
                   "hidden_kernel" => wh,
                   "bias" => b
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert_equal(wi, zeros({4 * out_channel_n, in_channel_n, 1, 1}))
      assert_equal(wh, zeros({4 * out_channel_n, out_channel_n, 1, 1}))

      # Bias
      assert Nx.shape(b) == {4 * out_channel_n}
      assert Nx.type(b) == {:f, 32}

      model2 =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", bias_initializer: :zeros)
        |> Axon.container()

      assert {init_fn, _predict_fn} = Axon.build(model2)

      assert %ModelState{
               data: %{
                 "convlstm" => %{
                   "input_kernel" => wi,
                   "hidden_kernel" => wh,
                   "bias" => b
                 }
               }
             } = init_fn.(input, ModelState.empty())

      # Input kernel
      assert Nx.shape(wi) == {4 * out_channel_n, in_channel_n, 1, 1}
      assert Nx.type(wi) == {:f, 32}

      # Hidden kernel
      assert Nx.shape(wh) == {4 * out_channel_n, out_channel_n, 1, 1}
      assert Nx.type(wh) == {:f, 32}

      # Bias
      assert_equal(b, zeros({4 * out_channel_n}))
    end

    test "computes forward pass with dynamic unroll and equal number of input and output channels" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        height = 6
      }

      out_channel_n = 4
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, height}

      model =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", recurrent_initializer: :zeros)
        |> Axon.container()

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> random(type: {:f, 32})

      init_carry = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "convlstm" => %{
                   "input_kernel" => wi,
                   "hidden_kernel" => wh,
                   "bias" => b
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(
          &Axon.Layers.conv_lstm_cell/6,
          input,
          init_carry,
          Nx.tensor(0),
          wi,
          wh,
          b
        )
      )
    end

    test "computes forward pass with static unroll and different number of input and output channels" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        height = 6
      }

      out_channel_n = 7
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, height}

      model =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n,
          name: "convlstm",
          recurrent_initializer: :zeros,
          unroll: :static
        )
        |> Axon.container()

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> random(type: {:f, 32})

      init_carry = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "convlstm" => %{
                   "input_kernel" => wi,
                   "hidden_kernel" => wh,
                   "bias" => b
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.static_unroll(
          &Axon.Layers.conv_lstm_cell/6,
          input,
          init_carry,
          Nx.tensor(0),
          wi,
          wh,
          b
        )
      )
    end

    # First part fails by conv_lstm_cell:
    # no support for custom gate and activation functions
    test "computes forward pass with custom options" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        height = 6
      }

      out_channel_n = 3
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, height}

      model1 =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", recurrent_initializer: :zeros)
        |> Axon.container()

      input1 =
        input_shape
        |> put_elem(0, batch_real)
        |> random(type: {:f, 32})

      init_carry1 = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      cell_fn1 = fn i, c, mask, k, h, b ->
        Axon.Layers.conv_lstm_cell(
          i,
          c,
          mask,
          k,
          h,
          b
        )
      end

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{
               data: %{
                 "convlstm" => %{
                   "input_kernel" => wi,
                   "hidden_kernel" => wh,
                   "bias" => b
                 }
               }
             } = params = init_fn.(input1, ModelState.empty())

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.dynamic_unroll(cell_fn1, input1, init_carry1, Nx.tensor(0), wi, wh, b)
      )

      model2 =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n,
          name: "convlstm",
          unroll: :static,
          recurrent_initializer: :zeros
        )
        |> Axon.container()

      input2 =
        input_shape
        |> put_elem(0, batch_real)
        |> random(type: {:f, 32})

      init_carry2 = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      cell_fn2 = &Axon.Layers.conv_lstm_cell/6

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{
               data: %{
                 "convlstm" => %{
                   "input_kernel" => wi,
                   "hidden_kernel" => wh,
                   "bias" => b
                 }
               }
             } = params = init_fn.(input2, ModelState.empty())

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.static_unroll(cell_fn2, input2, init_carry2, Nx.tensor(0), wi, wh, b)
      )
    end

    test "computes forward pass with hidden state" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        height = 6
      }

      out_channel_n = 3
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, height}
      seq = Axon.input("input", shape: input_shape)

      {_, carry} =
        seq
        |> Axon.conv_lstm(out_channel_n, name: "encode", recurrent_initializer: :zeros)

      model =
        Axon.conv_lstm(seq, carry, out_channel_n, name: "decode")
        |> Axon.container()

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> random(type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model)

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        init_carry = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

        {_, carry} =
          Axon.Layers.dynamic_unroll(
            &Axon.Layers.conv_lstm_cell/6,
            inp,
            init_carry,
            Nx.tensor(0),
            ei,
            eh,
            eb
          )

        Axon.Layers.dynamic_unroll(
          &Axon.Layers.conv_lstm_cell/6,
          inp,
          carry,
          Nx.tensor(0),
          di,
          dh,
          db
        )
      end

      assert %ModelState{
               data: %{
                 "encode" => %{
                   "input_kernel" => ei,
                   "hidden_kernel" => eh,
                   "bias" => eb
                 },
                 "decode" => %{
                   "input_kernel" => di,
                   "hidden_kernel" => dh,
                   "bias" => db
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      enc = {ei, eh, eb}
      dec = {di, dh, db}

      assert_equal(predict_fn.(params, input), equiv_fn.(input, enc, dec))
    end

    test "computes forward pass with use_bias false" do
      input_shape = {
        _batch = nil,
        _sequence_length = 10,
        _in_channel_n = 3,
        width = 6,
        height = 6
      }

      out_channel_n = 3
      batch_real = 1
      hidden_shape_real = {batch_real, 1, out_channel_n, width, height}

      model =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n,
          name: "convlstm",
          use_bias: false,
          recurrent_initializer: :zeros
        )
        |> Axon.container()

      input =
        input_shape
        |> put_elem(0, batch_real)
        |> random(type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "convlstm" => %{
                   "input_kernel" => k,
                   "hidden_kernel" => h
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      b = Nx.broadcast(0, 4 * out_channel_n)

      c = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(&Axon.Layers.conv_lstm_cell/6, input, c, Nx.tensor(0), k, h, b)
      )
    end
  end

  describe "gru" do
    test "initializes in default case" do
      model =
        Axon.input("input", shape: {nil, 32, 10}) |> Axon.gru(64, name: "gru") |> Axon.container()

      input = random({1, 32, 10})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "gru" => %{
                   "input_kernel" => %{"wir" => wir, "wiz" => wiz, "win" => win},
                   "hidden_kernel" => %{"whr" => whr, "whz" => whz, "whn" => whn},
                   "bias" => %{"br" => br, "bz" => bz, "bhn" => bhn, "bin" => bin}
                 }
               }
             } = init_fn.(input, ModelState.empty())

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
      input = random({1, 32, 10})

      model1 =
        Axon.input("input", shape: {nil, 32, 10})
        |> Axon.gru(64, name: "gru", kernel_initializer: :zeros)
        |> Axon.container()

      assert {init_fn, _} = Axon.build(model1)

      assert %ModelState{
               data: %{
                 "gru" => %{
                   "input_kernel" => %{"wir" => wir, "wiz" => wiz, "win" => win},
                   "hidden_kernel" => %{"whr" => whr, "whz" => whz, "whn" => whn},
                   "bias" => %{"br" => br, "bz" => bz, "bhn" => bhn, "bin" => bin}
                 }
               }
             } = init_fn.(input, ModelState.empty())

      assert_equal(wir, zeros({10, 64}))
      assert_equal(wiz, zeros({10, 64}))
      assert_equal(win, zeros({10, 64}))
      assert_equal(whr, zeros({64, 64}))
      assert_equal(whz, zeros({64, 64}))
      assert_equal(whn, zeros({64, 64}))
      assert Nx.shape(br) == {64}
      assert Nx.type(br) == {:f, 32}
      assert Nx.shape(bz) == {64}
      assert Nx.type(bz) == {:f, 32}
      assert Nx.shape(bhn) == {64}
      assert Nx.type(bhn) == {:f, 32}
      assert Nx.shape(bin) == {64}
      assert Nx.type(bin) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 32, 10})
        |> Axon.gru(64, name: "gru", bias_initializer: :zeros)
        |> Axon.container()

      assert {init_fn, _} = Axon.build(model2)

      assert %ModelState{
               data: %{
                 "gru" => %{
                   "input_kernel" => %{"wir" => wir, "wiz" => wiz, "win" => win},
                   "hidden_kernel" => %{"whr" => whr, "whz" => whz, "whn" => whn},
                   "bias" => %{"br" => br, "bz" => bz, "bhn" => bhn, "bin" => bin}
                 }
               }
             } = init_fn.(input, ModelState.empty())

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
      assert_equal(br, zeros({64}))
      assert_equal(bz, zeros({64}))
      assert_equal(bhn, zeros({64}))
      assert_equal(bin, zeros({64}))
    end

    test "computes forward pass with default options" do
      model =
        Axon.input("input", shape: {nil, 8, 2})
        |> Axon.gru(2, name: "gru", recurrent_initializer: :zeros)
        |> Axon.container()

      input = random({1, 8, 2})
      carry = {zeros({1, 2})}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "gru" => %{
                   "input_kernel" => k,
                   "hidden_kernel" => h,
                   "bias" => b
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(&Axon.Layers.gru_cell/6, input, carry, Nx.tensor(0), k, h, b)
      )
    end

    test "computes forward pass with custom options" do
      model1 =
        Axon.input("input", shape: {nil, 8, 2})
        |> Axon.gru(2,
          name: "gru",
          recurrent_initializer: :zeros,
          gate: :relu,
          activation: :sigmoid
        )
        |> Axon.container()

      input1 = random({1, 8, 2})
      carry1 = {zeros({1, 2})}

      cell_fn1 = fn i, c, mask, k, h, b ->
        Axon.Layers.gru_cell(
          i,
          c,
          mask,
          k,
          h,
          b,
          &Axon.Activations.relu/1,
          &Axon.Activations.sigmoid/1
        )
      end

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %ModelState{
               data: %{
                 "gru" => %{
                   "input_kernel" => %{"wir" => wir, "wiz" => wiz, "win" => win},
                   "hidden_kernel" => %{"whr" => whr, "whz" => whz, "whn" => whn},
                   "bias" => %{"br" => br, "bz" => bz, "bhn" => bhn, "bin" => bin}
                 }
               }
             } = params = init_fn.(input1, ModelState.empty())

      k = %{"wir" => wir, "wiz" => wiz, "win" => win}
      h = %{"whr" => whr, "whz" => whz, "whn" => whn}
      b = %{"br" => br, "bz" => bz, "bhn" => bhn, "bin" => bin}

      assert_all_close(
        predict_fn.(params, input1),
        Axon.Layers.dynamic_unroll(cell_fn1, input1, carry1, Nx.tensor(0), k, h, b)
      )

      model2 =
        Axon.input("input", shape: {nil, 8, 2})
        |> Axon.gru(2, name: "gru", recurrent_initializer: :zeros, unroll: :static)
        |> Axon.container()

      input2 = random({1, 8, 2})
      carry2 = {zeros({1, 2})}

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %ModelState{
               data: %{
                 "gru" => %{
                   "input_kernel" => %{"wir" => wir, "wiz" => wiz, "win" => win},
                   "hidden_kernel" => %{"whr" => whr, "whz" => whz, "whn" => whn},
                   "bias" => %{"br" => br, "bz" => bz, "bhn" => bhn, "bin" => bin}
                 }
               }
             } = params = init_fn.(input2, ModelState.empty())

      k = %{"wir" => wir, "wiz" => wiz, "win" => win}
      h = %{"whr" => whr, "whz" => whz, "whn" => whn}
      b = %{"br" => br, "bz" => bz, "bhn" => bhn, "bin" => bin}

      assert_all_close(
        predict_fn.(params, input2),
        Axon.Layers.static_unroll(&Axon.Layers.gru_cell/6, input2, carry2, Nx.tensor(0), k, h, b)
      )
    end

    test "computes forward pass with hidden state" do
      seq = Axon.input("input", shape: {nil, 8, 2})
      {_, carry} = Axon.gru(seq, 2, name: "encode", recurrent_initializer: :zeros)
      model = Axon.gru(seq, carry, 2, name: "decode") |> Axon.container()

      input = random({1, 8, 2})
      carry = {zeros({1, 2})}

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        {_, carry} =
          Axon.Layers.dynamic_unroll(
            &Axon.Layers.gru_cell/6,
            inp,
            carry,
            Nx.tensor(0),
            ei,
            eh,
            eb
          )

        Axon.Layers.dynamic_unroll(&Axon.Layers.gru_cell/6, inp, carry, Nx.tensor(0), di, dh, db)
      end

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "encode" => %{
                   "input_kernel" => eik,
                   "hidden_kernel" => ehk,
                   "bias" => eb
                 },
                 "decode" => %{
                   "input_kernel" => dik,
                   "hidden_kernel" => dhk,
                   "bias" => db
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      enc = {eik, ehk, eb}
      dec = {dik, dhk, db}

      assert_equal(predict_fn.(params, input), equiv_fn.(input, enc, dec))
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.gru(2, name: "gru", use_bias: false)
        |> Axon.container()

      input = random({1, 2, 1})

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "gru" =>
                   %{
                     "input_kernel" => %{"wir" => _, "wiz" => _, "win" => _},
                     "hidden_kernel" => %{"whr" => _, "whz" => _, "whn" => _}
                   } = gru_params
               }
             } = init_fn.(input, ModelState.empty())

      assert Map.has_key?(gru_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.gru(2, name: "gru", use_bias: false, recurrent_initializer: :zeros)
        |> Axon.container()

      input = random({1, 2, 1})
      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "gru" => %{
                   "input_kernel" => k,
                   "hidden_kernel" => h
                 }
               }
             } = params = init_fn.(input, ModelState.empty())

      b = %{
        "br" => Nx.tensor(0),
        "bz" => Nx.tensor(0),
        "bin" => Nx.tensor(0),
        "bhn" => Nx.tensor(0)
      }

      c = {zeros({1, 2})}

      assert_all_close(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(&Axon.Layers.gru_cell/6, input, c, Nx.tensor(0), k, h, b)
      )
    end

    test "mask actually works" do
      sequence = Axon.input("review")
      mask = Axon.mask(sequence, 0)
      embedded = sequence |> Axon.embedding(2048, 64)
      {rnn_sequence, _state} = Axon.gru(embedded, 64, mask: mask)

      {init_fn, predict_fn} = Axon.build(rnn_sequence)
      params = init_fn.(Nx.template({64, 64}, :s64), ModelState.empty())

      input = Nx.tensor([[1, 2, 3, 4]])
      padded = Nx.pad(input, 0, [{0, 0, 0}, {0, 60, 0}])
      out = predict_fn.(params, padded)

      last_token = out[[.., 3, ..]]

      for i <- 4..63 do
        # all eos tokens will be ignored so we just propagate the value
        # to the next token and thus these should all be the same as the
        # last non eos token
        assert_equal(last_token, out[[.., i, ..]])
      end
    end
  end

  @binary_layers [:add, :subtract, :multiply]

  describe "binary operations" do
    test "initializes with no params" do
      for op <- @binary_layers do
        model =
          apply(Axon, op, [
            Axon.input("input_0", shape: {nil, 32}),
            Axon.input("input_1", shape: {nil, 32})
          ])

        input = %{
          "input_0" => random({1, 32}),
          "input_1" => random({1, 32})
        }

        assert {init_fn, _} = Axon.build(model)
        assert ModelState.empty() == init_fn.(input, ModelState.empty())
      end
    end

    test "computes forward pass with default options" do
      for op <- @binary_layers do
        model1 =
          apply(Axon, op, [
            Axon.input("input_0", shape: {nil, 32}),
            Axon.input("input_1", shape: {nil, 32})
          ])

        input1_1 = random({1, 32})
        input1_2 = random({1, 32})
        assert {_, predict_fn} = Axon.build(model1)

        assert_all_close(
          predict_fn.(ModelState.empty(), %{"input_0" => input1_1, "input_1" => input1_2}),
          apply(Nx, op, [input1_1, input1_2])
        )

        model2 =
          apply(Axon, op, [
            [
              Axon.input("input_0", shape: {nil, 32}),
              Axon.input("input_1", shape: {nil, 32}),
              Axon.input("input_2", shape: {nil, 32})
            ]
          ])

        input2_1 = random({1, 32})
        input2_2 = random({1, 32})
        input2_3 = random({1, 32})
        assert {_, predict_fn} = Axon.build(model2)

        assert_all_close(
          predict_fn.(ModelState.empty(), %{
            "input_0" => input2_1,
            "input_1" => input2_2,
            "input_2" => input2_3
          }),
          apply(Nx, op, [apply(Nx, op, [input2_1, input2_2]), input2_3])
        )
      end
    end

    test "computes forward pass with output policy" do
      for op <- @binary_layers do
        model =
          apply(Axon, op, [
            Axon.input("input_0", shape: {nil, 32}),
            Axon.input("input_1", shape: {nil, 32})
          ])

        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = %{
          "input_0" => random({1, 32}),
          "input_1" => random({1, 32})
        }

        assert {_, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(ModelState.empty(), input)) == {:bf, 16}
      end
    end

    test "computes forward pass with broadcasting" do
      inp1 = random({1, 1})
      inp2 = random({1, 2})

      for op <- @binary_layers do
        model =
          apply(Axon, op, [
            Axon.input("input_0", shape: {nil, 1}),
            Axon.input("input_1", shape: {nil, 2})
          ])

        assert {_, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(ModelState.empty(), %{"input_0" => inp1, "input_1" => inp2}),
          apply(Nx, op, [inp1, inp2])
        )
      end
    end

    test "raises on bad shapes" do
      for op <- @binary_layers do
        assert_raise Axon.CompileError, ~r/cannot broadcast tensor/, fn ->
          inp1 = random({1, 32})
          inp2 = random({1, 64})

          model =
            apply(Axon, op, [
              [Axon.input("input_0", shape: {nil, 32}), Axon.input("input_1", shape: {nil, 64})]
            ])

          Axon.predict(model, ModelState.empty(), %{"input_0" => inp1, "input_1" => inp2})
        end
      end
    end
  end

  describe "concatenate" do
    test "initializes with no params" do
      model =
        Axon.concatenate(
          Axon.input("input_0", shape: {nil, 32}),
          Axon.input("input_1", shape: {nil, 32})
        )

      input = %{"input_0" => random({1, 32}), "input_1" => random({1, 32})}

      assert {init_fn, _} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
    end

    test "computes forward pass with default options" do
      model1 =
        Axon.concatenate(
          Axon.input("input_0", shape: {nil, 32}),
          Axon.input("input_1", shape: {nil, 32})
        )

      input1_1 = random({1, 32})
      input1_2 = random({1, 32})

      assert {_, predict_fn} = Axon.build(model1)

      assert_equal(
        predict_fn.(ModelState.empty(), %{"input_0" => input1_1, "input_1" => input1_2}),
        Nx.concatenate([input1_1, input1_2], axis: 1)
      )

      model2 =
        Axon.concatenate([
          Axon.input("input_0", shape: {nil, 32}),
          Axon.input("input_1", shape: {nil, 32}),
          Axon.input("input_2", shape: {nil, 32})
        ])

      input2_1 = random({1, 32})
      input2_2 = random({1, 32})
      input2_3 = random({1, 32})

      assert {_, predict_fn} = Axon.build(model2)

      assert_equal(
        predict_fn.(ModelState.empty(), %{
          "input_0" => input2_1,
          "input_1" => input2_2,
          "input_2" => input2_3
        }),
        Nx.concatenate([input2_1, input2_2, input2_3], axis: 1)
      )
    end

    test "computes forward pass with custom options" do
      model1 =
        Axon.concatenate(
          Axon.input("input_0", shape: {nil, 1, 32}),
          Axon.input("input_1", shape: {nil, 1, 32}),
          axis: 1
        )

      input1_1 = random({1, 1, 32})
      input1_2 = random({1, 1, 32})

      assert {_, predict_fn} = Axon.build(model1)

      assert_equal(
        predict_fn.(ModelState.empty(), %{"input_0" => input1_1, "input_1" => input1_2}),
        Nx.concatenate([input1_1, input1_2], axis: 1)
      )
    end

    test "computes forward pass with output policy" do
      model1 =
        Axon.concatenate(
          Axon.input("input_0", shape: {nil, 1, 32}),
          Axon.input("input_1", shape: {nil, 1, 32}),
          axis: 1
        )

      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model1, policy)
      input1_1 = random({1, 1, 32})
      input1_2 = random({1, 1, 32})

      assert {_, predict_fn} = Axon.build(mp_model)

      assert Nx.type(
               predict_fn.(ModelState.empty(), %{"input_0" => input1_1, "input_1" => input1_2})
             ) ==
               {:bf, 16}
    end
  end

  describe "pad" do
    test "initializes with no params" do
      model = Axon.input("input", shape: {nil, 3, 3}) |> Axon.pad([{1, 0}])
      input = random({1, 3, 3})

      assert {init_fn, _} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 3, 3}) |> Axon.pad([{1, 0}])
      input1 = random({1, 3, 3})

      assert {_, predict_fn} = Axon.build(model1)

      assert_equal(
        predict_fn.(ModelState.empty(), input1),
        Nx.pad(input1, 0, [{0, 0, 0}, {1, 0, 0}, {0, 0, 0}])
      )

      model2 = Axon.input("input", shape: {nil, 3, 3, 3}) |> Axon.pad([{0, 1}, {0, 1}])
      input2 = random({1, 3, 3, 3})

      assert {_, predict_fn} = Axon.build(model2)

      assert_equal(
        predict_fn.(ModelState.empty(), input2),
        Nx.pad(input2, 0, [{0, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 0}])
      )

      model3 = Axon.input("input", shape: {nil, 3, 3, 3, 3}) |> Axon.pad([{0, 1}, {0, 1}, {1, 0}])
      input3 = random({1, 3, 3, 3, 3})

      assert {_, predict_fn} = Axon.build(model3)

      assert_equal(
        predict_fn.(ModelState.empty(), input3),
        Nx.pad(input3, 0, [{0, 0, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 0}])
      )
    end

    test "computes forward pass with custom options" do
      model = Axon.input("input", shape: {nil, 3, 3}) |> Axon.pad([{1, 0}], 2)
      input = random({1, 3, 3})

      assert {_, predict_fn} = Axon.build(model)

      assert_equal(
        predict_fn.(ModelState.empty(), input),
        Nx.pad(input, 2, [{0, 0, 0}, {1, 0, 0}, {0, 0, 0}])
      )
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 3, 3}) |> Axon.pad([{1, 0}])
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)
      input = random({1, 3, 3})

      assert {_, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(ModelState.empty(), input)) == {:bf, 16}
    end
  end

  describe "nx" do
    test "computes special nx functions" do
      model = Axon.input("input", shape: {nil, 10}) |> Axon.nx(&Nx.sin/1)
      input = random({1, 10})

      assert {_, predict_fn} = Axon.build(model)
      assert_all_close(predict_fn.(ModelState.empty(), input), Nx.sin(input))
    end
  end

  describe "cond" do
    test "initializes with no params" do
      inp = Axon.input("input_0", shape: {nil, 1})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.all(x) end
      model = Axon.cond(inp, cond_fn, on_true, on_false)

      input = Nx.tensor([[1.0]])

      assert {init_fn, _} = Axon.build(model)
      assert ModelState.empty() == init_fn.(input, ModelState.empty())
    end

    test "computes forward pass with default options" do
      inp = Axon.input("input", shape: {nil, 2})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.all(x) end

      input_1 = Nx.tensor([[1.0, 1.0]])
      input_2 = Nx.tensor([[0.0, 0.0]])

      model = Axon.cond(inp, cond_fn, on_true, on_false)

      assert {_, predict_fn} = Axon.build(model)
      assert_equal(predict_fn.(ModelState.empty(), input_1), Axon.Activations.relu(input_1))
      assert_equal(predict_fn.(ModelState.empty(), input_2), Axon.Activations.sigmoid(input_2))
    end

    test "computes forward pass with output policy" do
      inp = Axon.input("input", shape: {nil, 1, 32})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.all(x) end
      model1 = Axon.cond(inp, cond_fn, on_true, on_false)
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model1, policy)

      input1_1 = random({1, 1, 32})

      assert {_, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(ModelState.empty(), input1_1)) == {:bf, 16}
    end

    test "raises on bad condition" do
      inp = Axon.input("input", shape: {nil, 1, 10})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.equal(x, 1) end

      model = Axon.cond(inp, cond_fn, on_true, on_false)

      assert_raise Axon.CompileError, ~r/cond_fn must return a scalar/, fn ->
        {_, predict_fn} = Axon.build(model)
        predict_fn.(ModelState.empty(), random({1, 1, 10}))
      end
    end
  end

  describe "split" do
    test "initializes with no parameters" do
      model = Axon.input("input", shape: {nil, 10}) |> Axon.split(5) |> Axon.container()

      input = random({1, 10})

      assert {init_fn, _} = Axon.build(model)
      assert init_fn.(input, ModelState.empty()) == ModelState.empty()
    end

    test "computes forward pass with default options" do
      model = Axon.input("input", shape: {nil, 10}) |> Axon.split(5) |> Axon.container()
      input = Nx.iota({1, 10}, type: {:f, 32})

      assert {_, predict_fn} = Axon.build(model)

      assert_equal(
        predict_fn.(ModelState.empty(), input),
        {
          Nx.tensor([[0.0, 1.0]]),
          Nx.tensor([[2.0, 3.0]]),
          Nx.tensor([[4.0, 5.0]]),
          Nx.tensor([[6.0, 7.0]]),
          Nx.tensor([[8.0, 9.0]])
        }
      )
    end
  end

  describe "hooks" do
    test "initialize hook", config do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(1, kernel_initializer: :ones)
        |> Axon.attach_hook(fn x -> send(config.test, x) end, on: :initialize)

      {init_fn, _} = Axon.build(model)
      init_fn.(Nx.tensor([[1.0]]), ModelState.empty())
      assert_receive %{"kernel" => kernel, "bias" => bias}
      assert_equal(kernel, Nx.tensor([[1.0]]))
      assert_equal(bias, Nx.tensor([0.0]))
    end

    test "pre forward hook", config do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.relu()
        |> Axon.attach_hook(fn x -> send(config.test, {x, :from_relu}) end, on: :pre_forward)

      inp = Nx.tensor([[1.0]])

      Axon.predict(model, ModelState.empty(), inp)

      assert_receive {pre_relu, :from_relu}
      assert_equal(pre_relu, inp)
    end

    test "forward hook", config do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.attach_hook(fn x -> send(config.test, {x, :from_input}) end, on: :forward)
        |> Axon.relu()

      inp = Nx.tensor([[1.0]])

      Axon.predict(model, ModelState.empty(), inp)

      assert_receive {from_inp, :from_input}
      assert_equal(from_inp, inp)
    end

    test "backward hook", config do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(10)
        |> Axon.attach_hook(fn x -> send(config.test, {x, :from_dense}) end, on: :backward)
        |> Axon.relu()
        |> Axon.attach_hook(fn x -> send(config.test, {x, :from_relu}) end, on: :backward)
        |> Axon.sigmoid()
        |> Axon.attach_hook(fn x -> send(config.test, {x, :from_sigmoid}) end, on: :backward)

      {init_fn, predict_fn} = Axon.build(model)

      inp = random({1, 1})
      %ModelState{} = params = init_fn.(inp, ModelState.empty())

      axon_loss = fn inp, params -> Nx.sum(predict_fn.(params, inp)) end

      loss = fn inp, %{data: params} ->
        inp
        |> Axon.Layers.dense(params["dense_0"]["kernel"], params["dense_0"]["bias"])
        |> Axon.Activations.relu()
        |> Axon.Activations.sigmoid()
        |> Nx.sum()
      end

      %{data: axon_grad_params} =
        Nx.Defn.jit(fn inp, x -> Nx.Defn.grad(x, &axon_loss.(inp, &1)) end).(inp, params)

      %{data: actual_grad_params} =
        Nx.Defn.jit(fn inp, x -> Nx.Defn.grad(x, &loss.(inp, &1)) end).(inp, params)

      assert_all_close(
        axon_grad_params["dense_0"]["kernel"],
        actual_grad_params["dense_0"]["kernel"]
      )

      assert_all_close(
        axon_grad_params["dense_0"]["bias"],
        actual_grad_params["dense_0"]["bias"]
      )

      assert_receive {%Nx.Tensor{}, :from_dense}
      assert_receive {%Nx.Tensor{}, :from_relu}
      assert_receive {%Nx.Tensor{}, :from_sigmoid}
    end
  end

  describe "integrated models" do
    test "basic feed forward model initializes correctly" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(8)
        |> Axon.dense(1)

      {init_fn, _} = Axon.build(model)

      inp = random({1, 2})

      assert %ModelState{data: %{"dense_0" => dense_0_params, "dense_1" => dense_1_params}} =
               init_fn.(inp, ModelState.empty())

      assert %{"kernel" => k0, "bias" => b0} = dense_0_params
      assert %{"kernel" => k1, "bias" => b1} = dense_1_params
      assert Nx.shape(k0) == {2, 8}
      assert Nx.shape(b0) == {8}
      assert Nx.shape(k1) == {8, 1}
      assert Nx.shape(b1) == {1}
    end

    test "recurrent model initializes correctly" do
      input = Axon.input("input", shape: {nil, 8, 2})

      {_, state} = input |> Axon.lstm(8)
      {out, _} = input |> Axon.lstm(state, 8)

      {init_fn, _} = Axon.build(out)

      inp = Nx.template({1, 8, 2}, {:f, 32})

      assert %ModelState{data: %{"lstm_0" => lstm_0_params, "lstm_1" => lstm_1_params}} =
               init_fn.(inp, ModelState.empty())

      assert %{
               "input_kernel" => %{"wii" => wii_0, "wif" => wif_0, "wig" => wig_0, "wio" => wio_0},
               "hidden_kernel" => %{
                 "whi" => whi_0,
                 "whf" => whf_0,
                 "whg" => whg_0,
                 "who" => who_0
               },
               "bias" => %{"bi" => bi_0, "bf" => bf_0, "bg" => bg_0, "bo" => bo_0}
             } = lstm_0_params

      assert Nx.shape(wii_0) == {2, 8}
      assert Nx.shape(wif_0) == {2, 8}
      assert Nx.shape(wig_0) == {2, 8}
      assert Nx.shape(wio_0) == {2, 8}
      assert Nx.shape(whi_0) == {8, 8}
      assert Nx.shape(whf_0) == {8, 8}
      assert Nx.shape(whg_0) == {8, 8}
      assert Nx.shape(who_0) == {8, 8}
      assert Nx.shape(bi_0) == {8}
      assert Nx.shape(bf_0) == {8}
      assert Nx.shape(bg_0) == {8}
      assert Nx.shape(bo_0) == {8}

      assert %{
               "input_kernel" => %{"wii" => wii_1, "wif" => wif_1, "wig" => wig_1, "wio" => wio_1},
               "hidden_kernel" => %{
                 "whi" => whi_1,
                 "whf" => whf_1,
                 "whg" => whg_1,
                 "who" => who_1
               },
               "bias" => %{"bi" => bi_1, "bf" => bf_1, "bg" => bg_1, "bo" => bo_1}
             } = lstm_1_params

      assert Nx.shape(wii_1) == {2, 8}
      assert Nx.shape(wif_1) == {2, 8}
      assert Nx.shape(wig_1) == {2, 8}
      assert Nx.shape(wio_1) == {2, 8}
      assert Nx.shape(whi_1) == {8, 8}
      assert Nx.shape(whf_1) == {8, 8}
      assert Nx.shape(whg_1) == {8, 8}
      assert Nx.shape(who_1) == {8, 8}
      assert Nx.shape(bi_1) == {8}
      assert Nx.shape(bf_1) == {8}
      assert Nx.shape(bg_1) == {8}
      assert Nx.shape(bo_1) == {8}
    end
  end

  describe "custom layers" do
    test "initializes with no parameters" do
      model = Axon.layer(fn x, _opts -> x end, [Axon.input("input_0", shape: {nil, 1})])
      inp = random({1, 1})

      {init_fn, _} = Axon.build(model)
      assert ModelState.empty() == init_fn.(inp, ModelState.empty())
    end

    test "initializes with parameters" do
      kernel_param = Axon.param("kernel", fn shape -> shape end)

      model =
        Axon.layer(
          fn x, _kernel, _opts -> x end,
          [Axon.input("input_0", shape: {nil, 1}), kernel_param],
          name: "layer_0"
        )

      {init_fn, _} = Axon.build(model)

      inp = random({1, 1})

      assert %ModelState{data: %{"layer_0" => %{"kernel" => kernel}}} =
               init_fn.(inp, ModelState.empty())

      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(kernel) == {1, 1}
    end

    test "initializes with two custom layers and no op_name" do
      k1 = Axon.param("kernel", fn shape -> shape end)
      k2 = Axon.param("kernel", fn shape -> shape end)

      layer = fn x, k -> Axon.layer(fn y, _kernel, _opts -> y end, [x, k]) end

      input = Axon.input("input_0", shape: {nil, 1})

      model = input |> layer.(k1) |> layer.(k2)

      {init_fn, _} = Axon.build(model)

      inp = random({1, 1})

      assert %ModelState{data: %{"custom_0" => %{"kernel" => _}, "custom_1" => %{"kernel" => _}}} =
               init_fn.(inp, ModelState.empty())
    end

    test "computes forward pass with parameters" do
      input = Axon.input("input_0", shape: {nil, 1})
      kernel_param = Axon.param("kernel", fn shape -> shape end)

      model =
        Axon.layer(fn x, kernel, _opts -> Nx.multiply(x, kernel) end, [input, kernel_param],
          name: "layer_0"
        )

      {init_fn, _} = Axon.build(model)

      input = random({1, 1})

      assert %ModelState{
               data: %{"layer_0" => %{"kernel" => kernel}},
               parameters: %{"layer_0" => ["kernel"]}
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(Axon.predict(model, params, input), Nx.multiply(input, kernel))
    end

    defn layer_with_options(x, kernel, opts \\ []) do
      if opts[:add] do
        Nx.add(x, kernel)
      else
        Nx.multiply(x, kernel)
      end
    end

    test "computes forward pass with options" do
      kernel_param = Axon.param("kernel", fn shape -> shape end)

      input = random({1, 1})

      model1 =
        Axon.layer(&layer_with_options/3, [Axon.input("input_0", shape: {nil, 1}), kernel_param],
          name: "add",
          add: true
        )

      {init_fn, _} = Axon.build(model1)

      assert %ModelState{
               data: %{"add" => %{"kernel" => kernel}},
               parameters: %{"add" => ["kernel"]}
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(Axon.predict(model1, params, input), Nx.add(input, kernel))

      model2 =
        Axon.layer(&layer_with_options/3, [Axon.input("input_0", shape: {nil, 1}), kernel_param],
          name: "multiply",
          add: false
        )

      {init_fn, _} = Axon.build(model2)

      assert %ModelState{
               data: %{"multiply" => %{"kernel" => kernel}},
               parameters: %{"multiply" => ["kernel"]}
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(Axon.predict(model2, params, input), Nx.multiply(input, kernel))
    end
  end

  describe "block" do
    test "initializes correctly with single dense layer, used once" do
      block = Axon.block(&Axon.dense(&1, 32))
      model = block.(Axon.input("features"))

      {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data: %{"block_0" => %{"dense_0" => %{"kernel" => k, "bias" => b}}},
               parameters: %{"block_0" => %{"dense_0" => ["bias", "kernel"]}}
             } =
               init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert Nx.shape(k) == {1, 32}
      assert Nx.shape(b) == {32}
      assert Nx.type(k) == {:f, 32}
      assert Nx.type(b) == {:f, 32}
    end

    test "initializes correctly with single dense layer, used twice" do
      block = Axon.block(&Axon.dense(&1, 1))

      model =
        Axon.input("features")
        |> block.()
        |> block.()

      {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data:
                 %{"block_0" => %{"dense_0" => %{"kernel" => k, "bias" => b}} = block_params} =
                   params,
               parameters: %{"block_0" => %{"dense_0" => ["bias", "kernel"]}}
             } = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert Nx.shape(k) == {1, 1}
      assert Nx.shape(b) == {1}
      assert Nx.type(k) == {:f, 32}
      assert Nx.type(b) == {:f, 32}

      # no additional dense layers in block
      assert map_size(block_params) == 1
      # no additional blocks
      assert map_size(params) == 1
    end

    test "initializes correctly with multiple dense layer, used once" do
      block =
        Axon.block(fn x ->
          x
          |> Axon.dense(32, activation: :relu)
          |> Axon.dense(32, activation: :relu)
        end)

      model = block.(Axon.input("features"))
      {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data:
                 %{
                   "block_0" =>
                     %{
                       "dense_0" => %{"kernel" => k1, "bias" => b1},
                       "dense_1" => %{"kernel" => k2, "bias" => b2}
                     } = block_params
                 } = params,
               parameters: %{
                 "block_0" => %{"dense_0" => ["bias", "kernel"], "dense_1" => ["bias", "kernel"]}
               }
             } = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert Nx.shape(k1) == {1, 32}
      assert Nx.shape(b1) == {32}
      assert Nx.shape(k2) == {32, 32}
      assert Nx.shape(b2) == {32}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.type(b2) == {:f, 32}

      # no additional dense layers in block
      assert map_size(block_params) == 2
      # no additional blocks
      assert map_size(params) == 1
    end

    test "initializes correctly with multiple dense layer, used multiple times" do
      block =
        Axon.block(fn x ->
          x
          |> Axon.dense(32, activation: :relu)
          |> Axon.dense(1, activation: :relu)
        end)

      model = Enum.reduce(0..9, Axon.input("features"), fn _, x -> block.(x) end)

      {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data:
                 %{
                   "block_0" =>
                     %{
                       "dense_0" => %{"kernel" => k1, "bias" => b1},
                       "dense_1" => %{"kernel" => k2, "bias" => b2}
                     } = block_params
                 } = params,
               parameters: %{
                 "block_0" => %{"dense_0" => ["bias", "kernel"], "dense_1" => ["bias", "kernel"]}
               }
             } = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert Nx.shape(k1) == {1, 32}
      assert Nx.shape(b1) == {32}
      assert Nx.shape(k2) == {32, 1}
      assert Nx.shape(b2) == {1}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.type(b2) == {:f, 32}

      # no additional dense layers in block
      assert map_size(block_params) == 2
      # no additional blocks
      assert map_size(params) == 1
    end

    test "initializes correctly with multiple blocks in network" do
      block1 = Axon.block(&Axon.dense(&1, 32))
      block2 = Axon.block(&Axon.dense(&1, 32))

      model =
        Axon.input("features")
        |> block1.()
        |> block2.()

      {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data:
                 %{
                   "block_0" =>
                     %{
                       "dense_0" => %{"kernel" => k1, "bias" => b1}
                     } = block_0_params,
                   "block_1" =>
                     %{
                       "dense_0" => %{"kernel" => k2, "bias" => b2}
                     } = block_1_params
                 } = params,
               parameters: %{
                 "block_0" => %{"dense_0" => ["bias", "kernel"]},
                 "block_1" => %{"dense_0" => ["bias", "kernel"]}
               }
             } = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert Nx.shape(k1) == {1, 32}
      assert Nx.shape(b1) == {32}
      assert Nx.shape(k2) == {32, 32}
      assert Nx.shape(b2) == {32}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.type(b2) == {:f, 32}

      # no additional dense layers in block
      assert map_size(block_0_params) == 1
      assert map_size(block_1_params) == 1
      # no additional blocks
      assert map_size(params) == 2
    end

    test "initializes correctly with block inside of a block" do
      block =
        Axon.block(fn x ->
          inner_block = Axon.block(&Axon.dense(&1, 1))

          x |> inner_block.() |> inner_block.()
        end)

      model =
        Axon.input("features")
        |> block.()
        |> block.()

      {init_fn, _} = Axon.build(model)

      assert %ModelState{
               data:
                 %{
                   "block_0" =>
                     %{
                       "block_0" =>
                         %{"dense_0" => %{"kernel" => k, "bias" => b}} = inner_block_params
                     } = block_params
                 } = params,
               parameters: %{"block_0" => %{"block_0" => %{"dense_0" => ["bias", "kernel"]}}}
             } = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert Nx.shape(k) == {1, 1}
      assert Nx.shape(b) == {1}
      assert Nx.type(k) == {:f, 32}
      assert Nx.type(b) == {:f, 32}

      assert map_size(inner_block_params) == 1
      assert map_size(block_params) == 1
      assert map_size(params) == 1
    end

    test "predicts correctly with single dense, used once" do
      block = Axon.block(&Axon.dense(&1, 32))
      model = block.(Axon.input("features"))

      {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"block_0" => %{"dense_0" => %{"kernel" => k, "bias" => b}}},
               parameters: %{"block_0" => %{"dense_0" => ["bias", "kernel"]}}
             } =
               params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      input = random({1, 1})

      assert_equal(predict_fn.(params, input), Axon.Layers.dense(input, k, b))
    end

    test "predicts correctly with single dense, used twice" do
      block = Axon.block(&Axon.dense(&1, 1))

      model =
        Axon.input("features")
        |> block.()
        |> block.()

      {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"block_0" => %{"dense_0" => %{"kernel" => k, "bias" => b}}},
               parameters: %{"block_0" => %{"dense_0" => ["bias", "kernel"]}}
             } =
               params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      input = random({1, 1})

      assert_equal(
        predict_fn.(params, input),
        input |> Axon.Layers.dense(k, b) |> Axon.Layers.dense(k, b)
      )
    end

    test "predicts correctly with multiple dense, used once" do
      block =
        Axon.block(fn x ->
          x
          |> Axon.dense(32, activation: :relu)
          |> Axon.dense(1, activation: :relu)
        end)

      model = block.(Axon.input("features"))
      {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "block_0" => %{
                   "dense_0" => %{"kernel" => k1, "bias" => b1},
                   "dense_1" => %{"kernel" => k2, "bias" => b2}
                 }
               },
               parameters: %{
                 "block_0" => %{"dense_0" => ["bias", "kernel"], "dense_1" => ["bias", "kernel"]}
               }
             } = params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      expected_predict_fn = fn x, k1, b1, k2, b2 ->
        x
        |> Axon.Layers.dense(k1, b1)
        |> Axon.Activations.relu()
        |> Axon.Layers.dense(k2, b2)
        |> Axon.Layers.relu()
      end

      input = random({1, 1})

      assert_equal(predict_fn.(params, input), expected_predict_fn.(input, k1, b1, k2, b2))
    end

    test "predicts correctly with multiple dense, used twice" do
      block =
        Axon.block(fn x ->
          x
          |> Axon.dense(32, activation: :relu)
          |> Axon.dense(1, activation: :relu)
        end)

      model =
        Axon.input("features")
        |> block.()
        |> block.()

      {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{
                 "block_0" => %{
                   "dense_0" => %{"kernel" => k1, "bias" => b1},
                   "dense_1" => %{"kernel" => k2, "bias" => b2}
                 }
               },
               parameters: %{
                 "block_0" => %{"dense_0" => ["bias", "kernel"], "dense_1" => ["bias", "kernel"]}
               }
             } = params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      expected_predict_fn = fn x, k1, b1, k2, b2 ->
        x
        |> Axon.Layers.dense(k1, b1)
        |> Axon.Activations.relu()
        |> Axon.Layers.dense(k2, b2)
        |> Axon.Layers.relu()
        |> Axon.Layers.dense(k1, b1)
        |> Axon.Activations.relu()
        |> Axon.Layers.dense(k2, b2)
        |> Axon.Layers.relu()
      end

      input = random({1, 1})

      assert_equal(predict_fn.(params, input), expected_predict_fn.(input, k1, b1, k2, b2))
    end

    test "predicts correctly with multiple blocks in network" do
      block1 = Axon.block(&Axon.dense(&1, 32))
      block2 = Axon.block(&Axon.dense(&1, 32))

      model =
        Axon.input("features")
        |> block1.()
        |> block2.()

      {init_fn, predict_fn} = Axon.build(model)

      actual_predict_fn = fn x, k1, b1, k2, b2 ->
        x
        |> Axon.Layers.dense(k1, b1)
        |> Axon.Layers.dense(k2, b2)
      end

      assert %ModelState{
               data: %{
                 "block_0" => %{
                   "dense_0" => %{"kernel" => k1, "bias" => b1}
                 },
                 "block_1" => %{
                   "dense_0" => %{"kernel" => k2, "bias" => b2}
                 }
               },
               parameters: %{
                 "block_0" => %{"dense_0" => ["bias", "kernel"]},
                 "block_1" => %{"dense_0" => ["bias", "kernel"]}
               }
             } = params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      input = random({1, 1})

      assert_equal(predict_fn.(params, input), actual_predict_fn.(input, k1, b1, k2, b2))
    end

    test "predicts correctly with block inside of a block" do
      block =
        Axon.block(fn x ->
          inner_block = Axon.block(&Axon.dense(&1, 1))

          x |> inner_block.() |> inner_block.()
        end)

      model =
        Axon.input("features")
        |> block.()
        |> block.()

      {init_fn, predict_fn} = Axon.build(model)

      actual_predict_fn = fn x, k, b ->
        x
        |> Axon.Layers.dense(k, b)
        |> Axon.Layers.dense(k, b)
        |> Axon.Layers.dense(k, b)
        |> Axon.Layers.dense(k, b)
      end

      assert %ModelState{
               data: %{
                 "block_0" => %{
                   "block_0" => %{"dense_0" => %{"kernel" => k, "bias" => b}}
                 }
               },
               parameters: %{"block_0" => %{"block_0" => %{"dense_0" => ["bias", "kernel"]}}}
             } = params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      input = random({1, 1})
      assert_equal(predict_fn.(params, input), actual_predict_fn.(input, k, b))
    end

    test "works with multiple block inputs" do
      block =
        Axon.block(fn x, y ->
          dense = Axon.block(&Axon.dense(&1, 4))
          Axon.add(dense.(y), dense.(x))
        end)

      input1 = Axon.input("input1")
      input2 = Axon.input("input2")

      model = block.(input1, input2) |> Axon.dense(1)

      {init_fn, predict_fn} = Axon.build(model)

      actual_predict_fn = fn %{"input1" => x, "input2" => y}, k1, b1, k2, b2 ->
        x = Axon.Layers.dense(x, k1, b1)
        y = Axon.Layers.dense(y, k1, b1)

        x
        |> Nx.add(y)
        |> Axon.Layers.dense(k2, b2)
      end

      input = %{"input1" => Nx.tensor([[0.5]]), "input2" => Nx.tensor([[0.75]])}

      assert %ModelState{
               data: %{
                 "block_0" => %{
                   "block_0" => %{"dense_0" => %{"kernel" => k1, "bias" => b1}}
                 },
                 "dense_0" => %{"kernel" => k2, "bias" => b2}
               }
             } = params = init_fn.(input, ModelState.empty())

      assert_equal(predict_fn.(params, input), actual_predict_fn.(input, k1, b1, k2, b2))
    end
  end

  describe "initializers" do
    test "work with functions" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(2,
          kernel_initializer:
            Axon.Initializers.variance_scaling(
              scale: 1.0e-4,
              distribution: :uniform,
              mode: :fan_avg
            )
        )

      {init_fn, _} = Axon.build(model)

      inp = Nx.tensor([[1.0]])

      assert %ModelState{
               data: %{"dense_0" => %{"kernel" => k, "bias" => b}},
               parameters: %{"dense_0" => ["bias", "kernel"]}
             } = init_fn.(inp, ModelState.empty())

      assert Nx.shape(k) == {1, 2}
      assert Nx.shape(b) == {2}
    end

    test "do not initialize same shape to same params" do
      model =
        Axon.input("data")
        |> Axon.dense(10)
        |> Axon.dense(10)

      {init_fn, _} = Axon.build(model)

      inp = Nx.iota({1, 10})

      assert %ModelState{
               data: %{"dense_0" => dense_0, "dense_1" => dense_1},
               parameters: %{"dense_0" => ["bias", "kernel"], "dense_1" => ["bias", "kernel"]}
             } = init_fn.(inp, ModelState.empty())

      assert %{"kernel" => k0, "bias" => _} = dense_0
      assert %{"kernel" => k1, "bias" => _} = dense_1

      assert_not_equal(k0, k1)
    end
  end

  describe "initialize from fixed model" do
    test "initializes entire model from start point" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2)

      {init_fn_1, _} = Axon.build(model)
      {init_fn_2, _} = Axon.build(model)

      inp = Nx.tensor([[1.0]])

      assert %ModelState{data: params_1} = init_fn_1.(inp, ModelState.empty())
      assert %ModelState{data: params_2} = init_fn_2.(inp, ModelState.new(params_1))

      assert_equal(params_1, params_2)
    end
  end

  describe "containers" do
    test "allows accessors with custom layers" do
      input1 = random({1, 1})
      input2 = random({1, 2})
      inputs = %{"input_0" => input1, "input_1" => input2}

      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 2})
      tuple_model = Axon.container({inp1, inp2})
      first_elem = Axon.nx(tuple_model, &elem(&1, 0))
      second_elem = Axon.nx(tuple_model, &elem(&1, 1))

      assert_equal(Axon.predict(first_elem, ModelState.empty(), inputs), input1)
      assert_equal(Axon.predict(second_elem, ModelState.empty(), inputs), input2)

      map_model = Axon.container(%{foo: inp1, bar: inp2})
      foo_elem = Axon.nx(map_model, & &1.foo)
      bar_elem = Axon.nx(map_model, & &1.bar)

      assert_equal(Axon.predict(foo_elem, ModelState.empty(), inputs), input1)
      assert_equal(Axon.predict(bar_elem, ModelState.empty(), inputs), input2)

      nested_model = Axon.container({{inp1}, %{foo: {inp2}}})
      first_elem = Axon.nx(nested_model, &elem(elem(&1, 0), 0))
      second_elem = Axon.nx(nested_model, &elem(elem(&1, 1).foo, 0))

      assert_equal(Axon.predict(first_elem, ModelState.empty(), inputs), input1)
      assert_equal(Axon.predict(second_elem, ModelState.empty(), inputs), input2)
    end
  end

  describe "edge cases" do
    test "raises clean error on missing layer" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      input = Nx.tensor([[1.0]])

      assert_raise ArgumentError, ~r/layer \"dense_0\" does not exist/, fn ->
        Axon.predict(model, ModelState.empty(), input)
      end
    end

    test "raises clean error on missing parameter" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      input = Nx.tensor([[1.0]])

      assert_raise ArgumentError, ~r/parameter \"kernel\" for layer:/, fn ->
        Axon.predict(model, ModelState.new(%{"dense_0" => %{}}), input)
      end
    end

    test "initializes a non-linear model" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2, name: "dense_0")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2, name: "dense_1")
      model = Axon.add(x, y)

      {init_fn, _} = Axon.build(model)

      input = %{"input_0" => Nx.tensor([[1.0]]), "input_1" => Nx.tensor([[2.0]])}

      assert %ModelState{
               data: %{"dense_0" => _, "dense_1" => _},
               parameters: %{"dense_0" => ["bias", "kernel"], "dense_1" => ["bias", "kernel"]}
             } = init_fn.(input, ModelState.empty())
    end
  end

  describe "instrumentation" do
    @describetag :capture_log

    test "predict logs debug utilities when debug true" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)

      {init_fn, _} = Axon.build(model, debug: true)

      assert capture_log(fn ->
               init_fn.(Nx.tensor([[1.0]]), ModelState.empty())
             end) =~ "Axon finished init"
    end

    test "init logs debug utilities when debug true" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      input = Nx.tensor([[1.0]])

      {init_fn, _} = Axon.build(model)

      params = init_fn.(Nx.template({1, 1}, {:f, 32}), ModelState.empty())

      assert capture_log(fn ->
               Axon.predict(model, params, input, debug: true)
             end) =~ "Axon finished predict"
    end

    test "compile logs debug utilities when debug true" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      input = Nx.tensor([[1.0]])

      {init_fn, predict_fn} = Axon.build(model, debug: true)

      assert capture_log(fn ->
               init_fn.(input, ModelState.empty())
             end) =~ "Axon finished init"

      params = init_fn.(input, ModelState.empty())

      assert capture_log(fn ->
               predict_fn.(params, input)
             end) =~ "Axon finished predict"
    end
  end

  describe "stack_columns" do
    test "works with lazy container" do
      model = Axon.input("lazy_container") |> Axon.stack_columns(ignore: [:b])

      input = %LazyOnly{a: [[1]], b: [[2]], c: [[3]]}

      assert_equal(
        Axon.predict(model, ModelState.empty(), %{"lazy_container" => input}),
        Nx.tensor([[1.0, 3.0]])
      )
    end
  end

  describe "determinism" do
    test "builds the same model multiple times" do
      builder = fn -> Axon.input("input", shape: {nil, 784}) |> Axon.dense(128) end
      {_, predict_fn1} = Axon.Compiler.build(builder.(), [])
      {_, predict_fn2} = Axon.Compiler.build(builder.(), [])
      assert predict_fn1 == predict_fn2
    end

    test "builds a model with dropout" do
      builder = fn ->
        node = Axon.input("input", shape: {nil, 784})
        Axon.add(Axon.dropout(node), node)
      end

      {_, predict_fn1} = Axon.Compiler.build(builder.(), [])
      {_, predict_fn2} = Axon.Compiler.build(builder.(), [])
      assert predict_fn1 == predict_fn2
    end
  end

  describe "metadata" do
    test "axon compiler attaches layer name as metadata to subgraphs" do
      model = Axon.input("input", shape: {nil, 784}) |> Axon.dense(128)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 784}, :f32), ModelState.empty())
      input = Nx.broadcast(0.0, {1, 784})

      expr_fn = Nx.Defn.jit(predict_fn, compiler: Axon.Defn)
      expr = expr_fn.(params, input)

      assert %{data: %{op: :metadata, args: [_tensor, %{axon_layer: :dense}]}} = expr
    end
  end

  describe "parameters" do
    test "supports passing a tuple instead of a function as the shape" do
      a = Axon.param("a", {1, 1})
      input = Axon.input("input")

      model =
        Axon.layer(fn x, a, _opts -> Nx.add(x, a) end, [input, a], name: "custom")

      x = random({1, 1})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"custom" => %{"a" => a}}} =
               params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert_equal(predict_fn.(params, x), Nx.add(x, a))
    end

    test "supports composite/map parameter types" do
      inner_param = Axon.param("inner", fn _ -> {1, 1} end)
      param = Axon.param("composite", {:map, [inner_param]})
      input = Axon.input("input")

      model =
        Axon.layer(fn x, %{"inner" => inner}, _opts -> Nx.add(x, inner) end, [input, param],
          name: "custom"
        )

      x = random({1, 1})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{data: %{"custom" => %{"composite" => %{"inner" => inner}}}} =
               params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert_equal(predict_fn.(params, x), Nx.add(x, inner))
    end

    test "inner params in composite parameters initialize to different values" do
      a = Axon.param("a", fn _ -> {1, 1} end)
      b = Axon.param("b", fn _ -> {1, 1} end)
      param = Axon.param("composite", {:map, [a, b]})

      input = Axon.input("input")

      model =
        Axon.layer(fn x, %{"a" => a}, _opts -> Nx.add(x, a) end, [input, param], name: "custom")

      assert {init_fn, _} = Axon.build(model)

      assert %ModelState{data: %{"custom" => %{"composite" => %{"a" => a, "b" => b}}}} =
               init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert_not_equal(a, b)
    end

    test "supports a composite of composites" do
      a = Axon.param("a", fn _ -> {1, 1} end)
      inner_composite = Axon.param("inner_composite", {:map, [a]})
      composite = Axon.param("composite", {:map, [inner_composite]})

      input = Axon.input("input")

      model =
        Axon.layer(
          fn x, %{"inner_composite" => %{"a" => a}}, _opts -> Nx.add(x, a) end,
          [input, composite],
          name: "custom"
        )

      x = random({1, 1})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %ModelState{
               data: %{"custom" => %{"composite" => %{"inner_composite" => %{"a" => a}}}}
             } =
               params = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert_equal(predict_fn.(params, x), Nx.add(x, a))
    end
  end

  describe "global layer options" do
    test "global options are forwarded to the layer when declared" do
      input = Axon.input("input")

      model =
        Axon.layer(
          fn input, opts ->
            assert Keyword.has_key?(opts, :option1)
            refute Keyword.has_key?(opts, :option2)
            input
          end,
          [input],
          global_options: [:option1]
        )

      {_, predict_fn} = Axon.build(model, global_layer_options: [option1: true, option2: true])

      params = ModelState.empty()
      input = random({1, 1}, type: {:f, 32})
      predict_fn.(params, input)
    end
  end

  describe "bidirectional" do
    test "works properly with LSTMs" do
      input = Axon.input("input")

      model =
        input
        |> Axon.embedding(10, 16)
        |> Axon.bidirectional(
          &Axon.lstm(&1, 32, name: "lstm"),
          &Nx.concatenate([&1, &2], axis: 1),
          name: "bidirectional"
        )
        |> Axon.nx(&elem(&1, 0))

      {init_fn, predict_fn} = Axon.build(model)

      input = Nx.broadcast(1, {1, 10})

      assert %ModelState{
               data: %{
                 "bidirectional" => %{"lstm" => _}
               }
             } = params = init_fn.(input, ModelState.empty())

      out = predict_fn.(params, input)
      assert Nx.shape(out) == {1, 20, 32}
    end
  end

  describe "inspect values" do
    test "prints intermediate layer values to the screen" do
      model =
        Axon.input("x")
        |> Axon.dense(10, name: "foo")
        |> Axon.dense(4, name: "bar")

      {init_fn, predict_fn} = Axon.build(model, print_values: true)
      input = Nx.broadcast(1, {1, 10})

      model_state = init_fn.(input, ModelState.empty())

      out =
        ExUnit.CaptureIO.capture_io(fn ->
          predict_fn.(model_state, input)
        end)

      assert out =~ "x:"
      assert out =~ "foo:"
      assert out =~ "bar:"
    end
  end

  describe "weight tying" do
    test "initializes with shared parameters" do
      model =
        Axon.input("x")
        |> Axon.embedding(32, 32, name: "embed")
        |> Axon.dense(32, name: "dense")

      init_state =
        ModelState.empty()
        |> ModelState.tie(["embed", "kernel"], ["dense", "kernel"])

      {init_fn, _} = Axon.build(model)
      input = Nx.template({1, 4}, :u32)
      assert %Axon.ModelState{data: %{"embed" => %{"kernel" => %Axon.ModelState.SharedParameter{}}}} = init_fn.(input, init_state)
    end

    test "performs inference with weights tied after initialization" do
      model =
        Axon.input("x")
        |> Axon.embedding(32, 32, name: "embed")
        |> Axon.dense(32, name: "dense")

      {init_fn, predict_fn} = Axon.build(model)

      %Axon.ModelState{data: %{"dense" => %{"kernel" => k, "bias" => b}}} =
        model_state = init_fn.(Nx.template({1, 4}, :u32), ModelState.empty())

      model_state =
        Axon.ModelState.tie(model_state, ["embed", "kernel"], ["dense", "kernel"])

      input = Nx.tensor([[0, 1, 2, 3]])

      actual_predict_fn = fn input, kernel, bias ->
        input
        |> Axon.Layers.embedding(kernel)
        |> Axon.Layers.dense(kernel, bias)
      end

      assert_equal(actual_predict_fn.(input, k, b), predict_fn.(model_state, input))
    end
  end
end
