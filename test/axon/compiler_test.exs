defmodule CompilerTest do
  use Axon.Case, async: true
  import ExUnit.CaptureLog

  alias Axon.MixedPrecision, as: AMP

  describe "input" do
    test "single input, single output" do
      model = Axon.input("input_0", shape: {nil, 1})
      input = Nx.random_uniform({1, 1}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{} = init_fn.(input, %{})
      assert_equal(predict_fn.(%{}, input), input)
    end

    test "multi-input, map with default names" do
      model1 =
        {Axon.input("input_0", shape: {nil, 1}), Axon.input("input_1", shape: {nil, 1})}
        |> Axon.container()

      input1 = Nx.random_uniform({1, 1})
      input2 = Nx.random_uniform({1, 1})
      input = %{"input_0" => input1, "input_1" => input2}

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{} = init_fn.(input, %{})

      assert_equal(
        {input1, input2},
        predict_fn.(%{}, input)
      )
    end

    test "output map" do
      model = %{foo: Axon.input("input_0", shape: {nil, 1})} |> Axon.container()

      input = Nx.random_uniform({1, 1})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{} = init_fn.(input, %{})
      assert_equal(%{foo: input}, predict_fn.(%{}, %{"input_0" => input}))
    end

    test "multi-input, multi-output, nested" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})

      model1 = {input1, {input1, {input2, {}}, input2, %{foo: input1}}} |> Axon.container()

      inp1 = Nx.random_uniform({1, 1})
      inp2 = Nx.random_uniform({1, 2})
      input = %{"input_0" => inp1, "input_1" => inp2}

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{} = init_fn.(input, %{})

      assert_equal(
        {inp1, {inp1, {inp2, {}}, inp2, %{foo: inp1}}},
        predict_fn.(%{}, input)
      )
    end

    test "multi-input, map with custom names" do
      x = Axon.input("x", shape: {nil, 1})
      y = Axon.input("y", shape: {nil, 1})
      z = Axon.input("z", shape: {nil, 1})
      model = {z, x, y} |> Axon.container()

      x_val = Nx.random_uniform({1, 1})
      y_val = Nx.random_uniform({1, 1})
      z_val = Nx.random_uniform({1, 1})
      input = %{"x" => x_val, "y" => y_val, "z" => z_val}

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{} = init_fn.(input, %{})

      assert_equal(
        {z_val, x_val, y_val},
        predict_fn.(%{}, input)
      )
    end

    test "allows container inputs" do
      model = Axon.input("input_0", shape: %{foo: {nil, 1}, bar: {{nil, 2}, {nil, 3}}})

      input = %{foo: Nx.tensor([[1]]), bar: {Nx.tensor([[1, 2]]), Nx.tensor([[1, 2, 3]])}}

      assert_equal(Axon.predict(model, %{}, %{"input_0" => input}), input)
    end

    test "allows lazy container inputs" do
      model = Axon.input("lazy_container") |> Axon.nx(fn x -> Nx.add(x.a, x.c) end)

      input = %LazyOnly{a: [[1]], b: [[2]], c: [[3]]}

      assert_equal(Axon.predict(model, %{}, %{"lazy_container" => input}), Nx.tensor([[4]]))
    end

    test "raises if input not found, no default value" do
      model = Axon.input("input_0", shape: {nil, 32})
      input = Nx.random_uniform({1, 16})
      assert {_, predict_fn} = Axon.build(model)

      exception = assert_raise ArgumentError, fn -> predict_fn.(%{}, %{foo: input}) end

      assert Exception.message(exception) =~
               "unable to find input"
    end

    test "raises helpful error messages" do
      input = Axon.input("input")
      x1 = Axon.dense(input, 32)
      x2 = Axon.dense(input, 64)
      model = Axon.add(x1, x2)

      {init_fn, _predict_fn} = Axon.build(model)
      %Axon.CompileError{} = exception = catch_error(init_fn.(Nx.template({1, 16}, :f32), %{}))

      message = Exception.message(exception)
      assert message =~ "exception found when compiling layer Axon.Layers.add/2 named add_0"
      assert message =~ "cannot broadcast tensor of dimensions {1, 32} to {1, 64}"
      assert message =~ "cannot broadcast tensor of dimensions {1, 32} to {1, 64}"
      assert message =~ "The layer was defined at:"
      assert message =~ "test/axon/compiler_test.exs:#{__ENV__.line - 10}: CompilerTest.\"test"
      assert message =~ "Compiling of the model was initiated at:"
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
                     Axon.predict(model, %{}, %{})
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

      assert init_fn.(%{"input_0" => Nx.tensor([[20]])}, %{}) == %{}
      assert init_fn.(%{}, %{}) == %{}

      assert_equal(
        predict_fn.(%{}, %{"input_0" => Nx.tensor([[20]])}),
        Nx.tensor(1)
      )

      assert_equal(
        predict_fn.(%{}, %{}),
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

      params = init_fn.(inputs, %{})
      assert Map.keys(params) == ["bias_0"]

      assert_equal(predict_fn.(params, inputs), Nx.tensor([0]))

      inputs = %{"input_0" => Nx.tensor([[20]]), "input_1" => Nx.tensor([[20]])}

      params = init_fn.(inputs, %{})
      assert params |> Map.keys() |> Enum.sort() == ["bias_0", "dense_0"]

      assert_equal(predict_fn.(params, inputs), Nx.tensor([1]))
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

      assert_equal(Axon.predict(model, %{}, %{}), Nx.tensor([1]))
    end
  end

  describe "constant" do
    test "initializes with no params" do
      model = Axon.constant(Nx.tensor(1.0))

      assert {init_fn, _} = Axon.build(model)

      assert %{} == init_fn.(%{}, %{})
    end

    test "computes forward pass with default options" do
      model = Axon.constant(Nx.tensor(1.0))

      assert {_, predict_fn} = Axon.build(model)
      assert_equal(predict_fn.(%{}, {}), Nx.tensor(1.0))
    end

    test "computes forward pass with other layers" do
      model = Axon.add(Axon.constant(Nx.tensor(1.0)), Axon.constant(Nx.tensor(2.0)))

      assert {_, predict_fn} = Axon.build(model)
      assert_equal(predict_fn.(%{}, {}), Nx.tensor(3.0))
    end

    test "computes forward pass with output policy" do
      model = Axon.constant(Nx.tensor(1.0))
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {_, predict_fn} = Axon.build(mp_model)
      assert_equal(predict_fn.(%{}, {}), Nx.tensor(1.0, type: {:bf, 16}))
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
        input = Nx.random_uniform({1, 32})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert %{} = init_fn.(input, %{})
      end
    end

    test "computes forward pass with default options" do
      for activation <- @activation_layers do
        model = Axon.input("input_0", shape: {nil, 1}) |> Axon.activation(activation)
        input = Nx.random_uniform({1, 1})

        assert {_init_fn, predict_fn} = Axon.build(model)
        assert_equal(predict_fn.(%{}, input), apply(Axon.Activations, activation, [input]))
      end
    end

    test "computes forward pass with custom options" do
      for activation <- [:celu, :elu, :leaky_relu] do
        model = Axon.input("input_0", shape: {nil, 32}) |> Axon.activation(activation, alpha: 0.8)
        input = Nx.random_uniform({1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(%{}, input),
          apply(Axon.Activations, activation, [input, [alpha: 0.8]])
        )
      end
    end

    test "computes forward pass with output policy" do
      for activation <- @activation_layers do
        model = Axon.input("input_0", shape: {nil, 1}) |> Axon.activation(activation)
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 1})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, %{}), Nx.random_uniform({1, 1}))) == {:bf, 16}
      end
    end
  end

  describe "bias" do
    test "initializes in default case" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.bias(name: "bias")

      input = Nx.random_uniform({1, 1})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{"bias" => %{"bias" => bias}} = init_fn.(input, %{})
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end
  end

  describe "dense" do
    test "initializes in default case" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(1, name: "dense")

      input = Nx.random_uniform({1, 1})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.shape(kernel) == {1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      input = Nx.random_uniform({1, 1})

      model1 =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(1, name: "dense", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model1)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert_equal(kernel, zeros({1, 1}))
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(1, name: "dense", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
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
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input, %{})
      assert_equal(predict_fn.(params, input), Axon.Layers.dense(input, kernel, bias))
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(1, name: "dense")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.build(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      input = Nx.random_uniform({1, 1})

      assert %{"dense" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(kernel_grad, Nx.broadcast(0.0, {1, 1}))
      assert_equal(bias_grad, Nx.broadcast(0.0, {1}))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, name: "dense")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 2})

      assert {init_fn, _} = Axon.build(mp_model)
      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, name: "dense")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, name: "dense", use_bias: false)

      input = Nx.random_uniform({1, 2})

      assert {init_fn, _} = Axon.build(model)
      assert %{"dense" => %{"kernel" => _} = dense_params} = init_fn.(input, %{})
      assert Map.has_key?(dense_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2}) |> Axon.dense(1, name: "dense", use_bias: false)

      input = Nx.random_uniform({1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"dense" => %{"kernel" => k}} = params = init_fn.(input, %{})

      assert_all_close(predict_fn.(params, input), Axon.Layers.dense(input, k, Nx.tensor(0.0)))
    end
  end

  describe "bilinear" do
    test "initializes in default case" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")

      inputs = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 2})}

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(inputs, %{})
      assert Nx.shape(kernel) == {1, 1, 2}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model1 = Axon.bilinear(input1, input2, 1, name: "bilinear", kernel_initializer: :zeros)

      inputs = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 2})}

      assert {init_fn, _predict_fn} = Axon.build(model1)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(inputs, %{})
      assert_equal(kernel, zeros({1, 1, 2}))
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}

      model2 = Axon.bilinear(input1, input2, 1, name: "bilinear", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(inputs, %{})
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

      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} =
               params = init_fn.(inputs, %{})

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

      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} =
               params = init_fn.(input1, %{})

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.bilinear(input1, input2, kernel, bias)
      )
    end

    test "returns zero gradient for frozen parameters" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear") |> Axon.freeze()

      input = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 2})}

      assert {init_fn, predict_fn} = Axon.build(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"bilinear" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(kernel_grad, Nx.broadcast(0.0, {1, 1, 2}))
      assert_equal(bias_grad, Nx.broadcast(0.0, {1}))
    end

    test "initializes with parameter policy" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 2})}

      assert {init_fn, _} = Axon.build(mp_model)
      assert %{"bilinear" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 2})}

      assert {init_fn, predict_fn} = Axon.build(mp_model)

      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear", use_bias: false)

      input = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 2})}

      assert {init_fn, _} = Axon.build(model)
      assert %{"bilinear" => %{"kernel" => _} = bilinear_params} = init_fn.(input, %{})
      assert Map.has_key?(bilinear_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      input1 = Axon.input("input_0", shape: {nil, 1})
      input2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.bilinear(input1, input2, 1, name: "bilinear", use_bias: false)

      inp1 = Nx.random_uniform({1, 1})
      inp2 = Nx.random_uniform({1, 2})

      input = %{"input_0" => inp1, "input_1" => inp2}

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"bilinear" => %{"kernel" => k}} = params = init_fn.(input, %{})

      assert_all_close(
        predict_fn.(params, input),
        Axon.Layers.bilinear(inp1, inp2, k, Nx.tensor(0.0))
      )
    end
  end

  describe "embedding" do
    test "initializes in default case" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.embedding(1, 1, name: "embedding")

      input = Nx.random_uniform({1, 1})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{"embedding" => %{"kernel" => kernel}} = init_fn.(input, %{})
      assert Nx.shape(kernel) == {1, 1}
      assert Nx.type(kernel) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.embedding(1, 1, name: "embedding", kernel_initializer: :zeros)

      input = Nx.random_uniform({1, 1})

      assert {init_fn, _predict_fn} = Axon.build(model1)
      assert %{"embedding" => %{"kernel" => kernel}} = init_fn.(input, %{})
      assert_equal(kernel, zeros({1, 1}))
    end

    test "computes forward pass" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.embedding(1, 1, name: "embedding", kernel_initializer: :identity)

      input = Nx.tensor([[0]])

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"embedding" => %{"kernel" => kernel}} = params = init_fn.(input, %{})
      assert_equal(predict_fn.(params, input), Axon.Layers.embedding(input, kernel))
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.embedding(1, 1, name: "embedding")
        |> Axon.freeze()

      input = Nx.tensor([[0]])

      assert {init_fn, predict_fn} = Axon.build(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"embedding" => %{"kernel" => kernel_grad}} =
               apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(kernel_grad, Nx.broadcast(0.0, {1, 1}))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.embedding(1, 1, name: "embedding")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.tensor([[0, 1]])

      assert {init_fn, _} = Axon.build(mp_model)
      assert %{"embedding" => %{"kernel" => kernel}} = init_fn.(input, %{})
      assert Nx.type(kernel) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.embedding(1, 1, name: "embedding")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.tensor([[0]])

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end
  end

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  describe "pooling" do
    test "initializes with no params" do
      for pool <- @pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])

        input = Nx.random_uniform({1, 32, 1})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert %{} = init_fn.(input, %{})
      end
    end

    test "computes forward pass with default options" do
      default_options = [kernel_size: 1]

      for pool <- @pooling_layers do
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])
        input1 = Nx.random_uniform({1, 32, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(%{}, input1),
          apply(Axon.Layers, pool, [input1, default_options])
        )

        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 1})])
        input2 = Nx.random_uniform({1, 8, 4, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)

        assert_equal(
          predict_fn.(%{}, input2),
          apply(Axon.Layers, pool, [input2, default_options])
        )

        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 2, 1})])
        input3 = Nx.random_uniform({1, 8, 4, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)

        assert_equal(
          predict_fn.(%{}, input3),
          apply(Axon.Layers, pool, [input3, default_options])
        )
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @pooling_layers do
        opts1 = [kernel_size: 6]
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1}), opts1])
        input1 = Nx.random_uniform({1, 32, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)
        assert_equal(predict_fn.(%{}, input1), apply(Axon.Layers, pool, [input1, opts1]))

        opts2 = [kernel_size: 2, strides: 2, padding: :same]
        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 1}), opts2])
        input2 = Nx.random_uniform({1, 8, 4, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)
        assert_equal(predict_fn.(%{}, input2), apply(Axon.Layers, pool, [input2, opts2]))

        opts3 = [kernel_size: {2, 1, 2}, strides: [1, 2, 1], padding: [{0, 1}, {1, 1}, {0, 2}]]
        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 2, 1}), opts3])
        input3 = Nx.random_uniform({1, 8, 4, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)
        assert_equal(predict_fn.(%{}, input3), apply(Axon.Layers, pool, [input3, opts3]))
      end
    end

    test "lp_pool computes forward pass with custom norm" do
      model = Axon.input("input", shape: {nil, 32, 1}) |> Axon.lp_pool(norm: 3)
      input = Nx.random_uniform({1, 32, 1}, type: {:f, 32})

      assert {_, predict_fn} = Axon.build(model)
      assert_equal(predict_fn.(%{}, input), Axon.Layers.lp_pool(input, kernel_size: {1}, norm: 3))
    end

    test "computes forward pass with output policy" do
      for pool <- @pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 32, 1})

        assert {init_fn, predict_fn} = Axon.build(mp_model)

        assert Nx.type(predict_fn.(init_fn.(input, %{}), Nx.random_uniform({1, 32, 1}))) ==
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

        inp = Nx.random_uniform({1, 32, 1})

        assert {_, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(%{}, inp),
          apply(Axon.Layers, pool, [inp, [kernel_size: {2}, strides: [2], channels: :last]])
        )
      end
    end

    # TODO: Add back in with transform validations
    # test "fails on bad options" do
    #   for pool <- @pooling_layers do
    #     assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
    #       apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 28, 28}), [strides: :foo]])
    #     end

    #     assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
    #       apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 28, 28}), [kernel_size: :foo]])
    #     end

    #     assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
    #       apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 28, 28}), [padding: :foo]])
    #     end
    #   end
    # end
  end

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool, :adaptive_lp_pool]

  describe "adaptive pooling" do
    test "initializes with no params" do
      for pool <- @adaptive_pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])

        input = Nx.random_uniform({1, 32, 1})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert %{} = init_fn.(input, %{})
      end
    end

    test "computes forward pass with default options" do
      for pool <- @adaptive_pooling_layers do
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])
        input1 = Nx.random_uniform({1, 32, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(%{}, input1),
          apply(Axon.Layers, pool, [input1, [output_size: 32]])
        )

        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 1})])
        input2 = Nx.random_uniform({1, 8, 4, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)

        assert_equal(
          predict_fn.(%{}, input2),
          apply(Axon.Layers, pool, [input2, [output_size: {8, 4}]])
        )

        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 2, 1})])
        input3 = Nx.random_uniform({1, 8, 4, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)

        assert_equal(
          predict_fn.(%{}, input3),
          apply(Axon.Layers, pool, [input3, [output_size: {8, 4, 2}]])
        )
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @adaptive_pooling_layers do
        opts1 = [output_size: 27]
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1}), opts1])
        input1 = Nx.random_uniform({1, 32, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)
        assert_equal(predict_fn.(%{}, input1), apply(Axon.Layers, pool, [input1, opts1]))

        opts2 = [output_size: {2, 3}]
        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 1}), opts2])
        input2 = Nx.random_uniform({1, 8, 4, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)
        assert_equal(predict_fn.(%{}, input2), apply(Axon.Layers, pool, [input2, opts2]))

        opts3 = [output_size: {4, 3, 1}]
        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 8, 4, 2, 1}), opts3])
        input3 = Nx.random_uniform({1, 8, 4, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)
        assert_equal(predict_fn.(%{}, input3), apply(Axon.Layers, pool, [input3, opts3]))
      end
    end

    test "computes forward pass with output policy" do
      for pool <- @adaptive_pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 32, 1})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 32, 1})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
      end
    end

    test "computes forward pass with channels last" do
      for pool <- @adaptive_pooling_layers do
        model =
          apply(Axon, pool, [
            Axon.input("input", shape: {nil, 32, 1}),
            [channels: :last, output_size: {27}]
          ])

        inp = Nx.random_uniform({1, 32, 1})

        assert {_, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(%{}, inp),
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

        input = Nx.random_uniform({1, 1, 32})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert %{} = init_fn.(input, %{})
      end
    end

    test "computes forward pass with default options" do
      for pool <- @global_pooling_layers do
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 4})])
        input1 = Nx.random_uniform({1, 1, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)
        assert_equal(predict_fn.(%{}, input1), apply(Axon.Layers, pool, [input1]))

        model2 = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 2, 2})])
        input2 = Nx.random_uniform({1, 1, 2, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2)
        assert_equal(predict_fn.(%{}, input2), apply(Axon.Layers, pool, [input2]))

        model3 = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 2, 2, 1})])
        input3 = Nx.random_uniform({1, 1, 2, 2, 1}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3)
        assert_equal(predict_fn.(%{}, input3), apply(Axon.Layers, pool, [input3]))
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @global_pooling_layers do
        opts1 = [keep_axes: true]
        model1 = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 2}), opts1])
        input1 = Nx.random_uniform({1, 1, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1)
        assert_equal(predict_fn.(%{}, input1), apply(Axon.Layers, pool, [input1, opts1]))
      end
    end

    test "computes forward pass with output policy" do
      for pool <- @global_pooling_layers do
        model = apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 2})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 1, 2})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
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

        inp = Nx.random_uniform({1, 32, 1})

        assert {_, predict_fn} = Axon.build(model1)

        assert_equal(
          predict_fn.(%{}, inp),
          apply(Axon.Layers, pool, [inp, [keep_axes: true, channels: :last]])
        )

        assert {_, predict_fn} = Axon.build(model2)

        assert_equal(
          predict_fn.(%{}, inp),
          apply(Axon.Layers, pool, [inp, [keep_axes: false, channels: :last]])
        )
      end
    end
  end

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  describe "dropout" do
    test "initializes with no params" do
      for dropout <- @dropout_layers do
        model = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 32})])

        input = Nx.random_uniform({1, 1, 32})

        assert {init_fn, _predict_fn} = Axon.build(model)
        assert %{} = init_fn.(input, %{})
      end
    end

    test "computes forward pass with default options" do
      for dropout <- @dropout_layers do
        model1 = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 32})])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1, mode: :train)
        %{prediction: result1} = predict_fn.(%{}, input1)

        assert Nx.shape(result1) == {1, 1, 32}
        assert Nx.type(result1) == {:f, 32}
        assert_not_equal(result1, input1)

        model2 = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 8, 4})])
        input2 = Nx.random_uniform({1, 1, 8, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model2, mode: :train)
        %{prediction: result2} = predict_fn.(%{}, input2)

        assert Nx.shape(result2) == {1, 1, 8, 4}
        assert Nx.type(result2) == {:f, 32}
        assert_not_equal(result2, input2)

        model3 = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 8, 4, 2})])
        input3 = Nx.random_uniform({1, 1, 8, 4, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model3, mode: :train)
        %{prediction: result3} = predict_fn.(%{}, input3)

        assert Nx.shape(result3) == {1, 1, 8, 4, 2}
        assert Nx.type(result3) == {:f, 32}
        assert_not_equal(result3, input3)
      end
    end

    test "computes forward pass with custom options" do
      for dropout <- @dropout_layers do
        opts1 = [rate: 0.25]
        model1 = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 32}), opts1])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.build(model1, mode: :train)

        %{prediction: result} = predict_fn.(%{}, input1)

        assert Nx.shape(result) == {1, 1, 32}
        assert Nx.type(result) == {:f, 32}
        assert_not_equal(result, input1)
      end
    end

    test "computes forward pass with output policy" do
      for dropout <- @dropout_layers do
        model = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 32})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 1, 32})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
      end
    end

    test "not present in inference mode" do
      for dropout <- @dropout_layers do
        model = apply(Axon, dropout, [Axon.input("input", shape: {nil, 1, 32})])
        input = Nx.random_uniform({1, 1, 32})

        assert_equal(Axon.predict(model, %{}, input), input)
      end
    end
  end

  describe "convolution" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 32, 32, 3}) |> Axon.conv(64, name: "conv")

      input = Nx.template({1, 32, 32, 3}, {:f, 32})

      assert {init_fn, _} = Axon.build(model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
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
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert_equal(kernel, zeros({1, 1, 3, 32}))
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 3, 32, 32})
        |> Axon.conv(32, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.shape(kernel) == {1, 1, 3, 32}
      assert Nx.type(kernel) == {:f, 32}
      assert_equal(bias, zeros({32}))
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv(2, name: "conv")
      input1 = Nx.random_uniform({1, 1, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input1, %{})
      assert_equal(predict_fn.(params, input1), Axon.Layers.conv(input1, kernel, bias))

      model2 = Axon.input("input", shape: {nil, 1, 2, 2}) |> Axon.conv(3, name: "conv")
      input2 = Nx.random_uniform({1, 1, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input2, %{})
      assert_equal(predict_fn.(params, input2), Axon.Layers.conv(input2, kernel, bias))

      model3 = Axon.input("input", shape: {nil, 1, 2, 2, 2}) |> Axon.conv(4, name: "conv")
      input3 = Nx.random_uniform({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input3, %{})
      assert_equal(predict_fn.(params, input3), Axon.Layers.conv(input3, kernel, bias))
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, padding: :same, input_dilation: 2]

      model1 =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.conv(2, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = Nx.random_uniform({1, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input1, %{})
      assert_equal(predict_fn.(params, input1), Axon.Layers.conv(input1, kernel, bias, opts1))

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input("input", shape: {nil, 4, 4, 1})
        |> Axon.conv(2, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 4, 4, 1})

      assert {init_fn, predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input2, %{})
      assert_equal(predict_fn.(params, input2), Axon.Layers.conv(input2, kernel, bias, opts2))

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input("input", shape: {nil, 2, 2, 2, 1})
        |> Axon.conv(4, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 2, 2, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input3, %{})
      assert_equal(predict_fn.(params, input3), Axon.Layers.conv(input3, kernel, bias, opts3))
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input("input", shape: {nil, 1, 32})
        |> Axon.conv(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.build(model)

      input = Nx.random_uniform({1, 1, 32})

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"conv" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(kernel_grad, Nx.broadcast(0.0, {1, 1, 1}))
      assert_equal(bias_grad, Nx.broadcast(0.0, {1}))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 1, 32}) |> Axon.conv(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.template({1, 1, 32}, {:f, 32})

      assert {init_fn, _} = Axon.build(mp_model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 32}) |> Axon.conv(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 32})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, _} = Axon.build(model)
      assert %{"conv" => %{"kernel" => _} = conv_params} = init_fn.(input, %{})
      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"conv" => %{"kernel" => k}} = params = init_fn.(input, %{})
      assert_equal(predict_fn.(params, input), Axon.Layers.conv(input, k, Nx.tensor(0)))
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 6}) |> Axon.conv(2, name: "conv", channels: :last)

      input = Nx.random_uniform({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"conv" => %{"kernel" => k, "bias" => b}} = params = init_fn.(input, %{})
      assert_equal(predict_fn.(params, input), Axon.Layers.conv(input, k, b, channels: :last))
    end

    # TODO: Add these back in with deftransform validations
    # so the error message is more clear
    # test "raises on bad options" do
    #   assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
    #     model = Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(128, kernel_size: :foo)
    #     Axon.init(model, Nx.template({1, 1, 28, 28}, :f32))
    #   end

    #   assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
    #     model = Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(128, strides: :foo)
    #     Axon.init(model, Nx.template({1, 1, 28, 28}, :f32))
    #   end

    #   assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
    #     model = Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(128, padding: :foo)
    #     Axon.init(model, Nx.template({1, 1, 28, 28}, :f32))
    #   end

    #   assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
    #     model = Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(128, input_dilation: :foo)
    #     Axon.init(model, Nx.template({1, 1, 28, 28}, :f32))
    #   end

    #   assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
    #     model = Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(128, kernel_dilation: :foo)
    #     Axon.init(model, Nx.template({1, 1, 28, 28}, :f32))
    #   end
    # end
  end

  describe "depthwise convolution" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.depthwise_conv(3, name: "conv")

      input = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.shape(kernel) == {1, 1, 1, 9}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {9}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input", shape: {nil, 2, 2, 3})
        |> Axon.depthwise_conv(3, name: "conv", kernel_initializer: :zeros)

      input = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, _predict_fn} = Axon.build(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert_equal(kernel, zeros({1, 1, 1, 9}))
      assert Nx.shape(bias) == {9}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 2, 2, 3})
        |> Axon.depthwise_conv(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.shape(kernel) == {1, 1, 1, 9}
      assert Nx.type(kernel) == {:f, 32}
      assert_equal(bias, zeros({9}))
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 1, 8}) |> Axon.depthwise_conv(3, name: "conv")
      input1 = Nx.random_uniform({1, 1, 8}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input1, %{})
      assert_equal(predict_fn.(params, input1), Axon.Layers.depthwise_conv(input1, kernel, bias))

      model2 = Axon.input("input", shape: {nil, 1, 2, 2}) |> Axon.depthwise_conv(4, name: "conv")
      input2 = Nx.random_uniform({1, 1, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input2, %{})
      assert_equal(predict_fn.(params, input2), Axon.Layers.depthwise_conv(input2, kernel, bias))

      model3 =
        Axon.input("input", shape: {nil, 1, 2, 2, 2}) |> Axon.depthwise_conv(5, name: "conv")

      input3 = Nx.random_uniform({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input3, %{})
      assert_equal(predict_fn.(params, input3), Axon.Layers.depthwise_conv(input3, kernel, bias))
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, padding: :same, input_dilation: 2]

      model1 =
        Axon.input("input", shape: {nil, 8, 1})
        |> Axon.depthwise_conv(1, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = Nx.random_uniform({1, 8, 1})

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input1, %{})

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.depthwise_conv(input1, kernel, bias, opts1)
      )

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input("input", shape: {nil, 4, 4, 1})
        |> Axon.depthwise_conv(8, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 4, 4, 1})

      assert {init_fn, predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input2, %{})

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.depthwise_conv(input2, kernel, bias, opts2)
      )

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input("input", shape: {nil, 3, 2, 2, 1})
        |> Axon.depthwise_conv(2, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 3, 2, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input3, %{})

      assert_equal(
        predict_fn.(params, input3),
        Axon.Layers.depthwise_conv(input3, kernel, bias, opts3)
      )
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.depthwise_conv(1, name: "conv")
        |> Axon.freeze()

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"conv" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(kernel_grad, Nx.broadcast(0.0, {1, 1, 1}))
      assert_equal(bias_grad, Nx.broadcast(0.0, {1}))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 1, 2}) |> Axon.depthwise_conv(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, _} = Axon.build(mp_model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 2}) |> Axon.depthwise_conv(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.depthwise_conv(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, _} = Axon.build(model)
      assert %{"conv" => %{"kernel" => _} = conv_params} = init_fn.(input, %{})
      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.depthwise_conv(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"conv" => %{"kernel" => k}} = params = init_fn.(input, %{})
      assert_equal(predict_fn.(params, input), Axon.Layers.depthwise_conv(input, k, Nx.tensor(0)))
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 6})
        |> Axon.depthwise_conv(2, name: "conv", channels: :last)

      input = Nx.random_uniform({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"conv" => %{"kernel" => k, "bias" => b}} = params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.depthwise_conv(input, k, b, channels: :last)
      )
    end

    # TODO: Validate at layer level and bring these back
    # test "fails on bad options" do
    #   assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.depthwise_conv(3, kernel_size: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.depthwise_conv(3, strides: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.depthwise_conv(3, padding: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.depthwise_conv(3, input_dilation: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.depthwise_conv(3, kernel_dilation: :foo)
    #   end
    # end
  end

  describe "convolution transpose" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.conv_transpose(32, name: "conv")

      input = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.shape(kernel) == {1, 1, 3, 32}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input", shape: {nil, 2, 2, 3})
        |> Axon.conv_transpose(32, name: "conv", kernel_initializer: :zeros)

      input = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, _predict_fn} = Axon.build(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert_equal(kernel, zeros({1, 1, 3, 32}))
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 32, 32, 3})
        |> Axon.conv_transpose(32, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.shape(kernel) == {1, 1, 3, 32}
      assert Nx.type(kernel) == {:f, 32}
      assert_equal(bias, zeros({32}))
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 1, 4}) |> Axon.conv_transpose(3, name: "conv")
      input1 = Nx.random_uniform({1, 1, 4}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input1, %{})
      assert_equal(predict_fn.(params, input1), Axon.Layers.conv_transpose(input1, kernel, bias))

      model2 = Axon.input("input", shape: {nil, 1, 4, 4}) |> Axon.conv_transpose(4, name: "conv")
      input2 = Nx.random_uniform({1, 1, 4, 4}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input2, %{})
      assert_equal(predict_fn.(params, input2), Axon.Layers.conv_transpose(input2, kernel, bias))

      model3 =
        Axon.input("input", shape: {nil, 1, 2, 2, 2}) |> Axon.conv_transpose(5, name: "conv")

      input3 = Nx.random_uniform({1, 1, 2, 2, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input3, %{})
      assert_equal(predict_fn.(params, input3), Axon.Layers.conv_transpose(input3, kernel, bias))
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, kernel_dilation: 1]

      model1 =
        Axon.input("input", shape: {nil, 4, 1})
        |> Axon.conv_transpose(1, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = Nx.random_uniform({1, 4, 1})

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input1, %{})

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.conv_transpose(input1, kernel, bias, opts1)
      )

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input("input", shape: {nil, 4, 4, 1})
        |> Axon.conv_transpose(8, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 4, 4, 1})

      assert {init_fn, predict_fn} = Axon.build(model2)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input2, %{})

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.conv_transpose(input2, kernel, bias, opts2)
      )

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input("input", shape: {nil, 2, 2, 2, 1})
        |> Axon.conv_transpose(2, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 2, 2, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model3)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(input3, %{})

      assert_equal(
        predict_fn.(params, input3),
        Axon.Layers.conv_transpose(input3, kernel, bias, opts3)
      )
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.conv_transpose(1, name: "conv")
        |> Axon.freeze()

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"conv" => %{"kernel" => kernel_grad, "bias" => bias_grad}} =
               apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(kernel_grad, Nx.broadcast(0.0, {1, 1, 1}))
      assert_equal(bias_grad, Nx.broadcast(0.0, {1}))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv_transpose(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, _} = Axon.build(mp_model)
      assert %{"conv" => %{"kernel" => kernel, "bias" => bias}} = init_fn.(input, %{})
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv_transpose(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.conv_transpose(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, _} = Axon.build(model)
      assert %{"conv" => %{"kernel" => _} = conv_params} = init_fn.(input, %{})
      assert Map.has_key?(conv_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2})
        |> Axon.conv_transpose(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"conv" => %{"kernel" => k}} = params = init_fn.(input, %{})
      assert_equal(predict_fn.(params, input), Axon.Layers.conv_transpose(input, k, Nx.tensor(0)))
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 6})
        |> Axon.conv_transpose(2, name: "conv", channels: :last)

      input = Nx.random_uniform({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"conv" => %{"kernel" => k, "bias" => b}} = params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.conv_transpose(input, k, b, channels: :last)
      )
    end
  end

  describe "separable convolution 2d" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.separable_conv2d(3, name: "conv")

      input = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model1)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = init_fn.(input, %{})

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

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = init_fn.(input, %{})

      assert_equal(b1, zeros({9}))
      assert_equal(b2, zeros({9}))
      assert Nx.shape(k1) == {1, 1, 1, 9}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {1, 1, 1, 9}
      assert Nx.type(k2) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model = Axon.input("input", shape: {nil, 3, 2, 2}) |> Axon.separable_conv2d(3, name: "conv")
      input = Nx.random_uniform({1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = params = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 3, 3, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv2d(input, k1, b1, k2, b2, opts)
      )
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2})
        |> Axon.separable_conv2d(1, name: "conv")
        |> Axon.freeze()

      input = Nx.random_uniform({1, 1, 3, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

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
             } = apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(k1_grad, Nx.broadcast(0.0, {1, 1, 1, 1}))
      assert_equal(b1_grad, Nx.broadcast(0.0, {1}))
      assert_equal(k2_grad, Nx.broadcast(0.0, {1, 1, 1, 1}))
      assert_equal(b2_grad, Nx.broadcast(0.0, {1}))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 1, 3, 2}) |> Axon.separable_conv2d(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 3, 2})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2
               }
             } = init_fn.(input, %{})

      assert Nx.type(k1) == {:bf, 16}
      assert Nx.type(b1) == {:bf, 16}
      assert Nx.type(k2) == {:bf, 16}
      assert Nx.type(b2) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 3, 2}) |> Axon.separable_conv2d(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 3, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2, 2})
        |> Axon.separable_conv2d(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2, 2})

      assert {init_fn, _} = Axon.build(model)
      assert %{"conv" => %{"kernel_1" => _, "kernel_2" => _} = conv_params} = init_fn.(input, %{})
      assert Map.has_key?(conv_params, "bias_1") == false
      assert Map.has_key?(conv_params, "bias_2") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 2, 2})
        |> Axon.separable_conv2d(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"conv" => %{"kernel_1" => k1, "kernel_2" => k2}} = params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv2d(input, k1, Nx.tensor(0), k2, Nx.tensor(0))
      )
    end

    test "computes forward pass with channels last" do
      model =
        Axon.input("input", shape: {nil, 3, 3, 6})
        |> Axon.separable_conv2d(2, name: "conv", channels: :last)

      input = Nx.random_uniform({1, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{"conv" => %{"kernel_1" => k1, "kernel_2" => k2, "bias_1" => b1, "bias_2" => b2}} =
               params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv2d(input, k1, b1, k2, b2, channels: :last)
      )
    end

    # TODO: Add these back in with validations
    # test "fails on bad options" do
    #   assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.separable_conv2d(3, kernel_size: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.separable_conv2d(3, strides: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.separable_conv2d(3, padding: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.separable_conv2d(3, input_dilation: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.separable_conv2d(3, kernel_dilation: :foo)
    #   end
    # end
  end

  describe "separable convolution 3d" do
    test "initializes in default case" do
      model =
        Axon.input("input", shape: {nil, 3, 2, 2, 3}) |> Axon.separable_conv3d(3, name: "conv")

      input = Nx.random_uniform({1, 3, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 3, 2, 2, 3})

      assert {init_fn, _} = Axon.build(model1)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = init_fn.(input, %{})

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

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 3, 2, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = params = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 3, 2, 3, 3})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3, opts)
      )
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2, 2})
        |> Axon.separable_conv3d(1, name: "conv")
        |> Axon.freeze()

      input = Nx.random_uniform({1, 1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

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
             } = apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(k1_grad, Nx.broadcast(0.0, {1, 1, 1, 1, 1}))
      assert_equal(b1_grad, Nx.broadcast(0.0, {1}))
      assert_equal(k2_grad, Nx.broadcast(0.0, {1, 1, 1, 1, 1}))
      assert_equal(b2_grad, Nx.broadcast(0.0, {1}))
      assert_equal(k3_grad, Nx.broadcast(0.0, {1, 1, 1, 1, 1}))
      assert_equal(b3_grad, Nx.broadcast(0.0, {1}))
    end

    test "initializes with parameter policy" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2, 2}) |> Axon.separable_conv3d(1, name: "conv")

      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 3, 2, 2})

      assert {init_fn, _} = Axon.build(mp_model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "bias_1" => b1,
                 "kernel_2" => k2,
                 "bias_2" => b2,
                 "kernel_3" => k3,
                 "bias_3" => b3
               }
             } = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2, 2})
        |> Axon.separable_conv3d(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 3, 2, 2})

      assert {init_fn, _} = Axon.build(model)

      assert %{"conv" => %{"kernel_1" => _, "kernel_2" => _, "kernel_3" => _} = conv_params} =
               init_fn.(input, %{})

      assert Map.has_key?(conv_params, "bias_1") == false
      assert Map.has_key?(conv_params, "bias_2") == false
      assert Map.has_key?(conv_params, "bias_3") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 1, 3, 2, 2})
        |> Axon.separable_conv3d(1, name: "conv", use_bias: false)

      input = Nx.random_uniform({1, 1, 3, 2, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{"conv" => %{"kernel_1" => k1, "kernel_2" => k2, "kernel_3" => k3}} =
               params = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 3, 3, 3, 6})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "conv" => %{
                 "kernel_1" => k1,
                 "kernel_2" => k2,
                 "kernel_3" => k3,
                 "bias_1" => b1,
                 "bias_2" => b2,
                 "bias_3" => b3
               }
             } = params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3, channels: :last)
      )
    end

    # TODO: Bring these back with layer-level validations
    # test "fails on bad options" do
    #   assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, kernel_size: {1, 1})
    #   end

    #   assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, strides: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, padding: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, input_dilation: :foo)
    #   end

    #   assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
    #     Axon.input("input", shape: {nil, 1, 28, 28, 3})
    #     |> Axon.separable_conv3d(3, kernel_dilation: [1, 1])
    #   end
    # end
  end

  @normalization_with_stats_layers [:batch_norm, :instance_norm]

  describe "normalization with stats" do
    test "initializes in default case" do
      for norm <- @normalization_with_stats_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"]])

          input = Nx.random_uniform({1, 2})

          assert {init_fn, _predict_fn} = Axon.build(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
          assert Nx.shape(gamma) == {2}
          assert Nx.type(gamma) == {:f, 32}
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2, 2, 3}), [name: "norm"]])

        input = Nx.random_uniform({1, 2, 2, 3})

        assert {init_fn, _predict_fn} = Axon.build(model2)

        assert %{"norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}} =
                 init_fn.(input, %{})

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

          input = Nx.random_uniform({1, 2})

          assert {init_fn, _predict_fn} = Axon.build(model1)

          assert %{"norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}} =
                   init_fn.(input, %{})

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

        input = Nx.random_uniform({1, 2, 2, 3})

        assert {init_fn, _predict_fn} = Axon.build(model2)

        assert %{"norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}} =
                 init_fn.(input, %{})

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
          input1 = Nx.random_uniform({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.build(model1)

          assert %{"norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}} =
                   params = init_fn.(input1, %{})

          assert_equal(
            predict_fn.(params, input1),
            apply(Axon.Layers, norm, [input1, gamma, beta, mean, var])
          )
        end

        model2 = apply(Axon, norm, [Axon.input("input", shape: {nil, 3, 2, 2}), [name: "norm"]])
        input2 = Nx.random_uniform({1, 3, 2, 2}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2)

        assert %{"norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}} =
                 params = init_fn.(input2, %{})

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

          input1 = Nx.random_uniform({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.build(model1)

          assert %{"norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}} =
                   params = init_fn.(input1, %{})

          assert_equal(
            predict_fn.(params, input1),
            apply(Axon.Layers, norm, [input1, gamma, beta, mean, var, opts1])
          )
        end

        opts2 = [channel_index: 3, epsilon: 1.0e-4]

        model2 =
          apply(Axon, norm, [Axon.input("input", shape: {nil, 2, 2, 3}), [name: "norm"] ++ opts2])

        input2 = Nx.random_uniform({1, 2, 2, 3}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2)

        assert %{"norm" => %{"gamma" => gamma, "beta" => beta, "mean" => mean, "var" => var}} =
                 params = init_fn.(input2, %{})

        assert_equal(
          predict_fn.(params, input2),
          apply(Axon.Layers, norm, [input2, gamma, beta, mean, var, opts2])
        )
      end
    end

    test "returns zero gradient for frozen parameters" do
      for norm <- @normalization_with_stats_layers do
        model =
          apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
          |> Axon.freeze()

        input = Nx.random_uniform({1, 1, 2})

        assert {init_fn, predict_fn} = Axon.build(model)

        backward = fn params, input ->
          Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
        end

        assert %{"norm" => %{"gamma" => gamma_grad, "beta" => beta_grad}} =
                 apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

        assert_equal(gamma_grad, Nx.broadcast(0.0, {1}))
        assert_equal(beta_grad, Nx.broadcast(0.0, {1}))
      end
    end

    test "initializes with parameter policy" do
      for norm <- @normalization_with_stats_layers do
        model = apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(params: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 1, 2})

        assert {init_fn, _} = Axon.build(mp_model)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
        assert Nx.type(gamma) == {:bf, 16}
        assert Nx.type(beta) == {:bf, 16}
      end
    end

    test "computes forward pass with output policy" do
      for norm <- @normalization_with_stats_layers do
        model = apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 1, 2})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
      end
    end
  end

  @normalization_layers [:layer_norm]

  describe "normalization" do
    test "initializes in default case" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"]])

          input = Nx.random_uniform({1, 2})

          assert {init_fn, _predict_fn} = Axon.build(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
          assert Nx.shape(gamma) == {2}
          assert Nx.type(gamma) == {:f, 32}
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2, 2, 3}), [name: "norm"]])

        input = Nx.random_uniform({1, 2, 2, 3})

        assert {init_fn, _predict_fn} = Axon.build(model2)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
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

          input = Nx.random_uniform({1, 2})

          assert {init_fn, _predict_fn} = Axon.build(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
          assert_equal(gamma, zeros({2}))
          assert Nx.shape(beta) == {2}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 =
          apply(Axon, norm, [
            Axon.input("input", shape: {nil, 2, 2, 3}),
            [name: "norm", beta_initializer: :zeros]
          ])

        input = Nx.random_uniform({1, 2, 2, 3})

        assert {init_fn, _predict_fn} = Axon.build(model2)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
        assert Nx.shape(gamma) == {3}
        assert Nx.type(gamma) == {:f, 32}
        assert_equal(beta, zeros({3}))
      end
    end

    test "computes forward pass with default options" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"]])
          input1 = Nx.random_uniform({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.build(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.(input1, %{})

          assert_equal(
            predict_fn.(params, input1),
            apply(Axon.Layers, norm, [input1, gamma, beta])
          )
        end

        model2 = apply(Axon, norm, [Axon.input("input", shape: {nil, 3, 2, 2}), [name: "norm"]])
        input2 = Nx.random_uniform({1, 3, 2, 2}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.(input2, %{})
        assert_equal(predict_fn.(params, input2), apply(Axon.Layers, norm, [input2, gamma, beta]))
      end
    end

    test "computes forward pass with custom options" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          opts1 = [channel_index: 1, epsilon: 1.0e-3]

          model1 =
            apply(Axon, norm, [Axon.input("input", shape: {nil, 2}), [name: "norm"] ++ opts1])

          input1 = Nx.random_uniform({1, 2}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.build(model1)
          assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.(input1, %{})

          assert_equal(
            predict_fn.(params, input1),
            apply(Axon.Layers, norm, [input1, gamma, beta, opts1])
          )
        end

        opts2 = [channel_index: 3, epsilon: 1.0e-4]

        model2 =
          apply(Axon, norm, [Axon.input("input", shape: {nil, 2, 2, 3}), [name: "norm"] ++ opts2])

        input2 = Nx.random_uniform({1, 2, 2, 3}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.build(model2)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.(input2, %{})

        assert_equal(
          predict_fn.(params, input2),
          apply(Axon.Layers, norm, [input2, gamma, beta, opts2])
        )
      end
    end

    test "returns zero gradient for frozen parameters" do
      for norm <- @normalization_layers do
        model =
          apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
          |> Axon.freeze()

        assert {init_fn, predict_fn} = Axon.build(model)

        input = Nx.random_uniform({1, 1, 2})

        backward = fn params, input ->
          Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
        end

        assert %{"norm" => %{"gamma" => gamma_grad, "beta" => beta_grad}} =
                 apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

        assert_equal(gamma_grad, Nx.broadcast(0.0, {1}))
        assert_equal(beta_grad, Nx.broadcast(0.0, {1}))
      end
    end

    test "initializes with parameter policy" do
      for norm <- @normalization_layers do
        model = apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(params: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 1, 2})

        assert {init_fn, _} = Axon.build(mp_model)
        assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
        assert Nx.type(gamma) == {:bf, 16}
        assert Nx.type(beta) == {:bf, 16}
      end
    end

    test "computes forward pass with output policy" do
      for norm <- @normalization_layers do
        model = apply(Axon, norm, [Axon.input("input", shape: {nil, 1, 2}), [name: "norm"]])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        input = Nx.random_uniform({1, 1, 2})

        assert {init_fn, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
      end
    end
  end

  describe "group normalization" do
    test "initializes in default case" do
      model = Axon.input("input", shape: {nil, 3}) |> Axon.group_norm(3, name: "norm")

      input = Nx.random_uniform({1, 3})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
      assert Nx.shape(gamma) == {3}
      assert Nx.type(gamma) == {:f, 32}
      assert Nx.shape(beta) == {3}
      assert Nx.type(beta) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input("input", shape: {nil, 3})
        |> Axon.group_norm(3, name: "norm", gamma_initializer: :zeros)

      input = Nx.random_uniform({1, 3})

      assert {init_fn, _predict_fn} = Axon.build(model1)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
      assert_equal(gamma, zeros({3}))
      assert Nx.shape(beta) == {3}
      assert Nx.type(beta) == {:f, 32}

      model2 =
        Axon.input("input", shape: {nil, 3, 3})
        |> Axon.group_norm(3, name: "norm", beta_initializer: :zeros)

      input = Nx.random_uniform({1, 3, 3})

      assert {init_fn, _predict_fn} = Axon.build(model2)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
      assert_equal(beta, zeros({3}))
      assert Nx.shape(gamma) == {3}
      assert Nx.type(gamma) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 2}) |> Axon.group_norm(2, name: "norm")
      input1 = Nx.random_uniform({1, 2})

      assert {init_fn, predict_fn} = Axon.build(model1)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.(input1, %{})

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.group_norm(input1, gamma, beta, num_groups: 2)
      )

      model2 = Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.group_norm(3, name: "norm")
      input2 = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, predict_fn} = Axon.build(model2)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.(input2, %{})

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.group_norm(input2, gamma, beta, num_groups: 3)
      )
    end

    test "computes forward pass with custom options" do
      opts = [epsilon: 1.0e-3, channel_index: 3]

      model =
        Axon.input("input", shape: {nil, 2, 2, 3}) |> Axon.group_norm(3, [name: "norm"] ++ opts)

      input = Nx.random_uniform({1, 2, 2, 3})

      assert {init_fn, predict_fn} = Axon.build(model)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.group_norm(input, gamma, beta, [num_groups: 3] ++ opts)
      )
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.group_norm(1, name: "norm")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.build(model)

      input = Nx.random_uniform({1, 2})

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"norm" => %{"gamma" => gamma_grad, "beta" => beta_grad}} =
               apply(Nx.Defn.jit(backward), [init_fn.(input, %{}), input])

      assert_equal(gamma_grad, Nx.broadcast(0.0, {2}))
      assert_equal(beta_grad, Nx.broadcast(0.0, {2}))
    end

    test "initializes with parameter policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.group_norm(1, name: "norm")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 2})

      assert {init_fn, _} = Axon.build(mp_model)
      assert %{"norm" => %{"gamma" => gamma, "beta" => beta}} = init_fn.(input, %{})
      assert Nx.type(gamma) == {:bf, 16}
      assert Nx.type(beta) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 2}) |> Axon.group_norm(1, name: "norm")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 3})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end
  end

  describe "flatten" do
    test "initializes with no params" do
      model = Axon.input("input_0", shape: {nil, 32}) |> Axon.flatten()

      input = Nx.random_uniform({1, 32})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{} = init_fn.(input, %{})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input_0", shape: {nil, 32}) |> Axon.flatten()
      input1 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.build(model1)
      assert_equal(predict_fn.(%{}, input1), Axon.Layers.flatten(input1))

      model2 = Axon.input("input", shape: {nil, 3, 32, 32}) |> Axon.flatten()
      input2 = Nx.random_uniform({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.build(model2)
      assert_equal(predict_fn.(%{}, input2), Axon.Layers.flatten(input2))
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input_0", shape: {nil, 32}) |> Axon.flatten()
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 3})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end
  end

  describe "transpose" do
    test "initializes with no params" do
      model = Axon.input("input", shape: {nil, 3, 32}) |> Axon.transpose()

      input = Nx.random_uniform({1, 3, 32})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{} = init_fn.(input, %{})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input_0", shape: {nil, 32}) |> Axon.transpose([0, 1])
      input1 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.build(model1)
      assert_equal(predict_fn.(%{}, input1), Nx.transpose(input1, axes: [0, 1]))

      model2 = Axon.input("input", shape: {nil, 3, 32, 32}) |> Axon.transpose([0, 2, 1, 3])
      input2 = Nx.random_uniform({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.build(model2)
      assert_equal(predict_fn.(%{}, input2), Nx.transpose(input2, axes: [0, 2, 1, 3]))
    end

    test "computes forward pass with constant" do
      model = Axon.constant(Nx.iota({1, 2, 3})) |> Axon.transpose([2, 1, 0])

      assert {_, predict_fn} = Axon.build(model)
      assert_equal(predict_fn.(%{}, {}), Nx.transpose(Nx.iota({1, 2, 3}, type: {:f, 32})))
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input_0", shape: {nil, 32}) |> Axon.transpose([0, 1])
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 32})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end
  end

  describe "reshape" do
    test "initializes with no params" do
      model = Axon.input("input", shape: {nil, 1, 32}) |> Axon.reshape({16, 2})

      input = Nx.random_uniform({1, 1, 32})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{} = init_fn.(input, %{})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input_0", shape: {nil, 32}) |> Axon.reshape({16, 2})
      input1 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.build(model1)
      assert_equal(predict_fn.(%{}, input1), Nx.reshape(input1, {1, 16, 2}))

      model2 = Axon.input("input", shape: {nil, 3, 32, 32}) |> Axon.reshape({3, 16, 2, 32})
      input2 = Nx.random_uniform({1, 3, 32, 32})

      assert {_, predict_fn} = Axon.build(model2)
      assert_equal(predict_fn.(%{}, input2), Nx.reshape(input2, {1, 3, 16, 2, 32}))
    end

    test "computes forward pass with constant input" do
      model = Axon.constant(Nx.iota({6})) |> Axon.reshape({1, 2, 3})

      assert {_, predict_fn} = Axon.build(model)
      assert_equal(predict_fn.(%{}, {}), Nx.tensor([[[0, 1, 2], [3, 4, 5]]], type: {:f, 32}))
    end

    test "computes forward pass with magic :batch and :auto" do
      model = Axon.input("input") |> Axon.reshape({:batch, 3, :auto})

      assert {_, predict_fn} = Axon.build(model)

      input = Nx.random_uniform({2, 4, 6})
      assert_equal(predict_fn.(%{}, input), Nx.reshape(input, {2, 3, 8}))
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input_0", shape: {nil, 32}) |> Axon.reshape({2, 16})
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 32})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end
  end

  describe "resize" do
    test "initializes with no params" do
      model = Axon.input("input", shape: {nil, 1, 3, 3}) |> Axon.resize({4, 4})

      input = Nx.random_uniform({1, 1, 3, 3})

      assert {init_fn, _predict_fn} = Axon.build(model)
      assert %{} = init_fn.(input, %{})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 1, 3, 3}) |> Axon.resize({4, 4})
      input1 = Nx.random_uniform({1, 1, 3, 3})

      assert {_, predict_fn} = Axon.build(model1)
      assert_equal(predict_fn.(%{}, input1), Axon.Layers.resize(input1, size: {4, 4}))
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 1, 3, 3}) |> Axon.resize({4, 4})
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      input = Nx.random_uniform({1, 1, 3, 3})

      assert {init_fn, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(init_fn.(input, %{}), input)) == {:bf, 16}
    end
  end

  describe "lstm" do
    test "initializes in default case" do
      model =
        Axon.input("input", shape: {nil, 32, 10})
        |> Axon.lstm(64, name: "lstm")
        |> Axon.container()

      input = Nx.random_uniform({1, 32, 10})

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %{
               "lstm" => %{
                 "input_kernel" => {wii, wif, wig, wio},
                 "hidden_kernel" => {whi, whf, whg, who},
                 "bias" => {bi, bf, bg, bo}
               }
             } = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 32, 10})

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %{
               "lstm" => %{
                 "input_kernel" => {wii, wif, wig, wio},
                 "hidden_kernel" => {whi, whf, whg, who},
                 "bias" => {bi, bf, bg, bo}
               }
             } = init_fn.(input, %{})

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

      assert %{
               "lstm" => %{
                 "input_kernel" => {wii, wif, wig, wio},
                 "hidden_kernel" => {whi, whf, whg, who},
                 "bias" => {bi, bf, bg, bo}
               }
             } = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 8, 2}, type: {:f, 32})

      init_carry = {zeros({1, 1, 2}), zeros({1, 1, 2})}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "lstm" => %{
                 "input_kernel" => {wii, wif, wig, wio},
                 "hidden_kernel" => {whi, whf, whg, who},
                 "bias" => {bi, bf, bg, bo}
               }
             } = params = init_fn.(input, %{})

      k = {wii, wif, wig, wio}
      h = {whi, whf, whg, who}
      b = {bi, bf, bg, bo}

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(
          &Axon.Layers.lstm_cell/5,
          input,
          init_carry,
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

      input1 = Nx.random_uniform({1, 8, 2}, type: {:f, 32})

      init_carry1 = {zeros({1, 1, 2}), zeros({1, 1, 2})}

      cell_fn1 = fn i, c, k, h, b ->
        Axon.Layers.lstm_cell(
          i,
          c,
          k,
          h,
          b,
          &Axon.Activations.relu/1,
          &Axon.Activations.sigmoid/1
        )
      end

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %{
               "lstm" => %{
                 "input_kernel" => {wii, wif, wig, wio},
                 "hidden_kernel" => {whi, whf, whg, who},
                 "bias" => {bi, bf, bg, bo}
               }
             } = params = init_fn.(input1, %{})

      k = {wii, wif, wig, wio}
      h = {whi, whf, whg, who}
      b = {bi, bf, bg, bo}

      assert_all_close(
        predict_fn.(params, input1),
        Axon.Layers.dynamic_unroll(cell_fn1, input1, init_carry1, k, h, b)
      )

      model2 =
        Axon.input("input", shape: {nil, 8, 2})
        |> Axon.lstm(2, name: "lstm", unroll: :static, recurrent_initializer: :zeros)
        |> Axon.container()

      input2 = Nx.random_uniform({1, 8, 2}, type: {:f, 32})

      init_carry2 = {zeros({1, 1, 2}), zeros({1, 1, 2})}

      cell_fn2 = &Axon.Layers.lstm_cell/5

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %{
               "lstm" => %{
                 "input_kernel" => {wii, wif, wig, wio},
                 "hidden_kernel" => {whi, whf, whg, who},
                 "bias" => {bi, bf, bg, bo}
               }
             } = params = init_fn.(input2, %{})

      k = {wii, wif, wig, wio}
      h = {whi, whf, whg, who}
      b = {bi, bf, bg, bo}

      assert_all_close(
        predict_fn.(params, input2),
        Axon.Layers.static_unroll(cell_fn2, input2, init_carry2, k, h, b)
      )
    end

    test "computes forward pass with hidden state" do
      seq = Axon.input("input", shape: {nil, 8, 2})
      {_, carry} = seq |> Axon.lstm(2, name: "encode", recurrent_initializer: :zeros)
      model = Axon.lstm(seq, carry, 2, name: "decode") |> Axon.container()
      input = Nx.random_uniform({1, 8, 2})

      assert {init_fn, predict_fn} = Axon.build(model)

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        init_carry = {zeros({1, 1, 2}), zeros({1, 1, 2})}

        {_, carry} =
          Axon.Layers.dynamic_unroll(&Axon.Layers.lstm_cell/5, inp, init_carry, ei, eh, eb)

        Axon.Layers.dynamic_unroll(&Axon.Layers.lstm_cell/5, inp, carry, di, dh, db)
      end

      assert %{
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
             } = params = init_fn.(input, %{})

      enc = {ek, eh, eb}
      dec = {dk, dh, db}

      assert_equal(predict_fn.(params, input), equiv_fn.(input, enc, dec))
    end

    # TODO(seanmor5): Update this with https://github.com/elixir-nx/axon/issues/90
    # test "returns zero gradient for frozen parameters" do
    # end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.lstm(2, name: "lstm", use_bias: false)
        |> Axon.container()

      input = Nx.random_uniform({1, 2, 1})

      assert {init_fn, _} = Axon.build(model)

      assert %{
               "lstm" =>
                 %{
                   "input_kernel" => {_, _, _, _},
                   "hidden_kernel" => {_, _, _, _}
                 } = lstm_params
             } = init_fn.(input, %{})

      assert Map.has_key?(lstm_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.lstm(2, name: "lstm", use_bias: false, recurrent_initializer: :zeros)
        |> Axon.container()

      input = Nx.random_uniform({1, 2, 1})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "lstm" => %{
                 "input_kernel" => k,
                 "hidden_kernel" => h
               }
             } = params = init_fn.(input, %{})

      b = {Nx.tensor(0), Nx.tensor(0), Nx.tensor(0), Nx.tensor(0)}
      c = {zeros({1, 1, 2}), zeros({1, 1, 2})}

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(&Axon.Layers.lstm_cell/5, input, c, k, h, b)
      )
    end

    # TODO(seanmor5): https://github.com/elixir-nx/axon/issues/90
    # test "initializes with parameter policy" do
    # end
    # TODO(seanmor5): https://github.com/elixir-nx/axon/issues/90
    # test "computes forward pass with output policy" do
    # end
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

      input = Nx.random_uniform({1, 10, 3, 6, 6})

      assert {init_fn, _predict_fn} = Axon.build(model)

      assert %{
               "convlstm" => %{
                 "input_kernel" => {wi},
                 "hidden_kernel" => {wh},
                 "bias" => {b}
               }
             } = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 10, 3, 6, 6})

      out_channel_n = 4

      model1 =
        Axon.input("input", shape: input_shape)
        |> Axon.conv_lstm(out_channel_n, name: "convlstm", kernel_initializer: :zeros)
        |> Axon.container()

      assert {init_fn, _predict_fn} = Axon.build(model1)

      assert %{
               "convlstm" => %{
                 "input_kernel" => {wi},
                 "hidden_kernel" => {wh},
                 "bias" => {b}
               }
             } = init_fn.(input, %{})

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

      assert %{
               "convlstm" => %{
                 "input_kernel" => {wi},
                 "hidden_kernel" => {wh},
                 "bias" => {b}
               }
             } = init_fn.(input, %{})

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
        |> Nx.random_uniform(type: {:f, 32})

      init_carry = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "convlstm" => %{
                 "input_kernel" => {wi},
                 "hidden_kernel" => {wh},
                 "bias" => {b}
               }
             } = params = init_fn.(input, %{})

      k = {wi}
      h = {wh}
      b = {b}

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(
          &Axon.Layers.conv_lstm_cell/5,
          input,
          init_carry,
          k,
          h,
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
        |> Nx.random_uniform(type: {:f, 32})

      init_carry = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "convlstm" => %{
                 "input_kernel" => {wi},
                 "hidden_kernel" => {wh},
                 "bias" => {b}
               }
             } = params = init_fn.(input, %{})

      k = {wi}
      h = {wh}
      b = {b}

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.static_unroll(
          &Axon.Layers.conv_lstm_cell/5,
          input,
          init_carry,
          k,
          h,
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
        |> Nx.random_uniform(type: {:f, 32})

      init_carry1 = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      cell_fn1 = fn i, c, k, h, b ->
        Axon.Layers.conv_lstm_cell(
          i,
          c,
          k,
          h,
          b
        )
      end

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %{
               "convlstm" => %{
                 "input_kernel" => {wi},
                 "hidden_kernel" => {wh},
                 "bias" => {b}
               }
             } = params = init_fn.(input1, %{})

      k = {wi}
      h = {wh}
      b = {b}

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.dynamic_unroll(cell_fn1, input1, init_carry1, k, h, b)
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
        |> Nx.random_uniform(type: {:f, 32})

      init_carry2 = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      cell_fn2 = &Axon.Layers.conv_lstm_cell/5

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %{
               "convlstm" => %{
                 "input_kernel" => {wi},
                 "hidden_kernel" => {wh},
                 "bias" => {b}
               }
             } = params = init_fn.(input2, %{})

      k = {wi}
      h = {wh}
      b = {b}

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.static_unroll(cell_fn2, input2, init_carry2, k, h, b)
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
        |> Nx.random_uniform(type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model)

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        init_carry = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

        {_, carry} =
          Axon.Layers.dynamic_unroll(
            &Axon.Layers.conv_lstm_cell/5,
            inp,
            init_carry,
            ei,
            eh,
            eb
          )

        Axon.Layers.dynamic_unroll(&Axon.Layers.conv_lstm_cell/5, inp, carry, di, dh, db)
      end

      assert %{
               "encode" => %{
                 "input_kernel" => {ei},
                 "hidden_kernel" => {eh},
                 "bias" => {eb}
               },
               "decode" => %{
                 "input_kernel" => {di},
                 "hidden_kernel" => {dh},
                 "bias" => {db}
               }
             } = params = init_fn.(input, %{})

      enc = {{ei}, {eh}, {eb}}
      dec = {{di}, {dh}, {db}}

      assert_equal(predict_fn.(params, input), equiv_fn.(input, enc, dec))
    end

    # TODO
    # test "returns zero gradient for frozen parameters" do
    # end

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
        |> Nx.random_uniform(type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "convlstm" => %{
                 "input_kernel" => k,
                 "hidden_kernel" => h
               }
             } = params = init_fn.(input, %{})

      b = {Nx.broadcast(0, 4 * out_channel_n)}

      c = {zeros(hidden_shape_real), zeros(hidden_shape_real)}

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(&Axon.Layers.conv_lstm_cell/5, input, c, k, h, b)
      )
    end
  end

  describe "gru" do
    test "initializes in default case" do
      model =
        Axon.input("input", shape: {nil, 32, 10}) |> Axon.gru(64, name: "gru") |> Axon.container()

      input = Nx.random_uniform({1, 32, 10})

      assert {init_fn, _} = Axon.build(model)

      assert %{
               "gru" => %{
                 "input_kernel" => {wir, wiz, win},
                 "hidden_kernel" => {whr, whz, whn},
                 "bias" => {br, bz, bhn, bin}
               }
             } = init_fn.(input, %{})

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
      input = Nx.random_uniform({1, 32, 10})

      model1 =
        Axon.input("input", shape: {nil, 32, 10})
        |> Axon.gru(64, name: "gru", kernel_initializer: :zeros)
        |> Axon.container()

      assert {init_fn, _} = Axon.build(model1)

      assert %{
               "gru" => %{
                 "input_kernel" => {wir, wiz, win},
                 "hidden_kernel" => {whr, whz, whn},
                 "bias" => {br, bz, bhn, bin}
               }
             } = init_fn.(input, %{})

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

      assert %{
               "gru" => %{
                 "input_kernel" => {wir, wiz, win},
                 "hidden_kernel" => {whr, whz, whn},
                 "bias" => {br, bz, bhn, bin}
               }
             } = init_fn.(input, %{})

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

      input = Nx.random_uniform({1, 8, 2})
      carry = {zeros({1, 1, 2})}

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "gru" => %{
                 "input_kernel" => k,
                 "hidden_kernel" => h,
                 "bias" => b
               }
             } = params = init_fn.(input, %{})

      assert_equal(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(&Axon.Layers.gru_cell/5, input, carry, k, h, b)
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

      input1 = Nx.random_uniform({1, 8, 2})
      carry1 = {zeros({1, 1, 2})}

      cell_fn1 = fn i, c, k, h, b ->
        Axon.Layers.gru_cell(
          i,
          c,
          k,
          h,
          b,
          &Axon.Activations.relu/1,
          &Axon.Activations.sigmoid/1
        )
      end

      assert {init_fn, predict_fn} = Axon.build(model1)

      assert %{
               "gru" => %{
                 "input_kernel" => {wir, wiz, win},
                 "hidden_kernel" => {whr, whz, whn},
                 "bias" => {br, bz, bhn, bin}
               }
             } = params = init_fn.(input1, %{})

      k = {wir, wiz, win}
      h = {whr, whz, whn}
      b = {br, bz, bin, bhn}

      assert_equal(
        predict_fn.(params, input1),
        Axon.Layers.dynamic_unroll(cell_fn1, input1, carry1, k, h, b)
      )

      model2 =
        Axon.input("input", shape: {nil, 8, 2})
        |> Axon.gru(2, name: "gru", recurrent_initializer: :zeros, unroll: :static)
        |> Axon.container()

      input2 = Nx.random_uniform({1, 8, 2})
      carry2 = {zeros({1, 1, 2})}

      assert {init_fn, predict_fn} = Axon.build(model2)

      assert %{
               "gru" => %{
                 "input_kernel" => {wir, wiz, win},
                 "hidden_kernel" => {whr, whz, whn},
                 "bias" => {br, bz, bhn, bin}
               }
             } = params = init_fn.(input2, %{})

      k = {wir, wiz, win}
      h = {whr, whz, whn}
      b = {br, bz, bin, bhn}

      assert_equal(
        predict_fn.(params, input2),
        Axon.Layers.static_unroll(&Axon.Layers.gru_cell/5, input2, carry2, k, h, b)
      )
    end

    test "computes forward pass with hidden state" do
      seq = Axon.input("input", shape: {nil, 8, 2})
      {_, carry} = Axon.gru(seq, 2, name: "encode", recurrent_initializer: :zeros)
      model = Axon.gru(seq, carry, 2, name: "decode") |> Axon.container()
      input = Nx.random_uniform({1, 8, 2})
      carry = {zeros({1, 1, 2})}

      equiv_fn = fn inp, enc, dec ->
        {ei, eh, eb} = enc
        {di, dh, db} = dec

        {_, carry} = Axon.Layers.dynamic_unroll(&Axon.Layers.gru_cell/5, inp, carry, ei, eh, eb)

        Axon.Layers.dynamic_unroll(&Axon.Layers.gru_cell/5, inp, carry, di, dh, db)
      end

      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "encode" => %{
                 "input_kernel" => {eir, eiz, ein},
                 "hidden_kernel" => {ehr, ehz, ehn},
                 "bias" => {ebr, ebz, ebhn, ebin}
               },
               "decode" => %{
                 "input_kernel" => {dir, diz, din},
                 "hidden_kernel" => {dhr, dhz, dhn},
                 "bias" => {dbr, dbz, dbhn, dbin}
               }
             } = params = init_fn.(input, %{})

      enc = {{eir, eiz, ein}, {ehr, ehz, ehn}, {ebr, ebz, ebin, ebhn}}
      dec = {{dir, diz, din}, {dhr, dhz, dhn}, {dbr, dbz, dbin, dbhn}}

      assert_equal(predict_fn.(params, input), equiv_fn.(input, enc, dec))
    end

    test "initializes with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.gru(2, name: "gru", use_bias: false)
        |> Axon.container()

      input = Nx.random_uniform({1, 2, 1})

      assert {init_fn, _} = Axon.build(model)

      assert %{
               "gru" =>
                 %{
                   "input_kernel" => {_, _, _},
                   "hidden_kernel" => {_, _, _}
                 } = gru_params
             } = init_fn.(input, %{})

      assert Map.has_key?(gru_params, "bias") == false
    end

    test "computes forward pass with use_bias false" do
      model =
        Axon.input("input", shape: {nil, 2, 1})
        |> Axon.gru(2, name: "gru", use_bias: false, recurrent_initializer: :zeros)
        |> Axon.container()

      input = Nx.random_uniform({1, 2, 1})
      assert {init_fn, predict_fn} = Axon.build(model)

      assert %{
               "gru" => %{
                 "input_kernel" => k,
                 "hidden_kernel" => h
               }
             } = params = init_fn.(input, %{})

      b = {Nx.tensor(0), Nx.tensor(0), Nx.tensor(0), Nx.tensor(0)}
      c = {zeros({1, 1, 2})}

      assert_all_close(
        predict_fn.(params, input),
        Axon.Layers.dynamic_unroll(&Axon.Layers.gru_cell/5, input, c, k, h, b)
      )
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
        model =
          apply(Axon, op, [
            Axon.input("input_0", shape: {nil, 32}),
            Axon.input("input_1", shape: {nil, 32})
          ])

        input = %{
          "input_0" => Nx.random_uniform({1, 32}),
          "input_1" => Nx.random_uniform({1, 32})
        }

        assert {init_fn, _} = Axon.build(model)
        assert %{} == init_fn.(input, %{})
      end
    end

    test "computes forward pass with default options" do
      for op <- @binary_layers do
        model1 =
          apply(Axon, op, [
            Axon.input("input_0", shape: {nil, 32}),
            Axon.input("input_1", shape: {nil, 32})
          ])

        input1_1 = Nx.random_uniform({1, 32})
        input1_2 = Nx.random_uniform({1, 32})
        assert {_, predict_fn} = Axon.build(model1)

        assert_all_close(
          predict_fn.(%{}, %{"input_0" => input1_1, "input_1" => input1_2}),
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

        input2_1 = Nx.random_uniform({1, 32})
        input2_2 = Nx.random_uniform({1, 32})
        input2_3 = Nx.random_uniform({1, 32})
        assert {_, predict_fn} = Axon.build(model2)

        assert_all_close(
          predict_fn.(%{}, %{"input_0" => input2_1, "input_1" => input2_2, "input_2" => input2_3}),
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
          "input_0" => Nx.random_uniform({1, 32}),
          "input_1" => Nx.random_uniform({1, 32})
        }

        assert {_, predict_fn} = Axon.build(mp_model)
        assert Nx.type(predict_fn.(%{}, input)) == {:bf, 16}
      end
    end

    test "computes forward pass with broadcasting" do
      inp1 = Nx.random_uniform({1, 1})
      inp2 = Nx.random_uniform({1, 2})

      for op <- @binary_layers do
        model =
          apply(Axon, op, [
            Axon.input("input_0", shape: {nil, 1}),
            Axon.input("input_1", shape: {nil, 2})
          ])

        assert {_, predict_fn} = Axon.build(model)

        assert_equal(
          predict_fn.(%{}, %{"input_0" => inp1, "input_1" => inp2}),
          apply(Nx, op, [inp1, inp2])
        )
      end
    end

    test "raises on bad shapes" do
      for op <- @binary_layers do
        assert_raise Axon.CompileError, ~r/cannot broadcast tensor/, fn ->
          inp1 = Nx.random_uniform({1, 32})
          inp2 = Nx.random_uniform({1, 64})

          model =
            apply(Axon, op, [
              [Axon.input("input_0", shape: {nil, 32}), Axon.input("input_1", shape: {nil, 64})]
            ])

          Axon.predict(model, %{}, %{"input_0" => inp1, "input_1" => inp2})
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

      input = %{"input_0" => Nx.random_uniform({1, 32}), "input_1" => Nx.random_uniform({1, 32})}

      assert {init_fn, _} = Axon.build(model)
      assert %{} == init_fn.(input, %{})
    end

    test "computes forward pass with default options" do
      model1 =
        Axon.concatenate(
          Axon.input("input_0", shape: {nil, 32}),
          Axon.input("input_1", shape: {nil, 32})
        )

      input1_1 = Nx.random_uniform({1, 32})
      input1_2 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.build(model1)

      assert_equal(
        predict_fn.(%{}, %{"input_0" => input1_1, "input_1" => input1_2}),
        Nx.concatenate([input1_1, input1_2], axis: 1)
      )

      model2 =
        Axon.concatenate([
          Axon.input("input_0", shape: {nil, 32}),
          Axon.input("input_1", shape: {nil, 32}),
          Axon.input("input_2", shape: {nil, 32})
        ])

      input2_1 = Nx.random_uniform({1, 32})
      input2_2 = Nx.random_uniform({1, 32})
      input2_3 = Nx.random_uniform({1, 32})

      assert {_, predict_fn} = Axon.build(model2)

      assert_equal(
        predict_fn.(%{}, %{"input_0" => input2_1, "input_1" => input2_2, "input_2" => input2_3}),
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

      input1_1 = Nx.random_uniform({1, 1, 32})
      input1_2 = Nx.random_uniform({1, 1, 32})

      assert {_, predict_fn} = Axon.build(model1)

      assert_equal(
        predict_fn.(%{}, %{"input_0" => input1_1, "input_1" => input1_2}),
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
      input1_1 = Nx.random_uniform({1, 1, 32})
      input1_2 = Nx.random_uniform({1, 1, 32})

      assert {_, predict_fn} = Axon.build(mp_model)

      assert Nx.type(predict_fn.(%{}, %{"input_0" => input1_1, "input_1" => input1_2})) ==
               {:bf, 16}
    end
  end

  describe "pad" do
    test "initializes with no params" do
      model = Axon.input("input", shape: {nil, 3, 3}) |> Axon.pad([{1, 0}])
      input = Nx.random_uniform({1, 3, 3})

      assert {init_fn, _} = Axon.build(model)
      assert %{} == init_fn.(input, %{})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input("input", shape: {nil, 3, 3}) |> Axon.pad([{1, 0}])
      input1 = Nx.random_uniform({1, 3, 3})

      assert {_, predict_fn} = Axon.build(model1)

      assert_equal(
        predict_fn.(%{}, input1),
        Nx.pad(input1, 0, [{0, 0, 0}, {1, 0, 0}, {0, 0, 0}])
      )

      model2 = Axon.input("input", shape: {nil, 3, 3, 3}) |> Axon.pad([{0, 1}, {0, 1}])
      input2 = Nx.random_uniform({1, 3, 3, 3})

      assert {_, predict_fn} = Axon.build(model2)

      assert_equal(
        predict_fn.(%{}, input2),
        Nx.pad(input2, 0, [{0, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 0}])
      )

      model3 = Axon.input("input", shape: {nil, 3, 3, 3, 3}) |> Axon.pad([{0, 1}, {0, 1}, {1, 0}])
      input3 = Nx.random_uniform({1, 3, 3, 3, 3})

      assert {_, predict_fn} = Axon.build(model3)

      assert_equal(
        predict_fn.(%{}, input3),
        Nx.pad(input3, 0, [{0, 0, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 0}])
      )
    end

    test "computes forward pass with custom options" do
      model = Axon.input("input", shape: {nil, 3, 3}) |> Axon.pad([{1, 0}], 2)
      input = Nx.random_uniform({1, 3, 3})

      assert {_, predict_fn} = Axon.build(model)

      assert_equal(
        predict_fn.(%{}, input),
        Nx.pad(input, 2, [{0, 0, 0}, {1, 0, 0}, {0, 0, 0}])
      )
    end

    test "computes forward pass with output policy" do
      model = Axon.input("input", shape: {nil, 3, 3}) |> Axon.pad([{1, 0}])
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)
      input = Nx.random_uniform({1, 3, 3})

      assert {_, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(%{}, input)) == {:bf, 16}
    end
  end

  describe "nx" do
    test "computes special nx functions" do
      model = Axon.input("input", shape: {nil, 10}) |> Axon.nx(&Nx.sin/1)
      input = Nx.random_uniform({1, 10})

      assert {_, predict_fn} = Axon.build(model)
      assert_all_close(predict_fn.(%{}, input), Nx.sin(input))
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
      assert %{} == init_fn.(input, %{})
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
      assert_equal(predict_fn.(%{}, input_1), Axon.Activations.relu(input_1))
      assert_equal(predict_fn.(%{}, input_2), Axon.Activations.sigmoid(input_2))
    end

    test "computes forward pass with output policy" do
      inp = Axon.input("input", shape: {nil, 1, 32})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.all(x) end
      model1 = Axon.cond(inp, cond_fn, on_true, on_false)
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model1, policy)

      input1_1 = Nx.random_uniform({1, 1, 32})

      assert {_, predict_fn} = Axon.build(mp_model)
      assert Nx.type(predict_fn.(%{}, input1_1)) == {:bf, 16}
    end

    test "raises on bad condition" do
      inp = Axon.input("input", shape: {nil, 1, 10})
      on_true = Axon.relu(inp)
      on_false = Axon.sigmoid(inp)
      cond_fn = fn x -> Nx.equal(x, 1) end

      model = Axon.cond(inp, cond_fn, on_true, on_false)

      assert_raise Axon.CompileError, ~r/cond_fn must return a scalar/, fn ->
        {_, predict_fn} = Axon.build(model)
        predict_fn.(%{}, Nx.random_uniform({1, 1, 10}))
      end
    end
  end

  describe "split" do
    test "initializes with no parameters" do
      model = Axon.input("input", shape: {nil, 10}) |> Axon.split(5) |> Axon.container()

      input = Nx.random_uniform({1, 10})

      assert {init_fn, _} = Axon.build(model)
      assert init_fn.(input, %{}) == %{}
    end

    test "computes forward pass with default options" do
      model = Axon.input("input", shape: {nil, 10}) |> Axon.split(5) |> Axon.container()
      input = Nx.iota({1, 10}, type: {:f, 32})

      assert {_, predict_fn} = Axon.build(model)

      assert_equal(
        predict_fn.(%{}, input),
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
      init_fn.(Nx.tensor([[1.0]]), %{})
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

      Axon.predict(model, %{}, inp)

      assert_receive {pre_relu, :from_relu}
      assert_equal(pre_relu, inp)
    end

    test "forward hook", config do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.attach_hook(fn x -> send(config.test, {x, :from_input}) end, on: :forward)
        |> Axon.relu()

      inp = Nx.tensor([[1.0]])

      Axon.predict(model, %{}, inp)

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

      inp = Nx.random_uniform({1, 1})
      params = init_fn.(inp, %{})

      axon_loss = fn inp, params -> Nx.sum(predict_fn.(params, inp)) end

      loss = fn inp, params ->
        inp
        |> Axon.Layers.dense(params["dense_0"]["kernel"], params["dense_0"]["bias"])
        |> Axon.Activations.relu()
        |> Axon.Activations.sigmoid()
        |> Nx.sum()
      end

      axon_grad_params =
        Nx.Defn.jit(fn inp, x -> Nx.Defn.grad(x, &axon_loss.(inp, &1)) end).(inp, params)

      actual_grad_params =
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

      inp = Nx.random_uniform({1, 2})

      assert %{"dense_0" => dense_0_params, "dense_1" => dense_1_params} = init_fn.(inp, %{})

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

      assert %{"lstm_0" => lstm_0_params, "lstm_1" => lstm_1_params} = init_fn.(inp, %{})

      assert %{
               "input_kernel" => {wii_0, wif_0, wig_0, wio_0},
               "hidden_kernel" => {whi_0, whf_0, whg_0, who_0},
               "bias" => {bi_0, bf_0, bg_0, bo_0}
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
               "input_kernel" => {wii_1, wif_1, wig_1, wio_1},
               "hidden_kernel" => {whi_1, whf_1, whg_1, who_1},
               "bias" => {bi_1, bf_1, bg_1, bo_1}
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
      inp = Nx.random_uniform({1, 1})

      {init_fn, _} = Axon.build(model)
      assert Enum.empty?(init_fn.(inp, %{}))
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

      inp = Nx.random_uniform({1, 1})

      assert %{"layer_0" => %{"kernel" => kernel}} = init_fn.(inp, %{})

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

      inp = Nx.random_uniform({1, 1})

      assert %{"custom_0" => %{"kernel" => _}, "custom_1" => %{"kernel" => _}} =
               init_fn.(inp, %{})
    end

    test "computes forward pass with parameters" do
      input = Axon.input("input_0", shape: {nil, 1})
      kernel_param = Axon.param("kernel", fn shape -> shape end)

      model =
        Axon.layer(fn x, kernel, _opts -> Nx.multiply(x, kernel) end, [input, kernel_param],
          name: "layer_0"
        )

      {init_fn, _} = Axon.build(model)

      input = Nx.random_uniform({1, 1})

      assert %{"layer_0" => %{"kernel" => kernel}} = params = init_fn.(input, %{})

      assert_equal(Axon.predict(model, params, input), Nx.multiply(input, kernel))
    end

    defn layer_with_options(x, kernel, opts \\ []) do
      transform({x, kernel, opts}, fn {x, kernel, opts} ->
        if opts[:add] do
          Nx.add(x, kernel)
        else
          Nx.multiply(x, kernel)
        end
      end)
    end

    test "computes forward pass with options" do
      kernel_param = Axon.param("kernel", fn shape -> shape end)

      input = Nx.random_uniform({1, 1})

      model1 =
        Axon.layer(&layer_with_options/3, [Axon.input("input_0", shape: {nil, 1}), kernel_param],
          name: "add",
          add: true
        )

      {init_fn, _} = Axon.build(model1)

      assert %{"add" => %{"kernel" => kernel}} = params = init_fn.(input, %{})

      assert_equal(Axon.predict(model1, params, input), Nx.add(input, kernel))

      model2 =
        Axon.layer(&layer_with_options/3, [Axon.input("input_0", shape: {nil, 1}), kernel_param],
          name: "multiply",
          add: false
        )

      {init_fn, _} = Axon.build(model2)

      assert %{"multiply" => %{"kernel" => kernel}} = params = init_fn.(input, %{})

      assert_equal(Axon.predict(model2, params, input), Nx.multiply(input, kernel))
    end
  end

  describe "namespace" do
    test "initializes correctly with single namespace" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("model")

      {init_fn, _} = Axon.build(model)

      inp = Nx.random_uniform({1, 1})

      assert %{"model" => %{"dense_0" => %{"kernel" => k, "bias" => b}}} = init_fn.(inp, %{})

      assert Nx.shape(k) == {1, 2}
      assert Nx.type(k) == {:f, 32}
      assert Nx.shape(b) == {2}
      assert Nx.type(b) == {:f, 32}
    end

    test "initializes correctly with nested namespace" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(2)
        |> Axon.namespace("model")
        |> Axon.namespace("nested")

      {init_fn, _} = Axon.build(model)

      inp = Nx.random_uniform({1, 1})

      assert %{"nested" => %{"model" => %{"dense_0" => %{"kernel" => k, "bias" => b}}}} =
               init_fn.(inp, %{})

      assert Nx.shape(k) == {1, 2}
      assert Nx.type(k) == {:f, 32}
      assert Nx.shape(b) == {2}
      assert Nx.type(b) == {:f, 32}
    end

    test "initializes correclty with single namespace no params" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.namespace("model")

      {init_fn, _} = Axon.build(model)

      inp = Nx.random_uniform({1, 1})

      assert Enum.empty?(init_fn.(inp, %{}))
    end

    test "initializes correctly with nested namespace no params" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.namespace("model")
        |> Axon.namespace("nested")

      {init_fn, _} = Axon.build(model)

      inp = Nx.random_uniform({1, 1})

      assert Enum.empty?(init_fn.(inp, %{}))
    end

    test "initializes correctly with multiple single namespaces" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("y")

      inp = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 1})}

      model = Axon.add(x, y)

      {init_fn, _} = Axon.build(model)

      assert %{
               "x" => %{"dense_0" => %{"kernel" => k1, "bias" => b1}},
               "y" => %{"dense_0" => %{"kernel" => k2, "bias" => b2}}
             } = init_fn.(inp, %{})

      assert Nx.shape(k1) == {1, 2}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(b1) == {2}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(k2) == {1, 2}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(b2) == {2}
      assert Nx.type(b2) == {:f, 32}
    end

    test "initializes correctly single and nested namespace" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")

      z =
        Axon.input("input_1", shape: {nil, 1})
        |> Axon.dense(2)
        |> Axon.namespace("y")
        |> Axon.namespace("z")

      inp = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 1})}

      model = Axon.add(x, z)

      {init_fn, _} = Axon.build(model)

      assert %{
               "x" => %{"dense_0" => %{"kernel" => k1, "bias" => b1}},
               "z" => %{"y" => %{"dense_0" => %{"kernel" => k2, "bias" => b2}}}
             } = init_fn.(inp, %{})

      assert Nx.shape(k1) == {1, 2}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(b1) == {2}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(k2) == {1, 2}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(b2) == {2}
      assert Nx.type(b2) == {:f, 32}
    end

    test "initializes correctly with single namespace and no namespace" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2)

      inp = %{"input_0" => Nx.random_uniform({1, 1}), "input_1" => Nx.random_uniform({1, 1})}

      model = Axon.add(x, y)

      {init_fn, _} = Axon.build(model)

      assert %{
               "x" => %{"dense_0" => %{"kernel" => k1, "bias" => b1}},
               "dense_0" => %{"kernel" => k2, "bias" => b2}
             } = init_fn.(inp, %{})

      assert Nx.shape(k1) == {1, 2}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(b1) == {2}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(k2) == {1, 2}
      assert Nx.type(k2) == {:f, 32}
      assert Nx.shape(b2) == {2}
      assert Nx.type(b2) == {:f, 32}
    end

    test "initializes correctly reusing namespace" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")

      inp = Nx.random_uniform({1, 1})
      model = Axon.add(x, x)

      {init_fn, _} = Axon.build(model)

      assert %{"x" => %{"dense_0" => %{"kernel" => k, "bias" => b}}} = init_fn.(inp, %{})

      assert Nx.shape(k) == {1, 2}
      assert Nx.type(k) == {:f, 32}
      assert Nx.shape(b) == {2}
      assert Nx.type(b) == {:f, 32}
    end

    test "initializes correctly with layers after namespace, re-using params" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")
      model = Axon.dense(x, 2)

      {init_fn, _} = Axon.build(model)

      assert %{"x" => x_params_1} = init_params = init_fn.(Nx.tensor([[1]]), %{})

      assert %{"x" => x_params_2, "dense_0" => _} = init_fn.(Nx.tensor([[1]]), init_params)

      assert_equal(x_params_1, x_params_2)
    end

    # TODO: I actually don't know what the correct behavior is here?
    # test "initializes correctly re-using part of inner namespace" do
    #   inner = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2)
    #   x = Axon.namespace(inner, "x")

    #   model = Axon.add(inner, x)

    #   assert %{"x" => %{"dense_0" => %{"kernel" => k, "bias" => b}}} = Axon.init(model)

    #   assert Nx.shape(k) == {1, 2}
    #   assert Nx.type(k) == {:f, 32}
    #   assert Nx.shape(b) == {2}
    #   assert Nx.type(b) == {:f, 32}
    # end

    test "predicts correctly with single namespace" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("model")
      input = Nx.random_uniform({1, 1})

      {init_fn, _} = Axon.build(model)

      assert %{"model" => %{"dense_0" => %{"kernel" => k, "bias" => b}}} =
               params = init_fn.(input, %{})

      assert_equal(Axon.predict(model, params, input), Axon.Layers.dense(input, k, b))
    end

    test "predicts correctly with single namespace no parameters" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.namespace("model")
      input = Nx.random_uniform({1, 1})

      assert_equal(Axon.predict(model, %{}, input), input)
    end

    test "predicts correctly with nested namespace" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(2)
        |> Axon.namespace("model")
        |> Axon.namespace("nested")

      {init_fn, _} = Axon.build(model)

      input = Nx.random_uniform({1, 1})

      assert %{"nested" => %{"model" => %{"dense_0" => %{"kernel" => k, "bias" => b}}}} =
               params = init_fn.(input, %{})

      assert_equal(Axon.predict(model, params, input), Axon.Layers.dense(input, k, b))
    end

    test "predicts correctly with nested namespace and no params" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.namespace("model")
        |> Axon.namespace("nested")

      input = Nx.random_uniform({1, 1})

      assert_equal(Axon.predict(model, %{}, input), input)
    end

    test "predicts correctly with multiple single namespace" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("y")

      model = Axon.add(x, y)

      {init_fn, _} = Axon.build(model)

      input_0 = Nx.random_uniform({1, 1})
      input_1 = Nx.random_uniform({1, 1})
      inputs = %{"input_0" => input_0, "input_1" => input_1}

      assert %{
               "x" => %{"dense_0" => %{"kernel" => k1, "bias" => b1}},
               "y" => %{"dense_0" => %{"kernel" => k2, "bias" => b2}}
             } = params = init_fn.(inputs, %{})

      expected = Nx.add(Axon.Layers.dense(input_0, k1, b1), Axon.Layers.dense(input_1, k2, b2))
      assert_equal(Axon.predict(model, params, inputs), expected)
    end

    test "predicts correctly with single and nested namespace" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")

      z =
        Axon.input("input_1", shape: {nil, 1})
        |> Axon.dense(2)
        |> Axon.namespace("y")
        |> Axon.namespace("z")

      model = Axon.add(x, z)

      {init_fn, _} = Axon.build(model)

      input_0 = Nx.random_uniform({1, 1})
      input_1 = Nx.random_uniform({1, 1})
      inputs = %{"input_0" => input_0, "input_1" => input_1}

      assert %{
               "x" => %{"dense_0" => %{"kernel" => k1, "bias" => b1}},
               "z" => %{"y" => %{"dense_0" => %{"kernel" => k2, "bias" => b2}}}
             } = params = init_fn.(inputs, %{})

      expected = Nx.add(Axon.Layers.dense(input_0, k1, b1), Axon.Layers.dense(input_1, k2, b2))
      assert_equal(Axon.predict(model, params, inputs), expected)
    end

    test "predicts correctly with single and no namespace" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2)

      model = Axon.add(x, y)

      {init_fn, _} = Axon.build(model)

      input_0 = Nx.random_uniform({1, 1})
      input_1 = Nx.random_uniform({1, 1})
      inputs = %{"input_0" => input_0, "input_1" => input_1}

      assert %{
               "x" => %{"dense_0" => %{"kernel" => k1, "bias" => b1}},
               "dense_0" => %{"kernel" => k2, "bias" => b2}
             } = params = init_fn.(inputs, %{})

      expected = Nx.add(Axon.Layers.dense(input_0, k1, b1), Axon.Layers.dense(input_1, k2, b2))
      assert_equal(Axon.predict(model, params, inputs), expected)
    end

    test "predicts correctly re-using namespace" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")

      model = Axon.add(x, x)
      {init_fn, _} = Axon.build(model)
      input = Nx.random_uniform({1, 1})

      assert %{"x" => %{"dense_0" => %{"kernel" => k, "bias" => b}}} =
               params = init_fn.(input, %{})

      expected = Nx.add(Axon.Layers.dense(input, k, b), Axon.Layers.dense(input, k, b))
      assert_equal(Axon.predict(model, params, input), expected)
    end

    # TODO: I actually don't know what the correct behavior is here?
    # test "predicts correctly re-using part of inner namespace" do
    #   inner = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2)
    #   x = Axon.namespace(inner, "x")

    #   model = Axon.add(inner, x)

    #   input = Nx.random_uniform({1, 1})
    #   assert %{"x" => %{"dense_0" => %{"kernel" => k, "bias" => b}}} = params = Axon.init(model)

    #   expected = Nx.add(Axon.Layers.dense(input, k, b), Axon.Layers.dense(input, k, b))
    #   assert_equal(Axon.predict(model, params, input), expected)
    # end
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

      assert %{"dense_0" => %{"kernel" => k, "bias" => b}} = init_fn.(inp, %{})
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

      assert %{"dense_0" => dense_0, "dense_1" => dense_1} = init_fn.(inp, %{})
      assert %{"kernel" => k0, "bias" => _} = dense_0
      assert %{"kernel" => k1, "bias" => _} = dense_1

      assert_not_equal(k0, k1)
    end
  end

  describe "initialize from fixed model" do
    test "initializes entire model from start point" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2)

      {init_fn, _} = Axon.build(model)

      inp = Nx.tensor([[1.0]])

      params_1 = init_fn.(inp, %{})
      params_2 = init_fn.(inp, params_1)

      assert_equal(params_1, params_2)
    end

    test "initializes partial model from start point" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("y")
      model = Axon.add(x, y)

      input = %{"input_0" => Nx.tensor([[1.0]]), "input_1" => Nx.tensor([[2.0]])}

      {init_fn, _} = Axon.build(model)

      %{"x" => x_params_1} = init_fn.(input, %{})

      assert %{"x" => x_params_2, "y" => y_params_1} = init_fn.(input, %{"x" => x_params_1})
      assert_equal(x_params_1, x_params_2)

      # This exercises that a namespace will always have the same
      # parameter names regardless of where it appears in a model
      assert %{"y" => y_params_2} = init_fn.(input, %{"y" => y_params_1})
      assert_equal(y_params_1, y_params_2)
    end

    test "raises when unknown parameters are passed" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2, name: "dense_1")

      {init_fn, _} = Axon.build(model)

      inp = Nx.random_uniform({1, 1})

      assert_raise ArgumentError,
                   ~s{found unexpected key in the initial parameters map: "dense_2"},
                   fn ->
                     init_fn.(inp, %{"dense_2" => %{"kernel" => Nx.tensor([[2.0]])}})
                   end
    end
  end

  describe "containers" do
    test "allows accessors with custom layers" do
      input1 = Nx.random_uniform({1, 1})
      input2 = Nx.random_uniform({1, 2})
      inputs = %{"input_0" => input1, "input_1" => input2}

      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 2})
      tuple_model = Axon.container({inp1, inp2})
      first_elem = Axon.nx(tuple_model, &elem(&1, 0))
      second_elem = Axon.nx(tuple_model, &elem(&1, 1))

      assert_equal(Axon.predict(first_elem, %{}, inputs), input1)
      assert_equal(Axon.predict(second_elem, %{}, inputs), input2)

      map_model = Axon.container(%{foo: inp1, bar: inp2})
      foo_elem = Axon.nx(map_model, & &1.foo)
      bar_elem = Axon.nx(map_model, & &1.bar)

      assert_equal(Axon.predict(foo_elem, %{}, inputs), input1)
      assert_equal(Axon.predict(bar_elem, %{}, inputs), input2)

      nested_model = Axon.container({{inp1}, %{foo: {inp2}}})
      first_elem = Axon.nx(nested_model, &elem(elem(&1, 0), 0))
      second_elem = Axon.nx(nested_model, &elem(elem(&1, 1).foo, 0))

      assert_equal(Axon.predict(first_elem, %{}, inputs), input1)
      assert_equal(Axon.predict(second_elem, %{}, inputs), input2)
    end
  end

  describe "edge cases" do
    test "raises clean error on missing parameter" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      input = Nx.tensor([[1.0]])

      assert_raise ArgumentError, ~r/parameter "kernel" for layer:/, fn ->
        Axon.predict(model, %{}, input)
      end
    end

    test "initializes a non-linear model" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2, name: "dense_0")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2, name: "dense_1")
      model = Axon.add(x, y)

      {init_fn, _} = Axon.build(model)

      input = %{"input_0" => Nx.tensor([[1.0]]), "input_1" => Nx.tensor([[2.0]])}

      assert %{"dense_0" => _, "dense_1" => _} = init_fn.(input, %{})
    end
  end

  describe "instrumentation" do
    @describetag :capture_log

    test "predict logs debug utilities when debug true" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)

      {init_fn, _} = Axon.build(model, debug: true)

      assert capture_log(fn ->
               init_fn.(Nx.tensor([[1.0]]), %{})
             end) =~ "Axon finished init"
    end

    test "init logs debug utilities when debug true" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      input = Nx.tensor([[1.0]])

      {init_fn, _} = Axon.build(model)

      params = init_fn.(Nx.template({1, 1}, {:f, 32}), %{})

      assert capture_log(fn ->
               Axon.predict(model, params, input, debug: true)
             end) =~ "Axon finished predict"
    end

    test "compile logs debug utilities when debug true" do
      model = Axon.input("input", shape: {nil, 1}) |> Axon.dense(2)
      input = Nx.tensor([[1.0]])

      {init_fn, predict_fn} = Axon.build(model, debug: true)

      assert capture_log(fn ->
               init_fn.(input, %{})
             end) =~ "Axon finished init"

      params = init_fn.(input, %{})

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
        Axon.predict(model, %{}, %{"lazy_container" => input}),
        Nx.tensor([[1.0, 3.0]])
      )
    end
  end
end
