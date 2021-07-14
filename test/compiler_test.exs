defmodule CompilerTest do
  use ExUnit.Case, async: true

  alias Axon.MixedPrecision, as: AMP

  describe "input" do
    test "single input, single output" do
      model = Axon.input({nil, 32})
      input = Nx.random_uniform({1, 32}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{} = init_fn.()
      assert predict_fn.(%{}, input) == input
    end

    test "multi-input, multi-output" do
      model = {Axon.input({nil, 32}), Axon.input({nil, 16})}

      input =
        {Nx.random_uniform({1, 32}, type: {:f, 32}), Nx.random_uniform({1, 16}, type: {:f, 32})}

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{} = init_fn.()
      assert predict_fn.(%{}, input) == input
    end

    # test "multi-input, multi-output nested" do
    #   model = {Axon.input({nil, 32}), {Axon.input({nil, 16})}}
    #   input = {Nx.random_uniform({1, 32}, type: {:f, 32}), {Nx.random_uniform({1, 16}, type: {:f, 32})}}

    #   assert {init_fn, predict_fn} = Axon.compile(model)
    #   assert %{} = init_fn.()
    #   assert predict_fn.(%{}, input) == input
    # end
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
        model = Axon.input({nil, 32}) |> Axon.activation(activation)
        input = Nx.random_uniform({1, 32})

        assert {_init_fn, predict_fn} = Axon.compile(model)
        assert predict_fn.(%{}, input) == apply(Axon.Activations, activation, [input])
      end
    end

    # test "computes forward pass with custom options" do
    # end

    test "computes forward pass with output policy" do
      for activation <- @activation_layers do
        model = Axon.input({nil, 32}) |> Axon.activation(activation)
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 32}))) == {:bf, 16}
      end
    end
  end

  describe "dense" do
    test "initializes in default case" do
      model = Axon.input({nil, 32}) |> Axon.dense(1, name: "dense")

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{"dense_kernel" => kernel, "dense_bias" => bias} = init_fn.()
      assert Nx.shape(kernel) == {32, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 = Axon.input({nil, 32}) |> Axon.dense(1, name: "dense", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"dense_kernel" => kernel, "dense_bias" => bias} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {32, 1})
      assert Nx.shape(bias) == {1}
      assert Nx.type(bias) == {:f, 32}

      model2 = Axon.input({nil, 32}) |> Axon.dense(1, name: "dense", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"dense_kernel" => kernel, "dense_bias" => bias} = init_fn.()
      assert Nx.shape(kernel) == {32, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {1})
    end

    test "computes forward pass" do
      model = Axon.input({nil, 1}) |> Axon.dense(1, name: "dense", kernel_initializer: :identity)
      input = Nx.iota({1, 1}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"dense_kernel" => kernel, "dense_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input) == Axon.Layers.dense(input, kernel, bias)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 32})
        |> Axon.dense(1, name: "dense")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"dense_kernel" => kernel_grad, "dense_bias" => bias_grad} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 32})])

      assert kernel_grad == Nx.broadcast(0.0, {32, 1})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 32}) |> Axon.dense(1, name: "dense")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"dense_kernel" => kernel, "dense_bias" => bias} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 32}) |> Axon.dense(1, name: "dense")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 32}))) == {:bf, 16}
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

    test "computes forward pass with output policy" do
      for pool <- @pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 1, 32})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
      end
    end
  end

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool]

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
        model1 = apply(Axon, pool, [Axon.input({nil, 1, 32})])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        assert predict_fn.(%{}, input1) == apply(Axon.Layers, pool, [input1])

        model2 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4})])
        input2 = Nx.random_uniform({1, 1, 8, 4}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model2)
        assert predict_fn.(%{}, input2) == apply(Axon.Layers, pool, [input2])

        model3 = apply(Axon, pool, [Axon.input({nil, 1, 8, 4, 2})])
        input3 = Nx.random_uniform({1, 1, 8, 4, 2}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model3)
        assert predict_fn.(%{}, input3) == apply(Axon.Layers, pool, [input3])
      end
    end

    test "computes forward pass with custom options" do
      for pool <- @global_pooling_layers do
        opts1 = [keep_axes: true]
        model1 = apply(Axon, pool, [Axon.input({nil, 1, 32}), opts1])
        input1 = Nx.random_uniform({1, 1, 32}, type: {:f, 32})

        assert {_, predict_fn} = Axon.compile(model1)
        assert predict_fn.(%{}, input1) == apply(Axon.Layers, pool, [input1, opts1])
      end
    end

    test "computes forward pass with output policy" do
      for pool <- @global_pooling_layers do
        model = apply(Axon, pool, [Axon.input({nil, 1, 32})])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
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
  end

  describe "convolution" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 32, 32}) |> Axon.conv(64, name: "conv")

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert Nx.shape(kernel) == {64, 3, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {64}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 3, 32, 32}) |> Axon.conv(32, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {32, 3, 1, 1})
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 32, 32}) |> Axon.conv(32, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert Nx.shape(kernel) == {32, 3, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {32})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 1, 28}) |> Axon.conv(32, name: "conv")
      input1 = Nx.random_uniform({1, 1, 28}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.conv(input1, kernel, bias)

      model2 = Axon.input({nil, 1, 28, 28}) |> Axon.conv(32, name: "conv")
      input2 = Nx.random_uniform({1, 1, 28, 28}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.conv(input2, kernel, bias)

      model3 = Axon.input({nil, 1, 28, 28, 2}) |> Axon.conv(32, name: "conv")
      input3 = Nx.random_uniform({1, 1, 28, 28, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input3) == Axon.Layers.conv(input3, kernel, bias)
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, padding: :same, input_dilation: 2]
      model1 = Axon.input({nil, 1, 28}) |> Axon.conv(32, [name: "conv", kernel_size: 2] ++ opts1)
      input1 = Nx.random_uniform({1, 1, 28})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.conv(input1, kernel, bias, opts1)

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input({nil, 1, 28, 28}) |> Axon.conv(32, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 1, 28, 28})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.conv(input2, kernel, bias, opts2)

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input({nil, 1, 28, 28, 2})
        |> Axon.conv(32, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 1, 28, 28, 2})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
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

      assert %{"conv_kernel" => kernel_grad, "conv_bias" => bias_grad} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 32})])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1, 1})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 32}) |> Axon.conv(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
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
  end

  describe "depthwise convolution" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 32, 32}) |> Axon.depthwise_conv(3, name: "conv")

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert Nx.shape(kernel) == {9, 1, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {9}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 3, 32, 32})
        |> Axon.depthwise_conv(3, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {9, 1, 1, 1})
      assert Nx.shape(bias) == {9}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 32, 32})
        |> Axon.depthwise_conv(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert Nx.shape(kernel) == {9, 1, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {9})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 1, 28}) |> Axon.depthwise_conv(3, name: "conv")
      input1 = Nx.random_uniform({1, 1, 28}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.depthwise_conv(input1, kernel, bias)

      model2 = Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(4, name: "conv")
      input2 = Nx.random_uniform({1, 1, 28, 28}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.depthwise_conv(input2, kernel, bias)

      model3 = Axon.input({nil, 1, 28, 28, 2}) |> Axon.depthwise_conv(5, name: "conv")
      input3 = Nx.random_uniform({1, 1, 28, 28, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input3) == Axon.Layers.depthwise_conv(input3, kernel, bias)
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, padding: :same, input_dilation: 2]

      model1 =
        Axon.input({nil, 1, 28})
        |> Axon.depthwise_conv(1, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = Nx.random_uniform({1, 1, 28})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()

      assert predict_fn.(params, input1) ==
               Axon.Layers.depthwise_conv(input1, kernel, bias, opts1)

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input({nil, 1, 28, 28})
        |> Axon.depthwise_conv(8, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 1, 28, 28})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()

      assert predict_fn.(params, input2) ==
               Axon.Layers.depthwise_conv(input2, kernel, bias, opts2)

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input({nil, 1, 28, 28, 2})
        |> Axon.depthwise_conv(2, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 1, 28, 28, 2})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()

      assert predict_fn.(params, input3) ==
               Axon.Layers.depthwise_conv(input3, kernel, bias, opts3)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 32})
        |> Axon.depthwise_conv(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"conv_kernel" => kernel_grad, "conv_bias" => bias_grad} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 32})])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1, 1})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 32}) |> Axon.depthwise_conv(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 32}) |> Axon.depthwise_conv(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
    end
  end

  describe "convolution transpose" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 32, 32}) |> Axon.conv_transpose(32, name: "conv")

      assert {init_fn, _} = Axon.compile(model)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert Nx.shape(kernel) == {32, 3, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 3, 32, 32})
        |> Axon.conv_transpose(32, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert kernel == Axon.Initializers.zeros(shape: {32, 3, 1, 1})
      assert Nx.shape(bias) == {32}
      assert Nx.type(bias) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 32, 32})
        |> Axon.conv_transpose(32, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert Nx.shape(kernel) == {32, 3, 1, 1}
      assert Nx.type(kernel) == {:f, 32}
      assert bias == Axon.Initializers.zeros(shape: {32})
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 1, 28}) |> Axon.conv_transpose(3, name: "conv")
      input1 = Nx.random_uniform({1, 1, 28}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.conv_transpose(input1, kernel, bias)

      model2 = Axon.input({nil, 1, 28, 28}) |> Axon.conv_transpose(16, name: "conv")
      input2 = Nx.random_uniform({1, 1, 28, 28}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.conv_transpose(input2, kernel, bias)

      model3 = Axon.input({nil, 1, 28, 28, 2}) |> Axon.conv_transpose(5, name: "conv")
      input3 = Nx.random_uniform({1, 1, 28, 28, 2}, type: {:f, 32})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()
      assert predict_fn.(params, input3) == Axon.Layers.conv_transpose(input3, kernel, bias)
    end

    test "computes forward pass with custom options" do
      opts1 = [strides: 2, kernel_dilation: 1]

      model1 =
        Axon.input({nil, 1, 28})
        |> Axon.conv_transpose(1, [name: "conv", kernel_size: 2] ++ opts1)

      input1 = Nx.random_uniform({1, 1, 28})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()

      assert predict_fn.(params, input1) ==
               Axon.Layers.conv_transpose(input1, kernel, bias, opts1)

      opts2 = [strides: [1, 2], padding: [{0, 1}, {1, 2}], kernel_dilation: 2]

      model2 =
        Axon.input({nil, 1, 28, 28})
        |> Axon.conv_transpose(8, [name: "conv", kernel_size: 2] ++ opts2)

      input2 = Nx.random_uniform({1, 1, 28, 28})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()

      assert predict_fn.(params, input2) ==
               Axon.Layers.conv_transpose(input2, kernel, bias, opts2)

      opts3 = [strides: [2, 1, 1]]

      model3 =
        Axon.input({nil, 1, 28, 28, 2})
        |> Axon.conv_transpose(2, [name: "conv", kernel_size: {2, 1, 1}] ++ opts3)

      input3 = Nx.random_uniform({1, 1, 28, 28, 2})

      assert {init_fn, predict_fn} = Axon.compile(model3)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = params = init_fn.()

      assert predict_fn.(params, input3) ==
               Axon.Layers.conv_transpose(input3, kernel, bias, opts3)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 32})
        |> Axon.conv_transpose(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"conv_kernel" => kernel_grad, "conv_bias" => bias_grad} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 32})])

      assert kernel_grad == Nx.broadcast(0.0, {1, 1, 1})
      assert bias_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 32}) |> Axon.conv_transpose(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"conv_kernel" => kernel, "conv_bias" => bias} = init_fn.()
      assert Nx.type(kernel) == {:bf, 16}
      assert Nx.type(bias) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 32}) |> Axon.conv_transpose(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
    end
  end

  describe "separable convolution 2d" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 32, 32}) |> Axon.separable_conv2d(3, name: "conv")

      assert {init_fn, _} = Axon.compile(model)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2
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
        Axon.input({nil, 3, 32, 32})
        |> Axon.separable_conv2d(3, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model1)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2
             } = init_fn.()

      assert k1 == Axon.Initializers.zeros(shape: {9, 1, 1, 1})
      assert k2 == Axon.Initializers.zeros(shape: {9, 1, 1, 1})
      assert Nx.shape(b1) == {9}
      assert Nx.type(b1) == {:f, 32}
      assert Nx.shape(b2) == {9}
      assert Nx.type(b2) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 32, 32})
        |> Axon.separable_conv2d(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model2)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2
             } = init_fn.()

      assert b1 == Axon.Initializers.zeros(shape: {9})
      assert b2 == Axon.Initializers.zeros(shape: {9})
      assert Nx.shape(k1) == {9, 1, 1, 1}
      assert Nx.type(k1) == {:f, 32}
      assert Nx.shape(k2) == {9, 1, 1, 1}
      assert Nx.type(k2) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model = Axon.input({nil, 3, 32, 32}) |> Axon.separable_conv2d(3, name: "conv")
      input = Nx.random_uniform({1, 3, 32, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2
             } = params = init_fn.()

      assert predict_fn.(params, input) == Axon.Layers.separable_conv2d(input, k1, b1, k2, b2)
    end

    test "computes forward pass with custom options" do
      opts = [strides: [2, 1], input_dilation: [1, 2], kernel_dilation: 1, padding: :same]

      model =
        Axon.input({nil, 3, 32, 32})
        |> Axon.separable_conv2d(3, [name: "conv", kernel_size: {2, 2}] ++ opts)

      input = Nx.random_uniform({1, 3, 32, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2
             } = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv2d(input, k1, b1, k2, b2, opts)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 32, 32})
        |> Axon.separable_conv2d(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{
               "conv_kernel_1" => k1_grad,
               "conv_bias_1" => b1_grad,
               "conv_kernel_2" => k2_grad,
               "conv_bias_2" => b2_grad
             } = Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 32, 32})])

      assert k1_grad == Nx.broadcast(0.0, {1, 1, 1, 1})
      assert b1_grad == Nx.broadcast(0.0, {1})
      assert k2_grad == Nx.broadcast(0.0, {1, 1, 1, 1})
      assert b2_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 32, 32}) |> Axon.separable_conv2d(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2
             } = init_fn.()

      assert Nx.type(k1) == {:bf, 16}
      assert Nx.type(b1) == {:bf, 16}
      assert Nx.type(k2) == {:bf, 16}
      assert Nx.type(b2) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 32, 32}) |> Axon.separable_conv2d(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32, 32}))) == {:bf, 16}
    end
  end

  describe "separable convolution 3d" do
    test "initializes in default case" do
      model = Axon.input({nil, 3, 3, 32, 32}) |> Axon.separable_conv3d(3, name: "conv")

      assert {init_fn, _} = Axon.compile(model)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2,
               "conv_kernel_3" => k3,
               "conv_bias_3" => b3
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
        Axon.input({nil, 3, 3, 32, 32})
        |> Axon.separable_conv3d(3, name: "conv", kernel_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model1)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2,
               "conv_kernel_3" => k3,
               "conv_bias_3" => b3
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
        Axon.input({nil, 3, 3, 32, 32})
        |> Axon.separable_conv3d(3, name: "conv", bias_initializer: :zeros)

      assert {init_fn, _} = Axon.compile(model2)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2,
               "conv_kernel_3" => k3,
               "conv_bias_3" => b3
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
      model = Axon.input({nil, 3, 3, 32, 32}) |> Axon.separable_conv3d(3, name: "conv")
      input = Nx.random_uniform({1, 3, 3, 32, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2,
               "conv_kernel_3" => k3,
               "conv_bias_3" => b3
             } = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3)
    end

    test "computes forward pass with custom options" do
      opts = [strides: [2, 1, 1], input_dilation: [1, 2, 1], kernel_dilation: 1, padding: :same]

      model =
        Axon.input({nil, 3, 3, 32, 32})
        |> Axon.separable_conv3d(3, [name: "conv", kernel_size: {2, 2, 1}] ++ opts)

      input = Nx.random_uniform({1, 3, 3, 32, 32})

      assert {init_fn, predict_fn} = Axon.compile(model)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2,
               "conv_kernel_3" => k3,
               "conv_bias_3" => b3
             } = params = init_fn.()

      assert predict_fn.(params, input) ==
               Axon.Layers.separable_conv3d(input, k1, b1, k2, b2, k3, b3, opts)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 1, 3, 32, 32})
        |> Axon.separable_conv3d(1, name: "conv")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{
               "conv_kernel_1" => k1_grad,
               "conv_bias_1" => b1_grad,
               "conv_kernel_2" => k2_grad,
               "conv_bias_2" => b2_grad,
               "conv_kernel_3" => k3_grad,
               "conv_bias_3" => b3_grad
             } = Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 3, 32, 32})])

      assert k1_grad == Nx.broadcast(0.0, {1, 1, 1, 1, 1})
      assert b1_grad == Nx.broadcast(0.0, {1})
      assert k2_grad == Nx.broadcast(0.0, {1, 1, 1, 1, 1})
      assert b2_grad == Nx.broadcast(0.0, {1})
      assert k3_grad == Nx.broadcast(0.0, {1, 1, 1, 1, 1})
      assert b3_grad == Nx.broadcast(0.0, {1})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 1, 3, 32, 32}) |> Axon.separable_conv3d(1, name: "conv")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)

      assert %{
               "conv_kernel_1" => k1,
               "conv_bias_1" => b1,
               "conv_kernel_2" => k2,
               "conv_bias_2" => b2,
               "conv_kernel_3" => k3,
               "conv_bias_3" => b3
             } = init_fn.()

      assert Nx.type(k1) == {:bf, 16}
      assert Nx.type(b1) == {:bf, 16}
      assert Nx.type(k2) == {:bf, 16}
      assert Nx.type(b2) == {:bf, 16}
      assert Nx.type(k3) == {:bf, 16}
      assert Nx.type(b3) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 1, 3, 32, 32}) |> Axon.separable_conv3d(1, name: "conv")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 3, 32, 32}))) == {:bf, 16}
    end
  end

  @normalization_layers [:batch_norm, :layer_norm, :instance_norm]

  describe "normalization" do
    test "initializes in default case" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input({nil, 32}), [name: "norm"]])

          assert {init_fn, _predict_fn} = Axon.compile(model1)
          assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
          assert Nx.shape(gamma) == {32}
          assert Nx.type(gamma) == {:f, 32}
          assert Nx.shape(beta) == {32}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 = apply(Axon, norm, [Axon.input({nil, 3, 32, 32}), [name: "norm"]])

        assert {init_fn, _predict_fn} = Axon.compile(model2)
        assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
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
            apply(Axon, norm, [Axon.input({nil, 32}), [name: "norm", gamma_initializer: :zeros]])

          assert {init_fn, _predict_fn} = Axon.compile(model1)
          assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
          assert gamma == Axon.Initializers.zeros(shape: {32})
          assert Nx.shape(beta) == {32}
          assert Nx.type(beta) == {:f, 32}
        end

        model2 =
          apply(Axon, norm, [
            Axon.input({nil, 3, 32, 32}),
            [name: "norm", beta_initializer: :zeros]
          ])

        assert {init_fn, _predict_fn} = Axon.compile(model2)
        assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
        assert Nx.shape(gamma) == {3}
        assert Nx.type(gamma) == {:f, 32}
        assert beta == Axon.Initializers.zeros(shape: {3})
      end
    end

    test "computes forward pass with default options" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          model1 = apply(Axon, norm, [Axon.input({nil, 32}), [name: "norm"]])
          input1 = Nx.random_uniform({1, 32}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.compile(model1)
          assert %{"norm_gamma" => gamma, "norm_beta" => beta} = params = init_fn.()
          assert predict_fn.(params, input1) == apply(Axon.Layers, norm, [input1, gamma, beta])
        end

        model2 = apply(Axon, norm, [Axon.input({nil, 3, 32, 32}), [name: "norm"]])
        input2 = Nx.random_uniform({1, 3, 32, 32}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.compile(model2)
        assert %{"norm_gamma" => gamma, "norm_beta" => beta} = params = init_fn.()
        assert predict_fn.(params, input2) == apply(Axon.Layers, norm, [input2, gamma, beta])
      end
    end

    test "computes forward pass with custom options" do
      for norm <- @normalization_layers do
        if norm != :instance_norm do
          opts1 = [channel_index: 1, epsilon: 1.0e-3]
          model1 = apply(Axon, norm, [Axon.input({nil, 32}), [name: "norm"] ++ opts1])
          input1 = Nx.random_uniform({1, 32}, type: {:f, 32})

          assert {init_fn, predict_fn} = Axon.compile(model1)
          assert %{"norm_gamma" => gamma, "norm_beta" => beta} = params = init_fn.()

          assert predict_fn.(params, input1) ==
                   apply(Axon.Layers, norm, [input1, gamma, beta, opts1])
        end

        opts2 = [channel_index: 3, epsilon: 1.0e-4]
        model2 = apply(Axon, norm, [Axon.input({nil, 32, 32, 3}), [name: "norm"] ++ opts2])
        input2 = Nx.random_uniform({1, 32, 32, 3}, type: {:f, 32})

        assert {init_fn, predict_fn} = Axon.compile(model2)
        assert %{"norm_gamma" => gamma, "norm_beta" => beta} = params = init_fn.()

        assert predict_fn.(params, input2) ==
                 apply(Axon.Layers, norm, [input2, gamma, beta, opts2])
      end
    end

    test "returns zero gradient for frozen parameters" do
      for norm <- @normalization_layers do
        model =
          apply(Axon, norm, [Axon.input({nil, 1, 32}), [name: "norm"]])
          |> Axon.freeze()

        assert {init_fn, predict_fn} = Axon.compile(model)

        backward = fn params, input ->
          Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
        end

        assert %{"norm_gamma" => gamma_grad, "norm_beta" => beta_grad} =
                 Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 1, 32})])

        assert gamma_grad == Nx.broadcast(0.0, {1})
        assert beta_grad == Nx.broadcast(0.0, {1})
      end
    end

    test "initializes with parameter policy" do
      for norm <- @normalization_layers do
        model = apply(Axon, norm, [Axon.input({nil, 1, 32}), [name: "norm"]])
        policy = AMP.create_policy(params: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, _} = Axon.compile(mp_model)
        assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
        assert Nx.type(gamma) == {:bf, 16}
        assert Nx.type(beta) == {:bf, 16}
      end
    end

    test "computes forward pass with output policy" do
      for norm <- @normalization_layers do
        model = apply(Axon, norm, [Axon.input({nil, 1, 32}), [name: "norm"]])
        policy = AMP.create_policy(output: {:bf, 16})
        mp_model = AMP.apply_policy(model, policy)

        assert {init_fn, predict_fn} = Axon.compile(mp_model)
        assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 1, 32}))) == {:bf, 16}
      end
    end
  end

  describe "group normalization" do
    test "initializes in default case" do
      model = Axon.input({nil, 32}) |> Axon.group_norm(3, name: "norm")

      assert {init_fn, _predict_fn} = Axon.compile(model)
      assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
      assert Nx.shape(gamma) == {32}
      assert Nx.type(gamma) == {:f, 32}
      assert Nx.shape(beta) == {32}
      assert Nx.type(beta) == {:f, 32}
    end

    test "initializes with custom initializers" do
      model1 =
        Axon.input({nil, 32}) |> Axon.group_norm(3, name: "norm", gamma_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model1)
      assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
      assert gamma == Axon.Initializers.zeros(shape: {32})
      assert Nx.shape(beta) == {32}
      assert Nx.type(beta) == {:f, 32}

      model2 =
        Axon.input({nil, 3, 32}) |> Axon.group_norm(3, name: "norm", beta_initializer: :zeros)

      assert {init_fn, _predict_fn} = Axon.compile(model2)
      assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
      assert beta == Axon.Initializers.zeros(shape: {3})
      assert Nx.shape(gamma) == {3}
      assert Nx.type(gamma) == {:f, 32}
    end

    test "computes forward pass with default options" do
      model1 = Axon.input({nil, 32}) |> Axon.group_norm(2, name: "norm")
      input1 = Nx.random_uniform({1, 32})

      assert {init_fn, predict_fn} = Axon.compile(model1)
      assert %{"norm_gamma" => gamma, "norm_beta" => beta} = params = init_fn.()
      assert predict_fn.(params, input1) == Axon.Layers.group_norm(input1, gamma, beta, group_size: 2)

      model2 = Axon.input({nil, 3, 16, 16}) |> Axon.group_norm(3, name: "norm")
      input2 = Nx.random_uniform({1, 3, 16, 16})

      assert {init_fn, predict_fn} = Axon.compile(model2)
      assert %{"norm_gamma" => gamma, "norm_beta" => beta} = params = init_fn.()
      assert predict_fn.(params, input2) == Axon.Layers.group_norm(input2, gamma, beta, group_size: 3)
    end

    test "computes forward pass with custom options" do
      opts = [epsilon: 1.0e-3, channel_index: 3]
      model = Axon.input({nil, 16, 16, 3}) |> Axon.group_norm(3, [name: "norm"] ++ opts)
      input = Nx.random_uniform({1, 16, 16, 3})

      assert {init_fn, predict_fn} = Axon.compile(model)
      assert %{"norm_gamma" => gamma, "norm_beta" => beta} = params = init_fn.()
      assert predict_fn.(params, input) == Axon.Layers.group_norm(input, gamma, beta, [group_size: 3] ++ opts)
    end

    test "returns zero gradient for frozen parameters" do
      model =
        Axon.input({nil, 32})
        |> Axon.group_norm(1, name: "norm")
        |> Axon.freeze()

      assert {init_fn, predict_fn} = Axon.compile(model)

      backward = fn params, input ->
        Nx.Defn.grad(params, &Nx.mean(predict_fn.(&1, input)))
      end

      assert %{"norm_gamma" => gamma_grad, "norm_beta" => beta_grad} =
               Nx.Defn.jit(backward, [init_fn.(), Nx.random_uniform({1, 32})])

      assert gamma_grad == Nx.broadcast(0.0, {32})
      assert beta_grad == Nx.broadcast(0.0, {32})
    end

    test "initializes with parameter policy" do
      model = Axon.input({nil, 32}) |> Axon.group_norm(1, name: "norm")
      policy = AMP.create_policy(params: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, _} = Axon.compile(mp_model)
      assert %{"norm_gamma" => gamma, "norm_beta" => beta} = init_fn.()
      assert Nx.type(gamma) == {:bf, 16}
      assert Nx.type(beta) == {:bf, 16}
    end

    test "computes forward pass with output policy" do
      model = Axon.input({nil, 32}) |> Axon.group_norm(1, name: "norm")
      policy = AMP.create_policy(output: {:bf, 16})
      mp_model = AMP.apply_policy(model, policy)

      assert {init_fn, predict_fn} = Axon.compile(mp_model)
      assert Nx.type(predict_fn.(init_fn.(), Nx.random_uniform({1, 32}))) == {:bf, 16}
    end
  end
end
