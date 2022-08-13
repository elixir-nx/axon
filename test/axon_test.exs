defmodule AxonTest do
  use ExUnit.Case
  doctest Axon
  import AxonTestUtil

  describe "input" do
    test "works with defaults" do
      assert %Axon{op: :input, parent: []} = Axon.input("input", shape: {32, 1, 28, 28})
    end
  end

  describe "constant" do
    test "works with defaults" do
      assert %Axon{op: :constant, opts: [value: value]} = Axon.constant(Nx.tensor(1.0))
      assert value == Nx.tensor(1.0)
    end

    test "raises on bad value" do
      assert_raise ArgumentError, ~r/value passed to constant/, fn ->
        Axon.constant(:foo)
      end

      assert_raise ArgumentError, ~r/value passed to constant/, fn ->
        Axon.constant(1)
      end
    end
  end

  describe "dense" do
    test "works with defaults" do
      assert %Axon{
               op: :dense,
               parameters: [weight, bias]
             } = Axon.input("input", shape: {nil, 784}) |> Axon.dense(128)

      assert %Axon.Parameter{initializer: :glorot_uniform} = weight
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with parameter initializer" do
      assert %Axon{op: :dense, parameters: [weight, bias]} =
               Axon.input("input", shape: {nil, 784})
               |> Axon.dense(128, kernel_initializer: :lecun_normal, bias_initializer: :ones)

      assert %Axon.Parameter{initializer: :lecun_normal} = weight
      assert %Axon.Parameter{initializer: :ones} = bias
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :dense}]} =
               Axon.input("input", shape: {nil, 784}) |> Axon.dense(128, activation: :relu)
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 784}) |> Axon.dense(128, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 784}) |> Axon.dense(128, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{parameters: [_]} =
               Axon.input("input", shape: {nil, 784}) |> Axon.dense(128, use_bias: false)
    end
  end

  describe "conv" do
    test "works with defaults" do
      assert %Axon{op: :conv, parameters: [kernel, bias], opts: opts} =
               Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(64)

      assert opts[:padding] == :valid
      assert opts[:strides] == 1
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{initializer: :glorot_uniform} = kernel
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :conv}]} =
               Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(64, activation: :relu)
    end

    test "works with options" do
      assert %Axon{op: :conv, opts: opts, parameters: [kernel, bias]} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.conv(64, padding: :same, strides: [2, 1], kernel_size: 2)

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{initializer: :glorot_uniform} = kernel
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(128, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(128, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{parameters: [_]} =
               Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv(2, use_bias: false)
    end
  end

  describe "depthwise_conv" do
    test "works with defaults" do
      assert %Axon{op: :depthwise_conv, parameters: [kernel, bias], opts: opts} =
               Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.depthwise_conv(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == 1
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{initializer: :glorot_uniform} = kernel
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :depthwise_conv}]} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.depthwise_conv(64, activation: :relu)
    end

    test "works with options" do
      assert %Axon{op: :depthwise_conv, opts: opts, parameters: [kernel, bias]} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.depthwise_conv(3, padding: :same, strides: [2, 1], kernel_size: 2)

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{initializer: :glorot_uniform} = kernel
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 1, 28, 28})
        |> Axon.depthwise_conv(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 1, 28, 28})
        |> Axon.depthwise_conv(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{parameters: [_]} =
               Axon.input("input", shape: {nil, 1, 2}) |> Axon.depthwise_conv(1, use_bias: false)
    end
  end

  describe "separable_conv2d" do
    test "works with defaults" do
      assert %Axon{
               op: :separable_conv2d,
               parameters: [k1, b1, k2, b2],
               opts: opts
             } = Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.separable_conv2d(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == 1
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{initializer: :glorot_uniform} = k1
      assert %Axon.Parameter{initializer: :glorot_uniform} = k2
      assert %Axon.Parameter{initializer: :zeros} = b1
      assert %Axon.Parameter{initializer: :zeros} = b2
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :separable_conv2d}]} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.separable_conv2d(3, activation: :relu)
    end

    test "works with options" do
      assert %Axon{
               op: :separable_conv2d,
               opts: opts,
               parameters: [k1, b1, k2, b2]
             } =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.separable_conv2d(3, padding: :same, strides: [2, 1], kernel_size: 2)

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{initializer: :glorot_uniform} = k1
      assert %Axon.Parameter{initializer: :glorot_uniform} = k2
      assert %Axon.Parameter{initializer: :zeros} = b1
      assert %Axon.Parameter{initializer: :zeros} = b2
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 1, 28, 28})
        |> Axon.separable_conv2d(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 1, 28, 28})
        |> Axon.separable_conv2d(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: _, parameters: [_, _]} =
               Axon.input("input", shape: {nil, 1, 2, 2})
               |> Axon.separable_conv2d(1, use_bias: false)
    end
  end

  describe "separable_conv3d" do
    test "works with defaults" do
      assert %Axon{
               op: :separable_conv3d,
               parameters: [k1, b1, k2, b2, k3, b3],
               opts: opts
             } = Axon.input("input", shape: {nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == 1
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{initializer: :glorot_uniform} = k1
      assert %Axon.Parameter{initializer: :glorot_uniform} = k2
      assert %Axon.Parameter{initializer: :glorot_uniform} = k3
      assert %Axon.Parameter{initializer: :zeros} = b1
      assert %Axon.Parameter{initializer: :zeros} = b2
      assert %Axon.Parameter{initializer: :zeros} = b3
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :separable_conv3d}]} =
               Axon.input("input", shape: {nil, 1, 28, 28, 3})
               |> Axon.separable_conv3d(3, activation: :relu)
    end

    test "works with options" do
      assert %Axon{
               op: :separable_conv3d,
               opts: opts,
               parameters: [k1, b1, k2, b2, k3, b3]
             } =
               Axon.input("input", shape: {nil, 1, 28, 28, 3})
               |> Axon.separable_conv3d(3,
                 padding: :same,
                 strides: [2, 1, 1],
                 kernel_size: {2, 2, 2}
               )

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1, 1]
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{} = k1
      assert %Axon.Parameter{} = k2
      assert %Axon.Parameter{} = k3
      assert %Axon.Parameter{} = b1
      assert %Axon.Parameter{} = b2
      assert %Axon.Parameter{} = b3
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 1, 28, 28, 3})
        |> Axon.separable_conv3d(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input("input", shape: {nil, 1, 28, 28, 3})
        |> Axon.separable_conv3d(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: _, parameters: [_, _, _]} =
               Axon.input("input", shape: {nil, 1, 2, 2, 2})
               |> Axon.separable_conv3d(1, use_bias: false)
    end
  end

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  describe "activation" do
    test "works with valid activation" do
      for act <- @activation_layers do
        assert %Axon{op: act1} = Axon.input("input", shape: {nil, 32}) |> Axon.activation(act)
        assert act1 == act
        assert %Axon{op: act2} = apply(Axon, act, [Axon.input("input", shape: {nil, 32})])
        assert act2 == act
      end
    end
  end

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  describe "dropout" do
    test "works with defaults" do
      for dropout <- @dropout_layers do
        assert %Axon{op: drop1, opts: opts} =
                 apply(Axon, dropout, [Axon.input("input", shape: {nil, 32})])

        assert drop1 == dropout
        assert opts[:rate] == 0.5
      end
    end
  end

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  describe "pooling" do
    test "works with defaults" do
      for pool <- @pooling_layers do
        assert %Axon{op: pool1, opts: opts} =
                 apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 28, 28})])

        assert pool1 == pool

        assert opts[:padding] == :valid
        assert opts[:strides] == nil
        assert opts[:kernel_size] == 1
      end
    end
  end

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool, :adaptive_lp_pool]

  describe "adaptive pooling" do
    test "works with options" do
      for pool <- @adaptive_pooling_layers do
        assert %Axon{} =
                 apply(Axon, pool, [
                   Axon.input("input", shape: {nil, 1, 28, 28}),
                   [output_size: {26, 26}, name: "pool"]
                 ])
      end
    end
  end

  @global_pooling_layers [:global_avg_pool, :global_max_pool, :global_lp_pool]

  describe "global pooling" do
    test "works with options" do
      for pool <- @global_pooling_layers do
        assert %Axon{} =
                 apply(Axon, pool, [
                   Axon.input("input", shape: {nil, 1, 28, 28}),
                   [keep_axes: false, name: "pool"]
                 ])
      end
    end
  end

  @stateful_normalization [:batch_norm, :instance_norm]

  describe "stateful normalization" do
    test "works with defaults" do
      for norm <- @stateful_normalization do
        assert %Axon{op: norm1, opts: opts, parameters: [gamma, beta, mean, var]} =
                 apply(Axon, norm, [Axon.input("input", shape: {nil, 784})])

        assert norm1 == norm

        assert opts[:channel_index] == 1
        assert opts[:epsilon] == 1.0e-5

        assert %Axon.Parameter{initializer: :glorot_uniform} = gamma
        assert %Axon.Parameter{initializer: :zeros} = beta
        assert %Axon.Parameter{initializer: :zeros} = mean
        assert %Axon.Parameter{initializer: :ones} = var
      end
    end

    test "works with parameter initializer" do
      for norm <- @stateful_normalization do
        assert %Axon{parameters: [gamma, beta, mean, var]} =
                 apply(Axon, norm, [
                   Axon.input("input", shape: {nil, 784}),
                   [gamma_initializer: :lecun_normal, beta_initializer: :ones]
                 ])

        assert %Axon.Parameter{initializer: :lecun_normal} = gamma
        assert %Axon.Parameter{initializer: :ones} = beta
        assert %Axon.Parameter{initializer: :zeros} = mean
        assert %Axon.Parameter{initializer: :ones} = var
      end
    end

    test "fails on bad initializers" do
      for norm <- @stateful_normalization do
        assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
          apply(Axon, norm, [Axon.input("input", shape: {nil, 784}), [gamma_initializer: :foo]])
        end

        assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
          apply(Axon, norm, [Axon.input("input", shape: {nil, 784}), [beta_initializer: :foo]])
        end
      end
    end
  end

  describe "layer normalization" do
    test "works with defaults" do
      assert %Axon{op: :layer_norm, opts: opts, parameters: [gamma, beta]} =
               Axon.layer_norm(Axon.input("input", shape: {nil, 784}))

      assert opts[:channel_index] == 1
      assert opts[:epsilon] == 1.0e-5

      assert %Axon.Parameter{initializer: :glorot_uniform} = gamma
      assert %Axon.Parameter{initializer: :zeros} = beta
    end

    test "works with parameter initializer" do
      assert %Axon{parameters: [gamma, beta]} =
               Axon.layer_norm(Axon.input("input", shape: {nil, 784}),
                 gamma_initializer: :lecun_normal,
                 beta_initializer: :ones
               )

      assert %Axon.Parameter{initializer: :lecun_normal} = gamma
      assert %Axon.Parameter{initializer: :ones} = beta
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.layer_norm(Axon.input("input", shape: {nil, 784}), gamma_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.layer_norm(Axon.input("input", shape: {nil, 784}), beta_initializer: :foo)
      end
    end
  end

  describe "group normalization" do
    test "works with defaults" do
      assert %Axon{op: :group_norm, parameters: [gamma, beta], opts: opts} =
               Axon.input("input", shape: {nil, 3, 28, 28}) |> Axon.group_norm(3)

      assert opts[:channel_index] == 1
      assert opts[:epsilon] == 1.0e-5
      assert opts[:group_size] == 3

      assert %Axon.Parameter{initializer: :glorot_uniform} = gamma
      assert %Axon.Parameter{initializer: :zeros} = beta
    end

    test "works with parameter initializer" do
      assert %Axon{parameters: [gamma, beta]} =
               Axon.input("input", shape: {nil, 3, 28, 28})
               |> Axon.group_norm(3, gamma_initializer: :lecun_normal, beta_initializer: :ones)

      assert %Axon.Parameter{initializer: :lecun_normal} = gamma
      assert %Axon.Parameter{initializer: :ones} = beta
    end

    test "fails on bad initializer" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        apply(Axon, :group_norm, [
          Axon.input("input", shape: {nil, 784}),
          3,
          [gamma_initializer: :foo]
        ])
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        apply(Axon, :group_norm, [
          Axon.input("input", shape: {nil, 784}),
          3,
          [beta_initializer: :foo]
        ])
      end
    end
  end

  describe "flatten" do
    test "works with defaults" do
      assert %Axon{op: :flatten} = Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.flatten()
    end
  end

  describe "concatenate" do
    test "works with 2 inputs" do
      assert %Axon{op: :concatenate, parent: [%Axon{parent: [{%Axon{}, %Axon{}}]}]} =
               Axon.concatenate(
                 Axon.input("input", shape: {nil, 32}),
                 Axon.input("input", shape: {nil, 32})
               )
    end

    test "works with many inputs" do
      assert %Axon{op: :concatenate, parent: [%Axon{parent: [{%Axon{}, %Axon{}, %Axon{}}]}]} =
               Axon.concatenate([
                 Axon.input("input", shape: {nil, 32}),
                 Axon.input("input", shape: {nil, 32}),
                 Axon.input("input", shape: {nil, 32})
               ])
    end
  end

  @element_wise_layers [:add, :subtract, :multiply]

  describe "element-wise layers" do
    test "works with 2 inputs" do
      for op <- @element_wise_layers do
        assert %Axon{op: op1, parent: [%Axon{parent: [{%Axon{}, %Axon{}}]}]} =
                 apply(Axon, op, [
                   Axon.input("input", shape: {nil, 32}),
                   Axon.input("input", shape: {nil, 32})
                 ])

        assert op1 == op
      end
    end

    test "works with many inputs" do
      for op <- @element_wise_layers do
        assert %Axon{op: op1, parent: [%Axon{parent: [{%Axon{}, %Axon{}, %Axon{}}]}]} =
                 apply(Axon, op, [
                   [
                     Axon.input("input", shape: {nil, 32}),
                     Axon.input("input", shape: {nil, 32}),
                     Axon.input("input", shape: {nil, 32})
                   ]
                 ])

        assert op1 == op
      end
    end
  end

  describe "nx" do
    test "works with defaults" do
      assert %Axon{} = Axon.input("input", shape: {nil, 32}) |> Axon.nx(fn x -> Nx.erf(x) end)
    end
  end

  describe "embedding" do
    test "works with defaults" do
      assert %Axon{} =
               Axon.input("input", shape: {nil, 10}) |> Axon.embedding(128, 32, name: "embedding")
    end
  end

  describe "reshape" do
    test "works with batch input" do
      assert %Axon{} = Axon.input("input", shape: {nil, 9}) |> Axon.reshape({3, 3})
    end

    test "works with constant input" do
      assert %Axon{} = Axon.constant(Nx.iota({6})) |> Axon.reshape({1, 2, 3})
    end
  end

  describe "transpose" do
    test "works with batch input" do
      assert %Axon{} = Axon.input("input", shape: {nil, 2, 1}) |> Axon.transpose([1, 0])
    end

    test "works with constant input" do
      assert %Axon{} =
               Axon.constant(Nx.iota({3, 2, 1}))
               |> Axon.transpose([2, 1, 0], ignore_batch?: false)
    end
  end

  # TODO(seanmor5): Move/replace all with compiler_test
  describe "execution" do
    test "compile returns init and predict" do
      inp = Nx.iota({1, 6}, type: {:f, 32})

      {init_fn, predict_fn} =
        Axon.input("input", shape: {nil, 6})
        |> Axon.dense(6, kernel_initializer: :identity, name: "dense")
        |> Axon.build()

      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(inp, %{})
      assert kernel == Nx.eye({6, 6}, type: {:f, 32})
      assert bias == zeros({6})

      assert predict_fn.(params, inp) == inp
    end

    def model do
      Axon.input("input", shape: {nil, 6})
      |> Axon.dense(6, kernel_initializer: :identity, name: "dense")
    end

    test "predict works outside defn" do
      inp = Nx.iota({1, 6}, type: {:f, 32})
      model = model()

      {init_fn, _} = Axon.build(model)
      params = init_fn.(inp, %{})

      assert Axon.predict(model, params, inp) == inp
    end
  end

  describe "model freezing" do
    test "sets metadata correctly" do
      model =
        Axon.input("input", shape: {nil, 784})
        |> Axon.dense(128)
        |> Axon.freeze()

      assert %Axon{parameters: [%{name: "kernel", frozen: true}, %{name: "bias", frozen: true}]} =
               model

      assert %Axon{parameters: [%{name: "kernel", frozen: false}, %{name: "bias", frozen: false}]} =
               model |> Axon.dense(10)
    end
  end

  describe "inspection" do
    test "works with basic model" do
      model =
        Axon.input("input", shape: {nil, 784})
        |> Axon.dense(128, name: "dense1")
        |> Axon.dense(10, name: "dense2")
        |> Axon.softmax(name: "softmax")

      assert inspect(model) == """
             #Axon<
               inputs: %{"input" => {nil, 784}}
               outputs: "softmax"
               nodes: 4
             >\
             """
    end

    test "works with complex model" do
      residual = fn x ->
        x
        |> Axon.dense(128, name: "residual_dense")
        |> Axon.add(x, name: "residual_add")
      end

      model =
        Axon.input("input", shape: {nil, 784})
        |> Axon.dense(128, name: "dense")
        |> residual.()
        |> Axon.dense(10, name: "dense2")
        |> Axon.softmax(name: "softmax")

      assert inspect(model) == """
             #Axon<
               inputs: %{"input" => {nil, 784}}
               outputs: "softmax"
               nodes: 7
             >\
             """
    end

    test "works with rnns" do
      {out_sequence, _} =
        Axon.input("input_0", shape: {nil, 32, 10}) |> Axon.lstm(64, name: "lstm")

      assert inspect(out_sequence) == """
             #Axon<
               inputs: %{"input_0" => {nil, 32, 10}}
               outputs: "lstm_output_sequence"
               nodes: 6
             >\
             """
    end

    test "works with single namespace" do
      model = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")

      assert inspect(model) == """
             #Axon<
               inputs: %{"input_0" => {nil, 1}}
               outputs: "x"
               nodes: 3
             >\
             """
    end

    test "works with nested namespace" do
      model =
        Axon.input("input_0", shape: {nil, 1})
        |> Axon.dense(2)
        |> Axon.namespace("x")
        |> Axon.namespace("y")

      assert inspect(model) == """
             #Axon<
               inputs: %{"input_0" => {nil, 1}}
               outputs: "y"
               nodes: 4
             >\
             """
    end

    test "works with multiple namespaces" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("y")

      model = Axon.add(x, y)

      assert inspect(model) == """
             #Axon<
               inputs: %{"input_0" => {nil, 1}, "input_1" => {nil, 1}}
               outputs: "add_0"
               nodes: 8
             >\
             """
    end

    test "works with single namespace and no namespace" do
      x = Axon.input("input_0", shape: {nil, 1}) |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input("input_1", shape: {nil, 1}) |> Axon.dense(2)

      model = Axon.add(x, y)

      assert inspect(model) == """
             #Axon<
               inputs: %{"input_0" => {nil, 1}, "input_1" => {nil, 1}}
               outputs: "add_0"
               nodes: 7
             >\
             """
    end
  end

  describe "container" do
    test "correctly derives container" do
      model = Axon.input("input", shape: {nil, 1})
      assert Axon.predict(model, %{}, Nx.tensor([[1.0]])) == Nx.tensor([[1.0]])
    end

    test "shape inference works" do
      last_hidden_state = Axon.input("last_hidden_state", shape: {5, 128, 768})
      pooled = Axon.input("pooled", shape: {5, 768})

      assert %Axon{} = Axon.container({last_hidden_state, pooled}) |> Axon.nx(&elem(&1, 0))

      assert %Axon{} = Axon.container({last_hidden_state, pooled}) |> Axon.nx(&elem(&1, 1))
    end
  end

  describe "serialization" do
    test "correctly serializes and deserializes simple container" do
      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.container(%{a: inp1, b: inp2})

      serialized = Axon.serialize(model, %{})
      {deserialized, params} = Axon.deserialize(serialized)

      input1 = Nx.tensor([[1.0]])
      input2 = Nx.tensor([[1.0, 2.0]])

      assert %{a: a, b: b} =
               Axon.predict(deserialized, params, %{"input_0" => input1, "input_1" => input2})

      assert a == input1
      assert b == input2
    end

    test "correctly serializes and deserializes nested container" do
      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 2})
      model = Axon.container({{inp1, {}}, %{a: inp1}, {%{b: inp2, c: inp1, d: %{}}}})

      serialized = Axon.serialize(model, %{})
      {deserialized, params} = Axon.deserialize(serialized)

      input1 = Nx.tensor([[1.0]])
      input2 = Nx.tensor([[1.0, 2.0]])

      assert {{a, {}}, %{a: b}, {%{b: c, c: d, d: %{}}}} =
               Axon.predict(deserialized, params, %{"input_0" => input1, "input_1" => input2})

      assert a == input1
      assert b == input1
      assert c == input2
      assert d == input1
    end
  end

  describe "layer names" do
    test "only accepts binaries, functions or nil" do
      %Axon{name: name_fn, op: op} = Axon.input("a_binary_name", shape: {nil, 1})

      assert "a_binary_name" == name_fn.(op, input: 1)

      %Axon{name: name_fn, op: op} =
        Axon.input("input", shape: {nil, 1})
        |> Axon.dense(2, name: fn op, _ -> "custom_#{op}" end)

      assert "custom_#{op}" == name_fn.(op, input: 1)

      %Axon{name: name_fn, op: op} =
        Axon.input("input", shape: {nil, 1}) |> Axon.dense(2, name: nil)

      assert "dense_10" == name_fn.(op, dense: 10)
    end

    @invalid_names [:atom, {"tuple"}, ["list"], 123]

    test "raises on invalid names" do
      Enum.each(@invalid_names, fn name ->
        assert_raise ArgumentError, fn ->
          Axon.input("input", shape: {nil, 1}) |> Axon.dense(2, name: name)
        end
      end)
    end
  end
end
