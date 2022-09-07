defmodule AxonTest do
  use ExUnit.Case
  doctest Axon

  import ExUnit.CaptureLog
  import AxonTestUtil

  describe "input" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} = Axon.input("input", shape: {32, 1, 28, 28})
      assert %Axon.Node{op: :input, parent: []} = nodes[id]
    end
  end

  describe "constant" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} = Axon.constant(Nx.tensor(1.0))
      assert %Axon.Node{op: :constant, opts: [value: value]} = nodes[id]
      assert value == Nx.tensor(1.0)
    end

    test "raises on bad value" do
      assert_raise ArgumentError, ~r/value passed to constant/, fn ->
        Axon.constant(:foo)
      end
    end
  end

  describe "dense" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 784}) |> Axon.dense(128)

      assert %Axon.Node{
               op: :dense,
               parameters: [weight, bias]
             } = nodes[id]

      assert %Axon.Parameter{} = weight
      assert %Axon.Parameter{} = bias
    end

    test "works with parameter initializer" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 784})
               |> Axon.dense(128, kernel_initializer: :lecun_normal, bias_initializer: :ones)

      assert %Axon.Node{op: :dense, parameters: [weight, bias]} = nodes[id]

      assert %Axon.Parameter{} = weight
      assert %Axon.Parameter{} = bias
    end

    test "works with activation" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 784}) |> Axon.dense(128, activation: :relu)

      assert %Axon.Node{op: :relu, parent: [_]} = nodes[id]
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
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 784}) |> Axon.dense(128, use_bias: false)

      assert %Axon.Node{parameters: [_]} = nodes[id]
    end
  end

  describe "conv" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(64)

      assert %Axon.Node{op: :conv, parameters: [kernel, bias], opts: opts} = nodes[id]

      assert opts[:padding] == :valid
      assert opts[:strides] == 1
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{} = kernel
      assert %Axon.Parameter{} = bias
    end

    test "works with activation" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.conv(64, activation: :relu)

      assert %Axon.Node{op: :relu, parent: [_]} = nodes[id]
    end

    test "works with options" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.conv(64, padding: :same, strides: [2, 1], kernel_size: 2)

      assert %Axon.Node{op: :conv, opts: opts, parameters: [kernel, bias]} = nodes[id]

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{} = kernel
      assert %Axon.Parameter{} = bias
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
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 2}) |> Axon.conv(2, use_bias: false)

      assert %Axon.Node{parameters: [_]} = nodes[id]
    end
  end

  describe "depthwise_conv" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.depthwise_conv(3)

      assert %Axon.Node{op: :depthwise_conv, parameters: [kernel, bias], opts: opts} = nodes[id]

      assert opts[:padding] == :valid
      assert opts[:strides] == 1
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{} = kernel
      assert %Axon.Parameter{} = bias
    end

    test "works with activation" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.depthwise_conv(64, activation: :relu)

      assert %Axon.Node{op: :relu, parent: [_]} = nodes[id]
    end

    test "works with options" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.depthwise_conv(3, padding: :same, strides: [2, 1], kernel_size: 2)

      assert %Axon.Node{op: :depthwise_conv, opts: opts, parameters: [kernel, bias]} = nodes[id]

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{} = kernel
      assert %Axon.Parameter{} = bias
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
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 2}) |> Axon.depthwise_conv(1, use_bias: false)

      assert %Axon.Node{parameters: [_]} = nodes[id]
    end
  end

  describe "separable_conv2d" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.separable_conv2d(3)

      assert %Axon.Node{
               op: :separable_conv2d,
               parameters: [k1, b1, k2, b2],
               opts: opts
             } = nodes[id]

      assert opts[:padding] == :valid
      assert opts[:strides] == 1
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{} = k1
      assert %Axon.Parameter{} = b1
      assert %Axon.Parameter{} = k2
      assert %Axon.Parameter{} = b2
    end

    test "works with activation" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.separable_conv2d(3, activation: :relu)

      assert %Axon.Node{op: :relu, parent: [_]} = nodes[id]
    end

    test "works with options" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28})
               |> Axon.separable_conv2d(3, padding: :same, strides: [2, 1], kernel_size: 2)

      assert %Axon.Node{
               op: :separable_conv2d,
               opts: opts,
               parameters: [k1, b1, k2, b2]
             } = nodes[id]

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{} = k1
      assert %Axon.Parameter{} = b1
      assert %Axon.Parameter{} = k2
      assert %Axon.Parameter{} = b2
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
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 2, 2})
               |> Axon.separable_conv2d(1, use_bias: false)

      assert %Axon.Node{op: _, parameters: [_, _]} = nodes[id]
    end
  end

  describe "separable_conv3d" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3)

      assert %Axon.Node{
               op: :separable_conv3d,
               parameters: [k1, b1, k2, b2, k3, b3],
               opts: opts
             } = nodes[id]

      assert opts[:padding] == :valid
      assert opts[:strides] == 1
      assert opts[:kernel_dilation] == 1
      assert opts[:input_dilation] == 1

      assert %Axon.Parameter{} = k1
      assert %Axon.Parameter{} = b1
      assert %Axon.Parameter{} = k2
      assert %Axon.Parameter{} = b2
      assert %Axon.Parameter{} = k3
      assert %Axon.Parameter{} = b3
    end

    test "works with activation" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28, 3})
               |> Axon.separable_conv3d(3, activation: :relu)

      assert %Axon.Node{op: :relu, parent: [_]} = nodes[id]
    end

    test "works with options" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28, 3})
               |> Axon.separable_conv3d(3,
                 padding: :same,
                 strides: [2, 1, 1],
                 kernel_size: {2, 2, 2}
               )

      assert %Axon.Node{
               op: :separable_conv3d,
               opts: opts,
               parameters: [k1, b1, k2, b2, k3, b3]
             } = nodes[id]

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
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 2, 2, 2})
               |> Axon.separable_conv3d(1, use_bias: false)

      assert %Axon.Node{op: _, parameters: [_, _, _]} = nodes[id]
    end
  end

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sumexp, :log_sigmoid, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  describe "activation" do
    test "works with valid activation" do
      for act <- @activation_layers do
        assert %Axon{output: id, nodes: nodes} =
                 Axon.input("input", shape: {nil, 32}) |> Axon.activation(act)

        assert %Axon.Node{op: act1} = nodes[id]

        assert act1 == act

        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, act, [Axon.input("input", shape: {nil, 32})])

        assert %Axon.Node{op: act2} = nodes[id]
        assert act2 == act
      end
    end
  end

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  describe "dropout" do
    test "works with defaults" do
      for dropout <- @dropout_layers do
        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, dropout, [Axon.input("input", shape: {nil, 32})])

        assert %Axon.Node{op: drop1, opts: opts} = nodes[id]

        assert drop1 == dropout
        assert opts[:rate] == 0.5
      end
    end

    test "raises for rates below zero" do
      opts = [rate: -0.1]

      for dropout <- @dropout_layers do
        assert_raise ArgumentError,
                     "The dropout rate needs to be >= 0 and < 1, got #{inspect(opts[:rate])}",
                     fn ->
                       apply(Axon, dropout, [Axon.input("input", shape: {nil, 32}), opts])
                     end
      end
    end

    test "raises for rates above or equal to one" do
      opts = [rate: 1]

      for dropout <- @dropout_layers do
        assert_raise ArgumentError,
                     "The dropout rate needs to be >= 0 and < 1, got #{inspect(opts[:rate])}",
                     fn ->
                       apply(Axon, dropout, [Axon.input("input", shape: {nil, 32}), opts])
                     end
      end
    end
  end

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  describe "pooling" do
    test "works with defaults" do
      for pool <- @pooling_layers do
        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, pool, [Axon.input("input", shape: {nil, 1, 28, 28})])

        assert %Axon.Node{op: pool1, opts: opts} = nodes[id]

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
        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, pool, [
                   Axon.input("input", shape: {nil, 1, 28, 28}),
                   [output_size: {26, 26}, name: "pool"]
                 ])

        assert %Axon.Node{} = nodes[id]
      end
    end
  end

  @global_pooling_layers [:global_avg_pool, :global_max_pool, :global_lp_pool]

  describe "global pooling" do
    test "works with options" do
      for pool <- @global_pooling_layers do
        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, pool, [
                   Axon.input("input", shape: {nil, 1, 28, 28}),
                   [keep_axes: false, name: "pool"]
                 ])

        assert %Axon.Node{} = nodes[id]
      end
    end
  end

  @stateful_normalization [:batch_norm, :instance_norm]

  describe "stateful normalization" do
    test "works with defaults" do
      for norm <- @stateful_normalization do
        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, norm, [Axon.input("input", shape: {nil, 784})])

        assert %Axon.Node{op: norm1, opts: opts, parameters: [gamma, beta, mean, var]} = nodes[id]

        assert norm1 == norm

        assert opts[:channel_index] == -1
        assert opts[:epsilon] == 1.0e-5

        assert %Axon.Parameter{} = gamma
        assert %Axon.Parameter{} = beta
        assert %Axon.Parameter{} = mean
        assert %Axon.Parameter{} = var
      end
    end

    test "works with parameter initializer" do
      for norm <- @stateful_normalization do
        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, norm, [
                   Axon.input("input", shape: {nil, 784}),
                   [gamma_initializer: :lecun_normal, beta_initializer: :ones]
                 ])

        assert %Axon.Node{parameters: [gamma, beta, mean, var]} = nodes[id]

        assert %Axon.Parameter{} = gamma
        assert %Axon.Parameter{} = beta
        assert %Axon.Parameter{} = mean
        assert %Axon.Parameter{} = var
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
      assert %Axon{output: id, nodes: nodes} =
               Axon.layer_norm(Axon.input("input", shape: {nil, 784}))

      assert %Axon.Node{op: :layer_norm, opts: opts, parameters: [gamma, beta]} = nodes[id]

      assert opts[:channel_index] == -1
      assert opts[:epsilon] == 1.0e-5

      assert %Axon.Parameter{} = gamma
      assert %Axon.Parameter{} = beta
    end

    test "works with parameter initializer" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.layer_norm(Axon.input("input", shape: {nil, 784}),
                 gamma_initializer: :lecun_normal,
                 beta_initializer: :ones
               )

      assert %Axon.Node{parameters: [gamma, beta]} = nodes[id]

      assert %Axon.Parameter{} = gamma
      assert %Axon.Parameter{} = beta
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
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 3, 28, 28}) |> Axon.group_norm(3)

      assert %Axon.Node{op: :group_norm, parameters: [gamma, beta], opts: opts} = nodes[id]

      assert opts[:channel_index] == -1
      assert opts[:epsilon] == 1.0e-5
      assert opts[:num_groups] == 3

      assert %Axon.Parameter{} = gamma
      assert %Axon.Parameter{} = beta
    end

    test "works with parameter initializer" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 3, 28, 28})
               |> Axon.group_norm(3, gamma_initializer: :lecun_normal, beta_initializer: :ones)

      assert %Axon.Node{parameters: [gamma, beta]} = nodes[id]

      assert %Axon.Parameter{} = gamma
      assert %Axon.Parameter{} = beta
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
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 1, 28, 28}) |> Axon.flatten()

      assert %Axon.Node{op: :flatten} = nodes[id]
    end
  end

  describe "concatenate" do
    test "works with 2 inputs" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.concatenate(
                 Axon.input("input", shape: {nil, 32}),
                 Axon.input("input", shape: {nil, 32})
               )

      assert %Axon.Node{
               op: :concatenate,
               parent: [_]
             } = nodes[id]
    end

    test "works with many inputs" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.concatenate([
                 Axon.input("input", shape: {nil, 32}),
                 Axon.input("input", shape: {nil, 32}),
                 Axon.input("input", shape: {nil, 32})
               ])

      assert %Axon.Node{
               op: :concatenate,
               parent: [_]
             } = nodes[id]
    end
  end

  @element_wise_layers [:add, :subtract, :multiply]

  describe "element-wise layers" do
    test "works with 2 inputs" do
      for op <- @element_wise_layers do
        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, op, [
                   Axon.input("input", shape: {nil, 32}),
                   Axon.input("input", shape: {nil, 32})
                 ])

        assert %Axon.Node{op: op1, parent: [_]} = nodes[id]

        assert op1 == op
      end
    end

    test "works with many inputs" do
      for op <- @element_wise_layers do
        assert %Axon{output: id, nodes: nodes} =
                 apply(Axon, op, [
                   [
                     Axon.input("input", shape: {nil, 32}),
                     Axon.input("input", shape: {nil, 32}),
                     Axon.input("input", shape: {nil, 32})
                   ]
                 ])

        assert %Axon.Node{
                 op: op1,
                 parent: [_]
               } = nodes[id]

        assert op1 == op
      end
    end
  end

  describe "nx" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 32}) |> Axon.nx(fn x -> Nx.erf(x) end)

      assert %Axon.Node{} = nodes[id]
    end
  end

  describe "embedding" do
    test "works with defaults" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 10}) |> Axon.embedding(128, 32, name: "embedding")

      assert %Axon.Node{} = nodes[id]
    end
  end

  describe "reshape" do
    test "works with batch input" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 9}) |> Axon.reshape({3, 3})

      assert %Axon.Node{} = nodes[id]
    end

    test "works with constant input" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.constant(Nx.iota({6})) |> Axon.reshape({1, 2, 3})

      assert %Axon.Node{} = nodes[id]
    end
  end

  describe "transpose" do
    test "works with batch input" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.input("input", shape: {nil, 2, 1}) |> Axon.transpose([0, 2, 1])

      assert %Axon.Node{} = nodes[id]
    end

    test "works with constant input" do
      assert %Axon{output: id, nodes: nodes} =
               Axon.constant(Nx.iota({3, 2, 1}))
               |> Axon.transpose([2, 1, 0])

      assert %Axon.Node{} = nodes[id]
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

      assert %Axon{output: id, nodes: nodes} = model

      assert %Axon.Node{
               parameters: [%{name: "kernel", frozen: true}, %{name: "bias", frozen: true}]
             } = nodes[id]

      assert %Axon{output: id, nodes: nodes} = model |> Axon.dense(10)

      assert %Axon.Node{
               parameters: [%{name: "kernel", frozen: false}, %{name: "bias", frozen: false}]
             } = nodes[id]
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

      assert %Axon{output: id, nodes: nodes} =
               Axon.container({last_hidden_state, pooled}) |> Axon.nx(&elem(&1, 0))

      assert %Axon.Node{} = nodes[id]

      assert %Axon{output: id, nodes: nodes} =
               Axon.container({last_hidden_state, pooled}) |> Axon.nx(&elem(&1, 1))

      assert %Axon.Node{} = nodes[id]
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

    # TODO: Raise on next release
    test "warns when serializing anonymous function" do
      model = Axon.input("input") |> Axon.nx(fn x -> Nx.cos(x) end)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({1, 1}, :f32), %{})

      assert capture_log(fn -> Axon.serialize(model, params) end) =~ "Attempting to serialize"
    end

    test "warns when deserializing anonymous function" do
      model = Axon.input("input") |> Axon.nx(fn x -> Nx.cos(x) end)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({1, 1}, :f32), %{})

      assert capture_log(fn ->
               serialized = Axon.serialize(model, params)
               Axon.deserialize(serialized)
             end) =~ "Attempting to deserialize"
    end
  end

  describe "layer names" do
    test "only accepts binaries, functions or nil" do
      %Axon{output: id, nodes: nodes} = Axon.input("a_binary_name", shape: {nil, 1})
      %Axon.Node{name: name_fn, op: op} = nodes[id]

      assert "a_binary_name" == name_fn.(op, input: 1)

      %Axon{output: id, nodes: nodes} =
        Axon.input("input", shape: {nil, 1})
        |> Axon.dense(2, name: fn op, _ -> "custom_#{op}" end)

      %Axon.Node{name: name_fn, op: op} = nodes[id]

      assert "custom_#{op}" == name_fn.(op, input: 1)

      %Axon{output: id, nodes: nodes} =
        Axon.input("input", shape: {nil, 1}) |> Axon.dense(2, name: nil)

      %Axon.Node{name: name_fn, op: op} = nodes[id]

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

  describe "get_output_shape" do
    test "works with container shapes" do
      out = Axon.input("input") |> Axon.dense(2)
      model = Axon.container({out, out})

      assert shape = Axon.get_output_shape(model, Nx.template({1, 1}, :f32))
      assert shape == {{1, 2}, {1, 2}}
    end
  end
end
