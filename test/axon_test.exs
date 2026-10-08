defmodule AxonTest do
  use ExUnit.Case
  doctest Axon

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

  describe "elem" do
    test "extracts a tuple element from a container output" do
      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 2})

      model = Axon.container({inp1, inp2}) |> Axon.elem(1)

      inputs = %{
        "input_0" => Nx.tensor([[1.0]]),
        "input_1" => Nx.tensor([[3.0, 4.0]])
      }

      assert Axon.predict(model, Axon.ModelState.empty(), inputs) ==
               Nx.tensor([[3.0, 4.0]])
    end

    test "tags the resulting layer with the :elem op_name" do
      inp = Axon.input("input", shape: {nil, 1})

      assert %Axon{output: id, nodes: nodes} =
               Axon.container({inp, inp}) |> Axon.elem(0)

      assert %Axon.Node{op_name: :elem} = nodes[id]
    end
  end

  describe "fetch" do
    test "extracts a map value from a container output" do
      inp1 = Axon.input("input_0", shape: {nil, 1})
      inp2 = Axon.input("input_1", shape: {nil, 2})

      model = Axon.container(%{a: inp1, b: inp2}) |> Axon.fetch(:b)

      inputs = %{
        "input_0" => Nx.tensor([[1.0]]),
        "input_1" => Nx.tensor([[3.0, 4.0]])
      }

      assert Axon.predict(model, Axon.ModelState.empty(), inputs) ==
               Nx.tensor([[3.0, 4.0]])
    end

    test "tags the resulting layer with the :fetch op_name" do
      inp = Axon.input("input", shape: {nil, 1})

      assert %Axon{output: id, nodes: nodes} =
               Axon.container(%{a: inp}) |> Axon.fetch(:a)

      assert %Axon.Node{op_name: :fetch} = nodes[id]
    end

    test "raises at predict time when key is missing" do
      inp = Axon.input("input", shape: {nil, 1})
      model = Axon.container(%{a: inp}) |> Axon.fetch(:missing)

      assert_raise Axon.CompileError, fn ->
        Axon.predict(model, Axon.ModelState.empty(), %{"input" => Nx.tensor([[1.0]])})
      end
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

      assert %Axon.ModelState{data: %{"dense" => %{"kernel" => kernel, "bias" => bias}}} =
               params = init_fn.(inp, Axon.ModelState.empty())

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
      params = init_fn.(inp, Axon.ModelState.empty())

      assert Axon.predict(model, params, inp) == inp
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
               nodes: 7
             >\
             """
    end
  end

  describe "container" do
    test "correctly derives container" do
      model = Axon.input("input", shape: {nil, 1})

      assert Axon.predict(model, Axon.ModelState.empty(), Nx.tensor([[1.0]])) ==
               Nx.tensor([[1.0]])
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

      assert {t1, t2} = Axon.get_output_shape(model, Nx.template({1, 1}, :f32))
      assert Nx.shape(t1) == {1, 2}
      assert Nx.shape(t2) == {1, 2}
    end

    test "doesn't raise on none output" do
      values = Axon.input("values")
      mask = Axon.input("mask", optional: true)

      model =
        values
        |> Axon.dense(10)
        |> Axon.multiply(mask)
        |> Axon.dense(1)
        |> Axon.sigmoid()

      assert %Axon.None{} = Axon.get_output_shape(model, %{"values" => Nx.template({1, 1}, :f32)})
    end
  end

  describe "namespace" do
    defp init_param_keys(model) do
      {init_fn, _predict_fn} = Axon.build(model)
      state = init_fn.(%{"input" => Nx.template({1, 4}, :f32)}, Axon.ModelState.empty())
      state.data |> Map.keys() |> Enum.sort()
    end

    test "single namespace prefixes leaf names" do
      input = Axon.input("input", shape: {nil, 4})

      model =
        Axon.namespace("attention", fn ->
          input
          |> Axon.dense(8, name: "q_proj")
          |> Axon.dense(8, name: "out_proj")
        end)

      assert init_param_keys(model) == ["attention.out_proj", "attention.q_proj"]
    end

    test "pipe form threads an input into an arity-1 closure" do
      input = Axon.input("input", shape: {nil, 4})

      model =
        Axon.namespace(input, "ffn", fn x ->
          x
          |> Axon.dense(8, name: "up_proj")
          |> Axon.dense(4, name: "down_proj")
        end)

      assert init_param_keys(model) == ["ffn.down_proj", "ffn.up_proj"]
    end

    test "nested namespaces stack with a dot separator" do
      input = Axon.input("input", shape: {nil, 4})

      model =
        Axon.namespace("outer", fn ->
          Axon.namespace("inner", fn ->
            Axon.dense(input, 8, name: "proj")
          end)
        end)

      assert init_param_keys(model) == ["outer.inner.proj"]
    end

    test "layers from outer scope are not renamed" do
      input = Axon.input("input", shape: {nil, 4})
      preprocessed = Axon.dense(input, 4, name: "preprocess")

      model =
        Axon.namespace(preprocessed, "head", fn x ->
          Axon.dense(x, 1, name: "logits")
        end)

      assert init_param_keys(model) == ["head.logits", "preprocess"]
    end

    test "auto-generated leaf names are also prefixed" do
      input = Axon.input("input", shape: {nil, 4})

      model =
        Axon.namespace("block", fn ->
          input
          |> Axon.dense(8)
          |> Axon.dense(8)
        end)

      keys = init_param_keys(model)
      assert Enum.all?(keys, &String.starts_with?(&1, "block."))
      assert length(keys) == 2
    end

    test "namespace introduces no extra graph node" do
      input = Axon.input("input", shape: {nil, 4})
      bare = Axon.dense(input, 8, name: "proj")
      wrapped = Axon.namespace("ns", fn -> Axon.dense(input, 8, name: "proj") end)

      bare_ops = Axon.get_op_counts(bare)
      wrapped_ops = Axon.get_op_counts(wrapped)

      assert bare_ops == wrapped_ops
      refute Map.has_key?(wrapped_ops, :namespace)
      refute Map.has_key?(wrapped_ops, :block)
    end

    test "predict produces the same numerical output as the bare form" do
      input = Axon.input("input", shape: {nil, 4})

      bare = Axon.dense(input, 8, name: "proj")
      wrapped = Axon.namespace("ns", fn -> Axon.dense(input, 8, name: "proj") end)

      {init_bare, predict_bare} = Axon.build(bare)
      {init_wrapped, predict_wrapped} = Axon.build(wrapped)

      template = %{"input" => Nx.template({1, 4}, :f32)}
      state_bare = init_bare.(template, Axon.ModelState.empty())
      state_wrapped = init_wrapped.(template, Axon.ModelState.empty())

      # Same fan-in shape => identical default initializer values for the
      # kernel and bias, just stored under different paths. Patch the
      # wrapped state's params over from the bare state so we compare
      # apples to apples on the prediction side.
      patched =
        update_in(state_wrapped.data, fn _ ->
          %{"ns.proj" => state_bare.data["proj"]}
        end)

      inputs = %{"input" => Nx.tensor([[1.0, 2.0, 3.0, 4.0]])}
      assert_all_close(predict_bare.(state_bare, inputs), predict_wrapped.(patched, inputs))
    end

    test "raises when the closure does not return an Axon graph" do
      assert_raise ArgumentError, ~r/expected the closure to return an %Axon{}/, fn ->
        Axon.namespace("oops", fn -> :not_an_axon end)
      end
    end

    test "raises when the pipe form's closure has the wrong arity" do
      input = Axon.input("input", shape: {nil, 4})

      # Applied dynamically so the compiler's type checker does not flag the
      # deliberately mismatched arity at compile time.
      args = [input, "ns", fn -> Axon.dense(input, 8) end]

      assert_raise FunctionClauseError, fn ->
        apply(Axon, :namespace, args)
      end
    end
  end

  describe "capture" do
    test "captures entire model with single input" do
      model =
        Axon.input("features", shape: {nil, 10})
        |> Axon.dense(32, name: "dense1")
        |> Axon.dense(16, name: "dense2")

      captured = Axon.capture(model)
      assert is_function(captured, 1)

      # Use with new input
      new_input = Axon.input("my_input", shape: {nil, 10})
      new_model = captured.(new_input)

      # Verify new model has the new input
      assert %{"my_input" => _} = Axon.get_inputs(new_model)
      refute Map.has_key?(Axon.get_inputs(new_model), "features")

      # Verify layers are preserved
      props = Axon.properties(new_model)
      assert Map.has_key?(props, "dense1")
      assert Map.has_key?(props, "dense2")
    end

    test "captures up to specific layer with :to option" do
      model =
        Axon.input("features", shape: {nil, 10})
        |> Axon.dense(32, name: "hidden1")
        |> Axon.relu(name: "relu1")
        |> Axon.dense(16, name: "hidden2")
        |> Axon.dense(2, name: "output")

      captured = Axon.capture(model, to: "hidden2")

      new_input = Axon.input("x", shape: {nil, 10})
      new_model = captured.(new_input)

      props = Axon.properties(new_model)
      assert Map.has_key?(props, "hidden1")
      assert Map.has_key?(props, "relu1")
      assert Map.has_key?(props, "hidden2")
      refute Map.has_key?(props, "output")
    end

    test "captures multi-output models with :to before or after the head split" do
      shared =
        Axon.input("x", shape: {nil, 4})
        |> Axon.dense(8, name: "trunk")
        |> Axon.relu(name: "shared")

      model =
        Axon.container(%{
          a: Axon.dense(shared, 2, name: "head_a"),
          b: Axon.dense(shared, 3, name: "head_b")
        })

      template = %{"y" => Nx.template({1, 4}, :f32)}
      new_input = Axon.input("y", shape: {nil, 4})

      # Capture shared trunk: single tensor, both heads and container dropped
      to_shared = Axon.capture(model, to: "shared").(new_input)
      shared_props = Axon.properties(to_shared)
      assert Map.has_key?(shared_props, "shared")
      assert Map.has_key?(shared_props, "trunk")
      refute Map.has_key?(shared_props, "head_a")
      refute Map.has_key?(shared_props, "head_b")
      refute Map.has_key?(shared_props, "container_0")
      assert %Nx.Tensor{shape: {1, 8}} = Axon.get_output_shape(to_shared, template)

      # Capture one head after the split: sibling head and container dropped
      to_head = Axon.capture(model, to: "head_a").(new_input)
      head_props = Axon.properties(to_head)
      assert Map.has_key?(head_props, "head_a")
      assert Map.has_key?(head_props, "shared")
      refute Map.has_key?(head_props, "head_b")
      refute Map.has_key?(head_props, "container_0")
      assert %Nx.Tensor{shape: {1, 2}} = Axon.get_output_shape(to_head, template)

      # No :to keeps the multi-output container
      full = Axon.capture(model).(new_input)
      full_props = Axon.properties(full)
      assert Map.has_key?(full_props, "head_a")
      assert Map.has_key?(full_props, "head_b")
      assert Map.has_key?(full_props, "container_0")

      assert %{a: %Nx.Tensor{shape: {1, 2}}, b: %Nx.Tensor{shape: {1, 3}}} =
               Axon.get_output_shape(full, template)
    end

    test "works with multiple inputs using map" do
      input1 = Axon.input("image", shape: {nil, 784})
      input2 = Axon.input("text", shape: {nil, 128})

      model =
        Axon.concatenate(input1, input2)
        |> Axon.dense(64, name: "combined")

      captured = Axon.capture(model)

      new_image = Axon.input("my_image", shape: {nil, 784})
      new_text = Axon.input("my_text", shape: {nil, 128})

      new_model = captured.(%{"image" => new_image, "text" => new_text})

      inputs = Axon.get_inputs(new_model)
      assert Map.has_key?(inputs, "my_image")
      assert Map.has_key?(inputs, "my_text")
      refute Map.has_key?(inputs, "image")
      refute Map.has_key?(inputs, "text")
    end

    test "single input and single output keep their values" do
      model =
        Axon.input("x", shape: {nil, 2})
        |> Axon.dense(3, name: "out", kernel_initializer: :ones, bias_initializer: :zeros)

      input = Nx.tensor([[1.0, 2.0]])
      captured = Axon.capture(model).(Axon.input("x2", shape: {nil, 2}))

      assert Axon.get_inputs(captured) == %{"x2" => {nil, 2}}
      assert predict_capture(captured, input) == predict_capture(model, input)
      assert predict_capture(captured, input) == Nx.tensor([[3.0, 3.0, 3.0]])
    end

    test "single input and multiple outputs keep their values" do
      shared =
        Axon.input("x", shape: {nil, 2})
        |> Axon.dense(3, name: "trunk", kernel_initializer: :ones, bias_initializer: :zeros)

      model =
        Axon.container(%{
          a:
            Axon.dense(shared, 1,
              name: "head_a",
              kernel_initializer: :ones,
              bias_initializer: :zeros
            ),
          b:
            Axon.dense(shared, 2,
              name: "head_b",
              kernel_initializer: :ones,
              bias_initializer: :zeros
            )
        })

      input = Nx.tensor([[1.0, 2.0]])
      captured = Axon.capture(model).(Axon.input("x2", shape: {nil, 2}))

      assert Axon.get_inputs(captured) == %{"x2" => {nil, 2}}

      assert %{a: a, b: b} = predict_capture(captured, input)
      assert predict_capture(captured, input) == predict_capture(model, input)
      assert a == Nx.tensor([[9.0]])
      assert b == Nx.tensor([[9.0, 9.0]])
    end

    test "multiple inputs and a single output keep their values" do
      left = Axon.input("left", shape: {nil, 2})
      right = Axon.input("right", shape: {nil, 2})

      model =
        Axon.add(left, right)
        |> Axon.dense(2, name: "out", kernel_initializer: :ones, bias_initializer: :zeros)

      captured =
        Axon.capture(model).(%{
          "left" => Axon.input("left2", shape: {nil, 2}),
          "right" => Axon.input("right2", shape: {nil, 2})
        })

      assert Axon.get_inputs(captured) == %{"left2" => {nil, 2}, "right2" => {nil, 2}}

      original = %{"left" => Nx.tensor([[1.0, 2.0]]), "right" => Nx.tensor([[3.0, 4.0]])}
      replaced = %{"left2" => Nx.tensor([[1.0, 2.0]]), "right2" => Nx.tensor([[3.0, 4.0]])}

      assert predict_capture(captured, replaced) == predict_capture(model, original)
      assert predict_capture(captured, replaced) == Nx.tensor([[10.0, 10.0]])
    end

    test "multiple inputs and multiple outputs keep their values" do
      left = Axon.input("left", shape: {nil, 2})
      right = Axon.input("right", shape: {nil, 2})
      summed = Axon.add(left, right)

      model =
        Axon.container(%{
          a:
            Axon.dense(summed, 1,
              name: "head_a",
              kernel_initializer: :ones,
              bias_initializer: :zeros
            ),
          b:
            Axon.dense(summed, 2,
              name: "head_b",
              kernel_initializer: :ones,
              bias_initializer: :zeros
            )
        })

      captured =
        Axon.capture(model).(%{
          "left" => Axon.input("left2", shape: {nil, 2}),
          "right" => Axon.input("right2", shape: {nil, 2})
        })

      assert Axon.get_inputs(captured) == %{"left2" => {nil, 2}, "right2" => {nil, 2}}

      original = %{"left" => Nx.tensor([[1.0, 2.0]]), "right" => Nx.tensor([[3.0, 4.0]])}
      replaced = %{"left2" => Nx.tensor([[1.0, 2.0]]), "right2" => Nx.tensor([[3.0, 4.0]])}

      assert %{a: a, b: b} = predict_capture(captured, replaced)
      assert predict_capture(captured, replaced) == predict_capture(model, original)
      assert a == Nx.tensor([[10.0]])
      assert b == Nx.tensor([[10.0, 10.0]])
    end

    test "raises on invalid layer name" do
      model =
        Axon.input("features", shape: {nil, 10})
        |> Axon.dense(32, name: "dense1")

      assert_raise ArgumentError, ~r/layer "nonexistent" not found/, fn ->
        Axon.capture(model, to: "nonexistent")
      end
    end

    test "raises on missing inputs for multi-input model" do
      input1 = Axon.input("a", shape: {nil, 10})
      input2 = Axon.input("b", shape: {nil, 10})

      model = Axon.add(input1, input2) |> Axon.dense(5)
      captured = Axon.capture(model)

      new_a = Axon.input("new_a", shape: {nil, 10})

      assert_raise ArgumentError, ~r/missing inputs/, fn ->
        captured.(%{"a" => new_a})
      end
    end

    test "raises when single input passed to multi-input model" do
      input1 = Axon.input("a", shape: {nil, 10})
      input2 = Axon.input("b", shape: {nil, 10})

      model = Axon.add(input1, input2)
      captured = Axon.capture(model)

      single_input = Axon.input("x", shape: {nil, 10})

      assert_raise ArgumentError, ~r/model has 2 inputs/, fn ->
        captured.(single_input)
      end
    end

    test "captured model can be executed" do
      model =
        Axon.input("features", shape: {nil, 2})
        |> Axon.dense(4, name: "dense1", kernel_initializer: :ones, bias_initializer: :zeros)
        |> Axon.relu(name: "relu")
        |> Axon.dense(2, name: "dense2", kernel_initializer: :ones, bias_initializer: :zeros)

      # Capture up to relu
      captured = Axon.capture(model, to: "relu")
      new_input = Axon.input("x", shape: {nil, 2})
      new_model = captured.(new_input)

      # Build and run
      {init_fn, predict_fn} = Axon.build(new_model)
      params = init_fn.(Nx.template({1, 2}, :f32), Axon.ModelState.empty())

      input = Nx.tensor([[1.0, 2.0]])
      result = predict_fn.(params, input)

      # With ones kernel: [1,2] dot ones(2,4) = [3,3,3,3], relu keeps it positive
      assert Nx.shape(result) == {1, 4}
    end
  end

  defp predict_capture(model, input) do
    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(capture_template(input), Axon.ModelState.empty())
    predict_fn.(params, input)
  end

  defp capture_template(%Nx.Tensor{} = tensor), do: Nx.template(Nx.shape(tensor), Nx.type(tensor))

  defp capture_template(inputs) when is_map(inputs) do
    Map.new(inputs, fn {name, tensor} -> {name, capture_template(tensor)} end)
  end
end
