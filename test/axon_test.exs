defmodule AxonTest do
  use ExUnit.Case
  doctest Axon

  import Nx.Defn

  describe "input" do
    test "works with defaults" do
      assert %Axon{op: :input, parent: nil} = Axon.input({32, 1, 28, 28})
    end

    test "works with name" do
      assert %Axon{op: :input, parent: nil, name: "input"} =
               Axon.input({nil, 1, 28, 28}, name: "input")
    end
  end

  describe "constant" do
    test "works with defaults" do
      assert %Axon{op: :constant, opts: [value: value]} = Axon.constant(Nx.tensor(1.0))
      assert value == Nx.tensor(1.0)
    end

    test "works with name" do
      assert %Axon{op: :constant, name: "constant"} =
               Axon.constant(Nx.tensor(1.0), name: "constant")
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
               params: %{"kernel" => weight, "bias" => bias},
               opts: [use_bias: true]
             } = Axon.input({nil, 784}) |> Axon.dense(128)

      assert %Axon.Parameter{initializer: :glorot_uniform} = weight
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with name" do
      assert %Axon{op: :dense, name: "dense1"} =
               Axon.input({nil, 784}) |> Axon.dense(128, name: "dense1")
    end

    test "works with parameter initializer" do
      assert %Axon{op: :dense, params: %{"kernel" => weight, "bias" => bias}} =
               Axon.input({nil, 784})
               |> Axon.dense(128, kernel_initializer: :lecun_normal, bias_initializer: :ones)

      assert %Axon.Parameter{initializer: :lecun_normal} = weight
      assert %Axon.Parameter{initializer: :ones} = bias
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: %Axon{op: :dense}} =
               Axon.input({nil, 784}) |> Axon.dense(128, activation: :relu)
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 784}) |> Axon.dense(128, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 784}) |> Axon.dense(128, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: :dense, opts: [use_bias: false]} =
               Axon.input({nil, 784}) |> Axon.dense(128, use_bias: false)
    end
  end

  describe "conv" do
    test "works with defaults" do
      assert %Axon{op: :conv, params: %{"kernel" => kernel, "bias" => bias}, opts: opts} =
               Axon.input({nil, 1, 28, 28}) |> Axon.conv(64)

      assert opts[:padding] == :valid
      assert opts[:strides] == [1, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]
      assert opts[:use_bias] == true

      assert %Axon.Parameter{initializer: :glorot_uniform} = kernel
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with name" do
      assert %Axon{op: :conv, name: "conv1"} =
               Axon.input({nil, 1, 28, 28}) |> Axon.conv(64, name: "conv1")
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: %Axon{op: :conv}} =
               Axon.input({nil, 1, 28, 28}) |> Axon.conv(64, activation: :relu)
    end

    test "works with options" do
      assert %Axon{op: :conv, opts: opts, params: %{"kernel" => kernel, "bias" => bias}} =
               Axon.input({nil, 1, 28, 28})
               |> Axon.conv(64, padding: :same, strides: [2, 1], kernel_size: 2)

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]

      assert %Axon.Parameter{shape: {64, 1, 2, 2}} = kernel
      assert %Axon.Parameter{shape: {64}} = bias
    end

    test "fails on bad options" do
      assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.conv(128, kernel_size: :foo)
      end

      assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.conv(128, strides: :foo)
      end

      assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.conv(128, padding: :foo)
      end

      assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.conv(128, input_dilation: :foo)
      end

      assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.conv(128, kernel_dilation: :foo)
      end
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.conv(128, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.conv(128, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: :conv, opts: opts} =
               Axon.input({nil, 1, 2}) |> Axon.conv(2, use_bias: false)

      assert opts[:use_bias] == false
    end
  end

  describe "depthwise_conv" do
    test "works with defaults" do
      assert %Axon{op: :depthwise_conv, params: %{"kernel" => kernel, "bias" => bias}, opts: opts} =
               Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == [1, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]
      assert opts[:use_bias] == true

      assert %Axon.Parameter{initializer: :glorot_uniform} = kernel
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with name" do
      assert %Axon{op: :depthwise_conv, name: "depthwise_conv1"} =
               Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3, name: "depthwise_conv1")
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: %Axon{op: :depthwise_conv}} =
               Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(64, activation: :relu)
    end

    test "works with options" do
      assert %Axon{op: :depthwise_conv, opts: opts, params: %{"kernel" => kernel, "bias" => bias}} =
               Axon.input({nil, 1, 28, 28})
               |> Axon.depthwise_conv(3, padding: :same, strides: [2, 1], kernel_size: 2)

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]

      assert %Axon.Parameter{shape: {3, 1, 2, 2}} = kernel
      assert %Axon.Parameter{shape: {3}} = bias
    end

    test "fails on bad options" do
      assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3, kernel_size: :foo)
      end

      assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3, strides: :foo)
      end

      assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3, padding: :foo)
      end

      assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3, input_dilation: :foo)
      end

      assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3, kernel_dilation: :foo)
      end
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.depthwise_conv(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: :depthwise_conv, opts: opts} =
               Axon.input({nil, 1, 2}) |> Axon.depthwise_conv(1, use_bias: false)

      assert opts[:use_bias] == false
    end
  end

  describe "separable_conv2d" do
    test "works with defaults" do
      assert %Axon{
               op: :separable_conv2d,
               params: %{"k1" => k1, "b1" => b1, "k2" => k2, "b2" => b2},
               opts: opts
             } = Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == [1, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]
      assert opts[:use_bias] == true

      assert %Axon.Parameter{initializer: :glorot_uniform} = k1
      assert %Axon.Parameter{initializer: :glorot_uniform} = k2
      assert %Axon.Parameter{initializer: :zeros} = b1
      assert %Axon.Parameter{initializer: :zeros} = b2
    end

    test "works with name" do
      assert %Axon{op: :separable_conv2d, name: "separable_conv1"} =
               Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, name: "separable_conv1")
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: %Axon{op: :separable_conv2d}} =
               Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, activation: :relu)
    end

    test "works with options" do
      assert %Axon{
               op: :separable_conv2d,
               opts: opts,
               params: %{"k1" => k1, "b1" => b1, "k2" => k2, "b2" => b2}
             } =
               Axon.input({nil, 1, 28, 28})
               |> Axon.separable_conv2d(3, padding: :same, strides: [2, 1], kernel_size: 2)

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]

      assert %Axon.Parameter{shape: {3, 1, 2, 1}} = k1
      assert %Axon.Parameter{shape: {3, 1, 1, 2}} = k2
      assert %Axon.Parameter{shape: {3}} = b1
      assert %Axon.Parameter{shape: {3}} = b2
    end

    test "fails on bad options" do
      assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, kernel_size: :foo)
      end

      assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, strides: :foo)
      end

      assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, padding: :foo)
      end

      assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, input_dilation: :foo)
      end

      assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, kernel_dilation: :foo)
      end
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}) |> Axon.separable_conv2d(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: :separable_conv2d, opts: opts} =
               Axon.input({nil, 1, 2, 2}) |> Axon.separable_conv2d(1, use_bias: false)

      assert opts[:use_bias] == false
    end
  end

  describe "separable_conv3d" do
    test "works with defaults" do
      assert %Axon{
               op: :separable_conv3d,
               params: %{"k1" => k1, "b1" => b1, "k2" => k2, "b2" => b2, "k3" => k3, "b3" => b3},
               opts: opts
             } = Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == [1, 1, 1]
      assert opts[:kernel_dilation] == [1, 1, 1]
      assert opts[:input_dilation] == [1, 1, 1]
      assert opts[:use_bias] == true

      assert %Axon.Parameter{initializer: :glorot_uniform} = k1
      assert %Axon.Parameter{initializer: :glorot_uniform} = k2
      assert %Axon.Parameter{initializer: :glorot_uniform} = k3
      assert %Axon.Parameter{initializer: :zeros} = b1
      assert %Axon.Parameter{initializer: :zeros} = b2
      assert %Axon.Parameter{initializer: :zeros} = b3
    end

    test "works with name" do
      assert %Axon{op: :separable_conv3d, name: "separable_conv1"} =
               Axon.input({nil, 1, 28, 28, 3})
               |> Axon.separable_conv3d(3, name: "separable_conv1")
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: %Axon{op: :separable_conv3d}} =
               Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, activation: :relu)
    end

    test "works with options" do
      assert %Axon{
               op: :separable_conv3d,
               opts: opts,
               params: %{"k1" => k1, "b1" => b1, "k2" => k2, "b2" => b2, "k3" => k3, "b3" => b3}
             } =
               Axon.input({nil, 1, 28, 28, 3})
               |> Axon.separable_conv3d(3,
                 padding: :same,
                 strides: [2, 1, 1],
                 kernel_size: {2, 2, 2}
               )

      assert opts[:padding] == :same
      assert opts[:strides] == [2, 1, 1]
      assert opts[:kernel_dilation] == [1, 1, 1]
      assert opts[:input_dilation] == [1, 1, 1]

      assert %Axon.Parameter{shape: {3, 1, 2, 1, 1}} = k1
      assert %Axon.Parameter{shape: {3, 1, 1, 2, 1}} = k2
      assert %Axon.Parameter{shape: {3, 1, 1, 1, 2}} = k3
      assert %Axon.Parameter{shape: {3}} = b1
      assert %Axon.Parameter{shape: {3}} = b2
      assert %Axon.Parameter{shape: {3}} = b3
    end

    test "fails on bad options" do
      assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
        Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, kernel_size: {1, 1})
      end

      assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
        Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, strides: :foo)
      end

      assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
        Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, padding: :foo)
      end

      assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, input_dilation: :foo)
      end

      assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, kernel_dilation: [1, 1])
      end
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28, 3}) |> Axon.separable_conv3d(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: :separable_conv3d, opts: opts} =
               Axon.input({nil, 1, 2, 2, 2}) |> Axon.separable_conv3d(1, use_bias: false)

      assert opts[:use_bias] == false
    end
  end

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  describe "activation" do
    test "works with valid activation" do
      for act <- @activation_layers do
        assert %Axon{op: act1} = Axon.input({nil, 32}) |> Axon.activation(act)
        assert act1 == act
        assert %Axon{op: act2} = apply(Axon, act, [Axon.input({nil, 32})])
        assert act2 == act
      end
    end

    test "works with names" do
      for act <- @activation_layers do
        assert %Axon{name: "activation"} =
                 Axon.input({nil, 32}) |> Axon.activation(act, name: "activation")

        assert %Axon{name: "activation"} =
                 apply(Axon, act, [Axon.input({nil, 32}), [name: "activation"]])
      end
    end
  end

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  describe "dropout" do
    test "works with defaults" do
      for dropout <- @dropout_layers do
        assert %Axon{op: drop1, opts: opts} = apply(Axon, dropout, [Axon.input({nil, 32})])
        assert drop1 == dropout
        assert opts[:rate] == 0.5
      end
    end

    test "works with names" do
      for dropout <- @dropout_layers do
        assert %Axon{name: "dropout"} =
                 apply(Axon, dropout, [Axon.input({nil, 32}), [name: "dropout"]])
      end
    end
  end

  @pooling_layers [:max_pool, :avg_pool, :lp_pool]

  describe "pooling" do
    test "works with defaults" do
      for pool <- @pooling_layers do
        assert %Axon{op: pool1, opts: opts} = apply(Axon, pool, [Axon.input({nil, 1, 28, 28})])
        assert pool1 == pool

        assert opts[:padding] == :valid
        assert opts[:strides] == [1, 1]
        assert opts[:kernel_size] == {1, 1}
      end
    end

    test "works with names" do
      for pool <- @pooling_layers do
        assert %Axon{name: "pool"} =
                 apply(Axon, pool, [Axon.input({nil, 1, 28, 28}), [name: "pool"]])
      end
    end

    test "fails on bad options" do
      for pool <- @pooling_layers do
        assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
          apply(Axon, pool, [Axon.input({nil, 1, 28, 28}), [strides: :foo]])
        end

        assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
          apply(Axon, pool, [Axon.input({nil, 1, 28, 28}), [kernel_size: :foo]])
        end

        assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
          apply(Axon, pool, [Axon.input({nil, 1, 28, 28}), [padding: :foo]])
        end
      end
    end
  end

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool, :adaptive_lp_pool]

  describe "adaptive pooling" do
    test "works with options" do
      for pool <- @adaptive_pooling_layers do
        assert %Axon{name: "pool"} =
                 apply(Axon, pool, [
                   Axon.input({nil, 1, 28, 28}),
                   [output_size: {26, 26}, name: "pool"]
                 ])
      end
    end
  end

  @global_pooling_layers [:global_avg_pool, :global_max_pool, :global_lp_pool]

  describe "global pooling" do
    test "works with options" do
      for pool <- @global_pooling_layers do
        assert %Axon{name: "pool", output_shape: {nil, 1}} =
                 apply(Axon, pool, [
                   Axon.input({nil, 1, 28, 28}),
                   [keep_axes: false, name: "pool"]
                 ])
      end
    end
  end

  @normalization_layers [:batch_norm, :layer_norm, :instance_norm]

  describe "normalization" do
    test "works with defaults" do
      for norm <- @normalization_layers do
        assert %Axon{op: norm1, opts: opts, params: %{"gamma" => gamma, "beta" => beta}} =
                 apply(Axon, norm, [Axon.input({nil, 784})])

        assert norm1 == norm

        assert opts[:channel_index] == 1
        assert opts[:epsilon] == 1.0e-5

        assert %Axon.Parameter{initializer: :glorot_uniform} = gamma
        assert %Axon.Parameter{initializer: :zeros} = beta
      end
    end

    test "works with name" do
      for norm <- @normalization_layers do
        assert %Axon{name: "norm"} = apply(Axon, norm, [Axon.input({nil, 784}), [name: "norm"]])
      end
    end

    test "works with parameter initializer" do
      for norm <- @normalization_layers do
        assert %Axon{params: %{"gamma" => gamma, "beta" => beta}} =
                 apply(Axon, norm, [
                   Axon.input({nil, 784}),
                   [gamma_initializer: :lecun_normal, beta_initializer: :ones]
                 ])

        assert %Axon.Parameter{initializer: :lecun_normal} = gamma
        assert %Axon.Parameter{initializer: :ones} = beta
      end
    end

    test "fails on bad initializers" do
      for norm <- @normalization_layers do
        assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
          apply(Axon, norm, [Axon.input({nil, 784}), [gamma_initializer: :foo]])
        end

        assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
          apply(Axon, norm, [Axon.input({nil, 784}), [beta_initializer: :foo]])
        end
      end
    end
  end

  describe "group normalization" do
    test "works with defaults" do
      assert %Axon{op: :group_norm, params: %{"gamma" => gamma, "beta" => beta}, opts: opts} =
               Axon.input({nil, 3, 28, 28}) |> Axon.group_norm(3)

      assert opts[:channel_index] == 1
      assert opts[:epsilon] == 1.0e-5
      assert opts[:group_size] == 3

      assert %Axon.Parameter{initializer: :glorot_uniform} = gamma
      assert %Axon.Parameter{initializer: :zeros} = beta
    end

    test "works with names" do
      assert %Axon{name: "norm"} = Axon.input({nil, 784}) |> Axon.group_norm(3, name: "norm")
    end

    test "works with parameter initializer" do
      assert %Axon{params: %{"gamma" => gamma, "beta" => beta}} =
               Axon.input({nil, 3, 28, 28})
               |> Axon.group_norm(3, gamma_initializer: :lecun_normal, beta_initializer: :ones)

      assert %Axon.Parameter{initializer: :lecun_normal} = gamma
      assert %Axon.Parameter{initializer: :ones} = beta
    end

    test "fails on bad initializer" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        apply(Axon, :group_norm, [Axon.input({nil, 784}), 3, [gamma_initializer: :foo]])
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        apply(Axon, :group_norm, [Axon.input({nil, 784}), 3, [beta_initializer: :foo]])
      end
    end
  end

  describe "flatten" do
    test "works with defaults" do
      assert %Axon{op: :flatten, output_shape: {nil, 784}} =
               Axon.input({nil, 1, 28, 28}) |> Axon.flatten()
    end

    test "works with names" do
      assert %Axon{name: "flatten"} =
               Axon.input({nil, 1, 28, 28}) |> Axon.flatten(name: "flatten")
    end
  end

  describe "concatenate" do
    test "works with 2 inputs" do
      assert %Axon{op: :concatenate, parent: [%Axon{}, %Axon{}]} =
               Axon.concatenate(Axon.input({nil, 32}), Axon.input({nil, 32}))
    end

    test "works with many inputs" do
      assert %Axon{op: :concatenate, parent: [%Axon{}, %Axon{}, %Axon{}]} =
               Axon.concatenate([
                 Axon.input({nil, 32}),
                 Axon.input({nil, 32}),
                 Axon.input({nil, 32})
               ])
    end
  end

  @element_wise_layers [:add, :subtract, :multiply]

  describe "element-wise layers" do
    test "works with 2 inputs" do
      for op <- @element_wise_layers do
        assert %Axon{op: op1, parent: [%Axon{}, %Axon{}]} =
                 apply(Axon, op, [Axon.input({nil, 32}), Axon.input({nil, 32})])

        assert op1 == op
      end
    end

    test "works with many inputs" do
      for op <- @element_wise_layers do
        assert %Axon{op: op1, parent: [%Axon{}, %Axon{}, %Axon{}]} =
                 apply(Axon, op, [
                   [Axon.input({nil, 32}), Axon.input({nil, 32}), Axon.input({nil, 32})]
                 ])

        assert op1 == op
      end
    end

    test "raises on bad shapes" do
      for op <- @element_wise_layers do
        assert_raise ArgumentError, ~r/all input shapes must match/, fn ->
          apply(Axon, op, [[Axon.input({nil, 32}), Axon.input({nil, 64})]])
        end
      end
    end
  end

  describe "nx" do
    test "works with defaults" do
      assert %Axon{output_shape: {nil, 32}} =
               Axon.input({nil, 32}) |> Axon.nx(fn x -> Nx.erf(x) end)
    end
  end

  describe "embedding" do
    test "works with defaults" do
      assert %Axon{output_shape: {nil, 10, 32}} =
               Axon.input({nil, 10}) |> Axon.embedding(128, 32, name: "embedding")
    end
  end

  describe "reshape" do
    test "works with batch input" do
      assert %Axon{output_shape: {nil, 3, 3}} = Axon.input({nil, 9}) |> Axon.reshape({3, 3})
    end

    test "works with constant input" do
      assert %Axon{output_shape: {1, 2, 3}} =
               Axon.constant(Nx.iota({6})) |> Axon.reshape({1, 2, 3})
    end
  end

  describe "transpose" do
    test "works with batch input" do
      assert %Axon{output_shape: {nil, 1, 2}} = Axon.input({nil, 2, 1}) |> Axon.transpose([1, 0])
    end

    test "works with constant input" do
      assert %Axon{output_shape: {1, 2, 3}} =
               Axon.constant(Nx.iota({3, 2, 1})) |> Axon.transpose([2, 1, 0])
    end
  end

  # TODO(seanmor5): Move/replace all with compiler_test
  describe "execution" do
    test "compile returns init and predict" do
      {init_fn, predict_fn} =
        Axon.input({nil, 6})
        |> Axon.dense(6, kernel_initializer: :identity, name: "dense")
        |> Axon.compile()

      assert %{"dense_kernel" => kernel, "dense_bias" => bias} = params = init_fn.()
      assert kernel == Nx.eye({6, 6}, type: {:f, 32})
      assert bias == Axon.Initializers.zeros(shape: {6})

      assert predict_fn.(params, Nx.iota({1, 6})) == Nx.iota({1, 6}, type: {:f, 32})
    end

    def model do
      Axon.input({nil, 6}) |> Axon.dense(6, kernel_initializer: :identity, name: "dense")
    end

    defn init do
      Axon.init(model())
    end

    test "init works inside defn" do
      assert init() == %{
               "dense_kernel" => Nx.eye({6, 6}, type: {:f, 32}),
               "dense_bias" => Axon.Initializers.zeros(shape: {6})
             }
    end

    test "init works outside defn" do
      assert Axon.init(model()) == %{
               "dense_kernel" => Nx.eye({6, 6}, type: {:f, 32}),
               "dense_bias" => Axon.Initializers.zeros(shape: {6})
             }
    end

    defn predict(params, input) do
      Axon.predict(model(), params, input)
    end

    test "predict works inside defn" do
      assert predict(init(), Nx.iota({1, 6})) == Nx.iota({1, 6}, type: {:f, 32})
    end

    test "predict works outside defn" do
      assert Axon.predict(model(), init(), Nx.iota({1, 6})) == Nx.iota({1, 6}, type: {:f, 32})
    end
  end

  describe "model freezing" do
    test "sets metadata correctly" do
      model =
        Axon.input({nil, 784})
        |> Axon.dense(128)
        |> Axon.freeze()

      assert %Axon{params: %{"kernel" => %{frozen: true}, "bias" => %{frozen: true}}} = model

      assert %Axon{params: %{"kernel" => %{frozen: false}, "bias" => %{frozen: false}}} =
               model |> Axon.dense(10)
    end
  end

  describe "inspection" do
    test "works with basic model" do
      model =
        Axon.input({nil, 784}, name: "input")
        |> Axon.dense(128, name: "dense1")
        |> Axon.dense(10, name: "dense2")
        |> Axon.softmax(name: "softmax")

      assert inspect(model) == """
             -----------------------------------------------
                                  Model
             ===============================================
              Layer                 Shape        Parameters
             ===============================================
              input ( input )       {nil, 784}   0
              dense1 ( dense )      {nil, 128}   100480
              dense2 ( dense )      {nil, 10}    1290
              softmax ( softmax )   {nil, 10}    0
             -----------------------------------------------
             """
    end

    test "works with complex model" do
      residual = fn x ->
        x
        |> Axon.dense(128, name: "residual_dense")
        |> Axon.add(x, name: "residual_add")
      end

      model =
        Axon.input({nil, 784}, name: "input")
        |> Axon.dense(128, name: "dense")
        |> residual.()
        |> Axon.dense(10, name: "dense2")
        |> Axon.softmax(name: "softmax")

      assert inspect(model) == """
             ----------------------------------------------------------------------------
                                                Model
             ============================================================================
              Layer                                              Shape        Parameters
             ============================================================================
              input ( input )                                    {nil, 784}   0
              dense ( dense )                                    {nil, 128}   100480
              residual_dense ( dense )                           {nil, 128}   16512
              residual_add ( add ["residual_dense", "dense"] )   {nil, 128}   0
              dense2 ( dense )                                   {nil, 10}    1290
              softmax ( softmax )                                {nil, 10}    0
             ----------------------------------------------------------------------------
             """
    end
  end
end
