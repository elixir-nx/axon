defmodule AxonTest do
  use ExUnit.Case
  doctest Axon
  import AxonTestUtil

  describe "input" do
    test "works with defaults" do
      assert %Axon{op: :input, parent: []} = Axon.input({32, 1, 28, 28}, "input")
    end

    test "raises on bad default value" do
      assert_raise ArgumentError, ~r/default input value/, fn ->
        Axon.input({nil, 1}, "foo", default: :bad_default)
      end
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
             } = Axon.input({nil, 784}, "input") |> Axon.dense(128)

      assert %Axon.Parameter{initializer: :glorot_uniform} = weight
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with parameter initializer" do
      assert %Axon{op: :dense, parameters: [weight, bias]} =
               Axon.input({nil, 784}, "input")
               |> Axon.dense(128, kernel_initializer: :lecun_normal, bias_initializer: :ones)

      assert %Axon.Parameter{initializer: :lecun_normal} = weight
      assert %Axon.Parameter{initializer: :ones} = bias
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :dense}]} =
               Axon.input({nil, 784}, "input") |> Axon.dense(128, activation: :relu)
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 784}, "input") |> Axon.dense(128, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 784}, "input") |> Axon.dense(128, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{parameters: [_]} =
               Axon.input({nil, 784}, "input") |> Axon.dense(128, use_bias: false)
    end
  end

  describe "conv" do
    test "works with defaults" do
      assert %Axon{op: :conv, parameters: [kernel, bias], opts: opts} =
               Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(64)

      assert opts[:padding] == :valid
      assert opts[:strides] == [1, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]

      assert %Axon.Parameter{initializer: :glorot_uniform} = kernel
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :conv}]} =
               Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(64, activation: :relu)
    end

    test "works with options" do
      assert %Axon{op: :conv, opts: opts, parameters: [kernel, bias]} =
               Axon.input({nil, 1, 28, 28}, "input")
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
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(128, kernel_size: :foo)
      end

      assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(128, strides: :foo)
      end

      assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(128, padding: :foo)
      end

      assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(128, input_dilation: :foo)
      end

      assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(128, kernel_dilation: :foo)
      end
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(128, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.conv(128, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{parameters: [_]} =
               Axon.input({nil, 1, 2}, "input") |> Axon.conv(2, use_bias: false)
    end
  end

  describe "depthwise_conv" do
    test "works with defaults" do
      assert %Axon{op: :depthwise_conv, parameters: [kernel, bias], opts: opts} =
               Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == [1, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]

      assert %Axon.Parameter{initializer: :glorot_uniform} = kernel
      assert %Axon.Parameter{initializer: :zeros} = bias
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :depthwise_conv}]} =
               Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(64, activation: :relu)
    end

    test "works with options" do
      assert %Axon{op: :depthwise_conv, opts: opts, parameters: [kernel, bias]} =
               Axon.input({nil, 1, 28, 28}, "input")
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
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(3, kernel_size: :foo)
      end

      assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(3, strides: :foo)
      end

      assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(3, padding: :foo)
      end

      assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(3, input_dilation: :foo)
      end

      assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(3, kernel_dilation: :foo)
      end
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.depthwise_conv(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{parameters: [_]} =
               Axon.input({nil, 1, 2}, "input") |> Axon.depthwise_conv(1, use_bias: false)
    end
  end

  describe "separable_conv2d" do
    test "works with defaults" do
      assert %Axon{
               op: :separable_conv2d,
               parameters: [k1, b1, k2, b2],
               opts: opts
             } = Axon.input({nil, 1, 28, 28}, "input") |> Axon.separable_conv2d(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == [1, 1]
      assert opts[:kernel_dilation] == [1, 1]
      assert opts[:input_dilation] == [1, 1]

      assert %Axon.Parameter{initializer: :glorot_uniform} = k1
      assert %Axon.Parameter{initializer: :glorot_uniform} = k2
      assert %Axon.Parameter{initializer: :zeros} = b1
      assert %Axon.Parameter{initializer: :zeros} = b2
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :separable_conv2d}]} =
               Axon.input({nil, 1, 28, 28}, "input")
               |> Axon.separable_conv2d(3, activation: :relu)
    end

    test "works with options" do
      assert %Axon{
               op: :separable_conv2d,
               opts: opts,
               parameters: [k1, b1, k2, b2]
             } =
               Axon.input({nil, 1, 28, 28}, "input")
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
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.separable_conv2d(3, kernel_size: :foo)
      end

      assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.separable_conv2d(3, strides: :foo)
      end

      assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.separable_conv2d(3, padding: :foo)
      end

      assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.separable_conv2d(3, input_dilation: :foo)
      end

      assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.separable_conv2d(3, kernel_dilation: :foo)
      end
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}, "input")
        |> Axon.separable_conv2d(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28}, "input") |> Axon.separable_conv2d(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: _, parameters: [_, _]} =
               Axon.input({nil, 1, 2, 2}, "input") |> Axon.separable_conv2d(1, use_bias: false)
    end
  end

  describe "separable_conv3d" do
    test "works with defaults" do
      assert %Axon{
               op: :separable_conv3d,
               parameters: [k1, b1, k2, b2, k3, b3],
               opts: opts
             } = Axon.input({nil, 1, 28, 28, 3}, "input") |> Axon.separable_conv3d(3)

      assert opts[:padding] == :valid
      assert opts[:strides] == [1, 1, 1]
      assert opts[:kernel_dilation] == [1, 1, 1]
      assert opts[:input_dilation] == [1, 1, 1]

      assert %Axon.Parameter{initializer: :glorot_uniform} = k1
      assert %Axon.Parameter{initializer: :glorot_uniform} = k2
      assert %Axon.Parameter{initializer: :glorot_uniform} = k3
      assert %Axon.Parameter{initializer: :zeros} = b1
      assert %Axon.Parameter{initializer: :zeros} = b2
      assert %Axon.Parameter{initializer: :zeros} = b3
    end

    test "works with activation" do
      assert %Axon{op: :relu, parent: [%Axon{op: :separable_conv3d}]} =
               Axon.input({nil, 1, 28, 28, 3}, "input")
               |> Axon.separable_conv3d(3, activation: :relu)
    end

    test "works with options" do
      assert %Axon{
               op: :separable_conv3d,
               opts: opts,
               parameters: [k1, b1, k2, b2, k3, b3]
             } =
               Axon.input({nil, 1, 28, 28, 3}, "input")
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
        Axon.input({nil, 1, 28, 28, 3}, "input") |> Axon.separable_conv3d(3, kernel_size: {1, 1})
      end

      assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
        Axon.input({nil, 1, 28, 28, 3}, "input") |> Axon.separable_conv3d(3, strides: :foo)
      end

      assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
        Axon.input({nil, 1, 28, 28, 3}, "input") |> Axon.separable_conv3d(3, padding: :foo)
      end

      assert_raise ArgumentError, ~r/expected :input_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28, 3}, "input") |> Axon.separable_conv3d(3, input_dilation: :foo)
      end

      assert_raise ArgumentError, ~r/expected :kernel_dilation to be/, fn ->
        Axon.input({nil, 1, 28, 28, 3}, "input")
        |> Axon.separable_conv3d(3, kernel_dilation: [1, 1])
      end
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28, 3}, "input")
        |> Axon.separable_conv3d(3, kernel_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.input({nil, 1, 28, 28, 3}, "input")
        |> Axon.separable_conv3d(3, bias_initializer: :foo)
      end
    end

    test "works with use_bias false" do
      assert %Axon{op: _, parameters: [_, _, _]} =
               Axon.input({nil, 1, 2, 2, 2}, "input") |> Axon.separable_conv3d(1, use_bias: false)
    end
  end

  @activation_layers [:celu, :elu, :exp, :gelu, :hard_sigmoid, :hard_silu, :hard_tanh] ++
                       [:leaky_relu, :linear, :log_sigmoid, :relu, :relu6] ++
                       [:sigmoid, :silu, :selu, :softmax, :softplus, :softsign, :tanh]

  describe "activation" do
    test "works with valid activation" do
      for act <- @activation_layers do
        assert %Axon{op: act1} = Axon.input({nil, 32}, "input") |> Axon.activation(act)
        assert act1 == act
        assert %Axon{op: act2} = apply(Axon, act, [Axon.input({nil, 32}, "input")])
        assert act2 == act
      end
    end
  end

  @dropout_layers [:dropout, :feature_alpha_dropout, :spatial_dropout, :alpha_dropout]

  describe "dropout" do
    test "works with defaults" do
      for dropout <- @dropout_layers do
        assert %Axon{op: drop1, opts: opts} =
                 apply(Axon, dropout, [Axon.input({nil, 32}, "input")])

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
                 apply(Axon, pool, [Axon.input({nil, 1, 28, 28}, "input")])

        assert pool1 == pool

        assert opts[:padding] == :valid
        assert opts[:strides] == [1, 1]
        assert opts[:kernel_size] == {1, 1}
      end
    end

    test "fails on bad options" do
      for pool <- @pooling_layers do
        assert_raise ArgumentError, ~r/expected :strides to be/, fn ->
          apply(Axon, pool, [Axon.input({nil, 1, 28, 28}, "input"), [strides: :foo]])
        end

        assert_raise ArgumentError, ~r/expected :kernel_size to be/, fn ->
          apply(Axon, pool, [Axon.input({nil, 1, 28, 28}, "input"), [kernel_size: :foo]])
        end

        assert_raise ArgumentError, ~r/invalid padding mode/, fn ->
          apply(Axon, pool, [Axon.input({nil, 1, 28, 28}, "input"), [padding: :foo]])
        end
      end
    end
  end

  @adaptive_pooling_layers [:adaptive_avg_pool, :adaptive_max_pool, :adaptive_lp_pool]

  describe "adaptive pooling" do
    test "works with options" do
      for pool <- @adaptive_pooling_layers do
        assert %Axon{} =
                 apply(Axon, pool, [
                   Axon.input({nil, 1, 28, 28}, "input"),
                   [output_size: {26, 26}, name: "pool"]
                 ])
      end
    end
  end

  @global_pooling_layers [:global_avg_pool, :global_max_pool, :global_lp_pool]

  describe "global pooling" do
    test "works with options" do
      for pool <- @global_pooling_layers do
        assert %Axon{output_shape: {nil, 1}} =
                 apply(Axon, pool, [
                   Axon.input({nil, 1, 28, 28}, "input"),
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
                 apply(Axon, norm, [Axon.input({nil, 784}, "input")])

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
                   Axon.input({nil, 784}, "input"),
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
          apply(Axon, norm, [Axon.input({nil, 784}, "input"), [gamma_initializer: :foo]])
        end

        assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
          apply(Axon, norm, [Axon.input({nil, 784}, "input"), [beta_initializer: :foo]])
        end
      end
    end
  end

  describe "layer normalization" do
    test "works with defaults" do
      assert %Axon{op: :layer_norm, opts: opts, parameters: [gamma, beta]} =
               Axon.layer_norm(Axon.input({nil, 784}, "input"))

      assert opts[:channel_index] == 1
      assert opts[:epsilon] == 1.0e-5

      assert %Axon.Parameter{initializer: :glorot_uniform} = gamma
      assert %Axon.Parameter{initializer: :zeros} = beta
    end

    test "works with parameter initializer" do
      assert %Axon{parameters: [gamma, beta]} =
               Axon.layer_norm(Axon.input({nil, 784}, "input"),
                 gamma_initializer: :lecun_normal,
                 beta_initializer: :ones
               )

      assert %Axon.Parameter{initializer: :lecun_normal} = gamma
      assert %Axon.Parameter{initializer: :ones} = beta
    end

    test "fails on bad initializers" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.layer_norm(Axon.input({nil, 784}, "input"), gamma_initializer: :foo)
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        Axon.layer_norm(Axon.input({nil, 784}, "input"), beta_initializer: :foo)
      end
    end
  end

  describe "group normalization" do
    test "works with defaults" do
      assert %Axon{op: :group_norm, parameters: [gamma, beta], opts: opts} =
               Axon.input({nil, 3, 28, 28}, "input") |> Axon.group_norm(3)

      assert opts[:channel_index] == 1
      assert opts[:epsilon] == 1.0e-5
      assert opts[:group_size] == 3

      assert %Axon.Parameter{initializer: :glorot_uniform} = gamma
      assert %Axon.Parameter{initializer: :zeros} = beta
    end

    test "works with parameter initializer" do
      assert %Axon{parameters: [gamma, beta]} =
               Axon.input({nil, 3, 28, 28}, "input")
               |> Axon.group_norm(3, gamma_initializer: :lecun_normal, beta_initializer: :ones)

      assert %Axon.Parameter{initializer: :lecun_normal} = gamma
      assert %Axon.Parameter{initializer: :ones} = beta
    end

    test "fails on bad initializer" do
      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        apply(Axon, :group_norm, [Axon.input({nil, 784}, "input"), 3, [gamma_initializer: :foo]])
      end

      assert_raise ArgumentError, ~r/initializer must be one of/, fn ->
        apply(Axon, :group_norm, [Axon.input({nil, 784}, "input"), 3, [beta_initializer: :foo]])
      end
    end
  end

  describe "flatten" do
    test "works with defaults" do
      assert %Axon{op: :flatten, output_shape: {nil, 784}} =
               Axon.input({nil, 1, 28, 28}, "input") |> Axon.flatten()
    end
  end

  describe "concatenate" do
    test "works with 2 inputs" do
      assert %Axon{op: :concatenate, parent: [%Axon{parent: [{%Axon{}, %Axon{}}]}]} =
               Axon.concatenate(Axon.input({nil, 32}, "input"), Axon.input({nil, 32}, "input"))
    end

    test "works with many inputs" do
      assert %Axon{op: :concatenate, parent: [%Axon{parent: [{%Axon{}, %Axon{}, %Axon{}}]}]} =
               Axon.concatenate([
                 Axon.input({nil, 32}, "input"),
                 Axon.input({nil, 32}, "input"),
                 Axon.input({nil, 32}, "input")
               ])
    end
  end

  @element_wise_layers [:add, :subtract, :multiply]

  describe "element-wise layers" do
    test "works with 2 inputs" do
      for op <- @element_wise_layers do
        assert %Axon{op: op1, parent: [%Axon{parent: [{%Axon{}, %Axon{}}]}]} =
                 apply(Axon, op, [Axon.input({nil, 32}, "input"), Axon.input({nil, 32}, "input")])

        assert op1 == op
      end
    end

    test "works with many inputs" do
      for op <- @element_wise_layers do
        assert %Axon{op: op1, parent: [%Axon{parent: [{%Axon{}, %Axon{}, %Axon{}}]}]} =
                 apply(Axon, op, [
                   [
                     Axon.input({nil, 32}, "input"),
                     Axon.input({nil, 32}, "input"),
                     Axon.input({nil, 32}, "input")
                   ]
                 ])

        assert op1 == op
      end
    end

    test "raises on bad shapes" do
      for op <- @element_wise_layers do
        assert_raise ArgumentError, ~r/cannot broadcast tensor/, fn ->
          apply(Axon, op, [[Axon.input({nil, 32}, "input"), Axon.input({nil, 64}, "input")]])
        end
      end
    end
  end

  describe "nx" do
    test "works with defaults" do
      assert %Axon{output_shape: {nil, 32}} =
               Axon.input({nil, 32}, "input") |> Axon.nx(fn x -> Nx.erf(x) end)
    end
  end

  describe "embedding" do
    test "works with defaults" do
      assert %Axon{output_shape: {nil, 10, 32}} =
               Axon.input({nil, 10}, "input") |> Axon.embedding(128, 32, name: "embedding")
    end
  end

  describe "reshape" do
    test "works with batch input" do
      assert %Axon{output_shape: {nil, 3, 3}} =
               Axon.input({nil, 9}, "input") |> Axon.reshape({3, 3})
    end

    test "works with constant input" do
      assert %Axon{output_shape: {1, 2, 3}} =
               Axon.constant(Nx.iota({6})) |> Axon.reshape({1, 2, 3})
    end
  end

  describe "transpose" do
    test "works with batch input" do
      assert %Axon{output_shape: {nil, 1, 2}} =
               Axon.input({nil, 2, 1}, "input") |> Axon.transpose([1, 0])
    end

    test "works with constant input" do
      assert %Axon{output_shape: {1, 2, 3}} =
               Axon.constant(Nx.iota({3, 2, 1}))
               |> Axon.transpose([2, 1, 0], ignore_batch?: false)
    end
  end

  # TODO(seanmor5): Move/replace all with compiler_test
  describe "execution" do
    test "compile returns init and predict" do
      {init_fn, predict_fn} =
        Axon.input({nil, 6}, "input")
        |> Axon.dense(6, kernel_initializer: :identity, name: "dense")
        |> Axon.compile()

      assert %{"dense" => %{"kernel" => kernel, "bias" => bias}} = params = init_fn.(%{})
      assert kernel == Nx.eye({6, 6}, type: {:f, 32})
      assert bias == zeros({6})

      assert predict_fn.(params, Nx.iota({1, 6})) == Nx.iota({1, 6}, type: {:f, 32})
    end

    def model do
      Axon.input({nil, 6}, "input") |> Axon.dense(6, kernel_initializer: :identity, name: "dense")
    end

    test "init works outside defn" do
      assert Axon.init(model()) == %{
               "dense" => %{
                 "kernel" => Nx.eye({6, 6}, type: {:f, 32}),
                 "bias" => zeros({6})
               }
             }
    end

    test "predict works outside defn" do
      model = model()

      assert Axon.predict(model, Axon.init(model), Nx.iota({1, 6})) ==
               Nx.iota({1, 6}, type: {:f, 32})
    end
  end

  describe "model freezing" do
    test "sets metadata correctly" do
      model =
        Axon.input({nil, 784}, "input")
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
        Axon.input({nil, 784}, "input")
        |> Axon.dense(128, name: "dense1")
        |> Axon.dense(10, name: "dense2")
        |> Axon.softmax(name: "softmax")

      assert inspect(model) == """
             -------------------------------------------------------------------------------------------------
                                                           Model
             =================================================================================================
              Layer                           Shape        Policy              Parameters   Parameters Memory
             =================================================================================================
              input ( input )                 {nil, 784}   p=f32 c=f32 o=f32   0            0 bytes
              dense1 ( dense["input"] )       {nil, 128}   p=f32 c=f32 o=f32   100480       401920 bytes
              dense2 ( dense["dense1"] )      {nil, 10}    p=f32 c=f32 o=f32   1290         5160 bytes
              softmax ( softmax["dense2"] )   {nil, 10}    p=f32 c=f32 o=f32   0            0 bytes
             -------------------------------------------------------------------------------------------------
             Total Parameters: 101770
             Total Parameters Memory: 407080 bytes
             Inputs: %{"input" => {nil, 784}}
             """
    end

    test "works with complex model" do
      residual = fn x ->
        x
        |> Axon.dense(128, name: "residual_dense")
        |> Axon.add(x, name: "residual_add")
      end

      model =
        Axon.input({nil, 784}, "input")
        |> Axon.dense(128, name: "dense")
        |> residual.()
        |> Axon.dense(10, name: "dense2")
        |> Axon.softmax(name: "softmax")

      assert inspect(model) == """
             ---------------------------------------------------------------------------------------------------------------------------------------
                                                                              Model
             =======================================================================================================================================
              Layer                                                   Shape                      Policy              Parameters   Parameters Memory
             =======================================================================================================================================
              input ( input )                                         {nil, 784}                 p=f32 c=f32 o=f32   0            0 bytes
              dense ( dense["input"] )                                {nil, 128}                 p=f32 c=f32 o=f32   100480       401920 bytes
              residual_dense ( dense["dense"] )                       {nil, 128}                 p=f32 c=f32 o=f32   16512        66048 bytes
              container_0 ( container {"residual_dense", "dense"} )   {{nil, 128}, {nil, 128}}   p=f32 c=f32 o=f32   0            0 bytes
              residual_add ( add["container_0"] )                     {nil, 128}                 p=f32 c=f32 o=f32   0            0 bytes
              dense2 ( dense["residual_add"] )                        {nil, 10}                  p=f32 c=f32 o=f32   1290         5160 bytes
              softmax ( softmax["dense2"] )                           {nil, 10}                  p=f32 c=f32 o=f32   0            0 bytes
             ---------------------------------------------------------------------------------------------------------------------------------------
             Total Parameters: 118282
             Total Parameters Memory: 473128 bytes
             Inputs: %{"input" => {nil, 784}}
             """
    end

    test "works with rnns" do
      {_, out_sequence} = Axon.input({nil, 32, 10}, "input_0") |> Axon.lstm(64, name: "lstm")

      assert inspect(out_sequence) == """
             -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                                                     Model
             =====================================================================================================================================================================================
              Layer                                                                            Shape                                           Policy              Parameters   Parameters Memory
             =====================================================================================================================================================================================
              input_0 ( input )                                                                {nil, 32, 10}                                   p=f32 c=f32 o=f32   0            0 bytes
              lstm_c_hidden_state ( recurrent_state["input_0"] )                               {nil, 1, 64}                                    p=f32 c=f32 o=f32   0            0 bytes
              lstm_h_hidden_state ( recurrent_state["input_0"] )                               {nil, 1, 64}                                    p=f32 c=f32 o=f32   0            0 bytes
              lstm_hidden_state ( container {"lstm_c_hidden_state", "lstm_h_hidden_state"} )   {{nil, 1, 64}, {nil, 1, 64}}                    p=f32 c=f32 o=f32   0            0 bytes
              lstm ( lstm["input_0", "lstm_hidden_state"] )                                    {{{nil, 1, 64}, {nil, 1, 64}}, {nil, 32, 64}}   p=f32 c=f32 o=f32   19200        76800 bytes
              lstm_output_sequence ( elem["lstm"] )                                            {nil, 32, 64}                                   p=f32 c=f32 o=f32   0            0 bytes
             -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
             Total Parameters: 19200
             Total Parameters Memory: 76800 bytes
             Inputs: %{"input_0" => {nil, 32, 10}}
             """
    end

    test "works with single namespace" do
      model = Axon.input({nil, 1}, "input_0") |> Axon.dense(2) |> Axon.namespace("x")

      assert inspect(model) == """
             -------------------------------------------------------------------------------
                                                  Model
             ===============================================================================
              Layer           Shape      Policy              Parameters   Parameters Memory
             ===============================================================================
              x ( input_0 )   {nil, 2}   p=f32 c=f32 o=f32   4            16 bytes
             -------------------------------------------------------------------------------
             Total Parameters: 4
             Total Parameters Memory: 16 bytes
             Inputs: %{"input_0" => {nil, 1}}
             """
    end

    test "works with nested namespace" do
      model =
        Axon.input({nil, 1}, "input_0")
        |> Axon.dense(2)
        |> Axon.namespace("x")
        |> Axon.namespace("y")

      assert inspect(model) == """
             -------------------------------------------------------------------------------
                                                  Model
             ===============================================================================
              Layer           Shape      Policy              Parameters   Parameters Memory
             ===============================================================================
              y ( input_0 )   {nil, 2}   p=f32 c=f32 o=f32   4            16 bytes
             -------------------------------------------------------------------------------
             Total Parameters: 4
             Total Parameters Memory: 16 bytes
             Inputs: %{"input_0" => {nil, 1}}
             """
    end

    test "works with multiple namespaces" do
      x = Axon.input({nil, 1}, "input_0") |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input({nil, 1}, "input_1") |> Axon.dense(2) |> Axon.namespace("y")

      model = Axon.add(x, y)

      assert inspect(model) == """
             ------------------------------------------------------------------------------------------------------------------
                                                                   Model
             ==================================================================================================================
              Layer                                  Shape                  Policy              Parameters   Parameters Memory
             ==================================================================================================================
              x ( input_0 )                          {nil, 2}               p=f32 c=f32 o=f32   4            16 bytes
              y ( input_1 )                          {nil, 2}               p=f32 c=f32 o=f32   4            16 bytes
              container_0 ( container {"x", "y"} )   {{nil, 2}, {nil, 2}}   p=f32 c=f32 o=f32   0            0 bytes
              add_0 ( add["container_0"] )           {nil, 2}               p=f32 c=f32 o=f32   0            0 bytes
             ------------------------------------------------------------------------------------------------------------------
             Total Parameters: 8
             Total Parameters Memory: 32 bytes
             Inputs: %{"input_0" => {nil, 1}, "input_1" => {nil, 1}}
             """
    end

    test "works with single namespace and no namespace" do
      x = Axon.input({nil, 1}, "input_0") |> Axon.dense(2) |> Axon.namespace("x")
      y = Axon.input({nil, 1}, "input_1") |> Axon.dense(2)

      model = Axon.add(x, y)

      assert inspect(model) == """
             ------------------------------------------------------------------------------------------------------------------------
                                                                      Model
             ========================================================================================================================
              Layer                                        Shape                  Policy              Parameters   Parameters Memory
             ========================================================================================================================
              x ( input_0 )                                {nil, 2}               p=f32 c=f32 o=f32   4            16 bytes
              input_1 ( input )                            {nil, 1}               p=f32 c=f32 o=f32   0            0 bytes
              dense_1 ( dense["input_1"] )                 {nil, 2}               p=f32 c=f32 o=f32   4            16 bytes
              container_0 ( container {"x", "dense_1"} )   {{nil, 2}, {nil, 2}}   p=f32 c=f32 o=f32   0            0 bytes
              add_0 ( add["container_0"] )                 {nil, 2}               p=f32 c=f32 o=f32   0            0 bytes
             ------------------------------------------------------------------------------------------------------------------------
             Total Parameters: 8
             Total Parameters Memory: 32 bytes
             Inputs: %{"input_0" => {nil, 1}, "input_1" => {nil, 1}}
             """
    end
  end

  describe "container" do
    test "correctly derives container" do
      model = Axon.input({nil, 1}, "input")
      assert Axon.predict(model, %{}, Nx.tensor([[1.0]])) == Nx.tensor([[1.0]])
    end

    test "shape inference works" do
      last_hidden_state = Axon.input({5, 128, 768}, "last_hidden_state")
      pooled = Axon.input({5, 768}, "pooled")

      assert %Axon{output_shape: {5, 128, 768}} =
               Axon.container({last_hidden_state, pooled}) |> Axon.nx(&elem(&1, 0))

      assert %Axon{output_shape: {5, 768}} =
               Axon.container({last_hidden_state, pooled}) |> Axon.nx(&elem(&1, 1))
    end
  end

  describe "serialization" do
    test "correctly serializes and deserializes simple container" do
      inp1 = Axon.input({nil, 1}, "input_0")
      inp2 = Axon.input({nil, 2}, "input_1")
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
      inp1 = Axon.input({nil, 1}, "input_0")
      inp2 = Axon.input({nil, 2}, "input_1")
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
      %Axon{name: name_fn, op: op} = Axon.input({nil, 1}, "a_binary_name")

      assert "a_binary_name" == name_fn.(op, input: 1)

      %Axon{name: name_fn, op: op} =
        Axon.input({nil, 1}, "input") |> Axon.dense(2, name: fn op, _ -> "custom_#{op}" end)

      assert "custom_#{op}" == name_fn.(op, input: 1)

      %Axon{name: name_fn, op: op} = Axon.input({nil, 1}, "input") |> Axon.dense(2, name: nil)

      assert "dense_10" == name_fn.(op, dense: 10)
    end

    @invalid_names [:atom, {"tuple"}, ["list"], 123]

    test "raises on invalid names" do
      Enum.each(@invalid_names, fn name ->
        assert_raise ArgumentError, fn ->
          Axon.input({nil, 1}, "input") |> Axon.dense(2, name: name)
        end
      end)
    end
  end
end
