defmodule Axon.QuantizationTest do
  use Axon.Case, async: true

  alias Axon.ModelState
  alias Axon.Quantization.QTensor

  describe "quantize_model_state" do
    test "replaces dense kernels with quantized versions" do
      model =
        Axon.input("input")
        |> Axon.dense(10, activation: :relu)

      assert {init_fn, _} = Axon.build(model)
      assert %ModelState{} = model_state = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert %{data: %{"dense_0" => %{"kernel" => %QTensor{}}}} =
               Axon.Quantization.quantize_model_state(model, model_state)
    end
  end

  describe "quantize" do
    test "returns model and state that execute properly" do
      model =
        Axon.input("input")
        |> Axon.dense(10, activation: :relu)

      assert {init_fn, _} = Axon.build(model)
      assert %ModelState{} = model_state = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert {quantized_model, quantized_model_state} =
               Axon.Quantization.quantize(model, model_state)

      assert {_, predict_fn} = Axon.build(quantized_model)

      real_fn = fn %{data: %{"dense_0" => %{"kernel" => k, "bias" => b}}}, input ->
        input
        |> Axon.Quantization.Layers.weight_only_quantized_dense(k, b)
        |> Axon.Activations.relu()
      end

      inp = Nx.broadcast(1.0, {1, 1})
      assert_equal(predict_fn.(quantized_model_state, inp), real_fn.(quantized_model_state, inp))
    end
  end

  describe "quantize_model" do
    test "preserves metadata from the layer it replaces" do
      model =
        Axon.input("input", shape: {nil, 8})
        |> Axon.dense(4, name: "d", meta: %{units: 4, use_bias: true, tag: :attn})

      quantized = Axon.Quantization.quantize_model(model)

      metas =
        Axon.reduce_nodes(quantized, [], fn
          %Axon.Node{op_name: :input}, acc -> acc
          %Axon.Node{meta: meta}, acc -> [meta | acc]
        end)

      assert [meta] = metas
      assert meta[:tag] == :attn
      assert meta[:units] == 4
      assert meta[:use_bias] == true
    end
  end

  describe "QTensor.from_tensor/2" do
    test "quantizes float tensors regardless of precision" do
      for type <- [{:f, 32}, {:f, 64}, {:bf, 16}] do
        input = Nx.iota({4, 8}, type: type) |> Nx.subtract(16) |> Nx.divide(7)

        assert %QTensor{value: value, scale: scale} = QTensor.from_tensor(input)
        assert Nx.type(value) == {:s, 8}
        assert Nx.shape(value) == {4, 8}
        assert Nx.shape(scale) == {8}
      end
    end

    test "raises on non-float tensors" do
      assert_raise ArgumentError, ~r/expected a float tensor/, fn ->
        QTensor.from_tensor(Nx.iota({4, 8}, type: {:s, 32}))
      end
    end

    test "raises on tensors that are not rank 2" do
      assert_raise ArgumentError, ~r/expected a 2d tensor/, fn ->
        QTensor.from_tensor(Nx.iota({4, 8, 2}, type: {:f, 32}))
      end
    end
  end

  describe "weight_only_quantized_dense" do
    test "inits and executes properly" do
      model =
        Axon.input("input")
        |> Axon.Quantization.weight_only_quantized_dense(10)

      assert {init_fn, _} = Axon.build(model)
      assert %ModelState{} = model_state = init_fn.(Nx.template({1, 1}, :f32), ModelState.empty())

      assert {_, predict_fn} = Axon.build(model)
      assert predict_fn.(model_state, Nx.broadcast(1.0, {1, 1}))
    end
  end
end
