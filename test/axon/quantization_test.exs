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
