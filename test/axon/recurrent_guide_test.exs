defmodule Axon.RecurrentGuideTest do
  @moduledoc """
  Tests that validate the examples in guides/model_creation/custom_recurrent_layers.livemd.
  Run with: mix test test/axon/recurrent_guide_test.exs
  """
  use Axon.Case, async: false

  # The custom recurrent layer from the guide, verbatim.
  defmodule SimpleRNN do
    import Nx.Defn

    # h_t = tanh(x_t · Wi + h_{t-1} · Wh + b)
    defn cell(input, {hidden}, mask, input_kernel, recurrent_kernel, bias) do
      candidate =
        Nx.tanh(
          Axon.Layers.dense(input, input_kernel, bias) +
            Axon.Layers.dense(hidden, recurrent_kernel, 0)
        )

      # mask is {batch, 1}; 1 means this step is padding, so keep the old state
      mask = Nx.broadcast(Nx.as_type(mask, :u8), hidden)
      new_hidden = Nx.select(mask, hidden, candidate)

      {new_hidden, {new_hidden}}
    end

    def simple_rnn(%Axon{} = x, units, opts \\ []) when is_integer(units) and units > 0 do
      opts = Keyword.validate!(opts, [:name, unroll: :dynamic, mask: Axon.constant(0)])

      # Shape functions receive one shape per Axon input of the layer, in order:
      # here the sequence and the mask.
      input_kernel =
        Axon.param("input_kernel", fn input_shape, _mask_shape ->
          {elem(input_shape, tuple_size(input_shape) - 1), units}
        end)

      recurrent_kernel = Axon.param("recurrent_kernel", {units, units})
      bias = Axon.param("bias", {units}, initializer: :zeros)

      out =
        Axon.layer(
          &simple_rnn_impl/6,
          [x, opts[:mask], input_kernel, recurrent_kernel, bias],
          name: opts[:name],
          op_name: :simple_rnn,
          unroll: opts[:unroll]
        )

      {Axon.elem(out, 0), out |> Axon.elem(1) |> Axon.elem(0)}
    end

    defn simple_rnn_impl(input, mask, input_kernel, recurrent_kernel, bias, opts \\ []) do
      opts = keyword!(opts, mode: :inference, unroll: :dynamic)

      batch = Nx.axis_size(input, 0)
      units = Nx.axis_size(recurrent_kernel, 0)
      carry = {Nx.broadcast(Nx.tensor(0, type: Nx.type(input)), {batch, units})}

      unroll(opts[:unroll], input, carry, mask, input_kernel, recurrent_kernel, bias)
    end

    deftransformp unroll(:static, input, carry, mask, wi, wh, b),
      do: Axon.Layers.static_unroll(&cell/6, input, carry, mask, wi, wh, b)

    deftransformp unroll(:dynamic, input, carry, mask, wi, wh, b),
      do: Axon.Layers.dynamic_unroll(&cell/6, input, carry, mask, wi, wh, b)
  end

  # A scaled-down version of the guide's synthetic "names" dataset.
  defmodule Names do
    @max_len 8

    def max_len, do: @max_len

    def word(lang) do
      for _ <- 1..Enum.random(3..@max_len), do: letter(lang)
    end

    defp letter(0),
      do: if(:rand.uniform() < 0.8, do: Enum.random(?a..?m), else: Enum.random(?n..?z))

    defp letter(1),
      do: if(:rand.uniform() < 0.8, do: Enum.random(?n..?z), else: Enum.random(?a..?m))

    # letters -> ids 1..26, then pad with 0 to max_len
    def encode(word) do
      ids = Enum.map(word, &(&1 - ?a + 1))
      ids ++ List.duplicate(0, @max_len - length(ids))
    end

    def batch(size) do
      labels = for _ <- 1..size, do: Enum.random(0..1)
      words = for lang <- labels, do: encode(word(lang))

      x = Nx.tensor(words)
      y = labels |> Nx.tensor() |> Nx.new_axis(-1) |> Nx.equal(Nx.tensor([0, 1]))
      {%{"tokens" => x}, y}
    end
  end

  @embedding_size 8
  @units 16

  defp build_model(opts \\ []) do
    tokens = Axon.input("tokens", shape: {nil, Names.max_len()})
    mask = Axon.mask(tokens, 0)

    {sequence, hidden} =
      tokens
      |> Axon.embedding(27, @embedding_size)
      |> SimpleRNN.simple_rnn(@units, [mask: mask, name: "rnn"] ++ opts)

    model = Axon.dense(hidden, 2, activation: :softmax)

    {model, sequence, hidden}
  end

  defp init_params(model) do
    {init_fn, _} = Axon.build(model)
    init_fn.(Nx.template({1, Names.max_len()}, :s64), Axon.ModelState.empty())
  end

  describe "custom recurrent layers guide examples" do
    test "static_unroll, dynamic_unroll and a map_reduce scan agree" do
      key = Nx.Random.key(42)
      {input, key} = Nx.Random.normal(key, shape: {2, 5, 3})
      {wi, key} = Nx.Random.normal(key, shape: {3, 4})
      {wh, _key} = Nx.Random.normal(key, shape: {4, 4})
      b = Nx.broadcast(0.0, {4})
      carry = {Nx.broadcast(0.0, {2, 4})}
      no_mask = Nx.tensor(0)

      {static_out, {static_carry}} =
        Axon.Layers.static_unroll(&SimpleRNN.cell/6, input, carry, no_mask, wi, wh, b)

      assert Nx.shape(static_out) == {2, 5, 4}
      assert Nx.shape(static_carry) == {2, 4}

      {dynamic_out, {dynamic_carry}} =
        Axon.Layers.dynamic_unroll(&SimpleRNN.cell/6, input, carry, no_mask, wi, wh, b)

      assert_all_close(static_out, dynamic_out)
      assert_all_close(static_carry, dynamic_carry)

      # The unroll is just a scan over the time axis
      {steps, _} =
        Enum.map_reduce(0..4, carry, fn t, carry ->
          SimpleRNN.cell(input[[.., t, ..]], carry, Nx.broadcast(0, {2, 1}), wi, wh, b)
        end)

      assert_all_close(Nx.stack(steps, axis: 1), static_out)

      # The final carry is the output of the last step
      assert_all_close(static_out[[.., 4, ..]], static_carry)
    end

    test "custom rnn layer initializes the expected parameters" do
      {model, _sequence, _hidden} = build_model()
      params = init_params(model)

      assert "rnn" in Map.keys(params.data)
      assert Nx.shape(params.data["rnn"]["input_kernel"]) == {@embedding_size, @units}
      assert Nx.shape(params.data["rnn"]["recurrent_kernel"]) == {@units, @units}
      assert Nx.shape(params.data["rnn"]["bias"]) == {@units}
    end

    test "mask freezes the state through padding" do
      {model, sequence, hidden} = build_model()
      params = init_params(model)

      # five real letters followed by three padding tokens
      word = Nx.tensor([[3, 7, 12, 5, 9, 0, 0, 0]])

      {_, seq_fn} = Axon.build(sequence)
      {_, hidden_fn} = Axon.build(hidden)
      outputs = seq_fn.(params, %{"tokens" => word})
      final = hidden_fn.(params, %{"tokens" => word})

      assert Nx.shape(outputs) == {1, Names.max_len(), @units}
      assert Nx.shape(final) == {1, @units}

      # the final state is the state after the last real letter...
      assert_all_close(outputs[[.., 4, ..]], final)
      # ...and it does not change through the padding
      assert_all_close(outputs[[.., 7, ..]], final)

      # without a mask the padding keeps updating the state; the layer names
      # match so the same params can be reused
      {_, unmasked_hidden} =
        Axon.input("tokens", shape: {nil, Names.max_len()})
        |> Axon.embedding(27, @embedding_size)
        |> SimpleRNN.simple_rnn(@units, name: "rnn")

      {_, unmasked_fn} = Axon.build(unmasked_hidden)
      unmasked = unmasked_fn.(params, %{"tokens" => word})
      refute Nx.to_number(Nx.all_close(unmasked, final)) == 1

      # static and dynamic unrolling agree inside a model too
      {_, _, static_hidden} = build_model(unroll: :static)
      {_, static_fn} = Axon.build(static_hidden)
      assert_all_close(static_fn.(params, %{"tokens" => word}), final)
    end

    test "trains on padded sequences" do
      :rand.seed(:exsss, {1, 2, 3})
      train_data = for _ <- 1..8, do: Names.batch(16)

      {model, _sequence, _hidden} = build_model()

      state =
        model
        |> Axon.Loop.trainer(
          :categorical_cross_entropy,
          Polaris.Optimizers.adam(learning_rate: 0.01),
          log: 0,
          seed: 42
        )
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.run(train_data, Axon.ModelState.empty(), epochs: 2)

      assert %Axon.ModelState{data: %{"rnn" => rnn_params}} = state.step_state[:model_state]
      assert Map.keys(rnn_params) |> Enum.sort() == ["bias", "input_kernel", "recurrent_kernel"]

      first_loss = Nx.to_number(state.metrics[0]["loss"])
      last_loss = Nx.to_number(state.metrics[1]["loss"])
      assert last_loss < first_loss

      assert Nx.to_number(state.metrics[1]["accuracy"]) > 0.8
    end
  end
end
