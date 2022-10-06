defmodule Axon.ServingTest do
  use Axon.Case

  defp single_in_single_out() do
    Axon.input("input") |> Axon.dense(32)
  end

  defp multi_in_single_out() do
    inp1 = Axon.input("input1")
    inp2 = Axon.input("input2")
    Axon.add(Axon.dense(inp1, 32), Axon.dense(inp2, 32))
  end

  defp optional() do
    inp1 = Axon.input("input1")
    inp2 = Axon.input("input2", optional: true)
    Axon.container(%{foo: Axon.dense(inp1, 32), bar: Axon.optional(inp2)})
  end

  defp deeply_nested() do
    inp1 = Axon.input("input1")
    inp2 = Axon.input("input2")

    out = %{
      foo: {inp1, %{bar: Axon.dense(inp1, 32)}},
      baz: %{bang: {inp2, Axon.dense(inp2, 32)}, buzz: {}}
    }

    Axon.container(out)
  end

  defp model(model, shapes) do
    templates = Map.new(shapes, fn {k, v} -> {k, Nx.template(v, :f32)} end)
    {init_fn, _} = Axon.build(model)
    {model, init_fn.(templates, %{})}
  end

  defp setup_predict(_context) do
    single_in_shape = %{"input" => {8, 16}}
    single_in_model = model(single_in_single_out(), single_in_shape)

    Axon.Serving.start_link(
      name: :single_in,
      model: single_in_model,
      shape: single_in_shape,
      batch_size: 8,
      batch_timeout: 50,
      compiler: test_compiler()
    )

    multi_in_single_out_shape = %{"input1" => {8, 8}, "input2" => {8, 16}}
    multi_in_single_out_model = model(multi_in_single_out(), multi_in_single_out_shape)

    Axon.Serving.start_link(
      name: :multi_in_single_out,
      model: multi_in_single_out_model,
      shape: multi_in_single_out_shape,
      batch_size: 8,
      batch_timeout: 50,
      compiler: test_compiler()
    )

    optional_shape = %{"input1" => {8, 8}}
    optional_model = model(optional(), optional_shape)

    Axon.Serving.start_link(
      name: :optional,
      model: optional_model,
      shape: optional_shape,
      batch_size: 8,
      batch_timeout: 50,
      compiler: test_compiler()
    )

    deeply_nested_shape = %{"input1" => {8, 8}, "input2" => {8, 16}}
    deeply_nested_model = model(deeply_nested(), deeply_nested_shape)

    Axon.Serving.start_link(
      name: :deeply_nested,
      model: deeply_nested_model,
      shape: deeply_nested_shape,
      batch_size: 8,
      batch_timeout: 50,
      compiler: test_compiler()
    )

    {:ok,
     %{
       single_in: single_in_model,
       multi_in_single_out: multi_in_single_out_model,
       optional: optional_model,
       deeply_nested: deeply_nested_model
     }}
  end

  describe "initialization" do
    test "initializes correctly with single-in, single-out model" do
      single_in_shape = %{"input" => {8, 16}}
      single_in_model = model(single_in_single_out(), single_in_shape)

      assert {:ok, pid} =
               Axon.Serving.start_link(
                 name: :single_in,
                 model: single_in_model,
                 shape: single_in_shape,
                 batch_size: 8,
                 batch_timeout: 50,
                 compiler: test_compiler()
               )

      GenServer.stop(pid)
    end

    test "initializes correctly with single-in, single-out model, nil batch" do
      single_in_shape = %{"input" => {nil, 16}}
      single_in_model = model(single_in_single_out(), %{"input" => {8, 16}})

      assert {:ok, pid} =
               Axon.Serving.start_link(
                 name: :single_in,
                 model: single_in_model,
                 shape: single_in_shape,
                 batch_size: 8,
                 batch_timeout: 50,
                 compiler: test_compiler()
               )

      GenServer.stop(pid)
    end

    test "initializes correctly with nil shape specified in model" do
      model = Axon.input("foo", shape: {nil, 16}) |> Axon.dense(32)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({1, 16}, :f32), %{})

      assert {:ok, pid} =
               Axon.Serving.start_link(
                 name: :model,
                 model: {model, params},
                 shape: %{"foo" => nil},
                 batch_size: 8,
                 batch_timeout: 50
               )

      GenServer.stop(pid)
    end

    test "initializes correctly without optional input" do
      inp1 = Axon.input("foo")
      inp2 = Axon.input("bar", optional: true)
      model = Axon.container(%{foo: inp1, bar: Axon.optional(inp2)})
      params = %{}

      assert {:ok, pid} =
               Axon.Serving.start_link(
                 name: :model,
                 model: {model, params},
                 shape: %{"foo" => {8, 16}},
                 batch_size: 8,
                 batch_timeout: 50
               )

      GenServer.stop(pid)
    end

    test "raises when shape is nil in both graph and config" do
      model = Axon.input("foo") |> Axon.dense(32)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({1, 16}, :f32), %{})

      assert_raise ArgumentError, ~r/invalid shape/, fn ->
        Axon.Serving.start_link(
          name: :bad,
          model: {model, params},
          shape: %{"foo" => nil},
          batch_size: 8,
          compiler: test_compiler()
        )
      end
    end

    test "raises when required shape is not present in config" do
      model = Axon.input("foo") |> Axon.dense(32)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({1, 16}, :f32), %{})

      assert_raise ArgumentError, ~r/must provide shape/, fn ->
        Axon.Serving.start_link(
          name: :bad,
          model: {model, params},
          shape: %{"bar" => {nil, 16}},
          batch_size: 8,
          compiler: test_compiler()
        )
      end
    end

    test "raises on mismatched shapes" do
      model = Axon.input("foo", shape: {nil, 8}) |> Axon.dense(32)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8}, :f32), %{})

      assert_raise ArgumentError, ~r/invalid shape/, fn ->
        Axon.Serving.start_link(
          name: :bad,
          model: {model, params},
          shape: %{"foo" => {nil, 16}},
          batch_size: 8,
          compiler: test_compiler()
        )
      end
    end

    test "raises on mismatched batch sizes in config" do
      model = Axon.input("foo") |> Axon.dense(32)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({1, 16}, :f32), %{})

      assert_raise ArgumentError, ~r/invalid shape/, fn ->
        Axon.Serving.start_link(
          name: :bad,
          model: {model, params},
          shape: %{"foo" => {8, 16}},
          batch_size: 16,
          compiler: test_compiler()
        )
      end
    end

    test "raises on mismatched batch sizes in graph" do
      model = Axon.input("foo", shape: {8, 16}) |> Axon.dense(32)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(Nx.template({8, 16}, :f32), %{})

      assert_raise ArgumentError, ~r/invalid shape/, fn ->
        Axon.Serving.start_link(
          name: :bad,
          model: {model, params},
          shape: %{"foo" => nil},
          batch_size: 16,
          compiler: test_compiler()
        )
      end
    end
  end

  describe "predict" do
    setup [:setup_predict]

    test "returns correctly with single unbatched predict and single in, single out model", %{
      single_in: {model, params}
    } do
      input = %{"input" => Nx.random_uniform({1, 16})}
      expected = Axon.predict(model, params, input)
      actual = Axon.Serving.predict(:single_in, input)

      assert_all_close(expected, actual, atol: 1.0e-4)
    end

    test "returns correctly with single predict multi-in, single out model", %{
      multi_in_single_out: {model, params}
    } do
      input = %{"input1" => Nx.random_uniform({1, 8}), "input2" => Nx.random_uniform({1, 16})}
      expected = Axon.predict(model, params, input)
      actual = Axon.Serving.predict(:multi_in_single_out, input)

      assert_all_close(expected, actual, atol: 1.0e-4)
    end

    test "returns correctly with single predict optional input model", %{
      optional: {model, params}
    } do
      input = %{"input1" => Nx.random_uniform({1, 8})}
      expected = Axon.predict(model, params, input)
      actual = Axon.Serving.predict(:optional, input)

      assert_all_close(expected, actual, atol: 1.0e-4)
    end

    test "returns correctly with single predict deeply nested model", %{
      deeply_nested: {model, params}
    } do
      input = %{"input1" => Nx.random_uniform({1, 8}), "input2" => Nx.random_uniform({1, 16})}
      expected = Axon.predict(model, params, input)
      actual = Axon.Serving.predict(:deeply_nested, input)

      assert_all_close(expected, actual, atol: 1.0e-4)
    end

    test "returns correctly with full-batch predict and single in, single out model", %{
      single_in: {model, params}
    } do
      0..7
      |> Enum.map(fn _idx ->
        inp = %{"input" => Nx.random_uniform({1, 16})}
        {inp, Axon.predict(model, params, inp)}
      end)
      |> Enum.map(fn {inp, expected} ->
        {Task.async(fn -> Axon.Serving.predict(:single_in, inp) end), expected}
      end)
      |> Enum.each(fn {actual_pid, expected} ->
        actual = Task.await(actual_pid)
        assert_all_close(expected, actual, atol: 1.0e-4)
      end)
    end

    test "returns correctly with full-batch predict and multi-in, single out model", %{
      multi_in_single_out: {model, params}
    } do
      0..7
      |> Enum.map(fn _idx ->
        inp = %{"input1" => Nx.random_uniform({1, 8}), "input2" => Nx.random_uniform({1, 16})}
        {inp, Axon.predict(model, params, inp)}
      end)
      |> Enum.map(fn {inp, expected} ->
        {Task.async(fn -> Axon.Serving.predict(:multi_in_single_out, inp) end), expected}
      end)
      |> Enum.each(fn {actual_pid, expected} ->
        actual = Task.await(actual_pid)
        assert_all_close(expected, actual, atol: 1.0e-4)
      end)
    end

    test "returns correctly with full-batch predict and optional model", %{
      optional: {model, params}
    } do
      0..7
      |> Enum.map(fn _idx ->
        inp = %{"input1" => Nx.random_uniform({1, 8})}
        {inp, Axon.predict(model, params, inp)}
      end)
      |> Enum.map(fn {inp, expected} ->
        {Task.async(fn -> Axon.Serving.predict(:optional, inp) end), expected}
      end)
      |> Enum.each(fn {actual_pid, expected} ->
        actual = Task.await(actual_pid)
        assert_all_close(expected, actual, atol: 1.0e-4)
      end)
    end

    test "returns correctly with full-batch predict and deeply nested model", %{
      deeply_nested: {model, params}
    } do
      0..7
      |> Enum.map(fn _idx ->
        inp = %{"input1" => Nx.random_uniform({1, 8}), "input2" => Nx.random_uniform({1, 16})}
        {inp, Axon.predict(model, params, inp)}
      end)
      |> Enum.map(fn {inp, expected} ->
        {Task.async(fn -> Axon.Serving.predict(:deeply_nested, inp) end), expected}
      end)
      |> Enum.each(fn {actual_pid, expected} ->
        actual = Task.await(actual_pid)
        assert_all_close(expected, actual, atol: 1.0e-4)
      end)
    end

    test "returns correctly with partial-batch predict timeout and single in, single out model",
         %{single_in: {model, params}} do
      0..4
      |> Enum.map(fn _idx ->
        inp = %{"input" => Nx.random_uniform({1, 16})}
        {inp, Axon.predict(model, params, inp)}
      end)
      |> Enum.map(fn {inp, expected} ->
        {Task.async(fn -> Axon.Serving.predict(:single_in, inp) end), expected}
      end)
      |> Enum.each(fn {actual_pid, expected} ->
        actual = Task.await(actual_pid)
        assert_all_close(expected, actual, atol: 1.0e-4)
      end)
    end

    test "returns correctly with partial-batch predict timeout and multi-in, single out model", %{
      multi_in_single_out: {model, params}
    } do
      0..4
      |> Enum.map(fn _idx ->
        inp = %{"input1" => Nx.random_uniform({1, 8}), "input2" => Nx.random_uniform({1, 16})}
        {inp, Axon.predict(model, params, inp)}
      end)
      |> Enum.map(fn {inp, expected} ->
        {Task.async(fn -> Axon.Serving.predict(:multi_in_single_out, inp) end), expected}
      end)
      |> Enum.each(fn {actual_pid, expected} ->
        actual = Task.await(actual_pid)
        assert_all_close(expected, actual, atol: 1.0e-4)
      end)
    end

    test "returns correctly with partial-batch predict timeout and optional model",
         %{optional: {model, params}} do
      0..4
      |> Enum.map(fn _idx ->
        inp = %{"input1" => Nx.random_uniform({1, 8})}
        {inp, Axon.predict(model, params, inp)}
      end)
      |> Enum.map(fn {inp, expected} ->
        {Task.async(fn -> Axon.Serving.predict(:optional, inp) end), expected}
      end)
      |> Enum.each(fn {actual_pid, expected} ->
        actual = Task.await(actual_pid)
        assert_all_close(expected, actual, atol: 1.0e-4)
      end)
    end

    test "returns correctly with partial-batch predict timeout and deeply nested model", %{
      deeply_nested: {model, params}
    } do
      0..4
      |> Enum.map(fn _idx ->
        inp = %{"input1" => Nx.random_uniform({1, 8}), "input2" => Nx.random_uniform({1, 16})}
        {inp, Axon.predict(model, params, inp)}
      end)
      |> Enum.map(fn {inp, expected} ->
        {Task.async(fn -> Axon.Serving.predict(:deeply_nested, inp) end), expected}
      end)
      |> Enum.each(fn {actual_pid, expected} ->
        actual = Task.await(actual_pid)
        assert_all_close(expected, actual, atol: 1.0e-4)
      end)
    end
  end
end
