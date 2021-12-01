defmodule Axon.RecurrentTest do
  use ExUnit.Case

  import Nx.Defn

  describe "dynamic_unroll" do
    test "computes carry and output identical to static_unroll" do
      input = Nx.random_uniform({1, 4, 2})
      carry = {Nx.random_uniform({1, 1, 8})}

      input_kernel =
        {Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8})}

      hidden_kernel =
        {Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8})}

      bias = {Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({})}

      cell_fn = &Axon.Recurrent.gru_cell/5

      {{s_carry}, s_output} =
          Axon.Recurrent.static_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, bias)

      {{d_carry}, d_output} =
          Axon.Recurrent.dynamic_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, bias)

      assert s_carry == d_carry
      assert s_output == d_output
    end

    defn grad_static_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {_, output} =
          Axon.Recurrent.static_unroll(cell_fn, input, carry, input_kernel, x, bias)

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {_, output} =
          Axon.Recurrent.dynamic_unroll(cell_fn, input, carry, input_kernel, x, bias)

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for hidden kernel w.r.t. output" do
      input = Nx.random_uniform({1, 4, 2})
      carry = {Nx.random_uniform({1, 1, 8})}

      input_kernel =
        {Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8})}

      hidden_kernel =
        {Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8})}

      bias = {Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({})}

      cell_fn = &Axon.Recurrent.gru_cell/5

      assert grad_static_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) ==
          grad_dynamic_hidden_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
    end

    defn grad_static_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {{carry}, _} =
          Axon.Recurrent.static_unroll(cell_fn, input, carry, input_kernel, x, bias)

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(hidden_kernel, fn x ->
        {{carry}, _} =
          Axon.Recurrent.dynamic_unroll(cell_fn, input, carry, input_kernel, x, bias)

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static_unroll for hidden kernel w.r.t carry" do
      input = Nx.random_uniform({1, 4, 2})
      carry = {Nx.random_uniform({1, 1, 8})}

      input_kernel =
        {Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8})}

      hidden_kernel =
        {Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8})}

      bias = {Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({})}

      cell_fn = &Axon.Recurrent.gru_cell/5

      assert grad_static_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) ==
        grad_dynamic_hidden_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
    end

    defn grad_static_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {_, output} =
          Axon.Recurrent.static_unroll(cell_fn, input, carry, x, hidden_kernel, bias)

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {_, output} =
          Axon.Recurrent.dynamic_unroll(cell_fn, input, carry, x, hidden_kernel, bias)

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for input kernel w.r.t. output" do
      input = Nx.random_uniform({1, 4, 2})
      carry = {Nx.random_uniform({1, 1, 8})}

      input_kernel =
        {Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8})}

      hidden_kernel =
        {Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8})}

      bias = {Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({})}

      cell_fn = &Axon.Recurrent.gru_cell/5

      assert grad_static_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) ==
          grad_dynamic_input_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
    end

    defn grad_static_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {{carry}, _} =
          Axon.Recurrent.static_unroll(cell_fn, input, carry, x, hidden_kernel, bias)

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(input_kernel, fn x ->
        {{carry}, _} =
          Axon.Recurrent.dynamic_unroll(cell_fn, input, carry, x, hidden_kernel, bias)

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static unroll for input kernel w.r.t. carry" do
      input = Nx.random_uniform({1, 4, 2})
      carry = {Nx.random_uniform({1, 1, 8})}

      input_kernel =
        {Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8})}

      hidden_kernel =
        {Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8})}

      bias = {Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({})}

      cell_fn = &Axon.Recurrent.gru_cell/5

      assert grad_static_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) ==
          grad_dynamic_input_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
    end

    defn grad_static_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {_, output} =
          Axon.Recurrent.static_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, x)

        Nx.mean(output)
      end)
    end

    defn grad_dynamic_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {_, output} =
          Axon.Recurrent.dynamic_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, x)

        Nx.mean(output)
      end)
    end

    test "computes gradient identical to static unroll for bias w.r.t. output" do
      input = Nx.random_uniform({1, 4, 2})
      carry = {Nx.random_uniform({1, 1, 8})}

      input_kernel =
        {Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8})}

      hidden_kernel =
        {Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8})}

      bias = {Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({})}

      cell_fn = &Axon.Recurrent.gru_cell/5

      assert grad_static_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn) ==
          grad_dynamic_bias_output(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
    end

    defn grad_static_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {{carry}, _} =
          Axon.Recurrent.static_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, x)

        Nx.mean(carry)
      end)
    end

    defn grad_dynamic_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) do
      grad(bias, fn x ->
        {{carry}, _} =
          Axon.Recurrent.dynamic_unroll(cell_fn, input, carry, input_kernel, hidden_kernel, x)

        Nx.mean(carry)
      end)
    end

    test "computes gradient identical to static unroll for bias w.r.t. carry" do
      input = Nx.random_uniform({1, 4, 2})
      carry = {Nx.random_uniform({1, 1, 8})}

      input_kernel =
        {Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8}), Nx.random_uniform({2, 8})}

      hidden_kernel =
        {Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8}), Nx.random_uniform({8, 8})}

      bias = {Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({}), Nx.random_uniform({})}

      cell_fn = &Axon.Recurrent.gru_cell/5

      assert grad_static_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn) ==
          grad_dynamic_bias_carry(input, carry, input_kernel, hidden_kernel, bias, cell_fn)
    end
  end
end