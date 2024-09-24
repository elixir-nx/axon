defmodule Axon.LossScaleTest do
  use ExUnit.Case
  import AxonTestUtil

  import Axon.LossScale

  describe "identity/1" do
    test "creates a loss scale tuple" do
      assert {init_fn, scale_fn, adjust_fn} = identity()
      assert is_function(init_fn, 0)
      assert is_function(scale_fn, 2)
      assert is_function(adjust_fn, 2)
    end

    test "accepts options" do
      assert {init_fn, scale_fn, adjust_fn} = identity([])
      assert is_function(init_fn, 0)
      assert is_function(scale_fn, 2)
      assert is_function(adjust_fn, 2)
    end

    test "initializes to empty state" do
      assert {init_fn, _, _} = identity()
      assert init_fn.() == %{}
    end

    test "scale function returns identity operation on x" do
      assert {init_fn, scale_fn, _} = identity()
      state = init_fn.()
      x = Nx.tensor([1.0, 2.0, 3.0])

      new_x = scale_fn.(x, state)
      assert new_x == x
    end

    test "adjust function returns identity operation on x and state" do
      assert {init_fn, _, adjust_fn} = identity()
      state = init_fn.()
      x = Nx.tensor([1.0, 2.0, 3.0])

      assert {new_x, new_state} = adjust_fn.(x, state)
      assert new_x == x
      assert new_state == state
    end
  end

  describe "static/1" do
    test "creates a loss scale tuple" do
      assert {init_fn, scale_fn, adjust_fn} = static()
      assert is_function(init_fn, 0)
      assert is_function(scale_fn, 2)
      assert is_function(adjust_fn, 2)
    end

    test "accepts options" do
      assert {init_fn, scale_fn, adjust_fn} = static([])
      assert is_function(init_fn, 0)
      assert is_function(scale_fn, 2)
      assert is_function(adjust_fn, 2)
    end

    test "initializes state with default loss scale" do
      assert {init_fn, _, _} = static()
      assert %{loss_scale: loss_scale} = init_fn.()
      assert_equal(loss_scale, Nx.pow(2, 15))
    end

    test "initializes state with specified loss scale" do
      init_scale = Nx.pow(3, 15)
      assert {init_fn, _, _} = static(init_scale: init_scale)
      assert %{loss_scale: loss_scale} = init_fn.()
      assert_equal(loss_scale, init_scale)
    end

    test "scale function returns a tree scaled by static scale" do
      assert {init_fn, scale_fn, _} = static()
      state = init_fn.()
      a = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.tensor([4.0, 5.0, 6.0])
      x = %{a: a, b: %{c: c}}

      assert %{a: scaled_a, b: %{c: scaled_c}} = scale_fn.(x, state)
      assert_equal(scaled_a, Nx.multiply(a, Nx.pow(2, 15)))
      assert_equal(scaled_c, Nx.multiply(c, Nx.pow(2, 15)))
    end

    test "scale function returns a tree scaled by static scale with custom scale" do
      init_scale = Nx.pow(3, 15)
      assert {init_fn, scale_fn, _} = static(init_scale: init_scale)
      state = init_fn.()
      a = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.tensor([4.0, 5.0, 6.0])
      x = %{a: a, b: %{c: c}}

      assert %{a: scaled_a, b: %{c: scaled_c}} = scale_fn.(x, state)
      assert_equal(scaled_a, Nx.multiply(a, init_scale))
      assert_equal(scaled_c, Nx.multiply(c, init_scale))
    end

    test "adjust function returns unscaled tree with static state" do
      assert {init_fn, scale_fn, adjust_fn} = static()
      state = init_fn.()
      a = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.tensor([4.0, 5.0, 6.0])
      x = %{a: a, b: %{c: c}}

      scaled_x = scale_fn.(x, state)
      assert {unscaled_x, new_state} = adjust_fn.(scaled_x, state)
      assert %{a: unscaled_a, b: %{c: unscaled_c}} = unscaled_x
      assert %{loss_scale: new_loss_scale} = new_state

      assert_all_close(unscaled_a, a)
      assert_all_close(unscaled_c, c)
      assert_equal(new_loss_scale, Nx.pow(2, 15))
    end

    test "adjust function returns unscaled tree with static state and custom scale" do
      init_scale = Nx.pow(3, 15)

      assert {init_fn, scale_fn, adjust_fn} = static(init_scale: init_scale)
      state = init_fn.()
      a = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.tensor([4.0, 5.0, 6.0])
      x = %{a: a, b: %{c: c}}

      scaled_x = scale_fn.(x, state)
      assert {unscaled_x, new_state} = adjust_fn.(scaled_x, state)
      assert %{a: unscaled_a, b: %{c: unscaled_c}} = unscaled_x
      assert %{loss_scale: new_loss_scale} = new_state

      assert_all_close(unscaled_a, a)
      assert_all_close(unscaled_c, c)
      assert_equal(new_loss_scale, init_scale)
    end
  end

  describe "dynamic/1" do
    test "creates a loss scale tuple" do
      assert {init_fn, scale_fn, adjust_fn} = dynamic()
      assert is_function(init_fn, 0)
      assert is_function(scale_fn, 2)
      assert is_function(adjust_fn, 2)
    end

    test "accepts options" do
      assert {init_fn, scale_fn, adjust_fn} = dynamic([])
      assert is_function(init_fn, 0)
      assert is_function(scale_fn, 2)
      assert is_function(adjust_fn, 2)
    end

    test "initializes state with default loss scale" do
      assert {init_fn, _, _} = dynamic()
      assert %{loss_scale: loss_scale, counter: counter} = init_fn.()
      assert_equal(loss_scale, Nx.pow(2, 15))
      assert_equal(counter, Nx.tensor(0))
    end

    test "initializes state with specified loss scale" do
      init_scale = Nx.pow(3, 15)
      assert {init_fn, _, _} = dynamic(init_scale: init_scale)
      assert %{loss_scale: loss_scale, counter: counter} = init_fn.()
      assert_equal(counter, Nx.tensor(0))
      assert_equal(loss_scale, init_scale)
    end

    test "scale function returns a tree scaled by scale" do
      assert {init_fn, scale_fn, _} = dynamic()
      state = init_fn.()
      a = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.tensor([4.0, 5.0, 6.0])
      x = %{a: a, b: %{c: c}}

      assert %{a: scaled_a, b: %{c: scaled_c}} = scale_fn.(x, state)
      assert_equal(scaled_a, Nx.multiply(a, Nx.pow(2, 15)))
      assert_equal(scaled_c, Nx.multiply(c, Nx.pow(2, 15)))
    end

    test "scale function returns a tree scaled by scale with custom scale" do
      init_scale = Nx.pow(3, 15)
      assert {init_fn, scale_fn, _} = dynamic(init_scale: init_scale)
      state = init_fn.()
      a = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.tensor([4.0, 5.0, 6.0])
      x = %{a: a, b: %{c: c}}

      assert %{a: scaled_a, b: %{c: scaled_c}} = scale_fn.(x, state)
      assert_equal(scaled_a, Nx.multiply(a, init_scale))
      assert_equal(scaled_c, Nx.multiply(c, init_scale))
    end

    test "adjust function unscales correctly" do
      init_scale = Nx.tensor(10)
      assert {init_fn, scale_fn, adjust_fn} = dynamic(init_scale: init_scale)
      state = init_fn.()

      a = Nx.tensor([1.0, 2.0, 3.0])
      c = Nx.tensor([4.0, 5.0, 6.0])
      x = %{a: a, b: %{c: c}}

      scaled_x = scale_fn.(x, state)
      assert {unscaled_x, _new_state} = adjust_fn.(scaled_x, state)
      assert %{a: unscaled_a, b: %{c: unscaled_c}} = unscaled_x

      assert_all_close(unscaled_a, a)
      assert_all_close(unscaled_c, c)
    end

    test "adjust function increases loss scale according to period and factor when grads are finite" do
      init_scale = Nx.tensor(10)
      period = 5
      assert {init_fn, _, adjust_fn} = dynamic(init_scale: init_scale, period: period)
      state = init_fn.()

      finite = Nx.tensor([1.0, 1.0, 1.0])

      final_state =
        for i <- 1..(period - 1), reduce: state do
          new_state ->
            {_, %{loss_scale: loss_scale, counter: counter} = new_state} =
              adjust_fn.(finite, new_state)

            assert_equal(loss_scale, init_scale)
            assert_equal(counter, Nx.tensor(i))
            new_state
        end

      assert {_, %{loss_scale: final_scale, counter: final_counter}} =
               adjust_fn.(finite, final_state)

      assert_equal(final_scale, Nx.tensor(20.0))
      assert_equal(final_counter, Nx.tensor(0))
    end

    test "adjust function reduces loss scale on non finite" do
      init_scale = Nx.tensor(10)
      period = 5
      factor = 2

      assert {init_fn, _, adjust_fn} =
               dynamic(init_scale: init_scale, period: period, factor: factor)

      state = init_fn.()

      non_finite = Nx.tensor([:infinity, :infinity, :infinity])

      for i <- 0..99, reduce: state do
        new_state ->
          {_, %{loss_scale: loss_scale, counter: counter} = new_state} =
            adjust_fn.(non_finite, new_state)

          # We want to check if init_scale / factor ** (i + 1) is greater than 1.
          # If we rely on `i` directly, we run into integer overflow issues.
          # Instead, we accumulate the divisor on the reduce.

          scale_divisor = 2 ** (i + 1)

          expected_new_scale =
            if scale_divisor >= 2 ** 32 do
              Nx.tensor(1)
            else
              Nx.max(1, Nx.divide(init_scale, scale_divisor))
            end

          assert_equal(counter, Nx.tensor(0))

          assert_all_close(loss_scale, expected_new_scale)

          new_state
      end
    end

    test "adjust function reduces loss scale to min loss scale" do
      init_scale = Nx.tensor(20)
      period = 5
      factor = 2
      min_loss_scale = 2

      assert {init_fn, _, adjust_fn} =
               dynamic(
                 init_scale: init_scale,
                 period: period,
                 factor: factor,
                 min_loss_scale: min_loss_scale
               )

      state = init_fn.()

      non_finite = Nx.tensor([:infinity, :infinity, :infinity])

      for i <- 0..99, reduce: state do
        new_state ->
          {_, %{loss_scale: loss_scale, counter: counter} = new_state} =
            adjust_fn.(non_finite, new_state)

          scale_divisor = 2 ** (i + 1)

          expected_new_scale =
            if scale_divisor >= 2 ** 32 do
              Nx.tensor(min_loss_scale)
            else
              Nx.max(min_loss_scale, Nx.divide(init_scale, scale_divisor))
            end

          assert_equal(counter, Nx.tensor(0))
          assert_all_close(loss_scale, expected_new_scale)

          new_state
      end
    end
  end
end
