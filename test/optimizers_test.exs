defmodule OptimizersTest do
  use ExUnit.Case, async: true

  import AxonTestUtil

  describe "adabelief" do
    test "correctly accepts options" do
      {init_fn, update_fn} = Axon.Optimizers.adabelief(1.0e-2, b1: 0.95,
                                                               b2: 0.90,
                                                               eps: 1.0e-5,
                                                               eps_root: 1.0e-5)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "correctly optimizes simple loss" do
      optimizer = Axon.Optimizers.adabelief(1.0e-2)
      loss_fn = fn x -> x * x end
      num_steps = 100
      x0 = %{"x0" => 1.0}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "adagrad" do
    test "correctly accepts options" do
    end

    test "correctly optimizers simple loss" do
      optimizer = Axon.Optimizers.adagrad(1.0e-2)
      loss_fn = fn x -> x * x end
      num_steps = 100
      x0 = %{"x0" => 1.0}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    describe "adam" do
      test "correctly accepts options" do
      end

      test "correctly optimizers simple loss" do
        optimizer = Axon.Optimizers.adam(1.0e-2)
        loss_fn = fn x -> x * x end
        num_steps = 100
        x0 = %{"x0" => 1.0}

        check_optimizer!(optimizer, loss_fn, x0, num_steps)
      end
    end
  end
end