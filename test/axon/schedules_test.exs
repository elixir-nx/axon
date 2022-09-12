defmodule Axon.SchedulesTest do
  use Axon.Case
  doctest Axon.Schedules

  import Axon.Schedules
  import Nx.Defn

  describe "exponential_decay" do
    test "returns arity-1 function with defaults" do
      fun = exponential_decay()
      assert is_function(fun, 1)
    end

    test "returns arity-1 function with options" do
      fun = exponential_decay(init_value: 1.0e-3)
      assert is_function(fun, 1)
    end

    test "can be called as anonymous function" do
      fun = exponential_decay()
      assert_all_close(fun.(0), 1.0e-2)

      fun = exponential_decay(init_value: 1.0e-3)
      assert_all_close(fun.(0), 1.0e-3)
    end

    test "can be called within JIT" do
      fun = exponential_decay()
      assert_all_close(apply(jit(fun), [0]), 1.0e-2)

      fun = exponential_decay(init_value: 1.0e-3)
      assert_all_close(apply(jit(fun), [0]), 1.0e-3)
    end

    test "matches optax values at different counts" do
      fun1 = exponential_decay(init_value: 1.0e-2, decay_rate: 0.9, transition_steps: 15)

      assert_all_close(fun1.(0), 1.0e-2)
      assert_all_close(fun1.(25), 0.008389527)
      assert_all_close(fun1.(50), 0.007038417)
      assert_all_close(fun1.(1000), 8.902254e-06)
      assert_all_close(fun1.(100_000), 0.0)

      fun2 = exponential_decay(init_value: 1.0e-3, decay_rate: 0.99, transition_steps: 100)

      assert_all_close(fun2.(0), 1.0e-3)
      assert_all_close(fun2.(25), 0.0009974906)
      assert_all_close(fun2.(50), 0.0009949874)
      assert_all_close(fun2.(1000), 0.0009043822)
      assert_all_close(fun2.(100_000), 4.3171664e-08)

      fun3 =
        exponential_decay(
          init_value: 1.0e-1,
          decay_rate: 0.99,
          transition_begin: 100,
          transition_steps: 25
        )

      assert_all_close(fun3.(0), 0.1)
      assert_all_close(fun3.(25), 0.1)
      assert_all_close(fun3.(50), 0.1)
      assert_all_close(fun3.(1000), 0.069641344)
      assert_all_close(fun3.(100_000), 3.6162157e-19)
    end
  end

  describe "cosine_decay" do
    test "returns arity-1 function with defaults" do
      fun = cosine_decay()
      assert is_function(fun, 1)
    end

    test "returns arity-1 function with options" do
      fun = cosine_decay(init_value: 1.0e-3)
      assert is_function(fun, 1)
    end

    test "can be called as anonymous function" do
      fun = cosine_decay()
      assert_all_close(fun.(0), 1.0e-2)

      fun = cosine_decay(init_value: 1.0e-3)
      assert_all_close(fun.(0), 1.0e-3)
    end

    test "can be called within JIT" do
      fun = cosine_decay()
      assert_all_close(apply(jit(fun), [0]), 1.0e-2)

      fun = cosine_decay(init_value: 1.0e-3)
      assert_all_close(apply(jit(fun), [0]), 1.0e-3)
    end

    test "matches optax values at different counts" do
      fun1 = cosine_decay(init_value: 1.0e-3, decay_steps: 10, alpha: 0.0)

      assert_all_close(fun1.(0), 0.001)
      assert_all_close(fun1.(25), 0.0)
      assert_all_close(fun1.(50), 0.00)
      assert_all_close(fun1.(1000), 0.0)
      assert_all_close(fun1.(100_000), 0.0)

      fun2 = cosine_decay(init_value: 1.0e-2, decay_steps: 1000, alpha: 0.5)

      assert_all_close(fun2.(0), 0.01)
      assert_all_close(fun2.(25), 0.009992293)
      assert_all_close(fun2.(50), 0.0099692205)
      assert_all_close(fun2.(1000), 0.005)
      assert_all_close(fun2.(100_000), 0.005)

      fun3 = cosine_decay(init_value: 1.0e-1, decay_steps: 1)

      assert_all_close(fun3.(0), 0.1)
      assert_all_close(fun3.(25), 0.0)
      assert_all_close(fun3.(50), 0.0)
      assert_all_close(fun3.(1000), 0.0)
      assert_all_close(fun3.(100_000), 0.0)
    end
  end

  describe "constant" do
    test "returns arity-1 function with defaults" do
      fun = constant()
      assert is_function(fun, 1)
    end

    test "returns arity-1 function with options" do
      fun = constant(init_value: 1.0e-3)
      assert is_function(fun, 1)
    end

    test "can be called as anonymous function" do
      fun = constant()
      assert_all_close(fun.(0), 1.0e-2)

      fun = cosine_decay(init_value: 1.0e-3)
      assert_all_close(fun.(0), 1.0e-3)
    end

    test "can be called within JIT" do
      fun = constant()
      assert_all_close(apply(jit(fun), [0]), 1.0e-2)

      fun = constant(init_value: 1.0e-3)
      assert_all_close(apply(jit(fun), [0]), 1.0e-3)
    end

    test "matches optax values at different counts" do
      fun1 = constant(init_value: 1.0e-3)

      assert_all_close(fun1.(0), 0.001)
      assert_all_close(fun1.(25), 0.001)
      assert_all_close(fun1.(50), 0.001)
      assert_all_close(fun1.(1000), 0.001)
      assert_all_close(fun1.(100_000), 0.001)

      fun2 = constant(init_value: 1.0e-2)

      assert_all_close(fun2.(0), 0.01)
      assert_all_close(fun2.(25), 0.01)
      assert_all_close(fun2.(50), 0.01)
      assert_all_close(fun2.(1000), 0.01)
      assert_all_close(fun2.(100_000), 0.01)

      fun3 = constant(init_value: 1.0e-1)

      assert_all_close(fun3.(0), 0.1)
      assert_all_close(fun3.(25), 0.1)
      assert_all_close(fun3.(50), 0.1)
      assert_all_close(fun3.(1000), 0.1)
      assert_all_close(fun3.(100_000), 0.1)
    end
  end

  describe "polynomial_decay" do
    test "returns arity-1 function with defaults" do
      fun = polynomial_decay()
      assert is_function(fun, 1)
    end

    test "returns arity-1 function with options" do
      fun = polynomial_decay(init_value: 1.0e-3)
      assert is_function(fun, 1)
    end

    test "can be called as anonymous function" do
      fun = polynomial_decay()
      assert_all_close(fun.(0), 1.0e-2)

      fun = polynomial_decay(init_value: 1.0e-3)
      assert_all_close(fun.(0), 1.0e-3)
    end

    test "can be called within JIT" do
      fun = polynomial_decay()
      assert_all_close(apply(jit(fun), [0]), 1.0e-2)

      fun = polynomial_decay(init_value: 1.0e-3)
      assert_all_close(apply(jit(fun), [0]), 1.0e-3)
    end

    test "matches optax values at different counts" do
      fun1 =
        polynomial_decay(init_value: 1.0e-2, end_value: 1.0e-3, power: 3, transition_steps: 1000)

      assert_all_close(fun1.(0), 0.01)
      assert_all_close(fun1.(25), 0.009341734)
      assert_all_close(fun1.(50), 0.008716375)
      assert_all_close(fun1.(1000), 0.001)
      assert_all_close(fun1.(100_000), 0.001)

      fun2 =
        polynomial_decay(init_value: 1.0e-3, end_value: 1.0e-4, transition_begin: 100, power: 2)

      assert_all_close(fun2.(0), 0.001)
      assert_all_close(fun2.(25), 0.001)
      assert_all_close(fun2.(50), 0.001)
      assert_all_close(fun2.(1000), 0.0001)
      assert_all_close(fun2.(100_000), 0.0001)

      fun3 =
        polynomial_decay(
          init_value: 1.0e-1,
          end_value: 1.0e-3,
          transition_steps: 10000,
          power: 1.5
        )

      assert_all_close(fun3.(0), 0.1)
      assert_all_close(fun3.(25), 0.099628985)
      assert_all_close(fun3.(50), 0.09925843)
      assert_all_close(fun3.(1000), 0.08552768)
      assert_all_close(fun3.(100_000), 0.001)
    end
  end
end
