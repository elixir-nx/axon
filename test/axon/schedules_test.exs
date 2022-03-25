defmodule Axon.SchedulesTest do
  use ExUnit.Case
  doctest Axon.Schedules

  import Axon.Schedules

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
      assert Nx.all_close(fun.(0), 1.0e-2) == Nx.tensor(1, type: {:u, 8})

      fun = exponential_decay(init_value: 1.0e-3)
      assert Nx.all_close(fun.(0), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
    end

    test "can be called within JIT" do
      fun = exponential_decay()
      assert Nx.all_close(Nx.Defn.jit(fun, [0]), 1.0e-2) == Nx.tensor(1, type: {:u, 8})

      fun = exponential_decay(init_value: 1.0e-3)
      assert Nx.all_close(Nx.Defn.jit(fun, [0]), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
    end

    test "matches optax values at different counts" do
      fun1 = exponential_decay(init_value: 1.0e-2, decay_rate: 0.9, transition_steps: 15)

      assert Nx.all_close(fun1.(0), 1.0e-2) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(25), 0.008389527) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(50), 0.007038417) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(1000), 8.902254e-06) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(100_000), 0.0) == Nx.tensor(1, type: {:u, 8})

      fun2 = exponential_decay(init_value: 1.0e-3, decay_rate: 0.99, transition_steps: 100)

      assert Nx.all_close(fun2.(0), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(25), 0.0009974906) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(50), 0.0009949874) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(1000), 0.0009043822) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(100_000), 4.3171664e-08) == Nx.tensor(1, type: {:u, 8})

      fun3 =
        exponential_decay(
          init_value: 1.0e-1,
          decay_rate: 0.99,
          transition_begin: 100,
          transition_steps: 25
        )

      assert Nx.all_close(fun3.(0), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(25), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(50), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(1000), 0.069641344) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(100_000), 3.6162157e-19) == Nx.tensor(1, type: {:u, 8})
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
      assert Nx.all_close(fun.(0), 1.0e-2) == Nx.tensor(1, type: {:u, 8})

      fun = cosine_decay(init_value: 1.0e-3)
      assert Nx.all_close(fun.(0), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
    end

    test "can be called within JIT" do
      fun = cosine_decay()
      assert Nx.all_close(Nx.Defn.jit(fun, [0]), 1.0e-2) == Nx.tensor(1, type: {:u, 8})

      fun = cosine_decay(init_value: 1.0e-3)
      assert Nx.all_close(Nx.Defn.jit(fun, [0]), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
    end

    test "matches optax values at different counts" do
      fun1 = cosine_decay(init_value: 1.0e-3, decay_steps: 10, alpha: 0.0)

      assert Nx.all_close(fun1.(0), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(25), 0.0) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(50), 0.00) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(1000), 0.0) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(100_000), 0.0) == Nx.tensor(1, type: {:u, 8})

      fun2 = cosine_decay(init_value: 1.0e-2, decay_steps: 1000, alpha: 0.5)

      assert Nx.all_close(fun2.(0), 0.01) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(25), 0.009992293) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(50), 0.0099692205) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(1000), 0.005) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(100_000), 0.005) == Nx.tensor(1, type: {:u, 8})

      fun3 = cosine_decay(init_value: 1.0e-1, decay_steps: 1)

      assert Nx.all_close(fun3.(0), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(25), 0.0) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(50), 0.0) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(1000), 0.0) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(100_000), 0.0) == Nx.tensor(1, type: {:u, 8})
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
      assert Nx.all_close(fun.(0), 1.0e-2) == Nx.tensor(1, type: {:u, 8})

      fun = cosine_decay(init_value: 1.0e-3)
      assert Nx.all_close(fun.(0), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
    end

    test "can be called within JIT" do
      fun = constant()
      assert Nx.all_close(Nx.Defn.jit(fun, [0]), 1.0e-2) == Nx.tensor(1, type: {:u, 8})

      fun = constant(init_value: 1.0e-3)
      assert Nx.all_close(Nx.Defn.jit(fun, [0]), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
    end

    test "matches optax values at different counts" do
      fun1 = constant(init_value: 1.0e-3)

      assert Nx.all_close(fun1.(0), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(25), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(50), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(1000), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(100_000), 0.001) == Nx.tensor(1, type: {:u, 8})

      fun2 = constant(init_value: 1.0e-2)

      assert Nx.all_close(fun2.(0), 0.01) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(25), 0.01) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(50), 0.01) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(1000), 0.01) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(100_000), 0.01) == Nx.tensor(1, type: {:u, 8})

      fun3 = constant(init_value: 1.0e-1)

      assert Nx.all_close(fun3.(0), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(25), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(50), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(1000), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(100_000), 0.1) == Nx.tensor(1, type: {:u, 8})
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
      assert Nx.all_close(fun.(0), 1.0e-2) == Nx.tensor(1, type: {:u, 8})

      fun = polynomial_decay(init_value: 1.0e-3)
      assert Nx.all_close(fun.(0), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
    end

    test "can be called within JIT" do
      fun = polynomial_decay()
      assert Nx.all_close(Nx.Defn.jit(fun, [0]), 1.0e-2) == Nx.tensor(1, type: {:u, 8})

      fun = polynomial_decay(init_value: 1.0e-3)
      assert Nx.all_close(Nx.Defn.jit(fun, [0]), 1.0e-3) == Nx.tensor(1, type: {:u, 8})
    end

    test "matches optax values at different counts" do
      fun1 =
        polynomial_decay(init_value: 1.0e-2, end_value: 1.0e-3, power: 3, transition_steps: 1000)

      assert Nx.all_close(fun1.(0), 0.01) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(25), 0.009341734) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(50), 0.008716375) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(1000), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun1.(100_000), 0.001) == Nx.tensor(1, type: {:u, 8})

      fun2 =
        polynomial_decay(init_value: 1.0e-3, end_value: 1.0e-4, transition_begin: 100, power: 2)

      assert Nx.all_close(fun2.(0), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(25), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(50), 0.001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(1000), 0.0001) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun2.(100_000), 0.0001) == Nx.tensor(1, type: {:u, 8})

      fun3 =
        polynomial_decay(
          init_value: 1.0e-1,
          end_value: 1.0e-3,
          transition_steps: 10000,
          power: 1.5
        )

      assert Nx.all_close(fun3.(0), 0.1) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(25), 0.099628985) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(50), 0.09925843) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(1000), 0.08552768) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(fun3.(100_000), 0.001) == Nx.tensor(1, type: {:u, 8})
    end
  end
end
