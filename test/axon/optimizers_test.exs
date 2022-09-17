defmodule OptimizersTest do
  use Axon.Case, async: true

  @learning_rate 1.0e-1
  @iterations 100

  describe "adabelief" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.adabelief(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.adabelief(@learning_rate, b1: 0.95, b2: 0.99)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.adabelief(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "adagrad" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.adagrad(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.adagrad(@learning_rate, eps: 1.0e-3)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.adagrad(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "adam" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.adam(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.adam(@learning_rate, b1: 0.95, b2: 0.99)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.adam(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "adamw" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.adamw(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.adamw(@learning_rate, decay: 0.9)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.adamw(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "lamb" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.lamb(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.lamb(@learning_rate, decay: 0.9, min_norm: 0.1)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.lamb(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "noisy_sgd" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.noisy_sgd(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.noisy_sgd(@learning_rate, eta: 0.2, gamma: 0.6)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.noisy_sgd(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "radam" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.radam(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.radam(@learning_rate, threshold: 2.0)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.radam(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "rmsprop" do
    test "correctly optimizes simple loss default case" do
      optimizer = Axon.Optimizers.rmsprop(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss centered case" do
      optimizer =
        Axon.Optimizers.rmsprop(@learning_rate, centered: true, initial_scale: 0.1, decay: 0.8)

      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss rms case" do
      optimizer = Axon.Optimizers.rmsprop(@learning_rate, initial_scale: 0.1, decay: 0.8)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with momentum" do
      optimizer =
        Axon.Optimizers.rmsprop(@learning_rate, initial_scale: 0.1, decay: 0.8, momentum: 0.9)

      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.rmsprop(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "sgd" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.sgd(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.sgd(@learning_rate, momentum: 0.9)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.sgd(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end

  describe "yogi" do
    test "correctly optimizes simple loss with default options" do
      optimizer = Axon.Optimizers.yogi(@learning_rate)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with custom options" do
      optimizer = Axon.Optimizers.yogi(@learning_rate, initial_accumulator_value: 0.1)
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor([1.0])}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end

    test "correctly optimizes simple loss with schedule" do
      optimizer = Axon.Optimizers.yogi(Axon.Schedules.constant(init_value: @learning_rate))
      loss_fn = fn %{"x0" => x} -> Nx.multiply(x, x) end
      num_steps = @iterations
      x0 = %{"x0" => Nx.tensor(1.0)}

      check_optimizer!(optimizer, loss_fn, x0, num_steps)
    end
  end
end
