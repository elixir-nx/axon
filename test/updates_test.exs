defmodule Axon.UpdatesTest do
  use ExUnit.Case
  doctest Axon.Updates

  import Axon.Updates

  describe "identity" do
    test "composes with others" do
      {init_fn, apply_fn} = scale(0.5) |> identity()
      assert is_function(init_fn, 1)
      assert is_function(apply_fn, 3)
    end

    test "initializes to empty state" do
      {init_fn, _} = identity()
      assert init_fn.({}) == {}
    end

    test "returns input" do
      {_, update_fn} = identity()
      assert update_fn.({Nx.tensor(1.0)}, {}, {}) == {{Nx.tensor(1.0)}, {}}

      {_, update_fn} = scale(0.5) |> identity()
      assert update_fn.({Nx.tensor(1.0)}, {}, {}) == {{Nx.tensor(0.5)}, {}}
    end
  end

  describe "scale" do
    test "composes with others" do
      {init_fn, update_fn} = identity() |> scale(0.5)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "initializes to empty state" do
      {init_fn, _} = scale(0.5)
      assert init_fn.({}) == {}
    end

    test "scales input by rate" do
      {_, update_fn} = scale(0.5)
      assert update_fn.({Nx.tensor(1.0)}, {}, {}) == {{Nx.tensor(0.5)}, {}}
    end
  end

  describe "scale_by_adam" do
    test "composes with others" do
      {init_fn, update_fn} = identity() |> scale_by_adam(b1: 0.99)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "initializes to correct state" do
      {init_fn, _} = scale_by_adam(b1: 0.99)

      assert init_fn.({Nx.tensor([1.0])}) ==
               {{{Nx.tensor([0.0])}, {Nx.tensor([0.0])}, Nx.tensor(0)}}
    end
  end

  describe "clip" do
    test "composes with others" do
      {init_fn, update_fn} = identity() |> clip(delta: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "initializes to empty state" do
      {init_fn, _} = clip(delta: 1.0)
      assert init_fn.({}) == {}
    end

    test "clips input" do
      {_, update_fn} = clip(delta: 1.0)
      assert update_fn.({Nx.tensor([2.0, 1.0])}, {}, {}) == {{Nx.tensor([1.0, 1.0])}, {}}
    end
  end

  describe "clip_by_global_norm" do
    test "composes with others" do
      {init_fn, update_fn} = identity() |> clip_by_global_norm(max_norm: 0.5)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "initializes to empty state" do
      {init_fn, _} = clip_by_global_norm(max_norm: 0.5)
      assert init_fn.({}) == {}
    end

    test "clips input by global norm" do
      {_, update_fn} = clip_by_global_norm(max_norm: 1.0)
      {{res}, {}} = update_fn.({Nx.tensor([1.0, 1.0])}, {}, {})

      assert Nx.all_close?(res, Nx.tensor([0.7071067690849304, 0.7071067690849304])) ==
               Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "centralize" do
    test "composes with others" do
      {init_fn, update_fn} = identity() |> centralize()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "initializes to empty state" do
      {init_fn, _} = centralize()
      assert init_fn.({}) == {}
    end

    test "centralizes input" do
      {_, update_fn} = centralize()
      one_dim = Nx.tensor([1.0, 2.0])
      two_dim = Nx.tensor([[1.0, 2.0]])
      assert update_fn.({one_dim}, {}, {}) == {{Nx.tensor([1.0, 2.0])}, {}}
      assert update_fn.({two_dim}, {}, {}) == {{Nx.tensor([[-0.5, 0.5]])}, {}}
    end
  end

  describe "add_decayed_weights" do
    test "composes with others" do
      {init_fn, update_fn} = identity() |> add_decayed_weights(decay: 0.1)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
    end

    test "initializes to empty state" do
      {init_fn, _} = add_decayed_weights(decay: 0.1)
      assert init_fn.({}) == {}
    end

    test "adds decayed weights to input" do
      weights = Nx.tensor(1.0)
      updates = Nx.tensor(1.0)

      {_, update_fn} = add_decayed_weights(decay: 0.5)
      assert update_fn.({updates}, {}, {weights}) == {{Nx.tensor(1.5)}, {}}
    end
  end
end
