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
      {init_fn, apply_fn} = identity() |> scale(0.5)
      assert is_function(init_fn, 1)
      assert is_function(apply_fn, 3)
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
end
