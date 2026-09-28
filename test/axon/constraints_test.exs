defmodule Axon.ConstraintsTest do
  use Axon.Case, async: true

  doctest Axon.Constraints

  defmodule InDefn do
    import Nx.Defn

    defn apply_constraint(constraint, x), do: constraint.(x)
  end

  # Column norms are 5 and 1
  @x Nx.tensor([[3.0, 0.0], [4.0, 1.0]])

  describe "max_norm/1" do
    test "rescales vectors whose norm exceeds max" do
      expected = Nx.tensor([[0.6, 0.0], [0.8, 1.0]])
      assert_all_close(Axon.Constraints.max_norm(max: 1.0).(@x), expected, atol: 1.0e-5)
    end

    test "leaves vectors within max alone" do
      assert_all_close(Axon.Constraints.max_norm(max: 10.0).(@x), @x, atol: 1.0e-5)
    end

    test "computes norms over the given axes" do
      s = :math.sqrt(17.0)

      assert_all_close(
        Axon.Constraints.max_norm(max: 1.0, axes: [1]).(@x),
        Nx.tensor([[1.0, 0.0], [4.0 / s, 1.0 / s]]),
        atol: 1.0e-5
      )
    end
  end

  describe "unit_norm/1" do
    test "rescales every vector to unit norm" do
      assert_all_close(Axon.Constraints.unit_norm().(@x), Nx.tensor([[0.6, 0.0], [0.8, 1.0]]),
        atol: 1.0e-5
      )
    end
  end

  describe "non_neg/0" do
    test "clips negative entries to zero" do
      x = Nx.tensor([[-1.0, 2.0], [3.0, -4.0]])
      assert_equal(Axon.Constraints.non_neg().(x), Nx.tensor([[0.0, 2.0], [3.0, 0.0]]))
    end

    test "preserves the parameter type" do
      x = Nx.tensor([[-1.0, 2.0], [3.0, -4.0]], type: {:bf, 16})
      out = Axon.Constraints.non_neg().(x)
      assert Nx.type(out) == {:bf, 16}
      assert_equal(out, Nx.tensor([[0.0, 2.0], [3.0, 0.0]], type: {:bf, 16}))
    end
  end

  describe "min_max_norm/1" do
    test "rescales vectors outside of [min, max]" do
      assert_all_close(
        Axon.Constraints.min_max_norm(min: 2.0, max: 4.0).(@x),
        Nx.tensor([[2.4, 0.0], [3.2, 2.0]]),
        atol: 1.0e-5
      )
    end

    test "moves norms toward the feasible region according to rate" do
      assert_all_close(
        Axon.Constraints.min_max_norm(min: 2.0, max: 4.0, rate: 0.5).(@x),
        Nx.tensor([[2.7, 0.0], [3.6, 1.5]]),
        atol: 1.0e-5
      )
    end
  end

  describe "inside defn" do
    test "constraints are callable from defn" do
      x = Nx.tensor([[-1.0, 2.0], [3.0, -4.0]])

      assert_equal(
        InDefn.apply_constraint(Axon.Constraints.non_neg(), x),
        Nx.tensor([[0.0, 2.0], [3.0, 0.0]])
      )

      assert_all_close(
        InDefn.apply_constraint(Axon.Constraints.max_norm(max: 1.0), @x),
        Nx.tensor([[0.6, 0.0], [0.8, 1.0]]),
        atol: 1.0e-5
      )
    end
  end
end
