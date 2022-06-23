defmodule Axon.InitializersTest do
  use ExUnit.Case, async: true

  import AxonTestUtil

  doctest Axon.Initializers

  describe "orthogonal/1" do
    test "property" do
      init_fn = Axon.Initializers.orthogonal()
      t1 = init_fn.({3, 3}, {:f, 32})
      identity_left_t1 = t1 |> Nx.transpose() |> Nx.dot(t1)

      assert_all_close(identity_left_t1, Nx.eye(identity_left_t1), atol: 1.0e-3, rtol: 1.0e-3)

      identity_right_t1 = t1 |> Nx.dot(t1 |> Nx.transpose())

      assert_all_close(identity_right_t1, Nx.eye(identity_right_t1),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )

      init_fn = Axon.Initializers.orthogonal()
      t2 = init_fn.({1, 2, 3, 4, 5}, {:f, 32}) |> Nx.reshape({24, 5})

      identity_left_t2 = t2 |> Nx.transpose() |> Nx.dot(t2)

      assert Nx.all_close(identity_left_t2, Nx.eye(identity_left_t2), atol: 1.0e-3, rtol: 1.0e-3) ==
               Nx.tensor(1, type: {:u, 8})

      # Since the matrix is "tall", it's transpose will only be it's left inverse
      identity_right_t2 = t2 |> Nx.dot(Nx.transpose(t2))

      assert Nx.all_close(identity_right_t2, Nx.eye(identity_right_t2),
               atol: 1.0e-3,
               rtol: 1.0e-3
             ) ==
               Nx.tensor(0, type: {:u, 8})
    end

    test "raises on input rank less than 2" do
      assert_raise ArgumentError,
                   ~r/Axon.Initializers.orthogonal: expected input_shape shape to have at least rank 2/,
                   fn ->
                     init_fn = Axon.Initializers.orthogonal()
                     init_fn.({1}, {:f, 32})
                   end
    end
  end
end
