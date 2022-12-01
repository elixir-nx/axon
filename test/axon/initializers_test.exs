defmodule Axon.InitializersTest do
  use Axon.Case, async: true

  doctest Axon.Initializers

  describe "lecun_uniform/1" do
    test "matches jax with defaults" do
      init_fn = Axon.Initializers.lecun_uniform()

      actual = init_fn.({6, 4}, :f32, Nx.Random.key(0))

      expected =
        Nx.tensor([
          [0.20278636, -0.44988236, -0.67542195, 0.09019998],
          [0.07444432, -0.5715227, -0.22269602, -0.65556777],
          [-0.5834403, 0.41140956, -0.20922656, 0.04757705],
          [-0.6660935, -0.11757841, 0.11348342, 0.58670807],
          [-0.31940702, -0.4950906, 0.6199206, 0.02958001],
          [0.01707217, 0.57443, 0.3266003, 0.64393777]
        ])

      assert_all_close(expected, actual)
    end
  end

  describe "lecun_normal/1" do
    test "matches jax with defaults" do
      init_fn = Axon.Initializers.lecun_normal()

      actual = init_fn.({6, 4}, :f32, Nx.Random.key(0))

      expected =
        Nx.tensor([
          [0.16248588, -0.39667854, -0.7911283, 0.07110167],
          [0.05860865, -0.5588784, -0.17921554, -0.73135567],
          [-0.5787071, 0.35475218, -0.16787331, 0.03739766],
          [-0.7614683, -0.09293934, 0.08966116, 0.58433205],
          [-0.26443285, -0.4505209, 0.6471739, 0.02323576],
          [0.01340684, 0.56362104, 0.27110383, 0.7013128]
        ])

      assert_all_close(expected, actual)
    end
  end

  describe "glorot_uniform/1" do
    test "matches jax with defaults" do
      init_fn = Axon.Initializers.glorot_uniform()

      actual = init_fn.({6, 4}, :f32, Nx.Random.key(0))

      expected =
        Nx.tensor([
          [0.22214133, -0.49282146, -0.7398877, 0.09880914],
          [0.08154967, -0.6260718, -0.24395128, -0.7181385],
          [-0.6391269, 0.4506766, -0.22919622, 0.05211805],
          [-0.7296689, -0.1288007, 0.12431487, 0.6427065],
          [-0.34989288, -0.54234457, 0.67908907, 0.03240328],
          [0.01870163, 0.6292566, 0.35777274, 0.7053985]
        ])

      assert_all_close(expected, actual)
    end
  end

  describe "glorot_normal/1" do
    test "matches jax with defaults" do
      init_fn = Axon.Initializers.glorot_normal()

      actual = init_fn.({6, 4}, :f32, Nx.Random.key(0))

      expected =
        Nx.tensor([
          [0.17799434, -0.43453953, -0.8666375, 0.07788797],
          [0.06420256, -0.6122206, -0.19632077, -0.8011599],
          [-0.63394177, 0.3886115, -0.18389598, 0.04096708],
          [-0.8341466, -0.10180994, 0.09821887, 0.64010364],
          [-0.28967166, -0.49352086, 0.7089434, 0.0254535],
          [0.01468645, 0.61741585, 0.29697934, 0.7682496]
        ])

      assert_all_close(expected, actual)
    end
  end

  describe "he_uniform/1" do
    test "matches jax with defaults" do
      init_fn = Axon.Initializers.he_uniform()

      actual = init_fn.({6, 4}, :f32, Nx.Random.key(0))

      expected =
        Nx.tensor([
          [0.28678322, -0.63622975, -0.9551909, 0.12756205],
          [0.10528016, -0.8082552, -0.31493974, -0.9271128],
          [-0.82510924, 0.58182096, -0.29589105, 0.06728411],
          [-0.9419985, -0.16628098, 0.1604898, 0.8297305],
          [-0.45170975, -0.70016384, 0.87670016, 0.04183245],
          [0.0241437, 0.8123667, 0.4618826, 0.9106655]
        ])

      assert_all_close(expected, actual)
    end
  end

  describe "he_normal/1" do
    test "matches jax with defaults" do
      init_fn = Axon.Initializers.he_normal()

      actual = init_fn.({6, 4}, :f32, Nx.Random.key(0))

      expected =
        Nx.tensor([
          [0.22978972, -0.5609881, -1.1188242, 0.10055294],
          [0.08288515, -0.7903734, -0.25344902, -1.034293],
          [-0.81841534, 0.5016953, -0.2374087, 0.05288828],
          [-1.0768787, -0.13143606, 0.12680002, 0.8263703],
          [-0.37396452, -0.6371327, 0.915242, 0.03286032],
          [0.01896013, 0.79708046, 0.38339868, 0.991806]
        ])

      assert_all_close(expected, actual)
    end
  end

  describe "orthogonal/1" do
    test "property" do
      init_fn = Axon.Initializers.orthogonal()
      t1 = init_fn.({3, 3}, {:f, 32}, Nx.Random.key(1))
      identity_left_t1 = t1 |> Nx.transpose() |> Nx.dot(t1)

      assert_all_close(identity_left_t1, Nx.eye(Nx.shape(identity_left_t1)),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )

      identity_right_t1 = t1 |> Nx.dot(t1 |> Nx.transpose())

      assert_all_close(identity_right_t1, Nx.eye(Nx.shape(identity_right_t1)),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )

      init_fn = Axon.Initializers.orthogonal()
      t2 = init_fn.({1, 2, 3, 4, 5}, {:f, 32}, Nx.Random.key(1))
      t2 = Nx.reshape(t2, {24, 5})

      identity_left_t2 = t2 |> Nx.transpose() |> Nx.dot(t2)

      assert_all_close(identity_left_t2, Nx.eye(Nx.shape(identity_left_t2)),
        atol: 1.0e-3,
        rtol: 1.0e-3
      )

      # Since the matrix is "tall", it's transpose will only be it's left inverse
      identity_right_t2 = t2 |> Nx.dot(Nx.transpose(t2))

      assert_equal(
        Nx.all_close(identity_right_t2, Nx.eye(Nx.shape(identity_right_t2)),
          atol: 1.0e-3,
          rtol: 1.0e-3
        ),
        Nx.tensor(0, type: {:u, 8})
      )
    end

    test "raises on input rank less than 2" do
      assert_raise ArgumentError,
                   ~r/Axon.Initializers.orthogonal: expected input_shape shape to have at least rank 2/,
                   fn ->
                     init_fn = Axon.Initializers.orthogonal()
                     init_fn.({1}, {:f, 32}, Nx.Random.key(1))
                   end
    end
  end
end
