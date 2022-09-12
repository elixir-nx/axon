defmodule Axon.UpdatesTest do
  use Axon.Case
  doctest Axon.Updates

  import Axon.Updates

  describe "add_decayed_weights" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = add_decayed_weights()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "constructs a stateless transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = add_decayed_weights(decay: 0.95)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}

      assert {init_fn, update_fn} =
               add_decayed_weights(decay: 0.95) |> add_decayed_weights(decay: 0.95)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> add_decayed_weights(decay: 0.95)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = add_decayed_weights(decay: 0.95)
      params = %{a: Nx.tensor([0.18884168, 0.92323774, 0.4513516])}
      updates = %{a: Nx.tensor([0.62370003, 0.86674502, 0.11204521])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.80309962, 1.74382088, 0.54082923])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert new_state == {}
      assert_all_close(actual_a, expected_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = add_decayed_weights(decay: 0.95)

      params = %{
        a: %{
          b: Nx.tensor([0.26106195, 0.52850289, 0.19788291]),
          c: %{d: %{}, e: Nx.tensor([[0.7100145, 0.41356265, 0.35657979]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.83834362, 0.75873946, 0.54735649]),
          c: %{d: %{}, e: Nx.tensor([[0.7384456, 0.76676084, 0.72992148]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([1.08635247, 1.26081721, 0.73534525])
      expected_e = Nx.tensor([[1.41295937, 1.15964536, 1.06867228]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = add_decayed_weights(decay: 0.95)

      params = {
        {
          Nx.tensor([0.26106195, 0.52850289, 0.19788291]),
          {{}, Nx.tensor([[0.7100145, 0.41356265, 0.35657979]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.83834362, 0.75873946, 0.54735649]),
          {{}, Nx.tensor([[0.7384456, 0.76676084, 0.72992148]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([1.08635247, 1.26081721, 0.73534525])
      expected_e = Nx.tensor([[1.41295937, 1.15964536, 1.06867228]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end
  end

  describe "add_noise" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = add_noise()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {add_noise_state} = init_fn.(params)
      assert %{count: count} = add_noise_state
      assert_equal(count, Nx.tensor(0))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = add_noise(gamma: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {add_noise_state} = init_fn.(params)
      assert %{count: count} = add_noise_state
      assert_equal(count, Nx.tensor(0))
    end
  end

  describe "clip" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = clip()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "constructs a stateless transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = clip(delta: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = clip(delta: 2.0) |> clip(delta: 2.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> clip(delta: 2.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = clip(delta: 2.0)
      params = %{a: Nx.tensor([0.74794595, 0.99105549, 0.5621627])}
      updates = %{a: Nx.tensor([0.84208747, 0.69837738, 0.61840895])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.84208745, 0.6983774, 0.618409])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert new_state == {}
      assert_all_close(actual_a, expected_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = clip(delta: 1.0)

      params = %{
        a: %{
          b: Nx.tensor([0.62866726, 0.04867021, 0.66160428]),
          c: %{d: %{}, e: Nx.tensor([0.70566323, 0.52083707, 0.14541595])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.19084232, 0.09963277, 0.28141486]),
          c: %{d: %{}, e: Nx.tensor([0.91124607, 0.2248316, 0.79530217])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.19084232, 0.09963277, 0.28141487])
      expected_e = Nx.tensor([0.91124606, 0.2248316, 0.79530215])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = clip(delta: 1.0)

      params = {
        {
          Nx.tensor([0.62866726, 0.04867021, 0.66160428]),
          {{}, Nx.tensor([0.70566323, 0.52083707, 0.14541595])}
        }
      }

      updates = {
        {
          Nx.tensor([0.19084232, 0.09963277, 0.28141486]),
          {{}, Nx.tensor([0.91124607, 0.2248316, 0.79530217])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.19084232, 0.09963277, 0.28141487])
      expected_e = Nx.tensor([0.91124606, 0.2248316, 0.79530215])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end
  end

  describe "clip_by_global_norm" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = clip_by_global_norm()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "constructs a stateless transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = clip_by_global_norm(max_norm: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}

      assert {init_fn, update_fn} =
               clip_by_global_norm(max_norm: 1.0) |> clip_by_global_norm(max_norm: 1.0)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> clip_by_global_norm(max_norm: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = clip_by_global_norm(max_norm: 1.0)
      params = %{a: Nx.tensor([0.72673265, 0.35788219, 0.75329067])}
      updates = %{a: Nx.tensor([0.68235248, 0.56976359, 0.79599518])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.571844, 0.47748914, 0.667082])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert new_state == {}
      assert_all_close(actual_a, expected_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = clip_by_global_norm(max_norm: 1.0)

      params = %{
        a: %{
          b: Nx.tensor([0.85107357, 0.67088125, 0.59811338]),
          c: %{d: %{}, e: Nx.tensor([0.45385324, 0.05131562, 0.91526984])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.59629243, 0.86219328, 0.30155944]),
          c: %{d: %{}, e: Nx.tensor([0.83792943, 0.22030587, 0.72606433])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.3795878, 0.54885495, 0.1919667])
      expected_e = Nx.tensor([0.53340906, 0.14024231, 0.462198])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = clip_by_global_norm(max_norm: 1.0)

      params = {
        {
          Nx.tensor([0.85107357, 0.67088125, 0.59811338]),
          {{}, Nx.tensor([0.45385324, 0.05131562, 0.91526984])}
        }
      }

      updates = {
        {
          Nx.tensor([0.59629243, 0.86219328, 0.30155944]),
          {{}, Nx.tensor([0.83792943, 0.22030587, 0.72606433])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.3795878, 0.54885495, 0.1919667])
      expected_e = Nx.tensor([0.53340906, 0.14024231, 0.462198])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end
  end

  describe "centralize" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = centralize()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = centralize() |> centralize()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> centralize()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = centralize()
      params = %{a: Nx.tensor([0.14574998, 0.53619206, 0.68726124])}
      updates = %{a: Nx.tensor([0.05166196, 0.3979764, 0.84524461])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.05166196, 0.3979764, 0.84524461])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert new_state == {}
      assert_all_close(actual_a, expected_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = centralize()

      params = %{
        a: %{
          b: Nx.tensor([0.21855268, 0.21286796, 0.83114509]),
          c: %{d: %{}, e: Nx.tensor([[0.26958357, 0.59519575, 0.87732692]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.41087112, 0.97778015, 0.51054674]),
          c: %{d: %{}, e: Nx.tensor([[0.20577277, 0.95319838, 0.14168365]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.41087112, 0.97778015, 0.51054674])
      expected_e = Nx.tensor([[-0.22777883, 0.51964678, -0.29186795]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = centralize()

      params = {
        {
          Nx.tensor([0.21855268, 0.21286796, 0.83114509]),
          {{}, Nx.tensor([[0.26958357, 0.59519575, 0.87732692]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.41087112, 0.97778015, 0.51054674]),
          {{}, Nx.tensor([[0.20577277, 0.95319838, 0.14168365]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.41087112, 0.97778015, 0.51054674])
      expected_e = Nx.tensor([[-0.22777883, 0.51964678, -0.29186795]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end
  end

  describe "identity" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = identity()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = identity() |> identity()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> identity()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = identity()
      params = %{a: Nx.tensor([0.18884168, 0.92323774, 0.4513516])}
      updates = %{a: Nx.tensor([0.62370003, 0.86674502, 0.11204521])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.62370003, 0.86674502, 0.11204521])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert new_state == {}
      assert_all_close(actual_a, expected_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = identity()

      params = %{
        a: %{
          b: Nx.tensor([0.26106195, 0.52850289, 0.19788291]),
          c: %{d: %{}, e: Nx.tensor([[0.7100145, 0.41356265, 0.35657979]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.83834362, 0.75873946, 0.54735649]),
          c: %{d: %{}, e: Nx.tensor([[0.7384456, 0.76676084, 0.72992148]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.83834362, 0.75873946, 0.54735649])
      expected_e = Nx.tensor([[0.7384456, 0.76676084, 0.72992148]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = identity()

      params = {
        {
          Nx.tensor([0.26106195, 0.52850289, 0.19788291]),
          {{}, Nx.tensor([[0.7100145, 0.41356265, 0.35657979]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.83834362, 0.75873946, 0.54735649]),
          {{}, Nx.tensor([[0.7384456, 0.76676084, 0.72992148]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.83834362, 0.75873946, 0.54735649])
      expected_e = Nx.tensor([[0.7384456, 0.76676084, 0.72992148]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end
  end

  describe "scale" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale(1.0e-2) |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale(1.0e-2)
      params = %{a: Nx.tensor([0.29887561, 0.70429164, 0.43314898])}
      updates = %{a: Nx.tensor([0.2584395, 0.35890494, 0.84845509])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.00258439, 0.00358905, 0.00848455])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert new_state == {}
      assert_all_close(actual_a, expected_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale(1.0e-2)

      params = %{
        a: %{
          b: Nx.tensor([0.58813851, 0.27981229, 0.17335737]),
          c: %{d: %{}, e: Nx.tensor([0.21444265, 0.63923396, 0.12755156])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.48363215, 0.7147937, 0.32252682]),
          c: %{d: %{}, e: Nx.tensor([0.09518468, 0.38613084, 0.20729078])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.00483632, 0.00714794, 0.00322527])
      expected_e = Nx.tensor([0.00095185, 0.00386131, 0.00207291])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale(1.0e-2)

      params = {
        {
          Nx.tensor([0.58813851, 0.27981229, 0.17335737]),
          {{}, Nx.tensor([0.21444265, 0.63923396, 0.12755156])}
        }
      }

      updates = {
        {
          Nx.tensor([0.48363215, 0.7147937, 0.32252682]),
          {{}, Nx.tensor([0.09518468, 0.38613084, 0.20729078])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.00483632, 0.00714794, 0.00322527])
      expected_e = Nx.tensor([0.00095185, 0.00386131, 0.00207291])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end
  end

  describe "scale_by_state" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_state(1.0e-3)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {state} = init_fn.(params)
      assert %{scale: scale} = state
      assert_equal(scale, Nx.tensor(1.0e-3))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_state(1.0e-3) |> scale_by_state(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {state_1, state_2} = init_fn.(params)
      assert %{scale: scale_1} = state_1
      assert_equal(scale_1, Nx.tensor(1.0e-2))
      assert %{scale: scale_2} = state_2
      assert_equal(scale_2, Nx.tensor(1.0e-3))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_state(1.0e-3) |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {state} = init_fn.(params)
      assert %{scale: scale} = state
      assert_equal(scale, Nx.tensor(1.0e-3))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_state(1.0e-2)
      params = %{a: Nx.tensor([0.29887561, 0.70429164, 0.43314898])}
      updates = %{a: Nx.tensor([0.2584395, 0.35890494, 0.84845509])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.00258439, 0.00358905, 0.00848455])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert {%{scale: scale}} = new_state
      assert_all_close(actual_a, expected_a)
      assert_all_close(scale, Nx.tensor(1.0e-2))
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_state(1.0e-2)

      params = %{
        a: %{
          b: Nx.tensor([0.58813851, 0.27981229, 0.17335737]),
          c: %{d: %{}, e: Nx.tensor([0.21444265, 0.63923396, 0.12755156])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.48363215, 0.7147937, 0.32252682]),
          c: %{d: %{}, e: Nx.tensor([0.09518468, 0.38613084, 0.20729078])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.00483632, 0.00714794, 0.00322527])
      expected_e = Nx.tensor([0.00095185, 0.00386131, 0.00207291])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{scale: scale}} = new_state
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(scale, Nx.tensor(1.0e-2))
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_state(1.0e-2)

      params = {
        {
          Nx.tensor([0.58813851, 0.27981229, 0.17335737]),
          {{}, Nx.tensor([0.21444265, 0.63923396, 0.12755156])}
        }
      }

      updates = {
        {
          Nx.tensor([0.48363215, 0.7147937, 0.32252682]),
          {{}, Nx.tensor([0.09518468, 0.38613084, 0.20729078])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.00483632, 0.00714794, 0.00322527])
      expected_e = Nx.tensor([0.00095185, 0.00386131, 0.00207291])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{scale: scale}} = new_state
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(scale, Nx.tensor(1.0e-2))
    end
  end

  describe "scale_by_adam" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam(b1: 0.5)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> scale_by_adam()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state_1, adam_state_2} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state_1
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state_2
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_adam()
      params = %{a: Nx.tensor([0.29236649, 0.26508023, 0.05959644])}
      updates = %{a: Nx.tensor([0.01461005, 0.3796587, 0.76886989])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.99999267, 0.9999933, 0.9999933])
      expected_next_mu_a = Nx.tensor([0.00146101, 0.03796587, 0.07688699])
      expected_next_nu_a = Nx.tensor([2.1345357e-07, 1.4414072e-04, 5.9116090e-04])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates

      assert {%{mu: %{a: actual_next_mu_a}, nu: %{a: actual_next_nu_a}, count: actual_next_count}} =
               new_state

      assert_all_close(actual_a, expected_a)
      assert_all_close(actual_next_mu_a, expected_next_mu_a)
      assert_all_close(actual_next_nu_a, expected_next_nu_a)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_adam()

      params = %{
        a: %{
          b: Nx.tensor([0.16028131, 0.82155978, 0.67870557]),
          c: %{d: %{}, e: Nx.tensor([[0.42164469, 0.59406027, 0.24703223]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.37850456, 0.80079877, 0.16309247]),
          c: %{d: %{}, e: Nx.tensor([[0.29081831, 0.29872105, 0.48405271]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.9999934, 0.9999933, 0.99999315])
      expected_e = Nx.tensor([[0.9999933, 0.9999933, 0.9999933]])
      expected_next_mu_b = Nx.tensor([0.03785046, 0.08007988, 0.01630925])
      expected_next_mu_e = Nx.tensor([[0.02908183, 0.0298721, 0.04840527]])
      expected_next_nu_b = Nx.tensor([1.4326570e-04, 6.4127869e-04, 2.6599155e-05])
      expected_next_nu_e = Nx.tensor([[8.4575287e-05, 8.9234265e-05, 2.3430702e-04]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu, count: actual_next_count}} = new_state
      assert %{a: %{b: actual_next_mu_b, c: %{d: %{}, e: actual_next_mu_e}}} = new_mu
      assert %{a: %{b: actual_next_nu_b, c: %{d: %{}, e: actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_adam()

      params = {
        {
          Nx.tensor([0.16028131, 0.82155978, 0.67870557]),
          {{}, Nx.tensor([[0.42164469, 0.59406027, 0.24703223]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.37850456, 0.80079877, 0.16309247]),
          {{}, Nx.tensor([[0.29081831, 0.29872105, 0.48405271]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.9999934, 0.9999933, 0.99999315])
      expected_e = Nx.tensor([[0.9999933, 0.9999933, 0.9999933]])
      expected_next_mu_b = Nx.tensor([0.03785046, 0.08007988, 0.01630925])
      expected_next_mu_e = Nx.tensor([[0.02908183, 0.0298721, 0.04840527]])
      expected_next_nu_b = Nx.tensor([1.4326570e-04, 6.4127869e-04, 2.6599155e-05])
      expected_next_nu_e = Nx.tensor([[8.4575287e-05, 8.9234265e-05, 2.3430702e-04]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu, count: actual_next_count}} = new_state
      assert {{actual_next_mu_b, {{}, actual_next_mu_e}}} = new_mu
      assert {{actual_next_nu_b, {{}, actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
      assert_equal(actual_next_count, expected_next_count)
    end
  end

  describe "scale_by_belief" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_belief()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {belief_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = belief_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_belief(b1: 0.4)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {belief_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = belief_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_belief() |> scale_by_belief()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {belief_state_1, belief_state_2} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = belief_state_1
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = belief_state_2
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_belief() |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {belief_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = belief_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_belief()
      params = %{a: Nx.tensor([0.35582285, 0.02904734, 0.8684706])}
      updates = %{a: Nx.tensor([0.64641294, 0.19990149, 0.54263212])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.9999934, 0.99999326, 0.9999933])
      expected_next_mu_a = Nx.tensor([0.0646413, 0.01999015, 0.05426321])
      expected_next_nu_a = Nx.tensor([4.1784969e-04, 3.9960611e-05, 2.9444962e-04])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates

      assert {%{mu: %{a: actual_next_mu_a}, nu: %{a: actual_next_nu_a}, count: actual_next_count}} =
               new_state

      assert_all_close(actual_a, expected_a)
      assert_all_close(actual_next_mu_a, expected_next_mu_a)
      assert_all_close(actual_next_nu_a, expected_next_nu_a)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_belief()

      params = %{
        a: %{
          b: Nx.tensor([0.48266117, 0.21594939, 0.25310925]),
          c: %{d: %{}, e: Nx.tensor([[0.08780911, 0.25273182, 0.02973737]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.15456417, 0.03338711, 0.47241908]),
          c: %{d: %{}, e: Nx.tensor([[0.76352976, 0.86033023, 0.22758512]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.9999933, 0.9999933, 0.99999326])
      expected_e = Nx.tensor([[0.9999934, 0.99999326, 0.9999933]])
      expected_next_mu_b = Nx.tensor([0.01545642, 0.00333871, 0.04724191])
      expected_next_mu_e = Nx.tensor([[0.07635298, 0.08603302, 0.02275851]])
      expected_next_nu_b = Nx.tensor([2.3890085e-05, 1.1146991e-06, 2.2317980e-04])
      expected_next_nu_e = Nx.tensor([[5.8297772e-04, 7.4016815e-04, 5.1794988e-05]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu, count: actual_next_count}} = new_state
      assert %{a: %{b: actual_next_mu_b, c: %{d: %{}, e: actual_next_mu_e}}} = new_mu
      assert %{a: %{b: actual_next_nu_b, c: %{d: %{}, e: actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_belief()

      params = {
        {
          Nx.tensor([0.48266117, 0.21594939, 0.25310925]),
          {{}, Nx.tensor([[0.08780911, 0.25273182, 0.02973737]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.15456417, 0.03338711, 0.47241908]),
          {{}, Nx.tensor([[0.76352976, 0.86033023, 0.22758512]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.9999933, 0.9999933, 0.99999326])
      expected_e = Nx.tensor([[0.9999934, 0.99999326, 0.9999933]])
      expected_next_mu_b = Nx.tensor([0.01545642, 0.00333871, 0.04724191])
      expected_next_mu_e = Nx.tensor([[0.07635298, 0.08603302, 0.02275851]])
      expected_next_nu_b = Nx.tensor([2.3890085e-05, 1.1146991e-06, 2.2317980e-04])
      expected_next_nu_e = Nx.tensor([[5.8297772e-04, 7.4016815e-04, 5.1794988e-05]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu, count: actual_next_count}} = new_state
      assert {{actual_next_mu_b, {{}, actual_next_mu_e}}} = new_mu
      assert {{actual_next_nu_b, {{}, actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
      assert_equal(actual_next_count, expected_next_count)
    end
  end

  describe "scale_by_radam" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_radam()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_radam(b1: 0.5)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_radam() |> scale_by_radam()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state_1, adam_state_2} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state_1
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state_2
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_radam() |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_radam()
      params = %{a: Nx.tensor([0.71289699, 0.29554161, 0.50779425])}
      updates = %{a: Nx.tensor([0.88675452, 0.21455035, 0.53581422])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.88675433, 0.2145503, 0.53581405])
      expected_next_mu_a = Nx.tensor([0.08867545, 0.02145503, 0.05358142])
      expected_next_nu_a = Nx.tensor([7.863336e-04, 4.603185e-05, 2.870969e-04])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates

      assert {%{mu: %{a: actual_next_mu_a}, nu: %{a: actual_next_nu_a}, count: actual_next_count}} =
               new_state

      assert_all_close(actual_a, expected_a)
      assert_all_close(actual_next_mu_a, expected_next_mu_a)
      assert_all_close(actual_next_nu_a, expected_next_nu_a)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_radam()

      params = %{
        a: %{
          b: Nx.tensor([0.72504156, 0.86982723, 0.58679938]),
          c: %{d: %{}, e: Nx.tensor([[0.26001513, 0.62556789, 0.29528421]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.01536453, 0.61977439, 0.561842]),
          c: %{d: %{}, e: Nx.tensor([[0.03755132, 0.80392208, 0.87391938]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.01536453, 0.6197742, 0.56184185])
      expected_e = Nx.tensor([[0.03755131, 0.8039219, 0.8739191]])
      expected_next_mu_b = Nx.tensor([0.00153645, 0.06197744, 0.0561842])
      expected_next_mu_e = Nx.tensor([[0.00375513, 0.0803922, 0.08739194]])
      expected_next_nu_b = Nx.tensor([2.3606893e-07, 3.8412030e-04, 3.1566643e-04])
      expected_next_nu_e = Nx.tensor([[1.4101014e-06, 6.4629072e-04, 7.6373509e-04]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu, count: actual_next_count}} = new_state
      assert %{a: %{b: actual_next_mu_b, c: %{d: %{}, e: actual_next_mu_e}}} = new_mu
      assert %{a: %{b: actual_next_nu_b, c: %{d: %{}, e: actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_radam()

      params = {
        {
          Nx.tensor([0.72504156, 0.86982723, 0.58679938]),
          {{}, Nx.tensor([[0.26001513, 0.62556789, 0.29528421]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.01536453, 0.61977439, 0.561842]),
          {{}, Nx.tensor([[0.03755132, 0.80392208, 0.87391938]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.01536453, 0.6197742, 0.56184185])
      expected_e = Nx.tensor([[0.03755131, 0.8039219, 0.8739191]])
      expected_next_mu_b = Nx.tensor([0.00153645, 0.06197744, 0.0561842])
      expected_next_mu_e = Nx.tensor([[0.00375513, 0.0803922, 0.08739194]])
      expected_next_nu_b = Nx.tensor([2.3606893e-07, 3.8412030e-04, 3.1566643e-04])
      expected_next_nu_e = Nx.tensor([[1.4101014e-06, 6.4629072e-04, 7.6373509e-04]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu, count: actual_next_count}} = new_state
      assert {{actual_next_mu_b, {{}, actual_next_mu_e}}} = new_mu
      assert {{actual_next_nu_b, {{}, actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
      assert_equal(actual_next_count, expected_next_count)
    end
  end

  describe "scale_by_rms" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_rms()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {rms_state} = init_fn.(params)
      assert %{nu: %{a: nu_a}} = rms_state
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_rms(initial_scale: 0.1)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {rms_state} = init_fn.(params)
      assert %{nu: %{a: nu_a}} = rms_state
      assert_equal(nu_a, Nx.tensor([0.1, 0.1, 0.1]))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_rms() |> scale_by_rms()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {rms_state_1, rms_state_2} = init_fn.(params)
      assert %{nu: %{a: nu_a}} = rms_state_1
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert %{nu: %{a: nu_a}} = rms_state_2
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_rms() |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {rms_state} = init_fn.(params)
      assert %{nu: %{a: nu_a}} = rms_state
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_rms()
      params = %{a: Nx.tensor([0.77100057, 0.98078091, 0.78499164])}
      updates = %{a: Nx.tensor([0.25156708, 0.30524656, 0.97350756])}
      state = init_fn.(params)

      expected_a = Nx.tensor([3.162275, 3.162276, 3.1622777])
      expected_next_nu_a = Nx.tensor([0.0063286, 0.00931755, 0.0947717])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert {%{nu: %{a: actual_next_nu_a}}} = new_state
      assert_all_close(actual_a, expected_a)
      assert_all_close(expected_next_nu_a, actual_next_nu_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_rms()

      params = %{
        a: %{
          b: Nx.tensor([0.0553049, 0.21828064, 0.98751916]),
          c: %{d: %{}, e: Nx.tensor([[0.17757973, 0.67966022, 0.19382288]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.61220327, 0.73535765, 0.42179138]),
          c: %{d: %{}, e: Nx.tensor([[0.39331236, 0.27389305, 0.30131908]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([3.1622772, 3.1622772, 3.162277])
      expected_e = Nx.tensor([[3.1622767, 3.1622758, 3.162276]])
      expected_next_nu_b = Nx.tensor([0.03747929, 0.05407509, 0.0177908])
      expected_next_nu_e = Nx.tensor([[0.01546946, 0.00750174, 0.00907932]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{nu: new_nu}} = new_state
      assert %{a: %{b: actual_next_nu_b, c: %{d: %{}, e: actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_rms()

      params = {
        {
          Nx.tensor([0.0553049, 0.21828064, 0.98751916]),
          {{}, Nx.tensor([[0.17757973, 0.67966022, 0.19382288]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.61220327, 0.73535765, 0.42179138]),
          {{}, Nx.tensor([[0.39331236, 0.27389305, 0.30131908]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([3.1622772, 3.1622772, 3.162277])
      expected_e = Nx.tensor([[3.1622767, 3.1622758, 3.162276]])
      expected_next_nu_b = Nx.tensor([0.03747929, 0.05407509, 0.0177908])
      expected_next_nu_e = Nx.tensor([[0.01546946, 0.00750174, 0.00907932]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{nu: new_nu}} = new_state
      assert {{actual_next_nu_b, {{}, actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
    end
  end

  describe "scale_by_rss" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_rss()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {rss_state} = init_fn.(params)
      assert %{sum_of_squares: %{a: sum_of_squares_a}} = rss_state
      assert_equal(sum_of_squares_a, Nx.tensor([0.1, 0.1, 0.1]))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_rss(initial_accumulator_value: 0.2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {rss_state} = init_fn.(params)
      assert %{sum_of_squares: %{a: sum_of_squares_a}} = rss_state
      assert_equal(sum_of_squares_a, Nx.tensor([0.2, 0.2, 0.2]))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_rss() |> scale_by_rss()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {rss_state_1, rss_state_2} = init_fn.(params)
      assert %{sum_of_squares: %{a: sum_of_squares_a}} = rss_state_1
      assert_equal(sum_of_squares_a, Nx.tensor([0.1, 0.1, 0.1]))
      assert %{sum_of_squares: %{a: sum_of_squares_a}} = rss_state_2
      assert_equal(sum_of_squares_a, Nx.tensor([0.1, 0.1, 0.1]))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_rss() |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {rss_state} = init_fn.(params)
      assert %{sum_of_squares: %{a: sum_of_squares_a}} = rss_state
      assert_equal(sum_of_squares_a, Nx.tensor([0.1, 0.1, 0.1]))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_rss()
      params = %{a: Nx.tensor([0.41327447, 0.06948837, 0.03234601])}
      updates = %{a: Nx.tensor([0.2137085, 0.84399692, 0.63099467])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.55993116, 0.93642795, 0.89401275])
      expected_next_sum_of_squares_a = Nx.tensor([0.14567132, 0.81233084, 0.49815428])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert {%{sum_of_squares: %{a: actual_next_sum_of_squares_a}}} = new_state
      assert_all_close(actual_a, expected_a)
      assert_all_close(actual_next_sum_of_squares_a, expected_next_sum_of_squares_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_rss()

      params = %{
        a: %{
          b: Nx.tensor([0.92084601, 0.27218277, 0.56501597]),
          c: %{d: %{}, e: Nx.tensor([[0.92937211, 0.44536295, 0.95296635]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.79292352, 0.11484326, 0.84693855]),
          c: %{d: %{}, e: Nx.tensor([[0.13715272, 0.63276641, 0.5234425]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.92885643, 0.34135267, 0.9368279])
      expected_e = Nx.tensor([[0.39790204, 0.894515, 0.855929]])
      expected_next_sum_of_squares_b = Nx.tensor([0.72872776, 0.11318897, 0.8173049])
      expected_next_sum_of_squares_e = Nx.tensor([[0.11881087, 0.50039333, 0.37399206]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{sum_of_squares: new_sum_of_squares}} = new_state

      assert %{
               a: %{
                 b: actual_next_sum_of_squares_b,
                 c: %{d: %{}, e: actual_next_sum_of_squares_e}
               }
             } = new_sum_of_squares

      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_sum_of_squares_b, expected_next_sum_of_squares_b)
      assert_all_close(actual_next_sum_of_squares_e, expected_next_sum_of_squares_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_rss()

      params = {
        {
          Nx.tensor([0.92084601, 0.27218277, 0.56501597]),
          {{}, Nx.tensor([[0.92937211, 0.44536295, 0.95296635]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.79292352, 0.11484326, 0.84693855]),
          {{}, Nx.tensor([[0.13715272, 0.63276641, 0.5234425]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.92885643, 0.34135267, 0.9368279])
      expected_e = Nx.tensor([[0.39790204, 0.894515, 0.855929]])
      expected_next_sum_of_squares_b = Nx.tensor([0.72872776, 0.11318897, 0.8173049])
      expected_next_sum_of_squares_e = Nx.tensor([[0.11881087, 0.50039333, 0.37399206]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{sum_of_squares: new_sum_of_squares}} = new_state

      assert {
               {
                 actual_next_sum_of_squares_b,
                 {{}, actual_next_sum_of_squares_e}
               }
             } = new_sum_of_squares

      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_sum_of_squares_b, expected_next_sum_of_squares_b)
      assert_all_close(actual_next_sum_of_squares_e, expected_next_sum_of_squares_e)
    end
  end

  describe "scale_by_schedule" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_schedule(Axon.Schedules.polynomial_decay())
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {schedule_state} = init_fn.(params)
      assert %{count: count} = schedule_state
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}

      assert {init_fn, update_fn} =
               scale_by_schedule(Axon.Schedules.polynomial_decay())
               |> scale_by_schedule(Axon.Schedules.polynomial_decay())

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {schedule_state_2, schedule_state_1} = init_fn.(params)
      assert %{count: count} = schedule_state_1
      assert_equal(count, Nx.tensor(0))
      assert %{count: count} = schedule_state_2
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}

      assert {init_fn, update_fn} =
               scale_by_schedule(Axon.Schedules.polynomial_decay()) |> scale(1.0e-2)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {schedule_state} = init_fn.(params)
      assert %{count: count} = schedule_state
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_schedule(Axon.Schedules.polynomial_decay())
      params = %{a: Nx.tensor([0.77425031, 0.65418105, 0.86150202])}
      updates = %{a: Nx.tensor([0.56082198, 0.94549107, 0.54412826])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.00560822, 0.00945491, 0.00544128])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert {%{count: actual_next_count}} = new_state
      assert_all_close(actual_a, expected_a)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_schedule(Axon.Schedules.polynomial_decay())

      params = %{
        a: %{
          b: Nx.tensor([0.3440084, 0.16096481, 0.43997161]),
          c: %{d: %{}, e: Nx.tensor([[0.26168961, 0.40905451, 0.3061841]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.27159927, 0.37657519, 0.38219061]),
          c: %{d: %{}, e: Nx.tensor([[0.9613661, 0.30215168, 0.24110271]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.00271599, 0.00376575, 0.00382191])
      expected_e = Nx.tensor([[0.00961366, 0.00302152, 0.00241103]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{count: actual_next_count}} = new_state
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_schedule(Axon.Schedules.polynomial_decay())

      params = {
        {
          Nx.tensor([0.3440084, 0.16096481, 0.43997161]),
          {{}, Nx.tensor([[0.26168961, 0.40905451, 0.3061841]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.27159927, 0.37657519, 0.38219061]),
          {{}, Nx.tensor([[0.9613661, 0.30215168, 0.24110271]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.00271599, 0.00376575, 0.00382191])
      expected_e = Nx.tensor([[0.00961366, 0.00302152, 0.00241103]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{count: actual_next_count}} = new_state
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_equal(actual_next_count, expected_next_count)
    end
  end

  describe "scale_by_stddev" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_stddev()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {stddev_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}} = stddev_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_stddev(initial_scale: 0.5)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {stddev_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}} = stddev_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.5, 0.5, 0.5]))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}

      assert {init_fn, update_fn} =
               scale_by_stddev(initial_scale: 0.1) |> scale_by_stddev(initial_scale: 0.2)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {stddev_state_2, stddev_state_1} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}} = stddev_state_1
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.1, 0.1, 0.1]))
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}} = stddev_state_2
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.2, 0.2, 0.2]))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_stddev(initial_scale: 0.1) |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {stddev_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}} = stddev_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.1, 0.1, 0.1]))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_stddev()
      params = %{a: Nx.tensor([0.98013234, 0.0653057, 0.39361905])}
      updates = %{a: Nx.tensor([0.58050587, 0.04869076, 0.62340991])}
      state = init_fn.(params)

      expected_a = Nx.tensor([3.3333325, 3.333255, 3.3333328])
      expected_next_mu_a = Nx.tensor([0.05805059, 0.00486908, 0.06234099])
      expected_next_nu_a = Nx.tensor([0.03369871, 0.00023708, 0.03886399])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert {%{mu: %{a: actual_next_mu_a}, nu: %{a: actual_next_nu_a}}} = new_state
      assert_all_close(actual_a, expected_a)
      assert_all_close(actual_next_mu_a, expected_next_mu_a)
      assert_all_close(actual_next_nu_a, expected_next_nu_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_stddev()

      params = %{
        a: %{
          b: Nx.tensor([0.49792875, 0.04941673, 0.33815839]),
          c: %{d: %{}, e: Nx.tensor([[0.70057761, 0.3689184, 0.36608007]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.54587409, 0.04849768, 0.23020724]),
          c: %{d: %{}, e: Nx.tensor([[0.29348535, 0.79428645, 0.76129383]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([3.333333, 3.3332546, 3.33333])
      expected_e = Nx.tensor([[3.333331, 3.333333, 3.333333]])
      expected_next_mu_b = Nx.tensor([0.05458741, 0.00484977, 0.02302072])
      expected_next_mu_e = Nx.tensor([[0.02934854, 0.07942864, 0.07612938]])
      expected_next_nu_b = Nx.tensor([0.02979785, 0.0002352, 0.00529954])
      expected_next_nu_e = Nx.tensor([[0.00861336, 0.0630891, 0.05795683]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu}} = new_state
      assert %{a: %{b: actual_next_mu_b, c: %{d: %{}, e: actual_next_mu_e}}} = new_mu
      assert %{a: %{b: actual_next_nu_b, c: %{d: %{}, e: actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_stddev()

      params = {
        {
          Nx.tensor([0.49792875, 0.04941673, 0.33815839]),
          {{}, Nx.tensor([[0.70057761, 0.3689184, 0.36608007]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.54587409, 0.04849768, 0.23020724]),
          {{}, Nx.tensor([[0.29348535, 0.79428645, 0.76129383]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([3.333333, 3.3332546, 3.33333])
      expected_e = Nx.tensor([[3.333331, 3.333333, 3.333333]])
      expected_next_mu_b = Nx.tensor([0.05458741, 0.00484977, 0.02302072])
      expected_next_mu_e = Nx.tensor([[0.02934854, 0.07942864, 0.07612938]])
      expected_next_nu_b = Nx.tensor([0.02979785, 0.0002352, 0.00529954])
      expected_next_nu_e = Nx.tensor([[0.00861336, 0.0630891, 0.05795683]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu}} = new_state
      assert {{actual_next_mu_b, {{}, actual_next_mu_e}}} = new_mu
      assert {{actual_next_nu_b, {{}, actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
    end
  end

  describe "scale_by_trust_ratio" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_trust_ratio()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "constructs a stateless transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_trust_ratio(min_norm: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}

      assert {init_fn, update_fn} =
               scale_by_trust_ratio(min_norm: 1.0) |> scale_by_trust_ratio(min_norm: 1.0)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_adam() |> scale_by_trust_ratio(min_norm: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {adam_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = adam_state
      assert_equal(mu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(nu_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_trust_ratio(min_norm: 1.0)
      params = %{a: Nx.tensor([0.07719177, 0.1812708, 0.94959977])}
      updates = %{a: Nx.tensor([0.29626032, 0.328152, 0.20388144])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.29626033, 0.328152, 0.20388144])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert new_state == {}
      assert_all_close(actual_a, expected_a)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_trust_ratio(min_norm: 1.0)

      params = %{
        a: %{
          b: Nx.tensor([0.98282674, 0.34776357, 0.33319137]),
          c: %{d: %{}, e: Nx.tensor([[0.95596768, 0.67948137, 0.05268411]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.53616958, 0.24854466, 0.26695091]),
          c: %{d: %{}, e: Nx.tensor([[0.50354858, 0.91245821, 0.30518247]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.58683133, 0.27202922, 0.29217464])
      expected_e = Nx.tensor([[0.5443927, 0.98647004, 0.3299366]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_trust_ratio(min_norm: 1.0)

      params = {
        {
          Nx.tensor([0.98282674, 0.34776357, 0.33319137]),
          {{}, Nx.tensor([[0.95596768, 0.67948137, 0.05268411]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.53616958, 0.24854466, 0.26695091]),
          {{}, Nx.tensor([[0.50354858, 0.91245821, 0.30518247]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.58683133, 0.27202922, 0.29217464])
      expected_e = Nx.tensor([[0.5443927, 0.98647004, 0.3299366]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert new_state == {}
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
    end
  end

  describe "scale_by_yogi" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_yogi()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {yogi_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = yogi_state
      assert_equal(mu_a, Nx.tensor([1.0e-6, 1.0e-6, 1.0e-6]))
      assert_equal(nu_a, Nx.tensor([1.0e-6, 1.0e-6, 1.0e-6]))
      assert_equal(count, Nx.tensor(0))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_yogi(initial_accumulator_value: 1.0e-4)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {yogi_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = yogi_state
      assert_equal(mu_a, Nx.tensor([1.0e-4, 1.0e-4, 1.0e-4]))
      assert_equal(nu_a, Nx.tensor([1.0e-4, 1.0e-4, 1.0e-4]))
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_yogi() |> scale_by_yogi()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {yogi_state_1, yogi_state_2} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = yogi_state_1
      assert_equal(mu_a, Nx.tensor([1.0e-6, 1.0e-6, 1.0e-6]))
      assert_equal(nu_a, Nx.tensor([1.0e-6, 1.0e-6, 1.0e-6]))
      assert_equal(count, Nx.tensor(0))
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = yogi_state_2
      assert_equal(mu_a, Nx.tensor([1.0e-6, 1.0e-6, 1.0e-6]))
      assert_equal(nu_a, Nx.tensor([1.0e-6, 1.0e-6, 1.0e-6]))
      assert_equal(count, Nx.tensor(0))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = scale_by_yogi() |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {yogi_state} = init_fn.(params)
      assert %{mu: %{a: mu_a}, nu: %{a: nu_a}, count: count} = yogi_state
      assert_equal(mu_a, Nx.tensor([1.0e-6, 1.0e-6, 1.0e-6]))
      assert_equal(nu_a, Nx.tensor([1.0e-6, 1.0e-6, 1.0e-6]))
      assert_equal(count, Nx.tensor(0))
    end

    test "matches optax with simple container" do
      assert {init_fn, update_fn} = scale_by_yogi()
      params = %{a: Nx.tensor([0.39152084, 0.86061072, 0.22693509])}
      updates = %{a: Nx.tensor([0.10820288, 0.73034528, 0.6741126])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.95148116, 0.99770474, 0.9974302])
      expected_next_mu_a = Nx.tensor([0.01082119, 0.07303543, 0.06741216])
      expected_next_nu_a = Nx.tensor([1.2707865e-05, 5.3440424e-04, 4.5542780e-04])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates

      assert {%{mu: %{a: actual_next_mu_a}, nu: %{a: actual_next_nu_a}, count: actual_next_count}} =
               new_state

      assert_all_close(actual_a, expected_a)
      assert_all_close(actual_next_mu_a, expected_next_mu_a)
      assert_all_close(actual_next_nu_a, expected_next_nu_a)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "matches optax with nested container" do
      assert {init_fn, update_fn} = scale_by_yogi()

      params = %{
        a: %{
          b: Nx.tensor([0.87690482, 0.80993702, 0.87935556]),
          c: %{d: %{}, e: Nx.tensor([[0.00528695, 0.06690531, 0.12589192]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.47019351, 0.72034131, 0.32043362]),
          c: %{d: %{}, e: Nx.tensor([[0.84200356, 0.76360484, 0.55381714]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.99564576, 0.9976599, 0.99210596])
      expected_e = Nx.tensor([[0.9981149, 0.99784315, 0.9965868]])
      expected_next_mu_b = Nx.tensor([0.04702025, 0.07203503, 0.03204427])
      expected_next_mu_e = Nx.tensor([[0.08420125, 0.07636139, 0.05538262]])
      expected_next_nu_b = Nx.tensor([0.00022208, 0.00051989, 0.00010368])
      expected_next_nu_e = Nx.tensor([[0.00070997, 0.00058409, 0.00030771]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu, count: actual_next_count}} = new_state
      assert %{a: %{b: actual_next_mu_b, c: %{d: %{}, e: actual_next_mu_e}}} = new_mu
      assert %{a: %{b: actual_next_nu_b, c: %{d: %{}, e: actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
      assert_equal(actual_next_count, expected_next_count)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = scale_by_yogi()

      params = {
        {
          Nx.tensor([0.87690482, 0.80993702, 0.87935556]),
          {{}, Nx.tensor([[0.00528695, 0.06690531, 0.12589192]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.47019351, 0.72034131, 0.32043362]),
          {{}, Nx.tensor([[0.84200356, 0.76360484, 0.55381714]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.99564576, 0.9976599, 0.99210596])
      expected_e = Nx.tensor([[0.9981149, 0.99784315, 0.9965868]])
      expected_next_mu_b = Nx.tensor([0.04702025, 0.07203503, 0.03204427])
      expected_next_mu_e = Nx.tensor([[0.08420125, 0.07636139, 0.05538262]])
      expected_next_nu_b = Nx.tensor([0.00022208, 0.00051989, 0.00010368])
      expected_next_nu_e = Nx.tensor([[0.00070997, 0.00058409, 0.00030771]])
      expected_next_count = Nx.tensor(1)

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{mu: new_mu, nu: new_nu, count: actual_next_count}} = new_state
      assert {{actual_next_mu_b, {{}, actual_next_mu_e}}} = new_mu
      assert {{actual_next_nu_b, {{}, actual_next_nu_e}}} = new_nu
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_mu_b, expected_next_mu_b)
      assert_all_close(actual_next_mu_e, expected_next_mu_e)
      assert_all_close(actual_next_nu_b, expected_next_nu_b)
      assert_all_close(actual_next_nu_e, expected_next_nu_e)
      assert_equal(actual_next_count, expected_next_count)
    end
  end

  describe "trace" do
    test "constructs a stateful transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = trace()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {trace_state} = init_fn.(params)
      assert %{trace: %{a: trace_a}} = trace_state
      assert_equal(trace_a, Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "constructs a stateful transformation with options" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = trace(decay: 0.8)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {trace_state} = init_fn.(params)
      assert %{trace: %{a: trace_a}} = trace_state
      assert_equal(trace_a, Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = trace() |> trace()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {trace_state_2, trace_state_1} = init_fn.(params)
      assert %{trace: %{a: trace_a}} = trace_state_1
      assert_equal(trace_a, Nx.tensor([0.0, 0.0, 0.0]))
      assert %{trace: %{a: trace_a}} = trace_state_2
      assert_equal(trace_a, Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "composes with stateless transformation" do
      params = %{a: Nx.tensor([1.0, 2.0, 3.0])}
      assert {init_fn, update_fn} = trace() |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert {trace_state} = init_fn.(params)
      assert %{trace: %{a: trace_a}} = trace_state
      assert_equal(trace_a, Nx.tensor([0.0, 0.0, 0.0]))
    end

    test "matches optax with simple container, nesterov: false" do
      assert {init_fn, update_fn} = trace(nesterov: false)
      params = %{a: Nx.tensor([0.54044065, 0.54168045, 0.14243068])}
      updates = %{a: Nx.tensor([0.76976679, 0.19561062, 0.84724249])}
      state = init_fn.(params)

      expected_a = Nx.tensor([0.7697668, 0.19561061, 0.8472425])
      expected_next_trace = Nx.tensor([0.7697668, 0.19561061, 0.8472425])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert {%{trace: %{a: actual_next_trace}}} = new_state
      assert_all_close(actual_a, expected_a)
      assert_all_close(actual_next_trace, expected_next_trace)
    end

    test "matches optax with nested container, nesterov: false" do
      assert {init_fn, update_fn} = trace(nesterov: false)

      params = %{
        a: %{
          b: Nx.tensor([0.23468207, 0.75940123, 0.06601013]),
          c: %{d: %{}, e: Nx.tensor([[0.68877159, 0.84383744, 0.15230977]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.60272336, 0.42772071, 0.39653623]),
          c: %{d: %{}, e: Nx.tensor([[0.25453278, 0.64759897, 0.71080799]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.60272336, 0.4277207, 0.39653623])
      expected_e = Nx.tensor([[0.25453278, 0.647599, 0.710808]])
      expected_next_trace_b = Nx.tensor([0.60272336, 0.4277207, 0.39653623])
      expected_next_trace_e = Nx.tensor([[0.25453278, 0.647599, 0.710808]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{trace: new_trace}} = new_state
      assert %{a: %{b: actual_next_trace_b, c: %{d: %{}, e: actual_next_trace_e}}} = new_trace
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_trace_b, expected_next_trace_b)
      assert_all_close(actual_next_trace_e, expected_next_trace_e)
    end

    test "matches optax with simple container, nesterov: true" do
      assert {init_fn, update_fn} = trace(nesterov: true)
      params = %{a: Nx.tensor([0.05727068, 0.71336316, 0.52111667])}
      updates = %{a: Nx.tensor([0.99510349, 0.38321624, 0.37485662])}
      state = init_fn.(params)

      expected_a = Nx.tensor([1.8906965, 0.7281108, 0.7122276])
      expected_next_trace = Nx.tensor([0.9951035, 0.38321623, 0.37485662])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: actual_a} = new_updates
      assert {%{trace: %{a: actual_next_trace}}} = new_state
      assert_all_close(actual_a, expected_a)
      assert_all_close(actual_next_trace, expected_next_trace)
    end

    test "matches optax with nested container, nesterov: true" do
      assert {init_fn, update_fn} = trace(nesterov: true)

      params = %{
        a: %{
          b: Nx.tensor([0.81068757, 0.89196671, 0.21672469]),
          c: %{d: %{}, e: Nx.tensor([[0.9194404, 0.19829658, 0.96960522]])}
        }
      }

      updates = %{
        a: %{
          b: Nx.tensor([0.21182614, 0.29456406, 0.50427876]),
          c: %{d: %{}, e: Nx.tensor([[0.26525984, 0.66349034, 0.11212149]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.40246966, 0.55967176, 0.95812964])
      expected_e = Nx.tensor([[0.5039937, 1.2606317, 0.21303083]])
      expected_next_trace_b = Nx.tensor([0.21182615, 0.29456407, 0.5042788])
      expected_next_trace_e = Nx.tensor([[0.26525983, 0.66349036, 0.11212149]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert %{a: %{b: actual_b, c: %{d: %{}, e: actual_e}}} = new_updates
      assert {%{trace: new_trace}} = new_state
      assert %{a: %{b: actual_next_trace_b, c: %{d: %{}, e: actual_next_trace_e}}} = new_trace
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_trace_b, expected_next_trace_b)
      assert_all_close(actual_next_trace_e, expected_next_trace_e)
    end

    test "supports generic container" do
      assert {init_fn, update_fn} = trace(nesterov: true)

      params = {
        {
          Nx.tensor([0.81068757, 0.89196671, 0.21672469]),
          {{}, Nx.tensor([[0.9194404, 0.19829658, 0.96960522]])}
        }
      }

      updates = {
        {
          Nx.tensor([0.21182614, 0.29456406, 0.50427876]),
          {{}, Nx.tensor([[0.26525984, 0.66349034, 0.11212149]])}
        }
      }

      state = init_fn.(params)

      expected_b = Nx.tensor([0.40246966, 0.55967176, 0.95812964])
      expected_e = Nx.tensor([[0.5039937, 1.2606317, 0.21303083]])
      expected_next_trace_b = Nx.tensor([0.21182615, 0.29456407, 0.5042788])
      expected_next_trace_e = Nx.tensor([[0.26525983, 0.66349036, 0.11212149]])

      assert {new_updates, new_state} = update_fn.(updates, state, params)
      assert {{actual_b, {{}, actual_e}}} = new_updates
      assert {%{trace: new_trace}} = new_state
      assert {{actual_next_trace_b, {{}, actual_next_trace_e}}} = new_trace
      assert_all_close(actual_b, expected_b)
      assert_all_close(actual_e, expected_e)
      assert_all_close(actual_next_trace_b, expected_next_trace_b)
      assert_all_close(actual_next_trace_e, expected_next_trace_e)
    end
  end
end
