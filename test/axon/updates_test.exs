defmodule Axon.UpdatesTest do
  use ExUnit.Case
  doctest Axon.Updates

  import Axon.Updates
  import AxonTestUtil

  describe "add_decayed_weights" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = add_decayed_weights(decay: 0.95)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1, 2, 3])}

      assert {init_fn, update_fn} =
               add_decayed_weights(decay: 0.95) |> add_decayed_weights(decay: 0.95)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    # TODO
    # test "composes with stateful transformation" do
    # end

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
  end

  describe "clip" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = clip(delta: 2.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = clip(delta: 2.0) |> clip(delta: 2.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    # TODO
    # test "composes with stateful transformation" do
    # end

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
  end

  describe "clip_by_global_norm" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = clip_by_global_norm(max_norm: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1, 2, 3])}

      assert {init_fn, update_fn} =
               clip_by_global_norm(max_norm: 1.0) |> clip_by_global_norm(max_norm: 1.0)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    # TODO
    # test "composes with stateful transformation" do
    # end

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
  end

  describe "centralize" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = centralize()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = centralize() |> centralize()
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    # TODO
    # test "composes with stateful transformation" do
    # end

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
  end

  describe "scale" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = scale(1.0e-2) |> scale(1.0e-2)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    # TODO
    # test "composes with stateful transformation" do
    # end

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
  end

  describe "scale_by_trust_ratio" do
    test "constructs a stateless transformation" do
      params = %{a: Nx.tensor([1, 2, 3])}
      assert {init_fn, update_fn} = scale_by_trust_ratio(min_norm: 1.0)
      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    test "composes with itself" do
      params = %{a: Nx.tensor([1, 2, 3])}

      assert {init_fn, update_fn} =
               scale_by_trust_ratio(min_norm: 1.0) |> scale_by_trust_ratio(min_norm: 1.0)

      assert is_function(init_fn, 1)
      assert is_function(update_fn, 3)
      assert init_fn.(params) == {}
    end

    # TODO
    # test "composes with stateful transformation" do
    # end

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
  end
end
