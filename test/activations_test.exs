defmodule Axon.ActivationsTest do
  use ExUnit.Case, async: true
  doctest Axon.Activations

  import Nx.Defn

  describe "celu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.534882128238678, 0.6075059175491333, 0.8809065222740173])
      expected = Nx.tensor([0.534882128238678, 0.6075059175491333, 0.8809065222740173])
      actual = Axon.Activations.celu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.5515183806419373, 0.01937619037926197],
          [0.11977817863225937, 0.11377150565385818]
        ])

      expected =
        Nx.tensor([
          [0.5515183806419373, 0.01937619037926197],
          [0.11977817863225937, 0.11377150565385818]
        ])

      actual = Axon.Activations.celu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.5185574293136597, 0.3031159043312073]],
          [[0.9700577855110168, 0.12915539741516113]]
        ])

      expected =
        Nx.tensor([
          [[0.5185574293136597, 0.3031159043312073]],
          [[0.9700577855110168, 0.12915539741516113]]
        ])

      actual = Axon.Activations.celu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 1 and type {:f, 32} and alpha: 0.5" do
      a = Nx.tensor([0.5254644751548767, 0.7386103868484497, 0.7302365303039551])
      expected = Nx.tensor([0.5254644751548767, 0.7386103868484497, 0.7302365303039551])
      actual = Axon.Activations.celu(a, alpha: 0.5)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 2 and type {:f, 32} and alpha: 0.5" do
      a =
        Nx.tensor([
          [0.3322221636772156, 0.8841397166252136],
          [0.8507494926452637, 0.307709664106369]
        ])

      expected =
        Nx.tensor([
          [0.3322221636772156, 0.8841397166252136],
          [0.8507494926452637, 0.307709664106369]
        ])

      actual = Axon.Activations.celu(a, alpha: 0.5)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 3 and type {:f, 32} and alpha: 0.5" do
      a =
        Nx.tensor([
          [[0.8232214450836182, 0.9869292378425598]],
          [[0.6635103821754456, 0.9175488948822021]]
        ])

      expected =
        Nx.tensor([
          [[0.8232214450836182, 0.9869292378425598]],
          [[0.6635103821754456, 0.9175488948822021]]
        ])

      actual = Axon.Activations.celu(a, alpha: 0.5)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.0006543696508742869, 0.9109137058258057, 0.5773505568504333])
      expected = Nx.tensor([1.0, 1.0, 1.0])

      actual =
        jit(
          fn x ->
            grad(x, &Nx.sum(Axon.Activations.celu(&1)))
          end,
          [a]
        )

      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.8410307765007019, 0.7737964987754822],
          [0.3063606917858124, 0.47949355840682983]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])

      actual =
        jit(
          fn x ->
            grad(x, &Nx.sum(Axon.Activations.celu(&1)))
          end,
          [a]
        )

      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.7361555695533752, 0.07279454171657562]],
          [[0.12938232719898224, 0.19182132184505463]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])

      actual =
        jit(
          fn x ->
            grad(x, &Nx.sum(Axon.Activations.celu(&1)))
          end,
          [a]
        )

      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "elu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.05110076069831848, 0.05315212532877922, 0.2063606083393097])
      expected = Nx.tensor([0.05110076069831848, 0.05315212532877922, 0.2063606083393097])
      actual = Axon.Activations.elu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.98786860704422, 0.7632235884666443],
          [0.12698937952518463, 0.2697761058807373]
        ])

      expected =
        Nx.tensor([
          [0.98786860704422, 0.7632235884666443],
          [0.12698937952518463, 0.2697761058807373]
        ])

      actual = Axon.Activations.elu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.9454386234283447, 0.8242208957672119]],
          [[0.26190897822380066, 0.7501916885375977]]
        ])

      expected =
        Nx.tensor([
          [[0.9454386234283447, 0.8242208957672119]],
          [[0.26190897822380066, 0.7501916885375977]]
        ])

      actual = Axon.Activations.elu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 1 and type {:f, 32} and alpha: 0.5" do
      a = Nx.tensor([0.5150681734085083, 0.8869504928588867, 0.6374541521072388])
      expected = Nx.tensor([0.5150681734085083, 0.8869504928588867, 0.6374541521072388])
      actual = Axon.Activations.elu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 2 and type {:f, 32} and alpha: 0.5" do
      a =
        Nx.tensor([
          [0.28927579522132874, 0.47236892580986023],
          [0.3614945411682129, 0.9077596664428711]
        ])

      expected =
        Nx.tensor([
          [0.28927579522132874, 0.47236892580986023],
          [0.3614945411682129, 0.9077596664428711]
        ])

      actual = Axon.Activations.elu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 3 and type {:f, 32} and alpha: 0.5" do
      a =
        Nx.tensor([
          [[0.46715274453163147, 0.39317840337753296]],
          [[0.47547057271003723, 0.3903312087059021]]
        ])

      expected =
        Nx.tensor([
          [[0.46715274453163147, 0.39317840337753296]],
          [[0.47547057271003723, 0.3903312087059021]]
        ])

      actual = Axon.Activations.elu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.2467288225889206, 0.047934457659721375, 0.6276589035987854])
      expected = Nx.tensor([1.0, 1.0, 1.0])
      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.elu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.35968613624572754, 0.2542852759361267],
          [0.19061528146266937, 0.5906847715377808]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])
      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.elu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.10630486905574799, 0.5689958333969116]],
          [[0.5406809449195862, 0.10776042938232422]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])
      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.elu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "gelu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.33406558632850647, 0.5005938410758972, 0.8558046817779541])
      expected = Nx.tensor([0.21074023842811584, 0.34624648094177246, 0.6880216598510742])
      actual = Axon.Activations.gelu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.65863037109375, 0.7466029524803162],
          [0.29895421862602234, 0.5063883662223816]
        ])

      expected =
        Nx.tensor([
          [0.49063578248023987, 0.5766375660896301],
          [0.18460796773433685, 0.3512856364250183]
        ])

      actual = Axon.Activations.gelu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.5786859393119812, 0.42829328775405884]],
          [[0.8231776356697083, 0.617345929145813]]
        ])

      expected =
        Nx.tensor([
          [[0.4158434271812439, 0.28514963388442993]],
          [[0.6542587280273438, 0.45158651471138]]
        ])

      actual = Axon.Activations.gelu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.5424693822860718, 0.17958815395832062, 0.6342728137969971])
      expected = Nx.tensor([0.89305579662323, 0.641761302947998, 0.9439805150032043])
      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.gelu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.27655771374702454, 0.9601503610610962],
          [0.06698627024888992, 0.1320214420557022]
        ])

      expected =
        Nx.tensor([
          [0.7151311039924622, 1.073091745376587],
          [0.5533674359321594, 0.6047282814979553]
        ])

      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.gelu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.6051793694496155, 0.11723373085260391]],
          [[0.7999136447906494, 0.4582436978816986]]
        ])

      expected =
        Nx.tensor([
          [[0.9285023212432861, 0.5931117534637451]],
          [[1.0198638439178467, 0.8412032127380371]]
        ])

      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.gelu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "relu" do
    defn grad_relu(x), do: grad(x, &Nx.mean(Axon.Activations.relu(&1)))

    test "returns correct gradient with custom grad" do
      assert Nx.all_close(
               grad_relu(Nx.iota({1, 3}, type: {:f, 32})),
               Nx.tensor([[0.0, 0.3333333432674408, 0.3333333432674408]])
             ) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.5017861723899841, 0.8087382912635803, 0.2827244997024536])
      expected = Nx.tensor([0.5017861723899841, 0.8087382912635803, 0.2827244997024536])
      actual = Axon.Activations.relu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.6976759433746338, 0.9168339371681213],
          [0.5028178691864014, 0.46735450625419617]
        ])

      expected =
        Nx.tensor([
          [0.6976759433746338, 0.9168339371681213],
          [0.5028178691864014, 0.46735450625419617]
        ])

      actual = Axon.Activations.relu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.38326719403266907, 0.5943251252174377]],
          [[0.6940548419952393, 0.8027238249778748]]
        ])

      expected =
        Nx.tensor([
          [[0.38326719403266907, 0.5943251252174377]],
          [[0.6940548419952393, 0.8027238249778748]]
        ])

      actual = Axon.Activations.relu(a)
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.2746083736419678, 0.23505471646785736, 0.5368936657905579])
      expected = Nx.tensor([1.0, 1.0, 1.0])
      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.7280408143997192, 0.4684365689754486],
          [0.5426321029663086, 0.7235316038131714]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])
      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.03652488812804222, 0.6321760416030884]],
          [[0.45222511887550354, 0.18244826793670654]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])
      actual = jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu(&1))) end, [a])
      assert Nx.all_close(expected, actual) == Nx.tensor(1, type: {:u, 8})
    end
  end

  describe "sigmoid" do
    defn value_and_grad_sigmoid(x), do: value_and_grad(x, &Axon.Activations.sigmoid(&1))

    defn value_and_grad_sum_sigmoid(x),
      do: value_and_grad(x, &Nx.sum(Axon.Activations.sigmoid(&1)))

    test "value_and_grad" do
      assert {value, grad} = value_and_grad_sigmoid(Nx.tensor(5.0))
      assert Nx.all_close(value, Nx.tensor(0.9933072)) == Nx.tensor(1, type: {:u, 8})
      assert Nx.all_close(grad, Nx.tensor(0.00664803)) == Nx.tensor(1, type: {:u, 8})

      assert {value, grad} =
               value_and_grad_sum_sigmoid(Nx.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]))

      assert Nx.all_close(value, Nx.tensor(3.5))

      assert Nx.all_close(
               grad,
               Nx.tensor([
                 0.04517666,
                 0.10499358,
                 0.19661194,
                 0.25,
                 0.19661193,
                 0.10499363,
                 0.04517666
               ])
             )
    end
  end

  describe "softmax" do
    test "raises on bad axis" do
      assert_raise ArgumentError, ~r/softmax axis must be within rank of tensor/, fn ->
        Axon.Activations.softmax(Nx.iota({1, 3}), axis: 2)
      end
    end
  end

  describe "softplus" do
    defn value_and_grad_softplus(x), do: value_and_grad(x, &Axon.Activations.softplus(&1))

    defn value_and_grad_sum_softplus(x),
      do: value_and_grad(x, &Nx.sum(Axon.Activations.softplus(&1)))

    test "value_and_grad" do
      assert {value, grad} = value_and_grad_softplus(Nx.tensor(5.0))
      assert Nx.all_close(value, Nx.tensor(5.0067153))
      assert Nx.all_close(grad, Nx.tensor(0.9933072))

      assert {value, grad} =
               value_and_grad_sum_softplus(
                 Nx.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
               )

      assert Nx.all_close(value, Nx.tensor(11.707001))

      assert Nx.all_close(
               grad,
               Nx.tensor([
                 0.01798621,
                 0.04742587,
                 0.11920291,
                 0.2689414,
                 0.5,
                 0.73105854,
                 0.880797,
                 0.95257413,
                 0.9820139
               ])
             )

      assert {value, grad} =
               value_and_grad_sum_softplus(
                 Nx.tensor([
                   [
                     3.91343785e-02,
                     2.02403838e-02,
                     1.12020537e-02,
                     2.83025027e-02,
                     2.39001730e-02,
                     3.32025503e-02,
                     1.57335941e-02,
                     3.85219835e-02,
                     4.17842921e-02,
                     3.35262782e-02,
                     7.44309500e-03,
                     4.03720339e-02,
                     4.42154224e-02,
                     3.90086390e-02,
                     4.04843100e-02,
                     4.23467114e-02,
                     2.15607638e-02,
                     3.81104307e-02,
                     4.93991938e-02,
                     4.31956985e-02,
                     3.86686089e-02,
                     2.52724580e-02,
                     5.28243431e-02,
                     4.63339678e-02,
                     5.84869638e-02,
                     4.19255988e-02,
                     3.79695809e-02,
                     4.37996404e-02,
                     6.22997236e-05,
                     5.94581819e-02,
                     4.50271399e-02,
                     4.27128846e-02
                   ]
                 ])
               )

      assert Nx.all_close(value, Nx.tensor(22.758692))

      assert Nx.all_close(
               grad,
               Nx.tensor([
                 [
                   0.5097823,
                   0.50505996,
                   0.5028005,
                   0.50707513,
                   0.50597477,
                   0.5082999,
                   0.5039333,
                   0.5096293,
                   0.5104446,
                   0.5083808,
                   0.5018608,
                   0.5100916,
                   0.5110521,
                   0.5097509,
                   0.5101197,
                   0.5105851,
                   0.50539,
                   0.50952643,
                   0.5123473,
                   0.51079726,
                   0.50966597,
                   0.5063178,
                   0.513203,
                   0.5115814,
                   0.51461756,
                   0.51047987,
                   0.50949126,
                   0.5109482,
                   0.50001556,
                   0.51486015,
                   0.5112549,
                   0.5106766
                 ]
               ])
             )
    end
  end
end
