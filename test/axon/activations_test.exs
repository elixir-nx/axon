defmodule Axon.ActivationsTest do
  use Axon.Case, async: true

  doctest Axon.Activations

  describe "celu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.534882128238678, 0.6075059175491333, 0.8809065222740173])
      expected = Nx.tensor([0.534882128238678, 0.6075059175491333, 0.8809065222740173])
      actual = Axon.Activations.celu(a)
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 1 and type {:f, 32} and alpha: 0.5" do
      a = Nx.tensor([0.5254644751548767, 0.7386103868484497, 0.7302365303039551])
      expected = Nx.tensor([0.5254644751548767, 0.7386103868484497, 0.7302365303039551])
      actual = Axon.Activations.celu(a, alpha: 0.5)
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.0006543696508742869, 0.9109137058258057, 0.5773505568504333])
      expected = Nx.tensor([1.0, 1.0, 1.0])

      actual =
        apply(
          jit(fn x ->
            grad(x, &Nx.sum(Axon.Activations.celu(&1)))
          end),
          [a]
        )

      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.8410307765007019, 0.7737964987754822],
          [0.3063606917858124, 0.47949355840682983]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])

      actual =
        apply(
          jit(fn x ->
            grad(x, &Nx.sum(Axon.Activations.celu(&1)))
          end),
          [a]
        )

      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.7361555695533752, 0.07279454171657562]],
          [[0.12938232719898224, 0.19182132184505463]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])

      actual =
        apply(
          jit(fn x ->
            grad(x, &Nx.sum(Axon.Activations.celu(&1)))
          end),
          [a]
        )

      assert_all_close(expected, actual)
    end
  end

  describe "elu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.05110076069831848, 0.05315212532877922, 0.2063606083393097])
      expected = Nx.tensor([0.05110076069831848, 0.05315212532877922, 0.2063606083393097])
      actual = Axon.Activations.elu(a)
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 1 and type {:f, 32} and alpha: 0.5" do
      a = Nx.tensor([0.5150681734085083, 0.8869504928588867, 0.6374541521072388])
      expected = Nx.tensor([0.5150681734085083, 0.8869504928588867, 0.6374541521072388])
      actual = Axon.Activations.elu(a)
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.2467288225889206, 0.047934457659721375, 0.6276589035987854])
      expected = Nx.tensor([1.0, 1.0, 1.0])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.elu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.35968613624572754, 0.2542852759361267],
          [0.19061528146266937, 0.5906847715377808]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.elu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.10630486905574799, 0.5689958333969116]],
          [[0.5406809449195862, 0.10776042938232422]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.elu(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "exp" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.4639144837856293, 0.3422740399837494, 0.6417020559310913])
      expected = Nx.tensor([1.5902869701385498, 1.4081461429595947, 1.8997116088867188])
      actual = Axon.Activations.exp(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.7983033061027527, 0.3828118145465851],
          [0.5582484006881714, 0.0010385476052761078]
        ])

      expected =
        Nx.tensor([
          [2.2217681407928467, 1.4664020538330078],
          [1.7476086616516113, 1.0010390281677246]
        ])

      actual = Axon.Activations.exp(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.1891545206308365, 0.23333953320980072]],
          [[0.3159049153327942, 0.26520010828971863]]
        ])

      expected =
        Nx.tensor([
          [[1.2082276344299316, 1.262810230255127]],
          [[1.37149977684021, 1.3036918640136719]]
        ])

      actual = Axon.Activations.exp(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.7080050706863403, 0.5748331546783447, 0.8570890426635742])
      expected = Nx.tensor([2.029937744140625, 1.7768340110778809, 2.3562915325164795])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.exp(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.690890371799469, 0.4090757966041565],
          [0.9405754804611206, 0.5836707949638367]
        ])

      expected =
        Nx.tensor([
          [1.9954915046691895, 1.5054258108139038],
          [2.561455011367798, 1.7926065921783447]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.exp(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.8088638186454773, 0.4728061258792877]],
          [[0.12738095223903656, 0.43700599670410156]]
        ])

      expected =
        Nx.tensor([
          [[2.2453553676605225, 1.6044902801513672]],
          [[1.1358495950698853, 1.548065423965454]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.exp(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "gelu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.33406558632850647, 0.5005938410758972, 0.8558046817779541])
      expected = Nx.tensor([0.21074023842811584, 0.34624648094177246, 0.6880216598510742])
      actual = Axon.Activations.gelu(a)
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.5424693822860718, 0.17958815395832062, 0.6342728137969971])
      expected = Nx.tensor([0.89305579662323, 0.641761302947998, 0.9439805150032043])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.gelu(&1))) end), [a])
      assert_all_close(expected, actual)
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

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.gelu(&1))) end), [a])
      assert_all_close(expected, actual)
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

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.gelu(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "hard_tanh" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.31174659729003906, 0.6287908554077148, 0.8463948369026184])
      expected = Nx.tensor([0.31174659729003906, 0.6287908554077148, 0.8463948369026184])
      actual = Axon.Activations.hard_tanh(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.8712798953056335, 0.26574787497520447],
          [0.5519879460334778, 0.24893896281719208]
        ])

      expected =
        Nx.tensor([
          [0.8712798953056335, 0.26574787497520447],
          [0.5519879460334778, 0.24893896281719208]
        ])

      actual = Axon.Activations.hard_tanh(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.20783449709415436, 0.046247102320194244]],
          [[0.12332560122013092, 0.7893155217170715]]
        ])

      expected =
        Nx.tensor([
          [[0.20783449709415436, 0.046247102320194244]],
          [[0.12332560122013092, 0.7893155217170715]]
        ])

      actual = Axon.Activations.hard_tanh(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.9485284090042114, 0.435951828956604, 0.5815699696540833])
      expected = Nx.tensor([1.0, 1.0, 1.0])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.hard_tanh(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.4597303569316864, 0.8040322661399841],
          [0.7624267935752869, 0.262974351644516]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.hard_tanh(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.22331906855106354, 0.967119038105011]],
          [[0.5433276295661926, 0.06428512930870056]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.hard_tanh(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "leaky_relu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.14366523921489716, 0.8074116110801697, 0.9015417098999023])
      expected = Nx.tensor([0.14366523921489716, 0.8074116110801697, 0.9015417098999023])
      actual = Axon.Activations.leaky_relu(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.7146409749984741, 0.004396744538098574],
          [0.7338429689407349, 0.7714938521385193]
        ])

      expected =
        Nx.tensor([
          [0.7146409749984741, 0.004396744538098574],
          [0.7338429689407349, 0.7714938521385193]
        ])

      actual = Axon.Activations.leaky_relu(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.5697717070579529, 0.2865411639213562]],
          [[0.6484665870666504, 0.46057072281837463]]
        ])

      expected =
        Nx.tensor([
          [[0.5697717070579529, 0.2865411639213562]],
          [[0.6484665870666504, 0.46057072281837463]]
        ])

      actual = Axon.Activations.leaky_relu(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 1 and type {:f, 32}, alpha: 0.5" do
      a = Nx.tensor([0.4840377867221832, 0.3402462601661682, 0.3884929120540619])
      expected = Nx.tensor([0.4840377867221832, 0.3402462601661682, 0.3884929120540619])
      actual = Axon.Activations.leaky_relu(a, alpha: 0.5)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}, alpha: 0.5" do
      a =
        Nx.tensor([
          [0.24476367235183716, 0.8360252380371094],
          [0.3812156319618225, 0.7486156821250916]
        ])

      expected =
        Nx.tensor([
          [0.24476367235183716, 0.8360252380371094],
          [0.3812156319618225, 0.7486156821250916]
        ])

      actual = Axon.Activations.leaky_relu(a, alpha: 0.5)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}, alpha: 0.5" do
      a =
        Nx.tensor([
          [[0.7986632585525513, 0.8708104491233826]],
          [[0.8820486664772034, 0.3485039472579956]]
        ])

      expected =
        Nx.tensor([
          [[0.7986632585525513, 0.8708104491233826]],
          [[0.8820486664772034, 0.3485039472579956]]
        ])

      actual = Axon.Activations.leaky_relu(a, alpha: 0.5)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.47335895895957947, 0.8662083745002747, 0.07783236354589462])
      expected = Nx.tensor([1.0, 1.0, 1.0])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.leaky_relu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.4009854197502136, 0.04092039540410042],
          [0.8223347067832947, 0.42723962664604187]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.leaky_relu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.5447808504104614, 0.6219799518585205]],
          [[0.22892318665981293, 0.552138090133667]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.leaky_relu(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "log_sigmoid" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.1184869334101677, 0.8524676561355591, 0.8292036652565002])
      expected = Nx.tensor([-0.6356576085090637, -0.35512682795524597, -0.3621376156806946])
      actual = Axon.Activations.log_sigmoid(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.9977004528045654, 0.6799364686012268],
          [0.5323657989501953, 0.5890879034996033]
        ])

      expected =
        Nx.tensor([
          [-0.31388065218925476, -0.4098881483078003],
          [-0.46198034286499023, -0.4413682520389557]
        ])

      actual = Axon.Activations.log_sigmoid(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.587926983833313, 0.8881919384002686]],
          [[0.6705363392829895, 0.6351630091667175]]
        ])

      expected =
        Nx.tensor([
          [[-0.4417826533317566, -0.34458136558532715]],
          [[-0.4130589962005615, -0.42516908049583435]]
        ])

      actual = Axon.Activations.log_sigmoid(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.007618676871061325, 0.2710806727409363, 0.7870540022850037])
      expected = Nx.tensor([0.4980953633785248, 0.43264180421829224, 0.3128015995025635])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.log_sigmoid(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.5251182317733765, 0.6292591691017151],
          [0.20837312936782837, 0.3905956745147705]
        ])

      expected =
        Nx.tensor([
          [0.3716561794281006, 0.347678542137146],
          [0.4480943977832794, 0.4035739302635193]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.log_sigmoid(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.7089985609054565, 0.9742392301559448]],
          [[0.8972120881080627, 0.5678815841674805]]
        ])

      expected =
        Nx.tensor([
          [[0.3298201560974121, 0.27403631806373596]],
          [[0.28962376713752747, 0.3617258071899414]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.log_sigmoid(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "log_softmax" do
    test "raises on bad axis" do
      assert_raise ArgumentError, ~r/log_softmax axis must be within rank of tensor/, fn ->
        Axon.Activations.log_softmax(Nx.iota({1, 3}), axis: 2)
      end
    end

    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.9053106307983398, 0.4941180944442749, 0.0404893234372139])
      expected = Nx.tensor([-0.7342825531959534, -1.145475149154663, -1.5991039276123047])
      actual = Axon.Activations.log_softmax(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.41777005791664124, 0.3814489245414734],
          [0.12367314100265503, 0.6880181431770325]
        ])

      expected =
        Nx.tensor([
          [-0.675151526927948, -0.7114726305007935],
          [-1.0146129131317139, -0.4502679705619812]
        ])

      actual = Axon.Activations.log_softmax(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.5805424451828003, 0.2838536202907562]],
          [[0.03359711915254593, 0.5723268985748291]]
        ])

      expected =
        Nx.tensor([
          [[-0.555765688419342, -0.8524545431137085]],
          [[-0.9983603954315186, -0.4596306085586548]]
        ])

      actual = Axon.Activations.log_softmax(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 1 and type {:f, 32}, axis: 0" do
      a = Nx.tensor([0.46805715560913086, 0.32387757301330566, 0.868580162525177])
      expected = Nx.tensor([-1.2114460468292236, -1.3556256294250488, -0.8109229803085327])
      actual = Axon.Activations.log_softmax(a, axis: 0)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}, axis: 0" do
      a =
        Nx.tensor([
          [0.38231733441352844, 0.5206525325775146],
          [0.7770483493804932, 0.4060186743736267]
        ])

      expected =
        Nx.tensor([
          [-0.909864068031311, -0.6374720335006714],
          [-0.5151330828666687, -0.7521058917045593]
        ])

      actual = Axon.Activations.log_softmax(a, axis: 0)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}, axis: 0" do
      a =
        Nx.tensor([
          [[0.5702528357505798, 0.5982637405395508]],
          [[0.8293957710266113, 0.6105955839157104]]
        ])

      expected =
        Nx.tensor([
          [[-0.8310896754264832, -0.6993321180343628]],
          [[-0.5719467401504517, -0.6870002746582031]]
        ])

      actual = Axon.Activations.log_softmax(a, axis: 0)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.06931844353675842, 0.772077202796936, 0.1932809054851532])
      expected = Nx.tensor([0.27733299136161804, -0.459293007850647, 0.1819601207971573])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.log_softmax(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.28714632987976074, 0.007128875702619553],
          [0.008051002398133278, 0.4765068590641022]
        ])

      expected =
        Nx.tensor([
          [-0.1391010284423828, 0.13910096883773804],
          [0.23003646731376648, -0.23003637790679932]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.log_softmax(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.36286887526512146, 0.755348265171051]],
          [[0.9639256000518799, 0.2673284113407135]]
        ])

      expected =
        Nx.tensor([
          [[0.19375887513160706, -0.19375884532928467]],
          [[-0.3348658084869385, 0.3348657786846161]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.log_softmax(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "relu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.5017861723899841, 0.8087382912635803, 0.2827244997024536])
      expected = Nx.tensor([0.5017861723899841, 0.8087382912635803, 0.2827244997024536])
      actual = Axon.Activations.relu(a)
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
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
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.2746083736419678, 0.23505471646785736, 0.5368936657905579])
      expected = Nx.tensor([1.0, 1.0, 1.0])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.7280408143997192, 0.4684365689754486],
          [0.5426321029663086, 0.7235316038131714]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.03652488812804222, 0.6321760416030884]],
          [[0.45222511887550354, 0.18244826793670654]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "relu6" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.842319667339325, 0.021900499239563942, 0.19451844692230225])
      expected = Nx.tensor([0.842319667339325, 0.021900499239563942, 0.19451844692230225])
      actual = Axon.Activations.relu6(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.01567154936492443, 0.26180967688560486],
          [0.6615678071975708, 0.409149169921875]
        ])

      expected =
        Nx.tensor([
          [0.01567154936492443, 0.26180967688560486],
          [0.6615678071975708, 0.409149169921875]
        ])

      actual = Axon.Activations.relu6(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.04743034392595291, 0.7952200770378113]],
          [[0.09659137576818466, 0.210462749004364]]
        ])

      expected =
        Nx.tensor([
          [[0.04743034392595291, 0.7952200770378113]],
          [[0.09659137576818466, 0.210462749004364]]
        ])

      actual = Axon.Activations.relu6(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.3808571994304657, 0.5188078880310059, 0.9164689183235168])
      expected = Nx.tensor([1.0, 1.0, 1.0])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu6(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.5499739646911621, 0.34733402729034424],
          [0.6983950734138489, 0.24491633474826813]
        ])

      expected = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu6(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.16589230298995972, 0.9246458411216736]],
          [[0.23529919981956482, 0.2500540614128113]]
        ])

      expected = Nx.tensor([[[1.0, 1.0]], [[1.0, 1.0]]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.relu6(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "selu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.5557087063789368, 0.09193431586027145, 0.9969830513000488])
      expected = Nx.tensor([0.5838837027549744, 0.09659548103809357, 1.0475311279296875])
      actual = Axon.Activations.selu(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.6687846183776855, 0.36514317989349365],
          [0.763460636138916, 0.8376092314720154]
        ])

      expected =
        Nx.tensor([
          [0.7026926875114441, 0.3836563229560852],
          [0.8021688461303711, 0.8800768852233887]
        ])

      actual = Axon.Activations.selu(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.9511207938194275, 0.9309434294700623]],
          [[0.02450866810977459, 0.42269641160964966]]
        ])

      expected =
        Nx.tensor([
          [[0.9993435740470886, 0.9781432151794434]],
          [[0.025751283392310143, 0.44412755966186523]]
        ])

      actual = Axon.Activations.selu(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.6316671967506409, 0.8289909958839417, 0.18112128973007202])
      expected = Nx.tensor([1.0507010221481323, 1.0507010221481323, 1.0507010221481323])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.selu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.4947720468044281, 0.15160438418388367],
          [0.44867998361587524, 0.808595597743988]
        ])

      expected =
        Nx.tensor([
          [1.0507010221481323, 1.0507010221481323],
          [1.0507010221481323, 1.0507010221481323]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.selu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.8145947456359863, 0.44749367237091064]],
          [[0.8953422904014587, 0.20643578469753265]]
        ])

      expected =
        Nx.tensor([
          [[1.0507010221481323, 1.0507010221481323]],
          [[1.0507010221481323, 1.0507010221481323]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.selu(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "sigmoid" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.06091778352856636, 0.5317422747612, 0.47350651025772095])
      expected = Nx.tensor([0.5152247548103333, 0.6298893690109253, 0.6162133812904358])
      actual = Axon.Activations.sigmoid(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.006493795197457075, 0.36615046858787537],
          [0.30910560488700867, 0.5146171450614929]
        ])

      expected =
        Nx.tensor([
          [0.5016234517097473, 0.5905284881591797],
          [0.5766669511795044, 0.6258881688117981]
        ])

      actual = Axon.Activations.sigmoid(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.9038936495780945, 0.007167538162320852]],
          [[0.8746536374092102, 0.1541392058134079]]
        ])

      expected =
        Nx.tensor([
          [[0.7117489576339722, 0.5017918348312378]],
          [[0.7057130932807922, 0.5384587049484253]]
        ])

      actual = Axon.Activations.sigmoid(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.9408628344535828, 0.2626379430294037, 0.3255162537097931])
      expected = Nx.tensor([0.20191897451877594, 0.2457379251718521, 0.24349266290664673])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.sigmoid(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.6852369904518127, 0.6258020401000977],
          [0.4455641210079193, 0.048214737325906754]
        ])

      expected =
        Nx.tensor([
          [0.22280584275722504, 0.22703653573989868],
          [0.2379913330078125, 0.24985475838184357]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.sigmoid(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.1957770735025406, 0.18241168558597565]],
          [[0.7026098370552063, 0.5788813829421997]]
        ])

      expected =
        Nx.tensor([
          [[0.24761967360973358, 0.2479318529367447]],
          [[0.22151799499988556, 0.2301725596189499]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.sigmoid(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "silu" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.042964693158864975, 0.21260398626327515, 0.040325723588466644])
      expected = Nx.tensor([0.021943766623735428, 0.11755973100662231, 0.020569348707795143])
      actual = Axon.Activations.silu(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.9929577708244324, 0.36843422055244446],
          [0.95541912317276, 0.19103989005088806]
        ])

      expected =
        Nx.tensor([
          [0.724533200263977, 0.21777431666851044],
          [0.690007209777832, 0.10461635142564774]
        ])

      actual = Axon.Activations.silu(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.6830660700798035, 0.1384754627943039]],
          [[0.5222190022468567, 0.5275246500968933]]
        ])

      expected =
        Nx.tensor([
          [[0.4538446068763733, 0.07402394711971283]],
          [[0.3277793526649475, 0.33176320791244507]]
        ])

      actual = Axon.Activations.silu(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.7756811380386353, 0.3616112768650055, 0.2904549837112427])
      expected = Nx.tensor([0.8521932363510132, 0.6769410967826843, 0.6432110667228699])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.silu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.9843150973320007, 0.673032283782959],
          [0.4259312152862549, 0.35117384791374207]
        ])

      expected =
        Nx.tensor([
          [0.9228900671005249, 0.8127371072769165],
          [0.7066973447799683, 0.6720436215400696]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.silu(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.016957957297563553, 0.2903658449649811]],
          [[0.09186866134405136, 0.6251020431518555]]
        ])

      expected =
        Nx.tensor([
          [[0.508478581905365, 0.6431683301925659]],
          [[0.545869767665863, 0.793329119682312]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.silu(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "softmax" do
    test "raises on bad axis" do
      assert_raise ArgumentError, "given axis (2) invalid for shape with rank 2", fn ->
        Axon.Activations.softmax(Nx.iota({1, 3}), axis: 2)
      end
    end

    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.7815847992897034, 0.22426098585128784, 0.6045002937316895])
      expected = Nx.tensor([0.4148605167865753, 0.23760710656642914, 0.34753239154815674])
      actual = Axon.Activations.softmax(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.15358823537826538, 0.15878894925117493],
          [0.438334584236145, 0.5666020512580872]
        ])

      expected =
        Nx.tensor([
          [0.4986998438835144, 0.5013002157211304],
          [0.46797701716423035, 0.5320229530334473]
        ])

      actual = Axon.Activations.softmax(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.6258360147476196, 0.5342917442321777]],
          [[0.36701539158821106, 0.7174903154373169]]
        ])

      expected =
        Nx.tensor([
          [[0.5228701233863831, 0.47712990641593933]],
          [[0.41326725482940674, 0.5867327451705933]]
        ])

      actual = Axon.Activations.softmax(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 1 and type {:f, 32}, axis: 0" do
      a = Nx.tensor([0.14293035864830017, 0.3950499892234802, 0.1608438789844513])
      expected = Nx.tensor([0.3025878965854645, 0.389354944229126, 0.30805712938308716])
      actual = Axon.Activations.softmax(a, axis: 0)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}, axis: 0" do
      a =
        Nx.tensor([
          [0.9145559072494507, 0.9167066216468811],
          [0.5215153098106384, 0.6520692706108093]
        ])

      expected =
        Nx.tensor([
          [0.5970144271850586, 0.5657759308815002],
          [0.402985543012619, 0.43422406911849976]
        ])

      actual = Axon.Activations.softmax(a, axis: 0)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}, axis: 0" do
      a =
        Nx.tensor([
          [[0.7535658478736877, 0.8726069331169128]],
          [[0.9226275086402893, 0.6262556910514832]]
        ])

      expected =
        Nx.tensor([
          [[0.45783495903015137, 0.561278223991394]],
          [[0.5421649813652039, 0.43872177600860596]]
        ])

      actual = Axon.Activations.softmax(a, axis: 0)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.21630559861660004, 0.1898617446422577, 0.9072079062461853])
      expected = Nx.tensor([0.0, 0.0, 0.0])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softmax(&1))) end), [a])
      assert_all_close(expected, actual, atol: 1.0e-5)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.3524329662322998, 0.7788003087043762],
          [0.4497735798358917, 0.6243307590484619]
        ])

      expected = Nx.tensor([[0.0, 0.0], [0.0, 0.0]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softmax(&1))) end), [a])
      assert_all_close(expected, actual, atol: 1.0e-7)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.06885724514722824, 0.08121756464242935]],
          [[0.04193086922168732, 0.8268604278564453]]
        ])

      expected = Nx.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softmax(&1))) end), [a])
      assert_all_close(expected, actual, atol: 1.0e-7)
    end
  end

  describe "softplus" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.7309077382087708, 0.13401681184768677, 0.12274937331676483])
      expected = Nx.tensor([1.12394380569458, 0.7623989582061768, 0.7564041018486023])
      actual = Axon.Activations.softplus(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.8193188309669495, 0.9130507707595825],
          [0.9200287461280823, 0.3547053635120392]
        ])

      expected =
        Nx.tensor([
          [1.1844699382781982, 1.250449776649475],
          [1.2554343938827515, 0.8861449956893921]
        ])

      actual = Axon.Activations.softplus(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.2834765613079071, 0.4217550754547119]],
          [[0.37744903564453125, 0.5455183982849121]]
        ])

      expected =
        Nx.tensor([
          [[0.8448967933654785, 0.926096498966217]],
          [[0.8995754718780518, 1.002652883529663]]
        ])

      actual = Axon.Activations.softplus(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.1869126409292221, 0.5769602656364441, 0.19408872723579407])
      expected = Nx.tensor([0.5465925931930542, 0.6403676867485046, 0.5483704805374146])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softplus(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.11179731041193008, 0.17767632007598877],
          [0.929054856300354, 0.13991422951221466]
        ])

      expected =
        Nx.tensor([
          [0.5279202461242676, 0.5443025827407837],
          [0.7168834209442139, 0.5349215865135193]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softplus(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.5481308102607727, 0.5958307981491089]],
          [[0.33531877398490906, 0.9774336814880371]]
        ])

      expected =
        Nx.tensor([
          [[0.6337018609046936, 0.6447018384933472]],
          [[0.5830529928207397, 0.7265986204147339]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softplus(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "softsign" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.6500415802001953, 0.7388723492622375, 0.5124310851097107])
      expected = Nx.tensor([0.39395466446876526, 0.4249146580696106, 0.33881282806396484])
      actual = Axon.Activations.softsign(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.584363579750061, 0.21967218816280365],
          [0.41985762119293213, 0.10589564591646194]
        ])

      expected =
        Nx.tensor([
          [0.3688317537307739, 0.18010756373405457],
          [0.29570403695106506, 0.09575554728507996]
        ])

      actual = Axon.Activations.softsign(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.9445214867591858, 0.16833151876926422]],
          [[0.753282368183136, 0.48777902126312256]]
        ])

      expected =
        Nx.tensor([
          [[0.4857346713542938, 0.1440785527229309]],
          [[0.4296412169933319, 0.3278571665287018]]
        ])

      actual = Axon.Activations.softsign(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.8105446696281433, 0.7689074873924255, 0.37004250288009644])
      expected = Nx.tensor([0.3050573468208313, 0.31958746910095215, 0.5327603816986084])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softsign(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.97364342212677, 0.5319579839706421],
          [0.10144895315170288, 0.6431559920310974]
        ])

      expected =
        Nx.tensor([
          [0.25672173500061035, 0.4260948896408081],
          [0.8242732882499695, 0.3703756332397461]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softsign(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.512872040271759, 0.010262660682201385]],
          [[0.0559421144425869, 0.8854929804801941]]
        ])

      expected =
        Nx.tensor([
          [[0.4369136691093445, 0.9797863960266113]],
          [[0.8968499898910522, 0.281287282705307]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.softsign(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end

  describe "tanh" do
    test "forward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.4583776295185089, 0.9545223712921143, 0.6616701483726501])
      expected = Nx.tensor([0.42876097559928894, 0.7418236136436462, 0.5794737935066223])
      actual = Axon.Activations.tanh(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.6983601450920105, 0.5971042513847351],
          [0.5811852216720581, 0.45667150616645813]
        ])

      expected =
        Nx.tensor([
          [0.6033258438110352, 0.5349857807159424],
          [0.5235263705253601, 0.42736750841140747]
        ])

      actual = Axon.Activations.tanh(a)
      assert_all_close(expected, actual)
    end

    test "forward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.2106747329235077, 0.6333889365196228]],
          [[0.5343267321586609, 0.06356880813837051]]
        ])

      expected =
        Nx.tensor([
          [[0.20761224627494812, 0.5603813529014587]],
          [[0.48868152499198914, 0.06348331272602081]]
        ])

      actual = Axon.Activations.tanh(a)
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 1 and type {:f, 32}" do
      a = Nx.tensor([0.006474076770246029, 0.0057099852710962296, 0.9349150061607361])
      expected = Nx.tensor([0.9999580979347229, 0.9999673962593079, 0.4628909230232239])
      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.tanh(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 2 and type {:f, 32}" do
      a =
        Nx.tensor([
          [0.09129361808300018, 0.23762618005275726],
          [0.018757252022624016, 0.28590673208236694]
        ])

      expected =
        Nx.tensor([
          [0.9917115569114685, 0.9455933570861816],
          [0.9996482133865356, 0.9225140810012817]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.tanh(&1))) end), [a])
      assert_all_close(expected, actual)
    end

    test "backward matches jax for rank 3 and type {:f, 32}" do
      a =
        Nx.tensor([
          [[0.5739016532897949, 0.3762475252151489]],
          [[0.5405635237693787, 0.9079052209854126]]
        ])

      expected =
        Nx.tensor([
          [[0.7314491271972656, 0.8707997798919678]],
          [[0.7565422058105469, 0.48141953349113464]]
        ])

      actual = apply(jit(fn x -> grad(x, &Nx.sum(Axon.Activations.tanh(&1))) end), [a])
      assert_all_close(expected, actual)
    end
  end
end
