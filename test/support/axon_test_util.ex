defmodule AxonTestUtil do
  import Nx.Defn

  def test_compiler do
    use_exla? = System.get_env("USE_EXLA")
    if use_exla?, do: EXLA, else: Nx.Defn.Evaluator
  end

  def test_backend do
    cond do
      System.get_env("USE_TORCHX") -> Torchx.Backend
      System.get_env("USE_EXLA") -> EXLA.Backend
      true -> Nx.BinaryBackend
    end
  end

  def check_optimizer!(optimizer, loss, x0, num_steps) do
    check_optimizer_functions!(optimizer)
    check_optimizer_run!(optimizer, loss, x0, num_steps)
  end

  def assert_all_close(lhs, rhs, opts \\ [])

  def assert_all_close(lhs, rhs, opts) when is_tuple(lhs) and is_tuple(rhs) do
    lhs
    |> Tuple.to_list()
    |> Enum.zip_with(Tuple.to_list(rhs), &assert_all_close(&1, &2, opts))
  end

  def assert_all_close(lhs, rhs, opts) do
    res = Nx.all_close(lhs, rhs, opts) |> Nx.backend_transfer(Nx.BinaryBackend)

    unless Nx.to_number(res) == 1 do
      raise """
      expected

      #{inspect(Nx.backend_transfer(lhs, Nx.BinaryBackend))}

      to be within tolerance of

      #{inspect(Nx.backend_transfer(rhs, Nx.BinaryBackend))}
      """
    end
  end

  def assert_equal(lhs, rhs) when is_tuple(lhs) and is_tuple(rhs) do
    lhs
    |> Tuple.to_list()
    |> Enum.zip_with(Tuple.to_list(rhs), &assert_equal/2)
  end

  def assert_equal(%Nx.Tensor{} = lhs, %Nx.Tensor{} = rhs) do
    res = Nx.equal(lhs, rhs) |> Nx.all() |> Nx.backend_transfer(Nx.BinaryBackend)

    unless Nx.to_number(res) == 1 do
      raise """
      expected

      #{inspect(Nx.backend_transfer(lhs, Nx.BinaryBackend))}

      to be equal to

      #{inspect(Nx.backend_transfer(rhs, Nx.BinaryBackend))}
      """
    end
  end

  def assert_equal(lhs, rhs) when is_map(lhs) and is_map(rhs) do
    lhs
    |> Map.values()
    |> Enum.zip_with(Map.values(rhs), &assert_equal/2)
  end

  def assert_not_equal(lhs, rhs) when is_tuple(lhs) and is_tuple(rhs) do
    lhs
    |> Tuple.to_list()
    |> Enum.zip_with(Tuple.to_list(rhs), &assert_not_equal/2)
  end

  def assert_not_equal(%Nx.Tensor{} = lhs, %Nx.Tensor{} = rhs) do
    res = Nx.equal(lhs, rhs) |> Nx.all() |> Nx.backend_transfer(Nx.BinaryBackend)

    unless Nx.to_number(res) == 0 do
      raise """
      expected

      #{inspect(Nx.backend_transfer(lhs, Nx.BinaryBackend))}

      to be not equal to

      #{inspect(Nx.backend_transfer(rhs, Nx.BinaryBackend))}
      """
    end
  end

  def assert_not_equal(lhs, rhs) when is_map(lhs) and is_map(rhs) do
    rhs
    |> Map.values()
    |> Enum.zip_with(Map.values(rhs), &assert_not_equal/2)
  end

  def assert_greater_equal(lhs, rhs) do
    res = Nx.greater_equal(lhs, rhs) |> Nx.all() |> Nx.backend_transfer(Nx.BinaryBackend)

    unless Nx.to_number(res) == 1 do
      raise """
      expected

      #{inspect(Nx.backend_transfer(lhs, Nx.BinaryBackend))}

      to be greater than or equal to

      #{inspect(Nx.backend_transfer(rhs, Nx.BinaryBackend))}
      """
    end
  end

  def zeros(shape) do
    fun = Axon.Initializers.zeros()
    fun.(shape, {:f, 32})
  end

  defp check_optimizer_functions!(optimizer) do
    {init_fn, update_fn} = optimizer
    is_function(init_fn, 1) and is_function(update_fn, 3)
  end

  defp check_optimizer_run!(optimizer, loss, x0, num_steps) do
    {init_fn, update_fn} = optimizer
    opt_state = init_fn.(x0)
    state = {x0, opt_state}

    step_fn = fn state ->
      {params, opt_state} = state
      gradients = Nx.Defn.grad(params, loss)
      {updates, new_state} = update_fn.(gradients, opt_state, params)
      {Axon.Updates.apply_updates(updates, params), new_state}
    end

    {params, _} =
      for _ <- 1..num_steps, reduce: state do
        state ->
          apply(Nx.Defn.jit(step_fn), [state])
      end

    lhs = loss.(params)
    rhs = 1.0e-2

    res = Nx.less_equal(lhs, rhs) |> Nx.all() |> Nx.backend_transfer(Nx.BinaryBackend)

    # Some optimizers require 1-D or 2-D input, so this potentially
    # could be multi-dimensional
    unless Nx.to_number(res) == 1 do
      raise """
        expected

        #{inspect(lhs)}

        to be less than or equal to

        #{inspect(rhs)}
      """
    end
  end

  def get_test_data(
        train_samples,
        test_samples,
        batch_size,
        input_shape,
        num_classes,
        random_seed
      ) do
    key = Nx.Random.key(random_seed)
    num_samples = train_samples + test_samples
    full_input_shape = [num_classes | Tuple.to_list(input_shape)] |> List.to_tuple()

    {noise, key} = Nx.Random.uniform(key, 0.0, 1.0, shape: full_input_shape)
    templates = 2 |> Nx.multiply(num_classes) |> Nx.multiply(noise)
    {y, key} = Nx.Random.randint(key, 0, num_classes, shape: {num_samples}, type: :s64)

    {xs, ys, _} =
      Enum.reduce(0..(num_samples - 1), {[], [], key}, fn i, {x_acc, y_acc, key} ->
        {noise, key} = Nx.Random.normal(key, 0.0, 1.0, shape: input_shape)
        y_i = y[[i]]
        x_i = templates[[y_i]] |> Nx.add(noise)

        {[x_i | x_acc], [y_i | y_acc], key}
      end)

    {x_train, x_test} = Enum.split(xs, train_samples)
    {y_train, y_test} = Enum.split(ys, train_samples)

    train =
      x_train
      |> Stream.zip(y_train)
      |> Stream.chunk_every(batch_size)
      |> Stream.map(fn chunks ->
        {xs, ys} = Enum.unzip(chunks)
        {Nx.stack(xs), Nx.stack(ys)}
      end)

    test =
      x_test
      |> Stream.zip(y_test)
      |> Stream.chunk_every(batch_size)
      |> Stream.map(fn chunks ->
        {xs, ys} = Enum.unzip(chunks)
        {Nx.stack(xs), Nx.stack(ys)}
      end)

    {train, test}
  end

  defn one_hot(tensor, opts \\ []) do
    Nx.new_axis(tensor, -1) == Nx.iota({1, opts[:num_classes]})
  end
end

# The result of lazy container traversal
defmodule LazyWrapped do
  @derive {Nx.Container, containers: [:a, :b, :c]}
  defstruct [:a, :b, :c]
end

# The lazy container itself (which is not a container)
defmodule LazyOnly do
  defstruct [:a, :b, :c]

  defimpl Nx.LazyContainer do
    def traverse(%LazyOnly{a: a, b: b, c: c}, acc, fun) do
      {a, acc} = fun.(a |> Nx.tensor() |> Nx.to_template(), fn -> Nx.tensor(a) end, acc)
      {b, acc} = fun.(b |> Nx.tensor() |> Nx.to_template(), fn -> raise "don't call b" end, acc)
      {c, acc} = fun.(c |> Nx.tensor() |> Nx.to_template(), fn -> Nx.tensor(c) end, acc)
      {%LazyWrapped{a: a, b: b, c: c}, acc}
    end
  end
end
