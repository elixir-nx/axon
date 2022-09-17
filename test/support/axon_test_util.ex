defmodule AxonTestUtil do
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
end
