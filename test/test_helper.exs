ExUnit.start()

defmodule AxonTestUtil do
  def test_compiler do
    use_exla? = System.get_env("USE_EXLA")
    if use_exla?, do: EXLA, else: Nx.Defn.Evaluator
  end

  def check_optimizer!(optimizer, loss, x0, num_steps) do
    check_optimizer_functions!(optimizer)
    check_optimizer_run!(optimizer, loss, x0, num_steps)
  end

  def assert_all_close(lhs, rhs) do
    unless Nx.all_close(lhs, rhs) == Nx.tensor(1, type: {:u, 8}) do
      raise """
      expected

      #{inspect(lhs)}

      to be within tolerance of

      #{inspect(rhs)}
      """
    end
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
          Nx.Defn.jit(step_fn, [state])
      end

    lhs = loss.(params)
    rhs = 1.0e-2

    # Some optimizers require 1-D or 2-D input, so this potentially
    # could be multi-dimensional
    unless Nx.all(Nx.less_equal(lhs, rhs)) == Nx.tensor(1, type: {:u, 8}) do
      raise """
        expected

        #{inspect(lhs)}

        to be less than or equal to

        #{inspect(rhs)}
      """
    end
  end
end
