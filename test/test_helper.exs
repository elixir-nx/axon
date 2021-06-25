ExUnit.start()

defmodule AxonTestUtil do

  def check_optimizer!(optimizer, loss, x0, num_steps) do
    check_optimizer_functions!(optimizer, x0)
    check_optimizer_run!(optimizer, x0, num_steps, learning_rate)
  end

  defp check_optimizer_functions!(optimizer, x0) do
    assert {init_fn, update_fn} = optimizer
    assert is_function(init_fn, 1)
    assert is_function(update_fn, 3)
  end

  defp check_optimizer_run!(optimizer, loss, x0, num_steps, learning_rate) do
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
          step_fn.(state)
      end

    assert loss.(params) <= 1.0e-2
  end
end