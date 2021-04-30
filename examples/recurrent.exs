timesteps =
  for t <- 0..9 do
    Nx.cos(Nx.multiply(Nx.multiply(t, 3.14 / 2), Nx.random_uniform({}, 0.0, 0.1)))
  end

sequence = Nx.stack(timesteps) |> Nx.reshape({1, 10, 1})

carry = {Nx.random_uniform({1, 32}, 0.1, 0.0)}
input_kernel = {Nx.random_uniform({1, 1}), Nx.random_uniform({1, 1}), Nx.random_uniform({1, 1})}
recurrent_kernel = {Nx.random_uniform({32, 1}), Nx.random_uniform({32, 1}), Nx.random_uniform({32, 1})}
bias = Nx.tensor(0.0)

fun =
  fn seq, carry, input_kernel, recurrent_kernel, bias ->
    Axon.Recurrent.static_unroll(&Axon.Recurrent.gru_cell/5, seq, carry, input_kernel, recurrent_kernel, bias)
  end

Nx.Defn.jit(fun, [sequence, carry, input_kernel, recurrent_kernel, bias], compiler: EXLA) |> IO.inspect