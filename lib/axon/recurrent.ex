defmodule Axon.Recurrent do
  @moduledoc """
  Functional implementations of common recurrent neural network
  routines.

  Recurrent Neural Networks are commonly used for working with
  sequences of data where there is some level of dependence between
  outputs at different timesteps.

  This module contains 3 RNN Cell functions and methods to "unroll"
  cells over an entire sequence. Each cell function returns a tuple:

      {new_carry, output}

  Where `new_carry` is an updated carry state and `output` is the output
  for a singular timestep. In order to apply an RNN across multiple timesteps,
  you need to use either `static_unroll` or `dynamic_unroll` (coming soon).

  Unrolling an RNN is equivalent to a `map_reduce` or `scan` starting
  from an initial carry state and ending with a final carry state and
  an output sequence.

  All of the functions in this module are implemented as
  numerical functions and can be JIT or AOT compiled with
  any supported `Nx` compiler.
  """
  import Nx.Defn
  import Axon.Layers
  import Axon.Activations

  @doc """
  GRU Cell.
  """
  defn gru_cell(
         input,
         carry,
         input_kernel,
         hidden_kernel,
         bias,
         gate_fn \\ &sigmoid/1,
         activation_fn \\ &tanh/1
       ) do
    {hidden} = carry
    {wir, wiz, win} = input_kernel
    {whr, whz, whn} = hidden_kernel
    {br, bz, bin, bhn} = bias

    r = gate_fn.(dense(input, wir, br) + dense(hidden, whr, 0))
    z = gate_fn.(dense(input, wiz, bz) + dense(hidden, whz, 0))
    n = activation_fn.(dense(input, win, bin) + r * dense(hidden, whn, bhn))

    new_h = (1.0 - z) * n + z * hidden

    {{new_h}, new_h}
  end

  @doc """
  LSTM Cell.
  """
  defn lstm_cell(
         input,
         carry,
         input_kernel,
         hidden_kernel,
         bias,
         gate_fn \\ &sigmoid/1,
         activation_fn \\ &tanh/1
       ) do
    {cell, hidden} = carry
    {wii, wif, wig, wio} = input_kernel
    {whi, whf, whg, who} = hidden_kernel

    {bi, bf, bg, bo} = bias

    i = gate_fn.(dense(input, wii, bi) + dense(hidden, whi, 0))
    f = gate_fn.(dense(input, wif, bf) + dense(hidden, whf, 0))
    g = activation_fn.(dense(input, wig, bg) + dense(hidden, whg, 0))
    o = gate_fn.(dense(input, wio, bo) + dense(hidden, who, 0))

    new_c = f * cell + i * g
    new_h = o * activation_fn.(new_c)

    {{new_c, new_h}, new_h}
  end

  @doc """
  ConvLSTM Cell.
  """
  defn conv_lstm_cell(input, carry, input_kernel, hidden_kernel, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :same)

    {cell, hidden} = carry
    {ih} = input_kernel
    {hh} = hidden_kernel
    {bi} = bias

    gates =
      conv(input, ih, bi, strides: opts[:strides], padding: opts[:padding]) +
        conv(hidden, hh, 0, strides: opts[:strides], padding: opts[:padding])

    {i, g, f, o} = split_gates(gates)

    f = sigmoid(f + 1)
    new_c = f * cell + sigmoid(i) * tanh(g)
    new_h = sigmoid(o) * tanh(new_c)

    {{new_c, new_h}, new_h}
  end

  defnp split_gates(gates) do
    transform(gates, fn gates ->
      channels = elem(Nx.shape(gates), 1)
      split_every = div(channels, 4)

      split_dims =
        for i <- 0..3 do
          {i * split_every, split_every}
        end

      split_dims
      |> Enum.map(fn {start, len} -> Nx.slice_axis(gates, start, len, 1) end)
      |> List.to_tuple()
    end)
  end

  @doc """
  Dynamically unrolls an RNN.
  """
  defn dynamic_unroll(cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias) do
    time_steps = transform(Nx.shape(input_sequence), &elem(&1, 1))

    feature_dims = transform(Nx.rank(input_sequence), &List.duplicate(0, &1 - 2))

    initial_shape =
      transform({Nx.shape(input_sequence), Nx.shape(elem(input_kernel, 0))}, fn {shape, kernel} ->
        put_elem(shape, 2, elem(kernel, 1))
      end)

    init_sequence = Nx.broadcast(0.0, initial_shape)
    i = Nx.tensor(0)

    {_, carry, output, _, _, _, _} =
      while {i, carry, init_sequence, input_sequence, input_kernel, recurrent_kernel, bias},
            Nx.less(i, time_steps) do
        sequence = Nx.slice_axis(input_sequence, i, 1, 1)
        indices = transform({feature_dims, i}, fn {feature_dims, i} -> [0, i] ++ feature_dims end)
        {carry, output} = cell_fn.(sequence, carry, input_kernel, recurrent_kernel, bias)
        update_sequence = Nx.put_slice(init_sequence, indices, output)
        {i + 1, carry, update_sequence, input_sequence, input_kernel, recurrent_kernel, bias}
      end

    {carry, output}
  end

  @doc """
  Statically unrolls an RNN.
  """
  defn static_unroll(cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias) do
    transform(
      {cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias},
      fn {cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias} ->
        time_steps = elem(Nx.shape(input_sequence), 1)

        {carry, outputs} =
          for t <- 0..(time_steps - 1), reduce: {carry, []} do
            {carry, outputs} ->
              input = Nx.slice_axis(input_sequence, t, 1, 1)
              {carry, output} = cell_fn.(input, carry, input_kernel, recurrent_kernel, bias)
              {carry, [output | outputs]}
          end

        {carry, Nx.concatenate(Enum.reverse(outputs), axis: 1)}
      end
    )
  end
end
