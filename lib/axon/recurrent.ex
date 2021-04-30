defmodule Axon.Recurrent do
  @moduledoc """
  Implementation of routines for creating Recurrent Neural Networks.
  """
  import Nx.Defn
  import Axon.Layers

  @doc """
  GRU Cell.
  """
  defn gru_cell(
         input,
         carry,
         input_kernel,
         recurrent_kernel,
         bias,
         gate_fn \\ &Axon.Activations.sigmoid/1,
         activation_fn \\ &Axon.Activations.tanh/1
       ) do
    {hidden} = carry
    {wir, wiz, win} = input_kernel
    {hir, hiz, hin} = recurrent_kernel

    r = gate_fn.(dense(input, wir, bias) + dense(hidden, hir, 0))
    z = gate_fn.(dense(input, wiz, bias) + dense(hidden, hiz, 0))
    n = activation_fn.(dense(input, win, bias) + r * dense(hidden, hin, bias))

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
         recurrent_kernel,
         bias,
         gate_fn \\ &Axon.Activations.sigmoid/1,
         activation_fn \\ &Axon.Activations.tanh/1
       ) do
    {cell, hidden} = carry
    {wii, wif, wig, wio} = input_kernel
    {whi, whf, whg, who} = recurrent_kernel

    i = gate_fn.(dense(input, wii, bias) + dense(hidden, whi, 0))
    f = gate_fn.(dense(input, wif, bias) + dense(hidden, whf, 0))
    g = activation_fn.(dense(input, wig, bias) + dense(hidden, whg, 0))
    o = gate_fn.(dense(input, wio, bias) + dense(hidden, who, 0))

    new_c = f * cell + i * g
    new_h = o * activation_fn.(new_c)

    {{new_c, new_h}, new_h}
  end

  @doc """
  ConvLSTM Cell.
  """
  defn conv_lstm_cell(input, carry, input_kernel, recurrent_kernel, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :same)

    {cell, hidden} = carry
    {ih} = input_kernel
    {hh} = recurrent_kernel

    gates =
      conv(input, ih, bias, strides: opts[:strides], padding: opts[:padding]) +
        conv(hidden, hh, 0, strides: opts[:strides], padding: opts[:padding])

    {i, g, f, o} = split_gates(gates)

    f = Axon.Activations.sigmoid(f + 1)
    new_c = f * cell + Axon.Activations.sigmoid(i) * Axon.Activations.tanh(g)
    new_h = Axon.Activations.sigmoid(o) * Axon.Activations.tanh(new_c)

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

        # TODO: This should be a stack along the time axis
        {carry, Nx.concatenate(Enum.reverse(outputs), axis: 1)}
      end
    )
  end
end
