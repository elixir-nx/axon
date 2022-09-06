defmodule Axon.Recurrent do
  @moduledoc false

  import Nx.Defn
  import Axon.Layers

  @doc """
  GRU Cell.

  When combined with `Axon.Recurrent.*_unroll`, implements a
  GRU-based RNN. More memory efficient than traditional LSTM.

  ## References

  * [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555v1.pdf)
  """
  @deprecated "Use Axon.Layers.gru_cell/7 instead"
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

  When combined with `Axon.Recurrent.*_unroll`, implements a
  LSTM-based RNN. More memory efficient than traditional LSTM.

  ## References

  * [Long Short-Term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf)
  """
  @deprecated "Use Axon.Layers.lstm_cell/7 instead"
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

  When combined with `Axon.Recurrent.*_unroll`, implements a
  ConvLSTM-based RNN. More memory efficient than traditional LSTM.

  ## Options

    * `:strides` - convolution strides. Defaults to `1`.

    * `:padding` - convolution padding. Defaults to `:same`.

  ## References

    * [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)
  """
  @deprecated "Use Axon.Layers.conv_lstm_cell/6 instead"
  defn conv_lstm_cell(input, carry, input_kernel, hidden_kernel, bias, opts \\ []) do
    opts = keyword!(opts, strides: 1, padding: :same)

    {ih} = input_kernel
    {hh} = hidden_kernel
    {bi} = bias

    {{cell, hidden}, input} = rank_down({carry, input})

    gates =
      Nx.add(
        conv(input, ih, bi, strides: opts[:strides], padding: opts[:padding]),
        conv(hidden, hh, 0, strides: opts[:strides], padding: opts[:padding])
      )

    {i, g, f, o} = split_gates(gates)

    f = sigmoid(f + 1)
    new_c = f * cell + sigmoid(i) * tanh(g)
    new_h = sigmoid(o) * tanh(new_c)

    rank_up({{new_c, new_h}, new_h})
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
      |> Enum.map(fn {start, len} -> Nx.slice_along_axis(gates, start, len, axis: 1) end)
      |> List.to_tuple()
    end)
  end

  defnp rank_down(rnn_data) do
    transform(rnn_data, fn {{cell, hidden}, input} ->
      [cell, hidden, input] =
        for tensor <- [cell, hidden, input] do
          Nx.squeeze(tensor, axes: [1])
        end

      {{cell, hidden}, input}
    end)
  end

  defnp rank_up(rnn_data) do
    transform(rnn_data, fn {{cell, hidden}, input} ->
      [cell, hidden, input] =
        for tensor <- [cell, hidden, input] do
          new_shape =
            Nx.shape(tensor)
            |> Tuple.insert_at(1, 1)

          Nx.reshape(tensor, new_shape)
        end

      {{cell, hidden}, input}
    end)
  end

  @doc """
  Dynamically unrolls an RNN.

  Unrolls implement a `scan` operation which applies a
  transformation on the leading axis of `input_sequence` carrying
  some state. In this instance `cell_fn` is an RNN cell function
  such as `lstm_cell` or `gru_cell`.

  This function will make use of an `defn` while-loop such and thus
  may be more efficient for long sequences.
  """
  @deprecated "Use Axon.Layers.dynamic_unroll/6 instead"
  defn dynamic_unroll(cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias) do
    time_steps = transform(Nx.shape(input_sequence), &elem(&1, 1))

    feature_dims = transform(Nx.rank(input_sequence), &List.duplicate(0, &1 - 2))

    initial_shape =
      transform({cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias}, fn
        {cell_fn, inp, carry, inp_kernel, hid_kernel, bias} ->
          seq = Nx.slice_along_axis(inp, 0, 1, axis: 1)
          {_, seq} = cell_fn.(seq, carry, inp_kernel, hid_kernel, bias)
          put_elem(Nx.shape(seq), 1, elem(Nx.shape(inp), 1))
      end)

    init_sequence = Nx.broadcast(0.0, initial_shape)
    i = Nx.tensor(0)

    {_, carry, output, _, _, _, _} =
      while {i, carry, init_sequence, input_sequence, input_kernel, recurrent_kernel, bias},
            Nx.less(i, time_steps) do
        sequence = Nx.slice_along_axis(input_sequence, i, 1, axis: 1)
        indices = transform({feature_dims, i}, fn {feature_dims, i} -> [0, i] ++ feature_dims end)
        {carry, output} = cell_fn.(sequence, carry, input_kernel, recurrent_kernel, bias)
        update_sequence = Nx.put_slice(init_sequence, indices, output)
        {i + 1, carry, update_sequence, input_sequence, input_kernel, recurrent_kernel, bias}
      end

    {carry, output}
  end

  @doc """
  Statically unrolls an RNN.

  Unrolls implement a `scan` operation which applies a
  transformation on the leading axis of `input_sequence` carrying
  some state. In this instance `cell_fn` is an RNN cell function
  such as `lstm_cell` or `gru_cell`.

  This function inlines the unrolling of the sequence such that
  the entire operation appears as a part of the compilation graph.
  This makes it suitable for shorter sequences.
  """
  @deprecated "Use Axon.Layers.static_unroll/6 instead"
  defn static_unroll(cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias) do
    transform(
      {cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias},
      fn {cell_fn, input_sequence, carry, input_kernel, recurrent_kernel, bias} ->
        time_steps = elem(Nx.shape(input_sequence), 1)

        {carry, outputs} =
          for t <- 0..(time_steps - 1), reduce: {carry, []} do
            {carry, outputs} ->
              input = Nx.slice_along_axis(input_sequence, t, 1, axis: 1)
              {carry, output} = cell_fn.(input, carry, input_kernel, recurrent_kernel, bias)
              {carry, [output | outputs]}
          end

        {carry, Nx.concatenate(Enum.reverse(outputs), axis: 1)}
      end
    )
  end
end
