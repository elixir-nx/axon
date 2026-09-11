# Bidirectional LSTM on IMDB
#
# An Axon port of https://keras.io/examples/nlp/bidirectional_lstm_imdb/,
# matching that example layer for layer:
#
#     Embedding(20000, 128)
#     Bidirectional(LSTM(64, return_sequences=True))
#     Bidirectional(LSTM(64))
#     Dense(1, activation="sigmoid")
#
# for 2,757,761 trainable parameters, trained with Adam for 2 epochs at
# batch size 32 against binary cross-entropy.
#
# The dataset is Keras' pre-tokenized IMDB dump. `imdb.npz` stores the
# reviews as pickled Python object arrays, which Nx cannot read, so it
# must be converted once into flat little-endian binaries under
# `IMDB_DIR` (default `/tmp/imdb`):
#
#     x_train.bin, x_val.bin   int32, 25000 x 200
#     y_train.bin, y_val.bin   float32, 25000
#
# i.e. the result of `keras.datasets.imdb.load_data(num_words=20000)`
# followed by `keras.utils.pad_sequences(..., maxlen=200)` (labels cast
# to float32), each array dumped with NumPy's `.tofile()`:
#
#     IMDB_DIR=/tmp/imdb elixir examples/text/bidirectional_lstm_imdb.exs
#
# Keras initializes its LSTM differently than Axon does, in ways that
# `Axon.lstm/3` cannot currently express. `KerasLSTMInit` below rewrites
# the initialized parameters to match it before training starts.
#
# Both this example and the Keras one shuffle the training set, so the
# numbers move by a few thousandths between runs. A representative run:
#
#             accuracy   loss   val_accuracy   val_loss
#     epoch 1   0.8237  0.3914       0.8701      0.3093
#     epoch 2   0.9201  0.3012       0.8644      0.3436
#
# against Keras' published 0.9151 / 0.2263 / 0.8428 / 0.3650.

# Axon currently tracks Nx main (see mix.exs), so EXLA must come from
# the same checkout — a hex EXLA would pull hex Nx and diverge.
Mix.install([
  {:axon, path: Path.expand("../..", __DIR__)},
  {:nx, github: "elixir-nx/nx", sparse: "nx", override: true},
  {:exla, github: "elixir-nx/nx", sparse: "exla", override: true}
])

Nx.global_default_backend(EXLA.Backend)

defmodule KerasLSTMInit do
  @moduledoc """
  Rewrites Axon's LSTM parameters to match `keras.layers.LSTM`'s
  initialization.

  Keras keeps the input and recurrent kernels as single
  `{input_dim, 4 * units}` and `{units, 4 * units}` matrices and slices
  them per gate, so its initializer always sees the full width. Axon
  stores one tensor per gate and initializes each on its own, which
  changes both the glorot fan-in/fan-out and — more importantly — means
  the four recurrent blocks are never orthogonal to one another. Here
  the wide matrix is drawn once and then sliced, exactly as Keras does.

  Three differences are corrected:

    * the recurrent kernel is orthogonal rather than glorot uniform;
    * the input kernel's glorot limit is computed over the full
      `4 * units` width rather than per gate;
    * `unit_forget_bias` — the forget gate's bias starts at one rather
      than zero, so the gate begins open and gradients can travel
      through time from the first step.

  Keras' gate order within those matrices is input, forget, cell,
  output, which lines up with Axon's `i`, `f`, `g`, `o` names.
  """

  @gates ~w(i f g o)

  def apply_to(%Axon.ModelState{data: data} = model_state, key) do
    {updates, _key} = walk(data, key)
    Axon.ModelState.update(model_state, updates)
  end

  defp walk(%{"input_kernel" => input_kernel, "hidden_kernel" => hidden_kernel} = layer, key) do
    units = Nx.axis_size(hidden_kernel["whi"], 1)
    input_dim = Nx.axis_size(input_kernel["wii"], 0)

    {input_kernel, key} =
      draw(Axon.Initializers.glorot_uniform(), {input_dim, 4 * units}, units, "wi", key)

    {hidden_kernel, key} =
      draw(Axon.Initializers.orthogonal(), {units, 4 * units}, units, "wh", key)

    updates = %{"input_kernel" => input_kernel, "hidden_kernel" => hidden_kernel}

    updates =
      if Map.has_key?(layer, "bias") do
        Map.put(updates, "bias", %{"bf" => Nx.broadcast(1.0, {units})})
      else
        updates
      end

    {updates, key}
  end

  defp walk(map, key) when is_map(map) and not is_struct(map) do
    Enum.reduce(map, {%{}, key}, fn {name, value}, {acc, key} ->
      {updates, key} = walk(value, key)

      if map_size(updates) == 0 do
        {acc, key}
      else
        {Map.put(acc, name, updates), key}
      end
    end)
  end

  defp walk(_other, key), do: {%{}, key}

  defp draw(initializer, shape, units, prefix, key) do
    keys = Nx.Random.split(key)
    matrix = initializer.(shape, {:f, 32}, keys[0])

    gates =
      @gates
      |> Enum.with_index()
      |> Map.new(fn {gate, index} ->
        {"#{prefix}#{gate}", Nx.slice_along_axis(matrix, index * units, units, axis: 1)}
      end)

    {gates, keys[1]}
  end
end

data_dir = System.get_env("IMDB_DIR", "/tmp/imdb")
epochs = String.to_integer(System.get_env("EPOCHS", "2"))
batch_size = String.to_integer(System.get_env("BATCH_SIZE", "32"))

# `nil` means "all of it" — the env vars exist to smoke-test the script
# on a slice before committing to a full run.
limit = fn name -> System.get_env(name) && String.to_integer(System.get_env(name)) end
train_limit = limit.("TRAIN_LIMIT")
val_limit = limit.("VAL_LIMIT")

max_features = 20_000
maxlen = 200

read = fn name, type, cols ->
  tensor =
    data_dir
    |> Path.join(name)
    |> File.read!()
    |> Nx.from_binary(type)

  Nx.reshape(tensor, {:auto, cols})
end

take = fn tensor, nil -> tensor
          tensor, n -> tensor[0..(n - 1)//1] end

x_train = read.("x_train.bin", :s32, maxlen) |> take.(train_limit)
y_train = read.("y_train.bin", :f32, 1) |> take.(train_limit)
x_val = read.("x_val.bin", :s32, maxlen) |> take.(val_limit)
y_val = read.("y_val.bin", :f32, 1) |> take.(val_limit)

IO.puts("#{Nx.axis_size(x_train, 0)} training sequences")
IO.puts("#{Nx.axis_size(x_val, 0)} validation sequences")

# Keras reshuffles the training set every epoch. `Stream.resource/3`
# re-runs its start function on each enumeration, and Axon.Loop
# enumerates the stream once per epoch, so the permutation is fresh
# each time.
#
# Every batch must have the same shape: the loop compiles against the
# first one and EXLA will not recompile for a ragged tail, so short
# final batches are discarded. 25000 training samples at batch size 32
# leaves a remainder of 8, i.e. 781 of 782 steps per epoch. Validation
# uses a batch size that divides 25000 exactly, so every validation
# sample is scored and the per-batch metric average is the true mean.
batched = fn x, y, shuffle?, size ->
  n = Nx.axis_size(x, 0)

  Stream.resource(
    fn ->
      order = if shuffle?, do: Enum.shuffle(0..(n - 1)), else: Enum.to_list(0..(n - 1))
      {Enum.chunk_every(order, size, size, :discard), 0}
    end,
    fn
      {[], _} = state ->
        {:halt, state}

      {[chunk | rest], i} ->
        idx = Nx.tensor(chunk)
        {[{Nx.take(x, idx), Nx.take(y, idx)}], {rest, i + 1}}
    end,
    fn _ -> :ok end
  )
end

val_batch_size = String.to_integer(System.get_env("VAL_BATCH_SIZE", "500"))

train_data = batched.(x_train, y_train, true, batch_size)
val_data = batched.(x_val, y_val, false, val_batch_size)

merge = fn forward, backward -> Nx.concatenate([forward, backward], axis: -1) end

# Keras' LSTM starts from a zero state; Axon's default is a random draw.
lstm = fn x -> Axon.lstm(x, 64, recurrent_initializer: :zeros) end

model =
  Axon.input("sequence", shape: {nil, maxlen})
  |> Axon.embedding(max_features, 128,
    kernel_initializer: Axon.Initializers.uniform(scale: 0.05)
  )
  |> Axon.bidirectional(lstm, merge, axis: 1)
  # return_sequences=True — keep the per-step outputs, drop the state.
  |> Axon.nx(&elem(&1, 0))
  |> Axon.bidirectional(lstm, merge, axis: 1)
  # return_sequences=False — keep each direction's final hidden state.
  |> Axon.nx(fn {_sequence, {_cell, hidden}} -> hidden end)
  |> Axon.dense(1)
  |> Axon.sigmoid()

{init_fn, _predict_fn} = Axon.build(model)

initial_state =
  Nx.template({batch_size, maxlen}, :s32)
  |> init_fn.(Axon.ModelState.empty())
  |> KerasLSTMInit.apply_to(Nx.Random.key(System.system_time()))

trainable_count =
  initial_state
  |> Axon.ModelState.trainable_parameters()
  |> then(&Nx.Defn.Composite.flatten_list([&1]))
  |> Enum.map(&Nx.size/1)
  |> Enum.sum()

IO.puts("Trainable params: #{trainable_count}")

# Keras' Adam defaults: learning_rate=0.001, beta_1=0.9, beta_2=0.999,
# epsilon=1e-7.
optimizer = Polaris.Optimizers.adam(learning_rate: 1.0e-3, eps: 1.0e-7)

# `validate/3` merges the validation metrics into the loop state at
# `:epoch_completed`, but the trainer's own logger only fires during
# iterations, so nothing prints them. Register a reporter after the
# validation handler — handlers run in registration order — to print
# the epoch summary Keras shows at the end of each epoch.
# A bare map pattern rather than %Axon.Loop.State{}: struct patterns
# expand while the script is being compiled, before `Mix.install`
# has fetched Axon, so naming the struct here breaks the script.
report = fn %{epoch: epoch, metrics: metrics} = state ->
  summary =
    metrics
    |> Enum.sort()
    |> Enum.map_join(" - ", fn {name, value} ->
      "#{name}: #{:io_lib.format(~c"~.4f", [Nx.to_number(value)])}"
    end)

  IO.puts("\nEpoch #{Nx.to_number(epoch) + 1}/#{epochs} - #{summary}")
  {:continue, state}
end

model
|> Axon.Loop.trainer(:binary_cross_entropy, optimizer, log: 100)
|> Axon.Loop.metric(:accuracy, "accuracy")
|> Axon.Loop.validate(model, val_data)
|> Axon.Loop.handle_event(:epoch_completed, report)
|> Axon.Loop.run(train_data, initial_state,
  epochs: epochs,
  compiler: EXLA
)
