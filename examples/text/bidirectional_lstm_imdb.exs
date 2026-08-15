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
# reviews as pickled Python object arrays, which Nx cannot read, so
# `prepare_imdb.py` (alongside this file) converts it once into flat
# binaries:
#
#     python prepare_imdb.py --out /tmp/imdb
#     IMDB_DIR=/tmp/imdb elixir examples/text/bidirectional_lstm_imdb.exs
#
# Known deviations from the Keras example, none of which are
# configurable through Axon's public API today:
#
#   * Keras initializes the recurrent kernel orthogonally; Axon uses
#     one `:kernel_initializer` (glorot uniform) for both the input and
#     recurrent kernels.
#   * Keras' `unit_forget_bias=True` initializes the forget gate bias
#     to one; Axon initializes all four gate biases to zero.
#
# Both this example and the Keras one shuffle the training set, so the
# numbers move by a few thousandths between runs.

Mix.install([
  {:axon, path: Path.expand("../..", __DIR__)},
  {:exla, "~> 0.13"}
])

Nx.global_default_backend(EXLA.Backend)

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

trainable_count =
  model
  |> Axon.build()
  |> elem(0)
  |> then(& &1.(Nx.template({batch_size, maxlen}, :s32), Axon.ModelState.empty()))
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
report = fn %Axon.Loop.State{epoch: epoch, metrics: metrics} = state ->
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
|> Axon.Loop.run(train_data, Axon.ModelState.empty(),
  epochs: epochs,
  compiler: EXLA
)
