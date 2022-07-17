# Based on https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
Mix.install([
  {:axon, github: "elixir-nx/axon"},
  {:nx, "~> 0.2.1"},
  {:exla, "~> 0.2.2"},
  {:req, "~> 0.3.0"}
])

EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

defmodule TextGenerator do
  require Axon

  @download_url "https://www.gutenberg.org/files/11/11-0.txt"
  @sequence_length 100
  @batch_size 128

  def build_model(characters_count) do
    Axon.input("input_chars", shape: {nil, @sequence_length, 1})
    |> Axon.lstm(256)
    |> then(fn {_, out} -> out end)
    |> Axon.nx(fn t -> t[[0..-1//1, -1]] end)
    |> Axon.dropout(rate: 0.2)
    |> Axon.dense(characters_count, activation: :softmax)
  end

  def generate(model, params, init_seq, char_to_idx, idx_to_char, characters_count) do
    init_seq =
      init_seq
      |> String.downcase()
      |> String.to_charlist()
      |> Enum.map(&Map.fetch!(char_to_idx, &1))

    Enum.reduce(1..80, init_seq, fn _, seq ->
      init_seq =
        seq
        |> Enum.take(-@sequence_length)
        |> Nx.tensor()
        |> Nx.divide(characters_count)
        |> Nx.reshape({1, @sequence_length, 1})

      char =
        Axon.predict(model, params, init_seq)
        |> Nx.argmax()
        |> Nx.to_number()

      seq ++ [char]
    end)
    |> Enum.map(&Map.fetch!(idx_to_char, &1))
  end

  def transform_text(text, char_to_idx, characters_count) do
    train_data =
      text
      |> Enum.map(&Map.fetch!(char_to_idx, &1))
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      # We don't want the last chunk since we don't have a prediction for it.
      |> Enum.drop(-1)
      |> Nx.tensor()
      |> Nx.divide(characters_count)
      |> Nx.reshape({:auto, @sequence_length, 1})
      |> Nx.to_batched_list(@batch_size)

    train_labels =
      text
      |> Enum.drop(@sequence_length)
      |> Enum.map(&Map.fetch!(char_to_idx, &1))
      |> Nx.tensor()
      |> Nx.reshape({:auto, 1})
      |> Nx.equal(Nx.iota({characters_count}))
      |> Nx.to_batched_list(@batch_size)

    {train_data, train_labels}
  end

  def run do
    normalized_book_text =
      Req.get!(@download_url).body
      |> String.downcase()
      |> String.replace(~r/[^a-z \.\n]/, "")
      |> String.to_charlist()

    # Extract all then unique characters we have. Optionally we can sort them.
    characters = normalized_book_text |> Enum.uniq() |> Enum.sort()
    characters_count = Enum.count(characters)
    # Create a mapping for every character
    char_to_idx = Enum.with_index(characters) |> Enum.into(%{})
    idx_to_char = Enum.with_index(characters, &{&2, &1}) |> Enum.into(%{})

    model = build_model(characters_count)
    IO.inspect(model)

    {train_data, train_labels} =
      transform_text(normalized_book_text, char_to_idx, characters_count)

    IO.puts("Total batches: #{Enum.count(train_data)}")

    params =
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(0.001))
      |> Axon.Loop.run(Stream.zip(train_data, train_labels), %{}, epochs: 20, compiler: EXLA)

    init_sequence = """
    not like to drop the jar for fear
    of killing somebody underneath so managed to put it into one of the
    cupboards as she fell past it.
    """

    generated = generate(model, params, init_sequence, char_to_idx, idx_to_char, characters_count)

    IO.puts("Generated text: #{generated}")
  end
end

TextGenerator.run()
