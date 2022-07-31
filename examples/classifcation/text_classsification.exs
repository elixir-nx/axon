# Based on https://valueml.com/named-entity-recognition-using-lstm-in-keras/

# References
# http://alexminnaar.com/2019/08/22/ner-rnns-tensorflow.html

# Test Data: https://github.com/bhuvanakundumani/NER_tensorflow2.2.0/blob/master/data/test.txt

# Train Data: https://github.com/bhuvanakundumani/NER_tensorflow2.2.0/blob/master/data/train.txt

# Validation Data: https://github.com/bhuvanakundumani/NER_tensorflow2.2.0/blob/master/data/valid.txt

Mix.install([
  {:axon, github: "elixir-nx/axon"},
  {:exla, "~> 0.3.0-dev", github: "elixir-nx/nx", sparse: "exla"},
  {:nx, "~> 0.3.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
])

EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

defmodule TextClassification do
  require Axon
  # From Example http://alexminnaar.com/2019/08/22/ner-rnns-tensorflow.html
  EXLA.set_as_nx_default([:tpu, :cuda, :rocl, :host])
  require Logger
  @sequence_length 75
  @batch_size 128
  @lstm 64
  @embed_dimension 256
  @def_split [" ", "\n"]
  @start "-DOCSTART- -X- -X- O\n"
  @ending "-------------------------------------------------------------"
  @incorrect [".", "?", "%", ",", ":", ";", "-docstart-", "\n"]
  @unknown_encode_token 0
  @unknown_decode_token "Unknown"

  def train(model, data) do
    model
    |> Axon.Loop.trainer(:kl_divergence, Axon.Optimizers.adam(0.001)) # Can use :categorical_cross_entropy
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(data, %{}, epochs: 20, iterations: 300)
  end

  defp kl_divergence(y_true, y_pred) do
    Axon.Losses.kl_divergence(y_true, y_pred, reduction: :mean)
  end

  def build_model(word_count, label_count) do
      Axon.input("inputs", shape: {nil, @sequence_length})
       |> Axon.embedding(word_count, @sequence_length)
       |> Axon.lstm(@lstm)
       |> elem(1)
       |> Axon.dropout(rate: 0.2)
       |> Axon.dense(label_count, activation: :softmax)
  end

  defp encode(dictionary, word), do: dictionary[word] || @unknown_encode_token
  defp decode(dictionary, word), do: dictionary[word] || @unknown_decode_token

  def transform_words(word_labels, word_to_idx, label_to_idx, wcount, lcount) do

    train_data =
      word_labels
      |> Enum.map(fn {word, label} ->
        encode(word_to_idx, word)
      end)
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Enum.drop(-1)
      |> Nx.tensor
      |> Nx.divide(wcount)
      |> Nx.reshape({:auto, @sequence_length})
      |> Nx.to_batched_list(@batch_size)


    train_labels =
      word_labels
      |> Enum.map(fn {word, label} ->
          encode(label_to_idx, label)
      end)
      |> Enum.chunk_every(@sequence_length, 1, :discard)
      |> Enum.drop(-1)
      |> Nx.tensor
      |> Nx.reshape({:auto, @sequence_length, 1})
      |> Nx.to_batched_list(@batch_size)


      {train_data, train_labels}
  end


  def predict(init_sequence, params, model, idx_to_char, idx_to_label, count, real) do

        Enum.reduce(1..10, [], fn _, seq ->

          p = Axon.predict(model, params, init_sequence)

          predicted =
            p
            |> Nx.argmax(axis: -1)
            |> Nx.to_flat_list
            |> Enum.map(fn predict_sequence ->
                decode(idx_to_label, predict_sequence)
              end)


          given =
            init_sequence
            |> Nx.to_flat_list
            |> Enum.map(fn inpseq ->
                {actual_result, _} =
                  inpseq*count
                  |> Float.floor
                  |> Float.to_string
                  |> Integer.parse

                decode(idx_to_char, actual_result)
              end)

          rl =
            real
            |> Nx.to_flat_list
            |> Enum.map(fn real_sequence ->
                decode(idx_to_label, real_sequence)
              end)

          Enum.zip([given, predicted, rl])

        end)
        |> Enum.map(fn {input, output, r} ->
                true? = if output == r do "CORRECT" else "INCORRECT" end

                if true? == "CORRECT" do
                  Logger.warn "Word was: #{input}, Predicted Category was: #{output} Real Result: #{r}, RETURNED: #{true?}"
                else
                  Logger.error "Word was: #{input}, Predicted Category was: #{output} Real Result: #{r}, RETURNED: #{true?}"
                end
            end)
  end

  defp from_file(filename, opts \\ []) do
    data_stream =
      filename
      |> File.stream!()
  end

  def pre_process(streamed_data) do
    char_label_data =
    streamed_data
    |> Stream.map(fn line ->
      [token | entities] = String.split(line, @def_split)
      if(
      token not in (["", "\n", @start, @ending | @incorrect]) &&
      String.length(token) !== 1)
      do
        single_label_classification = Enum.fetch!(entities, 2)
        {String.downcase(token), single_label_classification}
      else
        nil
      end
    end)
    |> Stream.reject(&is_nil/1)

    unique_words = uniq_seq(char_label_data, 0)
    unique_labels = uniq_seq(char_label_data, 1)

    {char_label_data, unique_words, unique_labels}
  end

  # Convert Words & Labels Sequences into a unique array
  defp uniq_seq(data, index), do: Enum.map(data, &(elem(&1, index))) |> Enum.uniq

  # Take a random sample from the data
  defp fetch_random_sample({sequence_words, sequence_labels} = seq_tuple) do
    {predict_words, predict_index} =
      Enum.with_index(sequence_words)
      |> Enum.random

    predict_labels = Enum.fetch!(sequence_labels, predict_index)

    {predict_words, predict_labels}
  end

  def run do

    {word_label_data, unique_words, unique_labels} =
      from_file("./data/train.txt")
      |> pre_process

    {test_word_label_data, unique_test_words, unique_test_labels} =
      from_file("./data/test.txt")
      |> pre_process

      streamed =
        word_label_data
        |> Stream.reject(&is_nil(&1))

      test_streamed =
        test_word_label_data
        |> Stream.reject(&is_nil(&1))


      word_count  = Enum.count(unique_words)
      label_count = Enum.count(unique_labels)

      word_to_idx =
        unique_words
        |> Stream.uniq
        |> Stream.with_index
        |> Enum.into(%{})


        label_to_idx =
          unique_labels
          |> Stream.uniq
          |> Stream.with_index
          |> Enum.into(%{})

          idx_to_word =
              unique_words
              |> Stream.uniq
              |> Enum.with_index(&{&2, &1})
              |> Enum.into(%{})

          idx_to_label =
            unique_labels
            |> Stream.uniq
            |> Enum.with_index(&{&2, &1})
            |> Enum.into(%{})

            {td, tl} = transform_words(streamed, word_to_idx, label_to_idx, word_count, label_count)

            model = build_model(word_count, label_count)

            data = Stream.zip(td, tl)

            {predict_words, predict_labels} =
              transform_words(test_streamed, word_to_idx, label_to_idx, word_count, label_count)
              |> fetch_random_sample

            params = train(model, data)

            predict(predict_words, params, model, idx_to_word, idx_to_label, word_count, predict_labels)

  end


end



TextClassification.run()
