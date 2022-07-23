# Based on https://valueml.com/named-entity-recognition-using-lstm-in-keras/

# References
# http://alexminnaar.com/2019/08/22/ner-rnns-tensorflow.html

# Test Data: https://github.com/bhuvanakundumani/NER_tensorflow2.2.0/blob/master/data/test.txt

# Train Data: https://github.com/bhuvanakundumani/NER_tensorflow2.2.0/blob/master/data/train.txt

# Validation Data: https://github.com/bhuvanakundumani/NER_tensorflow2.2.0/blob/master/data/valid.txt

Mix.install([
  {:axon, github: "elixir-nx/axon"},
  {:nx, "~> 0.2.1"},
  {:exla, "~> 0.2.2"}
])

EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

defmodule TextClassification do
  require Axon
  # From Example http://alexminnaar.com/2019/08/22/ner-rnns-tensorflow.html
  EXLA.set_as_nx_default([:tpu, :cuda, :rocl, :host]) # My :host option doesnt work, jit_apply is undefined
  require Logger
  @sequence_length 75
  @batch_size 128
  @lstm 128
  @embed_dimension 256
  @def_split " "
  @start "-DOCSTART- -X- -X- O\n"
  @ending "-------------------------------------------------------------"
  @incorrect [".", "?", "%", ",", ":", ";", "-docstart-"]

  # seqlen
  # word_count

  def train(model, data) do
    model
    |> Axon.Loop.trainer(&kl_divergence/2, Axon.Optimizers.adam(0.001))
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(data, %{}, epochs: 3, iterations: 5)

  end

  defp kl_divergence(y_true, y_pred) do
    Axon.Losses.kl_divergence(y_true, y_pred, reduction: :mean)
  end

  def build_model(word_count, label_count) do
      Axon.input({nil, @sequence_length, 1}, "inputs")
       |> Axon.embedding(word_count, @sequence_length)
       |> Axon.nx(fn t ->
         t[[0..-1//1, -1]] # Dropping out the sized 1 column created from the embed
       end)
       |> Axon.spatial_dropout(rate: 0.1)
       |> Axon.lstm(@lstm)
       |> then(fn {{new_cell, new_hidden}, out} ->
         out
       end)
      |> Axon.dropout(rate: 0.2)
      |> Axon.dense(label_count, activation: :softmax)
  end

  defp encode(dictionary, word) do
    with {:ok, encoded} <- Map.fetch(dictionary, word) do
      encoded
    else
      _->
      0
    end
  end

  defp decode(dictionary, word) do
    with {:ok, decoded} <- Map.fetch(dictionary, word) do
      decoded
    else
      _-> "Unknown"
    end
  end

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
      |> Nx.reshape({:auto, @sequence_length, 1})
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
      !is_nil(token) &&
      token !== "" &&
      String.length(token) !== 1 &&
      line !== "\n" &&
      token !== @start &&
      token !== @ending &&
      token not in @incorrect)
      do

        [ r | rest ] = entities |> List.last |> String.split("\n")

        {String.downcase(token), r}
      else
        nil
      end
    end)
    |> Stream.reject(fn x -> is_nil(x) end)
    # Here we start to remove unwanted things to reduce memory size


    unique_words = char_label_data |> Enum.map(&(elem(&1, 0))) |> Enum.uniq


    unique_labels = char_label_data |> Enum.map(&(elem(&1, 1))) |> Enum.uniq

    {char_label_data, unique_words, unique_labels}
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


      word_count  = unique_words |> Enum.count
      label_count  = unique_labels |> Enum.count

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

            {ttd, ttl} = transform_words(test_streamed, word_to_idx, label_to_idx, word_count, label_count)

            unpadded_model = build_model(word_count, label_count)

            data = Stream.zip(td, tl)
#
            unpadded_params = train(unpadded_model, data)

            random_int_from_sample = 5# Has to be in range
            sample_slice = Enum.fetch!(ttd, random_int_from_sample)
            real_results = Enum.fetch!(ttl, random_int_from_sample)

            predict(sample_slice, unpadded_params, unpadded_model, idx_to_word, idx_to_label, word_count, real_results)

  end


end


TextClassification.run()
