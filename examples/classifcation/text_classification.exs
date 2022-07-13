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
  require Logger
  @sequence_length 100
  @batch_size 128
  @lstm 1024
  @embed_dimension 256
  @def_split " "
  @start "-DOCSTART- -X- -X- O\n"
  @ending "-------------------------------------------------------------"

    def train(model, data) do
      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adam(0.001))
      |> Axon.Loop.metric(:accuracy, "Accuracy")
      |> Axon.Loop.run(data, %{}, epochs: 1, iterations: 1)

    end

    def build_model(word_count, label_count) do

        Axon.input({nil, @sequence_length, 1}, "inputs")
         |> Axon.embedding(word_count, @sequence_length)
         |> Axon.nx(fn t ->
           t[[0..-1//1, -1]]
       end)
         |> Axon.spatial_dropout(rate: 0.1)
         |> Axon.lstm(@lstm)
         |> then(fn {{new_cell, new_hidden}, out} ->
           out
         end)
        |> Axon.dropout(rate: 0.2)
        |> Axon.dense(label_count, activation: :softmax)

    end

    def transform_words(word_labels, word_to_idx, label_to_idx, wcount, lcount) do

      train_data =
        word_labels
        |> Enum.map(fn {word, label} ->
          encode(word_to_idx, word)
        end)
        |> Enum.chunk_every(@sequence_length, 1, :discard)
        |> Nx.tensor
        |> Nx.reshape({:auto, @sequence_length, 1})
        |> Nx.to_batched_list(@batch_size)


      train_labels =
        word_labels
        |> Enum.map(fn {word, label} ->
            encode(label_to_idx, label)
        end)
        |> Enum.chunk_every(@sequence_length, 1, :discard)
        |> Nx.tensor
        |> Nx.reshape({:auto, @sequence_length, 1})
        |> Nx.to_batched_list(@batch_size)


        {train_data, train_labels}
    end


    def predict(init_sequence, params, model, idx_to_char, idx_to_label, count, real) do

          Enum.reduce(1..100, [], fn _, seq ->

            p = Axon.predict(model, params, init_sequence)

            predicted =
              p
              |> Nx.argmax(axis: -1)
              |> Nx.to_flat_list

            given =
            init_sequence
            |> Nx.to_flat_list



            rl =
              real
              |> Nx.to_flat_list


            seq ++ [{given, predicted, rl}]

          end)
          |> Enum.map(fn {inp, res, rll} ->
  #
            input =
              inp
              |> Enum.map(fn inpseq ->
                decode(idx_to_char, inpseq)
              end)

            output =
              res
              |> Enum.map(fn inpseq ->
                decode(idx_to_label, inpseq)
              end)

              rl_d =
              rll
              |> Enum.map(fn inpseq ->
                decode(idx_to_label, inpseq)
              end)
          {input, output, rl_d}
          end)
          |> Enum.map(fn {i, o, r} ->
                Enum.zip([i, o, r])
                |> Enum.map(fn {input, output, r} ->
                  true? = if output == r do "CORRECT" else "INCORRECT" end

                  if true? == "CORRECT" do
                    Logger.warn "Word was: #{input}, Predicted Category was: #{output} Real Result: #{r}, RETURNED: #{true?}"
                  else
                    Logger.error "Word was: #{input}, Predicted Category was: #{output} Real Result: #{r}, RETURNED: #{true?}"
                  end
              end)
          end)
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




        word_to_idx = to_idx(unique_words)
        label_to_idx = to_idx(unique_labels)

        idx_to_word = idx_to(unique_words)
        idx_to_label = idx_to(unique_labels)


        word_count  = unique_words |> Enum.count
        label_count  = unique_labels |> Enum.count


        {td, tl} = transform_words(streamed, word_to_idx, label_to_idx, word_count, label_count)

        {ttd, ttl} = transform_words(test_streamed, word_to_idx, label_to_idx, word_count, label_count)

        unpadded_model = build_model(word_count, label_count)

        data = Stream.zip(td, tl)

        unpadded_params = train(unpadded_model, data)

        random_int_from_samples = 3 # Has to be in range
        sample_slice = Enum.fetch!(td, random_int_from_sample)
        real_results = Enum.fetch!(tl, random_int_from_sample)

        predict(i, unpadded_params, unpadded_model, idx_to_word, idx_to_label, word_count, real_results)

    end



    defp from_file(filename, opts \\ []) do

      data_stream =
        filename
        |> File.stream!()
    end

    defp pre_process(streamed_data) do

      char_label_data =
      streamed_data
      |> Stream.map(fn line ->
        [token | entities] = String.split(line, @def_split)
        if(
        !is_nil(token) &&
        token !== "" &&
        line !== "\n" &&
        token !== @start &&
        token !== @ending)
        do

          [ r | rest ] = entities |> List.last |> String.split("\n")

          {token, r}
        else
          nil
        end
      end)
      #|> Enum.take(1024) # Slice a small sample if needed
      |> Stream.reject(fn x -> is_nil(x) end)


      unique_words = char_label_data |> Enum.map(&(elem(&1, 0))) |> Enum.uniq


      unique_labels = char_label_data |> Enum.map(&(elem(&1, 1))) |> Enum.uniq

      {char_label_data, unique_words, unique_labels}
    end

    defp encode(dictionary, word) do
      with {:ok, id} <- Map.fetch(dictionary, word) do
        id
      else
        _->
        0
      end
    end

    defp decode(dictionary, word) do
      with {:ok, id} <- Map.fetch(dictionary, word) do
        word
      else
        _-> "Unknown"
      end
    end

    defp to_idx(str_array) do
      str_array
      |> Stream.uniq
      |> Stream.with_index
      |> Enum.into(%{})
    end

    defp idx_to(str_array) do
      str_array
      |> Stream.uniq
      |> Enum.with_index(&{&2, &1})
      |> Enum.into(%{})
    end



end

TextClassification.run()
