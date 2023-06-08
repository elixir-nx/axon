Mix.install([
  {:axon, "~> 0.5"},
  {:exla, "~> 0.5"},
  {:nx, "~> 0.5"},
  {:explorer, "~> 0.5"}
])

defmodule CreditCardFraud do
  alias Axon.Loop.State

  # Download data with a Kaggle account: https://www.kaggle.com/mlg-ulb/creditcardfraud/
  @file_name "examples/structured/creditcard.csv"

  defp data() do
    IO.puts("Loading #{@file_name}")
    df = Explorer.DataFrame.from_csv!(@file_name, dtypes: [{"Time", :float}])

    {train_df, test_df} = split_train_test(df, 0.8)

    IO.puts("Training Samples: #{inspect(Explorer.DataFrame.n_rows(train_df))}")
    IO.puts("Testing Samples: #{inspect(Explorer.DataFrame.n_rows(test_df))}")
    IO.write("\n\n")

    {train_features, train_targets} = split_features_targets(train_df)
    {test_features, test_targets} = split_features_targets(test_df)

    train_features = normalize_data(train_features)
    test_features = normalize_data(test_features)

    {
      {df_to_tensor(train_features), df_to_tensor(train_targets)},
      {df_to_tensor(test_features), df_to_tensor(test_targets)}
    }
  end

  defp split_train_test(df, portion) do
    num_examples = Explorer.DataFrame.n_rows(df)
    num_train = ceil(portion * num_examples)
    num_test = num_examples - num_train

    {
      Explorer.DataFrame.slice(df, 0, num_train),
      Explorer.DataFrame.slice(df, num_train, num_test)
    }
  end

  defp split_features_targets(df) do
    features = Explorer.DataFrame.select(df, &(&1 == "Class"), :drop)
    targets = Explorer.DataFrame.select(df, &(&1 == "Class"), :keep)
    {features, targets}
  end

  defp normalize(name),
    do: fn df ->
      Explorer.Series.divide(
        df[name],
        Explorer.Series.max(
          Explorer.Series.transform(df[name], fn x ->
            if x >= 0 do
              x
            else
              -x
            end
          end)
        )
      )
    end

  defp normalize_data(df) do
    df
    |> Explorer.DataFrame.names()
    |> Map.new(&{&1, normalize(&1)})
    |> then(&Explorer.DataFrame.mutate(df, &1))
  end

  defp df_to_tensor(df) do
    df
    |> Explorer.DataFrame.names()
    |> Enum.map(&(Explorer.Series.to_tensor(df[&1]) |> Nx.new_axis(-1)))
    |> Nx.concatenate(axis: 1)
  end

  defp build_model(num_features) do
    Axon.input("input", shape: {nil, num_features})
    |> Axon.dense(256)
    |> Axon.relu()
    |> Axon.dense(256)
    |> Axon.relu()
    |> Axon.dropout(rate: 0.3)
    |> Axon.dense(1)
    |> Axon.sigmoid()
  end

  defp summarize(%State{metrics: metrics} = state) do
    IO.write("\n\n")

    legit_transactions_declined = Nx.to_number(metrics["fp"])
    legit_transactions_accepted = Nx.to_number(metrics["tn"])
    fraud_transactions_accepted = Nx.to_number(metrics["fn"])
    fraud_transactions_declined = Nx.to_number(metrics["tp"])
    total_fraud = fraud_transactions_declined + fraud_transactions_accepted
    total_legit = legit_transactions_declined + legit_transactions_accepted

    fraud_denial_percent = 100 * (fraud_transactions_declined / total_fraud)
    legit_denial_percent = 100 * (legit_transactions_declined / total_legit)

    IO.puts("Legit Transactions Declined: #{legit_transactions_declined}")
    IO.puts("Fraudulent Transactions Caught: #{fraud_transactions_declined}")
    IO.puts("Fraudulent Transactions Missed: #{fraud_transactions_accepted}")
    IO.puts("Likelihood of catching fraud: #{fraud_denial_percent}%")
    IO.puts("Likelihood of denying legit transaction: #{legit_denial_percent}%")

    {:continue, state}
  end

  defp metrics(loop) do
    loop
    |> Axon.Loop.metric(:true_positives, "tp", :running_sum)
    |> Axon.Loop.metric(:true_negatives, "tn", :running_sum)
    |> Axon.Loop.metric(:false_positives, "fp", :running_sum)
    |> Axon.Loop.metric(:false_negatives, "fn", :running_sum)
  end

  defp test_model(model, model_state, test_data) do
    model
    |> Axon.Loop.evaluator()
    |> metrics()
    |> Axon.Loop.handle(:epoch_completed, &summarize/1)
    |> Axon.Loop.run(test_data, model_state, compiler: EXLA)
  end

  defp train_model(model, loss, optimizer, train_data) do
    model
    |> Axon.Loop.trainer(loss, optimizer)
    |> Axon.Loop.run(train_data, %{}, epochs: 30, compiler: EXLA)
  end

  def run() do
    {train, test} = data()
    {train_inputs, train_targets} = train
    {test_inputs, test_targets} = test

    fraud = Nx.sum(train_targets) |> Nx.to_number()
    legit = Nx.size(train_targets) - fraud

    batched_train_inputs = Nx.to_batched_list(train_inputs, 2048)
    batched_train_targets = Nx.to_batched_list(train_targets, 2048)
    batched_train = Stream.zip(batched_train_inputs, batched_train_targets)

    batched_test_inputs = Nx.to_batched_list(test_inputs, 2048)
    batched_test_targets = Nx.to_batched_list(test_targets, 2048)
    batched_test = Stream.zip(batched_test_inputs, batched_test_targets)

    IO.puts("# of legit transactions (train): #{legit}")
    IO.puts("# of fraudulent transactions (train): #{fraud}")
    IO.puts("% fraudlent transactions (train): #{100 * (fraud / (legit + fraud))}%")
    IO.write("\n\n")

    model = build_model(elem(train_inputs.shape, 1))

    loss =
      &Axon.Losses.binary_cross_entropy(
        &1,
        &2,
        negative_weight: 1 / legit,
        positive_weight: 1 / fraud,
        reduction: :mean
      )

    optimizer = Polaris.Optimizers.adam(1.0e-2)

    model
    |> train_model(loss, optimizer, batched_train)
    |> then(&test_model(model, &1, batched_test))
  end
end

CreditCardFraud.run()
