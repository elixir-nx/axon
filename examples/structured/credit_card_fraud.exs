Mix.install([
	{:axon, "~> 0.1.0-dev", path: "."},
	{:exla, path: "../nx/exla"},
	{:nx, path: "../nx/nx", override: true},
	{:explorer, "~> 0.1.0-dev", github: "elixir-nx/explorer"}
])

defmodule CreditCardFraud do
	alias Axon.Loop.State

	defp data() do
		fname = "examples/structured/creditcard.csv"

		IO.puts("Loading #{fname}")
		df = Explorer.DataFrame.read_csv!(fname, dtypes: [{"Time", :float}])

		{train_df, test_df} = split_train_test(df, 0.9)

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
		features = Explorer.DataFrame.select(df, & &1 == "Class", :drop)
		targets = Explorer.DataFrame.select(df, & &1 == "Class", :keep)
		{features, targets}
	end

	defp normalize_data(df) do
		df
		|> Explorer.DataFrame.names()
		|> Map.new(fn name ->
			normalized =
				df
				|> Explorer.DataFrame.pull(name)
				|> then(&Explorer.Series.divide(&1, Explorer.Series.max(&1)))
			{name, normalized}
		end)
		|> Explorer.DataFrame.from_map()
	end

	defp df_to_tensor(df) do
		df
		|> Explorer.DataFrame.names()
		|> Enum.reduce(:first, fn
				name, :first ->
					df
					|> Explorer.DataFrame.pull(name)
					|> Explorer.Series.to_tensor()
					|> Nx.new_axis(1)

				name, tensor ->
					df
					|> Explorer.DataFrame.pull(name)
					|> Explorer.Series.to_tensor()
					|> Nx.new_axis(1)
					|> then(&Nx.concatenate([tensor, &1], axis: 1))
		end)
	end

	defp build_model(num_features) do
		Axon.input({nil, num_features})
		|> Axon.dense(256)
		|> Axon.relu()
		|> Axon.dense(256)
		|> Axon.relu()
		|> Axon.dropout(rate: 0.3)
		|> Axon.dense(1)
		|> Axon.sigmoid()
	end

  defp log_metrics(
         %State{epoch: epoch, iteration: iter, metrics: metrics, step_state: pstate} = state,
         mode
       ) do
    loss =
      case mode do
        :train ->
          %{loss: loss} = pstate
          "Loss: #{:io_lib.format('~.5f', [Nx.to_scalar(loss)])}"

        :test ->
          ""
      end

    metrics =
      metrics
      |> Enum.map(fn {k, v} ->
      		v =
      			case Nx.type(v) do
      				{:f, _} ->
      					:io_lib.format('~.5f', [Nx.to_scalar(v)])

      				_ ->
      					:io_lib.format('~7.. B', [Nx.to_scalar(v)])
      			end
      		"#{k}: #{v}" end)
      |> Enum.join(" ")

    epoch = :io_lib.format('~3.. B', [Nx.to_scalar(epoch)])
    batch = :io_lib.format('~3.. B', [Nx.to_scalar(iter)])
    IO.write("\rEpoch: #{epoch}, Batch: #{batch}, #{loss} #{metrics}")

    {:continue, state}
  end

  defp summarize(%State{metrics: metrics} = state) do
  	IO.write("\n\n")

  	legit_transactions_declined = Nx.to_scalar(metrics["fp"])
  	legit_transactions_accepeted = Nx.to_scalar(metrics["tn"])
  	fraud_transactions_accepted = Nx.to_scalar(metrics["fn"])
  	fraud_transactions_declined = Nx.to_scalar(metrics["tp"])
  	total_fraud = fraud_transactions_declined + fraud_transactions_accepted
  	total_legit = legit_transactions_declined + legit_transactions_accepeted

  	fraud_denial_percent = 100 * (fraud_transactions_declined / total_fraud)
  	legit_denial_percent = 100 * (legit_transactions_declined / total_legit)

  	IO.puts("Legit Transactions Declined: #{legit_transactions_declined}")
  	IO.puts("Fraudulent Transactions Caught: #{fraud_transactions_declined}")
  	IO.puts("Fraudulent Transactions Missed: #{fraud_transactions_accepted}")
  	IO.puts("Likelihood of catching fraud: #{fraud_denial_percent}%")
  	IO.puts("Likelihood of denying legit transaction: #{legit_denial_percent}%")

  	{:continue, state}
  end

  defp test_model(model, model_state, test_data) do
  	model
  	|> Axon.Loop.evaluator(model_state)
  	|> Axon.Loop.metric(:true_positives, "tp", :running_sum)
		|> Axon.Loop.metric(:true_negatives, "tn", :running_sum)
		|> Axon.Loop.metric(:false_positives, "fp", :running_sum)
		|> Axon.Loop.metric(:false_negatives, "fn", :running_sum)
		|> Axon.Loop.handle(:epoch_completed, &summarize/1)
		|> Axon.Loop.run(test_data, compiler: EXLA)
  end

  defp train_model(model, train_data) do
		model
		|> Axon.Loop.trainer(:binary_cross_entropy, :adam)
		|> Axon.Loop.metric(:true_positives, "tp", :running_sum)
		|> Axon.Loop.metric(:true_negatives, "tn", :running_sum)
		|> Axon.Loop.metric(:false_positives, "fp", :running_sum)
		|> Axon.Loop.metric(:false_negatives, "fn", :running_sum)
		|> Axon.Loop.handle(:iteration_completed, &log_metrics(&1, :train), every: 10)
		|> Axon.Loop.run(train_data, epochs: 10, compiler: EXLA)
  end

	def run() do
		{train, test} = data()
		{train_inputs, train_targets} = train
		{test_inputs, test_targets} = test

		fraud = Nx.sum(train_targets) |> Nx.to_scalar()
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

		model = build_model(30)

		model
		|> train_model(batched_train)
		|> then(&test_model(model, &1, batched_test))
	end
end

CreditCardFraud.run()