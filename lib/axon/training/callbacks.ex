defmodule Axon.Training.Callbacks do
  @moduledoc """
  Axon training callbacks.
  """

  @doc """
  Standard IO Logger callback.

  Logs training results to standard out.
  """
  def standard_io_logger(train_state, :before_train, opts) do
    epochs = opts[:epochs]
    metrics = Map.keys(train_state[:metrics])

    IO.puts("Training model for #{epochs} epochs")
    IO.puts("Metrics: #{inspect(metrics)}")

    {:cont, train_state}
  end

  def standard_io_logger(train_state, :after_batch, opts) do
    log_every = opts[:log_every]

    case log_every do
      :none ->
        :ok

      :every ->
        log_batch(
          train_state[:epoch],
          train_state[:epoch_step],
          train_state[:epoch_loss],
          train_state[:metrics]
        )

      log_every when is_integer(log_every) ->
        if Nx.remainder(train_state[:epoch_step], log_every) == Nx.tensor(0) do
          log_batch(
            train_state[:epoch],
            train_state[:epoch_step],
            train_state[:epoch_loss],
            train_state[:metrics]
          )
        end
    end

    {:cont, train_state}
  end

  def standard_io_logger(train_state, :after_epoch, _opts) do
    epoch = Nx.to_scalar(train_state[:epoch])
    # Should this really be a part of train state, maybe an extra metadata argument?
    time = train_state[:time]
    epoch_loss = train_state[:epoch_loss]

    IO.puts("\n")
    IO.puts("Epoch #{epoch + 1} time: #{time / 1_000_000}s")
    IO.puts("Epoch #{epoch + 1} loss: #{:io_lib.format("~.5f", [Nx.to_scalar(epoch_loss)])}")

    train_state[:metrics]
    |> Enum.each(fn {k, v} ->
      IO.puts(
        "Epoch #{epoch + 1} #{Atom.to_string(k)}: #{:io_lib.format("~.5f", [Nx.to_scalar(v)])}"
      )
    end)

    IO.puts("\n")

    {:cont, train_state}
  end

  def standard_io_logger(train_state, :after_train, _opts) do
    IO.puts("Training finished")
    {:cont, train_state}
  end

  def standard_io_logger(train_state, _, _opts), do: {:cont, train_state}

  defp log_batch(epoch, step, loss, metrics) do
    metrics =
      metrics
      |> Enum.map(fn {k, v} ->
        "Average #{Atom.to_string(k)}: #{:io_lib.format("~.5f", [Nx.to_scalar(v)])}"
      end)

    metrics =
      Enum.join(
        ["Average Loss: #{:io_lib.format("~.5f", [Nx.to_scalar(loss)])}" | metrics],
        " - "
      )

    IO.write(
      "\rEpoch #{Nx.to_scalar(epoch) + 1}, batch #{Nx.to_scalar(step)} - " <>
        "#{metrics}"
    )
  end
end
