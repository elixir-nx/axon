Mix.install([
  {:axon, "~> 0.1.0"},
  # {:exla, "~> 0.2.2"},
  {:nx, "~> 0.2.1"}
])

# Specify EXLA as the default defn compiler
# EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

defmodule Power do
  require Axon

  defp build_model do
    fc =
      Axon.input("input", shape: {nil, 1})
      |> Axon.dense(32, activation: :relu)
      |> Axon.dense(64, activation: :relu)

    out1 = Axon.dense(fc, 1)
    out2 = Axon.dense(fc, 1)

    # Notice the "model" is just a tuple which matches the form we expect the targets
    # to be in, you can call Axon.container directly on this tuple and then you can
    # pass it to the Axon.Loop functions and you'll get valid results
    Axon.container({out1, out2})
  end

  def run do
    # Create the model
    model = build_model()

    # Create Data, this model will stream random numbers and 2 functions: x^2 and x^3
    # for a supervised model, Axon expects data to be in tuples of {x, y} where x is
    # model input and y is the target. Because we have multiple targets, we represent
    # y as a tuple. In the future, Axon will support any Nx container as an output
    data =
      Stream.repeatedly(fn ->
        # Batch size of 32
        x = Nx.random_uniform({32, 1}, -10, 10, type: {:f, 32})
        {x, {Nx.power(x, 2), Nx.power(x, 3)}}
      end)

    # Create the training loop, notice we specify 2 MSE objectives, 1 for the first
    # output and 1 for the second output. This will create a loss function which is
    # a weighted average of the two losses, if you had a problem which was predicting
    # something like the number of goals by Team A (regression) as well as the probability that
    # Team A wins (categorical), this would be something like [binary_cross_entropy: 0.5, mean_squared_error: 0.5]
    #
    # Internally Axon is interpreting each entry as {loss, weight}, so `loss` could also be a
    # custom function: [{fn y_true, y_pred -> Nx.mean(Nx.cos(y_true, y_pred)) end, 0.2}, log_cosh: 0.8]
    #
    # You can also just create a function yourself to compute loss which:
    #     fn {y_true1, y_true2}, {y_pred1, y_pred2} -> ... end
    #
    # There are a few Axon out-of-the box functions which expect multiple outputs and match
    # directly on the tuple like this.
    #
    # One thing to keep in mind is that there will be some give-and-take between each output
    # as you are trying to optimize for 2 things at once and optimal solutions for each
    # might not lie on a clean "manifold" - e.g. what's good for one output might not
    # be good for another output
    params =
      model
      |> Axon.Loop.trainer([mean_squared_error: 0.5, mean_squared_error: 0.5], :adam)
      |> Axon.Loop.run(data, %{}, iterations: 250, epochs: 5)

    IO.inspect(Axon.predict(model, params, Nx.tensor([[3]])))
  end
end

Power.run()
