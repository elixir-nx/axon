Mix.install([
  {:axon, "~> 0.1.0"},
  {:exla, "~> 0.2.2"},
  {:nx, "~> 0.2.1"},
  {:scidata, "~> 0.1.6"}
])

EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

defmodule MNISTGAN do
  require Axon
  alias Axon.Loop.State
  import Nx.Defn

  defp transform_images({bin, type, shape}) do
    bin
    |> Nx.from_binary(type)
    |> Nx.reshape({elem(shape, 0), 1, 28, 28})
    |> Nx.divide(255.0)
    |> Nx.to_batched_list(32)
  end

  defp build_generator(z_dim) do
    Axon.input("input", shape: {nil, z_dim})
    |> Axon.dense(256)
    |> Axon.leaky_relu(alpha: 0.9)
    |> Axon.batch_norm()
    |> Axon.dense(512)
    |> Axon.leaky_relu(alpha: 0.9)
    |> Axon.batch_norm()
    |> Axon.dense(1024)
    |> Axon.leaky_relu(alpha: 0.9)
    |> Axon.batch_norm()
    |> Axon.dense(784)
    |> Axon.tanh()
    |> Axon.reshape({1, 28, 28})
  end

  defp build_discriminator(input_shape) do
    Axon.input("input", shape: input_shape)
    |> Axon.flatten()
    |> Axon.dense(512)
    |> Axon.leaky_relu(alpha: 0.9)
    |> Axon.dense(256)
    |> Axon.leaky_relu(alpha: 0.9)
    |> Axon.dense(2, activation: :softmax)
  end

  defnp running_average(avg, obs, i) do
    avg
    |> Nx.multiply(i)
    |> Nx.add(obs)
    |> Nx.divide(Nx.add(i, 1))
  end

  defn init(init_d, init_g, init_optim_d, init_optim_g) do
    d_params = init_d.(%{})
    g_params = init_g.(%{})

    %{
      iteration: Nx.tensor(0),
      discriminator: %{
        model_state: d_params,
        optimizer_state: init_optim_d.(d_params),
        loss: Nx.tensor(0.0)
      },
      generator: %{
        model_state: g_params,
        optimizer_state: init_optim_g.(g_params),
        loss: Nx.tensor(0.0)
      }
    }
  end

  defn batch_step(d_model, g_model, optim_d, optim_g, real_images, state) do
    iter = state[:iteration]
    d_params = state[:discriminator][:model_state]
    g_params = state[:generator][:model_state]

    # Update D
    fake_labels = Nx.iota({32, 2}, axis: 1)
    real_labels = Nx.reverse(fake_labels)
    noise = Nx.random_normal({32, 100})

    {d_loss, d_grads} =
      value_and_grad(d_params, fn params ->
        %{prediction: fake_images} = g_model.(g_params, noise)

        %{prediction: d_fake_preds} = d_model.(params, fake_images)
        # Merge new state with params
        %{prediction: d_real_preds} = d_model.(d_params, real_images)

        joint_preds = Nx.concatenate([d_fake_preds, d_real_preds], axis: 0)
        joint_labels = Nx.concatenate([fake_labels, real_labels], axis: 0)

        Axon.Losses.categorical_cross_entropy(joint_labels, joint_preds, reduction: :mean)
      end)

    d_optimizer_state = state[:discriminator][:optimizer_state]

    {d_updates, d_optimizer_state} = optim_d.(d_grads, d_optimizer_state, d_params)
    d_params = Axon.Updates.apply_updates(d_params, d_updates)

    # Update G
    {g_loss, g_grads} =
      value_and_grad(g_params, fn params ->
        %{prediction: fake_images} = g_model.(params, noise)

        %{prediction: d_preds} = d_model.(d_params, fake_images)

        Axon.Losses.categorical_cross_entropy(real_labels, d_preds, reduction: :mean)
      end)

    g_optimizer_state = state[:generator][:optimizer_state]

    {g_updates, g_optimizer_state} = optim_g.(g_grads, g_optimizer_state, g_params)
    g_params = Axon.Updates.apply_updates(g_params, g_updates)

    %{
      iteration: iter + 1,
      discriminator: %{
        model_state: d_params,
        optimizer_state: d_optimizer_state,
        loss: running_average(state[:discriminator][:loss], d_loss, iter)
      },
      generator: %{
        model_state: g_params,
        optimizer_state: g_optimizer_state,
        loss: running_average(state[:generator][:loss], g_loss, iter)
      }
    }
  end

  defp train_loop(d_model, g_model) do
    {init_optim_d, optim_d} = Axon.Optimizers.adam(2.0e-3, b1: 0.5)
    {init_optim_g, optim_g} = Axon.Optimizers.adam(2.0e-3, b1: 0.5)

    {d_init, d_model} = Axon.compile(d_model, mode: :train)
    {g_init, g_model} = Axon.compile(g_model, mode: :train)

    step = &batch_step(d_model, g_model, optim_d, optim_g, &1, &2)
    init = fn %{} -> init(d_init, g_init, init_optim_d, init_optim_g) end

    Axon.Loop.loop(step, init)
  end

  defp log_iteration(state) do
    %State{epoch: epoch, iteration: iter, step_state: pstate} = state

    g_loss = "G: #{:io_lib.format('~.5f', [Nx.to_number(pstate[:generator][:loss])])}"
    d_loss = "D: #{:io_lib.format('~.5f', [Nx.to_number(pstate[:discriminator][:loss])])}"

    "\rEpoch: #{Nx.to_number(epoch)}, batch: #{Nx.to_number(iter)} #{g_loss} #{d_loss}"
  end

  defp view_generated_images(model, batch_size, state) do
    %State{step_state: pstate} = state
    noise = Nx.random_normal({batch_size, 100})
    preds = Axon.predict(model, pstate[:generator][:model_state], noise)

    preds
    |> Nx.reshape({batch_size, 28, 28})
    |> Nx.to_heatmap()
    |> IO.inspect()

    {:continue, state}
  end

  def run() do
    {images, _} = Scidata.MNIST.download()
    train_images = transform_images(images)

    generator = build_generator(100)
    discriminator = build_discriminator({nil, 1, 28, 28})

    discriminator
    |> train_loop(generator)
    |> Axon.Loop.log(:iteration_completed, &log_iteration/1, :stdio, every: 50)
    |> Axon.Loop.handle(:epoch_completed, &view_generated_images(generator, 3, &1))
    |> Axon.Loop.run(train_images, %{}, epochs: 10, compiler: EXLA)
  end
end

EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

MNISTGAN.run()
