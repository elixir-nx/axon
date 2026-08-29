defmodule Axon.Pruning do
  @moduledoc """
  Model pruning.

  Pruning removes weights from a trained model by setting them to zero.
  Most of the parameters in a trained network contribute very little to
  its output, so a large fraction of them can be removed with a small
  loss in accuracy, especially when the pruned model is fine-tuned for
  a few more steps afterwards.

  This module implements **unstructured magnitude pruning** of an
  `Axon.ModelState`: the parameters with the smallest absolute values
  are zeroed, either against a single global threshold or per tensor.
  Pruning produces a *mask* alongside the pruned state. The mask is a
  nested map that mirrors the model state data, with a `{:u, 8}` tensor
  (`1` = keep, `0` = pruned) for every pruned parameter. Use it with
  `masked_optimizer/2` to keep pruned weights at zero while fine-tuning:

      # Prune 90% of the kernel weights
      {pruned_state, mask} = Axon.Pruning.prune(model_state, 0.9)

      # Report the resulting sparsity
      Axon.Pruning.global_sparsity(pruned_state)
      #=> 0.87...

      # Fine-tune the remaining weights; pruned weights stay at zero
      trained_state =
        model
        |> Axon.Loop.trainer(
          :categorical_cross_entropy,
          Axon.Pruning.masked_optimizer(:adam, mask)
        )
        |> Axon.Loop.run(data, pruned_state, epochs: 2)

  ## Limitations

  Zeroed weights are still stored densely; this module reduces the number
  of non-zero parameters, not the memory footprint or the latency of the
  model. Taking advantage of sparsity requires backend support for sparse
  tensors or structured pruning, neither of which is implemented here. The
  model graph is never modified, only the model state. Ties between
  parameters of equal magnitude are broken arbitrarily but deterministically.
  """
  import Nx.Defn, only: [deftransform: 1, deftransform: 2]

  alias Axon.ModelState

  @doc """
  Computes a magnitude-based pruning mask for the given model state.

  `sparsity` is a number between `0` and `1` giving the fraction of the
  selected parameters to prune. The selected parameters are ranked by
  absolute value and the smallest ones are marked as pruned. The returned
  mask is a nested map that mirrors the model state data and contains a
  `{:u, 8}` tensor (`1` = keep, `0` = pruned) for each selected parameter.
  Parameters which were not selected are absent from the mask.

  Only parameters are considered, never state such as batch normalization
  running statistics. Tied parameters (see `Axon.ModelState.tie/4`),
  quantized parameters and non-float parameters are skipped.

  This function does not modify the model state. See `apply_mask/2` and
  `prune/3`.

  ## Options

    * `:scope` - `:global` ranks all selected parameters together, so a
      single threshold is used across the model and some tensors end up
      sparser than others. `:tensor` prunes each selected tensor to
      `sparsity` independently. Defaults to `:global`.

    * `:filter` - an arity-1 function which receives the access path of
      a parameter as a list of strings, such as `["dense_0", "kernel"]`
      or `["lstm_0", "input_kernel", "wii"]`, and returns whether that
      parameter should be considered for pruning. The default filter
      selects every parameter whose path contains a name ending in
      `"kernel"`, which covers dense, convolution and embedding kernels
      as well as the input and hidden kernels of recurrent layers, and
      leaves biases and normalization parameters untouched.

  ## Examples

      iex> model = Axon.input("x", shape: {nil, 2}) |> Axon.dense(4, name: "dense")
      iex> {init_fn, _} = Axon.build(model)
      iex> model_state = init_fn.(Nx.template({1, 2}, :f32), Axon.ModelState.empty())
      iex> kernel = Nx.tensor([[1.0, -8.0, 3.0, 0.5], [2.0, 7.0, -0.1, 4.0]])
      iex> model_state = put_in(model_state, [Access.key!(:data), "dense", "kernel"], kernel)
      iex> mask = Axon.Pruning.magnitude_mask(model_state, 0.5)
      iex> mask["dense"]["kernel"]
      #Nx.Tensor<
        u8[2][4]
        [
          [0, 1, 1, 0],
          [0, 1, 0, 1]
        ]
      >
      iex> Map.has_key?(mask["dense"], "bias")
      false

  """
  deftransform magnitude_mask(%ModelState{} = model_state, sparsity, opts \\ []) do
    unless is_number(sparsity) and sparsity >= 0 and sparsity <= 1 do
      raise ArgumentError,
            "expected sparsity to be a number between 0 and 1, got: #{inspect(sparsity)}"
    end

    opts = Keyword.validate!(opts, scope: :global, filter: &default_filter/1)
    scope = opts[:scope]
    filter = opts[:filter]

    unless scope in [:global, :tensor] do
      raise ArgumentError, "expected :scope to be :global or :tensor, got: #{inspect(scope)}"
    end

    unless is_function(filter, 1) do
      raise ArgumentError, "expected :filter to be an arity-1 function, got: #{inspect(filter)}"
    end

    candidates =
      model_state
      |> parameter_leaves()
      |> Enum.filter(fn {path, tensor} ->
        Nx.Type.float?(Nx.type(tensor)) and filter.(path)
      end)
      |> Enum.sort_by(fn {path, _} -> path end)

    masks =
      case scope do
        :tensor -> tensor_masks(candidates, sparsity)
        :global -> global_masks(candidates, sparsity)
      end

    Enum.reduce(masks, %{}, fn {path, mask}, acc -> put_path(acc, path, mask) end)
  end

  defp default_filter(path), do: Enum.any?(path, &String.ends_with?(&1, "kernel"))

  defp tensor_masks(candidates, sparsity) do
    Enum.map(candidates, fn {path, tensor} ->
      size = Nx.size(tensor)
      keep = size - round(sparsity * size)

      mask =
        tensor
        |> Nx.flatten()
        |> Nx.abs()
        |> keep_largest(keep)
        |> Nx.reshape(Nx.shape(tensor))

      {path, mask}
    end)
  end

  defp global_masks([], _sparsity), do: []

  defp global_masks(candidates, sparsity) do
    magnitudes = Enum.map(candidates, fn {_, tensor} -> Nx.abs(Nx.flatten(tensor)) end)
    total = Enum.reduce(magnitudes, 0, &(Nx.size(&1) + &2))
    keep = total - round(sparsity * total)
    global_mask = magnitudes |> Nx.concatenate() |> keep_largest(keep)

    {masks, _offset} =
      Enum.map_reduce(candidates, 0, fn {path, tensor}, offset ->
        size = Nx.size(tensor)

        mask =
          global_mask
          |> Nx.slice([offset], [size])
          |> Nx.reshape(Nx.shape(tensor))

        {{path, mask}, offset + size}
      end)

    masks
  end

  # Returns a {:u, 8} mask over a flat vector which keeps the `keep`
  # largest entries. Ties are broken by position, which keeps the exact
  # count and makes the result deterministic.
  defp keep_largest(flat, keep) do
    size = Nx.size(flat)

    cond do
      keep >= size ->
        Nx.broadcast(Nx.tensor(1, type: :u8), {size})

      keep <= 0 ->
        Nx.broadcast(Nx.tensor(0, type: :u8), {size})

      true ->
        flat
        |> Nx.argsort(direction: :desc, stable: true)
        |> Nx.argsort(stable: true)
        |> Nx.less(keep)
    end
  end

  @doc """
  Applies a pruning mask to the given model state or parameter map.

  Every leaf which has an entry in `mask` is replaced with the same tensor
  with zeros wherever the mask is `0`. Leaves without a mask entry and mask
  entries without a matching leaf are ignored. `mask` is typically produced
  by `magnitude_mask/3`.

  The first argument may be an `Axon.ModelState` or a nested parameter map
  such as the one returned by `Axon.ModelState.trainable_parameters/1`.

  ## Examples

      iex> params = %{"dense" => %{"kernel" => Nx.tensor([1.0, 2.0, 3.0]), "bias" => Nx.tensor([1.0])}}
      iex> mask = %{"dense" => %{"kernel" => Nx.tensor([1, 0, 1], type: :u8)}}
      iex> Axon.Pruning.apply_mask(params, mask)
      %{
        "dense" => %{
          "bias" => Nx.tensor([1.0]),
          "kernel" => Nx.tensor([1.0, 0.0, 3.0])
        }
      }

  """
  deftransform apply_mask(model_state_or_params, mask)

  deftransform apply_mask(%ModelState{} = model_state, mask) when is_map(mask) do
    update_in(model_state, [Access.key!(:data)], &mask_tree(&1, mask))
  end

  deftransform apply_mask(params, mask) when is_map(params) and is_map(mask) do
    mask_tree(params, mask)
  end

  @doc """
  Prunes the given model state.

  Computes a magnitude-based mask with `magnitude_mask/3` and applies it
  with `apply_mask/2`. Returns a tuple of the pruned model state and the
  mask. Accepts the same options as `magnitude_mask/3`.

  ## Examples

      {pruned_state, mask} = Axon.Pruning.prune(model_state, 0.5, scope: :tensor)

  """
  deftransform prune(%ModelState{} = model_state, sparsity, opts \\ []) do
    mask = magnitude_mask(model_state, sparsity, opts)
    {apply_mask(model_state, mask), mask}
  end

  @doc """
  Wraps an optimizer so that it never updates pruned parameters.

  `optimizer` is anything accepted by `Axon.Loop.trainer/4`: an atom
  naming an optimizer in `Polaris.Optimizers` or a tuple of
  `{init_fn, update_fn}`. `mask` is a pruning mask as returned by
  `magnitude_mask/3` or `prune/3`. The returned optimizer computes the
  regular updates and then zeroes them wherever the mask is `0`, so
  parameters which are zero when training starts stay exactly zero
  regardless of momentum or weight decay.

  Mask entries for parameters which are not being trained, such as
  frozen parameters, are ignored. The mask is kept in the optimizer
  state, so it lives on the same device as the rest of the training
  state and is included in checkpoints.

  ## Examples

      {pruned_state, mask} = Axon.Pruning.prune(model_state, 0.9)

      model
      |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Pruning.masked_optimizer(:adam, mask))
      |> Axon.Loop.run(data, pruned_state, epochs: 2)

  """
  def masked_optimizer(optimizer, mask) when is_map(mask) do
    # The optimizer init function is traced, so the mask is copied to the
    # binary backend, which can be embedded in the traced expression, and
    # placed in the optimizer state. From there on it is passed to the
    # update function as a regular argument.
    mask = Nx.backend_copy(mask, Nx.BinaryBackend)

    Polaris.Updates.stateful(
      optimizer_fns(optimizer),
      fn _params -> %{mask: mask} end,
      fn updates, %{mask: mask} = state, _params -> {mask_tree(updates, mask), state} end
    )
  end

  defp optimizer_fns(optimizer) when is_atom(optimizer) do
    if Code.ensure_loaded?(Polaris.Optimizers) and
         {optimizer, 0} in Polaris.Optimizers.__info__(:functions) do
      apply(Polaris.Optimizers, optimizer, [])
    else
      invalid_optimizer!(optimizer)
    end
  end

  defp optimizer_fns({init_fn, update_fn} = optimizer)
       when is_function(init_fn, 1) and is_function(update_fn, 3) do
    optimizer
  end

  defp optimizer_fns(optimizer), do: invalid_optimizer!(optimizer)

  defp invalid_optimizer!(optimizer) do
    raise ArgumentError,
          "invalid optimizer #{inspect(optimizer)}, expected an atom matching the name" <>
            " of an optimizer in Polaris.Optimizers or a tuple of {init_fn, update_fn}"
  end

  @doc """
  Returns the fraction of zero entries in each parameter of the model state.

  The result is a nested map mirroring the parameters of the model state
  with a float per parameter. Composite parameters such as the kernels of
  recurrent layers produce nested maps. Tied and quantized parameters are
  skipped, and state is not included.

  ## Examples

      Axon.Pruning.sparsity(pruned_state)
      #=> %{"dense_0" => %{"bias" => 0.0, "kernel" => 0.9}}

  """
  def sparsity(%ModelState{} = model_state) do
    model_state
    |> parameter_leaves()
    |> Enum.reduce(%{}, fn {path, tensor}, acc ->
      put_path(acc, path, ratio(zero_count(tensor), Nx.size(tensor)))
    end)
  end

  @doc """
  Returns the fraction of zero entries across all parameters of the model state.

  The fraction is computed over the total number of entries, so larger
  parameters weigh more than smaller ones. Parameters are counted as in
  `sparsity/1`. Returns `0.0` for a model state without parameters.

  ## Examples

      Axon.Pruning.global_sparsity(pruned_state)
      #=> 0.8625

  """
  def global_sparsity(%ModelState{} = model_state) do
    {zeros, total} =
      model_state
      |> parameter_leaves()
      |> Enum.reduce({0, 0}, fn {_path, tensor}, {zeros, total} ->
        {zeros + zero_count(tensor), total + Nx.size(tensor)}
      end)

    ratio(zeros, total)
  end

  defp zero_count(tensor) do
    tensor |> Nx.equal(0) |> Nx.sum() |> Nx.to_number()
  end

  defp ratio(_zeros, 0), do: 0.0
  defp ratio(zeros, total), do: zeros / total

  ## Tree helpers

  # Returns a list of {path, tensor} pairs for every tensor leaf
  # reachable from the parameters tree of the model state. Tied and
  # quantized parameters are skipped.
  defp parameter_leaves(%ModelState{data: data, parameters: parameters}) do
    collect_leaves(parameters, data, [])
  end

  defp collect_leaves(names, data, path) when is_list(names) and is_map(data) do
    Enum.flat_map(names, fn name -> data_leaves(Map.get(data, name), path ++ [name]) end)
  end

  defp collect_leaves(tree, data, path) when is_map(tree) and is_map(data) do
    Enum.flat_map(tree, fn {key, subtree} ->
      case data do
        %{^key => subdata} when is_map(subdata) and not is_struct(subdata) ->
          collect_leaves(subtree, subdata, path ++ [key])

        %{} ->
          []
      end
    end)
  end

  defp collect_leaves(_names, _data, _path), do: []

  defp data_leaves(%Nx.Tensor{} = tensor, path), do: [{path, tensor}]

  defp data_leaves(%_{}, _path), do: []

  defp data_leaves(map, path) when is_map(map) do
    Enum.flat_map(map, fn {key, value} -> data_leaves(value, path ++ [key]) end)
  end

  defp data_leaves(_other, _path), do: []

  defp mask_tree(tree, mask) do
    Enum.reduce(mask, tree, fn {key, mask_value}, acc ->
      case Map.fetch(acc, key) do
        {:ok, value} -> Map.put(acc, key, mask_value(value, mask_value))
        :error -> acc
      end
    end)
  end

  defp mask_value(%Nx.Tensor{} = tensor, %Nx.Tensor{} = mask), do: Nx.select(mask, tensor, 0)
  defp mask_value(%_{} = value, _mask), do: value

  defp mask_value(tree, mask) when is_map(tree) and is_map(mask) and not is_struct(mask),
    do: mask_tree(tree, mask)

  defp mask_value(value, _mask), do: value

  defp put_path(map, [key], value), do: Map.put(map, key, value)

  defp put_path(map, [key | rest], value) do
    Map.update(map, key, put_path(%{}, rest, value), &put_path(&1, rest, value))
  end
end
