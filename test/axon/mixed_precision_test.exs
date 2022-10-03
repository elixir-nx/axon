# defmodule MixedPrecisionTest do
#   use Axon.Case, async: true

#   alias Axon.MixedPrecision.Policy
#   alias Axon.MixedPrecision, as: AMP
#   alias Axon.Loop

#   describe "creation and application" do
#     test "create policy" do
#       assert %Policy{params: {:f, 32}, compute: {:bf, 16}, output: {:f, 32}} =
#                AMP.create_policy(compute: {:bf, 16})

#       assert %Policy{params: {:bf, 16}, compute: {:f, 32}, output: {:bf, 16}} =
#                AMP.create_policy(params: {:bf, 16}, output: {:bf, 16})
#     end

#     test "apply_policy" do
#       model =
#         Axon.input("input", shape: {nil, 784})
#         |> Axon.dense(128)
#         |> Axon.batch_norm()
#         |> Axon.dense(10)

#       policy = AMP.create_policy(compute: {:bf, 16})

#       assert %Axon{
#                op: :dense,
#                parent: [
#                  %Axon{
#                    op: :batch_norm,
#                    parent: [%Axon{op: :dense, policy: %Policy{compute: {:bf, 16}}}],
#                    policy: %Policy{compute: {:f, 32}}
#                  }
#                ],
#                policy: %Policy{compute: {:bf, 16}}
#              } = AMP.apply_policy(model, policy, except: [:batch_norm])
#     end
#   end

#   describe "compilation" do
#     # TODO(seanmor5): Now that everything else has moved, maybe this
#     # belongs in a train test or elsewhere
#     test "correctly maintains parameter type after train step" do
#       model =
#         Axon.input("input", shape: {nil, 32})
#         |> Axon.dense(2, name: "dense1")
#         |> Axon.batch_norm(name: "batch_norm")
#         |> Axon.dense(1, activation: :sigmoid, name: "dense2")

#       policy = AMP.create_policy(params: {:bf, 16})

#       mp_model = AMP.apply_policy(model, policy, except: [:batch_norm])

#       %Loop{init: init_fn, step: step_fn} =
#         Axon.Loop.trainer(mp_model, :binary_cross_entropy, Axon.Optimizers.sgd(0.01))

#       v1 = Nx.random_uniform({1, 32})
#       v2 = Nx.random_uniform({1, 1})

#       pstate =
#         apply(Nx.Defn.jit(step_fn), [
#           {v1, v2},
#           init_fn.({v1, v2}, %{})
#         ])

#       params = pstate[:model_state]

#       assert Nx.type(params["dense1"]["kernel"]) == {:bf, 16}
#       assert Nx.type(params["dense1"]["bias"]) == {:bf, 16}
#       assert Nx.type(params["dense2"]["kernel"]) == {:bf, 16}
#       assert Nx.type(params["dense2"]["bias"]) == {:bf, 16}
#       assert Nx.type(params["batch_norm"]["gamma"]) == {:f, 32}
#       assert Nx.type(params["batch_norm"]["beta"]) == {:f, 32}
#     end
#   end

#   describe "inspection" do
#     test "works" do
#       policy = AMP.create_policy()

#       assert inspect(policy) == """
#              p=f32 c=f32 o=f32\
#              """
#     end
#   end
# end
