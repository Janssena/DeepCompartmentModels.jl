import Optimisers
import Zygote
import Random
import Plots
import BSON
import CSV
import Lux

using Lux
using Bijectors
using DataFrames
using LinearAlgebra
using Distributions
using DistributionsAD
using DifferentialEquations

include("src/lib/population.jl");
include("src/lib/model.jl");
include("src/lib/error.jl");
include("src/lib/objectives.jl");
include("src/lib/compartment_models.jl");
include("src/lib/callbacks.jl");
include("src/lib/constrain.jl");
include("src/lib/dcm.jl");

df = DataFrame(CSV.File("data/warfarin.csv"))
df_group = groupby(df, :ID)

indvs = Vector{AbstractIndividual}(undef, length(df_group))
for (i, group) in enumerate(df_group)
    x = Vector{Float32}(group[1, [:WEIGHT, :AGE, :SEX]])
    ty = group[(group.DVID .== 1) .& (group.MDV .== 0), [:TIME, :DV]]
    𝐈 = Matrix{Float32}(group[group.MDV .== 1, [:TIME, :DOSE, :RATE, :DURATION]])
    callback = generate_dosing_callback(𝐈)
    indvs[i] = Individual(x, Float32.(ty.TIME), Float32.(ty.DV), callback; id = group.ID[1])
end
population = Population(indvs)

ann = Chain(
    Normalize([200, 100, 1]),
    Dense(3, 16, swish), 
    Dense(16, 4, softplus), 
    # AddGlobalParameters(4, [3, 4]; activation=Lux.softplus)
)

model = DCM(two_comp_abs!, 3, ann; objective=VariationalELBO([2]), dv_compartment=2)

phi = model.objective.init_phi(model, population)
objective(model, population, model.p, phi) # TODO: Reduce allocations

opt = Optimisers.ADAM(0.01)
opt_state = Optimisers.setup(opt, model.p)
opt_phi = Optimisers.ADAM(0.01)
opt_state_phi = Optimisers.setup(opt_phi, phi)

# for epoch in 1:200
#     loss, back = Zygote.pullback(p -> objective(SSE(), model, population, (weights = p.weights, st = p.st, error = p.error)), model.p);
#     grad_p = first(back(1))
#     if epoch == 1 || epoch % 50 == 0
#         println("Epoch $epoch: loss = $loss, omega = $(constrain(model.p).omega)")
#         plt = Plots.scatter(population.y, getfield.(predict(model, population)[1], :u), label=nothing, color=:dodgerblue)
#         # Plots.plot!(plt, [-1, 16], [-1, 16], color=:black)
#         Plots.plot!(plt, [-0.5, 1.5], [-0.5, 1.5], color=:black) # fviii
#         display(plt)
#     end
#     opt_state, p_opt = Optimisers.update(opt_state, model.p, grad_p)
#     update!(model, p_opt)
# end

for epoch in 1:2_000
    loss, back = Zygote.pullback((p, phi_) -> objective(model, population, p, phi_), model.p, phi);
    grad_p, grad_phi = back(1)
    if epoch == 1 || epoch % 10 == 0
        println("Epoch $epoch: loss = $loss")
        plt = Plots.scatter(population.y, getfield.(predict(model, population)[1], :u), label=nothing, color=:dodgerblue)
        Plots.plot!(plt, [-1, 16], [-1, 16], color=:black)
        display(plt)
    end
    opt_state, p_opt = Optimisers.update(opt_state, model.p, grad_p)
    update!(model, p_opt)
    opt_state_phi, phi = Optimisers.update(opt_state_phi, phi, grad_phi)
end


