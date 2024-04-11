import Optimisers
import Zygote
import Random
import BSON
import CSV
import Lux

using Lux
using Bijectors
using DataFrames
using LinearAlgebra
using Distributions
using DifferentialEquations

include("src/lib/population.jl");
include("src/lib/model.jl");
include("src/lib/error.jl");
include("src/lib/objectives.jl");
include("src/lib/compartment_models.jl");
include("src/lib/callbacks.jl");
include("src/lib/constrain.jl");
include("src/lib/dcm.jl");

# df = DataFrame(CSV.File("data/warfarin.csv"))
# df_group = groupby(df, :ID)

df = DataFrame(CSV.File("data/fviii_simulation.csv"))
df_group = groupby(df, :id)

indvs = Vector{AbstractIndividual}(undef, length(df_group))
for (i, group) in enumerate(df_group)
    # warfarin
    # x = Vector{Float32}(group[1, [:WEIGHT, :AGE, :SEX]])
    # ty = group[(group.DVID .== 1) .& (group.MDV .== 0), [:TIME, :DV]]
    # 𝐈 = Matrix{Float32}(group[group.MDV .== 1, [:TIME, :DOSE, :RATE, :DURATION]])
    # callback = generate_dosing_callback(𝐈)
    # indvs[i] = Individual(x, Float32.(ty.TIME), Float32.(ty.DV), callback; id = group.ID[1])
    # fviii
    x = Vector{Float32}(group[1, [:wt, :vwf]])
    ty = group[group.mdv .== 0, [:time, :dv]]
    𝐈 = Matrix{Float32}(group[group.mdv .== 1, [:time, :amt, :rate, :duration]])
    callback = generate_dosing_callback(𝐈; S1=1/1000.f0)
    indvs[i] = Individual(x, Float32.(ty.time), Float32.(ty.dv), callback; id = group.id[1])    
end
train_idxs = [428,463,265,348,374,364,112,357,181,259,483,498,287,342,410, 36,182,150,145,284,438,230,175,171, 16, 71,294,455, 78,265,450,317,109,342,470,425,340,455,231,159,413, 96,276,277,190,223,238, 36, 92, 471, 146, 376, 419, 498, 441, 282, 85, 170, 119, 434]
population = Population(indvs[train_idxs])

ann = Chain(
    # warfarin
    # Normalize([200, 100, 1]),
    # Dense(3, 16, swish), 
    # Dense(16, 4, softplus), 
    # fviii
    Normalize([150, 350]),
    Lux.Dense(2, 8, Lux.swish),
    Lux.Dense(8, 8, Lux.swish),
    Lux.Dense(8, 2, Lux.softplus, init_bias=Lux.ones32),
    AddGlobalParameters(4, [3, 4]; activation=Lux.softplus)
)

# model = DCM(two_comp_abs!, 3, ann; objective=VariationalELBO([1, 2]; approx=MonteCarlo(3)), dv_compartment=2)
model = DCM(two_comp!, 2, ann; objective=
    VariationalELBO(
        Additive(;init_f=init_sigma(init_dist=Normal(0.1f0, 0.03f0))), 
        [1, 2];
        init_prior=init_omega(;C_shape=10.f0)
    ), 
    dv_compartment=1) # FVIII

model.p
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
        println("Epoch $epoch: loss = $loss, omega = $(constrain(model.p).omega), ∇omega = $(grad_p.omega)")
        plt = Plots.scatter(population.y, getfield.(predict(model, population)[1], :u), label=nothing, color=:dodgerblue)
        # Plots.plot!(plt, [-1, 16], [-1, 16], color=:black)
        Plots.plot!(plt, [-0.5, 1.5], [-0.5, 1.5], color=:black) # fviii
        display(plt)
    end
    opt_state, p_opt = Optimisers.update(opt_state, model.p, grad_p)
    update!(model, p_opt)
    opt_state_phi, phi = Optimisers.update(opt_state_phi, phi, grad_phi)
end


include("src/lib/vi.jl")

for epoch in 1:2_000
    loss, back = Zygote.pullback((p, phi_) -> -partial_advi(model, population, p, phi_), model.p, phi);
    grad_p, grad_phi = back(1)
    if epoch == 1 || epoch % 10 == 0
        println("Epoch $epoch: loss = $loss, omega = $(constrain(model.p).omega), ∇omega = $(grad_p.omega)")
        plt = Plots.scatter(population.y, getfield.(predict(model, population)[1], :u), label=nothing, color=:dodgerblue)
        # Plots.plot!(plt, [-1, 16], [-1, 16], color=:black)
        Plots.plot!(plt, [-0.5, 1.5], [-0.5, 1.5], color=:black) # fviii
        display(plt)
    end
    opt_state, p_opt = Optimisers.update(opt_state, model.p, grad_p)
    update!(model, p_opt)
    opt_state_phi, phi = Optimisers.update(opt_state_phi, phi, grad_phi)
end


old_p = BSON.load("data/init_old_implementation.bson")[:p]
grad_old_p = Zygote.gradient(p -> -partial_advi_(model.ann, model.problem, population, p, model.p.st), old_p)