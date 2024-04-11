import Optimisers
import Zygote
import Random
import Plots
import CSV
import Lux

include("src/lib/population.jl");
include("src/lib/model.jl");
include("src/lib/error.jl");
include("src/lib/objectives.jl");
include("src/lib/compartment_models.jl");
include("src/lib/callbacks.jl");
include("src/lib/constrain.jl");
include("src/lib/dcm.jl");

using Lux
using DataFrames
using LinearAlgebra
using DifferentialEquations

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
)

model = DCM(two_comp_abs!, 3, ann; dv_compartment=2)
objective(model, population)

opt = Optimisers.ADAM(1e-2)
opt_state = Optimisers.setup(opt, model.p)

for epoch in 1:2_000
    loss, back = Zygote.pullback(p -> objective(model, population, p), model.p);
    grad = first(back(1))
    if epoch == 1 || epoch % 50 == 0
        println("Epoch $epoch: loss = $loss")
    end
    opt_state, p_opt = Optimisers.update(opt_state, model.p, grad)
    update!(model, p_opt)
end

model.ann(population.x, model.p.weights, model.p.st)[1]

i = rand(1:length(population))
sol, st = predict(model, population[i]; interpolate=true);
Plots.plot(sol, color=:black)
Plots.scatter!(population[i].t, population[i].y)

model_LL = DCM(two_comp_abs!, 3, ann; dv_compartment=2, objective=LogLikelihood(Combined()))

opt_LL = Optimisers.ADAM(1e-2)
opt_state_LL = Optimisers.setup(opt_LL, model_LL.p)
for epoch in 1:2_000
    loss, back = Zygote.pullback(p -> objective(model_LL, population, p), model_LL.p);
    grad = first(back(1))
    if epoch == 1 || epoch % 50 == 0
        println("Epoch $epoch: loss = $loss")
    end
    opt_state_LL, p_opt = Optimisers.update(opt_state_LL, model_LL.p, grad)
    update!(model_LL, p_opt)
end

i = rand(1:length(population))
sol, st = predict(model_LL, population[i]; interpolate=true);
Plots.plot(sol.t, sol.u, ribbon=std(model_LL, sol.u), color=:black, fillalpha=0.15)
Plots.scatter!(population[i].t, population[i].y)
