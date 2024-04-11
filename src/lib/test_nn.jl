import Optimisers
import Zygote
import Plots
import CSV

include("src/lib/nn.jl");

using Lux
using DataFrames
using LinearAlgebra
using DifferentialEquations

df = DataFrame(CSV.File("data/warfarin.csv"))
df_group = groupby(df, :ID)

indvs = Vector{AbstractIndividual}(undef, length(df_group))
for (i, group) in enumerate(df_group)
    x = Vector{Float32}(group[1, [:WEIGHT, :AGE, :SEX, :DOSE]])
    ty = group[(group.DVID .== 1) .& (group.MDV .== 0), [:TIME, :DV]]
    indvs[i] = Individual(x, Float32.(ty.TIME), Float32.(ty.DV), CallbackSet(); id = group.ID[1])
end
population = Population(indvs)

ann = Chain(
    Normalize([200, 100, 1, 200, 144]),
    Dense(5, 16, swish), 
    Chain(Dense(16, 4, swish), 
    Dense(4, 1, softplus))
)

model_SSE = SNN(ann)
objective(model_SSE, population)

opt = Optimisers.ADAM(1e-2)
opt_state = Optimisers.setup(opt, model_SSE.p)

for epoch in 1:10_000
    loss, back = Zygote.pullback(p -> objective(model_SSE, population, p), model_SSE.p);
    grad = first(back(1))
    if epoch == 1 || epoch % 500 == 0
        println("Epoch $epoch: loss = $loss")
    end
    opt_state, p_opt = Optimisers.update(opt_state, model_SSE.p, grad)
    update!(model_SSE, p_opt)
end

i = rand(1:length(population))
pred, st = predict(model_SSE, population[i]; saveat=0:144.)
Plots.plot(pred, color=:black)
Plots.scatter!(population[i].t, population[i].y)


model_LL = SNN(ann; objective=LogLikelihood(Combined()))
objective(model_LL, population)

opt_LL = Optimisers.ADAM(1e-2)
opt_state_LL = Optimisers.setup(opt_LL, model_LL.p)

for epoch in 1:10_000
    loss, back = Zygote.pullback(p -> objective(model_LL, population, p), model_LL.p);
    grad = first(back(1))
    if epoch == 1 || epoch % 500 == 0
        println("Epoch $epoch: loss = $loss")
    end
    opt_state_LL, p_opt = Optimisers.update(opt_state_LL, model_LL.p, grad)
    update!(model_LL, p_opt)
end

i = rand(1:length(population))
pred, st = predict(model_LL, population[i]; saveat=0:144.)
Plots.plot(pred, color=:black, ribbon=std(model_LL, pred), fillalpha=0.1)
Plots.scatter!(population[i].t, population[i].y)