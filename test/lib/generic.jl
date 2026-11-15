import Optimisers
import Random
import Plots
import CSV

using Lux
using DataFrames
using DeepCompartmentModels

T = Float32
population = load_theophylline(T)

num_covariates = size(get_x(population), 1)
num_pk = 3

ann = Chain(
    Dense(num_covariates, 12, swish),
    Dense(12, 12, swish),
    Dense(12, num_pk, softplus)
)
dcm = DeepCompartmentModel(one_comp_abs!, ann; target = 2) # target the plasma compartment

rng = Random.default_rng()
objective = MSE()
ps, st = setup(objective, rng, dcm, T)

predict(dcm, population, ps, st)
objective(dcm, population, ps, st)

opt_state = Optimisers.setup(Optimisers.Adam(0.01), ps)

for epoch in 1:250
    println("Epoch $epoch, loss = $(objective(dcm, population, ps, st))")
    grad = gradient(objective, dcm, population, ps, st)
    opt_state, ps = Optimisers.update(opt_state, ps, grad)
end

yhat = predict(dcm, population, ps, st)

# GOF plot:
plt = Plots.scatter(get_y(population), yhat, color=:lightblue, label=nothing)
Plots.plot!(plt, [-0.5, 15], [-0.5, 15], color=:black, label=nothing)

@testset "YourPackageName.jl Tests" begin
    
    @testset "Testing with DataFrames" begin
        # 1. Create a dummy DataFrame using your test-only dependency
        df = DataFrame(A = 1:5, B = 6:10)
        
        # 2. Call a function from your local package code
        # result = YourPackageName.process_data(df)
        
        # 3. Test that your local code did what it was supposed to do
        @test nrow(df) == 5
        # @test result == expected_value
    end

    @testset "Other Component Tests" begin
        # You can test other parts of your local code here
        @test 1 + 1 == 2
    end
    
end