import OrdinaryDiffEq: ODECompositeSolution
import DeepCompartmentModels

using Test

@testset "dataset" begin
    @testset "_get_rate_over_t" begin
        # Should result in the correct changing rate over time
        I = [0 1000 60000 1/60; 10 500 30000 1/60]
        result = DeepCompartmentModels._get_rate_over_t(I)
        @test result == [0 60000; 1/60 0; 10 30000; 10 + 1/60 0]

        # Should sum the effect of multiple events
        I = [0 1000 60000 1/60; 5 2000 100 20; 10 500 30000 1/60]
        result = DeepCompartmentModels._get_rate_over_t(I)
        @test result == [0 60000; 1/60 0; 5 100; 10 30100; 10 + 1/60 100; 25 0]

        # Should also work when not in chronological order
        I = [10 500 30000 1/60; 0 1000 60000 1/60]
        result = DeepCompartmentModels._get_rate_over_t(I)
        @test result == [0 60000; 1/60 0; 10 30000; 10 + 1/60 0]
    end

    @testset "_generate_dosing_callbacks" begin
        # Should return a DiscreteCallback when supplied an event matrix
        I = [[0 1000 60000 1/60; 10 500 30000 1/60]]
        result = DeepCompartmentModels._generate_dosing_callbacks(I)
        @test length(result) == 1
        @test typeof(first(result)) <: DeepCompartmentModels.DiscreteCallback

        # Iterates through the events and returns n callbacks
        I = [[0 1000 60000 1/60], [0 500 30000 1/60], [0 250 15000 1/60]]
        result = DeepCompartmentModels._generate_dosing_callbacks(I)
        @test length(result) == 3

        # Should adapt the rate and dose when setting S1
        factor = 1/1000
        I = [[0 1000 60000 1/60; 10 500 30000 1/60]]
        result = first(DeepCompartmentModels._generate_dosing_callbacks(I; S1=factor))
        @test result.affect!.rates == [60000 * factor, 0., 30000 * factor, 0.]
    end
    
    @testset "normalize" begin
        # Should normalize a float based on a minimum and maximum value
        @test DeepCompartmentModels.normalize(50, 0, 100) == 0.5

        # Should work for vectors
        @test DeepCompartmentModels.normalize([25, 50, 75], 0, 100) == [0.25, 0.5, 0.75]

        # Should work for matrices
        result = DeepCompartmentModels.normalize([25 50 75; 12 24 36], [0, 0], [100, 48])
        @test result == [0.25 0.5 0.75; 0.25 0.5 0.75]

        # When not provided a min/max will calculate them from the passed vector
        input = collect(1:5)
        result, scale = DeepCompartmentModels.normalize(input)
        @test result == [0.0, 0.25, 0.5, 0.75, 1.0]
        @test scale == ([minimum(input)], [maximum(input)])
        
        # When not provided a min/max will calculate them from the passed matrix
        input = [25 12 8; 50 24 16; 75 36 24]
        result, scale = DeepCompartmentModels.normalize(input)
        @test result == [0.0 0.0 0.0; 0.5 0.5 0.5; 1.0 1.0 1.0]
        @test scale == (minimum(input, dims=1), maximum(input, dims=1))

        # Accepts DataFrames
        input = DeepCompartmentModels.DataFrame([25 12 8; 50 24 16; 75 36 24], :auto)
        result, scale = DeepCompartmentModels.normalize(input)
        @test typeof(result) <: Matrix
        @test result == [0 0 0; 0.5 0.5 0.5; 1 1 1]
        @test scale == (minimum(Matrix(input), dims=1), maximum(Matrix(input), dims=1))

        # Can be called with a previous scale
        input = [25 12 8; 50 24 16; 75 36 24]
        scale = ([0. 0. 0.], [75. 48. 48.])
        result = DeepCompartmentModels.normalize(input, scale)
        @test result == [1/3 0.25 1/6; 2/3 0.5 1/3; 1. 0.75 0.5] 
    end

    @testset "normalize⁻¹" begin
        # Should revert the initialization for a Float
        @test DeepCompartmentModels.normalize⁻¹(0.5, 0, 100) == 50

        # Should revert the initialization for a Vector
        @test DeepCompartmentModels.normalize⁻¹([0.25, 0.5, 0.75], 0, 100) == [25, 50, 75]

        # Should work for matrices
        result = DeepCompartmentModels.normalize⁻¹([0.25 0.5 0.75; 0.25 0.5 0.75], [0, 0], [100, 48])
        @test result == [25 50 75; 12 24 36]

        # Works with populations
        input = [1/3 0.25 1/6; 2/3 0.5 1/3; 1. 0.75 0.5]
        scale = ([0. 0. 0.], [75. 48. 48.])
        cb = DeepCompartmentModels.DiscreteCallback(() -> nothing, () -> nothing)
        pop = DeepCompartmentModels.Population(input, [[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]], [cb], scale)
        result = DeepCompartmentModels.normalize⁻¹(pop)
        @test typeof(result) <: Matrix
        @test result == [25 12 8; 50 24 16; 75 36 24]

        # Has an alias normalize_inv which works for floats
        @test DeepCompartmentModels.normalize⁻¹(0.5, 0, 100) == DeepCompartmentModels.normalize_inv(0.5, (0, 100))
        
        # Has an alias normalize_inv which works for vectors
        @test DeepCompartmentModels.normalize⁻¹([0.25, 0.5, 0.75], 0, 100) == DeepCompartmentModels.normalize_inv([0.25, 0.5, 0.75], (0, 100))

        # Has an alias normalize_inv which works for populations
        input = [1/3 0.25 1/6; 2/3 0.5 1/3; 1. 0.75 0.5]
        scale = ([0. 0. 0.], [75. 48. 48.])
        cb = DeepCompartmentModels.DiscreteCallback(() -> nothing, () -> nothing)
        pop = DeepCompartmentModels.Population(input, [[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]], [cb], scale)
        @test DeepCompartmentModels.normalize⁻¹(pop) == DeepCompartmentModels.normalize_inv(pop)
    end

    @testset "create_split" begin
        # takes an integer and returns a random train and test set according to the ratio size.
        @test length.(DeepCompartmentModels.create_split(10; ratio=0.7)) == (7, 3)

        # Errors if ratio <= 0 or >= 1
        @test_throws ErrorException DeepCompartmentModels.create_split(10; ratio=0.)
        @test_throws ErrorException DeepCompartmentModels.create_split(10; ratio=-1.)
        @test_throws ErrorException DeepCompartmentModels.create_split(10; ratio=1.)
        @test_throws ErrorException DeepCompartmentModels.create_split(10; ratio=1124.)

        # Works on populations
        input = rand(100, 4)
        scale = (0., 1.)
        cb = DeepCompartmentModels.DiscreteCallback(() -> nothing, () -> nothing)
        pop = DeepCompartmentModels.Population(input, [rand(4) for i in 1:100], [rand(4) for i in 1:100], [cb for i in 1:100], scale)
        result = DeepCompartmentModels.create_split(pop; ratio=0.7)
        @test length.(result) == (70, 30)
        @test typeof(result[1]) == DeepCompartmentModels.Population
        @test typeof(result[2]) == DeepCompartmentModels.Population
    end
end

@testset "population" begin
    
end

@testset "model" begin
    @testset "DCM" begin
        # Has constructor that accepts ODEProblem and ann
        prob = DeepCompartmentModels.ODEProblem(DeepCompartmentModels.one_comp!, zeros(1), (0., 1.))
        ann = DeepCompartmentModels.Flux.Chain(DeepCompartmentModels.Flux.Dense(4, 1))

        model = DeepCompartmentModels.DCM(prob, ann)
        @test typeof(model) == DeepCompartmentModels.DCM
        # Has constructor that uses a ode_f, ann, and number of compartments
        model = DeepCompartmentModels.DCM(DeepCompartmentModels.one_comp!, ann, 1)
        @test typeof(model) == DeepCompartmentModels.DCM
        # When called using x, returns a prediction for ζ
        @test length(model(rand(4))) == 1
        @test typeof(model(rand(4))) == Vector{Float64}
        @test length(model(rand(4, 4))) == 4
        @test typeof(model(rand(4, 4))) == Matrix{Float64}
        # Can also be called using a population or individual

        # Can be copied using copy()

    end

    @testset "predict" begin
        # can be called with an Individual and predicts concentration
        𝐈 = [0 10 10*60 1/60]
        cb = first(DeepCompartmentModels._generate_dosing_callbacks([𝐈]))
        t = sort(rand(4)) 
        
        individual = DeepCompartmentModels.Individual(rand(4), rand(4), t, cb, 0.)
        prob = DeepCompartmentModels.ODEProblem(DeepCompartmentModels.one_comp!, zeros(1), (0., 1.))
        ann = DeepCompartmentModels.Flux.Chain(DeepCompartmentModels.Flux.Dense(4, 2))
        model = DeepCompartmentModels.DCM(prob, ann)

        pred = DeepCompartmentModels.predict(model, individual)
        @test typeof(pred) <: ODECompositeSolution
        @test length(pred.u) == length(t)
        @test all(isapprox.(pred.t, t))
        
        # Can be set to interpolate
        pred = DeepCompartmentModels.predict(model, individual; interpolating=true)
        @test typeof(pred) <: ODECompositeSolution
        @test length(pred.u) != length(t) 
        @test individual.callback.save_positions == BitVector(zeros(2)) # save_positions is set to true for predict(), but set to false after calling solve()

        # the tspan can be customized
        pred = DeepCompartmentModels.predict(model, individual; tspan=(0, 0.2))
        @test typeof(pred) <: ODECompositeSolution
        @test length(pred.t) == sum(t .<= .2)

        # Can also return the concentration in all compartments
        prob = DeepCompartmentModels.ODEProblem(DeepCompartmentModels.two_comp!, zeros(2), (0., 1.))
        ann = DeepCompartmentModels.Flux.Chain(DeepCompartmentModels.Flux.Dense(4, 4))
        model = DeepCompartmentModels.DCM(prob, ann)
        pred_not_full = DeepCompartmentModels.predict(model, individual)
        pred_full = DeepCompartmentModels.predict(model, individual; full=true)
        @test size(pred_not_full.u) == (4,)
        @test size(hcat(pred_full.u...)) == (length(prob.u0), 4)

        # Can be called on a population to calculate the concentration for each individual iteratively.
        𝐈 = [[0 10 10*60 1/60], [0 10 10*60 1/60]]
        cbs = DeepCompartmentModels._generate_dosing_callbacks(𝐈)
        population = DeepCompartmentModels.Population(rand(2,4), [rand(4), rand(4)], fill(t, 2), cbs, (zeros(4), ones(4)))
        pred = DeepCompartmentModels.predict(model, population)
        @test length(pred) == length(population)
        @test typeof(first(pred)) <: ODECompositeSolution
    end

    @testset "predict_adjoint" begin
        # Prep
        𝐈 = [0 10 10*60 1/60]
        cb = first(DeepCompartmentModels._generate_dosing_callbacks([𝐈]))
        t = sort(rand(4)) 
        
        individual = DeepCompartmentModels.Individual(rand(4), rand(4), t, cb, 0.)
        prob = DeepCompartmentModels.ODEProblem(DeepCompartmentModels.one_comp!, zeros(1), (0., 1.))
        ann = DeepCompartmentModels.Flux.Chain(DeepCompartmentModels.Flux.Dense(4, 2))
        model = DeepCompartmentModels.DCM(prob, ann)
        
        # Calculates the concentration for the measurement compartment when passed a set of weights for the ann
        pred = DeepCompartmentModels.predict(model, individual)
        pred_adjoint = DeepCompartmentModels.predict_adjoint(model, individual, model.weights)
        @test all(isapprox.(pred_adjoint, pred.u))
    end

    @testset "pullback" begin
        # Prep
        𝐈 = [0 10 10*60 1/60]
        cb = first(DeepCompartmentModels._generate_dosing_callbacks([𝐈]))
        t = sort(rand(4)) 
        
        individual = DeepCompartmentModels.Individual(rand(4), rand(4), t, cb, 0.)
        prob = DeepCompartmentModels.ODEProblem(DeepCompartmentModels.one_comp!, zeros(1), (0., 1.))
        ann = DeepCompartmentModels.Flux.Chain(DeepCompartmentModels.Flux.Dense(4, 2))
        model = DeepCompartmentModels.DCM(prob, ann)
        # Should succesfully calculate the gradient for a single individual
        loss, back = DeepCompartmentModels.pullback((𝑤) -> DeepCompartmentModels.mse(model, individual, 𝑤), model.weights)
        ∇𝑤 = first(back(1.0))
        @test length(∇𝑤) == length(model.weights)
        # succesfully updates weights when calling update! using an optimizer
        old_weights = copy(model.weights)
        DeepCompartmentModels.Flux.update!(DeepCompartmentModels.Flux.ADAM(1e-3), model.weights, ∇𝑤)
        @test all(isapprox.(model.weights, old_weights)) == false
    end

    @testset "fit!" begin
        # Prep
        𝐈 = [[0 10 10*60 1/60], [0 10 10*60 1/60]]
        cbs = DeepCompartmentModels._generate_dosing_callbacks(𝐈)
        t = sort(rand(4)) 

        population = DeepCompartmentModels.Population(rand(2,4), [rand(4), rand(4)], fill(t, 2), cbs, (zeros(4), ones(4)))
        prob = DeepCompartmentModels.ODEProblem(DeepCompartmentModels.one_comp!, zeros(1), (0., 1.))
        ann = DeepCompartmentModels.Flux.Chain(DeepCompartmentModels.Flux.Dense(4, 2))
        model = DeepCompartmentModels.DCM(prob, ann)

        # Can be used as a shorthand to update weights of the model
        old_weights = copy(model.weights)
        optimizer = DeepCompartmentModels.Flux.ADAM(1e-3)
        DeepCompartmentModels.fit!(model, population, optimizer; iterations=1)
        @test all(isapprox.(model.weights, old_weights)) == false
    
        # Can be provided callbacks that are evaluated each iteration
    end
end