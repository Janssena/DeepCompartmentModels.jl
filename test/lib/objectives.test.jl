using LinearAlgebra

@testset "_logpdf for objectives" begin
    error = AdditiveError()
    ps = (σ = [0.1],)
    yhat = [rand(3), rand(4), rand(2), rand(1)]
    y = [rand(3), rand(4), rand(2), rand(1)]

    population_dist = make_dist(error, yhat, ps)
    @test DeepCompartmentModels._logpdf(population_dist, y) ==
          sum(logpdf.(population_dist, y))

    individual_dist = make_dist(error, yhat[1], ps)
    @test DeepCompartmentModels._logpdf(individual_dist, y[1]) ==
          logpdf(individual_dist, y[1])
end

@testset "variational distributions" begin
    n_individuals = 4
    n_random_effects = 2
    means = [zeros(n_random_effects) for _ in 1:n_individuals]
    factors = [LowerTriangular(0.2I(n_random_effects)) for _ in 1:n_individuals]
    unconstrained_scales = [fill(-2.0, n_random_effects) for _ in 1:n_individuals]

    full_rank = getq((μ = means, L = factors))
    mean_field = getq((μ = means, σ = unconstrained_scales))

    @test length(full_rank) == n_individuals
    @test length(mean_field) == n_individuals
    @test all(length.(full_rank) .== n_random_effects)
    @test all(length.(mean_field) .== n_random_effects)

    eta = [rand(n_random_effects) for _ in 1:n_individuals]
    @test DeepCompartmentModels._logpdf(full_rank, eta) == sum(logpdf.(full_rank, eta))
    @test DeepCompartmentModels._logpdf(mean_field, eta) == sum(logpdf.(mean_field, eta))
end
