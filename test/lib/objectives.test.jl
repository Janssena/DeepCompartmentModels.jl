using LinearAlgebra
import Zygote

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

@testset "CustomError distribution hook" begin
    gaussian_model = (prediction, ps, st; kwargs...) ->
        MvNormal(prediction, Diagonal(fill(DeepCompartmentModels.softplus(only(ps.σ))^2,
                                           length(prediction))))
    error = CustomError([0.2]; model = gaussian_model)
    ps = (σ = [DeepCompartmentModels.invsoftplus(0.2)],)
    prediction = [1.0, 2.0, 3.0]

    dist = make_dist(error, prediction, ps)
    @test dist isa MvNormal
    @test mean(dist) == prediction
    @test diag(var(error, prediction, ps, NamedTuple())) ≈ fill(0.2^2, 3)

    invalid = CustomError([0.2]; model = (args...; kwargs...) -> 1.0)
    @test_throws ArgumentError make_dist(invalid, prediction, ps)
    @test_throws ErrorException make_dist(CustomError([0.2]), prediction, ps)
end

@testset "CustomError log-transform-both-sides" begin
    function ltbs_model(prediction, ps, st; kwargs...)
        all(>(zero(eltype(prediction))), prediction) || throw(DomainError(
            minimum(prediction), "LTBS requires strictly positive predictions."))
        sigma = DeepCompartmentModels.softplus(only(ps.σ))
        return product_distribution(LogNormal.(log.(prediction), sigma))
    end

    error = CustomError([0.3]; model = ltbs_model)
    raw_sigma = DeepCompartmentModels.invsoftplus(0.3)
    ps = (σ = [raw_sigma],)
    prediction = [0.25, 1.0, 10.0]
    observation = [0.3, 0.8, 12.0]

    dist = make_dist(error, prediction, ps)
    expected = sum(logpdf.(LogNormal.(log.(prediction), 0.3), observation))
    @test logpdf(dist, observation) ≈ expected
    @test all(rand(dist, 100) .> 0)

    expected_variance = prediction.^2 .* exp(0.3^2) .* expm1(0.3^2)
    @test diag(var(error, prediction, ps, NamedTuple())) ≈ expected_variance

    gradient = Zygote.gradient(raw ->
        logpdf(make_dist(error, prediction, (σ = [raw],)), observation), raw_sigma)[1]
    @test isfinite(gradient)
    @test_throws DomainError make_dist(error, [0.0, 1.0], ps)
end
