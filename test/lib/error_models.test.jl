using LinearAlgebra

@testset "Gaussian residual-error variances" begin
    prediction = Float64[0.5, 2.0, 8.0]

    additive_ps = (σ = [DeepCompartmentModels.invsoftplus(0.4)],)
    @test diag(var(AdditiveError(), prediction, additive_ps, NamedTuple())) ≈
          fill(0.4^2, length(prediction))

    proportional_ps = (σ = [DeepCompartmentModels.invsoftplus(0.2)],)
    proportional_variance = diag(var(
        ProportionalError(), prediction, proportional_ps, NamedTuple(); eps = 0.0))
    @test proportional_variance ≈ (0.2 .* prediction).^2

    combined_ps = (σ = DeepCompartmentModels.invsoftplus.([0.4, 0.2]),)
    independent = diag(var(
        CombinedError(), prediction, combined_ps, NamedTuple()))
    dependent = diag(var(
        CombinedError(; dependent = true), prediction, combined_ps, NamedTuple()))

    @test independent ≈ 0.4^2 .+ (0.2 .* prediction).^2
    @test dependent ≈ (0.4 .+ 0.2 .* prediction).^2
    @test all(dependent .>= independent)
end

@testset "error setup and validation" begin
    @test_throws ErrorException AdditiveError([0.1, 0.2])
    @test_throws ErrorException ProportionalError([0.1, 0.2])
    @test_throws ErrorException CombinedError([0.1])
    @test_throws ErrorException CombinedError([0.1, 0.2, 0.3])

    @test DeepCompartmentModels.setup(AdditiveError(0.3), nothing).σ ≈
          DeepCompartmentModels.invsoftplus.([0.3])
    @test DeepCompartmentModels.setup(CustomError([0.3]; model = identity), nothing).σ ≈
          DeepCompartmentModels.invsoftplus.([0.3])
end
