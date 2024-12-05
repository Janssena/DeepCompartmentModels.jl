using DistributionsAD
using LinearAlgebra

@testset "_logpdf for objectives" begin
    # setup
    M = 3
    error = AdditiveError()
    ps = (error = (σ = [0.1], ), )
    ŷ_pop = [rand(3), rand(4), rand(2), rand(1)]
    ŷ_pop_mc = [[rand(3), rand(4), rand(2), rand(1)] for _ in 1:M]
    ŷ_indv_mc = [rand(3) for _ in 1:M]
    y = [rand(3), rand(4), rand(2), rand(1)]
    # AbstractModel
    # ŷ = rand(10)
    # dist = make_dist(error, ŷ, ps)
    # y₁ = rand(10)
    # y₂ = [rand(3), rand(4), rand(2), rand(1)] → TODO: how to get pred in this shape for StandardNeuralNetwork?...

    # @test DeepCompartmentModels._logpdf(dist, y₁) == logpdf(dist, y₁)
    # @test DeepCompartmentModels._logpdf(dist, y₂) == sum(logpdf(dist, reduce(vcat, y₂)))

    # AbstractDEModel
    # Population prediction with MonteCarlo samples
    dist₁ = make_dist(error, ŷ_pop_mc, ps)
    @test length(DeepCompartmentModels._logpdf(dist₁, y)) == M
    @test all(isapprox.(DeepCompartmentModels._logpdf(dist₁, y), [sum(logpdf.(dist₁[i], y)) for i in eachindex(dist₁)]))

    # Population prediction
    dist₂ = make_dist(error, ŷ_pop, ps)
    @test DeepCompartmentModels._logpdf(dist₂, y) == sum(logpdf.(dist₂, y))

    # MonteCarlo sample for individual
    dist₃ = make_dist(error, ŷ_indv_mc, ps)
    @test length(DeepCompartmentModels._logpdf(dist₃, y[1])) == M
    @test DeepCompartmentModels._logpdf(dist₃, y[1]) == logpdf.(dist₃, (y[1], ))
    
    # Individual prediction
    idx = 1
    dist₄ = make_dist(error, ŷ_pop[idx], ps)
    @test DeepCompartmentModels._logpdf(dist₄, y[idx]) == logpdf(dist₄, y[idx])
end

@testset "logprior" begin
    M = 3
    omega_dist₁ = MvNormal(zeros(2), [0.1 0; 0 0.1])
    omega_dist₂ = Normal(0., 0.1)
    η₁ = rand(2, 10)
    η₂ = rand(10)
    # with Monte Carlo samples
    η₃ = [rand(2, 10) for _ in 1:M]
    η₄ = [rand(10) for _ in 1:M]

    # _get_prior
    @test DeepCompartmentModels._get_prior(Symmetric(omega_dist₁.Σ.mat)) isa TuringDenseMvNormal
    @test DeepCompartmentModels._get_prior(0.1) isa Normal
    
    # Multivariate eta for population
    @test DeepCompartmentModels._logpdf(omega_dist₁, η₁) == sum(logpdf(omega_dist₁, η₁))
    @test DeepCompartmentModels.logprior(omega_dist₁, η₁) == sum(logpdf(omega_dist₁, η₁))
    # Univariate eta for population
    @test DeepCompartmentModels._logpdf(omega_dist₂, η₂) == sum(map(Base.Fix1(logpdf, omega_dist₂), η₂))
    @test DeepCompartmentModels.logprior(omega_dist₂, η₂) == sum(map(Base.Fix1(logpdf, omega_dist₂), η₂))
    
    # Monte Carlo samples of eta for population
    # Multivariate eta
    @test all(isapprox.(DeepCompartmentModels.logprior(omega_dist₁, η₃), sum.([logpdf(omega_dist₁, η₃[i]) for i in eachindex(η₃)])))
    # Univariate eta
    @test all(isapprox.(DeepCompartmentModels.logprior(omega_dist₂, η₄), sum.([logpdf(omega_dist₂, η₄[i]) for i in eachindex(η₄)])))

    # How will this work for individuals? We need to keep the original type intact
end

@testset "logq" begin
    N = 10
    M = 3
    phi_fr = (μ = [zeros(2) for _ in 1:N], L = fill(LowerTriangular([0.316228 0; 0 0.316228]), N)) # full-rank multivariate omega
    phi_mf = (μ = [zeros(2) for _ in 1:N], σ = [fill(0.1, 2) for _ in 1:N]) # mean-field multivariate omega
    phi_uni = (μ = zeros(N), σ = fill(0.1, N)) # univariate omega
    
    η_mv = rand(2, N)
    η_uni = rand(N)
    η_mv_mc = [rand(2, N) for _ in 1:M]
    η_uni_mc = [rand(N) for _ in 1:M]

    q_fr = getq(VariationalELBO([1,2], FullRank()), phi_fr)
    q_mf = getq(VariationalELBO([1,2], MeanField()), phi_mf)
    q_uni = getq(VariationalELBO([1]), phi_uni)

    @test q_fr isa AbstractVector{<:TuringDenseMvNormal}
    @test q_mf isa AbstractVector{<:TuringDiagMvNormal}
    @test q_uni isa TuringDiagMvNormal
    # When eta is univariate, it shouldn't matter whether we are using Meanfield or FullRank
    @test getq(VariationalELBO([1], MeanField()), phi_uni) == getq(VariationalELBO([1], FullRank()), phi_uni)
    
    # Multivariate eta with Fullrank q
    @test DeepCompartmentModels._logpdf(q_fr, η_mv) == sum(logpdf.(q_fr, eachcol(η_mv)))
    @test logq(q_fr, η_mv) == sum(logpdf.(q_fr, eachcol(η_mv)))
    # Monte Carlo samples
    @test DeepCompartmentModels._logpdf.((q_fr, ), η_mv_mc) == [sum(logpdf.(q_fr, eachcol(η_mv_mc[i]))) for i in eachindex(η_mv_mc)]
    @test logq(q_fr, η_mv_mc) == [sum(logpdf.(q_fr, eachcol(η_mv_mc[i]))) for i in eachindex(η_mv_mc)]
    # Multivariate eta with meanfield q
    @test DeepCompartmentModels._logpdf(q_mf, η_mv) == sum(logpdf.(q_mf, eachcol(η_mv)))
    @test logq(q_mf, η_mv) == sum(logpdf.(q_mf, eachcol(η_mv)))

    # Univariate eta
    @test DeepCompartmentModels._logpdf(q_uni, η_uni) == logpdf(q_uni, η_uni)
    @test logq(q_uni, η_uni) == logpdf(q_uni, η_uni)
    # Monte Carlo samples
    @test logq(q_uni, η_uni_mc) == [logpdf(q_uni, η_uni_mc[i]) for i in eachindex(η_uni_mc)]
end
