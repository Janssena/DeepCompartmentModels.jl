# Gate 7A — fixed-effect observed-information uncertainty.

# Build a fixed-effect structural fit on data simulated WITHOUT random effects,
# so the fixed-effect (naive-pooled) estimand matches the generating model and
# interval coverage is meaningful.
function _uncertainty_fixture(; n=24, seed=7001, sigma=0.1, epochs=1500)
    truth = [1.4, 2.8, 30.0]
    layer = StructuralParameters(
        StructuralParameter(:ka, truth[1]; unit="1/h"),
        StructuralParameter(:CL, truth[2]; unit="L/h"),
        StructuralParameter(:V, truth[3]; unit="L"),
    )
    model = DCM(one_comp_abs!, layer, AdditiveError(sigma); target=2)
    truth_ps, st = setup(LogLikelihood(), Random.MersenneTwister(seed), model, Float64)

    rng = Random.MersenneTwister(seed + 1)
    times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
    individuals = map(1:n) do i
        callback = generate_dosing_callback([0.0 100.0 1000.0 0.1], Float64)
        template = Individual("sim$i", Float64[70.0], times,
                              zeros(length(times)), callback, Float64)
        clean = predict(model, template, truth_ps, st)
        noisy = clean .+ sigma .* randn(rng, length(times))
        Individual("sim$i", Float64[70.0], times, noisy, callback, Float64)
    end
    population = Population(individuals)

    fit_ps, _ = setup(LogLikelihood(), Random.MersenneTwister(seed + 2), model, Float64)
    result = fit(
        LogLikelihood(), model, population, Optimisers.Adam(0.05), fit_ps, st;
        epochs, min_epochs=50, patience=30,
        objective_rel_tol=1e-9, step_rel_tol=1e-7, on_failure=:throw)
    return (; truth, model, result)
end

@testset "fixed-effect Hessian uncertainty contract" begin
    fixture = _uncertainty_fixture()
    result = fixture.result
    @test isconverged(result)

    u = uncertainty(result)
    @test u isa FixedEffectUncertainty
    @test u.names == [:ka, :CL, :V, :σ]
    @test u.kinds == [:structural, :structural, :structural, :residual]
    @test u.units == Union{Nothing,String}["1/h", "L/h", "L", nothing]
    @test u.method === :forwarddiff

    # A valid covariance requires a positive-definite observed information.
    @test u.diagnostics.positive_definite
    @test all(isfinite, u.standard_error)
    @test all(>(0), u.standard_error)
    @test all(isfinite, u.relative_standard_error)
    @test all(>(0), u.relative_standard_error)

    # Numerical verification: the independent ForwardDiff and finite-difference
    # Hessians must agree, and the reported Hessian must be symmetric.
    @test u.diagnostics.hessian_cross_check < 1e-2
    @test u.diagnostics.hessian_symmetry < 1e-8
    @test u.diagnostics.gradient_norm < 1e-1

    # StatsAPI-style accessors.
    @test coef(u) == u.estimate
    @test coefnames(u) == u.names
    @test stderror(u) == u.standard_error
    @test vcov(u) === u.vcov
    @test diag(u.vcov) ≈ u.standard_error .^ 2

    # coef(result) (structural only) is the structural prefix of coef(u).
    @test coef(u)[1:3] ≈ coef(result)

    # Confidence intervals: default level round-trips; tighter level is narrower;
    # transforms keep natural-scale bounds inside the parameter domain and place
    # the estimate strictly inside a (generally asymmetric) interval.
    ci95 = confint(u)
    @test ci95 == collect(zip(u.lower, u.upper))
    ci90 = confint(u; level=0.90)
    for i in eachindex(u.estimate)
        @test ci90[i][1] > ci95[i][1]
        @test ci90[i][2] < ci95[i][2]
        @test ci95[i][1] < u.estimate[i] < ci95[i][2]
        @test ci95[i][1] > 0                      # log / softplus keep it positive
    end

    # Delta method for a LogTransform parameter: d(natural)/d(uncon) = natural.
    se_uncon = sqrt.(diag(u.vcov_unconstrained))
    @test u.standard_error[1:3] ≈ u.estimate[1:3] .* se_uncon[1:3]

    # Sanity against known truth (single dataset; formal coverage is in Gate 7A
    # analysis): each structural truth lies within four natural-scale SEs.
    @test all(abs.(u.estimate[1:3] .- fixture.truth) .<= 4 .* u.standard_error[1:3])

    @test occursin("FixedEffectUncertainty", sprint(show, u))
    @test occursin("forwarddiff", sprint(show, u))
end

@testset "uncertainty refuses non-identified and unsupported fits" begin
    population = toy_population(Float64; n=4, seed=7100)

    # MSE/SSE are not likelihoods: no observed-information covariance.
    layer = StructuralParameters(
        StructuralParameter(:ka, 1.5; unit="1/h"),
        StructuralParameter(:CL, 3.0; unit="L/h"),
        StructuralParameter(:V, 30.0; unit="L"),
    )
    structural_model = DCM(one_comp_abs!, layer, AdditiveError(0.1); target=2)
    mse_ps, mse_st = setup(MSE(), Random.MersenneTwister(7101), structural_model, Float64)
    mse_result = fit(
        MSE(), structural_model, population, Optimisers.Descent(0.0),
        mse_ps, mse_st; epochs=1)
    @test_throws ArgumentError uncertainty(mse_result)

    # Raw neural-network weights are not identified pharmacokinetic coefficients.
    neural_model = toy_dcm(Float64; error=AdditiveError(0.1f0))
    neural_ps, neural_st = setup(
        LogLikelihood(), Random.MersenneTwister(7102), neural_model, Float64)
    neural_result = fit(
        LogLikelihood(), neural_model, population, Optimisers.Descent(0.0),
        neural_ps, neural_st; epochs=1)
    @test_throws ArgumentError uncertainty(neural_result)

    # Mixed-effect uncertainty is conditional on Gate 5 and not implemented here.
    mixed_result = FitResult(
        structural_model, VariationalELBO([2]), population, mse_ps, mse_st)
    @test mixed_result.metadata.effects === :mixed
    @test_throws ArgumentError uncertainty(mixed_result)
end
