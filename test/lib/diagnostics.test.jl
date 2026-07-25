# Gate 7B — standardized model-evaluation records.

function _diagnostics_single_output(; n=8, sigma=0.1, seed=7300)
    layer = StructuralParameters(
        StructuralParameter(:ka, 1.4; unit="1/h"),
        StructuralParameter(:CL, 2.8; unit="L/h"),
        StructuralParameter(:V, 30.0; unit="L"),
    )
    model = DCM(one_comp_abs!, layer, AdditiveError(sigma); target=2)
    ps, st = setup(LogLikelihood(), Random.MersenneTwister(seed), model, Float64)

    rng = Random.MersenneTwister(seed + 1)
    times = [0.5, 1.0, 2.0, 4.0, 8.0]
    draws = Dict{String,Vector{Float64}}()
    individuals = map(1:n) do i
        callback = generate_dosing_callback([0.0 100.0 1000.0 0.1], Float64)
        template = Individual("s$i", Float64[70.0], times,
                              zeros(length(times)), callback, Float64)
        clean = predict(model, template, ps, st)
        z = randn(rng, length(times))
        draws["s$i"] = z
        Individual("s$i", Float64[70.0], times, clean .+ sigma .* z, callback, Float64)
    end
    return (; model, ps, st, population=Population(individuals), sigma, draws, times)
end

@testset "prediction record: single output and exact known-truth residuals" begin
    fixture = _diagnostics_single_output()
    record = prediction_record(fixture.model, fixture.population, fixture.ps, fixture.st)

    n_obs = sum(length, get_y(fixture.population))
    @test record isa PredictionRecord
    @test length(record) == n_obs
    @test all(==(1), record.dv)                    # single output
    @test Set(record.id) == Set("s" .* string.(1:8))
    @test occursin("PredictionRecord", sprint(show, record))

    # Gaussian error model: predictive mean equals the structural prediction, and
    # the predictive SD is the (constant) additive standard deviation.
    @test record.predictive_mean ≈ record.prediction
    @test all(isapprox.(record.predictive_std, fixture.sigma; atol=1e-8))
    @test record.residual ≈ record.observation .- record.prediction

    # At the true parameters the weighted residual must recover the exact standard
    # normal draws used to simulate each observation.
    for id in unique(record.id)
        @test record.weighted_residual[record.id .== id] ≈ fixture.draws[id]
    end
    # Standardized residuals therefore have (sample) mean ≈ 0 and SD ≈ 1.
    @test abs(mean(record.weighted_residual)) < 0.3
    @test isapprox(std(record.weighted_residual), 1.0; atol=0.3)
end

@testset "prediction record: multi-output asynchronous alignment" begin
    pop = mo_population(; n=2)
    model = mo_model()
    ps, st = setup(LogLikelihood(), Random.MersenneTwister(7310), model, MO_T)
    record = prediction_record(model, pop, ps, st)

    @test length(record) == sum(subject -> sum(length, get_y(subject)), pop)
    @test Set(record.dv) == Set([1, 2])

    # Each dependent variable keeps its OWN observation times; DV 1 and DV 2 have
    # different, asynchronous grids and must never be aligned to each other.
    t1 = [0.5, 1.0, 2.0, 4.0]
    t2 = [1.0, 3.0, 4.0]
    for subject in pop
        mask1 = (record.id .== subject.id) .& (record.dv .== 1)
        mask2 = (record.id .== subject.id) .& (record.dv .== 2)
        @test record.time[mask1] ≈ t1
        @test record.time[mask2] ≈ t2
        @test record.observation[mask1] ≈ get_y(subject)[1]
        @test record.observation[mask2] ≈ get_y(subject)[2]
    end
    @test all(isfinite, record.weighted_residual)
    @test all(>(0), record.predictive_std)
end

@testset "prediction record: ImplicitError and FitResult dispatch" begin
    population = toy_population(Float64; n=4, seed=7320)
    layer = StructuralParameters(
        StructuralParameter(:ka, 1.5; unit="1/h"),
        StructuralParameter(:CL, 3.0; unit="L/h"),
        StructuralParameter(:V, 30.0; unit="L"),
    )

    # MSE fit uses ImplicitError: no predictive distribution.
    implicit_model = DCM(one_comp_abs!, layer; target=2)
    mse_ps, mse_st = setup(MSE(), Random.MersenneTwister(7321), implicit_model, Float64)
    implicit_record = prediction_record(implicit_model, population, mse_ps, mse_st)
    @test all(isnan, implicit_record.predictive_std)
    @test all(isnan, implicit_record.weighted_residual)
    @test implicit_record.predictive_mean ≈ implicit_record.prediction
    @test implicit_record.residual ≈ implicit_record.observation .- implicit_record.prediction

    # FitResult method equals the explicit model-level call for a fixed-effect fit.
    like_model = DCM(one_comp_abs!, layer, AdditiveError(0.1); target=2)
    ll_ps, ll_st = setup(LogLikelihood(), Random.MersenneTwister(7322), like_model, Float64)
    result = fit(
        LogLikelihood(), like_model, population, Optimisers.Adam(0.05),
        ll_ps, ll_st; epochs=5, min_epochs=1)
    result_record = prediction_record(result)
    direct_record = prediction_record(result.model, result.data, result.ps, result.st)
    @test result_record.weighted_residual ≈ direct_record.weighted_residual
    @test result_record.prediction ≈ direct_record.prediction

    # Mixed-effect results are supported (individual/empirical-Bayes diagnostics);
    # the real EBE path is exercised in the mixed-effect workflow tests.
    mixed_result = FitResult(
        like_model, VariationalELBO([2]), population, ll_ps, ll_st)
    @test mixed_result.metadata.effects === :mixed
    @test prediction_record(mixed_result) isa PredictionRecord
end
