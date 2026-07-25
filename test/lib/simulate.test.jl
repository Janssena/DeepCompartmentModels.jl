# Gate 8B — forward simulation (core: deterministic + residual).

function _sim_single_model()
    layer = StructuralParameters(
        StructuralParameter(:ka, 1.4; unit="1/h"),
        StructuralParameter(:CL, 2.8; unit="L/h"),
        StructuralParameter(:V, 30.0; unit="L"),
    )
    model = DCM(one_comp_abs!, layer, AdditiveError(0.15); target=2)
    ps, st = setup(LogLikelihood(), Random.MersenneTwister(8000), model, Float64)
    return model, ps, st
end

@testset "dose_regimen: bolus, repeated and infusion" begin
    bolus = dose_regimen(amount=100)
    @test size(bolus) == (1, 2)
    @test bolus == [0.0 100.0]

    repeated = dose_regimen(amount=100, n_doses=4, interval=24)
    @test size(repeated) == (4, 2)
    @test repeated[:, 1] == [0.0, 24.0, 48.0, 72.0]
    @test all(==(100.0), repeated[:, 2])

    infusion = dose_regimen(amount=500, infusion_duration=2.0)
    @test size(infusion) == (1, 4)
    @test infusion[1, 3] == 250.0          # rate = amount / duration
    @test infusion[1, 4] == 2.0

    @test_throws ArgumentError dose_regimen(amount=-1)
    @test_throws ArgumentError dose_regimen(amount=100, n_doses=0)
    @test_throws ArgumentError dose_regimen(amount=100, n_doses=3, interval=0)
    @test_throws ArgumentError dose_regimen(amount=100, infusion_duration=0)
end

@testset "simulation_population: shared and per-subject schedules" begin
    shared = simulation_population(
        [[70.0], [55.0]]; regimen=dose_regimen(amount=100), obs_times=0:2:10)
    @test shared isa Population
    @test length(shared) == 2
    @test all(indv -> get_t(indv) == Float64.(0:2:10), shared)
    @test all(indv -> all(iszero, get_y(indv)), shared)

    per_subject = simulation_population(
        [[70.0], [55.0]];
        regimen=x -> dose_regimen(amount=2 * x[1]),
        obs_times=[[0.5, 4.0, 24.0], [1.0, 8.0]])
    @test get_t(per_subject[1]) == [0.5, 4.0, 24.0]
    @test get_t(per_subject[2]) == [1.0, 8.0]

    @test_throws ArgumentError simulation_population(
        [[70.0], [55.0]]; regimen=dose_regimen(amount=100),
        obs_times=[[0.5], [1.0], [2.0]])
    @test_throws ArgumentError simulation_population(
        []; regimen=dose_regimen(amount=100), obs_times=0:2:10)
end

@testset "simulate: deterministic, residual and reproducibility" begin
    model, ps, st = _sim_single_model()
    design = simulation_population(
        [[70.0], [60.0], [80.0]];
        regimen=dose_regimen(amount=320, n_doses=2, interval=12),
        obs_times=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0])

    # Deterministic simulation reproduces the typical prediction exactly.
    typical = predict(model, design, ps, st)
    deterministic = simulate(model, ps, st, design; residual=false)
    @test deterministic isa Population
    for i in eachindex(design)
        @test get_y(deterministic[i]) ≈ typical[i]
    end

    # Residual simulation perturbs the observations, is finite, and preserves the
    # design (times and covariates unchanged).
    noisy = simulate(model, ps, st, design; residual=true, seed=1)
    for i in eachindex(design)
        @test get_t(noisy[i]) == get_t(design[i])
        @test get_x(noisy[i]) == get_x(design[i])
        @test all(isfinite, get_y(noisy[i]))
        @test get_y(noisy[i]) != typical[i]
    end

    # Reproducibility: same seed reproduces, different seed differs.
    again = simulate(model, ps, st, design; residual=true, seed=1)
    different = simulate(model, ps, st, design; residual=true, seed=2)
    @test get_y(again[1]) == get_y(noisy[1])
    @test get_y(different[1]) != get_y(noisy[1])

    # n > 1 returns distinct replicate datasets.
    replicates = simulate(model, ps, st, design; residual=true, n=5, seed=3)
    @test replicates isa Vector{<:Population}
    @test length(replicates) == 5
    @test get_y(replicates[1][1]) != get_y(replicates[2][1])
end

@testset "simulate: multi-output asynchronous design" begin
    pop = mo_population(; n=2)
    model = mo_model()
    ps, st = setup(LogLikelihood(), Random.MersenneTwister(8100), model, MO_T)

    typical = predict(model, pop, ps, st)
    deterministic = simulate(model, ps, st, pop; residual=false)
    @test deterministic isa Population{<:MOIndividual}
    for i in eachindex(pop)
        # Per-DV observations equal the typical per-DV predictions...
        @test get_y(deterministic[i])[1] ≈ typical[i][1]
        @test get_y(deterministic[i])[2] ≈ typical[i][2]
        # ...and each DV keeps its own asynchronous time grid.
        @test deterministic[i].t == pop[i].t
        @test deterministic[i].dvid == pop[i].dvid
    end

    noisy = simulate(model, ps, st, pop; residual=true, seed=7)
    for i in eachindex(pop)
        @test all(all(isfinite, y) for y in get_y(noisy[i]))
        @test length(get_y(noisy[i])[1]) == length(get_y(pop[i])[1])
        @test length(get_y(noisy[i])[2]) == length(get_y(pop[i])[2])
    end
end

@testset "simulate: between-subject variability draws match Ω" begin
    model, ps, st = _sim_single_model()          # ka, CL, V — all LogTransform
    design = simulation_population(
        [[70.0] for _ in 1:6000]; regimen=dose_regimen(amount=320),
        obs_times=[1.0, 4.0])
    ζ, _ = predict_typ_parameters(model, design, ps, st)

    # Single random effect on CL (index 2): the log-multipliers must be ~N(0, Ω)
    # and the other parameters must be untouched.
    bsv1 = DeepCompartmentModels._build_bsv(model, ps, st, design, [0.09], [2])
    z1 = DeepCompartmentModels._simulated_parameters(ζ, bsv1, Random.MersenneTwister(1))
    @test z1[1, :] == ζ[1, :]                     # ka unchanged
    @test z1[3, :] == ζ[3, :]                     # V unchanged
    logmult = log.(z1[2, :] ./ ζ[2, :])
    @test abs(mean(logmult)) < 0.02
    @test isapprox(var(logmult), 0.09; atol=0.012)

    # Correlated random effects on CL and V recover the specified covariance.
    Ω = [0.09 0.02; 0.02 0.04]
    bsv2 = DeepCompartmentModels._build_bsv(model, ps, st, design, Ω, [2, 3])
    z2 = DeepCompartmentModels._simulated_parameters(ζ, bsv2, Random.MersenneTwister(2))
    a = log.(z2[2, :] ./ ζ[2, :]); b = log.(z2[3, :] ./ ζ[3, :])
    sample_cov = sum((a .- mean(a)) .* (b .- mean(b))) / (length(a) - 1)
    @test isapprox(var(a), Ω[1, 1]; atol=0.012)
    @test isapprox(var(b), Ω[2, 2]; atol=0.010)
    @test isapprox(sample_cov, Ω[1, 2]; atol=0.010)
    @test z2[1, :] == ζ[1, :]                     # ka (no random effect) unchanged
end

@testset "simulate: between-subject variability integration and guards" begin
    model, ps, st = _sim_single_model()
    design = simulation_population(
        [[70.0] for _ in 1:8]; regimen=dose_regimen(amount=320),
        obs_times=[0.5, 2.0, 6.0, 12.0])

    # With BSV and no residual, subjects with identical covariates now differ;
    # without BSV they are identical.
    deterministic = simulate(model, ps, st, design; residual=false)
    @test all(get_y(deterministic[i]) == get_y(deterministic[1]) for i in eachindex(design))

    bsv_sim = simulate(model, ps, st, design;
                       residual=false, omega=[0.09], random_effects=[2], seed=1)
    @test any(get_y(bsv_sim[i]) != get_y(bsv_sim[1]) for i in 2:length(design))
    again = simulate(model, ps, st, design;
                     residual=false, omega=[0.09], random_effects=[2], seed=1)
    @test all(get_y(again[i]) == get_y(bsv_sim[i]) for i in eachindex(design))

    @test_throws ArgumentError simulate(model, ps, st, design; omega=[0.09])
    @test_throws ArgumentError simulate(model, ps, st, design; random_effects=[2])
    @test_throws ArgumentError simulate(
        model, ps, st, design; omega=[0.09], random_effects=[9])
    @test_throws DimensionMismatch simulate(
        model, ps, st, design; omega=[0.09 0.0; 0.0 0.04], random_effects=[2])
    @test_throws ArgumentError simulate(
        model, ps, st, design; omega=[-0.1], random_effects=[2])
end

@testset "multi-output design builder" begin
    model = mo_model()
    ps, st = setup(LogLikelihood(), Random.MersenneTwister(8400), model, MO_T)

    # Shared multi-output schedule: a Tuple groups the dependent variables.
    shared = simulation_population(
        [[70.0], [60.0]]; regimen=dose_regimen(amount=100),
        obs_times=([0.5, 1.0, 2.0, 4.0], [1.0, 3.0, 4.0]), T=MO_T)
    @test shared isa Population{<:MOIndividual}
    @test length(shared) == 2
    for subject in shared
        @test [subject.t[m] for m in subject.dvid] == [[0.5, 1.0, 2.0, 4.0], [1.0, 3.0, 4.0]]
    end
    @test simulate(model, ps, st, shared; residual=false) isa Population{<:MOIndividual}

    # Per-subject multi-output schedules: a Vector of Tuples groups subjects.
    per_subject = simulation_population(
        [[70.0], [60.0]]; regimen=dose_regimen(amount=100),
        obs_times=[([0.5, 1.0], [1.0, 2.0]), ([2.0, 4.0], [3.0, 4.0])], T=MO_T)
    @test [per_subject[1].t[m] for m in per_subject[1].dvid] == [[0.5, 1.0], [1.0, 2.0]]
    @test [per_subject[2].t[m] for m in per_subject[2].dvid] == [[2.0, 4.0], [3.0, 4.0]]
    noisy = simulate(model, ps, st, per_subject; residual=true, seed=1)
    @test all(all(isfinite, y) for subject in noisy for y in get_y(subject))
end

@testset "parameter-uncertainty propagation via Gate 7A covariance" begin
    model, ps0, st = _sim_single_model()
    design = simulation_population(
        [[70.0] for _ in 1:24]; regimen=dose_regimen(amount=320),
        obs_times=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    data = simulate(model, ps0, st, design; residual=true, seed=99)
    result = fit(
        LogLikelihood(), model, data, Optimisers.Adam(0.05), deepcopy(ps0), st;
        epochs=1500, min_epochs=50, patience=30,
        objective_rel_tol=1e-9, step_rel_tol=1e-7, on_failure=:throw)
    u = uncertainty(result)

    draws = parameter_draws(u, 5; rng=Random.MersenneTwister(4))
    @test length(draws) == 5
    @test all(d -> length(d.theta.unconstrained) == 3 && length(d.error.σ) == 1, draws)
    @test draws[1].theta.unconstrained != draws[2].theta.unconstrained

    # Over many draws the sample mean and variance recover the 7A estimate/covariance.
    big = parameter_draws(u, 4000; rng=Random.MersenneTwister(5))
    stacked = reduce(hcat, [[d.theta.unconstrained; d.error.σ] for d in big])
    sample_mean = [mean(stacked[i, :]) for i in 1:4]
    sample_var = [var(stacked[i, :]) for i in 1:4]
    @test all(abs.(sample_mean .- u.unconstrained) .< 0.03)
    @test all(isapprox.(sample_var, diag(u.vcov_unconstrained); rtol=0.15))

    # Propagate parameter uncertainty into a simulation: one dataset per draw.
    sims = simulate(model, draws, st, design; residual=false)
    @test sims isa Vector{<:Population}
    @test length(sims) == 5
    @test get_y(sims[1][1]) != get_y(sims[2][1])
end

@testset "vpc binned summary" begin
    model, ps, st = _sim_single_model()
    design = simulation_population(
        [[70.0] for _ in 1:10]; regimen=dose_regimen(amount=320),
        obs_times=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    observed = simulate(model, ps, st, design; residual=true, seed=100)
    sims = simulate(model, ps, st, design; residual=true, n=100, seed=101)

    summary = vpc(sims; observed=observed, bins=5)
    @test summary isa VPCSummary
    @test length(summary.bin_centers) == 5
    @test size(summary.simulated) == (5, 3)
    @test summary.observed !== nothing
    @test size(summary.observed) == (5, 3)
    # Percentiles are ordered lower ≤ median ≤ upper in every populated bin.
    for b in 1:5
        summary.n_simulated[b] == 0 && continue
        @test summary.simulated[b, 1] <= summary.simulated[b, 2] <= summary.simulated[b, 3]
    end
    @test sum(summary.n_simulated) == 100 * sum(length, get_y(design))

    # Explicit edges give length(edges)-1 bins; simulated-only omits observed.
    explicit = vpc(sims; bins=[0.0, 2.0, 6.0, 12.0])
    @test length(explicit.bin_centers) == 3
    @test explicit.observed === nothing

    # Multi-output populations are not yet supported by vpc.
    mo_sims = [mo_population(; n=2)]
    @test_throws ArgumentError vpc(mo_sims)
end

@testset "simulate: FitResult convenience, ImplicitError and mixed guard" begin
    model, ps, st = _sim_single_model()
    population = toy_population(Float64; n=3, seed=8200)
    result = fit(
        LogLikelihood(), model, population, Optimisers.Adam(0.05), ps, st;
        epochs=3, min_epochs=1)
    from_result = simulate(result; residual=false)
    @test from_result isa Population
    @test length(from_result) == length(population)

    # ImplicitError (MSE fit): residual requested but no error model → typical.
    layer = StructuralParameters(
        StructuralParameter(:ka, 1.4; unit="1/h"),
        StructuralParameter(:CL, 2.8; unit="L/h"),
        StructuralParameter(:V, 30.0; unit="L"),
    )
    implicit_model = DCM(one_comp_abs!, layer; target=2)
    mse_ps, mse_st = setup(MSE(), Random.MersenneTwister(8201), implicit_model, Float64)
    typical = predict(implicit_model, population, mse_ps, mse_st)
    implicit_sim = simulate(implicit_model, mse_ps, mse_st, population; residual=true, seed=1)
    for i in eachindex(population)
        @test get_y(implicit_sim[i]) ≈ typical[i]
    end

    # The model-level method refuses a VEM parameter tree (omega/phi): BSV must be
    # specified explicitly there.
    mixed_ps, mixed_st = setup(
        VariationalELBO([2]), Random.MersenneTwister(8202), model, population, Float64)
    @test_throws ArgumentError simulate(model, mixed_ps, mixed_st, population)
    # The FitResult convenience DOES simulate from a mixed fit, drawing BSV from the
    # fitted (provisional, Gate-5-unqualified) Ω; it returns a usable Population.
    mixed_result = FitResult(model, VariationalELBO([2]), population, mixed_ps, mixed_st)
    @test simulate(mixed_result; residual=true, seed=1) isa Population
end
