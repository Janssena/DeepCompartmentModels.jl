# InitialScale layer + mixed-effect workflow (EBE / IPRED, provisional uncertainty,
# mixed simulate). Mixed-effect uncertainty is provisional (Gate 5) but must run.

@testset "InitialScale anchors encoder output" begin
    layer = InitialScale([1.5, 3.0, 30.0])
    ps, st = Lux.setup(Random.MersenneTwister(1), layer)
    @test Lux.parameterlength(layer) == 0
    @test Lux.statelength(layer) == 3
    @test ps == NamedTuple()
    out_vec, _ = layer(Float32[1.0, 1.0, 1.0], ps, st)
    @test out_vec ≈ Float32[1.5, 3.0, 30.0]
    out_mat, _ = layer(ones(Float32, 3, 4), ps, st)
    @test all(col -> col ≈ Float32[1.5, 3.0, 30.0], eachcol(out_mat))
    @test occursin("InitialScale", sprint(show, layer))

    # As the final encoder layer it anchors predictions near the init values.
    encoder = Lux.Chain(
        Normalize(Float64[100.0]), Lux.Dense(1, 4, Lux.swish),
        Lux.Dense(4, 3, Lux.softplus), InitialScale([1.5, 3.0, 30.0]))
    model = DCM(one_comp_abs!, encoder, AdditiveError(0.1); target=2)
    ps_m, st_m = setup(MSE(), Random.MersenneTwister(2), model, Float64)
    z, _ = predict_typ_parameters(model, toy_population(Float64; n=1), ps_m, st_m)
    @test size(z, 1) == 3
    @test all(isfinite, z)
end

function _mixed_fixture(; seed=9500)
    layer = StructuralParameters(
        StructuralParameter(:ka, 1.4; unit="1/h"),
        StructuralParameter(:CL, 2.8; unit="L/h"),
        StructuralParameter(:V, 30.0; unit="L"))
    model = DCM(one_comp_abs!, layer, AdditiveError(0.1); target=2)
    truth_ps, st = setup(LogLikelihood(), Random.MersenneTwister(seed), model, Float64)
    design = simulation_population(
        [[70.0] for _ in 1:20]; regimen=dose_regimen(amount=320, n_doses=2, interval=12),
        obs_times=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    # Data with genuine between-subject variability on CL and residual error.
    data = simulate(model, truth_ps, st, design;
                    residual=true, omega=[0.09], random_effects=[2], seed=seed + 1)
    result = fit(VariationalELBO([2]), model, data, Optimisers.Adam(0.1);
                 n_outer=20, n_inner=10, monitor_samples=5,
                 m_step_kwargs=(epochs=5, num_samples=10),
                 rng=Random.MersenneTwister(seed + 2), verbose=false)
    return (; model, data, result)
end

const MIXED_FIXTURE = _mixed_fixture()

@testset "mixed-effect EBE predictions and IPRED record" begin
    fixture = MIXED_FIXTURE
    result = fixture.result
    @test result.metadata.effects === :mixed

    ebe = empirical_bayes(result)
    @test length(ebe) == length(fixture.data)
    @test all(v -> length(v) == 1, ebe)            # one random effect (CL)

    # Empirical-Bayes predictions are deterministic (no sampling noise).
    ipred1 = predict(result; ebe=true)
    ipred2 = predict(result; ebe=true)
    @test all(ipred1[i] == ipred2[i] for i in eachindex(ipred1))
    @test all(all(isfinite, y) for y in ipred1)
    pred = predict(result; individual=false)       # population PRED
    @test all(all(isfinite, y) for y in pred)

    record = prediction_record(result)             # IPRED-based
    @test length(record) == sum(length, get_y(fixture.data))
    @test all(isfinite, record.weighted_residual)
    @test all(isfinite, record.prediction)
end

@testset "mixed-effect uncertainty is provisional but computed" begin
    fixture = MIXED_FIXTURE
    u = uncertainty(fixture.result)
    @test u isa MixedEffectUncertainty
    @test u.names == [:ka, :CL, :V, :σ]
    @test u.diagnostics.positive_definite
    @test all(isfinite, stderror(u))
    @test all(>(0), stderror(u))
    @test diag(vcov(u)) ≈ stderror(u) .^ 2
    @test length(coef(u)) == 4

    # Ω, its SD and η-shrinkage are reported for the random-effect indices.
    @test u.random_effect_indices == [2]
    @test length(u.omega_sd) == 1
    @test u.omega_sd[1] > 0
    @test length(u.shrinkage) == 1
    @test u.shrinkage[1] <= 1

    ci = confint(u)
    @test length(ci) == 4
    @test all(c -> c[1] < c[2], ci)
    # The display must flag that these numbers are provisional / unqualified.
    @test occursin("PROVISIONAL", sprint(show, u))
end

@testset "simulate and VPC from a mixed-effect fit" begin
    fixture = MIXED_FIXTURE
    result = fixture.result

    sims = simulate(result; residual=true, n=30, seed=1)
    @test sims isa Vector{<:Population}
    @test length(sims) == 30
    @test all(all(isfinite, get_y(sim[1])) for sim in sims)

    summary = vpc(sims; observed=fixture.data, bins=4)
    @test summary isa VPCSummary
    @test size(summary.simulated) == (4, 3)
    @test summary.observed !== nothing

    # Overriding omega explicitly reproduces deterministically and skips the warning.
    a = simulate(result; residual=true, n=3, seed=2, omega=[0.09], random_effects=[2])
    b = simulate(result; residual=true, n=3, seed=2, omega=[0.09], random_effects=[2])
    @test all(get_y(a[1][i]) == get_y(b[1][i]) for i in eachindex(fixture.data))
end
