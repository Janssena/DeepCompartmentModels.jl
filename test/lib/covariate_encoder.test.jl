# Gate 9A — neural covariate-encoder DCM qualification (fast known-truth).
# The encoder must recover a known covariate->clearance relationship and beat a
# covariate-free baseline on held-out subjects.

@testset "NN encoder recovers a covariate effect and beats a covariate-free baseline" begin
    true_cl(crcl) = 3.0f0 .* (crcl ./ 80.0f0) .^ 0.7f0
    true_encoder = Lux.WrappedFunction(
        x -> vcat(true_cl(x), fill(30.0f0, 1, size(x, 2))))
    truth = DCM(one_comp!, true_encoder, AdditiveError(0.1f0); target=1)
    tps, tst = setup(LogLikelihood(), Random.Xoshiro(1), truth, Float32)

    n = 28
    crcl = collect(range(20.0f0, 140.0f0; length=n))
    design = simulation_population([[c] for c in crcl];
        regimen=dose_regimen(amount=100), obs_times=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0], T=Float32)
    data = simulate(truth, tps, tst, design; residual=true, seed=7)

    Random.seed!(11); order = Random.randperm(n); ntr = 21
    train = Population(collect(data)[order[1:ntr]])
    test = Population(collect(data)[order[(ntr + 1):end]])

    encoder = Lux.Chain(
        Normalize(Float32[150.0]), Lux.Dense(1, 16, Lux.swish),
        Lux.Dense(16, 2, Lux.softplus), InitialScale([3.0, 30.0]))
    model = DCM(one_comp!, encoder, AdditiveError(0.1f0); target=1)
    result = fit(LogLikelihood(), model, train, Optimisers.Adam(0.02f0);
                 epochs=600, rng=Random.Xoshiro(2))

    # Recovery of the covariate->CL function over the observed range.
    grid = collect(range(20.0f0, 140.0f0; length=60))
    z, _ = model.model(reshape(grid, 1, :), result.ps.theta, result.st.theta)
    @test cor(z[1, :], true_cl(grid)) > 0.9         # CL trend recovered
    @test isapprox(mean(z[2, :]), 30.0; rtol=0.2)   # V recovered ~constant
    @test all(isfinite, z)

    # Held-out prediction beats a covariate-free structural baseline.
    baseline_layer = StructuralParameters(
        StructuralParameter(:CL, 3.0; unit="L/h"), StructuralParameter(:V, 30.0; unit="L"))
    baseline = DCM(one_comp!, baseline_layer, AdditiveError(0.1f0); target=1)
    bresult = fit(LogLikelihood(), baseline, train, Optimisers.Adam(0.02f0);
                  epochs=600, rng=Random.Xoshiro(3))
    rmse(m, r, pop) = sqrt(mean(vcat(
        [(get_y(pop[i]) .- predict(m, pop[i], r.ps, r.st)) .^ 2 for i in eachindex(pop)]...)))
    @test rmse(model, result, test) < rmse(baseline, bresult, test)
end
