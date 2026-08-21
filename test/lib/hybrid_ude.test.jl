# Gate 9B — mechanistic hybrid NeuralODE qualification (fast known-truth).
# A NeuralODE learning a time-varying clearance must recover the trend and beat a
# static (constant-clearance) baseline on held-out subjects. Kept small because
# NeuralODE-in-ODE fits are expensive.

# User-defined dynamics for the :two_comp_var hybrid model.
function (::UniversalDiffEq{P,TT})(model, u, p, t) where {P,TT<:CustomUDE{StaticSymbol{:hybrid_test}}}
    cl0, v1, q, v2 = p.latents
    cl = cl0 * only(model([t;;], p.node))
    k10 = cl / v1; k12 = q / v1; k21 = q / v2
    return [p.I / v1 - (k10 + k12) * u[1] + k21 * u[2], k12 * u[1] - k21 * u[2]]
end

@testset "hybrid NeuralODE recovers time-varying clearance and beats a static baseline" begin
    gtv(t) = 0.25f0 .+ 0.75f0 ./ (1f0 .+ exp.(-(t .- 14f0) ./ 3f0))   # CL 1 -> 4
    cl0, v1, q, v2, sigma = 4.0f0, 30.0f0, 2.0f0, 40.0f0, 0.2f0
    regimen = let times = collect(0.0f0:12.0f0:36.0f0)
        hcat(times, fill(100.0f0, length(times)), fill(100.0f0, length(times)),
             fill(1.0f0, length(times)))
    end
    subject(id, obs, y=zeros(Float32, length(obs))) =
        Individual(id, Float32[1.0], Float32.(obs), Float32.(y),
                   generate_dosing_callback(regimen, Float32), Float32)

    ude() = UniversalDiffEq(2; type=CustomUDE(:hybrid_test))
    truth = DCM(ude(), HybridModel(4,
        Lux.WrappedFunction(x -> repeat(Float32[cl0, v1, q, v2], 1, size(x, 2))),
        Lux.WrappedFunction(gtv)), AdditiveError(sigma); target=1)
    tps, tst = setup(MSE(), Random.Xoshiro(1), truth, Float32)

    obs_times = collect(1.0f0:4.0f0:37.0f0)
    rng = Random.MersenneTwister(3)
    n = 12
    data = Population([let clean = predict(truth, subject("t$i", obs_times), tps, tst)
        subject("s$i", obs_times, clean .+ sigma .* randn(rng, length(obs_times)))
    end for i in 1:n])
    train = Population(collect(data)[1:9]); test = Population(collect(data)[10:end])

    encoder = Lux.Chain(Normalize(Float32[2.0]), Lux.Dense(1, 6, Lux.swish),
                        Lux.Dense(6, 4, Lux.softplus), InitialScale([cl0, v1, q, v2]))
    node = Lux.Chain(Normalize(Float32[40.0]), Lux.Dense(1, 8, Lux.swish),
                     Lux.Dense(8, 1, Lux.softplus))
    hybrid = DCM(ude(), HybridModel(4, encoder, node), AdditiveError(sigma); target=1)
    hres = fit(LogLikelihood(), hybrid, train, Optimisers.Adam(0.01f0);
               epochs=250, rng=Random.Xoshiro(4))

    # predict works for the hybrid model (the base-UDE gap is closed here).
    @test all(all(isfinite, y) for y in predict(hybrid, test, hres.ps, hres.st))
    tgrid, node_out = interpret_node(hybrid, hres.ps, hres.st; t_dummy=0:2:36)
    @test all(isfinite, node_out)

    # Recovered effective clearance CL(t)=cl0*node(t) tracks the true trend.
    baseline, _ = hybrid.model.encoder(Float32[1.0;;], hres.ps.theta.encoder, hres.st.theta.encoder)
    cl_learned = baseline[1] .* node_out
    @test cor(cl_learned, cl0 .* gtv(Float32.(tgrid))) > 0.7

    # Held-out prediction beats a static (constant-clearance) baseline.
    static = DCM(two_comp!, StructuralParameters(
        StructuralParameter(:CL, 4.0), StructuralParameter(:V1, 30.0),
        StructuralParameter(:Q, 2.0), StructuralParameter(:V2, 40.0)),
        AdditiveError(sigma); target=1)
    sres = fit(LogLikelihood(), static, train, Optimisers.Adam(0.01f0);
               epochs=150, rng=Random.Xoshiro(5))
    rmse(m, r, pop) = sqrt(mean(vcat(
        [(get_y(pop[i]) .- predict(m, pop[i], r.ps, r.st)) .^ 2 for i in eachindex(pop)]...)))
    @test rmse(hybrid, hres, test) < rmse(static, sres, test)
end
