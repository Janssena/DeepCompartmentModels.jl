# Gate 9A0 — audit regression test for the existing UniversalDiffEq.
#
# Pins the CURRENT behaviour of the base UDE type so the Gate 9B HybridModel
# redesign is a deliberate, tracked change:
#   * the core (solve_for_target / objective / gradient) works for BasicUDE and
#     TimeConcatUDE;
#   * the user-facing `predict` does NOT yet support a pure UDE (it routes through
#     the covariate-encoder path) — this is the gap HybridModel closes.

function _ude_individual()
    callback = generate_dosing_callback(Float32[0.0 100.0], Float32)
    return Individual("ude1", Float32[70.0], Float32[1.0, 2.0, 4.0, 8.0],
                      Float32[1.0, 0.8, 0.5, 0.2], callback, Float32)
end

@testset "UniversalDiffEq core: BasicUDE" begin
    node = Lux.Chain(Lux.Dense(1, 8, Lux.tanh), Lux.Dense(8, 1))   # du/dt = NN(u)
    ude = UniversalDiffEq(1; type=BasicUDE())
    dcm = DCM(ude, node, AdditiveError(0.1f0); target=1)
    ps, st = setup(MSE(), Random.Xoshiro(1), dcm, Float32)

    # setup yields a ComponentVector theta carrying the intervention field `I`.
    @test ps.theta isa DeepCompartmentModels.ComponentArray
    @test :I in keys(ps.theta)

    individual = _ude_individual()
    population = Population([individual])

    yhat = solve_for_target(dcm, individual, ps, st)
    @test length(yhat) == length(get_t(individual))
    @test all(isfinite, yhat)

    value = MSE()(dcm, population, ps, st)
    @test isfinite(value)

    grad = gradient(MSE(), dcm, population, ps, st)
    @test all(isfinite, collect(grad.theta))

    # Known gap (closed by the Gate 9B HybridModel): the covariate-encoder
    # `predict` path does not support a pure UDE.
    @test_throws Exception predict(dcm, population, ps, st)
end

@testset "UniversalDiffEq core: TimeConcatUDE" begin
    node = Lux.Chain(Lux.Dense(2, 8, Lux.tanh), Lux.Dense(8, 1))   # (u, t) -> du
    ude = UniversalDiffEq(1; type=TimeConcatUDE())
    dcm = DCM(ude, node, AdditiveError(0.1f0); target=1)
    ps, st = setup(MSE(), Random.Xoshiro(2), dcm, Float32)

    individual = _ude_individual()
    yhat = solve_for_target(dcm, individual, ps, st)
    @test length(yhat) == length(get_t(individual))
    @test all(isfinite, yhat)
    @test isfinite(MSE()(dcm, Population([individual]), ps, st))
end
