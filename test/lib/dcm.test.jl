@testset "one fixed-effect training step" begin
    model = toy_dcm()
    population = toy_population(; n = 2)
    objective = SSE()
    ps, st = setup(objective, Random.default_rng(), model)

    loss_before = objective(model, population, ps, st)
    parameter_gradient = gradient(objective, model, population, ps, st)
    optimiser_state = Optimisers.setup(Optimisers.Adam(1.0f-3), ps)
    optimiser_state, ps_updated =
        Optimisers.update(optimiser_state, ps, parameter_gradient)
    loss_after = objective(model, population, ps_updated, st)

    @test isfinite(loss_before)
    @test isfinite(loss_after)
    @test ps_updated !== ps
    @test ps_updated.theta.layer_2.weight != ps.theta.layer_2.weight
end
