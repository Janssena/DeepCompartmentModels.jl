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

@testset "fixed-effect objectives are finite and distinct" begin
    population = toy_population(; n = 1)
    model = toy_dcm()
    ps, st = setup(SSE(), Random.MersenneTwister(12), model)

    mse_loss = MSE()(model, population, ps, st)
    sse_loss = SSE()(model, population, ps, st)
    @test isfinite(mse_loss)
    @test isfinite(sse_loss)
    @test sse_loss ≈ length(get_y(first(population))) * mse_loss
    @test all(isfinite, gradient(MSE(), model, population, ps, st).theta.layer_2.weight)

    likelihood_model = toy_dcm(; error = AdditiveError(0.2f0))
    ps_ll, st_ll = setup(
        LogLikelihood(), Random.MersenneTwister(12), likelihood_model)
    likelihood_loss = LogLikelihood()(likelihood_model, population, ps_ll, st_ll)
    likelihood_gradient = gradient(
        LogLikelihood(), likelihood_model, population, ps_ll, st_ll)
    @test isfinite(likelihood_loss)
    @test all(isfinite, likelihood_gradient.theta.layer_2.weight)
    @test all(isfinite, likelihood_gradient.error.σ)
end
