@testset "safe Float32 initial step at pre-dose equilibrium" begin
    model = toy_dcm()
    individual = first(toy_population(; n=1))
    parameters = Float32[1.2, 2.8, 32.0]

    prediction = @test_nowarn solve_for_target(model, individual, parameters)
    automatic = solve_for_target(
        model, individual, parameters; dt=nothing, verbose=false)
    overridden = @test_nowarn solve_for_target(
        model, individual, parameters; dt=1.0f-4)

    @test prediction ≈ automatic rtol=1e-4 atol=1e-6
    @test overridden ≈ automatic rtol=1e-4 atol=1e-6
    @test DeepCompartmentModels._safe_initial_dt(Float32) > 10eps(Float32)
    @test isnothing(DeepCompartmentModels._safe_initial_dt(Float64))

    objective = MSE()
    ps, st = setup(objective, Random.MersenneTwister(41), model)
    @test_nowarn gradient(objective, model, Population([individual]), ps, st)
end
