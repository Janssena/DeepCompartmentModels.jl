import Zygote

@testset "prediction and ODE solving" begin
    model = toy_dcm()
    population = toy_population(; n = 4)
    ps, st = setup(SSE(), Random.default_rng(), model)

    predictions = predict(model, population, ps, st)
    @test predictions isa AbstractVector
    @test length(predictions) == length(population)
    @test length(predictions[1]) == length(get_y(population[1]))
    @test all(all(isfinite, values) for values in predictions)

    parameters, _ = predict_typ_parameters(model, population, ps, st)
    @test size(parameters) == (3, length(population))
    @test all(parameters .> 0)

    solutions = predict(model, population, ps, st; target = false)
    @test length(solutions) == length(population)
    @test all(solution.retcode == ReturnCode.Success for solution in solutions)

    requested_times = Float32[0.5, 1.0, 2.0, 3.0]
    dense_predictions = predict(model, population, ps, st; saveat = requested_times)
    @test length(dense_predictions[1]) == length(requested_times)
end

@testset "prediction gradient" begin
    model = toy_dcm()
    individual = first(toy_population(; n = 1))
    ps, st = setup(SSE(), Random.default_rng(), model)

    loss(parameters) = sum(abs2, predict(model, individual, parameters, st))
    gradient = Zygote.gradient(loss, ps)[1]
    @test gradient !== nothing
    @test all(isfinite, gradient.theta.layer_2.weight)
end
