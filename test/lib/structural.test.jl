@testset "structural parameter transforms and metadata" begin
    layer = StructuralParameters(
        StructuralParameter(
            :baseline, -2.0f0; unit="mg/L",
            transform=IdentityTransform()),
        StructuralParameter(:CL, 3.0f0; unit="L/h", transform=LogTransform()),
        StructuralParameter(
            :fraction, 0.4f0; unit="1",
            transform=LogitTransform(0.0f0, 1.0f0)),
    )
    ps, st = Lux.setup(Random.MersenneTwister(91), layer)

    @test parameter_names(layer) == [:baseline, :CL, :fraction]
    @test parameter_units(layer) == ["mg/L", "L/h", "1"]
    @test parameter_transforms(layer) ==
        AbstractParameterTransform[
            IdentityTransform(), LogTransform(), LogitTransform(0.0f0, 1.0f0)]
    @test parameter_bounds(layer) == [
        (-Inf32, Inf32), (0.0f0, Inf32), (0.0f0, 1.0f0)]
    @test natural_parameters(layer, ps) ≈ [-2.0f0, 3.0f0, 0.4f0]
    @test unconstrained_parameters(
        layer, natural_parameters(layer, ps)) ≈ ps.unconstrained
    @test st == NamedTuple()
    @test Lux.parameterlength(layer) == 3
    @test Lux.statelength(layer) == 0
    @test occursin("baseline, CL, fraction", sprint(show, layer))

    vector_output, vector_state = layer(Float32[99], ps, st)
    matrix_output, matrix_state = layer(fill(99.0f0, 2, 4), ps, st)
    @test vector_output ≈ [-2.0f0, 3.0f0, 0.4f0]
    @test size(matrix_output) == (3, 4)
    @test all(column -> column ≈ vector_output, eachcol(matrix_output))
    @test vector_state === st
    @test matrix_state === st

    derivative = Zygote.gradient(
        values -> sum(natural_parameters(layer, values)), ps.unconstrained)[1]
    @test all(isfinite, derivative)
    @test all(>(0), derivative)

    @test_throws ArgumentError StructuralParameters()
    @test_throws ArgumentError StructuralParameters(
        StructuralParameter(:CL, 3.0), StructuralParameter(:CL, 4.0))
    @test_throws DomainError StructuralParameter(:CL, 0.0)
    @test_throws DomainError StructuralParameter(
        :fraction, 1.0; transform=LogitTransform(0.0, 1.0))
    @test_throws ArgumentError LogitTransform(1.0, 1.0)
    @test_throws DimensionMismatch natural_parameters(layer, [1.0, 2.0])
    @test_throws DimensionMismatch unconstrained_parameters(layer, [1.0, 2.0])

    concise = StructuralParameters(Float32[1.5, 3.0, 30.0])
    @test parameter_names(concise) == [:θ1, :θ2, :θ3]
    @test parameter_units(concise) == Union{Nothing,String}[nothing, nothing, nothing]

    named = StructuralParameters(
        [:ka, :CL, :V], Float32[1.5, 3.0, 30.0];
        units=["1/h", "L/h", "L"],
        transforms=[LogTransform(), LogTransform(), LogTransform()])
    @test parameter_names(named) == [:ka, :CL, :V]
    @test parameter_units(named) == ["1/h", "L/h", "L"]
    @test_throws DimensionMismatch StructuralParameters(
        [:CL], [3.0, 30.0])
end

@testset "structural layer DCM and FitResult contract" begin
    population = toy_population(Float64; n=1, seed=92)
    oral_layer = StructuralParameters(
        StructuralParameter(:ka, 1.5; unit="1/h"),
        StructuralParameter(:CL, 3.0; unit="L/h"),
        StructuralParameter(:V, 30.0; unit="L"),
    )
    oral_model = DCM(one_comp_abs!, oral_layer; target=2)
    oral_ps, oral_st = setup(
        MSE(), Random.MersenneTwister(93), oral_model, Float64)
    typical, _ = predict_typ_parameters(
        oral_model, population, oral_ps, oral_st)
    @test size(typical) == (3, 1)
    @test vec(typical) ≈ [1.5, 3.0, 30.0]

    result = fit(
        MSE(), oral_model, population, Optimisers.Descent(0.0),
        oral_ps, oral_st; epochs=1)
    @test result isa FitResult
    @test coef(result) ≈ [1.5, 3.0, 30.0]
    @test coefnames(result) == [:ka, :CL, :V]
    @test coefunits(result) == ["1/h", "L/h", "L"]
    @test result.metadata.n_fitted_scalars == 3
    @test result.metadata.structural_parameter_names == (:ka, :CL, :V)
    @test result.metadata.structural_parameter_units == ("1/h", "L/h", "L")
    @test result.metadata.structural_parameter_transforms ==
        (:LogTransform, :LogTransform, :LogTransform)
    @test result.metadata.structural_parameter_bounds ==
        ((0.0, Inf), (0.0, Inf), (0.0, Inf))

    two_layer = StructuralParameters(
        StructuralParameter(:CL, 3.0; unit="L/h"),
        StructuralParameter(:V1, 30.0; unit="L"),
        StructuralParameter(:Q, 2.0; unit="L/h"),
        StructuralParameter(:V2, 40.0; unit="L"),
    )
    two_model = DCM(two_comp!, two_layer; target=1)
    two_ps, two_st = setup(
        SSE(), Random.MersenneTwister(94), two_model, Float64)
    two_typical, _ = predict_typ_parameters(
        two_model, population, two_ps, two_st)
    @test vec(two_typical) ≈ [3.0, 30.0, 2.0, 40.0]

    neural_model = toy_dcm(Float64)
    neural_ps, neural_st = setup(
        MSE(), Random.MersenneTwister(95), neural_model, Float64)
    neural_result = FitResult(
        neural_model, MSE(), population, neural_ps, neural_st)
    @test_throws ArgumentError coef(neural_result)
    @test_throws ArgumentError coefnames(neural_result)
    @test_throws ArgumentError coefunits(neural_result)
end
