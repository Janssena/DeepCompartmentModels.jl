@testset "variational random effects" begin
    model = toy_dcm(; error = AdditiveError(0.1f0))
    population = toy_population(; n = 5)
    rng = Random.default_rng()

    for indices in ([2], [1, 2])
        ps, st = setup(VariationalELBO(indices), rng, model, population)
        update_epsilon!(rng, st)

        eta = sample_gaussian(ps.phi, st.phi)
        @test eta isa AbstractVector
        @test length(eta) == length(population)

        random_effects = get_random_effects(ps, st)
        @test random_effects isa AbstractMatrix
        @test size(random_effects, 1) == size(st.phi.mask, 1)
        @test size(random_effects, 2) == length(population)
    end
end

@testset "_to_random_eff_matrix" begin
    mask = Float32[1 0; 0 1; 0 0]
    eta = [Float32[0.1, -0.2], Float32[0.3, 0.4]]
    result = DeepCompartmentModels._to_random_eff_matrix(mask, eta)

    @test result isa AbstractMatrix
    @test size(result) == (3, 2)
    @test result[3, :] == zeros(Float32, 2)
end
