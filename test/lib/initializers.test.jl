@testset "setup (fixed effects)" begin
    rng = Random.default_rng()
    ps, st = setup(SSE(), rng, toy_dcm())
    @test haskey(ps, :theta)
    @test haskey(st, :theta)

    ps_ll, _ = setup(LogLikelihood(), rng,
        toy_dcm(; error = AdditiveError(0.1f0)))
    @test haskey(ps_ll, :error)
    @test haskey(ps_ll.error, :σ)
end

@testset "setup (VariationalELBO)" begin
    model = toy_dcm(; error = AdditiveError(0.1f0))
    population = toy_population(; n = 5)
    rng = Random.default_rng()

    ps, st = setup(VariationalELBO([1, 2]), rng, model, population)
    @test Set(keys(ps)) == Set((:theta, :error, :omega, :phi))
    @test size(ps.omega) == (2, 2)
    @test length(ps.phi.μ) == length(population)
    @test size(st.phi.mask, 2) == 2
    @test length(st.phi.epsilon) == length(population)

    ps_mean_field, _ = setup(
        VariationalELBO([2]; mean_field = true), rng, model, population)
    @test haskey(ps_mean_field.phi, :σ)
end
