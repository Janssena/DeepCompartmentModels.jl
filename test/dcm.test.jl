@testset "Run through entire processs" begin
    population = Population([
        Individual(Float32[1., 1.], Float32[12, 24], Float32[1., 0.3], generate_dosing_callback([0 1 60 1/60]); id = "fake_1"),
        Individual(Float32[0., 0.], Float32[12, 24], Float32[1., 0.3], generate_dosing_callback([0 1 60 1/60]); id = "fake_2")
    ])

    ann = Lux.Chain(
        Lux.Dense(2, 16, Lux.swish),
        Lux.Dense(16, 3, Lux.softplus),
    )

    obj_fn = SSE()
    model = DCM(one_comp_abs!, ann)

    ps, st = setup(obj_fn, model)

    opt = Optimisers.Adam(0.1f0)

    _, ps_update, _ = fit(SSE(), model, population, opt, ps, st; epochs = 1);
    
    @test ps !== ps_update
end