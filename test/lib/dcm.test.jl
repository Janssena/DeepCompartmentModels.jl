@testset "Run through entire processs" begin
    callback = generate_dosing_callback([0 1 60 1/60])
    
    population = Population([
        Individual("test_1", [1., 1.], [12., 24], [1., 0.3], callback)
        Individual("test_2", [0., 0.], [12., 24], [1., 0.3], callback)
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