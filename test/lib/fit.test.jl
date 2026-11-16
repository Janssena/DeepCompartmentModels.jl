using Test
using Random
using DeepCompartmentModels

# Population of one-compartment individuals (CL and V from a small network).
function _fit_population(n; rng = Random.Xoshiro(1))
    t_obs = Float32[0.5, 1, 2, 4, 6, 8, 12, 16, 24]
    indvs = map(1:n) do i
        cb = generate_dosing_callback(Float32[0 (50 + 25 * (i % 4))], Float32)
        w = 60f0 + 3f0 * i
        p = Float32[5f0 * (w / 70)^0.75f0, 40f0 * (w / 70), 0]
        prob = ODEProblem(DeepCompartmentModels.one_comp!, Float32[0], (-0.1f0, 24f0), p)
        sol = solve(prob, Tsit5(); callback = cb, saveat = t_obs, tstops = cb.condition.times)
        y = only.(sol.u) .* (1f0 .+ 0.1f0 .* randn(rng, Float32, length(t_obs)))
        Individual(i, Float32[w / 70], t_obs, y, cb, Float32)
    end
    return Population(indvs)
end

@testset "create_batches" begin
    sizes(batches) = sort(length.(batches))

    # batchsize: maximum number of individuals per batch, balanced
    b = create_batches(Random.Xoshiro(1), 100; batchsize = 16)
    @test length(b) == 7 && maximum(length.(b)) <= 16 && sizes(b)[end] - sizes(b)[1] <= 1
    @test sort(reduce(vcat, b)) == 1:100 # every individual exactly once
    @test sizes(create_batches(Random.Xoshiro(1), 12; batchsize = 4)) == [4, 4, 4]
    @test sizes(create_batches(Random.Xoshiro(1), 10; batchsize = 4)) == [3, 3, 4]

    # nbatch: number of batches
    b = create_batches(Random.Xoshiro(1), 100; nbatch = 6)
    @test length(b) == 6 && sort(reduce(vcat, b)) == 1:100
    @test sizes(b)[end] - sizes(b)[1] <= 1

    # larger than the population
    @test length(create_batches(Random.Xoshiro(1), 5; batchsize = 16)) == 1
    @test length(create_batches(Random.Xoshiro(1), 5; nbatch = 16)) == 5

    # default is batchsize = 16
    @test create_batches(Random.Xoshiro(1), 100) == create_batches(Random.Xoshiro(1), 100; batchsize = 16)

    # reproducible with the rng
    @test create_batches(Random.Xoshiro(1), 20; nbatch = 3) == create_batches(Random.Xoshiro(1), 20; nbatch = 3)
    @test create_batches(Random.Xoshiro(1), 20; nbatch = 3) != create_batches(Random.Xoshiro(2), 20; nbatch = 3)

    # invalid arguments
    @test_throws ArgumentError create_batches(10; batchsize = 4, nbatch = 3)
    @test_throws ArgumentError create_batches(10; batchsize = 0)
    @test_throws ArgumentError create_batches(10; nbatch = 0)
end

@testset "fit" begin
    population = _fit_population(6)
    nn = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 2, Lux.softplus))
    dcm = DeepCompartmentModel(DeepCompartmentModels.one_comp!, nn, AdditiveError([1f0]))

    @testset "value_and_gradient returns the loss and the gradient" begin
        settings = (; parallel = false), (; parallel = :individual), (; parallel = :batch, nbatch = 2)
        for obj in (SSE(), MSE(), LogLikelihood())
            ps, st = setup(obj, Random.Xoshiro(3), dcm)
            loss = obj(dcm, population, ps, st)
            ∇ = gradient(obj, dcm, population, ps, st)
            for kwargs in settings
                loss_par, ∇_par = value_and_gradient(obj, dcm, population, ps, st; kwargs...)
                @test loss_par ≈ loss rtol = 1e-4
                @test maximum(abs, ∇_par.theta.layer_2.bias .- ∇.theta.layer_2.bias) / maximum(abs, ∇.theta.layer_2.bias) < 1e-4
                @test gradient(obj, dcm, population, ps, st; kwargs...).theta.layer_2.bias ≈ ∇_par.theta.layer_2.bias
            end
        end

        # the loss of VariationalELBO is -elbo
        obj = VariationalELBO([1, 2])
        ps, st = setup(obj, Random.Xoshiro(5), dcm, population)
        loss_vg, ∇ = value_and_gradient(obj, dcm, population, ps, st)
        @test loss_vg ≈ obj(dcm, population, ps, st) rtol = 1e-4
        @test ∇.theta == gradient(obj, dcm, population, ps, st).theta
    end

    @testset "keywords" begin
        silent = (args...) -> nothing
        opt = Optimisers.Adam(1f-2)
        ps, st = setup(SSE(), Random.Xoshiro(3), dcm)
        opt_state = Optimisers.setup(opt, ps)

        # unknown keywords are not ignored, for every method of fit
        @test_throws ArgumentError fit(SSE(), dcm, population, opt; epochs = 1, callback = silent, bogus = 1)
        @test_throws ArgumentError fit(SSE(), dcm, population, opt, ps, st; epochs = 1, callback = silent, bogus = 1)
        @test_throws ArgumentError fit(SSE(), dcm, population, opt_state, ps, st; epochs = 1, callback = silent, bogus = 1)
        err = try fit(SSE(), dcm, population, opt; epochs = 1, callback = silent, batch_size = 4); nothing catch e; e end
        @test err isa ArgumentError && occursin("batch_size", err.msg)

        # keywords of setup only apply when ps and st are created by fit
        @test_throws ArgumentError fit(SSE(), dcm, population, opt, ps, st; epochs = 1, callback = silent, init_sigma = [2f0])
        @test_throws ArgumentError fit(SSE(), dcm, population, opt; epochs = 1, callback = silent, init_omega = 0.5) # SSE has no random effects

        # keywords of setup are passed on
        _, ps_sigma, _ = fit(LogLikelihood(), dcm, population, opt; epochs = 0, init_sigma = [3f0])
        @test Lux.softplus.(ps_sigma.error.σ) ≈ [3f0]
        _, ps_omega, _ = fit(VariationalELBO([1, 2]), dcm, population, opt; epochs = 0, init_omega = 0.5)
        @test [ps_omega.omega[i, i] for i in 1:2] ≈ fill(0.5, 2)

        # element type
        _, ps32, _ = fit(SSE(), dcm, population, opt; epochs = 0)
        _, ps64, _ = fit(SSE(), dcm, population, opt; epochs = 0, T = Float64)
        @test eltype(ps32.theta.layer_1.weight) == Float32
        @test eltype(ps64.theta.layer_1.weight) == Float64
    end

    @testset "VariationalELBO uses the given rng" begin
        obj = VariationalELBO([1, 2])
        function losses(; setup_seed, fit_seed, with_state)
            ls = Float32[]
            cb = (epoch, loss) -> push!(ls, loss)
            opt = Optimisers.Adam(1f-2)
            if with_state
                ps, st = setup(obj, Random.Xoshiro(setup_seed), dcm, population)
                fit(obj, dcm, population, opt, ps, st; rng = Random.Xoshiro(fit_seed), epochs = 3, callback = cb)
            else
                fit(obj, dcm, population, opt; rng = Random.Xoshiro(fit_seed), epochs = 3, callback = cb)
            end
            return ls
        end

        for with_state in (false, true)
            a = losses(; setup_seed = 5, fit_seed = 1, with_state)
            b = losses(; setup_seed = 5, fit_seed = 1, with_state)
            c = losses(; setup_seed = 5, fit_seed = 2, with_state)
            @test a == b # same seed, same fit
            @test a != c # the noise of the posteriors depends on the rng
        end
    end

    @testset "update_state" begin
        obj = VariationalELBO([1, 2])
        ps, st = setup(obj, Random.Xoshiro(5), dcm, population)
        ϵ_before = deepcopy(st.phi.epsilon)

        st1 = update_state(Random.Xoshiro(1), st, obj, dcm, population, ps)
        st2 = update_state(Random.Xoshiro(1), st, obj, dcm, population, ps)
        @test st1.phi.epsilon == st2.phi.epsilon
        @test st1.phi.epsilon != ϵ_before
        @test st.phi.epsilon == ϵ_before # the given state is not modified

        # nothing to update for fixed effect objectives
        ps_f, st_f = setup(SSE(), Random.Xoshiro(5), dcm)
        @test update_state(Random.Xoshiro(1), st_f, SSE(), dcm, population, ps_f) === st_f
    end

    @testset "predict for a single individual" begin
        obj = VariationalELBO([1, 2])
        ps, st = setup(obj, Random.Xoshiro(5), dcm, population)
        ŷ_population = predict(dcm, population, ps, st)

        for i in (1, 4)
            ps_i, st_i = take_batch(ps, st, [i])
            @test predict(dcm, population[i], ps_i, st_i) ≈ ŷ_population[i]
            @test predict(dcm, population[i], ps_i, st_i; individual = false) ≈
                predict(dcm, population[i:i], ps_i, st_i; individual = false)[1]
        end

        # the random effects of all individuals are given, but only one individual is predicted
        @test_throws ArgumentError predict(dcm, population[1], ps, st)
    end

    @testset "multiple outputs" begin
        cb = generate_dosing_callback(Float32[0 100], Float32)
        t_pk, t_pd = Float32[1, 2, 4, 8], Float32[2, 8]
        mo_pop = Population(map(1:3) do i
            MOIndividual(i, Float32[i / 3], [t_pk, t_pd], [Float32[3, 2, 1, 0.5], Float32[1, 2]], cb, Float32)
        end)
        nn_mo = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 3, Lux.softplus))
        dcm_mo = DeepCompartmentModel(DeepCompartmentModels.one_comp_abs!, 2, nn_mo, ErrorModelSet(ProportionalError(0.2f0), AdditiveError(1f0)); target = [1, 2])

        @test size(get_x(mo_pop)) == (1, 3)

        obj = SSE()
        ps, st = setup(obj, Random.Xoshiro(3), dcm_mo; init_sigma = [[0.2f0], [1f0]])
        ŷ = predict(dcm_mo, mo_pop[1], ps, st)
        @test length.(ŷ) == [4, 2] # each output at its own observation times

        # a custom `saveat` returns every output at every time
        t_dense = Float32[0.5, 1, 2, 3, 4, 6, 8]
        ŷ_dense = predict(dcm_mo, mo_pop[1], ps, st; saveat = t_dense)
        @test length.(ŷ_dense) == [7, 7]
        @test ŷ_dense[1][[2, 3, 5, 7]] ≈ ŷ[1]
    end

    @testset "predict for an individual with occasions" begin
        occasions = [0f0 => 12f0, 168f0 => 192f0]
        cb = generate_dosing_callback(Float32[0 100; 168 100], Float32)
        t_obs = Float32[1, 4, 8, 169, 172, 180]
        indv = Individual(1, Float32[1], t_obs, zeros(Float32, 6), cb, Float32; occasions)
        ps, st = setup(SSE(), Random.Xoshiro(3), dcm)

        ŷ = predict(dcm, indv, ps, st)
        @test length(ŷ) == 2 # one vector of predictions per occasion
        @test length.(ŷ) == [3, 3]
        @test all(>(0), ŷ[2]) # the second dose is given in the second occasion
    end
end
