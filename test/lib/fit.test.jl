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

        # the loss of VariationalELBO is -elbo (path_deriv = false currently fails in Zygote: Diagonal pullback)
        obj = VariationalELBO([1, 2])
        ps, st = setup(obj, Random.Xoshiro(5), dcm, population)
        loss_vg, ∇ = value_and_gradient(obj, dcm, population, ps, st)
        @test loss_vg ≈ obj(dcm, population, ps, st) rtol = 1e-4
        @test ∇.theta == gradient(obj, dcm, population, ps, st).theta
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
end
