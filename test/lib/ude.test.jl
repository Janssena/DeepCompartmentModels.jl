using Test
using Random
using DeepCompartmentModels

const _gradient = DeepCompartmentModels.gradient

# One-compartment model with Michaelis-Menten elimination, in amounts: dA = I - Vmax * A / (Km + A)
function _mm_ode!(dA, A, p, t)
    Vmax, Km, I = p
    dA[1] = I - (Vmax * A[1]) / (Km + A[1])
end

# Individuals receive different doses, so that mix-ups of the infusion rate between individuals are visible.
function _ude_population(n; rng = Random.Xoshiro(1))
    t_obs = Float32[0.5, 1, 2, 4, 6, 8, 12, 16, 24]
    indvs = map(1:n) do i
        cb = generate_dosing_callback(Float32[0 (50 + 25 * (i % 4))], Float32)
        prob = ODEProblem(_mm_ode!, Float32[0], (-0.1f0, 24f0), Float32[8, 20, 0])
        sol = solve(prob, Tsit5(); callback = cb, saveat = t_obs, tstops = cb.condition.times)
        Individual(i, Float32[1], t_obs, only.(sol.u) .+ 0.1f0 .* randn(rng, Float32, length(t_obs)), cb, Float32)
    end
    return Population(indvs)
end

_ude_nn(h = 8) = Lux.Chain(Lux.Dense(1, h, tanh), Lux.Dense(h, h, tanh), Lux.Dense(h, 1))

@testset "UniversalDiffEq" begin
    rng = Random.Xoshiro(2)
    population = _ude_population(8)

    @testset "setup" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn())
        ps, st = setup(SSE(), rng, dcm)

        @test keys(ps) == (:theta, )
        @test keys(st) == (:theta, )
        @test :I in keys(ps.theta)
        @test ps.theta.I == 0
        @test eltype(ps.theta) == Float32
    end

    @testset "solve matches analytical solution" begin
        # dA/dt = I - k * A, with a linear network: weight = -k and zero bias.
        k = 0.2f0
        dcm = DeepCompartmentModel(UniversalDiffEq(1), Lux.Chain(Lux.Dense(1, 1)))
        ps, st = setup(SSE(), rng, dcm)
        ps.theta.layer_1.weight .= -k
        ps.theta.layer_1.bias .= 0

        dose, τ = 100f0, 1f0 / 60f0
        callback = generate_dosing_callback(Float32[0 dose], Float32)
        ts = Float32[1, 2, 4, 8, 16]
        indv = Individual(1, Float32[1], ts, zeros(Float32, length(ts)), callback, Float32)

        R = dose / τ
        analytical = @. R / k * (1 - exp(-k * τ)) * exp(-k * (ts - τ))

        @test solve_for_target(dcm, indv, ps, st) ≈ analytical rtol = 1e-3
    end

    @testset "solving does not modify the parameters" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn())
        ps, st = setup(SSE(), rng, dcm)
        ps_before = deepcopy(ps)

        SSE()(dcm, population, ps, st)
        solve(dcm, population[1], ps, st; interpolate = true)

        @test ps.theta == ps_before.theta
        @test ps.theta.I == 0
    end

    @testset "solving does not modify the callback of the individual" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn())
        ps, st = setup(SSE(), rng, dcm)
        save_positions = copy(population[1].callback.save_positions)

        sol = solve(dcm, population[1], ps, st; interpolate = true)

        @test population[1].callback.save_positions == save_positions
        @test length(sol.t) > length(get_t(population[1])) # events are saved when interpolating
    end

    @testset "state can be passed with or without the theta key" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn())
        ps, st = setup(SSE(), rng, dcm)

        @test solve_for_target(dcm, population[1], ps, st) == solve_for_target(dcm, population[1], ps, st.theta)
        @test build_problem(dcm.problem, dcm.model, st.theta) isa SciMLBase.AbstractODEProblem
    end

    @testset "predict" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn())
        ps, st = setup(SSE(), rng, dcm)

        ŷ = predict(dcm, population[1], ps, st)
        @test ŷ == solve_for_target(dcm, population[1], ps, st)
        @test length(ŷ) == length(get_t(population[1]))

        sol = predict(dcm, population[1], ps, st; target = false, saveat = 0f0:2f0:24f0)
        @test sol isa SciMLBase.AbstractODESolution
        @test sol.t == collect(0f0:2f0:24f0)

        @test length(predict(dcm, population, ps, st)) == length(population)
    end

    @testset "objectives" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn(), AdditiveError([1f0]))
        ps, st = setup(LogLikelihood(), rng, dcm)

        @test isfinite(SSE()(dcm, population, ps, st))
        @test isfinite(MSE()(dcm, population, ps, st))
        @test isfinite(LogLikelihood()(dcm, population, ps, st))
    end

    @testset "gradients" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn())
        ps, st = setup(SSE(), rng, dcm)
        ∇ = _gradient(SSE(), dcm, population, ps, st)

        @test all(isfinite, ∇.theta)
        @test any(!iszero, ∇.theta)

        # Individuals get different doses: parallel gradients are only equal to the sequential one if
        # each individual uses its own infusion rate.
        scale = maximum(abs, ∇.theta)
        for kwargs in ((; parallel = :individual), (; parallel = :batch, batchsize = 4), (; parallel = :batch, nbatch = 3))
            ∇_par = _gradient(SSE(), dcm, population, ps, st; kwargs...)
            @test maximum(abs, ∇_par.theta .- ∇.theta) / scale < 1e-4
        end
        @test_throws ArgumentError _gradient(SSE(), dcm, population, ps, st; parallel = :batch, batchsize = 4, nbatch = 3)

        # MSE is a mean over individuals: summed per-individual/batch gradients must be weighted
        # (otherwise they are length(population) or the number of batches times too large).
        ∇_mse = _gradient(MSE(), dcm, population, ps, st)
        scale_mse = maximum(abs, ∇_mse.theta)
        for kwargs in ((; parallel = :individual), (; parallel = :batch, batchsize = 4), (; parallel = :batch, batchsize = 3))
            ∇_par = _gradient(MSE(), dcm, population, ps, st; kwargs...)
            @test maximum(abs, ∇_par.theta .- ∇_mse.theta) / scale_mse < 1e-4
        end

        # LogLikelihood is a sum over individuals: its batch gradients must NOT be weighted.
        dcm_ll = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn(), AdditiveError([5f0]))
        ps_ll, st_ll = setup(LogLikelihood(), Random.Xoshiro(4), dcm_ll) # own rng: keeps the shared stream of the other testsets unchanged
        ∇_ll = _gradient(LogLikelihood(), dcm_ll, population, ps_ll, st_ll)
        scale_ll = maximum(abs, ∇_ll.theta)
        for kwargs in ((; parallel = :individual), (; parallel = :batch, batchsize = 4), (; parallel = :batch, batchsize = 3))
            ∇_par = _gradient(LogLikelihood(), dcm_ll, population, ps_ll, st_ll; kwargs...)
            @test maximum(abs, ∇_par.theta .- ∇_ll.theta) / scale_ll < 1e-4
            @test maximum(abs, ∇_par.error.σ .- ∇_ll.error.σ) / maximum(abs, ∇_ll.error.σ) < 1e-4
        end
    end

    @testset "adjoint gradients agree with forward sensitivities" begin
        dcm_fwd = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn(); sensealg = ForwardDiffSensitivity())
        dcm_adj = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn(); sensealg = GaussAdjoint(; autojacvec = ReverseDiffVJP(true)))
        ps, st = setup(SSE(), rng, dcm_fwd)

        ∇_fwd = _gradient(SSE(), dcm_fwd, population, ps, st)
        ∇_adj = _gradient(SSE(), dcm_adj, population, ps, st)

        @test maximum(abs, ∇_adj.theta .- ∇_fwd.theta) / maximum(abs, ∇_fwd.theta) < 1e-2
    end

    @testset "in-place problems" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn())
        dcm_ip = DeepCompartmentModel(UniversalDiffEq(1; inplace = true), _ude_nn())
        ps, st = setup(SSE(), Random.Xoshiro(2), dcm)
        ps_ip, st_ip = setup(SSE(), Random.Xoshiro(2), dcm_ip)

        @test ps.theta == ps_ip.theta
        @test solve_for_target(dcm_ip, population[1], ps_ip, st_ip) ≈ solve_for_target(dcm, population[1], ps, st) rtol = 1e-4

        ∇ = _gradient(SSE(), dcm, population, ps, st)
        ∇_ip = _gradient(SSE(), dcm_ip, population, ps_ip, st_ip)
        @test maximum(abs, ∇_ip.theta .- ∇.theta) / maximum(abs, ∇.theta) < 1e-3
    end

    @testset "TimeConcatUDE" begin
        nn = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))
        dcm = DeepCompartmentModel(UniversalDiffEq(1; type = TimeConcatUDE()), nn)
        dcm_ip = DeepCompartmentModel(UniversalDiffEq(1; inplace = true, type = TimeConcatUDE()), nn)
        ps, st = setup(SSE(), rng, dcm)

        @test isfinite(SSE()(dcm, population, ps, st))
        @test SSE()(dcm_ip, population, ps, st) ≈ SSE()(dcm, population, ps, st) rtol = 1e-4
    end

    @testset "default sensitivity algorithm depends on the number of parameters" begin
        small = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn(8))
        large = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn(32))
        explicit = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn(32); sensealg = ForwardDiffSensitivity())

        @test small.sensealg isa ForwardDiffSensitivity
        @test large.sensealg isa GaussAdjoint
        @test explicit.sensealg isa ForwardDiffSensitivity
    end

    @testset "fit" begin
        dcm = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn())
        losses = Float32[]
        opt_state, ps, st = fit(SSE(), dcm, population, Optimisers.Adam(1f-2); 
            rng = Random.Xoshiro(2), epochs = 30, callback = (epoch, loss) -> push!(losses, loss))

        @test length(losses) == 30
        @test last(losses) < first(losses)
        @test keys(ps) == (:theta, ) && keys(st) == (:theta, )

        # fit does what the explicit loop does
        ps0, st0 = setup(SSE(), Random.Xoshiro(3), dcm)
        _, ps_fit, _ = fit(SSE(), dcm, population, Optimisers.Adam(1f-2), ps0, st0; epochs = 3, callback = (args...) -> nothing)
        opt_state_loop, ps_loop = Optimisers.setup(Optimisers.Adam(1f-2), ps0), ps0
        for _ in 1:3
            ∇ = _gradient(SSE(), dcm, population, ps_loop, st0)
            opt_state_loop, ps_loop = Optimisers.update(opt_state_loop, ps_loop, ∇)
        end
        @test ps_fit.theta ≈ ps_loop.theta

        # continuing from the returned optimiser state
        opt_state2, ps2, _ = fit(SSE(), dcm, population, opt_state, ps, st; epochs = 2, callback = (args...) -> nothing)
        @test ps2.theta != ps.theta

        # with an error model
        dcm_ll = DeepCompartmentModel(UniversalDiffEq(1), _ude_nn(), AdditiveError([5f0]))
        _, ps_ll, _ = fit(LogLikelihood(), dcm_ll, population, Optimisers.Adam(1f-2); epochs = 3, callback = (args...) -> nothing)
        @test keys(ps_ll) == (:theta, :error)
    end
end
