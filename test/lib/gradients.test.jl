using Test
using Random
using DeepCompartmentModels
import Zygote # loads the extension that enables `AutoZygote()` (not `using`: it exports another `gradient`)

# Optimisers and Lux are re-exported by DeepCompartmentModels
const Enzyme = DeepCompartmentModels.Enzyme
const ADTypes = DeepCompartmentModels.ADTypes

# Population of one-compartment individuals with an absorption compartment and different doses.
function _gradient_population(n; rng = Random.Xoshiro(1))
    t_obs = Float32[0.5, 1, 2, 4, 6, 8, 12, 24]
    return Population(map(1:n) do i
        cb = generate_dosing_callback(Float32[0 (50 + 25 * (i % 4))], Float32)
        Individual(i, Float32[1, i / n], t_obs, abs.(randn(rng, Float32, length(t_obs))), cb, Float32)
    end)
end

_gradient_nn() = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 3, Lux.softplus))

# Maximum difference between two gradients (any structure that Optimisers can flatten), relative to the largest entry.
function _relerr(a, b)
    x, _ = Optimisers.destructure(a)
    y, _ = Optimisers.destructure(b)
    return maximum(abs, x .- y) / maximum(abs, y)
end

@testset "gradients" begin
    population = _gradient_population(6)
    dcm = DeepCompartmentModel(one_comp_abs!, _gradient_nn(), AdditiveError(Float32[1]); target = 2)
    forward = AutoForwardDiff()

    @testset "Enzyme and ForwardDiff agree for the fixed effect objectives" begin
        for obj in (SSE(), MSE(), LogLikelihood())
            ps, st = setup(obj, Random.Xoshiro(3), dcm)
            loss, ∇ = value_and_gradient(obj, dcm, population, ps, st)
            loss_fd, ∇_fd = value_and_gradient(obj, dcm, population, ps, st; ad = forward)

            @test loss ≈ obj(dcm, population, ps, st)
            @test loss_fd ≈ loss rtol = 1e-4
            @test _relerr(∇, ∇_fd) < 1e-4
            @test keys(∇) == keys(ps)
            @test gradient(obj, dcm, population, ps, st) == ∇
        end
    end

    @testset "Enzyme and ForwardDiff agree for VariationalELBO" begin
        for kwargs in ((; path_deriv = true), (; path_deriv = false), (; path_deriv = false, mean_field = true), (; mean_field = true))
            obj = VariationalELBO([1, 2]; kwargs...)
            ps, st = setup(obj, Random.Xoshiro(3), dcm, population)
            st = update_state(Random.Xoshiro(4), st, obj, dcm, population, ps)

            loss, ∇ = value_and_gradient(obj, dcm, population, ps, st)
            loss_fd, ∇_fd = value_and_gradient(obj, dcm, population, ps, st; ad = forward)

            @test loss ≈ obj(dcm, population, ps, st) rtol = 1e-4
            @test loss_fd ≈ loss rtol = 1e-4
            @test isnothing(∇.omega) && isnothing(∇_fd.omega) # omega is not optimised
            for key in (:theta, :phi, :error)
                @test _relerr(∇[key], ∇_fd[key]) < 1e-3
            end
        end
    end

    @testset "parallel gradients equal the sequential gradient" begin
        for obj in (SSE(), MSE(), LogLikelihood())
            ps, st = setup(obj, Random.Xoshiro(3), dcm)
            loss, ∇ = value_and_gradient(obj, dcm, population, ps, st)
            for kwargs in ((; parallel = :individual), (; parallel = :batch, nbatch = 2), (; parallel = :batch, batchsize = 4))
                loss_par, ∇_par = value_and_gradient(obj, dcm, population, ps, st; kwargs...)
                @test loss_par ≈ loss rtol = 1e-4
                @test _relerr(∇_par, ∇) < 1e-4
            end
        end
    end

    @testset "universal differential equations" begin
        ude_population = _gradient_population(4)
        ude_nn = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
        for inplace in (false, true)
            dcm_ude = DeepCompartmentModel(UniversalDiffEq(1; inplace), ude_nn)
            ps, st = setup(SSE(), Random.Xoshiro(2), dcm_ude)
            loss, ∇ = value_and_gradient(SSE(), dcm_ude, ude_population, ps, st)
            loss_fd, ∇_fd = value_and_gradient(SSE(), dcm_ude, ude_population, ps, st; ad = forward)

            @test loss_fd ≈ loss rtol = 1e-4
            @test _relerr(∇.theta, ∇_fd.theta) < 1e-3
            @test iszero(∇.theta.I) # the infusion rate is set by the callback, it is not a parameter
        end
    end

    @testset "Zygote is an opt-in backend" begin
        zygote = AutoZygote()
        for obj in (SSE(), MSE(), LogLikelihood())
            ps, st = setup(obj, Random.Xoshiro(3), dcm)
            loss, ∇ = value_and_gradient(obj, dcm, population, ps, st)
            loss_z, ∇_z = value_and_gradient(obj, dcm, population, ps, st; ad = zygote)
            @test loss_z ≈ loss rtol = 1e-4
            @test _relerr(∇_z.theta, ∇.theta) < 1e-4
            # Zygote gives `nothing` for parameters that are not used (Enzyme gives zeros)
            obj isa LogLikelihood ? (@test _relerr(∇_z.error, ∇.error) < 1e-4) : (@test isnothing(∇_z.error) || iszero(∇_z.error.σ))
        end

        obj = VariationalELBO([1, 2])
        ps, st = setup(obj, Random.Xoshiro(3), dcm, population)
        st = update_state(Random.Xoshiro(4), st, obj, dcm, population, ps)
        _, ∇ = value_and_gradient(obj, dcm, population, ps, st)
        _, ∇_z = value_and_gradient(obj, dcm, population, ps, st; ad = zygote)
        for key in (:theta, :phi, :error)
            @test _relerr(∇_z[key], ∇[key]) < 1e-3
        end

        # In-place UDEs with an adjoint: this is what Zygote is for, Enzyme cannot do this yet.
        ude_population = _gradient_population(4)
        ude_nn = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 1))
        for sensealg in (ForwardDiffSensitivity(), GaussAdjoint(; autojacvec = ReverseDiffVJP(false)))
            dcm_ude = DeepCompartmentModel(UniversalDiffEq(1; inplace = true), ude_nn; sensealg)
            ps, st = setup(SSE(), Random.Xoshiro(2), dcm_ude)
            _, ∇_fd = value_and_gradient(SSE(), dcm_ude, ude_population, ps, st; ad = forward)
            _, ∇_z = value_and_gradient(SSE(), dcm_ude, ude_population, ps, st; ad = zygote)
            @test _relerr(∇_z.theta, ∇_fd.theta) < 1e-3
            @test iszero(∇_z.theta.I)
        end
    end

    @testset "unsupported backends" begin
        ps, st = setup(SSE(), Random.Xoshiro(3), dcm)
        # Enzyme cannot differentiate through the solvers in forward mode
        @test_throws ArgumentError value_and_gradient(SSE(), dcm, population, ps, st; ad = AutoEnzyme(; mode = Enzyme.Forward))
        @test_throws ArgumentError value_and_gradient(SSE(), dcm, population, ps, st; ad = ADTypes.AutoFiniteDiff())
        # AutoEnzyme() without a mode uses the default reverse mode
        ∇ = gradient(SSE(), dcm, population, ps, st; ad = AutoEnzyme())
        @test _relerr(∇, gradient(SSE(), dcm, population, ps, st)) < 1e-6
    end

    @testset "residual_error_value_and_gradient" begin
        obj = LogLikelihood()
        ps, st = setup(obj, Random.Xoshiro(3), dcm)
        loss_fwd, ∇_fwd = residual_error_value_and_gradient(Random.Xoshiro(1), dcm, population, ps, st; mode = :forward, num_samples = 10)
        loss_rev, ∇_rev = residual_error_value_and_gradient(Random.Xoshiro(1), dcm, population, ps, st; mode = :reverse, num_samples = 10)

        @test loss_rev ≈ loss_fwd rtol = 1e-4
        @test ∇_rev.error.σ ≈ ∇_fwd.error.σ rtol = 1e-4
        @test isnothing(∇_rev.theta)
        @test_throws ArgumentError residual_error_value_and_gradient(Random.Xoshiro(1), dcm, population, ps, st; mode = :backward)
    end

    # Wall-clock limits are too noisy on shared CI runners, so the speed of the backends is only compared
    # with each other, on the same machine and in the same process. Reverse mode costs about one extra
    # solve however many parameters there are, whereas forward mode scales with the number of parameters.
    # On a network with ~800 parameters Enzyme is around 100x faster (measured), the bound is deliberately loose.
    @testset "reverse mode scales better than forward mode with the number of parameters" begin
        wide = DeepCompartmentModel(one_comp_abs!, Lux.Chain(Lux.Dense(2, 128, tanh), Lux.Dense(128, 3, Lux.softplus)),
            AdditiveError(Float32[1]); target = 2) # same layer types as `dcm`, so Enzyme does not compile again
        pop = _gradient_population(20)
        ps, st = setup(SSE(), Random.Xoshiro(3), wide)
        @test length(Optimisers.destructure(ps)[1]) > 700

        fastest(f) = minimum(@elapsed(f()) for _ in 1:3)
        enzyme() = value_and_gradient(SSE(), wide, pop, ps, st)
        forwarddiff() = value_and_gradient(SSE(), wide, pop, ps, st; ad = forward)
        enzyme(); forwarddiff() # compile
        @test fastest(forwarddiff) > 5 * fastest(enzyme)
    end

    @testset "fit accepts an ad backend" begin
        silent = (args...) -> nothing
        ps, st = setup(SSE(), Random.Xoshiro(3), dcm)
        opt = Optimisers.Adam(1f-2)
        _, ps_enzyme, _ = fit(SSE(), dcm, population, opt, ps, st; epochs = 3, callback = silent)
        _, ps_forward, _ = fit(SSE(), dcm, population, opt, ps, st; epochs = 3, callback = silent, ad = forward)
        @test _relerr(ps_enzyme, ps_forward) < 1e-3
    end
end
