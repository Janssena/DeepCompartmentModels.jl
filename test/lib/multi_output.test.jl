import Random
using LinearAlgebra: diag

const MO_T = Float64

function mo_ode!(du, u, p, t)
    clearance, volume, second_rate, second_scale, input = unpack(p, t)
    du[1] = input / volume - (clearance / volume) * u[1]
    du[2] = second_scale * input / volume - second_rate * u[2]
end

function mo_population(; n=3, seed=11)
    rng = Random.MersenneTwister(seed)
    individuals = map(1:n) do i
        callback = generate_dosing_callback(MO_T[0.0 100.0 1000.0 0.1], MO_T)
        t1 = MO_T[0.5, 1.0, 2.0, 4.0]
        t2 = MO_T[1.0, 3.0, 4.0]
        y1 = MO_T.(max.(2.0 .* exp.(-0.20 .* t1) .+ 0.02 .* randn(rng, length(t1)), 0.01))
        y2 = MO_T.(max.(1.0 .* exp.(-0.35 .* t2) .+ 0.02 .* randn(rng, length(t2)), 0.01))
        MOIndividual("mo$i", MO_T[65 + 5randn(rng)], [t1, t2], [y1, y2], callback, MO_T)
    end
    Population(individuals)
end

function mo_model(; errors=ErrorModelSet(AdditiveError(0.2), AdditiveError(0.2)))
    network = Lux.Chain(
        Normalize(MO_T[100.0]),
        Lux.Dense(1, 6, Lux.tanh),
        Lux.Dense(6, 4, Lux.softplus),
        Lux.WrappedFunction(x -> MO_T[3.0, 30.0, 0.5, 1.0] .* x),
    )
    DCM(mo_ode!, 2, network, errors; target=[1, 2])
end

@testset "two-DV variational fit cannot skip its M-step" begin
    pop = mo_population(; n=1)
    model = mo_model()
    objective = VariationalELBO([1, 2]; mean_field=true)

    @test_throws ArgumentError fit(
        objective, model, pop, Optimisers.Adam(1e-3);
        n_outer=1, n_inner=1, monitor_samples=2,
        m_step_kwargs=(epochs=0, num_samples=2), verbose=false)
    @test_throws ArgumentError setup(
        VariationalELBO([1]; natural=true), Random.MersenneTwister(14),
        model, pop, MO_T)

    result = fit(
        objective, model, pop, Optimisers.Adam(1e-3);
        n_outer=1, n_inner=1, gradient_samples=1, monitor_samples=2,
        parameter_type=MO_T, init_omega=MO_T[0.1, 0.1],
        m_step_kwargs=(epochs=1, num_samples=2, opt=Optimisers.Adam(1e-3)),
        rng=Random.MersenneTwister(15), verbose=false)

    @test result.selection == :final
    @test result.outer_iterations_completed == 1
    @test length(result.monitor_history) == 2
    @test length(result.monitor_mcse_history) == 2
    @test length(result.monitor_difference_mcse_history) == 2
    @test isfinite(result.monitor_difference_mcse_history[2])
    @test length(result.monitor_window_endpoint_history) == 1
    @test isnan(result.monitor_window_endpoint_history[1])
    @test length(result.monitor_window_drift_history) == 1
    @test length(result.monitor_window_mcse_history) == 1
    @test length(result.outer_relative_step_history) == 1
    @test isfinite(result.outer_relative_step_history[1])
    @test length(result.m_step_history) == 1
    @test result.m_step_history[1].residual.executed
    @test result.m_step_history[1].omega_updated
    @test haskey(result, :checkpoint)
end

@testset "MOIndividual with a missing dependent variable is skipped, not evaluated" begin
    # A subject measured in only ONE matrix (empty second output) must contribute only its
    # present output to the likelihood; logpdf of an empty distribution is type-unstable under
    # AD, so _logpdf skips empty outputs. Regression for the joint plasma/whole-blood use-case.
    callback = generate_dosing_callback(MO_T[0.0 100.0 1000.0 0.1], MO_T)
    t1 = MO_T[0.5, 1.0, 2.0, 4.0]
    both = MOIndividual("both", MO_T[65.0], [t1, MO_T[1.0, 3.0]],
                        [MO_T[1.8, 1.4, 0.9, 0.5], MO_T[0.9, 0.4]], callback, MO_T)
    plasma_only = MOIndividual("po", MO_T[70.0], [t1, MO_T[]],
                               [MO_T[1.7, 1.3, 0.8, 0.45], MO_T[]], callback, MO_T)
    model = mo_model()
    ps, st = setup(LogLikelihood(), Random.MersenneTwister(3), model, MO_T)

    ll_mixed = DeepCompartmentModels.loglikelihood(model, Population([both, plasma_only]), ps, st)
    @test isfinite(ll_mixed)
    # the plasma-only subject contributes exactly its first-output likelihood (empty DV omitted)
    ll_both = DeepCompartmentModels.loglikelihood(model, both, ps, st)
    ll_po = DeepCompartmentModels.loglikelihood(model, plasma_only, ps, st)
    @test ll_mixed ≈ ll_both + ll_po
    # the fit's Zygote gradient path must run end-to-end through the empty output
    result = fit(LogLikelihood(), model, Population([both, plasma_only]),
                 Optimisers.Adam(1e-2); epochs = 2, rng = Random.MersenneTwister(5))
    @test result isa FitResult
    @test all(isfinite, result.history)
end

@testset "windowed mixed convergence and exact continuation" begin
    flat_history = fill(10.0, 4)
    flat_draws = [Float64[9, 11] for _ in flat_history]
    flat = DeepCompartmentModels._mixed_window_diagnostics(
        flat_history, flat_draws, fill(1e-4, 3);
        patience=3, monitor_rel_tol=1e-5, monitor_abs_tol=1e-6,
        monitor_mcse_multiplier=2, outer_step_rel_tol=1e-3)
    @test flat.stable
    @test flat.projected_drift == 0

    drifting_history = [10.0, 9.9, 9.8, 9.7]
    drifting_draws = [Float64[value - 0.1, value + 0.1]
                      for value in drifting_history]
    drifting = DeepCompartmentModels._mixed_window_diagnostics(
        drifting_history, drifting_draws, fill(1e-4, 3);
        patience=3, monitor_rel_tol=1e-2, monitor_abs_tol=1e-6,
        monitor_mcse_multiplier=2, outer_step_rel_tol=1e-3)
    @test !drifting.stable
    @test drifting.projected_drift > drifting.objective_threshold

    pop = mo_population(; n=1)
    model = mo_model()
    objective = VariationalELBO([1, 2]; mean_field=true)
    options = (
        n_inner=1, gradient_samples=1, monitor_samples=2, patience=2,
        parameter_type=MO_T, init_omega=MO_T[0.1, 0.1],
        m_step_kwargs=(epochs=1, num_samples=2, opt=Optimisers.Adam(1e-3)),
        verbose=false,
    )
    saved_checkpoints = Any[]
    staged = fit(
        objective, model, pop, Optimisers.Adam(1e-3);
        n_outer=1, options..., rng=Random.MersenneTwister(81),
        monitor_rng=Random.MersenneTwister(82),
        learning_rate_schedule=outer -> 1e-3 / outer,
        checkpoint_callback=(outer, checkpoint) -> push!(saved_checkpoints, checkpoint))
    @test staged isa FitResult
    @test staged.metadata.effects == :mixed
    @test staged.metadata.n_subjects == 1
    @test staged.metadata.n_outputs == 2
    @test staged.metadata.n_observations == sum(length, get_y(only(pop)))
    @test staged.metadata.parameter_blocks == (:theta, :error, :omega, :phi)
    @test staged.objective_value == last(staged.monitor_history)
    @test niterations(staged) == staged.outer_iterations_completed
    @test fit_status(staged) == :maximum_outer_iterations
    @test length(saved_checkpoints) == 1
    @test length(only(saved_checkpoints).m_step_history) == 1
    resumed = fit(
        objective, model, pop, Optimisers.Adam(1e-3);
        n_outer=2, options..., resume_from=staged.checkpoint,
        learning_rate_schedule=outer -> 1e-3 / outer)
    uninterrupted = fit(
        objective, model, pop, Optimisers.Adam(1e-3);
        n_outer=2, options..., rng=Random.MersenneTwister(81),
        monitor_rng=Random.MersenneTwister(82),
        learning_rate_schedule=outer -> 1e-3 / outer)

    @test resumed.outer_iterations_completed == 2
    @test resumed isa FitResult
    @test resumed.metadata.resumed
    @test length(resumed.monitor_history) == 3
    @test resumed.monitor_history == uninterrupted.monitor_history
    @test resumed.monitor_mcse_history == uninterrupted.monitor_mcse_history
    @test resumed.outer_relative_step_history == uninterrupted.outer_relative_step_history
    @test resumed.learning_rate_history == [1e-3, 5e-4]
    @test collect(DeepCompartmentModels.ComponentVector(resumed.ps)) ≈
          collect(DeepCompartmentModels.ComponentVector(uninterrupted.ps))
    @test_throws ArgumentError fit(
        objective, model, pop, Optimisers.Adam(1e-3);
        n_outer=1, options..., resume_from=staged.checkpoint)
end

@testset "multi-output construction and objective paths" begin
    pop = mo_population()
    model = mo_model()
    rng = Random.MersenneTwister(12)

    @test pop isa Population{<:MOIndividual}
    @test size(get_x(pop)) == (1, length(pop))
    @test_throws ArgumentError MOIndividual(
        "bad", MO_T[70], [MO_T[1], MO_T[1]], [MO_T[1]], pop[1].callback, MO_T)
    with_occasions = MOIndividual(
        "occasions", MO_T[70], [MO_T[1], MO_T[1]], [MO_T[1], MO_T[1]],
        pop[1].callback, MO_T; occasions=true)
    @test with_occasions.occasions === true

    ps_ll, st_ll = setup(LogLikelihood(), rng, model, MO_T)
    predictions = predict(model, pop, ps_ll, st_ll)
    @test length(predictions) == length(pop)
    @test length(first(predictions)) == 2
    @test length(first(predictions)[2]) == length(get_y(first(pop))[2])

    for objective in (MSE(), SSE(), LogLikelihood())
        ps, st = setup(objective, rng, model, MO_T)
        value = objective(model, pop, ps, st)
        grad = gradient(objective, model, pop, ps, st)
        @test isfinite(value)
        gradient_values = filter(!isnothing, collect(DeepCompartmentModels.ComponentVector(grad)))
        @test !isempty(gradient_values)
        @test all(isfinite, gradient_values)
    end

    one_error = ErrorModelSet(AdditiveError(0.2))
    @test_throws DimensionMismatch make_dist(one_error, first(predictions), (σ=[[0.0]],))
    @test_throws DimensionMismatch setup(model.error, [[0.2]])

    batch_gradient = gradient(
        LogLikelihood(), model, pop, ps_ll, st_ll; parallel=:batch, batchsize=2)
    batch_values = filter(!isnothing, collect(DeepCompartmentModels.ComponentVector(batch_gradient)))
    @test all(isfinite, batch_values)

    lognormal_error = CustomError([0.2]; model=(ŷ, ps, st; kwargs...) ->
        product_distribution(LogNormal.(log.(max.(ŷ, eps(eltype(ŷ)))), log1p(exp(only(ps.σ))))))
    custom_set = ErrorModelSet(lognormal_error, AdditiveError(0.2))
    custom_ps = setup(custom_set, nothing)
    custom_dists = make_dist(custom_set, first(predictions), custom_ps)
    @test length(custom_dists) == 2
    @test isfinite(sum(logpdf.(custom_dists, get_y(first(pop)))))
end

@testset "two-DV residual-error and omega M-step execute" begin
    pop = mo_population()
    model = mo_model()
    objective = VariationalELBO([1, 2]; mean_field=true)
    rng = Random.MersenneTwister(13)
    ps, st = setup(objective, rng, model, pop, MO_T;
                   init_omega=MO_T[0.1, 0.1], scale=0.1)

    loss, grad = residual_error_value_and_gradient(
        rng, model, pop, ps, st; num_samples=3, mode=:forward)
    @test isfinite(loss)
    @test all(isfinite, collect(DeepCompartmentModels.ComponentVector(grad.error)))
    @test_throws ArgumentError optimise_residual_error(
        objective, rng, model, pop, ps, st; epochs=0)

    omega_before = copy(ps.omega)
    ps_after, diagnostics = m_step(
        objective, rng, model, pop, ps, st;
        epochs=2, num_samples=3, opt=Optimisers.Adam(1e-3),
        verbose=false, return_diagnostics=true)

    @test diagnostics.residual.executed
    @test diagnostics.residual.epochs == 2
    @test diagnostics.residual.num_samples == 3
    @test diagnostics.residual.reused_prediction_draws
    @test length(diagnostics.residual.objective_history) == 2
    @test diagnostics.residual.error_changed
    @test diagnostics.omega_updated
    @test ps_after.omega != omega_before
    @test all(>(0), diag(Matrix(ps_after.omega)))
end
