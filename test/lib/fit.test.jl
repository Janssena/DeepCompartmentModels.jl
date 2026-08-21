struct InfiniteFixedObjective <: DeepCompartmentModels.FixedObjective end
(::InfiniteFixedObjective)(model, data, ps, st) = Inf

struct ThrowingFixedObjective <: DeepCompartmentModels.FixedObjective end
(::ThrowingFixedObjective)(model, data, ps, st) =
    error("synthetic ODE solve failure")

@testset "minimal fixed-effect fit" begin
    model = toy_dcm()
    population = toy_population(; n = 1)
    callbacks = Tuple{Int,Float32}[]

    result = fit(
        SSE(), model, population, Optimisers.Adam(1.0f-3);
        epochs = 3,
        rng = Random.MersenneTwister(31),
        callback = (epoch, loss) -> push!(callbacks, (epoch, loss)),
    )

    @test result isa FitResult
    @test all(haskey(result, key) for key in (
        :model, :objective, :data, :ps, :st, :optimiser_state, :history,
        :gradient_norm_history, :relative_step_history,
        :relative_objective_change_history, :window_endpoint_change_history,
        :window_projected_drift_history, :window_maximum_step_history,
        :final_loss, :final_epoch, :best_loss, :best_epoch, :selected_loss,
        :selected_epoch, :epochs_completed, :converged, :termination_reason,
        :recovered, :failure, :diagnostics, :metadata))
    @test result.model === model
    @test result.objective isa SSE
    @test result.data === population
    @test length(result.history) == 4
    @test length(result.gradient_norm_history) == 3
    @test length(result.relative_step_history) == 3
    @test length(result.relative_objective_change_history) == 3
    @test length(result.window_endpoint_change_history) == 3
    @test length(result.window_projected_drift_history) == 3
    @test length(result.window_maximum_step_history) == 3
    @test result.epochs_completed == 3
    @test result.final_epoch == 3
    @test result.final_loss == last(result.history)
    @test result.best_loss == minimum(result.history)
    @test result.best_epoch == argmin(result.history) - 1
    @test result.selected_loss == result.final_loss
    @test result.objective_value == result.selected_loss
    @test result.selected_epoch == result.final_epoch
    @test result.selected_iteration == result.selected_epoch
    @test result.best_loss < first(result.history)
    @test !result.converged
    @test result.termination_reason == :maximum_epochs
    @test !result.recovered
    @test isnothing(result.failure)
    @test length(callbacks) == 3
    @test first.(callbacks) == 1:3
    @test all(isfinite, result.history)
    @test all(isfinite, result.gradient_norm_history)
    @test all(isfinite, result.relative_step_history)
    @test niterations(result) == result.epochs_completed
    @test objective_history(result) === result.history
    @test !isconverged(result)
    @test fit_status(result) == :maximum_epochs
    @test result.metadata.effects == :fixed
    @test result.metadata.n_subjects == 1
    @test result.metadata.n_observations == 5
    @test result.metadata.n_outputs == 1
    @test result.metadata.parameter_blocks == (:theta,)
    @test result.metadata.n_fitted_scalars > 0
    @test result.metadata.parameter_type == Float32
    @test occursin("FitResult(SSE", sprint(show, result))

    prediction = predict(model, population, result.ps, result.st)
    @test all(subject -> all(isfinite, subject), prediction)
    @test predict(result) ≈ prediction

    io = IOBuffer()
    Serialization.serialize(io, result)
    seekstart(io)
    restored = Serialization.deserialize(io)
    @test restored isa FitResult
    @test restored.history == result.history
    @test restored.ps == result.ps
    @test restored.metadata == result.metadata

    likelihood_model = toy_dcm(; error = AdditiveError(0.2f0))
    likelihood_result = fit(
        LogLikelihood(), likelihood_model, population, Optimisers.Adam(1.0f-3);
        epochs = 1, rng = Random.MersenneTwister(32))
    @test isfinite(likelihood_result.selected_loss)
    @test :error in keys(likelihood_result.ps)
    @test all(isfinite, likelihood_result.history)
    @test likelihood_result isa FitResult
    @test likelihood_result.metadata.parameter_blocks == (:theta, :error)
end

@testset "fixed convergence uses the whole window" begin
    flat = DeepCompartmentModels._fixed_window_diagnostics(
        fill(10.0, 4), zeros(3), zeros(3);
        patience=3, objective_rel_tol=1e-3, step_rel_tol=1e-3,
        gradient_abs_tol=nothing)
    @test flat.stable

    # Every individual change is below 0.1%, but the accumulated trend across
    # the window is not. A consecutive-counter rule would stop incorrectly.
    drifting = DeepCompartmentModels._fixed_window_diagnostics(
        [10.0, 9.995, 9.990, 9.985], zeros(3), zeros(3);
        patience=3, objective_rel_tol=1e-3, step_rel_tol=1e-3,
        gradient_abs_tol=nothing)
    @test !drifting.stable
    @test drifting.projected_relative_drift > 1e-3
end

@testset "convergence, checkpoint selection, and recovery" begin
    model = toy_dcm()
    population = toy_population(; n = 1)
    objective = MSE()
    ps, st = setup(objective, Random.MersenneTwister(7), model)

    converged = fit(
        objective, model, population, Optimisers.Descent(0.0f0), ps, st;
        epochs = 20,
        min_epochs = 3,
        patience = 3,
        objective_rel_tol = 0,
        step_rel_tol = 0,
    )
    @test converged.converged
    @test converged.termination_reason == :converged
    @test converged.epochs_completed == 3
    @test converged.selected_epoch == 3
    @test all(==(0), converged.relative_step_history)
    @test all(==(0), converged.relative_objective_change_history)

    best_selected = fit(
        objective, model, population, Optimisers.Descent(-1.0f-4), ps, st;
        epochs = 2,
        selection = :best,
    )
    @test best_selected.selected_loss == best_selected.best_loss
    @test best_selected.selected_epoch == best_selected.best_epoch

    recovered = fit(
        objective, model, population, Optimisers.Descent(Inf), ps, st;
        epochs = 2,
        on_failure = :return_best,
    )
    @test recovered.recovered
    @test !recovered.converged
    @test recovered.termination_reason == :recovered_after_failure
    @test recovered.epochs_completed == 0
    @test recovered.selected_epoch == 0
    @test recovered.selected_loss == recovered.best_loss == first(recovered.history)
    @test recovered.failure isa DeepCompartmentModels._FitFailure
    @test !isconverged(recovered)
    @test_throws ArgumentError getproperty(recovered, :not_a_result_property)
end

@testset "fit continuation and failure messages" begin
    model = toy_dcm()
    population = toy_population(; n = 1)
    objective = MSE()
    ps, st = setup(objective, Random.MersenneTwister(7), model)

    first_stage = fit(
        objective, model, population, Optimisers.Adam(1.0f-4), ps, st;
        epochs = 1)
    continued = fit(
        objective, model, population, Optimisers.Adam(1.0f-4),
        first_stage.ps, first_stage.st;
        epochs = 1,
        optimiser_state = first_stage.optimiser_state,
    )
    @test continued.epochs_completed == 1
    @test length(continued.history) == 2
    @test continued.st === st

    @test_throws ArgumentError fit(
        objective, model, population, Optimisers.Adam(), ps, st; epochs = 0)
    @test_throws ArgumentError fit(
        objective, model, population, Optimisers.Adam(), ps, st;
        epochs = 2, min_epochs = 3)
    @test_throws ArgumentError fit(
        objective, model, population, Optimisers.Adam(), ps, st;
        epochs = 2, selection = :unknown)
    @test_throws ArgumentError fit(
        objective, model, population, Optimisers.Adam(), ps, st;
        epochs = 2, on_failure = :unknown)

    failure = try
        fit(InfiniteFixedObjective(), model, population, Optimisers.Adam();
            epochs = 1, rng = Random.MersenneTwister(8))
        nothing
    catch error
        error
    end
    @test failure isa DeepCompartmentModels._FitFailure
    @test occursin("objective during initialization", sprint(showerror, failure))
    @test occursin("not finite", sprint(showerror, failure))

    solve_failure = try
        fit(ThrowingFixedObjective(), model, population, Optimisers.Adam();
            epochs = 1, rng = Random.MersenneTwister(9))
        nothing
    catch error
        error
    end
    @test solve_failure isa DeepCompartmentModels._FitFailure
    @test occursin(
        "objective during initialization", sprint(showerror, solve_failure))
    @test occursin("synthetic ODE solve failure", sprint(showerror, solve_failure))
end
