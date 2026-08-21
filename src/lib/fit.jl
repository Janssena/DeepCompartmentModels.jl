struct _FitFailure <: Exception
    stage::Symbol
    epoch::Int
    cause
end

function _mixed_gradient_samples(rng, objective, model, data, ps, st, num_samples)
    gradients = map(1:num_samples) do _
        update_epsilon!(rng, st)
        gradient(objective, model, data, ps, st)
    end
    summed = fmap(_sum_grads, gradients...; exclude=_isleaf_or_vec_of_arrays)
    return fmap(x -> isnothing(x) ? nothing : x ./ num_samples, summed;
                exclude=_isleaf_or_vec_of_arrays)
end

function _monitor_states(rng, st, num_samples)
    return map(1:num_samples) do _
        state = deepcopy(st)
        update_epsilon!(rng, state)
        state
    end
end

function _monitor_elbo(objective, model, data, ps, states)
    losses = Float64[objective(model, data, ps, state) for state in states]
    all(isfinite, losses) || throw(ErrorException("ELBO monitoring produced non-finite values."))
    estimate = mean(losses)
    mcse = length(losses) == 1 ? NaN :
        sqrt(sum(abs2, losses .- estimate) / (length(losses) * (length(losses) - 1)))
    return estimate, mcse, losses
end

function _paired_mcse(current, previous)
    differences = current .- previous
    estimate = mean(differences)
    return length(differences) == 1 ? NaN :
        sqrt(sum(abs2, differences .- estimate) /
             (length(differences) * (length(differences) - 1)))
end

function _projected_window_drift(values)
    x = collect(0:(length(values) - 1))
    x_centered = x .- mean(x)
    slope = sum(x_centered .* (values .- mean(values))) / sum(abs2, x_centered)
    return abs(slope) * (length(values) - 1)
end

function _mixed_window_diagnostics(
    monitor_history,
    monitor_draw_history,
    outer_relative_step_history;
    patience,
    monitor_rel_tol,
    monitor_abs_tol,
    monitor_mcse_multiplier,
    outer_step_rel_tol,
)
    if length(outer_relative_step_history) < patience
        return (stable=false, endpoint_change=NaN, projected_drift=NaN,
                paired_mcse=NaN, objective_threshold=NaN,
                stochastic_threshold=NaN, maximum_parameter_step=NaN)
    end

    first_index = length(monitor_history) - patience
    recent_objectives = monitor_history[first_index:end]
    projected_drift = _projected_window_drift(recent_objectives)
    endpoint_change = abs(last(recent_objectives) - first(recent_objectives))
    paired_mcse = _paired_mcse(last(monitor_draw_history),
                               monitor_draw_history[first_index])
    objective_threshold = max(
        monitor_rel_tol * max(abs(first(recent_objectives)),
                              abs(last(recent_objectives)), 1.0),
        monitor_abs_tol)
    stochastic_threshold = monitor_mcse_multiplier * paired_mcse + monitor_abs_tol
    maximum_parameter_step = maximum(last(outer_relative_step_history, patience))
    stable = endpoint_change <= objective_threshold &&
             projected_drift <= objective_threshold &&
             endpoint_change <= stochastic_threshold &&
             projected_drift <= stochastic_threshold &&
             maximum_parameter_step <= outer_step_rel_tol
    return (; stable, endpoint_change, projected_drift, paired_mcse,
            objective_threshold, stochastic_threshold, maximum_parameter_step)
end

function _mixed_checkpoint(
    ps, st, optimiser_state, rng, monitoring_states, monitor_history,
    monitor_mcse_history, monitor_difference_mcse_history, monitor_draw_history,
    outer_relative_step_history, inner_loss_history, m_step_history,
    learning_rate_history,
)
    return (
        ps=deepcopy(ps), st=deepcopy(st), optimiser_state=deepcopy(optimiser_state),
        rng=deepcopy(rng), monitoring_states=deepcopy(monitoring_states),
        monitor_history=copy(monitor_history),
        monitor_mcse_history=copy(monitor_mcse_history),
        monitor_difference_mcse_history=copy(monitor_difference_mcse_history),
        monitor_draw_history=deepcopy(monitor_draw_history),
        outer_relative_step_history=copy(outer_relative_step_history),
        inner_loss_history=copy(inner_loss_history),
        m_step_history=copy(m_step_history),
        learning_rate_history=copy(learning_rate_history),
    )
end

"""
    fit(objective::VariationalELBO, model, population, optimiser; kwargs...)

Run the package's mixed-effects variational workflow. Each outer iteration uses
stochastic variational gradients for `theta` and `phi`, then performs a mandatory
residual-error optimisation and analytic omega update. Convergence is monitored
with a larger, fixed set of independent standard-normal draws (common random
numbers across outer iterations); no single-draw "best ELBO" checkpoint is
selected or restored.

This is a hybrid stochastic variational / M-step algorithm: `theta` and `phi`
are updated jointly inside each outer iteration, while residual error and omega
are updated only in the explicit M-step.

For a real modelling run (rather than a quick exploration) raise `n_outer` well
above the default (e.g. several hundred) and use a larger learning rate for the
optimiser (`Optimisers.Adam(0.1)` is a reasonable starting point for
mixed-effect estimation); the stochastic variational updates converge more
slowly than a fixed-effect fit. The returned `FitResult` is usable regardless of
whether the windowed convergence criterion was met; check `isconverged(result)`.

Convergence requires endpoint and fitted-trend stability across the complete
`patience` window, compatibility with the paired monitoring Monte Carlo error,
and small parameter movement throughout the window. `resume_from` accepts the
complete `checkpoint` returned by an earlier call; `n_outer` is then interpreted
as the total outer-iteration cap. `checkpoint_callback(outer, checkpoint)` can
persist intermediate state. `learning_rate_schedule(outer)` may return a
positive finite rate and is converted to `parameter_type` before the optimizer
is adjusted.
"""
function fit(
    objective::VariationalELBO,
    model::DeepCompartmentModel,
    data::Population,
    optimiser::Optimisers.AbstractRule;
    n_outer::Integer=20,
    n_inner::Integer=20,
    gradient_samples::Integer=1,
    monitor_samples::Integer=20,
    min_outer::Integer=min(3, n_outer),
    patience::Integer=5,
    monitor_rel_tol::Real=1e-3,
    monitor_abs_tol::Real=1e-3,
    monitor_mcse_multiplier::Real=2.0,
    outer_step_rel_tol::Real=1e-3,
    rng::Random.AbstractRNG=Random.default_rng(),
    monitor_rng::Union{Nothing,Random.AbstractRNG}=nothing,
    parameter_type::Type{<:AbstractFloat}=Float32,
    init_omega=0.1,
    init_sigma=nothing,
    params::Parameterisation=MeanSqrt(),
    scale::Real=0.1,
    m_step_kwargs=(epochs=20, num_samples=20),
    verbose::Bool=true,
    callback=nothing,
    checkpoint_callback=nothing,
    resume_from=nothing,
    learning_rate_schedule=nothing,
)
    n_outer > 0 || throw(ArgumentError("n_outer must be positive."))
    n_inner > 0 || throw(ArgumentError("n_inner must be positive."))
    gradient_samples > 0 || throw(ArgumentError("gradient_samples must be positive."))
    monitor_samples > 1 || throw(ArgumentError("monitor_samples must be at least 2 to estimate Monte Carlo error."))
    0 < min_outer <= n_outer || throw(ArgumentError("min_outer must be between 1 and n_outer."))
    patience >= 2 || throw(ArgumentError("patience must be at least 2 for windowed convergence."))
    monitor_rel_tol >= 0 || throw(ArgumentError("monitor_rel_tol must be non-negative."))
    monitor_abs_tol >= 0 || throw(ArgumentError("monitor_abs_tol must be non-negative."))
    monitor_mcse_multiplier >= 0 || throw(ArgumentError("monitor_mcse_multiplier must be non-negative."))
    outer_step_rel_tol >= 0 || throw(ArgumentError("outer_step_rel_tol must be non-negative."))
    haskey(m_step_kwargs, :epochs) || throw(ArgumentError(
        "m_step_kwargs must explicitly include a positive epochs value."))
    m_step_kwargs.epochs > 0 || throw(ArgumentError(
        "The residual-error M-step cannot be skipped: m_step_kwargs.epochs must be positive."))

    if resume_from === nothing
        ps, st = setup(objective, rng, model, data, parameter_type;
                           init_omega, init_sigma, params, scale)
        optimiser_state = Optimisers.setup(optimiser, ps)
        monitor_rng = monitor_rng === nothing ? Random.MersenneTwister(rand(rng, UInt)) : monitor_rng
        monitoring_states = _monitor_states(monitor_rng, st, monitor_samples)
        initial_monitor, initial_mcse, initial_monitor_draws = _monitor_elbo(
            objective, model, data, ps, monitoring_states)
        monitor_history = Float64[initial_monitor]
        monitor_mcse_history = Float64[initial_mcse]
        monitor_difference_mcse_history = Float64[NaN]
        monitor_draw_history = [initial_monitor_draws]
        outer_relative_step_history = Float64[]
        inner_loss_history = Float64[]
        m_step_history = NamedTuple[]
        learning_rate_history = Float64[]
    else
        required = (:ps, :st, :optimiser_state, :rng, :monitoring_states,
                    :monitor_history, :monitor_mcse_history,
                    :monitor_difference_mcse_history, :monitor_draw_history,
                    :outer_relative_step_history, :inner_loss_history,
                    :m_step_history)
        all(key -> haskey(resume_from, key), required) || throw(ArgumentError(
            "resume_from is not a complete mixed-fit checkpoint."))
        ps = deepcopy(resume_from.ps)
        st = deepcopy(resume_from.st)
        optimiser_state = deepcopy(resume_from.optimiser_state)
        rng = deepcopy(resume_from.rng)
        monitoring_states = deepcopy(resume_from.monitoring_states)
        monitor_history = copy(resume_from.monitor_history)
        monitor_mcse_history = copy(resume_from.monitor_mcse_history)
        monitor_difference_mcse_history = copy(resume_from.monitor_difference_mcse_history)
        monitor_draw_history = deepcopy(resume_from.monitor_draw_history)
        outer_relative_step_history = copy(resume_from.outer_relative_step_history)
        inner_loss_history = copy(resume_from.inner_loss_history)
        m_step_history = copy(resume_from.m_step_history)
        learning_rate_history = haskey(resume_from, :learning_rate_history) ?
            copy(resume_from.learning_rate_history) : fill(NaN, length(m_step_history))
        length(first(monitor_draw_history)) == monitor_samples || throw(ArgumentError(
            "monitor_samples does not match the resumed checkpoint."))
        length(m_step_history) < n_outer || throw(ArgumentError(
            "n_outer must exceed the $(length(m_step_history)) completed checkpoint iterations."))
    end

    monitor_window_endpoint_history = Float64[]
    monitor_window_drift_history = Float64[]
    monitor_window_mcse_history = Float64[]
    for completed in eachindex(m_step_history)
        window = _mixed_window_diagnostics(
            monitor_history[1:(completed + 1)],
            monitor_draw_history[1:(completed + 1)],
            outer_relative_step_history[1:completed];
            patience, monitor_rel_tol, monitor_abs_tol,
            monitor_mcse_multiplier, outer_step_rel_tol)
        push!(monitor_window_endpoint_history, window.endpoint_change)
        push!(monitor_window_drift_history, window.projected_drift)
        push!(monitor_window_mcse_history, window.paired_mcse)
    end
    converged = false
    termination_reason = :maximum_outer_iterations

    for outer in (length(m_step_history) + 1):n_outer
        if learning_rate_schedule !== nothing
            scheduled_learning_rate = learning_rate_schedule(outer)
            scheduled_learning_rate isa Real && isfinite(scheduled_learning_rate) &&
                scheduled_learning_rate > 0 ||
                throw(ArgumentError(
                    "learning_rate_schedule($outer) must return a positive finite Real."))
            learning_rate = parameter_type(scheduled_learning_rate)
            isfinite(learning_rate) && learning_rate > 0 ||
                throw(ArgumentError(
                    "learning_rate_schedule($outer) is not representable as a positive finite $parameter_type."))
            Optimisers.adjust!(optimiser_state, learning_rate)
            push!(learning_rate_history, Float64(learning_rate))
        else
            push!(learning_rate_history, NaN)
        end
        outer_start_ps = deepcopy(ps)
        for inner in 1:n_inner
            parameter_gradient = _mixed_gradient_samples(
                rng, objective, model, data, ps, st, gradient_samples)
            # Residual error has one explicit owner: the M-step below.
            parameter_gradient = Accessors.@set parameter_gradient.error = nothing
            _fit_all_finite(parameter_gradient) || throw(_FitFailure(
                :variational_gradient, (outer - 1) * n_inner + inner,
                ErrorException("gradient contained non-finite values")))
            optimiser_state, ps = Optimisers.update(
                optimiser_state, ps, parameter_gradient)
            _fit_all_finite(ps) || throw(_FitFailure(
                :variational_update, (outer - 1) * n_inner + inner,
                ErrorException("updated parameters contained non-finite values")))
            training_loss = objective(model, data, ps, st)
            isfinite(training_loss) || throw(_FitFailure(
                :variational_objective, (outer - 1) * n_inner + inner,
                ErrorException("objective was not finite")))
            push!(inner_loss_history, Float64(training_loss))
        end

        ps, diagnostics = m_step(
            objective, rng, model, data, ps, st;
            m_step_kwargs..., verbose=false, return_diagnostics=true)
        push!(m_step_history, diagnostics)
        outer_relative_step = _fit_relative_step(outer_start_ps, ps)
        push!(outer_relative_step_history, outer_relative_step)

        monitored, mcse, monitor_draws = _monitor_elbo(
            objective, model, data, ps, monitoring_states)
        difference_mcse = _paired_mcse(monitor_draws, last(monitor_draw_history))
        push!(monitor_history, monitored)
        push!(monitor_mcse_history, mcse)
        push!(monitor_difference_mcse_history, difference_mcse)
        push!(monitor_draw_history, monitor_draws)
        callback === nothing || callback(outer, monitored, mcse, diagnostics)
        verbose && println("Outer $outer, monitored negative ELBO = $monitored (MCSE = $mcse)")

        window = _mixed_window_diagnostics(
            monitor_history, monitor_draw_history, outer_relative_step_history;
            patience, monitor_rel_tol, monitor_abs_tol,
            monitor_mcse_multiplier, outer_step_rel_tol)
        push!(monitor_window_endpoint_history, window.endpoint_change)
        push!(monitor_window_drift_history, window.projected_drift)
        push!(monitor_window_mcse_history, window.paired_mcse)
        if checkpoint_callback !== nothing
            checkpoint_callback(outer, _mixed_checkpoint(
                ps, st, optimiser_state, rng, monitoring_states, monitor_history,
                monitor_mcse_history, monitor_difference_mcse_history,
                monitor_draw_history, outer_relative_step_history,
                inner_loss_history, m_step_history, learning_rate_history))
        end
        if outer >= min_outer && window.stable
            converged = true
            termination_reason = :converged
            break
        end
    end

    checkpoint = _mixed_checkpoint(
        ps, st, optimiser_state, rng, monitoring_states, monitor_history,
        monitor_mcse_history, monitor_difference_mcse_history,
        monitor_draw_history, outer_relative_step_history,
        inner_loss_history, m_step_history, learning_rate_history)
    diagnostics = (
        monitor_history,
        monitor_mcse_history,
        monitor_difference_mcse_history,
        monitor_window_endpoint_history,
        monitor_window_drift_history,
        monitor_window_mcse_history,
        outer_relative_step_history,
        inner_loss_history,
        m_step_history,
        learning_rate_history,
        outer_iterations_completed=length(m_step_history),
        checkpoint,
    )
    return FitResult(
        model, objective, data, ps, st;
        optimiser_state,
        optimiser,
        objective_value=last(monitor_history),
        history=monitor_history,
        iterations=length(m_step_history),
        selected_iteration=length(m_step_history),
        converged,
        termination_reason,
        recovered=false,
        failure=nothing,
        selection=:final,
        diagnostics,
        metadata=(
            parameter_type=parameter_type,
            gradient_samples=gradient_samples,
            monitor_samples=monitor_samples,
            resumed=resume_from !== nothing,
        ),
    )
end

function Base.showerror(io::IO, error::_FitFailure)
    location = error.epoch == 0 ? "during initialization" : "at epoch $(error.epoch)"
    print(io, "fit failed during $(error.stage) $location: ")
    showerror(io, error.cause)
end

_fit_all_finite(::Nothing) = true
_fit_all_finite(value::Number) = isfinite(value)
_fit_all_finite(values::AbstractArray) = all(_fit_all_finite, values)
_fit_all_finite(tree::NamedTuple) = all(_fit_all_finite, values(tree))
_fit_all_finite(tree::Tuple) = all(_fit_all_finite, tree)

_fit_sumsq(::Nothing) = 0.0
_fit_sumsq(value::Number) = abs2(Float64(value))
_fit_sumsq(values::AbstractArray{<:Number}) = sum(abs2, Float64.(values))
_fit_sumsq(values::AbstractArray) = sum(_fit_sumsq, values; init=0.0)
_fit_sumsq(tree::NamedTuple) = sum(_fit_sumsq, values(tree); init = 0.0)
_fit_sumsq(tree::Tuple) = sum(_fit_sumsq, tree; init = 0.0)

_fit_difference_sumsq(a::Number, b::Number) = abs2(Float64(a) - Float64(b))
_fit_difference_sumsq(a::AbstractArray{<:Number}, b::AbstractArray{<:Number}) =
    sum(abs2, Float64.(a) .- Float64.(b))
_fit_difference_sumsq(a::AbstractArray, b::AbstractArray) =
    sum((_fit_difference_sumsq(x, y) for (x, y) in zip(a, b)); init=0.0)
_fit_difference_sumsq(a::NamedTuple, b::NamedTuple) =
    sum((_fit_difference_sumsq(a[key], b[key]) for key in keys(a)); init = 0.0)
_fit_difference_sumsq(a::Tuple, b::Tuple) =
    sum((_fit_difference_sumsq(x, y) for (x, y) in zip(a, b)); init = 0.0)

_fit_norm(tree) = sqrt(_fit_sumsq(tree))
_fit_relative_step(previous, candidate) =
    sqrt(_fit_difference_sumsq(previous, candidate)) / max(_fit_norm(previous), 1.0)
_fit_relative_objective_change(previous, candidate) =
    abs(Float64(candidate) - Float64(previous)) /
    max(abs(Float64(previous)), abs(Float64(candidate)), 1.0)

function _fixed_window_diagnostics(
    history,
    relative_step_history,
    gradient_norm_history;
    patience,
    objective_rel_tol,
    step_rel_tol,
    gradient_abs_tol,
)
    if length(relative_step_history) < patience
        return (stable=false, endpoint_relative_change=NaN,
                projected_relative_drift=NaN, maximum_parameter_step=NaN,
                maximum_gradient_norm=NaN)
    end

    recent_objectives = last(history, patience + 1)
    scale = max(abs(first(recent_objectives)), abs(last(recent_objectives)), 1.0)
    endpoint_relative_change =
        abs(last(recent_objectives) - first(recent_objectives)) / scale
    projected_relative_drift = _projected_window_drift(recent_objectives) / scale
    maximum_parameter_step = maximum(last(relative_step_history, patience))
    maximum_gradient_norm = maximum(last(gradient_norm_history, patience))
    stable = endpoint_relative_change <= objective_rel_tol &&
             projected_relative_drift <= objective_rel_tol &&
             maximum_parameter_step <= step_rel_tol &&
             (gradient_abs_tol === nothing ||
              maximum_gradient_norm <= gradient_abs_tol)
    return (; stable, endpoint_relative_change, projected_relative_drift,
            maximum_parameter_step, maximum_gradient_norm)
end

function _fit_evaluate(objective, model, data, ps, st, epoch)
    loss = try
        objective(model, data, ps, st)
    catch cause
        cause isa InterruptException && rethrow()
        throw(_FitFailure(:objective, epoch, cause))
    end

    isfinite(loss) || throw(_FitFailure(
        :objective, epoch, ErrorException("objective was not finite: $loss")))
    return loss
end

function _validate_fit_options(
    epochs,
    min_epochs,
    patience,
    objective_rel_tol,
    step_rel_tol,
    gradient_abs_tol,
    selection,
    on_failure,
)
    epochs > 0 || throw(ArgumentError("epochs must be positive, got $epochs"))
    min_epochs > 0 || throw(ArgumentError("min_epochs must be positive, got $min_epochs"))
    min_epochs <= epochs || throw(ArgumentError(
        "min_epochs ($min_epochs) cannot exceed epochs ($epochs)"))
    patience > 0 || throw(ArgumentError("patience must be positive, got $patience"))
    objective_rel_tol >= 0 || throw(ArgumentError(
        "objective_rel_tol must be non-negative, got $objective_rel_tol"))
    step_rel_tol >= 0 || throw(ArgumentError(
        "step_rel_tol must be non-negative, got $step_rel_tol"))
    (gradient_abs_tol === nothing || gradient_abs_tol >= 0) || throw(ArgumentError(
        "gradient_abs_tol must be nothing or non-negative, got $gradient_abs_tol"))
    selection in (:final, :best) || throw(ArgumentError(
        "selection must be :final or :best, got $selection"))
    on_failure in (:throw, :return_best) || throw(ArgumentError(
        "on_failure must be :throw or :return_best, got $on_failure"))
    return nothing
end

"""
    fit(objective::FixedObjective, model, data, optimiser; kwargs...)

Fit a fixed-effects model through the package's canonical
`setup → gradient → Optimisers.update` path. The final iterate is returned by
default; convergence and any recovery are reported explicitly.

Supported objectives are `MSE`, `SSE`, and `LogLikelihood`. Mixed-effects
objectives are intentionally excluded from this method; they require the
Variational-EM workflow.

# Keywords

- `epochs=100`: Maximum number of optimizer updates.
- `min_epochs=min(10, epochs)`: Earliest update at which convergence can stop.
- `patience=10`: Width of the stable convergence window. Both endpoint change
  and fitted drift across the whole window must pass.
- `objective_rel_tol=1e-6`: Maximum relative objective change for a stable update.
- `step_rel_tol=1e-6`: Maximum relative parameter-step norm for a stable update.
- `gradient_abs_tol=nothing`: Optional additional absolute gradient-norm threshold.
- `selection=:final`: Return `:final` parameters by default. `:best` explicitly
  requests the lowest deterministic training-objective checkpoint.
- `on_failure=:throw`: Throw a numerical fitting failure. `:return_best` returns
  the best checkpoint with `recovered=true` and `converged=false`.
- `rng=Random.default_rng()`: Random generator used for parameter setup.
- `parameter_type=Float32`: Floating-point type passed to `setup`.
- `init_sigma=nothing`: Optional residual-error initialization passed to `setup`.
- `params=MeanSqrt()`: Parameterisation passed to `setup`.
- `callback=nothing`: Optional `callback(epoch, loss)` called after each update.

# Returns

A [`FitResult`](@ref) containing the selected parameters/state and optimizer
state; objective, gradient, and relative-step histories; final/best/selected
losses and epochs; and explicit `converged`, `termination_reason`, and
`recovered` fields. `history[1]` is the initial objective.
"""
function fit(
    objective::FixedObjective,
    model::DeepCompartmentModel,
    data::Population,
    optimiser::Optimisers.AbstractRule;
    epochs::Integer = 100,
    min_epochs::Integer = min(10, epochs),
    patience::Integer = 10,
    objective_rel_tol::Real = 1e-6,
    step_rel_tol::Real = 1e-6,
    gradient_abs_tol::Union{Nothing,Real} = nothing,
    selection::Symbol = :final,
    on_failure::Symbol = :throw,
    rng::Random.AbstractRNG = Random.default_rng(),
    parameter_type::Type{<:AbstractFloat} = Float32,
    init_sigma = nothing,
    params::Parameterisation = MeanSqrt(),
    callback = nothing,
)
    ps, st = setup(
        objective, rng, model, parameter_type; init_sigma, params)
    return fit(
        objective, model, data, optimiser, ps, st;
        epochs, min_epochs, patience, objective_rel_tol, step_rel_tol,
        gradient_abs_tol, selection, on_failure, callback,
        metadata=(parameter_type=parameter_type, initialized_by=:setup),
    )
end

"""
    fit(objective::FixedObjective, model, data, optimiser, ps, st; kwargs...)

Continue a fixed-effects fit from explicitly supplied parameters and state.
Pass the previously returned `optimiser_state` to continue Adam-like optimizers
without discarding their accumulated state.
"""
function fit(
    objective::FixedObjective,
    model::DeepCompartmentModel,
    data::Population,
    optimiser::Optimisers.AbstractRule,
    ps::NamedTuple,
    st::NamedTuple;
    epochs::Integer = 100,
    min_epochs::Integer = min(10, epochs),
    patience::Integer = 10,
    objective_rel_tol::Real = 1e-6,
    step_rel_tol::Real = 1e-6,
    gradient_abs_tol::Union{Nothing,Real} = nothing,
    selection::Symbol = :final,
    on_failure::Symbol = :throw,
    optimiser_state = nothing,
    callback = nothing,
    metadata::NamedTuple = NamedTuple(),
)
    _validate_fit_options(
        epochs, min_epochs, patience, objective_rel_tol, step_rel_tol,
        gradient_abs_tol, selection, on_failure)

    if optimiser_state === nothing
        optimiser_state = try
            Optimisers.setup(optimiser, ps)
        catch cause
            cause isa InterruptException && rethrow()
            throw(_FitFailure(:optimizer_setup, 0, cause))
        end
    end

    initial_loss = _fit_evaluate(objective, model, data, ps, st, 0)
    history = Float64[initial_loss]
    gradient_norm_history = Float64[]
    relative_step_history = Float64[]
    relative_objective_change_history = Float64[]
    window_endpoint_change_history = Float64[]
    window_projected_drift_history = Float64[]
    window_maximum_step_history = Float64[]

    best_loss = Float64(initial_loss)
    best_epoch = 0
    best_ps = deepcopy(ps)
    best_optimiser_state = deepcopy(optimiser_state)
    converged = false
    recovered = false
    termination_reason = :maximum_epochs
    failure = nothing

    for epoch in 1:epochs
        try
            parameter_gradient = try
                gradient(objective, model, data, ps, st)
            catch cause
                cause isa InterruptException && rethrow()
                throw(_FitFailure(:gradient, epoch, cause))
            end
            _fit_all_finite(parameter_gradient) || throw(_FitFailure(
                :gradient, epoch, ErrorException("gradient contained non-finite values")))
            gradient_norm = _fit_norm(parameter_gradient)

            candidate_optimiser_state, candidate_ps = try
                Optimisers.update(optimiser_state, ps, parameter_gradient)
            catch cause
                cause isa InterruptException && rethrow()
                throw(_FitFailure(:optimizer_update, epoch, cause))
            end
            _fit_all_finite(candidate_ps) || throw(_FitFailure(
                :optimizer_update, epoch,
                ErrorException("updated parameters contained non-finite values")))

            candidate_loss = _fit_evaluate(
                objective, model, data, candidate_ps, st, epoch)
            relative_step = _fit_relative_step(ps, candidate_ps)
            relative_objective_change = _fit_relative_objective_change(
                history[end], candidate_loss)

            push!(history, Float64(candidate_loss))
            push!(gradient_norm_history, gradient_norm)
            push!(relative_step_history, relative_step)
            push!(relative_objective_change_history, relative_objective_change)
            callback === nothing || callback(epoch, candidate_loss)

            ps = candidate_ps
            optimiser_state = candidate_optimiser_state
            if candidate_loss < best_loss
                best_loss = Float64(candidate_loss)
                best_epoch = epoch
                best_ps = deepcopy(candidate_ps)
                best_optimiser_state = deepcopy(candidate_optimiser_state)
            end

            window = _fixed_window_diagnostics(
                history, relative_step_history, gradient_norm_history;
                patience, objective_rel_tol, step_rel_tol, gradient_abs_tol)
            push!(window_endpoint_change_history, window.endpoint_relative_change)
            push!(window_projected_drift_history, window.projected_relative_drift)
            push!(window_maximum_step_history, window.maximum_parameter_step)
            if epoch >= min_epochs && window.stable
                converged = true
                termination_reason = :converged
                break
            end
        catch cause
            cause isa InterruptException && rethrow()
            error = cause isa _FitFailure ? cause : _FitFailure(:unknown, epoch, cause)
            if on_failure === :throw
                throw(error)
            end
            failure = error
            recovered = true
            termination_reason = :recovered_after_failure
            break
        end
    end

    final_ps = ps
    final_optimiser_state = optimiser_state
    final_loss = history[end]
    final_epoch = length(history) - 1

    use_best = recovered || selection === :best
    selected_ps = use_best ? best_ps : final_ps
    selected_optimiser_state = use_best ? best_optimiser_state : final_optimiser_state
    selected_loss = use_best ? best_loss : final_loss
    selected_epoch = use_best ? best_epoch : final_epoch

    diagnostics = (
        history,
        gradient_norm_history,
        relative_step_history,
        relative_objective_change_history,
        window_endpoint_change_history,
        window_projected_drift_history,
        window_maximum_step_history,
        final_loss,
        final_epoch,
        best_loss,
        best_epoch,
        selected_loss,
        selected_epoch,
        epochs_completed = final_epoch,
    )
    return FitResult(
        model, objective, data, selected_ps, st;
        optimiser_state=selected_optimiser_state,
        optimiser,
        objective_value=selected_loss,
        history,
        iterations=final_epoch,
        selected_iteration=selected_epoch,
        converged,
        termination_reason,
        recovered,
        failure,
        selection,
        diagnostics,
        metadata,
    )
end
