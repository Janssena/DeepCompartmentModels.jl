"""
    FitResult

Canonical result returned by [`fit`](@ref). It retains the fitted model,
objective, data, parameters, state, optimizer information, convergence status,
objective history, and compact provenance metadata.

Algorithm-specific histories remain available as properties (for example,
`result.gradient_norm_history` for fixed effects and
`result.monitor_mcse_history` for Variational-EM). They are stored in
`result.diagnostics` so the common result contract stays small.

`FitResult` intentionally reports the fitted parameter tree verbatim in `ps`.
It does not label arbitrary neural-network weights as pharmacokinetic
coefficients or silently transform them to a purported natural scale.
"""
struct FitResult{M,O,D,P,S,OS,R,DI,ME}
    model::M
    objective::O
    data::D
    ps::P
    st::S
    optimiser_state::OS
    optimiser::R
    objective_value::Float64
    history::Vector{Float64}
    iterations::Int
    selected_iteration::Int
    converged::Bool
    termination_reason::Symbol
    recovered::Bool
    failure::Any
    selection::Symbol
    diagnostics::DI
    metadata::ME
end

function FitResult(
    model,
    objective,
    data,
    ps,
    st;
    optimiser_state=nothing,
    optimiser=nothing,
    objective_value=nothing,
    history::AbstractVector{<:Real}=Float64[],
    iterations::Integer=max(length(history) - 1, 0),
    selected_iteration::Integer=iterations,
    converged::Bool=false,
    termination_reason::Symbol=:not_reported,
    recovered::Bool=false,
    failure=nothing,
    selection::Symbol=:final,
    diagnostics::NamedTuple=NamedTuple(),
    metadata::NamedTuple=NamedTuple(),
)
    iterations >= 0 || throw(ArgumentError("iterations must be non-negative."))
    0 <= selected_iteration <= iterations || throw(ArgumentError(
        "selected_iteration must be between zero and iterations."))
    selection in (:final, :best) || throw(ArgumentError(
        "selection must be :final or :best, got $selection"))
    recovered && converged && throw(ArgumentError(
        "a recovered numerical failure cannot be marked converged."))

    result_history = Float64.(collect(history))
    value = objective_value === nothing ?
        (isempty(result_history) ? NaN : last(result_history)) :
        Float64(objective_value)
    result_metadata = merge(_fit_result_metadata(model, objective, data, ps), metadata)
    return FitResult(
        model, objective, data, ps, st, optimiser_state, optimiser, value,
        result_history, Int(iterations), Int(selected_iteration), converged,
        termination_reason, recovered, failure, selection, diagnostics,
        result_metadata)
end

_fit_result_scalar_count(::Nothing) = 0
_fit_result_scalar_count(::Number) = 1
_fit_result_scalar_count(x::AbstractArray{<:Number}) = length(x)
_fit_result_scalar_count(x::AbstractArray) =
    sum(_fit_result_scalar_count, x; init=0)
_fit_result_scalar_count(x::NamedTuple) =
    sum(_fit_result_scalar_count, values(x); init=0)
_fit_result_scalar_count(x::Tuple) =
    sum(_fit_result_scalar_count, x; init=0)
_fit_result_scalar_count(_) = 0

_fit_result_observation_count(y::Number) = 1
_fit_result_observation_count(y::AbstractArray{<:Number}) = length(y)
_fit_result_observation_count(y::AbstractArray) =
    sum(_fit_result_observation_count, y; init=0)
_fit_result_observation_count(y::Tuple) =
    sum(_fit_result_observation_count, y; init=0)

_fit_result_output_count(individual::MOIndividual) = length(get_y(individual))
_fit_result_output_count(::AbstractIndividual) = 1
_fit_result_output_count(population::Population) =
    isempty(population) ? 0 : maximum(_fit_result_output_count, population)

function _fit_result_metadata(model, objective, data, ps)
    n_subjects = data isa Population ? length(data) : 1
    metadata = (
        package_version=pkgversion(@__MODULE__),
        model_type=nameof(typeof(model)),
        objective_type=nameof(typeof(objective)),
        effects=objective isa MixedObjective ? :mixed : :fixed,
        n_subjects=n_subjects,
        n_observations=_fit_result_observation_count(get_y(data)),
        n_outputs=_fit_result_output_count(data),
        parameter_blocks=Tuple(keys(ps)),
        n_fitted_scalars=_fit_result_scalar_count(ps),
    )
    if model isa DeepCompartmentModel && model.model isa StructuralParameters
        layer = model.model
        metadata = merge(metadata, (
            structural_parameter_names=Tuple(parameter_names(layer)),
            structural_parameter_units=Tuple(parameter_units(layer)),
            structural_parameter_transforms=
                Tuple(nameof.(typeof.(parameter_transforms(layer)))),
            structural_parameter_bounds=Tuple(parameter_bounds(layer)),
        ))
    end
    return metadata
end

"""Return `true` only when the fit's declared convergence criterion passed."""
isconverged(result::FitResult) = result.converged

"""Return the number of completed optimizer or outer Variational-EM updates."""
niterations(result::FitResult) = result.iterations

"""Return the objective history used to report and assess the fit."""
objective_history(result::FitResult) = result.history

"""Return the explicit termination reason recorded by the fitting algorithm."""
fit_status(result::FitResult) = result.termination_reason

function _fit_result_structural_layer(result::FitResult)
    model = result.model
    model isa DeepCompartmentModel || throw(ArgumentError(
        "structural coefficients are only defined for a DeepCompartmentModel."))
    layer = model.model
    layer isa StructuralParameters || throw(ArgumentError(
        "coef is only defined when the fitted model uses StructuralParameters; " *
        "raw neural-network weights are not structural coefficients."))
    return layer
end

"""Return fitted structural parameters on their declared natural scale."""
function coef(result::FitResult)
    layer = _fit_result_structural_layer(result)
    return natural_parameters(layer, result.ps.theta)
end

"""Return structural-parameter names in their declared ODE-output order."""
coefnames(result::FitResult) =
    parameter_names(_fit_result_structural_layer(result))

"""Return structural-parameter units without performing any unit conversion."""
coefunits(result::FitResult) =
    parameter_units(_fit_result_structural_layer(result))

"""
    predict(result::FitResult[, data]; ebe=false, kwargs...)

Forward prediction to the fitted model using the result's selected `ps` and
`st`. For mixed-effects results, pass `individual=false` for population
predictions (PRED). Pass `ebe=true` to make individual predictions (IPRED) at
the **deterministic empirical-Bayes** random effects (η = the variational
posterior means) instead of a random posterior draw — this is what individual
diagnostics should use. The ordinary model-level `predict` semantics otherwise
apply.
"""
predict(result::FitResult; ebe::Bool=false, kwargs...) =
    predict(result.model, result.data,
            result.ps, ebe ? _empirical_bayes_state(result.st) : result.st; kwargs...)
predict(result::FitResult, data; ebe::Bool=false, kwargs...) =
    predict(result.model, data,
            result.ps, ebe ? _empirical_bayes_state(result.st) : result.st; kwargs...)

"""
    empirical_bayes(result::FitResult)

Return the empirical-Bayes random-effect estimates for a mixed-effect fit: one
vector per subject holding the variational posterior means `η = μ` on the
random-effect subspace (in the order of the objective's random-effect indices).
"""
function empirical_bayes(result::FitResult)
    result.metadata.effects === :mixed || throw(ArgumentError(
        "empirical_bayes is only defined for mixed-effect fits."))
    (haskey(result.ps, :phi) && haskey(result.ps.phi, :μ)) || throw(ArgumentError(
        "the fitted parameters do not contain variational posterior means (ps.phi.μ)."))
    return result.ps.phi.μ
end

# Preserve property access used by the pre-FitResult named-tuple return values.
function Base.getproperty(result::FitResult, name::Symbol)
    name in fieldnames(typeof(result)) && return getfield(result, name)
    diagnostics = getfield(result, :diagnostics)
    haskey(diagnostics, name) && return getproperty(diagnostics, name)
    throw(ArgumentError("FitResult has no property $(repr(name))."))
end

function Base.propertynames(result::FitResult, private::Bool=false)
    core = fieldnames(typeof(result))
    diagnostic_names = keys(getfield(result, :diagnostics))
    return (core..., filter(name -> name ∉ core, diagnostic_names)...)
end

Base.keys(result::FitResult) = propertynames(result)
Base.haskey(result::FitResult, name::Symbol) = name in propertynames(result)

function Base.show(io::IO, result::FitResult)
    print(io, "FitResult(", result.metadata.objective_type,
          "; ", result.metadata.effects,
          "; iterations=", result.iterations,
          "; objective=", round(result.objective_value; sigdigits=6),
          "; status=", result.termination_reason,
          result.recovered ? "; recovered=true)" : ")")
end
