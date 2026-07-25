# Gate 7B — standardized model-evaluation records.
#
# `prediction_record` builds one honest, error-model-aware row per observation:
# the structural prediction, the predictive distribution's mean and standard
# deviation (from `make_dist`), the raw residual, and the population weighted
# residual. It is error-model-aware (Gaussian and distribution-backed custom
# likelihoods alike) and aligns multi-output / asynchronous dependent-variable
# observations by their own time vectors — it never silently reuses one DV's
# times for another.
#
# This is a core data layer with no plotting dependency; goodness-of-fit,
# residual and dense-profile figures live in the optional analysis layer and are
# built from these columns.

"""
    PredictionRecord

Column-oriented record produced by [`prediction_record`](@ref). One row per
observation. Fields are equal-length vectors:

- `id` — subject identifier;
- `dv` — dependent-variable index (`1` for single-output data);
- `time` — observation time;
- `observation` — observed value `y`;
- `prediction` — structural prediction `ŷ` (population PRED);
- `predictive_mean` — mean of the predictive distribution `make_dist(error, ŷ, ps.error)`;
- `predictive_std` — standard deviation of that predictive distribution;
- `residual` — `observation - prediction`;
- `weighted_residual` — `(observation - predictive_mean) / predictive_std`.

For Gaussian error models `predictive_mean == prediction`. When the model uses
`ImplicitError` (an `MSE`/`SSE` fit) there is no predictive distribution, so
`predictive_std` and `weighted_residual` are `NaN` while `prediction` and
`residual` remain defined.
"""
struct PredictionRecord
    id::Vector{Any}
    dv::Vector{Int}
    time::Vector{Float64}
    observation::Vector{Float64}
    prediction::Vector{Float64}
    predictive_mean::Vector{Float64}
    predictive_std::Vector{Float64}
    residual::Vector{Float64}
    weighted_residual::Vector{Float64}
end

Base.length(record::PredictionRecord) = length(record.time)

function Base.show(io::IO, record::PredictionRecord)
    n = length(record)
    subjects = length(unique(record.id))
    outputs = length(unique(record.dv))
    print(io, "PredictionRecord($n observations; $subjects subjects; ",
          "$outputs output(s))")
end

_dv_times(individual::AbstractIndividual, ::Int) = get_t(individual)
_dv_times(individual::MOIndividual, j::Int) = individual.t[individual.dvid[j]]

# Per-dependent-variable predictive mean and standard deviation from the error
# model. `ImplicitError` has no distribution; the mean defaults to the structural
# prediction and the standard deviation is undefined.
function _predictive_moments(error::ImplicitError, prediction, ps_error)
    return prediction, fill(NaN, length(prediction))
end

function _predictive_moments(error::AbstractErrorModel, prediction, ps_error)
    distribution = make_dist(error, prediction, ps_error)
    return Float64.(mean(distribution)), sqrt.(Float64.(var(distribution)))
end

function _push_dv!(columns, id, dv, times, observations, prediction,
                   predictive_mean, predictive_std)
    for k in eachindex(observations)
        push!(columns.id, id)
        push!(columns.dv, dv)
        push!(columns.time, times[k])
        push!(columns.observation, observations[k])
        push!(columns.prediction, prediction[k])
        push!(columns.predictive_mean, predictive_mean[k])
        push!(columns.predictive_std, predictive_std[k])
        push!(columns.residual, observations[k] - prediction[k])
        push!(columns.weighted_residual,
              (observations[k] - predictive_mean[k]) / predictive_std[k])
    end
    return nothing
end

"""
    prediction_record(model, data::Population, ps, st) -> PredictionRecord
    prediction_record(result::FitResult) -> PredictionRecord

Standardized observation/prediction/residual record (see [`PredictionRecord`]).

For a mixed-effect fit the `FitResult` method uses the **deterministic
empirical-Bayes** random effects (η = the variational posterior means), so
`prediction` is the individual prediction (IPRED) and `weighted_residual` is an
IWRES-like quantity standardised by the residual-error model. Mixed-effect
*uncertainty* (standard errors, coverage) remains provisional pending Gate 5;
these are point diagnostics, not uncertainty claims.
"""
function prediction_record(model::DeepCompartmentModel, data::Population, ps, st)
    predictions = predict(model, data, ps, st)
    columns = (
        id=Any[], dv=Int[], time=Float64[], observation=Float64[],
        prediction=Float64[], predictive_mean=Float64[], predictive_std=Float64[],
        residual=Float64[], weighted_residual=Float64[],
    )
    for (individual, prediction) in zip(data, predictions)
        _record_individual!(columns, model.error, individual, prediction, ps)
    end
    return PredictionRecord(columns.id, columns.dv, columns.time,
        columns.observation, columns.prediction, columns.predictive_mean,
        columns.predictive_std, columns.residual, columns.weighted_residual)
end

_ps_error(ps) = haskey(ps, :error) ? ps.error : NamedTuple()

# Single-output subject: one predictive distribution over the observation vector.
function _record_individual!(columns, error, individual::AbstractIndividual, prediction, ps)
    observations = get_y(individual)
    predictive_mean, predictive_std = _predictive_moments(
        error, prediction, _ps_error(ps))
    _push_dv!(columns, individual.id, 1, get_t(individual), observations,
              prediction, predictive_mean, predictive_std)
    return nothing
end

# Multi-output subject: one predictive distribution per dependent variable, each
# aligned to that dependent variable's own observation times.
function _record_individual!(columns, error, individual::MOIndividual, predictions, ps)
    observations = get_y(individual)
    moments = _mo_predictive_moments(error, predictions, _ps_error(ps))
    for j in eachindex(observations)
        _push_dv!(columns, individual.id, j, _dv_times(individual, j),
                  observations[j], predictions[j], moments[j][1], moments[j][2])
    end
    return nothing
end

function _mo_predictive_moments(error::ErrorModelSet, predictions, ps_error)
    distributions = make_dist(error, predictions, ps_error)
    return map(distributions) do d
        (Float64.(mean(d)), sqrt.(Float64.(var(d))))
    end
end

function _mo_predictive_moments(::ImplicitError, predictions, ps_error)
    return map(prediction -> (Float64.(prediction), fill(NaN, length(prediction))),
               predictions)
end

function prediction_record(result::FitResult)
    result.model isa DeepCompartmentModel || throw(ArgumentError(
        "prediction_record is only implemented for a DeepCompartmentModel."))
    # Mixed-effect fits use deterministic empirical-Bayes random effects (IPRED).
    st = result.metadata.effects === :mixed ?
        _empirical_bayes_state(result.st) : result.st
    return prediction_record(result.model, result.data, result.ps, st)
end
