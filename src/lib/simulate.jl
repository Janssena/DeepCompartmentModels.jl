# Gate 8B — forward simulation (deterministic + residual + between-subject).
#
# Compose a *design* (covariates + dosing regimen + sampling times) with a model
# and parameters, and get back a `Population` — the same structure `fit`/`load`
# consume, so a simulated dataset round-trips into another fit, a diagnostic, or
# a plot. Covers the Gate-5-independent sources of variability: the
# fitted/specified typical parameters, an explicitly specified between-subject
# random-effect distribution (`omega`/`random_effects`), and the residual-error
# distribution. Single-output and multi-output (asynchronous per-DV sampling)
# designs are both supported.
#
# Deliberately deferred to later 8B slices (recorded in REBUILD.md):
#   * parameter-uncertainty propagation (simulate over draws from the qualified
#     Gate 7A covariance or bootstrap refits);
#   * reading a VEM-fitted Ω directly (inherits Gate 5's open qualification);
#   * a multi-output design builder and packaged VPC summary helpers.

"""
    dose_regimen(; amount, n_doses=1, interval=24.0, start=0.0, infusion_duration=nothing)

Build a dosing matrix for [`generate_dosing_callback`](@ref). Returns a 2-column
`[time amount]` matrix for **bolus** doses, or a 4-column
`[time amount rate duration]` matrix for a **zero-order infusion** when
`infusion_duration` is supplied. `n_doses > 1` repeats the dose every `interval`
starting at `start`.

# Examples
```julia
dose_regimen(amount=100, n_doses=4, interval=24)      # 100 mg q24h × 4, bolus
dose_regimen(amount=500, infusion_duration=1.0)       # 500 mg over 1 h, infusion
```
"""
function dose_regimen(; amount::Real, n_doses::Integer=1, interval::Real=24.0,
                      start::Real=0.0, infusion_duration::Union{Nothing,Real}=nothing)
    n_doses >= 1 || throw(ArgumentError("n_doses must be at least 1."))
    amount > 0 || throw(ArgumentError("amount must be positive."))
    (n_doses == 1 || interval > 0) || throw(ArgumentError(
        "interval must be positive for repeated dosing."))
    times = float(start) .+ float(interval) .* (0:(n_doses - 1))
    amounts = fill(float(amount), n_doses)
    infusion_duration === nothing && return hcat(collect(times), amounts)
    infusion_duration > 0 || throw(ArgumentError("infusion_duration must be positive."))
    rate = amount / infusion_duration
    return hcat(collect(times), amounts, fill(float(rate), n_doses),
                fill(float(infusion_duration), n_doses))
end

"""
    simulation_population(covariates; regimen, obs_times, T=Float64) -> Population

Assemble a single-output design `Population` for simulation: one subject per
covariate vector, dosed by `regimen`, sampled at `obs_times`. Observations are
placeholder zeros — fill them with [`simulate`](@ref).

- `regimen`: a shared dosing matrix (from [`dose_regimen`](@ref)) **or** a
  function `covariates -> matrix` for covariate-dependent dosing.
- `obs_times`: the sampling schedule.
  - single-output: a vector/range shared by every subject, or a vector of
    per-subject schedules (one time vector per covariate row);
  - multi-output: a **`Tuple`** of per-dependent-variable time vectors shared by
    every subject (e.g. `([0.5,1,2,4], [1,3,4])`), or a vector of such tuples for
    per-subject multi-output schedules. A `Tuple` groups dependent variables;
    a `Vector` groups subjects, so the two are never ambiguous.

For full control, build the `Population` yourself and pass it to `simulate`,
which uses each subject's own design.
"""
function simulation_population(covariates; regimen, obs_times, T::Type=Float64)
    isempty(covariates) && throw(ArgumentError("covariates must be non-empty."))
    multi_output = obs_times isa Tuple ||
        (obs_times isa AbstractVector && !isempty(obs_times) && first(obs_times) isa Tuple)
    multi_output && return _mo_simulation_population(covariates, regimen, obs_times, T)

    get_regimen = regimen isa Function ? regimen : (_ -> regimen)
    n = length(covariates)
    per_subject = obs_times isa AbstractVector &&
        eltype(obs_times) <: Union{AbstractVector,AbstractRange,Tuple}
    per_subject && length(obs_times) != n && throw(ArgumentError(
        "per-subject obs_times has $(length(obs_times)) schedules but there " *
        "are $n subjects."))
    individuals = map(enumerate(covariates)) do (i, x)
        covariate = T.(collect(x))
        callback = generate_dosing_callback(T.(get_regimen(covariate)), T)
        times = T.(collect(per_subject ? obs_times[i] : obs_times))
        Individual(i, covariate, times, zeros(T, length(times)), callback, T)
    end
    return Population(individuals)
end

# Multi-output design: `obs_times` is a Tuple of per-DV time vectors (shared) or
# a Vector of such Tuples (per subject).
function _mo_simulation_population(covariates, regimen, obs_times, T)
    get_regimen = regimen isa Function ? regimen : (_ -> regimen)
    n = length(covariates)
    per_subject = obs_times isa AbstractVector
    per_subject && length(obs_times) != n && throw(ArgumentError(
        "per-subject obs_times has $(length(obs_times)) schedules but there " *
        "are $n subjects."))
    individuals = map(enumerate(covariates)) do (i, x)
        covariate = T.(collect(x))
        callback = generate_dosing_callback(T.(get_regimen(covariate)), T)
        schedule = per_subject ? obs_times[i] : obs_times
        times = [T.(collect(dv_times)) for dv_times in schedule]
        observations = [zeros(T, length(t)) for t in times]
        MOIndividual(i, covariate, times, observations, callback, T)
    end
    return Population(individuals)
end

# Reconstruct an individual with simulated observations, preserving its design.
_set_observations(individual::BasicIndividual{T}, y) where {T} =
    BasicIndividual(individual.id, individual.x, individual.t, T.(y),
                    individual.callback, T; u0=individual.u0,
                    occasions=individual.occasions)

_set_observations(individual::MOIndividual{T}, ys) where {T} =
    MOIndividual(individual.id, individual.x,
                 [individual.t[mask] for mask in individual.dvid],
                 [T.(y) for y in ys], individual.callback, T;
                 u0=individual.u0, occasions=individual.occasions)

_set_observations(individual::AbstractIndividual, _) = throw(ArgumentError(
    "simulate currently supports BasicIndividual and MOIndividual designs; " *
    "got $(nameof(typeof(individual)))."))

# Draw one subject's observations. Single-output predictions are a real vector;
# multi-output predictions are one vector per dependent variable.
_simulate_draw(error::AbstractErrorModel, prediction::AbstractVector{<:Real}, ps_error, rng) =
    rand(rng, make_dist(error, prediction, ps_error))

_simulate_draw(error::ErrorModelSet, predictions::AbstractVector{<:AbstractVector}, ps_error, rng) =
    map(d -> rand(rng, d), make_dist(error, predictions, ps_error))

function _simulate_observations(model, prediction, ps, rng, residual)
    draw = residual && !(model.error isa ImplicitError) && haskey(ps, :error)
    return draw ? _simulate_draw(model.error, prediction, ps.error, rng) : prediction
end

# Between-subject variability. Random effects follow the package's mixed-effect
# convention: multiplicative log-normal on the typical parameters, `z = ζ ⊙
# exp(mask·η)` with `η ~ N(0, Ω)`. `Ω` and the random-effect indices are supplied
# explicitly (a specified-Ω forward simulation, independent of Gate 5); a
# VEM-fitted Ω is not read here.
function _build_bsv(model, ps, st, design, omega, random_effects)
    (omega === nothing) == (random_effects === nothing) || throw(ArgumentError(
        "between-subject variability needs BOTH `omega` and `random_effects` " *
        "(the parameter indices carrying random effects), or neither."))
    omega === nothing && return nothing

    indices = collect(Int, random_effects)
    all(>(0), indices) || throw(ArgumentError("random_effects indices must be positive."))
    allunique(indices) || throw(ArgumentError("random_effects indices must be unique."))
    n_params = _estimate_typ_parameter_size(model, design[1:1], ps, st)
    maximum(indices) <= n_params || throw(ArgumentError(
        "random_effects index $(maximum(indices)) exceeds the $n_params typical " *
        "parameters."))

    covariance = omega isa AbstractVector ? Diagonal(collect(float.(omega))) :
                 Symmetric(Matrix(float.(omega)))
    size(covariance, 1) == length(indices) || throw(DimensionMismatch(
        "omega is $(size(covariance)) but there are $(length(indices)) random effects."))
    isposdef(covariance) || throw(ArgumentError(
        "omega must be positive definite to define a random-effect distribution."))

    mask = indicator(n_params, indices, Float64)
    return (mask=mask, dist=MvNormal(zeros(length(indices)), covariance))
end

# Per-subject typical parameters after applying between-subject variability.
_simulated_parameters(ζ, ::Nothing, rng) = ζ
_simulated_parameters(ζ, bsv, rng) = reduce(hcat, map(axes(ζ, 2)) do i
    ζ[:, i] .* exp.(bsv.mask * rand(rng, bsv.dist))
end)

function _simulate_once(model, ps, st, design::Population, rng, residual, solve_kwargs, bsv)
    ζ, _ = predict_typ_parameters(model, design, ps, st)
    z = _simulated_parameters(ζ, bsv, rng)
    individuals = map(enumerate(design)) do (i, individual)
        prediction = solve_for_target(model, individual, z[:, i]; solve_kwargs...)
        _set_observations(individual,
            _simulate_observations(model, prediction, ps, rng, residual))
    end
    return Population(individuals)
end

"""
    simulate(model, ps, st, design::Population; rng, seed, residual=true, n=1, solve_kwargs=NamedTuple())
    simulate(result::FitResult[, design]; kwargs...)

Simulate observations for the `design` `Population`, keeping its covariates,
dosing callbacks and sampling times and generating only the observations. Returns
a new `Population` (`n == 1`) or a `Vector{Population}` (`n > 1`), each ready to
re-fit, diagnose, or plot.

Sources of variability:
- the fitted/specified typical parameters (always);
- `residual=true`: draw from the model's error distribution
  (`make_dist(model.error, ŷ, ps.error)`). Skipped without error (an `MSE`/`SSE`
  fit uses `ImplicitError`), in which case the typical prediction is returned;
- between-subject variability: pass **both** `omega` (a covariance matrix, or a
  vector of diagonal variances) and `random_effects` (the typical-parameter
  indices carrying random effects). Random effects are multiplicative log-normal,
  `z = ζ ⊙ exp(η)` with `η ~ N(0, Ω)`, matching the package's mixed-effect model.

`omega`/`random_effects` describe an **explicitly specified** Ω — a forward
"what if" simulation that does not depend on Gate 5. A VEM-*fitted* Ω is not read
automatically; passing a mixed-effects parameter set (with `omega`/`phi` in `ps`)
is refused. Parameter-uncertainty propagation remains a later slice.

# Keywords
- `rng=Random.default_rng()`, `seed=nothing`: the same `rng` (or `seed`)
  reproduces the same dataset.
- `n=1`: number of replicate datasets, each with fresh random-effect and residual draws.
- `omega=nothing`, `random_effects=nothing`: between-subject variability (see above).
- `solve_kwargs=NamedTuple()`: passed to the ODE solve, e.g.
  `(abstol=1e-12, reltol=1e-10)`.
"""
function simulate(model::AbstractDEModel, ps::NamedTuple, st, design::Population;
                  rng::Random.AbstractRNG=Random.default_rng(), seed=nothing,
                  residual::Bool=true, n::Integer=1,
                  omega=nothing, random_effects=nothing,
                  solve_kwargs::NamedTuple=NamedTuple())
    (:omega in keys(ps) || :phi in keys(ps)) && throw(ArgumentError(
        "simulate does not read a VEM-fitted Ω; pass fixed-effect parameters " *
        "(theta[, error]) and specify between-subject variability explicitly with " *
        "the `omega` and `random_effects` keywords."))
    n >= 1 || throw(ArgumentError("n must be at least 1."))
    isempty(design) && throw(ArgumentError("design population must be non-empty."))
    bsv = _build_bsv(model, ps, st, design, omega, random_effects)
    seed === nothing || Random.seed!(rng, seed)
    n == 1 && return _simulate_once(model, ps, st, design, rng, residual, solve_kwargs, bsv)
    return [_simulate_once(model, ps, st, design, rng, residual, solve_kwargs, bsv)
            for _ in 1:n]
end

"""
    simulate(result::FitResult[, design]; omega=nothing, random_effects=nothing, kwargs...)

Simulate from a fit. For a fixed-effect result this forwards to the model method.
For a **mixed-effect** result it simulates with the fitted (typical θ, residual σ)
and draws between-subject variability from the **fitted Ω** and the objective's
random-effect indices unless `omega`/`random_effects` override them. The fitted Ω
comes from a Variational-EM fit and is **provisional / not qualified** (Gate 5);
a one-time warning is emitted. This enables a VPC from a mixed-effect fit while
keeping the assumption explicit.
"""
function simulate(result::FitResult, design::Population=result.data;
                  omega=nothing, random_effects=nothing, kwargs...)
    if result.metadata.effects === :fixed
        return simulate(result.model, result.ps, result.st, design;
                        omega, random_effects, kwargs...)
    end
    om = omega === nothing ? Matrix(result.ps.omega) : omega
    indices = random_effects === nothing ? collect(Int, result.objective.idxs) : random_effects
    omega === nothing && @warn(
        "Simulating between-subject variability from a VEM-fitted Ω, which is " *
        "provisional and not qualified (Gate 5). Pass `omega` explicitly to " *
        "override.", maxlog=1)
    fixed_ps = (theta = result.ps.theta, error = result.ps.error)
    return simulate(result.model, fixed_ps, result.st, design;
                    omega=om, random_effects=indices, kwargs...)
end

"""
    parameter_draws(u::FixedEffectUncertainty, n; rng=Random.default_rng(), parameter_type=Float64)

Draw `n` fixed-effect parameter sets from the qualified Gate 7A observed-
information covariance for use with the `simulate(model, ps_draws, ...)` route,
propagating parameter uncertainty into a simulation. Each draw samples the
identified parameters on the unconstrained (optimizer) scale from
`MvNormal(u.unconstrained, u.vcov_unconstrained)` and returns a fixed-effect
parameter tree `(theta=(unconstrained=…,), error=(σ=…,))`. Sampling on the
unconstrained scale keeps every draw inside each parameter's declared domain.
"""
function parameter_draws(u::FixedEffectUncertainty, n::Integer;
                         rng::Random.AbstractRNG=Random.default_rng(),
                         parameter_type::Type{<:AbstractFloat}=Float64)
    n >= 1 || throw(ArgumentError("n must be at least 1."))
    distribution = MvNormal(u.unconstrained, Symmetric(u.vcov_unconstrained))
    structural = u.kinds .== :structural
    residual = u.kinds .== :residual
    return map(1:n) do _
        draw = rand(rng, distribution)
        (theta=(unconstrained=parameter_type.(draw[structural]),),
         error=(σ=parameter_type.(draw[residual]),))
    end
end

"""
    simulate(model, ps_draws::AbstractVector{<:NamedTuple}, st, design; kwargs...)

Simulate one dataset per parameter set in `ps_draws` (for example from
[`parameter_draws`](@ref) or bootstrap refits), returning a `Vector{Population}`.
This propagates parameter uncertainty into the simulation. `residual`, `omega`
and `random_effects` behave as in the single-parameter method and apply to every
draw.
"""
function simulate(model::AbstractDEModel, ps_draws::AbstractVector{<:NamedTuple}, st,
                  design::Population; rng::Random.AbstractRNG=Random.default_rng(),
                  seed=nothing, residual::Bool=true,
                  omega=nothing, random_effects=nothing,
                  solve_kwargs::NamedTuple=NamedTuple())
    isempty(ps_draws) && throw(ArgumentError("ps_draws must be non-empty."))
    isempty(design) && throw(ArgumentError("design population must be non-empty."))
    seed === nothing || Random.seed!(rng, seed)
    return map(ps_draws) do ps
        (:omega in keys(ps) || :phi in keys(ps)) && throw(ArgumentError(
            "ps_draws must contain fixed-effect parameter trees (theta[, error])."))
        bsv = _build_bsv(model, ps, st, design, omega, random_effects)
        _simulate_once(model, ps, st, design, rng, residual, solve_kwargs, bsv)
    end
end

"""
    VPCSummary

Binned visual-predictive-check summary (see [`vpc`](@ref)). Holds the bin edges
and centres, the requested percentile `levels`, the `simulated` percentile matrix
(`length(bin_centers) × length(levels)`), the matching `observed` matrix (or
`nothing`), and the per-bin simulated/observed observation counts.
"""
struct VPCSummary
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    levels::Vector{Float64}
    simulated::Matrix{Float64}
    observed::Union{Nothing,Matrix{Float64}}
    n_simulated::Vector{Int}
    n_observed::Vector{Int}
end

function Base.show(io::IO, summary::VPCSummary)
    print(io, "VPCSummary($(length(summary.bin_centers)) bins; ",
          "levels=$(summary.levels); ",
          summary.observed === nothing ? "simulated only)" : "with observed)")
end

_vpc_pairs(population::Population{<:BasicIndividual}) =
    (reduce(vcat, [Float64.(get_t(i)) for i in population]),
     reduce(vcat, [Float64.(get_y(i)) for i in population]))
_vpc_pairs(::Population) = throw(ArgumentError(
    "vpc currently supports single-output (BasicIndividual) populations."))

function _vpc_quantiles(times, values, edges, levels)
    n_bins = length(edges) - 1
    quantiles = fill(NaN, n_bins, length(levels))
    counts = zeros(Int, n_bins)
    for b in 1:n_bins
        upper = b == n_bins ? (v -> v <= edges[b + 1]) : (v -> v < edges[b + 1])
        inside = findall(v -> v >= edges[b] && upper(v), times)
        counts[b] = length(inside)
        isempty(inside) && continue
        binned = values[inside]
        for (l, level) in enumerate(levels)
            quantiles[b, l] = quantile(binned, level)
        end
    end
    return quantiles, counts
end

"""
    vpc(simulations::AbstractVector{<:Population}; observed=nothing, bins=8, levels=(0.05,0.5,0.95))

Binned visual-predictive-check summary for single-output data. Pools the
observations of all simulated replicate populations, bins them by time, and
reports the `levels` percentiles per bin; if `observed` is supplied its
percentiles are reported on the same bins. `bins` is either a bin count (equal-
count edges from the pooled simulated times) or an explicit vector of edges.
Returns a [`VPCSummary`](@ref); plotting is left to the caller.
"""
function vpc(simulations::AbstractVector{<:Population};
             observed::Union{Nothing,Population}=nothing,
             bins::Union{Integer,AbstractVector{<:Real}}=8,
             levels=(0.05, 0.5, 0.95))
    isempty(simulations) && throw(ArgumentError("simulations must be non-empty."))
    level_vector = collect(Float64, levels)
    times = reduce(vcat, [first(_vpc_pairs(pop)) for pop in simulations])
    values = reduce(vcat, [last(_vpc_pairs(pop)) for pop in simulations])

    if bins isa Integer
        bins >= 1 || throw(ArgumentError("bins must be at least 1."))
        edges = collect(quantile(times, range(0, 1; length=bins + 1)))
        edges[1] = minimum(times); edges[end] = maximum(times)
        unique!(edges)
    else
        edges = collect(Float64, bins)
        issorted(edges) && length(edges) >= 2 || throw(ArgumentError(
            "explicit bin edges must be sorted with at least two entries."))
    end
    centers = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2

    simulated, n_simulated = _vpc_quantiles(times, values, edges, level_vector)
    observed_quantiles = nothing
    n_observed = zeros(Int, length(centers))
    if observed !== nothing
        obs_times, obs_values = _vpc_pairs(observed)
        observed_quantiles, n_observed = _vpc_quantiles(obs_times, obs_values, edges, level_vector)
    end
    return VPCSummary(edges, centers, level_vector, simulated, observed_quantiles,
                      n_simulated, n_observed)
end
