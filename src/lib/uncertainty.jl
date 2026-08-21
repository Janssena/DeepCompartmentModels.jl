struct SoftplusTransform <: AbstractParameterTransform end
_to_natural(::SoftplusTransform, value) = softplus(value)

# d(natural)/d(unconstrained) for the delta method.
_natural_derivative(::IdentityTransform, u) = one(u)
_natural_derivative(::LogTransform, u) = exp(u)
_natural_derivative(t::LogitTransform, u) =
    (t.upper - t.lower) * logistic(u) * (1 - logistic(u))
_natural_derivative(::SoftplusTransform, u) = logistic(u)

"""
    FixedEffectUncertainty

Observed-information uncertainty for the identified parameters of a fixed-effect
fit (see [`uncertainty`](@ref)). Fields hold the identified-parameter names,
kinds (`:structural` or `:residual`), units, natural-scale point estimates, the
observed information and covariance matrices, standard errors, relative standard
errors (percent), transform-respecting confidence-interval bounds, the parameter
correlation matrix, and numerical diagnostics.

Supports `coef`, `coefnames`, `vcov`, `stderror`, and `confint`.
"""
struct FixedEffectUncertainty
    names::Vector{Symbol}
    kinds::Vector{Symbol}
    units::Vector{Union{Nothing,String}}
    unconstrained::Vector{Float64}
    estimate::Vector{Float64}
    information::Matrix{Float64}          # observed information, unconstrained scale
    vcov_unconstrained::Matrix{Float64}
    vcov::Matrix{Float64}                 # natural scale (delta method)
    standard_error::Vector{Float64}       # natural scale
    relative_standard_error::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    level::Float64
    correlation::Matrix{Float64}
    method::Symbol
    diagnostics::NamedTuple
    transforms::Vector{AbstractParameterTransform}   # per-parameter natural map
end

# ---- identified-parameter layout ------------------------------------------

function _identified_layout(result::FitResult)
    (result.objective isa LogLikelihood || result.objective isa MixedObjective) ||
        throw(ArgumentError(
            "Hessian uncertainty requires a LogLikelihood or VariationalELBO " *
            "objective; MSE/SSE are not likelihoods and have no " *
            "observed-information covariance."))
    model = result.model
    model isa DeepCompartmentModel || throw(ArgumentError(
        "Hessian uncertainty is only implemented for a DeepCompartmentModel."))
    layer = model.model
    layer isa StructuralParameters || throw(ArgumentError(
        "Hessian uncertainty requires a StructuralParameters model; raw " *
        "neural-network weights are not identified coefficients."))
    model.error isa ErrorModelSet && throw(ArgumentError(
        "Multi-output (ErrorModelSet) residual uncertainty is not yet supported."))

    ps = result.ps
    (haskey(ps, :theta) && haskey(ps.theta, :unconstrained)) || throw(ArgumentError(
        "expected ps.theta.unconstrained for a structural fixed-effect fit."))
    (haskey(ps, :error) && haskey(ps.error, :σ)) || throw(ArgumentError(
        "expected ps.error.σ for a fixed-effect likelihood fit."))
    theta_u = ps.theta.unconstrained
    sigma_u = ps.error.σ
    (theta_u isa AbstractVector{<:Real} && sigma_u isa AbstractVector{<:Real}) ||
        throw(ArgumentError(
            "identified parameters must be real vectors; got a more complex " *
            "residual structure than Gate 7A supports."))

    names = Symbol[parameter_names(layer)...]
    kinds = fill(:structural, length(names))
    units = Union{Nothing,String}[parameter_units(layer)...]
    transforms = AbstractParameterTransform[parameter_transforms(layer)...]

    n_sigma = length(sigma_u)
    sigma_names = n_sigma == 1 ? [:σ] : [Symbol("σ", i) for i in 1:n_sigma]
    append!(names, sigma_names)
    append!(kinds, fill(:residual, n_sigma))
    append!(units, fill(nothing, n_sigma))
    append!(transforms, fill(SoftplusTransform(), n_sigma))

    k = length(theta_u)
    unconstrained = Float64[Float64.(theta_u)...; Float64.(sigma_u)...]
    return (; names, kinds, units, transforms, unconstrained, k)
end

# Rebuild the fixed-effect parameter tree from a flat unconstrained vector while
# preserving the element type (Float64 or a ForwardDiff.Dual).
_scatter_fixed(u, k) =
    (theta=(unconstrained=u[1:k],), error=(σ=u[(k+1):end],))

# ---- Hessian construction --------------------------------------------------

function _finite_difference_hessian(negll, u0, step)
    n = length(u0)
    grad(u) = first(Zygote.gradient(negll, u))
    H = Matrix{Float64}(undef, n, n)
    for j in 1:n
        h = step * max(abs(u0[j]), 1.0)
        up = copy(u0);
        up[j] += h
        um = copy(u0);
        um[j] -= h
        H[:, j] = (grad(up) .- grad(um)) ./ (2h)
    end
    return H
end

# Core observed-information computation shared by the fixed- and mixed-effect
# paths. `negll(u)` is the negative (conditional) log-likelihood as a function of
# the flat unconstrained identified-parameter vector; `transforms` map each entry
# to its natural scale. Returns the common numeric bundle.
function _information_bundle(negll, u0, transforms; level, fdstep)
    n = length(u0)
    gradient_norm = maximum(abs, first(Zygote.gradient(negll, u0)))
    hessian_fd = _finite_difference_hessian(negll, u0, fdstep)
    hessian_forward = try
        ForwardDiff.hessian(negll, u0)
    catch cause
        cause isa InterruptException && rethrow()
        nothing
    end
    if hessian_forward === nothing
        method = :finite_difference
        raw = hessian_fd
        alternate = _finite_difference_hessian(negll, u0, 2 * fdstep)
    else
        method = :forwarddiff
        raw = hessian_forward
        alternate = hessian_fd
    end
    information = Symmetric((raw .+ raw') ./ 2)
    scale = max(maximum(abs, information), 1.0)
    symmetry = maximum(abs, raw .- raw') / (2 * scale)
    cross_check = maximum(abs, Matrix(information) .- (alternate .+ alternate') ./ 2) / scale

    factors = eigen(information).values
    positive_definite = all(>(0), factors)
    positive_definite || throw(ArgumentError(
        "the observed information matrix is not positive definite " *
        "(eigenvalues $(factors)); the fit is not at a proper identified " *
        "minimum, so a covariance cannot be reported."))
    condition_number = maximum(factors) / minimum(factors)

    vcov_unconstrained = inv(information)
    se_unconstrained = sqrt.(diag(vcov_unconstrained))
    estimate = [_to_natural(transforms[i], u0[i]) for i in 1:n]
    jacobian = [_natural_derivative(transforms[i], u0[i]) for i in 1:n]
    J = Diagonal(jacobian)
    vcov_natural = J * vcov_unconstrained * J
    standard_error = sqrt.(diag(vcov_natural))
    relative_standard_error = 100 .* standard_error ./ abs.(estimate)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    lower = [_to_natural(transforms[i], u0[i] - z * se_unconstrained[i]) for i in 1:n]
    upper = [_to_natural(transforms[i], u0[i] + z * se_unconstrained[i]) for i in 1:n]
    correlation = _correlation_from_covariance(vcov_unconstrained)
    diagnostics = (; gradient_norm, hessian_symmetry=symmetry,
        hessian_cross_check=cross_check, condition_number, positive_definite,
        fdstep=Float64(fdstep))
    return (; estimate, information=Matrix(information), vcov_unconstrained,
        vcov=vcov_natural, standard_error, relative_standard_error, lower, upper,
        correlation, method, diagnostics)
end

"""
    uncertainty(result::FitResult; level=0.95, fdstep=1e-4)

Observed-information uncertainty for a fixed-effect `LogLikelihood` fit of a
`DeepCompartmentModel` with a `StructuralParameters` layer. Returns a
[`FixedEffectUncertainty`](@ref).

The observed information is the Hessian of the negative log-likelihood on the
unconstrained optimizer scale, computed with `ForwardDiff` and independently
cross-checked with a central finite-difference Hessian of the reverse-mode
gradient (step `fdstep`, relative to each parameter). If `ForwardDiff` cannot
differentiate the model the finite-difference Hessian is used instead and the
`method` field reports `:finite_difference`.

The covariance is the inverse observed information; it is mapped to the declared
natural scale with the delta method. Standard errors and relative standard
errors are reported on the natural scale. Confidence intervals are formed as
symmetric Wald intervals on the unconstrained scale and mapped through each
parameter's monotone transform, so natural-scale bounds always respect the
parameter's domain.

An error is raised if the observed information is not positive definite: that
signals the fit is not at a proper minimum or the parameters are not locally
identified, in which case a covariance would be meaningless.

`level` sets the confidence level. Uncertainty is reported only for identified
parameters (named structural parameters and residual-error parameters); raw
neural-network weights are refused.

For a **mixed-effect** (`VariationalELBO`) fit this returns a
[`MixedEffectUncertainty`](@ref) instead: population standard errors conditional
on the empirical-Bayes random effects, plus Ω and η-shrinkage. Those mixed-effect
standard errors are **provisional and not qualified** (Gate 5).
"""
function uncertainty(result::FitResult; level::Real=0.95, fdstep::Real=1e-4)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level."))
    fdstep > 0 || throw(ArgumentError("fdstep must be positive, got $fdstep."))
    result.metadata.effects === :mixed && return _mixed_uncertainty(result; level, fdstep)
    layout = _identified_layout(result)
    negll(u) = result.objective(
        result.model, result.data, _scatter_fixed(u, layout.k), result.st)
    b = _information_bundle(negll, layout.unconstrained, layout.transforms; level, fdstep)
    return FixedEffectUncertainty(
        layout.names, layout.kinds, layout.units, layout.unconstrained, b.estimate,
        b.information, b.vcov_unconstrained, b.vcov, b.standard_error,
        b.relative_standard_error, b.lower, b.upper, Float64(level),
        b.correlation, b.method, b.diagnostics, layout.transforms)
end

"""
    MixedEffectUncertainty

**Provisional** mixed-effect uncertainty (see [`uncertainty`](@ref)). The
population parameters (typical structural values and residual error) carry
standard errors from the observed information of the log-likelihood **conditional
on the empirical-Bayes random effects** — a FOCE-like conditional approximation
that ignores Ω-estimation and random-effect uncertainty and, being derived from a
variational fit, is **not qualified** (Gate 5). Also holds the estimated Ω, its
diagonal standard deviations, the random-effect indices, and the per-dimension
η-shrinkage.
"""
struct MixedEffectUncertainty
    names::Vector{Symbol}
    kinds::Vector{Symbol}
    units::Vector{Union{Nothing,String}}
    unconstrained::Vector{Float64}
    estimate::Vector{Float64}
    information::Matrix{Float64}
    vcov_unconstrained::Matrix{Float64}
    vcov::Matrix{Float64}
    standard_error::Vector{Float64}
    relative_standard_error::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    level::Float64
    correlation::Matrix{Float64}
    method::Symbol
    diagnostics::NamedTuple
    transforms::Vector{AbstractParameterTransform}
    omega::Matrix{Float64}
    omega_sd::Vector{Float64}
    random_effect_indices::Vector{Int}
    shrinkage::Vector{Float64}
end

# Rebuild a mixed-effect parameter tree from the flat unconstrained population
# vector, keeping omega/phi from the fitted parameters fixed.
_scatter_mixed(u, k, base_ps) = merge(base_ps,
    (theta=(unconstrained=u[1:k],), error=(σ=u[(k+1):end],)))

function _mixed_uncertainty(result::FitResult; level, fdstep)
    layout = _identified_layout(result)
    base_ps = result.ps
    st_ebe = _empirical_bayes_state(result.st)
    # Negative log-likelihood conditional on the empirical-Bayes random effects.
    negll(u) = -loglikelihood(
        result.model, result.data, _scatter_mixed(u, layout.k, base_ps), st_ebe)
    b = _information_bundle(negll, layout.unconstrained, layout.transforms; level, fdstep)

    omega = Matrix{Float64}(result.ps.omega)
    omega_sd = sqrt.(diag(omega))
    indices = collect(Int, result.objective.idxs)
    ebe = reduce(hcat, [Float64.(μ) for μ in result.ps.phi.μ])   # num_random × n
    ebe_sd = [std(view(ebe, j, :)) for j in axes(ebe, 1)]
    shrinkage = 1 .- ebe_sd ./ omega_sd

    return MixedEffectUncertainty(
        layout.names, layout.kinds, layout.units, layout.unconstrained, b.estimate,
        b.information, b.vcov_unconstrained, b.vcov, b.standard_error,
        b.relative_standard_error, b.lower, b.upper, Float64(level),
        b.correlation, b.method, b.diagnostics, layout.transforms,
        omega, omega_sd, indices, shrinkage)
end

function _correlation_from_covariance(covariance)
    d = sqrt.(diag(covariance))
    correlation = covariance ./ (d * d')
    correlation[diagind(correlation)] .= 1.0
    return correlation
end

# ---- StatsAPI-style accessors (shared by both uncertainty types) ----------

const _Uncertainty = Union{FixedEffectUncertainty,MixedEffectUncertainty}

coef(u::_Uncertainty) = u.estimate
coefnames(u::_Uncertainty) = u.names
vcov(u::_Uncertainty) = u.vcov
stderror(u::_Uncertainty) = u.standard_error

function confint(u::_Uncertainty; level::Real=u.level)
    level == u.level && return collect(zip(u.lower, u.upper))
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level."))
    se_unconstrained = sqrt.(diag(u.vcov_unconstrained))
    z = quantile(Normal(), 1 - (1 - level) / 2)
    return [(
        _to_natural(u.transforms[i], u.unconstrained[i] - z * se_unconstrained[i]),
        _to_natural(u.transforms[i], u.unconstrained[i] + z * se_unconstrained[i]),
    ) for i in eachindex(u.unconstrained)]
end

function _show_parameter_table(io, u)
    header = ("parameter", "kind", "estimate", "SE", "RSE%",
        "lower", "upper", "unit")
    widths = (12, 11, 12, 11, 8, 12, 12, 10)
    _row(io, header, widths)
    for i in eachindex(u.names)
        unit = u.units[i] === nothing ? "" : u.units[i]
        _row(io, (
                string(u.names[i]), string(u.kinds[i]),
                _fmt(u.estimate[i]), _fmt(u.standard_error[i]),
                _fmt(u.relative_standard_error[i]),
                _fmt(u.lower[i]), _fmt(u.upper[i]), unit), widths)
    end
    d = u.diagnostics
    print(io, "diagnostics: ‖grad‖∞=", _fmt(d.gradient_norm),
        ", cond(H)=", _fmt(d.condition_number),
        ", symmetry=", _fmt(d.hessian_symmetry),
        ", cross-check=", _fmt(d.hessian_cross_check))
end

function Base.show(io::IO, u::FixedEffectUncertainty)
    percent = round(Int, 100 * u.level)
    println(io, "FixedEffectUncertainty (observed information; $(percent)% CI; method=$(u.method))")
    _show_parameter_table(io, u)
end

function Base.show(io::IO, u::MixedEffectUncertainty)
    percent = round(Int, 100 * u.level)
    println(io, "MixedEffectUncertainty (PROVISIONAL — conditional on empirical-Bayes")
    println(io, "random effects; not qualified, Gate 5; $(percent)% CI; method=$(u.method))")
    _show_parameter_table(io, u)
    println(io)
    for (j, index) in enumerate(u.random_effect_indices)
        println(io, "  omega[z$index] SD = ", _fmt(u.omega_sd[j]),
            ",  shrinkage = ", _fmt(u.shrinkage[j]))
    end
end

_fmt(x::Real) = string(round(Float64(x); sigdigits=5))
function _row(io, cells, widths)
    for (cell, width) in zip(cells, widths)
        print(io, rpad(cell, width))
    end
    println(io)
end
