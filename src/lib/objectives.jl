import Zygote.ChainRules: ignore_derivatives

abstract type AbstractObjective end
abstract type FixedObjective <: AbstractObjective end
abstract type MixedObjective <: AbstractObjective end

objective(obj::MixedObjective, ::AbstractModel{T,M,E}, args...) where {T,M,E<:ImplicitError} = 
    throw(ErrorException("$(obj.name.name) objective does not support `ImplicitError`. Use a different error model."))


################################################################################
##########                 Evaluation of loglikelihoods               ##########
################################################################################

# Monte Carlo samples for p(y | ŷ)
_logpdf(dist::AbstractVector{<:AbstractVector{<:MultivariateDistribution}}, y::AbstractVector{<:AbstractVector{<:Real}}) = 
    _logpdf.(dist, (y, ))

# p(y | ŷ)
_logpdf(dist::AbstractVector{<:MultivariateDistribution}, y::AbstractVector{<:AbstractVector{<:Real}}) = 
    sum(_logpdf.(dist, y))

# p(yᵢ | ŷᵢ)
_logpdf(dist::MultivariateDistribution, y::AbstractVector{<:Real}) = 
    logpdf(dist, y)
_logpdf(dist::AbstractVector{<:MultivariateDistribution}, y::AbstractVector{<:Real}) = 
    _logpdf.(dist, (y, ))

# p(η) if multi-dimensional η
_logpdf(dist::MultivariateDistribution, y::AbstractMatrix{<:Real}) = 
    sum(_logpdf.((dist, ), eachcol(y)))
_logpdf(dist::MultivariateDistribution, y::AbstractVector{<:AbstractMatrix{<:Real}}) = 
    _logpdf.((dist, ), y)
# p(η) if single η per individual
_logpdf(dist::UnivariateDistribution, y::AbstractVector{<:Real}) = 
    sum(map(Base.Fix1(logpdf, dist), y))
_logpdf(dist::UnivariateDistribution, y::AbstractVector{<:AbstractVector{<:Real}}) = 
    _logpdf.((dist,), y)
# q(η)
_logpdf(dist::AbstractVector{<:MultivariateDistribution}, y::AbstractMatrix{<:Real}) = 
    sum(logpdf.(dist, eachcol(y)))
_logpdf(dist::AbstractVector{<:UnivariateDistribution}, y::AbstractVector{<:Real}) = 
    sum(logpdf.(dist, eachcol(y)))
# _logpdf(dist::MultivariateDistribution, y::AbstractVector{<:AbstractVector{<:Real}}) = 
#     logpdf.((dist, ), y)

################################################################################
##########                Fixed effects based objectives              ##########
################################################################################

forward(::AbstractObjective, model::AbstractModel, data, ps, st) = predict(model, data, ps, st)[1]

# TODO: Also return st
function forward(obj::AbstractObjective, model::AbstractDEModel, data, ps, st)
    z, _ = predict_de_parameters(obj, model, data, ps, st)
    return forward_ode_with_dv(model, data, z)
end

##### Sum of squared errors

"""
    SSE()

Sum of squared errors objective function:

    L(ps) = Σᵢ (yᵢ - f(xᵢ; ps))²
"""
struct SSE <: FixedObjective end

function sse(model, data, ps, st) 
    ŷ = forward(SSE(), model, data, ps, st)
    return sse(get_y(data), ŷ)
end

sse(y::AbstractVector{<:AbstractVector{<:Real}}, ŷ::AbstractVector{<:AbstractVector{<:Real}}) = sum(abs2, reduce(vcat, y - ŷ))
sse(y::AbstractVector{<:AbstractVector{<:Real}}, ŷ::AbstractVector{<:Real}) = sum(abs2, reduce(vcat, y) - ŷ)

"""Expects that forward returns a one-dimensional array"""
objective(obj::SSE, model::AbstractModel, data::Union{AbstractIndividual, Population}, ps, st) = 
    sse(model, data, constrain(obj, model, ps), st)

##### Loglikelihood based objectives
"""
    LogLikelihood()

LogLikelihood based objective function:

    L(ps) = p(y | ps)

Custom parameterizations and distributions can be controlled using CustomError.
"""
struct LogLikelihood <: FixedObjective end

#TODO: join these with a forward function?
function Distributions.loglikelihood(obj::LogLikelihood, model, data, ps, st)
    ŷ = forward(obj, model, data, ps, st)
    dist = make_dist(model, ŷ, ps)
    return _logpdf(dist, get_y(data))
end

# For models resulting in multiple dists due to MonteCarlo sampling
# _logpdf(dist::AbstractVector{<:AbstractVector{<:Distribution}}, y::AbstractVector{<:AbstractVector{<:Real}}) = 
#     _logpdf.(dist, (y, ))

# For dists related to a specific array in y
# _logpdf(dist::AbstractVector{<:Distribution}, y::AbstractVector{<:AbstractVector{<:Real}}) = 
#     sum(logpdf.(dist, y))

# _logpdf(dist::AbstractVector{<:Distribution}, y::AbstractVector{<:AbstractMatrix{<:Real}}) = 
#     logpdf.((dist, ), y)

# _logpdf(dist::Distribution, y::AbstractVector{<:AbstractArray{<:Real}}) = 
#     _logpdf.((dist, ), y)

# _logpdf(dist::Distribution, y::AbstractArray{<:Real}) = sum(logpdf(dist, y))
# _logpdf(dist::AbstractVector{<:Distribution}, y::AbstractVector{<:Real}) = logpdf.(dist, (y, ))
# _logpdf(dist::AbstractVector{<:Distribution}, y::AbstractMatrix{<:Real}) = sum(logpdf.(dist, eachcol(y)))





# _logpdf(dist::Sampleable, y::AbstractVector{<:AbstractVector{<:Real}}) = 
#     sum(logpdf.((dist, ), y))

# _logpdf(dist::Sampleable, y::AbstractVector{<:Real}) = logpdf(dist, y)

objective(obj::LogLikelihood, model::AbstractModel, data, ps, st) = 
    -loglikelihood(obj, model, data, constrain(model, ps), st)

objective(::LogLikelihood, ::AbstractModel{T,M,E}, args...) where {T,M,E<:ImplicitError} = 
    throw(ErrorException("The LogLikelihood objective does not support `ImplicitError`. Use a different error model."))

##### get differential equation parameters

function predict_de_parameters(::Union{FixedObjective, Type{FixedObjective}}, model::AbstractDEModel, data, ps, st)
    ζ, st = predict_typ_parameters(model, data, ps, st)
    return construct_p(ζ, data), st
end

################################################################################
##########                Mixed effects based objectives              ##########
################################################################################

# TODO: version that works with time-dependent z
function ∂forward_ode_∂η(obj::MixedObjective, model::AbstractDEModel, individual::AbstractIndividual, z::AbstractVector{<:Real})
    jac = FiniteDiff.finite_difference_jacobian(
        z_ -> forward_ode_with_dv(model, individual, z_), 
        z, 
        Val(:central); # :forward is mutating
        colorvec = maximum(obj.idxs)
    )
    return jac[:, obj.idxs] .* z[obj.idxs]' # Chain rule: ∂F/∂η = ∂F/∂z ⋅ ∂z/∂η
end

##### Get differential equation parameters
function predict_de_parameters(::Union{MixedObjective, Type{MixedObjective}}, model::AbstractDEModel, data, ps, st__)
    ζ, st_ = predict_typ_parameters(model, data, ps, st__)
    η, st = get_random_effects(ps, st_)
    return construct_p(ζ, η, data), st
end

##### First-order (FO) approximation

struct FO <: MixedObjective 
    idxs::Vector{Int}
    FO(idxs) = new(idxs)
end

function Distributions.loglikelihood(obj::FO, model::AbstractDEModel, container, ps, st_)
    ζ, st = predict_de_parameters(obj, model, container, ps, st_)
    return first_order_approx(obj, model, container, ζ, ps), st
end

first_order_approx(obj::FO, model::AbstractDEModel, population::Population, ζ::AbstractMatrix{<:Real}, ps) = 
    sum(first_order_approx.((obj, ), (model, ), population, eachcol(ζ), (ps, )))

first_order_approx(obj::FO, model::AbstractDEModel, population::Population, ζ::AbstractVector{<:AbstractVector{<:Real}}, ps) = 
    sum(first_order_approx.((obj, ), (model, ), population, ζ, (ps, )))

function first_order_approx(obj::FO, model::AbstractDEModel, individual::AbstractIndividual, ζ::AbstractVector, ps)
    ŷ = forward_ode_with_dv(model, individual, ζ)
    Gᵢ = ∂forward_ode_∂η(obj, model, individual, ζ)
    Σ = var(model, ŷ, ps)
    residuals = get_y(individual) - ŷ
    Cᵢ = Gᵢ * _get_omega_cov(ps) * Gᵢ' + Σ
    return logdet(Cᵢ) + residuals' * inv(Cᵢ) * residuals
end

objective(obj::FO, model::AbstractDEModel, container, ps, st) = 
    -first(loglikelihood(obj, model, container, constrain(obj, model, ps), st))

_get_omega_cov(ps::NamedTuple) = _get_omega_cov(ps.omega)
_get_omega_cov(omega::NamedTuple{(:Σ,),<:Any}) = only(omega)
_get_omega_cov(omega::NamedTuple{(:σ,),<:Any}) = fill(only(omega)^2, 1, 1)
_get_omega_cov(omega::NamedTuple{(:σ²,),<:Any}) = fill(only(omega), 1, 1)

##### get differential equation parameters

function predict_de_parameters(::FO, model::AbstractDEModel, data, ps, st)
    ζ, st = predict_typ_parameters(model, data, ps, st)
    return construct_p(ζ, data), st
end

##### First-order conditional estimation (FOCE) approximation

struct FOCE <: MixedObjective
    idxs::Vector{Int}
    FOCE(idxs::AbstractVector{<:Int}) = new(idxs)
end

function Distributions.loglikelihood(obj::FOCE, model::AbstractDEModel, container, ps, st_)
    z, st = predict_de_parameters(obj, model, container, ps, st_)
    return foce(obj, model, container, z, ps), st
end

# TODO: Check if this works on individuals?
logprior(obj::FOCE, ps::NamedTuple, st::NamedTuple) = 
    logprior(obj, inv(_get_omega_cov(ps)), st.phi.eta)

logprior(::FOCE, Ω⁻¹::AbstractMatrix, η::AbstractMatrix{<:Real}) = 
    sum(_approx_logprior.((Ω⁻¹, ), eachcol(η)))

logprior(::FOCE, Ω⁻¹::AbstractMatrix, η::AbstractVector{<:Real}) = 
    sum(η.^2 .* only(Ω⁻¹))

_approx_logprior(Ω⁻¹::AbstractMatrix, η::AbstractVector) = η' * Ω⁻¹ * η

foce(obj::FOCE, model::AbstractDEModel, population::Population, ζ::AbstractMatrix{<:Real}, ps) = 
    sum(foce.((obj, ), (model, ), population, eachcol(ζ), (ps, )))

foce(obj::FOCE, model::AbstractDEModel, population::Population, ζ::AbstractVector{<:AbstractVector{<:Real}}, ps) = 
    sum(foce.((obj, ), (model, ), population, ζ, (ps, )))

function foce(obj::FOCE, model::AbstractDEModel, individual::AbstractIndividual, z::AbstractVector, ps)
    ŷ = forward_ode_with_dv(model, individual, z)
    Gᵢ = ∂forward_ode_∂η(obj, model, individual, z)
    Σ = var(model, ŷ, ps)
    residuals = get_y(individual) - ŷ
    Cᵢ = Gᵢ * _get_omega_cov(ps) * Gᵢ' + Σ
    return logdet(Cᵢ) + residuals' * inv(Σ) * residuals # + individual.eta' * inv(Ω) * individual.eta
end

function objective(obj::FOCE, model::AbstractDEModel, container, ps_, st_) 
    ps = constrain(obj, model, ps_)
    LL, st = loglikelihood(obj, model, container, ps, st_)
    return -(LL + logprior(obj, ps, st))
end

##### LogJoint for the optimization of etas

function optimize_etas(model::AbstractDEModel, population::Population, ps_, st)
    ps = constrain(model, ps_)
    etas = optimize_etas.((model, ), population, (ps, ), (st, ))
    return reduce(hcat, etas)
end

function optimize_etas(model::AbstractDEModel{T,P,M,E,S}, individual::AbstractIndividual, ps_, st) where {T,P,M,E,S}
    ps = constrain(model, ps_)
    result = Optim.optimize(
        eta -> -logjoint(model, individual, ps, _etas_to_state(st, eta)), 
        zeros(T, size(st.phi.mask, 2)),
        # Optim.Options() # TODO: set obj_fn tolerance
        )
    return result.minimizer
end

function logjoint(model::AbstractDEModel, individual::AbstractIndividual, ps, st_)
    # TODO: would be nice if we could do this differently, i.e. 
    # loglikelihood(model, data, z, ps, st), so that we don't have to repeat 
    # this. This also works with stuff above.
    z, st = predict_de_parameters(MixedObjective, model, individual, ps, st_)
    ŷ = forward_ode_with_dv(model, individual, z)
    dist = make_dist(model, ŷ, ps)
    LL = _logpdf(dist, get_y(individual))

    prior = _get_prior(ps)
    
    return LL + logprior(prior, st.phi.eta)
end

_etas_to_state(st, etas) = merge(st, (phi = merge(st.phi, (eta = etas, )), ))

##### VariationalELBO

abstract type VariationalApproximation end
struct MeanField <: VariationalApproximation end
struct FullRank <: VariationalApproximation end

# TODO: replace Val{true} with Static.True (i.e. Lux.True)
struct VariationalELBO{approx<:VariationalApproximation,path_deriv} <: MixedObjective 
    idxs::Vector{Int}
    VariationalELBO(idxs, approx; path_deriv::Bool=true) = 
        new{typeof(approx),Val{path_deriv}}(idxs)
    function VariationalELBO(idxs; path_deriv::Bool=true)
        approx = length(idxs) > 1 ? FullRank() : MeanField()
        return new{typeof(approx),Val{path_deriv}}(idxs)
    end
end

function Distributions.loglikelihood(obj::VariationalELBO, model::AbstractDEModel, container, ps, st_)
    z, st = predict_de_parameters(obj, model, container, ps, st_)
    ŷs = forward_ode_with_dv(model, container, z)
    dist = make_dist(model, ŷs, ps)
    return _logpdf(dist, get_y(container)), st
end

logprior(::VariationalELBO, ps, st) = logprior(_get_prior(ps), st.phi.eta)

_get_prior(ps::NamedTuple) = _get_prior(only(ps.omega))
_get_prior(Ω::Symmetric) = TuringDenseMvNormal(zeros(eltype(Ω), size(Ω, 1)), Ω)
_get_prior(L::LowerTriangular) = TuringDenseMvNormal(zeros(eltype(L), size(L, 1)), Cholesky(L))
_get_prior(ω::Real) = Normal(zero(ω), ω)

logprior(prior::Distribution, η) = _logpdf(prior, η)

logq(obj::VariationalELBO{approx,path_deriv}, ps, st) where {approx,path_deriv<:Val{false}} = logq(getq(obj, ps.phi), st.phi.eta)
logq(obj::VariationalELBO{approx,path_deriv}, ps, st) where {approx,path_deriv<:Val{true}} = logq(getq(obj, ignore_derivatives(ps.phi)), st.phi.eta)

logq(q, η::AbstractArray{<:Real}) = _logpdf(q, η)
# For monte carlo samples:
logq(q::AbstractVector{<:MultivariateDistribution}, η::AbstractVector{<:AbstractMatrix{<:Real}}) = _logpdf.((q,), η) # Multivariate eta
logq(q::MultivariateDistribution, η::AbstractVector{<:AbstractVector{<:Real}}) = _logpdf.((q, ), η) # Univariate eta

# TODO: Remove the need for VariationalELBO?
getq(::VariationalELBO, phi::NamedTuple{(:μ,:σ,),<:Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}}) = TuringDiagMvNormal(phi.μ, phi.σ)
getq(::VariationalELBO, phi::NamedTuple{(:μ,:σ,),<:Any}) = TuringDiagMvNormal.(phi.μ, phi.σ)
# TODO: When running constrain(obj, ps) calculate L and put in ps.phi (we could also put q in ps.phi instead; will be easier to extend in the future)
getq(::VariationalELBO, phi::NamedTuple{(:μ,:L,)}) = TuringDenseMvNormal.(phi.μ, Cholesky.(phi.L))
getq(::VariationalELBO, phi::NamedTuple{(:μ,:Σ,)}) = TuringDenseMvNormal.(phi.μ, phi.Σ)
 
function elbo(obj::VariationalELBO, model::AbstractDEModel, container, ps, st_)
    LL, st = loglikelihood(obj, model, container, ps, st_)
    return LL + logprior(obj, ps, st) - logq(obj, ps, st)
end

objective(obj::VariationalELBO, model::AbstractDEModel, container, ps, st) = 
    -mean(elbo(obj, model, container, constrain(obj, model, ps), st))

# """
# Hack that solves accumulation of gradients of LowerTriangular variables in phi 
# and omega. Likely because we use Cholesky as in above.
# """
# Zygote.accum(x::NamedTuple{(:data,), <:Tuple{<:LowerTriangular}}, ::Nothing) = x.data

"""
Hack that solves accumulation of gradients of Diagonal variables in FO and FOCE.
"""
Zygote.accum(x::NamedTuple{(:diag,), <:Tuple{<:AbstractVector}}, ::Nothing) = Diagonal(x.diag)