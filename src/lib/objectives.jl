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

##### Get differential equation parameters
function predict_de_parameters(::Union{MixedObjective, Type{MixedObjective}}, model::AbstractDEModel, data, ps, st__)
    ζ, st_ = predict_typ_parameters(model, data, ps, st__)
    η, st = get_random_effects(ps, st_)
    return construct_p(ζ, η, data), st
end
    idxs::Vector{Int}

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

"""
Hack that solves accumulation of gradients of Diagonal variables in FO and FOCE.
"""
Zygote.accum(x::NamedTuple{(:diag,), <:Tuple{<:AbstractVector}}, ::Nothing) = Diagonal(x.diag)


