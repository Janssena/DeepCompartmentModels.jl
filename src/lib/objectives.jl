import Distributions: loglikelihood, kldivergence

abstract type AbstractObjective end
abstract type FixedObjective <: AbstractObjective end

struct MSE <: FixedObjective end
(::MSE)(dcm, data, ps, st) = mse(dcm, data, ps, st)

"""
    mse(dcm, data, ps, st)

Calculates the mean squared error of predictions with respect to the target `y` in the data.

L = 1/n * ∑ᵢ (yᵢ - ŷᵢ)²

## Arguments:
- `dcm`: DeepCompartmentModel to use.
- `data`: Data that is used to make predictions. Can be either a Population or any AbstractIndividual
- `ps`: Model parameters. Contains all learnable parameters.
- `st`: Model state. Contains additional parameters which are deemed constant when calculating gradients.
"""
function mse(model::AbstractDEModel{<:SciMLBase.AbstractDEProblem}, data::D, ps, st; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    z, _ = predict_de_parameters(model, data, ps, st)
    return mse(model, data, z; kwargs...)
end

function mse(model::AbstractDEModel{<:UniversalDiffEq}, data::D, ps, st; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    ŷ = solve_for_target(model, data, ps, st; kwargs...)
    return _mse(get_y(data), ŷ)
end

function mse(model::AbstractDEModel, data::D, z::AbstractArray; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    ŷ = solve_for_target(model, data, z; kwargs...)
    return _mse(get_y(data), ŷ)
end

_mse(y, ŷ) = mean(map(mean ∘ Base.Fix1(broadcast, abs2), y - ŷ))

struct SSE <: FixedObjective end
(::SSE)(dcm, data, ps, st) = sse(dcm, data, ps, st)

"""
    sse(dcm, data, ps, st)

Calculates the sum of squared errors of predictions with respect to the target `y` in the data.

L = ∑ᵢ (yᵢ - ŷᵢ)²

## Arguments:
- `dcm`: DeepCompartmentModel to use.
- `data`: Data that is used to make predictions. Can be either a Population or any AbstractIndividual
- `ps`: Model parameters. Contains all learnable parameters.
- `st`: Model state. Contains additional parameters which are deemed constant when calculating gradients.
"""
function sse(model::AbstractDEModel, data::D, ps, st; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    z, _ = predict_de_parameters(model, data, ps, st)
    return sse(model, data, z; kwargs...)
end

function sse(model::AbstractDEModel{<:UniversalDiffEq}, data::D, ps, st; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    ŷ = solve_for_target(model, data, ps, st; kwargs...)
    return _sse(get_y(data), ŷ)
end

function sse(model::AbstractDEModel, data::D, z::AbstractArray; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    ŷ = solve_for_target(model, data, z; kwargs...)
    return _sse(get_y(data), ŷ)
end

_sse(y, ŷ) = sum(map(sum ∘ Base.Fix1(broadcast, abs2), y - ŷ))


struct LogLikelihood <: FixedObjective end
(::LogLikelihood)(dcm, data, ps, st) = -loglikelihood(dcm, data, ps, st)

abstract type MixedObjective <: AbstractObjective end

_num_random_effects(obj::MixedObjective) = length(obj.idxs)

struct FO <: MixedObjective
    idxs::Vector{Int}
end
# (::FO)(dcm, data, ps, st) = FO(dcm, data, ps, st)

"""
    VariationalELBO(idxs)

Optimise the fixed and random effects by minimizing the ELBO:

ELBO(ps) = Ez~q[p(y | z) + p(z) - q(z)]
         = Ez~q[p(y | z)] - KL[q(z) || p(z)]

# Arguments
- `idxs`: Parameter indexes of ζ for which to estimate random effects.

# Keyword arguments
- `mean_field=false`: Boolean to indicate whether to use the mean field approximation.
- `path_deriv=true`: Boolean to indicate whether to use the path derivative estimator by Roeder et al. (2017).
- `natural=false`: Boolean to indicate whether to use natural gradient descent based optimisation using NaturalGradientOptimisers.jl.
Calculates the gradient with respect to the logjoint instead of the full ELBO.
"""
struct VariationalELBO{MF<:StaticBool, PD<:StaticBool, N<:StaticBool} <: MixedObjective
    idxs::Vector{Int}
    VariationalELBO(idxs::AbstractVector{Int}; mean_field = false, path_deriv = true, natural = false) = 
        new{static(mean_field), static(path_deriv), static(natural)}(idxs)
end

(::VariationalELBO{<:StaticBool,<:StaticBool,<:False})(dcm, data, ps, st) = -elbo(dcm, data, ps, st)

(::VariationalELBO{<:StaticBool,<:StaticBool,<:True})(dcm, data, ps, st) = 
    -loglikelihood(dcm, data, ps, st) + kldivergence(dcm, ps, st)

Base.show(io::IO, obj::VariationalELBO{MF,PD,N}) where {MF,PD,N} =
    print(io, "VariationalELBO{mean_field = $(dynamic(MF())), path_deriv = $(dynamic(PD())), natural = $(dynamic(N()))}(idxs = $(obj.idxs))")


"""
    loglikelihood(dcm, data, ps, st)

Calculates the loglikelihood of observations `y` from the data given a distribution obtained
through a call to `make_dist(model.error, ŷ, ps.error)`.

L = ∑ᵢ logpdf(yᵢ | M)

## Arguments:
- `model`: model to use to make predictions.
- `data`: Data that is used to make predictions. Can be either a Population or any AbstractIndividual.
- `ps`: Model parameters. Contains all learnable parameters.
- `st`: Model state. Contains additional parameters which are deemed constant when calculating gradients.
"""
function Distributions.loglikelihood(model::AbstractDEModel{<:SciMLBase.AbstractDEProblem}, data::D, ps::NamedTuple, st::NamedTuple; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    z, _ = predict_de_parameters(model, data, ps, st)
    return loglikelihood(model, data, z, ps; kwargs...)
end

function Distributions.loglikelihood(model::AbstractDEModel{<:UniversalDiffEq}, data::D, ps::NamedTuple, st::NamedTuple; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    ŷ = solve_for_target(model, data, ps, st; kwargs...)
    dist = make_dist(model.error, ŷ, ps.error)
    return _logpdf(dist, get_y(data))
end

function Distributions.loglikelihood(model::AbstractDEModel, data::D, z::AbstractArray, ps::NamedTuple; kwargs...) where D<:Union{<:Population, <:AbstractIndividual}
    ŷ = solve_for_target(model, data, z; kwargs...)
    dist = make_dist(model.error, ŷ, ps.error)
    return _logpdf(dist, get_y(data))
end

Distributions.loglikelihood(model::AbstractDEModel, data::D, ps, sts::AbstractVector{<:NamedTuple}; kwargs...) where D<:Union{<:Population, <:AbstractIndividual} = qmap(sts) do st
    loglikelihood(model, data, ps, st; kwargs...)
end

function Distributions.kldivergence(dcm::DeepCompartmentModel{D,M}, ps, st) where {D<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    qs = getq(dcm, ps, st)
    ps = getprior(dcm, ps, st)
    return sum(Distributions.kldivergence.(qs, ps))
end

function logprior(::DeepCompartmentModel, ps::NamedTuple{(:theta,:error,:omega,:phi)}, st)
    prior = MvNormal(zeros(eltype(ps.omega), size(ps.omega, 1)), ps.omega)
    η = sample_gaussian(ps.phi, st.phi)
    return _logpdf(prior, η)
end

function logq(dcm::DeepCompartmentModel, ps, st)
    qs = getq(dcm, ps, st)
    η = sample_gaussian(ps.phi, st.phi)
    return _logpdf(qs, η)
end

getq(::DeepCompartmentModel{P,M}, ps, st) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer} = 
    getq(ps.phi)

getq(ps::NamedTuple{(:μ, :Σ), <:Tuple{<:AbstractVector{<:AbstractVector{<:Real}}, <:AbstractVector{<:Symmetric}}}) = 
    MvNormal.(ps.μ, ps.Σ)

getq(ps::NamedTuple{(:μ, :σ²), <:Tuple{<:AbstractVector{<:AbstractVector{<:Real}}, <:AbstractVector{<:AbstractVector{<:Real}}}}) = 
    MvNormal.(ps.μ, Diagonal.(ps.σ²))

function getq(ps::NamedTuple{(:μ, :L), <:Tuple{<:AbstractVector{<:AbstractVector{<:Real}}, <:AbstractVector{<:LowerTriangular}}})
    Σ = map(ps.L) do L
        Symmetric(L * L') + eltype(L).(I(size(L, 1)) * 1e-6)
    end
    return MvNormal.(ps.μ, Σ)
end

getq(ps::NamedTuple{(:μ, :σ), <:Tuple{<:AbstractVector{<:AbstractVector{<:Real}}, <:AbstractVector{<:AbstractVector{<:Real}}}}) = 
    MvNormal.(ps.μ, map(Base.Fix1(broadcast, softplus), ps.σ)) # σ -> σ² is happening in Distributions.jl

getq(ps::NamedTuple{(:μ,:Σ)}) = MvNormal(ps.μ, ps.Σ)
getq(ps::NamedTuple{(:μ,:L)}) = MvNormal(ps.μ, Symmetric(ps.L * ps.L'))
getq(ps::NamedTuple{(:μ,:σ²)}) = MvNormal(ps.μ, Diagonal(ps.σ²))
getq(ps::NamedTuple{(:μ,:σ)}) = MvNormal(ps.μ, softplus.(ps.σ)) # σ -> σ² is happening in Distributions.jl

logjoint(dcm, data, ps, st; kwargs...) = 
    loglikelihood(dcm, data, ps, st; kwargs...) + logprior(dcm, ps, st)

logjoint(dcm::DeepCompartmentModel, data::D, ps, sts::AbstractVector{<:NamedTuple}; kwargs...) where D<:Union{<:Population, <:AbstractIndividual} = qmap(sts) do st
    logjoint(dcm, data, ps, st; kwargs...)
end

elbo(dcm::DeepCompartmentModel, data::D, ps, st::NamedTuple; kwargs...) where D<:Union{<:Population, <:AbstractIndividual} = 
    logjoint(dcm, data, ps, st; kwargs...) - logq(dcm, ps, st)

elbo(dcm::DeepCompartmentModel, data::D, ps, sts::AbstractVector{<:NamedTuple}; kwargs...) where D<:Union{<:Population, <:AbstractIndividual} = qmap(sts) do st
    elbo(dcm, data, ps, st; kwargs...)
end

_logpdf(dists::AbstractVector{<:AbstractVector{<:Distribution}}, x::AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}}) = 
    sum(map(_logpdf, dists, x))

_logpdf(dists::AbstractVector{<:Distribution}, x::AbstractVector{<:AbstractVector{<:Real}}) = 
    sum(logpdf.(dists, x))

_logpdf(dist::Distribution, x::AbstractVector{<:AbstractVector{<:Real}}) = 
    sum(map(Base.Fix1(logpdf, dist), x))

_logpdf(dist::Distribution, x::AbstractVector{<:Real}) = logpdf(dist, x)