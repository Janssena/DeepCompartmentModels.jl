import Statistics: Statistics, std
import LinearAlgebra: Diagonal
import Random: AbstractRNG, rand!
import Bijectors

using Distributions
using PartialFunctions

abstract type AbstractObjective end
abstract type FixedObjective <: AbstractObjective end
abstract type MixedObjective <: AbstractObjective end

########## FixedObjectives

"""
    SSE()

Sum of squared errors objective function:

    L(p) = Σᵢ (yᵢ - f(xᵢ; p))²
"""
struct SSE <: FixedObjective end

objective(model::M, args...) where {M<:AbstractModel} = objective(model.objective, model, args...)

"""Expects that forward returns a one-dimensional array"""
function objective(::SSE, model, container::Union{AbstractIndividual, Population}, p_)
    p = constrain(p_)
    ŷ, st = forward_adjoint(model, container, p)
    return sse(container.y, ŷ)
end

sse(y::T, ŷ::T) where T<:AbstractVector{<:AbstractVector} = sum(abs2, reduce(vcat, y - ŷ))
sse(y::AbstractVector{<:AbstractVector}, ŷ::AbstractVector) = sum(abs2, reduce(vcat, y) - ŷ)

# TODO: Potentially add a objective(::SSE, ŷ::AbstractVector{<:Real}) and objective(::SSE, ŷ::AbstractVector{<:AbstractVector}). The latter can do only a single reduce

# TODO: Force the D in LogLikelihood to have eltype Float32
"""
    LogLikelihood{D, E}()

LogLikelihood based objective function:

    L(p) = p(y | p, σ)

Default uses a Normal / MultivariateNormal loglikelihood function 
(as represented by `D`). Different distributions can be passed, and custom 
parameters can be controlled using the Custom ErrorModel.
"""
struct LogLikelihood{D<:Sampleable, E<:ErrorModel} <: FixedObjective 
    error::E
    # Constructors
    LogLikelihood(error::E=Additive()) where {E<:ErrorModel} = new{Normal{Float32},E}(error)
    LogLikelihood(::Type{D}, error::E=Additive()) where {D<:Sampleable,E<:ErrorModel} = new{D,E}(error)
end

"""Expects that forward returns a one-dimensional array"""
function objective(obj::LogLikelihood{D,E}, model::M, container::Union{AbstractIndividual, Population}, p_) where {D,E,M<:AbstractModel}
    p = constrain(p_)
    ŷ, st = forward_adjoint(model, container, p)
    σ² = variance(obj.error, p, ŷ)
    return -ll(D, ŷ, σ², container.y)
    # return -logpdf(MultivariateNormal(ŷ, σ²), reduce(vcat, container.y))
end

ll(::Type{<:Normal}, ŷ::T, σ², y::T) where T<:AbstractVector{<:AbstractVector} = sum(logpdf.(MultivariateNormal.(ŷ, σ²), y))
ll(::Type{<:Normal}, ŷ::T, σ², y) where T<:AbstractVector{<:Real} = logpdf(MultivariateNormal(ŷ, σ²), reduce(vcat, y))


# """Generic function for any likelihood distribution"""
# function objective(obj::LogLikelihood, model, container::Union{AbstractIndividual, Population}, p)
#     throw(ErrorException("Not implemented yet."))
# end

########## MixedObjectives

function indicator(n::Integer, a::AbstractVector{<:Integer}, ::Type{T}=Float32) where T
    Iₐ = zeros(T, n, length(a))
    for i in eachindex(a)
        Iₐ[a[i], i] = 1
    end
    return Iₐ
end

# TODO: update_mask!(objective, idxs, p_length) # When we want to choose new set of parameters with random effects. 
# TODO: set_mask!(objective, p_length) -> should also update model.p

"""
    init_omega(rng, n)
    
Initialization function for the MultivariateNormal prior on random effect 
parameters.
"""
function init_omega(rng::Random.AbstractRNG, n, ::Type{T}=Float32; omega_dist::Sampleable=Normal(0.2, 0.03), C_shape::Real=50.) where T<:Float32 
    ω_init = zeros(T, n)
    rand!(rng, omega_dist, ω_init)
    ω_init = max.(ω_init, zero(T)) .+ T(1e-3)

    C_init = zeros(T, n, n)
    rand!(rng, LKJ(n, C_shape), C_init)

    return (omega = (var = softplus_inv.(ω_init), corr = Bijectors.VecCorrBijector()(C_init)),)
end

(init_omega)(; kwargs...) = init_omega$(; kwargs...)

init_params(model) = init_params(model.rng, model.objective, model.ann)
init_params!(model) = update!(model, init_params(model.rng, model.objective, model.ann))
init_params(rng, objective, ann::Lux.AbstractExplicitLayer) = init_params(rng, objective, Lux.setup(rng, ann)...)

function init_params(rng, objective, ps::NamedTuple, st::NamedTuple)
    p = (weights = ps, st = st)
    # Init error model
    if !(objective isa SSE)
        p = merge(p, (error = objective.error.init_f(rng, objective.error), ))
    end
    # init random effect prior
    if objective isa MixedObjective
        p = merge(p, objective.init_prior(rng, length(objective.idxs)))
    end
    # Init objective-specific parameters:
    return init_params(rng, objective, p)
end

init_params(::Random.AbstractRNG, ::O, p::NamedTuple) where {O<:AbstractObjective} = p

function init_phi(model::AbstractModel{O,M,P}, pop::Population; sigma_dist::Sampleable=Normal(0.2, 0.03), C_shape::Real=50.) where {O<:MixedObjective,M,P} 
    n = length(pop)
    num_random_effects = size(model.p.omega.var, 1)
    T = eltype(model.p.omega.var)
    
    sigma_init = zeros(T, num_random_effects, n)
    rand!(model.rng, sigma_dist, sigma_init)
    sigma_init = max.(sigma_init, zero(T))  .+ T(1e-3)
    
    p_ = (mean = zeros(T, num_random_effects, n), sigma = softplus_inv.(sigma_init),)
    if FullRank in typeof(model.objective).parameters
        C_init = zeros(T, num_random_effects, num_random_effects, n)
        rand!(model.rng, LKJ(num_random_effects, C_shape), C_init)
        C_init_vec = reduce(hcat, [Bijectors.VecCholeskyBijector(:L)(C_init[:, :, i]) for i in 1:size(C_init)[end]])
        p_ = merge(p_, (corr = C_init_vec, ))
    end

    return p_
end

(init_phi)(; kwargs...) = init_phi$(; kwargs...)

abstract type AbstractVariationalApproximation end
struct MeanField <: AbstractVariationalApproximation end
struct FullRank <: AbstractVariationalApproximation end

abstract type AbstractQuadratureApproximation end
struct MonteCarlo <: AbstractQuadratureApproximation
    n_samples::Int # Number of samples per individual
    MonteCarlo() = new(1)
    MonteCarlo(i) = new(i)
end # Takes N samples for each individual at each iteration
struct SampleAverage{T} <: AbstractQuadratureApproximation 
    n_samples::Int # Number of samples per individual
    samples::Vector{T} # samples need to be initialized based on the population. We then push! additional matrices to this vector
    SampleAverage(n=20, samples::T=Matrix{Float32}[]) where T = new{T}(n, samples)
end # N fixed samples

Base.show(io::IO, sa::SampleAverage) = print(io, "SampleAverage($(sa.n_samples), $(isempty(sa.samples) ? "uninitialized" : "..."))")

struct VariationalELBO{V<:AbstractVariationalApproximation,A<:AbstractQuadratureApproximation,E<:ErrorModel,F1,F2} <: MixedObjective
    error::E
    approx::A # MonteCarlo or SampleAverage
    idxs::Vector{Int}
    # mask::ElasticArray{T} # Needs to be an ElasticArray in order to be extended at later time.
    init_prior::F1
    init_phi::F2
    VariationalELBO(idxs::AbstractVector{<:Int}; kwargs...) = VariationalELBO(Additive(), idxs; kwargs...)
    function VariationalELBO(error::E, idxs::AbstractVector{<:Int}; type::V=FullRank(), approx::A=MonteCarlo(), init_prior::F1=init_omega, init_phi::F2=init_phi) where {V<:AbstractVariationalApproximation,A<:AbstractQuadratureApproximation,E<:ErrorModel,F1,F2}
        new{V,A,E,F1,F2}(error, approx, idxs, init_prior, init_phi)
    end
end

init_samples!(obj::VariationalELBO{V,A,E,F1,F2}, population::Population) where {V,A<:SampleAverage,E,F1,F2} = init_samples!(obj.approx, length(obj.idxs), length(population))

function init_samples!(sa::SampleAverage, k::Int, n::Int) 
    empty!(sa.samples)
    push!(sa.samples, eachslice(randn(Float32, k, sa.n_samples, n), dims=3)...)
    return nothing
end

Base.show(io::IO, ::SSE) = print(io, "SSE")
Base.show(io::IO, obj::LogLikelihood{D,E}) where {D,E} = print(io, "LogLikelihood{$(D.name.name), $(obj.error)}")
Base.show(io::IO, obj::VariationalELBO{V,A,E,F1,F2}) where {V,A,E,F1,F2} = print(io, "VariationalELBO{$(V.name.name), $(obj.approx), $(obj.error)}")
Base.show(io::IO, obj::O) where {O<:AbstractObjective} = print(io, "$(O.name.name){$(obj.error)}")