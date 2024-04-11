# TODO: Put error model logic here.
import Distributions: Sampleable
import Statistics

abstract type ErrorModel end
abstract type ExplicitErrorModel <: ErrorModel end

Base.show(io::IO, error::E) where {E<:ExplicitErrorModel} = print(io, "$(E.name.name){init = $(error.init_f)}")

struct Additive{N<:Tuple,F} <: ExplicitErrorModel
    dims::N
    init_f::F
    Additive(;init_f = init_sigma) = new{Tuple{Int64},typeof(init_f)}((1,), init_f)
end
struct Proportional{N<:Tuple,F} <: ExplicitErrorModel
    dims::Int
    init_f::F
    Proportional(;init_f = init_sigma) = new{Tuple{Int64},typeof(init_f)}((1,), init_f)
end
struct Combined{N<:Tuple,F} <: ExplicitErrorModel
    dims::N
    init_f::F
    Combined(;init_f = init_sigma) = new{Tuple{Int64},typeof(init_f)}((2,), init_f)
end
struct Custom{N<:Tuple,F} <: ErrorModel 
    dims::N
    init_f::F
end

function init_sigma(rng::Random.AbstractRNG, error::E, ::Type{T}=Float32; init_dist::Sampleable=Uniform(0, 1)) where {E<:ExplicitErrorModel,T<:Real} 
    init = zeros(T, error.dims...)
    rand!(rng, init_dist, init)
    init = max.(init, zero(T)) .+ T(1e-6)
    return (sigma = softplus_inv.(init),)
end

(init_sigma)(; kwargs...) = init_sigma$(; kwargs...)

init_sigma(::Random.AbstractRNG, ::E) where E<:Custom = throw(ErrorException("`init_sigma` method not implemented. Overload this function (and the `variance` function) when using Custom error."))
variance(::E, p, ŷ::AbstractVector) where {E<:Custom} = throw(ErrorException("`variance` method not implemented. Overload this function (and the `init_sigma` function) when using Custom error."))

variance(model::AbstractModel, args...)  = variance(model.objective.error, args...)
variance(error::E, p, ŷs::AbstractVector{<:AbstractVector}) where E<:ErrorModel = variance.((error,), (p,), ŷs)
variance(::E, p, ŷ::AbstractVector{<:Real}) where {E<:Additive} = Diagonal(repeat(p.error.sigma.^2, length(ŷ)))
variance(::E, p, ŷ::AbstractVector{<:Real}) where {E<:Proportional} = Diagonal((p.error.sigma .* ŷ).^2)
variance(::E, p, ŷ::AbstractVector{<:Real}) where {E<:Combined} = Diagonal((p.error.sigma[1] .+ p.error.sigma[2] .* ŷ).^2)

function Statistics.std(model::M, prediction::AbstractVector) where M<:AbstractModel 
    if model.objective isa SSE 
        return throw(ErrorException("Models fit using the SSE objective have implicit error. Cannot obtain standard deviation of prediction."))
    end
    return sqrt.(diag(variance(model.objective.error, constrain(model.p), prediction)))
end
