# TODO: Put error model logic here.
import Distributions: Sampleable
import Statistics
import Random

abstract type ErrorModel end
abstract type ExplicitErrorModel <: ErrorModel end

Base.show(io::IO, error::E) where {E<:ExplicitErrorModel} = print(io, "$(E.name.name){init = $(error.init_f)}")

"""
    Additive

Additive error model following y = f(x) + ϵ

# Arguments
- `init_f`: Function to initialize parameters. Default = init_sigma.
"""
struct Additive{N<:Tuple,F} <: ExplicitErrorModel
    dims::N
    init_f::F
    Additive(;init_f = init_sigma) = new{Tuple{Int64},typeof(init_f)}((1,), init_f)
end

"""
    Proportional

Proportional error model following y = f(x) + f(x) ⋅ ϵ

# Arguments
- `init_f`: Function to initialize parameters. Default = init_sigma.
"""
struct Proportional{N<:Tuple,F} <: ExplicitErrorModel
    dims::Int
    init_f::F
    Proportional(;init_f = init_sigma) = new{Tuple{Int64},typeof(init_f)}((1,), init_f)
end

"""
    Proportional

Proportional error model following y = f(x) + ϵ₁ + f(x) ⋅ ϵ₂

# Arguments
- `init_f`: Function to initialize parameters. Default = init_sigma.
"""
struct Combined{N<:Tuple,F} <: ExplicitErrorModel
    dims::N
    init_f::F
    Combined(;init_f = init_sigma) = new{Tuple{Int64},typeof(init_f)}((2,), init_f)
end

"""
    Custom

Custom error model. Requires to definition of a variance(::Custom, p, y) function 
describing the variance of the observations and initialization function for its 
parameters.

# Arguments
- `init_f`: Function to initialize parameters. Default = init_sigma.
"""
struct Custom{N<:Tuple,F} <: ErrorModel 
    dims::N
    init_f::F
end

"""
    init_sigma(rng, error; init_dist)

Initialization function for σ parameters based on the error model.

# Arguments
- `rng::AbstractRNG`: Randomizer to use.
- `error`: Error model. One of Additive, Proportional, Combined, or Custom.
- `init_dist::Sampleable`: Distribution from which to sample the initial σ. Default = Uniform(0, 1)
"""
function init_sigma(rng::Random.AbstractRNG, error::E, ::Type{T}=Float32; init_dist::Sampleable=Uniform(0, 1)) where {E<:ExplicitErrorModel,T<:Real} 
    init = zeros(T, error.dims...)
    rand!(rng, init_dist, init)
    init = max.(init, zero(T)) .+ T(1e-6)
    return (sigma = softplus_inv.(init),)
end

(init_sigma)(; kwargs...) = init_sigma$(; kwargs...)

init_sigma(::Random.AbstractRNG, ::E) where E<:Custom = throw(ErrorException("`init_sigma` method not implemented. Overload this function (and the `variance` function) when using Custom error."))
variance(::E, p, ŷ::AbstractVector) where {E<:Custom} = throw(ErrorException("`variance` method not implemented. Overload this function (and the `init_sigma` function) when using Custom error."))

"""
    variance(model, p, y)

Returns the variance of `y` based on the `model`.

# Arguments:
- `model::AbstractModel`: The model.
- `p`: Constrained model parameters.
- `y`: Observations or predictions for which to calculate the variance.
"""
variance(model::AbstractModel, args...)  = variance(model.objective.error, args...)
variance(error::E, p, ŷs::AbstractVector{<:AbstractVector}) where E<:ErrorModel = variance.((error,), (p,), ŷs)
variance(::E, p, ŷ::AbstractVector{<:Real}) where {E<:Additive} = Diagonal(repeat(p.error.sigma.^2, length(ŷ)))
variance(::E, p, ŷ::AbstractVector{<:Real}) where {E<:Proportional} = Diagonal((p.error.sigma .* ŷ).^2)
variance(::E, p, ŷ::AbstractVector{<:Real}) where {E<:Combined} = Diagonal((p.error.sigma[1] .+ p.error.sigma[2] .* ŷ).^2)

"""
    std(model::AbstractModel, prediction::AbstractVector)

Returns the standard deviation of `predictions` based on the `model`.
# Arguments:
- `model::AbstractModel`: The model.
- `y`: Observations or predictions for which to calculate standard deviations.
"""
function Statistics.std(model::M, prediction::AbstractVector) where M<:AbstractModel 
    if model.objective isa SSE 
        return throw(ErrorException("Models fit using the SSE objective have implicit error. Cannot obtain standard deviation of prediction."))
    end
    return sqrt.(diag(variance(model.objective.error, constrain(model.objective, model.p), prediction)))
end
