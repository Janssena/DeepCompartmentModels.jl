import Statistics

abstract type AbstractErrorModel end

Statistics.std(model::AbstractModel, args...) = std(model.error, args...)
Statistics.var(model::AbstractModel, args...) = var(model.error, args...)
Statistics.var(error::AbstractErrorModel, ŷ, ps) = Diagonal(abs2.(std(error, ŷ, ps)))

make_dist(model::AbstractModel, args...) = make_dist(model.error, args...)

# For objectives with Monte Carlo samples
make_dist(error::AbstractErrorModel, ŷ::AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}}, ps) = 
    make_dist.((error, ), ŷ, (ps, ))

make_dist(error::AbstractErrorModel, ŷ::AbstractVector{<:AbstractVector{<:Real}}, ps) = 
    make_dist.((error, ), ŷ, (ps, ))

"""
    ImplicitError

No error model present, as is the case when optimizing the sum of squared 
errors.
"""
struct ImplicitError <: AbstractErrorModel end

Statistics.std(::ImplicitError, args...) = nothing

Base.show(io::IO, ::E) where {E<:ImplicitError} = print(io, "ImplicitError")

"""
    AdditiveError

AdditiveError error model following y = f(x) + ϵ

# Arguments
- `init_f`: Function to initialize parameters. Default = init_sigma.
"""
struct AdditiveError{F} <: AbstractErrorModel
    num_params::Int
    init_f::F
    function AdditiveError(; init::Real=0.1)
        f = init_sigma(; init_dist = LogNormal(log(init), 0.3))
        return new{typeof(f)}(1, f)
    end
end

make_dist(::AdditiveError, ŷ::AbstractVector{<:Real}, ps) = 
    TuringScalMvNormal(ŷ, only(ps.error.σ))

Statistics.std(::AdditiveError, ŷ::AbstractVector{T}, ps) where T<:Real = fill(only(ps.error.σ), length(ŷ))

"""
    ProportionalError

ProportionalError error model following y = f(x) + f(x) ⋅ ϵ

# Arguments
- `init_f`: Function to initialize parameters. Default = init_sigma.
"""
struct ProportionalError{F} <: AbstractErrorModel
    num_params::Int
    init_f::F
    function ProportionalError(; init::Real=0.1)
        f = init_sigma(; init_dist = LogNormal(log(init), 0.3))
        return new{typeof(f)}(1, f)
    end
end

make_dist(error::ProportionalError, ŷ::AbstractVector{T}, ps) where T<:Real = 
    TuringDiagMvNormal(ŷ, std(error, ŷ, ps))

Statistics.std(::ProportionalError, ŷ::AbstractVector{T}, ps) where T<:Real = ŷ .* only(ps.error.σ) .+ T(1e-6)

"""
    CombinedError

CombinedError error model following y = f(x) + ϵ₁ + f(x) ⋅ ϵ₂

# Arguments
- `init_f`: Function to initialize parameters. Default = init_sigma.
"""
struct CombinedError{F} <: AbstractErrorModel
    num_params::Int
    init_f::F
    function CombinedError(; init::AbstractVector{<:Real}=[0.1, 0.1])
        f = init_sigma(; init_dist = Product(LogNormal.(log.(init), (0.3, ))))
        return new{typeof(f)}(2, f)
    end
end

make_dist(error::CombinedError, ŷ::AbstractVector{T}, ps) where T<:Real = 
    TuringDiagMvNormal(ŷ, std(error, ŷ, ps))

Statistics.std(::CombinedError, ŷ::AbstractVector{T}, ps) where T<:Real = ps.error.σ[1] .+ ŷ .* ps.error.σ[2]

Base.show(io::IO, error::Union{AdditiveError{F}, ProportionalError{F}, CombinedError{F}}) where {F<:PartialFunctions.PartialFunction} = 
    print(io, "$(typeof(error).name.name){init = $(error.init_f)}")

"""
    CustomError

CustomError error model. Requires to definition of a variance(::CustomError, p, y) function 
describing the variance of the observations and initialization function for its 
parameters.

# Arguments
- `num_params::Int`: Number of parameters to use in the error function.
- `init_f`: Function to initialize parameters. Default = init_sigma.
"""
struct CustomError{F} <: AbstractErrorModel 
    num_params::Int
    init_f::F
    CustomError(num_params, init_f) = new{typeof(init_f)}(num_params, init_f)
end

# TODO: add the option to covariates in here, should be through the passing of individual and taking individual.x.error
make_dist(::CustomError, p, ŷ::AbstractVector) = 
    throw(ErrorException("`make_dist` method not implemented. Overload this function, `constrain_error`, and `Statistics.std` when using CustomError error."))

Base.show(io::IO, ::CustomError) = print(io, "CustomError()")

"""
    init_sigma(rng, error; init_dist)

Initialization function for σ parameters based on the error model.

# Arguments
- `rng::AbstractRNG`: Randomizer to use.
- `error`: Error model. One of AdditiveError, ProportionalError, CombinedError, or CustomError.
- `init_dist::Sampleable`: Distribution from which to sample the initial σ. Default = Uniform(0, 1)
"""
function init_sigma(rng::Random.AbstractRNG, error::AbstractErrorModel, ::MeanSqrt, ::Type{T}=Float32; init_dist::Sampleable) where {T<:Real} 
	init = _sample_sigma(rng, error, init_dist, T)
    return (σ = softplus_inv.(init), )
end

function init_sigma(rng::Random.AbstractRNG, error::AbstractErrorModel, ::MeanVar, ::Type{T}=Float32; init_dist::Sampleable) where {T<:Real} 
	init = _sample_sigma(rng, error, init_dist, T)
    return (σ² = init.^2, )
end

function _sample_sigma(rng::Random.AbstractRNG, error::AbstractErrorModel, init_dist::Sampleable, T)
	init_ = zeros(T, error.num_params)
    Random.rand!(rng, init_dist, init_)
    return max.(init_, zero(T)) .+ T(1e-6)
end

(init_sigma)(; kwargs...) = init_sigma$(; kwargs...)