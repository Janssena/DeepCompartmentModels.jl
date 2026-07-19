import Statistics: var

abstract type AbstractErrorModel end

(error::AbstractErrorModel)(ŷ, ps, st; kwargs...) = 
    var(error, ŷ, ps, st; kwargs...)

make_dist(error::AbstractErrorModel, ŷ, ps; kwargs...) = 
    make_dist(error, ŷ, ps, NamedTuple(); kwargs...)

make_dist(error::AbstractErrorModel, ŷ::AbstractVector{<:Real}, ps, st; kwargs...) = 
    MvNormal(ŷ, error(ŷ, ps, st; kwargs...))

make_dist(error::AbstractErrorModel, ŷ::AbstractVector{<:AbstractVector{<:Real}}, ps, st; kwargs...) = map(ŷ) do ŷᵢ
    make_dist(error, ŷᵢ, ps, st; kwargs...)
end

Base.show(io::IO, error::AbstractErrorModel) = 
    print(io, "$(nameof(typeof(error)))(...)")

"""
    ImplicitError

No error model present, as is the case when optimizing the mean or sum of squared 
errors.
"""
struct ImplicitError <: AbstractErrorModel end

var(::ImplicitError, args...) = nothing

Base.show(io::IO, ::E) where {E<:ImplicitError} = print(io, "ImplicitError")

"""
    AdditiveError(init)

AdditiveError error model for Gaussian likelihoods. 

y = f(x) + ϵ

# Arguments
- `init=[]`: Initial value of error standard deviation. Should have length 0 (no initial value) or 1.
"""
struct AdditiveError <: AbstractErrorModel 
    init
    function AdditiveError(init::AbstractVector=Float32[]) 
        if length(init) > 1
            throw(ErrorException("Length of `init` for AdditiveError cannot be greater than 1."))
        end
        new(init)
    end
end

AdditiveError(init::Real) = AdditiveError([init])

var(::AdditiveError, ŷ::AbstractVector{<:Real}, ps, ::NamedTuple) = 
    Diagonal(one.(ŷ) .* softplus(only(ps.σ))^2)

"""
    ProportionalError(init)

ProportionalError error model for Gaussian likelihoods. 
    
y = f(x) + f(x) ⋅ ϵ

# Arguments
- `init=[]`: Initial value of error standard deviation. Should have length 0 (no initial value) or 1.
"""
struct ProportionalError <: AbstractErrorModel
    init
    function ProportionalError(init::AbstractVector=Float32[]) 
        if length(init) > 1
            throw(ErrorException("Length of `init` for ProportionalError cannot be greater than 1."))
        end
        new(init)
    end
end

ProportionalError(init::Real) = ProportionalError([init])

var(::ProportionalError, ŷ::AbstractVector{<:Real}, ps, ::NamedTuple; eps=1e-6) = 
    Diagonal((ŷ .* softplus(only(ps.σ))).^2 .+ eltype(ps.σ)(eps)) # small eps to prevent 0 variances when ŷ = 0

Base.show(io::IO, error::Union{<:AdditiveError, <:ProportionalError}) = 
    print(io, "$(nameof(typeof(error)))(init = $(error.init))")

"""
    CombinedError

CombinedError error model following y = f(x) + ϵ₁ + f(x) ⋅ ϵ₂

# Arguments
- `init=[]`: Initial value of error standard deviations. Should have length 0 (no initial values) or 2.

# Keyword arguments
- `dependent::Bool=false`: Whether the two sources of error are dependent.
"""
struct CombinedError{T<:StaticBool} <: AbstractErrorModel
    init
    function CombinedError(init::AbstractVector=Float32[]; dependent::Bool = false) 
        if length(init) == 1 || length(init) > 2
            throw(ErrorException("Length of `init` for CombinedError cannot be equal to 1 or greater than 2."))
        end
        new{dependent ? True : False}(init)
    end
end

var(::CombinedError{<:True}, ŷ::AbstractVector{<:Real}, ps, ::NamedTuple) = 
    Diagonal((softplus(ps.σ[1]) .+ ŷ .* softplus(ps.σ[2])).^2)

var(::CombinedError{<:False}, ŷ::AbstractVector{<:Real}, ps, ::NamedTuple) = 
    Diagonal((softplus(ps.σ[1])^2 .+ ŷ.^2 .* softplus(ps.σ[2])^2))

Base.show(io::IO, error::CombinedError{T}) where T = 
    print(io, "CombinedError{dependent = $T}(init = $(error.init))")

"""
    CustomError(init; model = nothing)

Custom observation-error model. There are two supported extension routes:

1. With `model = nothing`, overload `Statistics.var(::CustomError, yhat, ps, st)`
   and return the observation covariance matrix. The likelihood is multivariate
   normal with mean `yhat`.
2. Pass a callable `model`. It is called as `model(yhat, ps, st; kwargs...)` and
   must return a `Distribution`. This route supports non-Gaussian likelihoods and
   transformed-scale models such as log-transform-both-sides.

For a statistically proper LTBS likelihood on positive observations, return a
product of `LogNormal(log(yhat), sigma)` distributions. This keeps data and ODE
predictions on their natural scale and includes the transformation Jacobian.

# Arguments
- `init=[]`: Initial values for the custom likelihood parameters.
- `model=nothing`: Optional callable returning a `Distribution`.
"""
struct CustomError{M} <: AbstractErrorModel 
    model::M
    init
    CustomError(init=Float32[]; model = nothing) = new{typeof(model)}(model, init)
end

function make_dist(error::CustomError{Nothing}, ŷ::AbstractVector{<:Real}, ps, st; kwargs...)
    return MvNormal(ŷ, error(ŷ, ps, st; kwargs...))
end

function make_dist(error::CustomError, ŷ::AbstractVector{<:Real}, ps, st; kwargs...)
    dist = error.model(ŷ, ps, st; kwargs...)
    dist isa Distribution || throw(ArgumentError(
        "CustomError model must return a Distributions.jl Distribution, got $(typeof(dist))."))
    return dist
end

# TODO: add covariates, potentially through individual.x.error or state.
var(::CustomError{Nothing}, ŷ::AbstractVector, ps, st) =
    throw(ErrorException("`var` method not implemented. Overload Statistics.var when using CustomError error."))

function var(error::CustomError, ŷ::AbstractVector, ps, st)
    marginal_variances = Distributions.var(make_dist(error, ŷ, ps, st))
    return marginal_variances isa AbstractMatrix ? marginal_variances :
           Diagonal(marginal_variances)
end

Base.show(io::IO, ::CustomError) = print(io, "CustomError(...)")

"""
    ErrorModelSet

Error model consisting of a set of AbstractErrorModels for multiple objectives. 

# Arguments
- `errors`: Tuple containing the different ErrorModels.
"""
struct ErrorModelSet{E} <: AbstractErrorModel
    errors::E
    ErrorModelSet(errors::Vararg{AbstractErrorModel}) = 
        new{typeof(errors)}(errors)
end

var(error::ErrorModelSet, ŷs::AbstractVector{<:AbstractVector{<:Real}}, ps, st) = map(eachindex(ŷs)) do j
    var(error.errors[j], ŷs[j], take_batch(ps, st, j)...)
end

Base.show(io::IO, error::ErrorModelSet) = 
    print(io, "ErrorModelSet($(join([nameof(typeof(e)) for e in error.errors], ", ")))")
    
make_dist(error::ErrorModelSet, ŷs::AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}}, ps, st) = map(ŷs) do ŷᵢ
    make_dist(error, ŷᵢ, ps, st)
end

make_dist(error::ErrorModelSet, ŷs::AbstractVector{<:AbstractVector{<:Real}}, ps, st; kwargs...) = 
    MvNormal.(ŷs, var(error, ŷs, ps, st; kwargs...))
