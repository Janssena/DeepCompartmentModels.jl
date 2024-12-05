"""
    softplus_inv(x::Real)

Returns the inverse of the softplus function such that: \\
`y = softplus(x)` \\
`x = softplus_inv(y)`
"""
softplus(x::T) where {T<:Real} = log(exp(x) + one(T))
softplus(x::AbstractArray{T}) where {T<:Real} = softplus.(x)

softplus_inv(x::T) where {T<:Real} = log(exp(x) - one(T))
softplus_inv(x::AbstractArray{T}) where {T<:Real} = softplus_inv.(x)

_chol_lower(a::Cholesky) = a.uplo === 'L' ? a.L : a.U'

########## Constrain functions

constrain(::SSE, ::AbstractModel, ps) = ps

constrain(::LogLikelihood, model::AbstractModel, ps::NamedTuple) = 
    merge(ps, (error = constrain_error(model.error, ps.error), ))

constrain(::MixedObjective, model::AbstractModel, ps::NamedTuple) = 
    merge(ps, (
        error = constrain_error(model.error, ps.error), 
        omega = constrain_omega(ps.omega), 
        phi = constrain_phi(ps.phi)
        )
    )

########## Error

constrain_error(::AbstractErrorModel, ps::NamedTuple{(:σ,)}) = (σ = softplus.(ps.σ), )
constrain_error(::AbstractErrorModel, ps::NamedTuple{(:σ²,)}) = (σ = sqrt.(ps.σ²), )

constrain_error(::CustomError, ps) = 
    throw(ErrorException("`constrain_error` method not implemented. Overload this function, `make_dist`, and `Statistics.std` when using CustomError error."))

########## Omega

constrain_omega(omega::NamedTuple{(:σ,),<:Any}) = (σ = softplus(only(omega.σ)), ) # If omega is σ, it is one-dimensional
constrain_omega(omega::NamedTuple{(:σ²,),<:Any}) = (σ² = only(omega.σ²), )
constrain_omega(omega::NamedTuple{(:L,),<:Any}) = (Σ = Symmetric(omega.L * omega.L'), )
constrain_omega(omega::NamedTuple{(:Σ,),<:Any}) = omega

########## Phi

constrain_phi(phi::@NamedTuple{}) = phi

constrain_phi(phi::NamedTuple{(:μ,:σ),<:Any}) = 
    (μ = phi.μ, σ = softplus.(phi.σ), )

constrain_phi(phi::NamedTuple{(:μ,:σ²),<:Any}) = 
    (μ = phi.μ, σ = sqrt.(phi.σ²), )

constrain_phi(phi::NamedTuple{(:μ,:L),<:Any}) = phi

constrain_phi(phi::NamedTuple{(:μ,:Σ),<:Any}) = 
    (μ = phi.μ, L = _chol_lower.(cholesky.(phi.Σ)), )

########## Detect what constrain function to use

"""Version that estimates what obj was used"""
constrain(model::AbstractModel, ps::NamedTuple{(:theta,),<:Any}) = constrain(SSE(), model, ps)
constrain(model::AbstractModel, ps::NamedTuple{(:theta,:error,),<:Any}) = constrain(LogLikelihood(), model, ps)
constrain(model::AbstractModel, ps::NamedTuple{(:theta,:error,:omega,:phi,),<:Any}) = constrain(VariationalELBO([1]), model, ps)
