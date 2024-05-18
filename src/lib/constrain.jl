import Bijectors: inverse, VecCorrBijector, VecCholeskyBijector

"""
    softplus_inv(x::Real)

Returns the inverse of the softplus function such that: \\
`y = softplus(x)` \\
`x = softplus_inv(y)`
"""
softplus_inv(x::T) where {T<:Real} = log(exp(x) - one(T))

"""
    constrain(p::NamedTuple)

Transforms the unconstrained parameter vector to constrained space.

# Examples
`σ* ∈ ℝ → softplus(σ*) ∈ ℝ⁺` \\
`ω, C  → ω ⋅ C ⋅ ω'`
"""
# function constrain(::O, p_::NamedTuple) where O<:SSE
#     p = (weights = p_.weights,)
#     if :error in keys(p_) # Constrain ErrorModel parameters
#         p = merge(p, (error = merge(p_.error, (sigma = softplus.(p_.error.sigma),)),))
#     end

#     if :omega in keys(p_)
#         ω = softplus.(p_.omega.var) # TODO: rename this to sigma or similar, e.g. (prior = (omega = ..., corr = ...), )
#         C = inverse(VecCorrBijector())(p_.omega.corr)
#         p = merge(p, (omega = Symmetric(ω .* C .* ω'),))
#     end
#     return p
# end

constrain_error(p) = (sigma = softplus.(p.error.sigma), )
function constrain_omega(p)
    ω = softplus.(p.omega.var) # TODO: rename this to sigma or similar, e.g. (prior = (omega = ..., corr = ...), )
    C = inverse(VecCorrBijector())(p.omega.corr)
    return Symmetric(ω .* C .* ω')
end

constrain(::O, p::NamedTuple) where O<:SSE = p
constrain(::O, p_::NamedTuple) where O<:LogLikelihood = (weights = p_.weights, error = constrain_error(p_))
constrain(::O, p_::NamedTuple) where O<:MixedObjective = (weights = p_.weights, error = constrain_error(p_), omega = constrain_omega(p_))

"""
    constrain_phi(::MeanField, 𝜙::NamedTuple)

Transforms unconstrained `𝜙` to constrained space. For a MeanField approximation
this function returns `μ` and standard deviations `σ`.
"""
constrain_phi(::Type{MeanField}, 𝜙::NamedTuple) = (mean = 𝜙.mean, sigma = softplus.(𝜙.sigma))

sigma_corr_to_L(sigma, corr) = sigma .* inverse(VecCholeskyBijector(:L))(corr).L

"""
    constrain_phi(::FullRank, 𝜙::NamedTuple)

Transforms unconstrained `𝜙` to constrained space. For a FullRank approximation
this function returns `μ` and the lower cholesky factor `L`.
"""
function constrain_phi(::Type{FullRank}, 𝜙::NamedTuple)
    σ = softplus.(𝜙.sigma)
    L = sigma_corr_to_L.(eachcol(σ), eachcol(𝜙.corr))
    return (mean = 𝜙.mean, L = L)
end
