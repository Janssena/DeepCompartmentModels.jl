function get_random_effects(ps::NamedTuple, st::NamedTuple) 
    etas = make_etas(ps.phi, st.phi)
    st_new = _etas_to_state(st, etas)
    return _to_random_eff_matrix(st.phi.mask, etas), st_new
end

"""
Case for the FOCE objective
"""
make_etas(::@NamedTuple{}, st_phi::NamedTuple{(:mask,:eta,),<:Any}) = st_phi.eta

"""
Case for the first-order objectives if no eta in the state.
"""
make_etas(::@NamedTuple{}, st_phi::NamedTuple{(:mask,),<:Any}) = 
    throw(ErrorException("Estimates for η are missing from the state. Run `optimize_etas` to obtain MAP estimates."))

"""
Case for Variational objectives
"""
make_etas(ps_phi::NamedTuple, st_phi::NamedTuple{(:epsilon,:mask,),<:Any}) = 
    make_etas(values(ps_phi)..., st_phi.epsilon)

make_etas(mu::AbstractVector{<:AbstractVector{T}}, var::AbstractVector{<:AbstractArray{T}}, ϵ::AbstractArray{T, 3}) where T<:Real = 
    make_etas.((mu, ), (var, ), eachslice(ϵ; dims = 3))::Vector{Matrix{T}}

make_etas(mu::AbstractVector{<:AbstractVector{<:Real}}, var::AbstractVector{<:AbstractArray{<:Real}}, ϵ::AbstractMatrix{<:Real}) = 
    reduce(hcat, make_etas.(mu, var, eachcol(ϵ)))

make_etas(μ::AbstractVector{<:Real}, Σ::Symmetric, ϵ::AbstractVector{<:Real}) = 
    make_etas(μ, _chol_lower(cholesky(Σ)), ϵ)

make_etas(μ::AbstractVector{<:Real}, L::LowerTriangular, ϵ::AbstractVector{<:Real}) = 
    μ + L * ϵ

make_etas(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, ϵ::AbstractMatrix{<:Real}) = 
    make_etas.((μ, ), (σ, ), eachcol(ϵ))

make_etas(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, ϵ::AbstractVector{<:Real}) = 
    μ + σ .* ϵ

_to_random_eff_matrix(mask::AbstractMatrix, eta::AbstractVector{<:AbstractArray{<:Real}}) = 
    _to_random_eff_matrix.((mask, ), eta)

_to_random_eff_matrix(mask::AbstractMatrix, eta::AbstractMatrix{<:Real}) = 
    mask * eta

_to_random_eff_matrix(mask::AbstractMatrix, eta::AbstractVector{<:Real}) = 
    mask * eta'
