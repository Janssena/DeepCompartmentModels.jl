function get_random_effects(ps::NamedTuple, st::NamedTuple{(:theta,:phi)}) 
    etas = sample_gaussian(ps.phi, st.phi)
    return _to_random_eff_matrix(st.phi.mask, etas)
end

_to_random_eff_matrix(mask, etas::AbstractVector{<:AbstractVector{<:Real}}) = 
    _to_random_eff_matrix(mask, reduce(hcat, etas))

_to_random_eff_matrix(mask, etas::AbstractArray{<:Real}) = mask * etas

# for a vectors of distribution parameters (i.e. those present in phi)

sample_gaussian(ps::NamedTuple{(:μ, :Σ), <:Tuple{<:AbstractVector{<:AbstractVector{<:Real}}, <:AbstractVector{<:Symmetric}}}, st) = 
    sample_gaussian.(ps.μ, getproperty.(cholesky.(ps.Σ), :L), st.epsilon)

sample_gaussian(ps::NamedTuple{(:μ, :L), <:Tuple{<:AbstractVector{<:AbstractVector{<:Real}}, <:AbstractVector{<:LowerTriangular}}}, st) = 
    sample_gaussian.(ps.μ, ps.L, st.epsilon)

sample_gaussian(ps::NamedTuple{(:μ, :σ), <:Tuple{<:AbstractVector{<:AbstractVector{<:Real}}, <:AbstractVector{<:AbstractVector{<:Real}}}}, st) = 
    sample_gaussian.(ps.μ, map(Base.Fix1(broadcast, softplus), ps.σ), st.epsilon)

sample_gaussian(ps::NamedTuple{(:μ, :σ²), <:Tuple{<:AbstractVector{<:AbstractVector{<:Real}}, <:AbstractVector{<:AbstractVector{<:Real}}}}, st) = 
    sample_gaussian.(ps.μ, map(Base.Fix1(broadcast, sqrt), ps.σ²), st.epsilon)

# for a single set of distribution parameters

sample_gaussian(ps::NamedTuple{(:μ, :Σ), <:Tuple{<:AbstractVector{<:Real}, <:Symmetric}}, st) = 
    sample_gaussian(ps.μ, cholesky(ps.Σ).L, st.epsilon)

sample_gaussian(ps::NamedTuple{(:μ, :L), <:Tuple{<:AbstractVector{<:Real}, <:LowerTriangular}}, st) = 
    sample_gaussian(ps.μ, ps.L, st.epsilon)

sample_gaussian(ps::NamedTuple{(:μ, :σ), <:Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}}, st) = 
    sample_gaussian(ps.μ, softplus.(ps.σ), st.epsilon)

sample_gaussian(ps::NamedTuple{(:μ, :σ²), <:Tuple{<:AbstractVector{<:Real}, <:AbstractVector{<:Real}}}, st) = 
    sample_gaussian(ps.μ, sqrt.(ps.σ²), st.epsilon)

sample_gaussian(μ::AbstractVector, L::LowerTriangular, ϵ::AbstractVector) = μ + L * ϵ
sample_gaussian(μ::AbstractVector, σ::AbstractVector, ϵ::AbstractVector) = μ + σ .* ϵ

function update_state!(rng::Random.AbstractRNG, st::NamedTuple) 
    fmap_with_path(st) do kp, x
        if :epsilon in kp
            x .= randn(rng, size(x))
        end
    end
    return nothing
end