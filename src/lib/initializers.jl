function indicator(n::Int, a::AbstractVector{Int}, ::Type{T}=Float32) where T
    Iₐ = zeros(T, n, length(a))
    for i in eachindex(a)
        Iₐ[a[i], i] = one(T)
    end
    return Iₐ
end


# At least these versions can have doc strings
setup(obj::AbstractObjective, model::AbstractModel; kwargs...) = _setup(obj, model; kwargs...)
setup(obj::AbstractObjective, model::AbstractModel, population::Population; kwargs...) = _setup(obj, model, population; kwargs...)

function _setup(::SSE, model::AbstractModel)
    ps, st = init_theta(model)
    return (theta = ps, ), (theta = st, )
end

function _setup(::LogLikelihood, model::AbstractModel; kwargs...)
    ps_θ, st_θ = init_theta(model)
    ps_σ = init_error(Random.GLOBAL_RNG, model; kwargs...)
    return (theta = ps_θ, error = ps_σ, ), (theta = st_θ, )
end

function _setup(obj::MixedObjective, model::AbstractModel{T,M}, population::Population; params::Parameterization = MeanSqrt(), omega_init = fill(0.1, length(obj.idxs))) where {T<:Real,M}
    omega_init = init_omega(T.(omega_init), params)
    Ω = only(omega_init)
    if size(Ω, 1) !== size(Ω, 2) || size(Ω, 1) !== length(obj.idxs) 
        throw(ArgumentError("The size of the initial omega matrix ($(size(Ω, 1))x$(size(Ω, 2))) is incompatible with the number of random effects designated by the objective function ($(length(obj.idxs)))"))
    end

    ps_θ, st_θ = init_theta(model, params)
    num_typ_p = _estimate_typ_parameter_size(model, population[1:1], (theta = ps_θ, ), (theta = st_θ, ))

    ps_σ = init_error(Random.GLOBAL_RNG, model; params)
    ps_𝜙, st_𝜙 = init_phi(obj, omega_init, length(population), num_typ_p)
    return (theta = ps_θ, error = ps_σ, omega = omega_init, phi = ps_𝜙), (theta = st_θ, phi = st_𝜙)
end

# TODO: Specific version for GPEnsemble that just takes the size of the typical_model.prior
_estimate_typ_parameter_size(m::AbstractModel, population::Population, ps, st) =
    _estimate_typ_parameter_size(first(predict_typ_parameters(m, population, ps, st)))

_estimate_typ_parameter_size(y::AbstractVector{<:Real}) = length(y)
_estimate_typ_parameter_size(y::AbstractMatrix{<:Real}) = size(y, 1)
_estimate_typ_parameter_size(y::AbstractVector{AbstractArray{<:Real}}) = _estimate_typ_parameter_size(first(y))


init_theta(m::AbstractModel) = Lux.setup(Random.GLOBAL_RNG, m.model)
# TODO: Add specific function for GPEnsemble that uses the parameterization
init_theta(m::AbstractModel, ::Parameterization) = Lux.setup(Random.GLOBAL_RNG, m.model)

init_error(rng, model::AbstractModel{T,M,E}; kwargs...) where {T,M,E} = init_error(rng, model.error, T; kwargs...)
init_error(rng, error::AbstractErrorModel, T; params=MeanSqrt()) = error.init_f(rng, error, params, T)
function init_error(rng, ::ImplicitError, T; kwargs...)
    @warn "Initializing error parameters when using ImplicitError. The objective function and the error model might not be compatible." 
    return NamedTuple()
end

function init_omega(omega, ::MeanVar)
    Ω = _to_init_omega(omega)
    return length(Ω) == 1 ? (σ² = [only(Ω)], ) : (Σ = Ω, )
end

function init_omega(omega, ::MeanSqrt)
    Ω = _to_init_omega(omega)
    return length(Ω) == 1 ? (σ = [softplus_inv(sqrt(only(Ω)))], ) : (L = _chol_lower(cholesky(Ω)), )
end

_to_init_omega(ω::Real) = _to_init_omega([ω])
_to_init_omega(ω::AbstractVector{<:Real}) = Symmetric(collect(Diagonal(ω.^2)))
_to_init_omega(Ω::AbstractMatrix{<:Real}) = Symmetric(collect(Ω))

init_phi(obj::Union{FO,FOCE}, Ωinit::NamedTuple, _, num_typ_p) = 
    NamedTuple(), (mask = indicator(num_typ_p, obj.idxs, eltype(only(Ωinit))), )

function init_phi(obj::VariationalELBO{approx,path_deriv}, Ωinit::NamedTuple, n, num_typ_p) where {approx,path_deriv}
    T = eltype(only(Ωinit))
    ps_𝜙 = _init_phi_ps(approx, Ωinit, n)
    st_𝜙 = (
        epsilon = length(obj.idxs) == 1 ? zeros(T, n, 1) : zeros(T, length(obj.idxs), n, 1), 
        mask = indicator(num_typ_p, obj.idxs, T),
    )
    return ps_𝜙, st_𝜙
end

_init_phi_ps(omega::NamedTuple, n::Int) = 
    NamedTuple{(:μ,only(keys(omega)))}(_get_mu_vars(only(omega), n))

_get_mu_vars(Ω::AbstractVector, n::Int) = (
    zeros(eltype(Ω), n), 
    fill(only(Ω), n)
)

_get_mu_vars(Ω::AbstractMatrix, n::Int) = (
    copy.(fill(zeros(eltype(Ω), size(Ω, 1)), n)),
    copy.(fill(Ω, n))
)

_init_phi_ps(::Type{FullRank}, omega::NamedTuple, n::Int) = _init_phi_ps(omega, n)
_init_phi_ps(::Type{MeanField}, omega::NamedTuple{<:Any,<:Tuple{<:AbstractVector}}, n::Int) = _init_phi_ps(omega, n)

function _init_phi_ps(::Type{MeanField}, omega::NamedTuple{<:Any,<:Tuple{<:Symmetric}}, n::Int) 
    phi_ = _init_phi_ps(omega, n)
    return NamedTuple{(:μ,:σ²)}((phi_.μ, diag.(phi_.Σ))) # Assumes phi[2] is always the variance
end

function _init_phi_ps(::Type{MeanField}, omega::NamedTuple{<:Any,<:Tuple{<:LowerTriangular}}, n::Int) 
    phi_ = _init_phi_ps(omega, n)
    σs = map(L -> sqrt.(diag(Symmetric(L * L'))), phi_.L)
    return NamedTuple{(:μ,:σ)}((phi_.μ, σs)) # Assumes phi[2] is always the variance
end
