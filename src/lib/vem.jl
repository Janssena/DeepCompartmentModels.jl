# TODO: 
function m_step(obj::VariationalELBO, rng::Random.AbstractRNG, dcm::DeepCompartmentModel{P,M}, population::Population, ps, st; kwargs...) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    @info "Optimising residual error parameters"
    ps = optimise_residual_error(obj, rng, dcm, population, ps, st; kwargs...)
    @info "Optimising omega based on Variational posteriors"
    omega_opt = optimise_omega(ps)
    
    return Accessors.@set ps.omega = omega_opt
end

function optimise_residual_error(obj::Union{<:LogLikelihood,<:MixedObjective}, rng, dcm, data, ps, st; opt=Optimisers.Adam(1e-2), epochs=100, verbose::Bool = true, kwargs...)
    opt_state = Optimisers.setup(opt, ps)
    for epoch in 1:epochs
        loss, grad = residual_error_value_and_gradient(rng, dcm, data, ps, st; kwargs...)
        if verbose
            println("Epoch $epoch, NLL = $(loss)")
        end
        opt_state, ps = Optimisers.update(opt_state, ps, grad)
    end

    return ps
end

function optimise_omega(ps::NamedTuple{(:theta,:error,:omega,:phi)})
    μμᵀ = map(ps.phi.μ) do μ
        μ * μ'
    end
    return mean(μμᵀ + _get_cov_matrix(ps.phi))
end

# These all assume that the variance parameters are vectors of parameters
_get_cov_matrix(ps::NamedTuple{(:μ,:Σ)}) = ps.Σ
_get_cov_matrix(ps::NamedTuple{(:μ,:L)}) = map(ps.L) do L
    Symmetric(L * L')
end
_get_cov_matrix(ps::NamedTuple{(:μ,:σ)}) = map(ps.σ) do σ
    collect(Diagonal(softplus.(σ).^2))
end
_get_cov_matrix(ps::NamedTuple{(:μ,:σ²)}) = map(ps.σ²) do σ²
    collect(Diagonal(σ²))
end