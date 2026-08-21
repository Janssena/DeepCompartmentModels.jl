# TODO: 
function m_step(obj::VariationalELBO, rng::Random.AbstractRNG, dcm::DeepCompartmentModel{P,M}, population::Population, ps, st; return_diagnostics::Bool=false, verbose::Bool=true, kwargs...) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    verbose && @info "Optimising residual error parameters"
    ps, residual_diagnostics = optimise_residual_error(
        obj, rng, dcm, population, ps, st;
        return_diagnostics=true, verbose, kwargs...)
    verbose && @info "Optimising omega based on Variational posteriors"
    omega_opt = optimise_omega(ps)
    ps = Accessors.@set ps.omega = omega_opt

    diagnostics = (
        residual = residual_diagnostics,
        omega_updated = true,
        omega_min_eigenvalue = minimum(eigvals(Matrix(omega_opt))),
    )
    return return_diagnostics ? (ps, diagnostics) : ps
end

function optimise_residual_error(obj::Union{<:LogLikelihood,<:MixedObjective}, rng, dcm, data, ps, st; opt=Optimisers.Adam(1e-2), epochs::Int=100, num_samples::Int=100, verbose::Bool=true, return_diagnostics::Bool=false, kwargs...)
    epochs > 0 || throw(ArgumentError(
        "The residual-error M-step cannot be skipped: epochs must be positive."))
    opt_state = Optimisers.setup(opt, ps.error)
    history = Float64[]
    error_start = deepcopy(ps.error)
    predictions = _residual_error_predictions(rng, dcm, data, ps, st, num_samples)
    for epoch in 1:epochs
        loss, grad = residual_error_value_and_gradient(
            rng, dcm, data, ps, st; num_samples, kwargs..., predictions)
        isfinite(loss) || throw(ErrorException(
            "Residual-error M-step produced a non-finite loss at epoch $epoch."))
        push!(history, Float64(loss))
        if verbose
            println("Epoch $epoch, NLL = $(loss)")
        end
        opt_state, error = Optimisers.update(opt_state, ps.error, grad.error)
        ps = Accessors.@set ps.error = error
    end

    diagnostics = (
        executed = true,
        epochs = epochs,
        num_samples = num_samples,
        reused_prediction_draws = true,
        objective_history = history,
        error_changed = !isequal(error_start, ps.error),
    )
    return return_diagnostics ? (ps, diagnostics) : ps
end

function optimise_omega(ps::NamedTuple{(:theta,:error,:omega,:phi)})
    μμᵀ = map(ps.phi.μ) do μ
        μ * μ'
    end
    omega = Symmetric(mean(μμᵀ + _get_cov_matrix(ps.phi)))
    all(isfinite, omega) || throw(ErrorException("The analytic omega M-step produced non-finite values."))
    isposdef(omega) || throw(DomainError(omega,
        "The analytic omega M-step produced a covariance matrix that is not positive definite."))
    return omega
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
