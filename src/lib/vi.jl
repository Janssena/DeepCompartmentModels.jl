function loglikelihood(prob, individual, z, σ)
    ŷ = predict_adjoint(prob, individual, z)
    Σ = fill(σ, length(ŷ))
    return logpdf(MultivariateNormal(ŷ, Σ), individual.y)
end
logprior(η::AbstractVector, Ω::AbstractMatrix) = logpdf(MultivariateNormal(zero.(η), Ω), η)
logjoint(model, individual, z, η, Ω, σ) = loglikelihood(model, individual, z, σ) + logprior(η, Ω)

_chol_lower(a::Cholesky) = a.uplo === 'L' ? a.L : a.U'
function partial_advi(model, population, p, phi::NamedTuple; num_samples = 1, with_entropy = false)
    m = 2 # num_rand_effects
    # ζ, _ = Lux.apply(model.ann, population.x, p.weights, p.st) # Lux
    ζ, _ = model.ann(Zygote.ignore_derivatives(population.x), p.weights, Zygote.ignore_derivatives(p.st)) # Lux
    σ_ω = softplus.([p.error.sigma; p.omega.var])
    σ = σ_ω[1]
    ω = σ_ω[2:3]
    C = inverse(Bijectors.VecCorrBijector())(p.omega.corr)
    Ω = Symmetric(ω .* C .* ω')

    ELBO = zero(eltype(ζ))
    for _ in 1:num_samples
        for i in eachindex(population)
            # Calculates p(y | η; σ) + p(η; Ω) - q_𝜙(η)
            Lᵢ = LowerTriangular(_chol_lower(inverse(Bijectors.VecCholeskyBijector(:L))(phi.corr[:, i])) .* softplus.(phi.sigma[:, i]))
            ηᵢ = phi.mean[:, i] + Lᵢ * randn(Float32, m)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(model.problem, population[i], zᵢ, ηᵢ, Ω, σ)
            qᵢ = @ignore_derivatives MultivariateNormal(phi.mean[:, i], Lᵢ * Lᵢ')
            ELBO += (logπ - logpdf(qᵢ, ηᵢ)) / num_samples
        end
    end
    
    return ELBO
end

function predict_adjoint(prob, individual::AbstractIndividual, z::AbstractVector{T}; measurement_idx = 1) where T
    prob_ = remake(prob, tspan = (prob.tspan[1], maximum(individual.t)), p = [z; zero(T)])
    return solve(
        prob_, Tsit5(), dtmin=1e-10, saveat=individual.t, 
        tstops=individual.callback.condition.times, callback=individual.callback, 
        sensealg=ForwardDiffSensitivity(;convert_tspan=true)
    )[measurement_idx, :]
end


function partial_advi_(ann, prob, population, 𝜙::NamedTuple, st; num_samples = 1, with_entropy = false)
    m = 2 # num_rand_effects
    ζ, _ = Lux.apply(ann, Zygote.ignore_derivatives(population.x), 𝜙.weights, st) # Lux
    σ_ω = softplus.(𝜙.theta[1:3])
    σ = σ_ω[1]
    ω = σ_ω[2:3]
    C = inverse(Bijectors.VecCorrBijector())(𝜙.theta[4:end])
    Ω = Symmetric(ω .* C .* ω')

    ELBO = zero(eltype(ζ))
    for _ in 1:num_samples
        for i in eachindex(population)
            # Calculates p(y | η; σ) + p(η; Ω) - q_𝜙(η)
            Lᵢ = LowerTriangular(_chol_lower(inverse(Bijectors.VecCholeskyBijector(:L))(𝜙.phi[2m+1:end, i])) .* softplus.(𝜙.phi[m+1:2m, i]))
            ηᵢ = 𝜙.phi[1:m, i] + Lᵢ * randn(Float32, m)
            zᵢ = ζ[:, i] .* exp.([1 0; 0 1; 0 0; 0 0] * ηᵢ)
            logπ = logjoint(prob, population[i], zᵢ, ηᵢ, Ω, σ)
            qᵢ = @ignore_derivatives MultivariateNormal(𝜙.phi[1:2, i], Lᵢ * Lᵢ')
            ELBO += (logπ - logpdf(qᵢ, ηᵢ)) / num_samples
        end
    end
    
    return ELBO
end