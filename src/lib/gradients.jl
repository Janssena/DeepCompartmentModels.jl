import ForwardDiff

"""
    gradient(objective, model, population, ps, st; parallel, batchsize, nbatch, rng)

Compute the gradient of `objective` with respect to `ps` using Zygote. Same as
`last(value_and_gradient(...))`, see [`value_and_gradient`](@ref) for the arguments.
"""
gradient(objective::AbstractObjective, dcm::DeepCompartmentModel, population::Population, ps, st; kwargs...) =
    last(value_and_gradient(objective, dcm, population, ps, st; kwargs...))

"""
    value_and_gradient(objective, model, population, ps, st; parallel, batchsize, nbatch, rng)

Compute the value of `objective` and its gradient with respect to `ps` using Zygote, in a single
forward pass. Returns `(loss, ∇)`.

## Arguments
- `objective`: Objective function.
- `model`: A `DeepCompartmentModel`.
- `population`: A `Population`.
- `ps`: Model parameters.
- `st`: Model state.

## Keyword Arguments
- `parallel=false`: Parallelism strategy. Options: `false` (sequential), `:individual`
  (one thread per individual), `:batch` (one thread per mini-batch).
- `batchsize=nothing`, `nbatch=nothing`: Only used when `parallel=:batch`. Set at most one of them:
  `batchsize` is the maximum number of individuals per batch, `nbatch` is the number of batches. The
  individuals are divided over the batches as evenly as possible. If neither is set, `batchsize = 16`.
  See [`create_batches`](@ref).
- `rng=Random.default_rng()`: Random number generator used to assign individuals to batches when
  `parallel=:batch`.
"""
function value_and_gradient(
        objective::AbstractObjective, dcm::DeepCompartmentModel, population::Population, ps, st;
        parallel = false, batchsize::Union{Nothing,Int} = nothing, nbatch::Union{Nothing,Int} = nothing,
        rng::Random.AbstractRNG = Random.default_rng()
    )
    if parallel == false
        return _value_and_gradient(objective, dcm, population, ps, st)
    elseif parallel == true || parallel == :individual
        results = qmap(eachindex(population)) do i
            _value_and_gradient(objective, dcm, population, i:i, ps, st)
        end
        return _combine_value_and_gradient(results)
    elseif parallel == :batch
        batches = create_batches(rng, population; batchsize, nbatch)
        results = qmap(batches) do batch
            _value_and_gradient(objective, dcm, population, batch, ps, st)
        end
        return _combine_value_and_gradient(results)
    else
        throw(ErrorException("Parallel = $(parallel) not recognized. Please use one of [false, :individual, :batch]"))
    end
end

# `results` are the `(loss, ∇)` of the individuals or batches. The (weighted) losses and gradients of
# the batches add up to the loss and gradient of the population.
_combine_value_and_gradient(results) = (sum(first, results), _sum_gradients(last.(results)))

function _value_and_gradient(objective::FixedObjective, dcm::DeepCompartmentModel, data, ps, st)
    loss, grad = Zygote.withgradient(ps) do p
        objective(dcm, data, p, st)
    end
    return loss, first(grad)
end

function _value_and_gradient(objective::FixedObjective, dcm::DeepCompartmentModel, population::Population, batch, ps, st)
    ps_batch, st_batch = take_batch(ps, st, batch; exclude = dcm.error isa ErrorModelSet ? (:error) : nothing)
    weight = _batch_weight(objective, length(batch), length(population))
    loss, grad = Zygote.withgradient(ps_batch) do p
        loss = objective(dcm, population[batch], p, st_batch)
        loss * oftype(loss, weight) # keep the element type of the loss (Float32)
    end
    return loss, first(grad)
end

function _value_and_gradient(objective::VariationalELBO{MF,PD,N}, dcm, data, ps, st) where {MF,PD<:False,N<:False}
    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    loss, grad = Zygote.withgradient(ps.theta, ps.phi, ps.error) do θ, 𝜙, σ
        ps_local = Accessors.@set ps.theta = θ
        ps_local = Accessors.@set ps_local.phi = 𝜙
        ps_local = Accessors.@set ps_local.error = σ
        objective(dcm, data, ps_local, st)
    end

    Accessors.@reset ∇.theta = grad[1]
    Accessors.@reset ∇.phi = grad[2]
    Accessors.@reset ∇.error = grad[3]
        
    return loss, ∇
end

function _value_and_gradient(::VariationalELBO{MF,PD,N}, dcm, data, ps, st) where {MF,PD<:True,N<:False}
    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    negative_logjoint, grad = Zygote.withgradient(ps.theta, ps.phi, ps.error) do θ, 𝜙, σ
        ps_local = Accessors.@set ps.theta = θ
        ps_local = Accessors.@set ps_local.phi = 𝜙
        ps_local = Accessors.@set ps_local.error = σ
        -logjoint(dcm, data, ps_local, st)
    end

    Accessors.@reset ∇.theta = grad[1]
    Accessors.@reset ∇.phi = grad[2]
    Accessors.@reset ∇.error = grad[3]

    # The gradient of m and S given z wrt logq(z)
    ∇_full = add_path_deriv_dlogq(∇, ps, st)

    # The objective is -elbo = -logjoint + logq. logq does not need the ODE solves.
    loss = negative_logjoint + logq(dcm, ps, st)
    
    return loss, ∇_full
end

function add_path_deriv_dlogq(∇, ps, st)
    qs = getq(ps.phi)
    η = sample_gaussian(ps.phi, st.phi)
    dη = gradlogpdf.(qs, η)

    Accessors.@reset ∇.phi.μ = ∇.phi.μ + dη
    if :L in keys(∇.phi)
        dL = LowerTriangular.(dη .* transpose.(st.phi.epsilon))
        Accessors.@reset ∇.phi.L = ∇.phi.L + dL # NOTE: checked and is correct
    elseif :σ in keys(∇.phi)
        dσ = map(.*, dη, st.phi.epsilon)
        dσ̃ = map(.*, dσ, map(Base.Fix1(broadcast, logistic), ps.phi.σ)) # add ∂σ/∂σ̃ = logistic(σ̃), where σ = softplus(σ̃)
        Accessors.@reset ∇.phi.σ = ∇.phi.σ + dσ̃ # NOTE: checked and is correct
    else
        throw(ErrorException("Not implemented: it could be that you are using natural = true and path_deriv = true. Either of these options should be chosen."))
    end

    return ∇
end

function _value_and_gradient(obj::VariationalELBO, dcm::DeepCompartmentModel{P,M}, data, batch::AbstractArray{<:Int}, ps, st) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    ps_batch, st_batch = take_batch(ps, st, batch; exclude = dcm.error isa ErrorModelSet ? (:error) : nothing)
    loss, grad_batch = _value_and_gradient(obj, dcm, data[batch], ps_batch, st_batch)
    # TODO: correct phi indices (potentially more to do for other models)
    grad_phi = fmap(ps.phi) do _
        nothing
    end
    if :z in keys(grad_batch.phi)
        grad_phi = NamedTuple{(:z,)}((grad_phi.μ, ))
        Accessors.@reset grad_phi.z[batch] = grad_batch.phi.z
    else
        Accessors.@reset grad_phi.μ[batch] = grad_batch.phi.μ
        variance_key = only(filter(!=(:μ), keys(ps.phi)))
        Accessors.@reset grad_phi[variance_key][batch] = grad_batch.phi[variance_key]
    end
    
    return loss, Accessors.@set grad_batch.phi = grad_phi
end

function residual_error_value_and_gradient(rng::Random.AbstractRNG, dcm::DeepCompartmentModel{P,M}, data, ps, st; mode::Symbol = :forward, num_samples::Int = 100) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    st_local = deepcopy(st)
    predictions = map(1:num_samples) do _
        update_epsilon!(rng, st_local)
        predict(dcm, data, ps, st_local)
    end
    
    if mode == :forward
        error = ComponentVector(ps.error)
        ad = ForwardDiff
    else   
        error = ps.error
        ad = Zygote
    end

    res = eltype(error)[]
    _grad = ad.gradient(error) do p
        lls = map(predictions) do ŷ
            dist = make_dist(dcm.error, ŷ, p)
            logpdf.(dist, get_y(data))
        end
        loss = -sum(logsumexp.(lls) .- log(num_samples))
        push!(res, loss isa ForwardDiff.Dual ? loss.value : loss)
        return loss
    end

    grad = mode == :forward ? NamedTuple(_grad) : first(_grad)

    return only(res), Accessors.@set ∇.error = grad
end