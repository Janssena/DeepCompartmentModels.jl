import ForwardDiff

function gradient(objective::AbstractObjective, dcm::DeepCompartmentModel, population::Population, ps, st; parallel = false, batchsize::Int=16)
    if parallel == false
        return _gradient(objective, dcm, population, ps, st)
    elseif parallel == true || parallel == :individual
        grads = qmap(eachindex(population)) do i
            _gradient(objective, dcm, population, i:i, ps, st)
        end
        return fmap(_sum_grads, grads...)
    elseif parallel == :batch
        batches = create_batches(population, batchsize)
        grads = qmap(batches) do batch
            _gradient(objective, dcm, population, batch, ps, st)
        end
        return fmap(_sum_grads, grads...)
    else
        throw(ErrorException("Parallel = $(parallel) not recognized. Please use one of [false, :individual, :batch]"))
    end
end

function _gradient(objective::FixedObjective, dcm, data, ps, st)
    grad = Zygote.gradient(ps) do p
        objective(dcm, data, p, st)
    end
    return first(grad)
end

function _gradient(objective::FixedObjective, dcm, data, idx, ps, st)
    grad = Zygote.gradient(ps) do p
        objective(dcm, data[idx], p, st)
    end
    return first(grad)
end

function _gradient(objective::VariationalELBO{MF,PD,N}, dcm, data, ps, st) where {MF,PD<:False,N<:False}
    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    grad = Zygote.gradient(ps.theta, ps.phi, ps.error) do θ, 𝜙, σ
        ps_local = Accessors.@set ps.theta = θ
        ps_local = Accessors.@set ps_local.phi = 𝜙
        ps_local = Accessors.@set ps_local.error = σ
        objective(dcm, data, ps_local, st)
    end

    Accessors.@reset ∇.theta = grad[1]
    Accessors.@reset ∇.phi = grad[2]
    Accessors.@reset ∇.error = grad[3]
        
    return ∇
end

function _gradient(::VariationalELBO{MF,PD,N}, dcm, data, ps, st) where {MF,PD<:True,N<:False}
    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    grad = Zygote.gradient(ps.theta, ps.phi, ps.error) do θ, 𝜙, σ
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
    
    return ∇_full
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

function _gradient(obj::VariationalELBO, dcm::DeepCompartmentModel{P,M}, data, batch::AbstractArray{<:Int}, ps, st) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    ps_batch, st_batch = take_batch(ps, st, batch)
    grad_batch = _gradient(obj, dcm, data[batch], ps_batch, st_batch)
    # TODO: correct phi indices (potentially more to do for other models)
    grad_phi = fmap(ps.phi) do _
        nothing
    end
    Accessors.@reset grad_phi.μ[batch] = grad_batch.phi.μ
    variance_key = only(filter(!=(:μ), keys(ps.phi)))
    Accessors.@reset grad_phi[variance_key][batch] = grad_batch.phi[variance_key]
    
    return Accessors.@set grad_batch.phi = grad_phi
end

function create_batches(n::Int, M::Int)
    batches = [collect(i:M:n) for i in 1:M]
    idxs = Random.shuffle(1:n)
    return map(Base.Fix1(getindex, idxs), batches) # randomize indexes in each batch
end

create_batches(population::Population, M::Int) = 
    create_batches(length(population), min(M, length(population)))

take_batch(ps::NamedTuple, st::NamedTuple, batch) = take_batch(ps, batch), take_batch(st, batch)

take_batch(x, ::Any) = x
take_batch(x::AbstractVector{<:AbstractArray{<:Real}}, i) = x[i]
take_batch(x::AbstractVector{<:Cholesky}, i) = x[i]
function take_batch(x::NamedTuple, i) 
    keys_ = keys(x)
    values_ = map(keys_) do key # Almost a fmap, but does not recurse into indexes
        take_batch(x[key], i)
    end
    return NamedTuple{keys_}(values_)
end

_sum_grads(::Vararg{Nothing}) = nothing
_sum_grads(xs::Vararg) = +(filter(!isnothing, xs)...)

function residual_error_value_and_gradient(rng::Random.AbstractRNG, dcm::DeepCompartmentModel{P,M}, data, ps, st; mode::Symbol = :forward, num_samples::Int = 100) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    st_local = deepcopy(st)
    predictions = map(1:num_samples) do _
        update_state!(rng, st_local)
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