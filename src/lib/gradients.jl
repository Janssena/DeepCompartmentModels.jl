import ForwardDiff

################################################################################
##########                     Automatic differentiation              ##########
################################################################################

"""
    default_ad()

The default automatic differentiation backend: Enzyme in reverse mode with runtime activity,
i.e. `AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))`.
"""
default_ad() = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))

# Calculate the value and gradient of `f(ps, args...)` with respect to `ps`. All of `args` are constant.
# `f` should be a function defined at the module level (not a closure over data) to keep Enzyme type stable.
function _ad_value_and_gradient(ad::AutoEnzyme, f, ps, args...)
    mode = _enzyme_reverse_mode(ad.mode)
    dps = Enzyme.make_zero(ps)
    _, loss = Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(ps, dps), map(Enzyme.Const, args)...)
    return loss, dps
end

_enzyme_reverse_mode(::Nothing) = _enzyme_reverse_mode(default_ad().mode)
_enzyme_reverse_mode(mode::Enzyme.EnzymeCore.ReverseMode) = Enzyme.EnzymeCore.WithPrimal(mode)
_enzyme_reverse_mode(mode) = throw(ArgumentError(
    "Enzyme can only be used in reverse mode in DeepCompartmentModels (got `$mode`): forward mode does not work through differential " *
    "equation solves yet. Use `AutoForwardDiff()` for forward mode differentiation."
))

function _ad_value_and_gradient(::AutoForwardDiff, f, ps, args...)
    flat, restructure = Optimisers.destructure(ps)
    result = ForwardDiff.DiffResults.GradientResult(flat)
    result = ForwardDiff.gradient!(result, v -> f(restructure(v), args...), flat)
    return ForwardDiff.DiffResults.value(result), restructure(ForwardDiff.DiffResults.gradient(result))
end

# Implemented by the extension DCMZygoteExt, which is loaded when Zygote is.
_ad_value_and_gradient(::AutoZygote, args...) = throw(ArgumentError(
    "`AutoZygote()` needs Zygote to be loaded: run `import Zygote` first."
))

_ad_value_and_gradient(ad::ADTypes.AbstractADType, args...) = throw(ArgumentError(
    "Unsupported automatic differentiation backend `$ad`. Use `AutoEnzyme()` (reverse mode, default), `AutoForwardDiff()` or `AutoZygote()` (needs `import Zygote`)."
))

# Loss functions that are differentiated. These are defined at the module level and get all data as
# (constant) arguments, which is what Enzyme needs to compile efficient code.
_objective_loss(ps, objective, dcm, data, st) = objective(dcm, data, ps, st)

function _weighted_objective_loss(ps, objective, dcm, data, st, weight)
    loss = objective(dcm, data, ps, st)
    return loss * oftype(loss, weight) # keep the element type of the loss (Float32)
end

# `sub` holds the parameters that are differentiated for the mixed effect objectives: (theta, phi, error).
_objective_loss_mixed(sub, objective, dcm, data, ps, st) = objective(dcm, data, merge(ps, sub), st)
_negative_logjoint_loss(sub, dcm, data, ps, st) = -logjoint(dcm, data, merge(ps, sub), st)

"""
    gradient(objective, model, population, ps, st; ad, parallel, batchsize, nbatch, rng)

Compute the gradient of `objective` with respect to `ps`. Same as `last(value_and_gradient(...))`,
see [`value_and_gradient`](@ref) for the arguments.
"""
gradient(objective::AbstractObjective, dcm::DeepCompartmentModel, population::Population, ps, st; kwargs...) =
    last(value_and_gradient(objective, dcm, population, ps, st; kwargs...))

"""
    value_and_gradient(objective, model, population, ps, st; ad, parallel, batchsize, nbatch, rng)

Compute the value of `objective` and its gradient with respect to `ps`, in a single pass. Returns
`(loss, ∇)`. The gradient has the same structure as `ps`; for `VariationalELBO` the entries of `ps`
that are not differentiated (`omega`) are `nothing`.

## Arguments
- `objective`: Objective function.
- `model`: A `DeepCompartmentModel`.
- `population`: A `Population`.
- `ps`: Model parameters.
- `st`: Model state.

## Keyword Arguments
- `ad=AutoEnzyme(; mode = set_runtime_activity(Reverse))`: Automatic differentiation backend, an
  [ADTypes.jl](https://github.com/SciML/ADTypes.jl) object. Supported are `AutoEnzyme()` (reverse mode)
  and `AutoForwardDiff()`. Forward mode with Enzyme is not supported, since Enzyme cannot yet differentiate
  through the differential equation solvers in forward mode. `AutoZygote()` is available after `import Zygote`,
  but is only advised for in-place `UniversalDiffEq`s with large networks, which Enzyme cannot yet
  differentiate with an adjoint `sensealg`. Large networks are better written as out-of-place UDEs and
  differentiated with Enzyme.
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
        ad::ADTypes.AbstractADType = default_ad(), parallel = false, batchsize::Union{Nothing,Int} = nothing,
        nbatch::Union{Nothing,Int} = nothing, rng::Random.AbstractRNG = Random.default_rng()
    )
    if parallel == false
        return _value_and_gradient(objective, dcm, population, ps, st, ad)
    elseif parallel == true || parallel == :individual
        results = qmap(eachindex(population)) do i
            _value_and_gradient(objective, dcm, population, i:i, ps, st, ad)
        end
        return _combine_value_and_gradient(results)
    elseif parallel == :batch
        batches = create_batches(rng, population; batchsize, nbatch)
        results = qmap(batches) do batch
            _value_and_gradient(objective, dcm, population, batch, ps, st, ad)
        end
        return _combine_value_and_gradient(results)
    else
        throw(ErrorException("Parallel = $(parallel) not recognized. Please use one of [false, :individual, :batch]"))
    end
end

# `results` are the `(loss, ∇)` of the individuals or batches. The (weighted) losses and gradients of
# the batches add up to the loss and gradient of the population.
_combine_value_and_gradient(results) = (sum(first, results), _sum_gradients(last.(results)))

_value_and_gradient(objective::FixedObjective, dcm::DeepCompartmentModel, data, ps, st, ad) =
    _ad_value_and_gradient(ad, _objective_loss, ps, objective, dcm, data, st)

function _value_and_gradient(objective::FixedObjective, dcm::DeepCompartmentModel, population::Population, batch, ps, st, ad)
    ps_batch, st_batch = take_batch(ps, st, batch; exclude = dcm.error isa ErrorModelSet ? (:error) : nothing)
    weight = _batch_weight(objective, length(batch), length(population))
    return _ad_value_and_gradient(ad, _weighted_objective_loss, ps_batch, objective, dcm, population[batch], st_batch, weight)
end

# The parameters that are differentiated for the mixed effect objectives. `omega` is not.
_differentiated_parameters(ps) = (theta = ps.theta, phi = ps.phi, error = ps.error)

function _value_and_gradient(objective::VariationalELBO{MF,PD,N}, dcm, data, ps, st, ad) where {MF,PD<:False,N<:False}
    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    loss, grad = _ad_value_and_gradient(ad, _objective_loss_mixed, _differentiated_parameters(ps), objective, dcm, data, ps, st)

    Accessors.@reset ∇.theta = grad.theta
    Accessors.@reset ∇.phi = grad.phi
    Accessors.@reset ∇.error = grad.error
        
    return loss, ∇
end

function _value_and_gradient(::VariationalELBO{MF,PD,N}, dcm, data, ps, st, ad) where {MF,PD<:True,N<:False}
    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    negative_logjoint, grad = _ad_value_and_gradient(ad, _negative_logjoint_loss, _differentiated_parameters(ps), dcm, data, ps, st)

    Accessors.@reset ∇.theta = grad.theta
    Accessors.@reset ∇.phi = grad.phi
    Accessors.@reset ∇.error = grad.error

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

function _value_and_gradient(obj::VariationalELBO, dcm::DeepCompartmentModel{P,M}, data, batch::AbstractArray{<:Int}, ps, st, ad) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    ps_batch, st_batch = take_batch(ps, st, batch; exclude = dcm.error isa ErrorModelSet ? (:error) : nothing)
    loss, grad_batch = _value_and_gradient(obj, dcm, data[batch], ps_batch, st_batch, ad)
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

# Negative log of the importance-weighted likelihood, where `predictions` holds the predictions for
# `length(predictions)` samples of the random effects.
function _residual_error_loss(error_ps, dcm, predictions, y)
    lls = map(predictions) do ŷ
        dist = make_dist(dcm.error, ŷ, error_ps)
        logpdf.(dist, y)
    end
    return -sum(logsumexp.(lls) .- log(length(predictions)))
end

"""
    residual_error_value_and_gradient(rng, model, data, ps, st; mode = :forward, num_samples = 100)

Value and gradient of the (importance sampled) negative log likelihood with respect to the parameters of the
residual error model `ps.error`. `mode = :forward` uses ForwardDiff, `mode = :reverse` uses Enzyme.
"""
function residual_error_value_and_gradient(rng::Random.AbstractRNG, dcm::DeepCompartmentModel{P,M}, data, ps, st; mode::Symbol = :forward, num_samples::Int = 100) where {P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer}
    ad = if mode == :forward
        AutoForwardDiff()
    elseif mode == :reverse
        default_ad()
    else
        throw(ArgumentError("`mode` = $mode not recognized. Please use one of [:forward, :reverse]"))
    end

    ∇ = NamedTuple{keys(ps)}(fill(nothing, length(keys(ps))))
    st_local = deepcopy(st)
    predictions = map(1:num_samples) do _
        update_epsilon!(rng, st_local)
        predict(dcm, data, ps, st_local)
    end

    loss, grad = _ad_value_and_gradient(ad, _residual_error_loss, ps.error, dcm, predictions, get_y(data))

    return loss, Accessors.@set ∇.error = grad
end
