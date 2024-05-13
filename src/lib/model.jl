import DifferentialEquations.SciMLBase: AbstractDEProblem

abstract type AbstractModel{O,M,P,S} end
abstract type AbstractDEModel{O,D,M,P,S} <: AbstractModel{O,M,P,S} end

"""
    predict(...)
Alias for the forward function.
"""
predict(model::AbstractModel, args...; kwargs...) = forward(model, args...; kwargs...)

objective(model::M, container::Union{AbstractIndividual, Population}; kwargs...) where {M<:AbstractModel} = objective(model, container, model.p; kwargs...)
forward(model::M, container::Union{AbstractIndividual, Population}, args...; kwargs...) where {M<:AbstractModel} = forward(model, container, model.p, args...; kwargs...)

"""
    update!(model::AbstractModel, p::Tuple)

Performs an in-place update of model parameters when P <: NamedTuple.
"""
function update!(model::AbstractModel{O,M,P,S}, p::P) where {O,M,P<:NamedTuple,S}
    Lux.fmap((x,y) -> x .= y, model.p, p)
    return nothing
end

"""
    fit!(model::AbstractModel, population::Population, opt, epochs; callback)

Fits the model in place to the data from the population. Updated model 
parameters are stored in the model. Can be passed a callback function that can 
be used to monitor training. The callback is called with the current epoch and 
loss after gradient calculation and before updating model parameters.
"""
function fit!(model::AbstractModel{O,M,P,S}, population::Population, opt, epochs::Int; callback=(e,l) -> nothing) where {O<:FixedObjective,M,P,S}
    opt_state = Optimisers.setup(opt, model.p)
    for epoch in 1:epochs
        loss, back = Zygote.pullback(p -> objective(model, population, p), model.p)
        grad = first(back(1))
        callback(epoch, loss)
        opt_state, new_p = Optimisers.update(opt_state, model.p, grad)
        update!(model, new_p)
    end
    return nothing
end

# TODO: allow for multiple optimizers (one for p and one for phi)
# TODO: allow passing previous phi
function fit!(model::AbstractModel{O,M,P,S}, population::Population, opt, epochs::Int; callback=(e,l) -> nothing) where {O<:VariationalELBO,M,P,S}
    phi = model.objective.init_phi(model, population)

    if typeof(model.objective.approx) <: SampleAverage
        init_samples!(model.objective, population)
    end # TODO: make this optional

    opt_state = Optimisers.setup(opt, model.p)
    opt_state_phi = Optimisers.setup(opt, phi)
    for epoch in 1:epochs
        loss, back = Zygote.pullback((p, phi) -> objective(model, population, p, phi), model.p, phi)
        grad_p, grad_phi = back(1)
        callback(epoch, loss)
        opt_state, new_p = Optimisers.update(opt_state, model.p, grad_p)
        opt_state_phi, phi = Optimisers.update(opt_state_phi, phi, grad_phi) # TODO: Natural Gradient descent.
        update!(model, new_p)
    end
    return phi
end

# forward_ode → solve ode
forward_ode(model::AbstractDEModel, args...; kwargs...) = forward_ode(model.problem, args...; dv_idx=model.dv_compartment, kwargs...)
forward_ode(prob::AbstractDEProblem, container::Union{Population, AbstractIndividual}, zᵢ::AbstractVecOrMat; get_dv=Val(false), kwargs...) = forward_ode(prob, container, zᵢ, get_dv; kwargs...)
forward_ode(prob::AbstractDEProblem, population::Population, z::AbstractMatrix, get_dv; kwargs...) = forward_ode.((prob,), population, eachcol(z), (get_dv,); kwargs...)
forward_ode(prob::AbstractDEProblem, population::Population, z::AbstractVector{<:AbstractMatrix}, get_dv; kwargs...) = forward_ode.((prob,), population, z, (get_dv,); kwargs...)
# Handle the case where we want the dv directly. TODO: Make type safe
forward_ode(prob::AbstractDEProblem, individual::AbstractIndividual, zᵢ::AbstractVecOrMat, ::Val{true}; dv_idx, kwargs...) = forward_ode(prob, individual, zᵢ, Val(false); kwargs...)[dv_idx, :]

function forward_ode(problem::AbstractDEProblem, individual::AbstractIndividual, zᵢ::AbstractVecOrMat, ::Val{false}; dv_idx=1, sensealg=nothing, interpolate::Bool=false, saveat=get_t(individual))
    u0 = isempty(individual.initial) ? problem.u0 : individual.initial
    saveat_ = interpolate ? empty(saveat) : saveat
    prob = remake(problem, u0 = u0, tspan = (problem.tspan[1], maximum(saveat)), p = zᵢ)
    interpolate && (individual.callback.save_positions .= 1)
    sol = solve(prob, Tsit5(),
        saveat = saveat_, callback=individual.callback, 
        tstops=individual.callback.condition.times, sensealg=sensealg
    )
    interpolate && (individual.callback.save_positions .= 0)
    return sol
end