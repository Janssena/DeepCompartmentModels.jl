abstract type AbstractModel{O,M,P} end
abstract type AbstractDEModel{O,D,M,P} <: AbstractModel{O,M,P} end

"""
    predict(...)
Alias for the forward function.
"""
predict(model::AbstractModel, args...; kwargs...) = forward(model, args...; kwargs...)

objective(model::M, container::Union{AbstractIndividual, Population}; kwargs...) where {M<:AbstractModel} = objective(model, container, model.p; kwargs...)
forward(model::M, container::Union{AbstractIndividual, Population}; kwargs...) where {M<:AbstractModel} = forward(model, container, model.p; kwargs...)

"""
    update!(model::AbstractModel, p::Tuple)

Performs an in-place update of model parameters when P <: NamedTuple.
"""
function update!(model::AbstractModel{O,M,P}, p::P) where {O,M,P<:NamedTuple}
    Lux.fmap((x,y) -> x .= y, model.p, p)
    return nothing
end

# forward_ode → solve ode
forward_ode(model::AbstractDEModel, population::Population, z::AbstractMatrix; kwargs...) = forward_ode.((model,), population, eachcol(z); kwargs...)
forward_ode(model::AbstractDEModel, population::Population, z::AbstractVector{<:AbstractMatrix}; kwargs...) = forward_ode.((model,), population, z; kwargs...)
function forward_ode(model::AbstractDEModel, individual::AbstractIndividual, zᵢ::AbstractVecOrMat; get_dv::Bool=false, sensealg=nothing, full::Bool=false, interpolate::Bool=false, saveat = is_timevariable(individual) ? individual.t.y : individual.t)
    u0 = isempty(individual.initial) ? model.problem.u0 : individual.initial
    saveat_ = interpolate ? empty(saveat) : saveat
    save_idxs = full ? (1:length(u0)) : model.dv_compartment
    prob = remake(model.problem, u0 = u0, tspan = (model.problem.tspan[1], maximum(saveat)), p = zᵢ)
    interpolate && (individual.callback.save_positions .= 1)
    sol = solve(prob, Tsit5(),
        save_idxs = save_idxs, saveat = saveat_, callback=individual.callback, 
        tstops=individual.callback.condition.times, sensealg=sensealg
    )
    interpolate && (individual.callback.save_positions .= 0)
    return get_dv ? sol[model.dv_compartment, :] : sol
end