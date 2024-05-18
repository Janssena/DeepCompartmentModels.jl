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

# forward_ode → solve ode
forward_ode(model::AbstractDEModel, args...; kwargs...) = forward_ode(model.problem, args...; dv_idx=model.dv_compartment, kwargs...)
forward_ode(prob::SciMLBase.AbstractDEProblem, container::Union{Population, AbstractIndividual}, zᵢ::AbstractVecOrMat; get_dv=Val(false), kwargs...) = forward_ode(prob, container, zᵢ, get_dv; kwargs...)
forward_ode(prob::SciMLBase.AbstractDEProblem, population::Population, z::AbstractMatrix, get_dv; kwargs...) = forward_ode.((prob,), population, eachcol(z), (get_dv,); kwargs...)
forward_ode(prob::SciMLBase.AbstractDEProblem, population::Population, z::AbstractVector{<:AbstractMatrix}, get_dv; kwargs...) = forward_ode.((prob,), population, z, (get_dv,); kwargs...)
# Handle the case where we want the dv directly.
forward_ode(prob::SciMLBase.AbstractDEProblem, individual::AbstractIndividual, zᵢ::AbstractVecOrMat, ::Val{true}; dv_idx, kwargs...) = forward_ode(prob, individual, zᵢ, Val(false); kwargs...)[dv_idx, :]

function forward_ode(problem::SciMLBase.AbstractDEProblem, individual::AbstractIndividual, zᵢ::AbstractVecOrMat, ::Val{false}; dv_idx=1, sensealg=nothing, interpolate::Bool=false, saveat=get_t(individual))
    u0 = all(iszero.(problem.u0)) && !isempty(individual.initial) ? individual.initial : problem.u0  # Set to individual.initial only when u0 is all zeros and initial is not empty
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