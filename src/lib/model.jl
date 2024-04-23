abstract type AbstractModel{O,M,P} end

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