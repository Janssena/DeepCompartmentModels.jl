abstract type AbstractModel{O,M,P} end

"""Alias for the forward function"""
predict(args...; kwargs...) = forward(args...; kwargs...)

objective(model::M, container::Union{AbstractIndividual, Population}; kwargs...) where {M<:AbstractModel} = objective(model, container, model.p; kwargs...)
forward(model::M, container::Union{AbstractIndividual, Population}; kwargs...) where {M<:AbstractModel} = forward(model, container, model.p; kwargs...)

"""Performs an in-place update of model parameters when P <: NamedTuple."""
function update!(model::AbstractModel{O,M,P}, p::P) where {O,M,P<:NamedTuple}
    Lux.fmap((x,y) -> x .= y, model.p, p)
    return nothing
end