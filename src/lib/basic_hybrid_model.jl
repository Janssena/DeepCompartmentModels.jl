struct BasicHybridModel <: Lux.AbstractExplicitContainerLayer{(:ann,:node)} 
    ann
    node
end

function forward(m::HybridDCM{O,D,M,P,R}, individual::AbstractIndividual, p; kwargs...) where {O<:FixedObjective,D,M<:BasicHybridModel,P,R}
    ζ, st = m.model.ann(get_x(individual), p.weights.ann, m.st.ann)
    return forward_ode(m, individual, ζ, p.weights.node; kwargs...), st
end

function forward(m::HybridDCM{O,D,M,P,R}, population::Population, p; kwargs...) where {O<:FixedObjective,D,M<:BasicHybridModel,P,R}
    ζ, st = m.model.ann(get_x(population), p.weights.ann, m.st.ann)
    return forward_ode.((m,), population, eachcol(ζ), (p.weights.node,); kwargs...), st
end

function forward_ode(m::HybridDCM{O,D,M,P,R}, individual::AbstractIndividual, z::AbstractVector, p_node::ComponentArray; kwargs...) where {O,D,M<:BasicHybridModel,P,R}
    p = _add_z_and_I(p_node, z)
    return forward_ode(m, individual, p; kwargs...)
end

forward_adjoint(args...) = forward(args...; get_dv = Val(true), sensealg = InterpolatingAdjoint(; autojacvec = ReverseDiffVJP(true)))