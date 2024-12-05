struct NeuralODE{T,P<:SciMLBase.AbstractDEProblem,M<:Lux.StatefulLuxLayer,E,S} <: AbstractDEModel{T,P,M,E,S}
    problem::P
    node::M # Not necessarily used during optimization, but useful for interpretation
    dv_compartment::Int
    error::E
    sensealg::S
    NeuralODE(problem::P, node::M, error::E, ::Type{T}=Float32; dv_compartment::Int = 1, sensealg::S = InterpolatingAdjoint(; autojacvec = ReverseDiffVJP())) where {T,P,M,E,S} = 
        new{T,P,M,E,S}(problem, node, dv_compartment, error, sensealg)
end

NeuralODE(f::Function, node, ::Type{T}=Float32; kwargs...) where T = 
    NeuralODE(f, node, ImplicitError(), T; kwargs...)

function NeuralODE(dudt::Function, node_::Lux.AbstractLuxLayer, error::AbstractErrorModel, ::Type{T}=Float32; kwargs...) where T
    num_partials = _estimate_num_partials_node(node_)
    return NeuralODE(dudt, node_, num_partials, error, T; kwargs...) 
end
    
function NeuralODE(dudt::Function, node_::Lux.AbstractLuxLayer, num_partials::Int, error::AbstractErrorModel, ::Type{T}=Float32; kwargs...) where T
    _, st = Lux.setup(Random.GLOBAL_RNG, node_)
    node = StatefulLuxLayer{true}(node_, nothing, st)

    dudt_ = ODEFunction{false}(dudt$(; model = node))
    problem = ODEProblem{false}(dudt_, zeros(T, num_partials), (-T(0.1), one(T)))
    return NeuralODE(problem, node, error, T; kwargs...)
end

function _estimate_num_partials_node(node::Lux.AbstractLuxLayer) 
    if node[end] isa Dense
        return node[end].out_dims
    else
        throw(ErrorException("Cannot estimate the number of patials in the NeuralODE. Call `NeuralODE(dudt, node, num_partials, error)` instead."))
    end
end

function init_theta(model::NeuralODE{T,P,M,E,S}) where {T,P,M<:StatefulLuxLayer,E,S}
    # Note: model.node here is a StatefulLuxLayer
    ps, _ = Lux.setup(Random.GLOBAL_RNG, model.node.model)
    return ComponentVector((weights = ps, I = zero(T))), model.node.st
end

Base.show(io::IO, model::NeuralODE{T,D,M,E,S}) where {T,D,M,E,S} = print(io, "NeuralODE{$T, $(model.error)}")

################################################################################
##########                        Model API                           ##########
################################################################################

predict(model::NeuralODE{T,P,M,E,S}, individual::AbstractIndividual, ps::NamedTuple, st; kwargs...) where {T,P,M,E,S} = 
    forward_ode(model, individual, ps.theta; kwargs...)

predict(model::NeuralODE{T,P,M,E,S}, population::Population, ps::NamedTuple, st; kwargs...) where {T,P,M,E,S} = 
    forward_ode.((model, ), population, (ps.theta, ); kwargs...)

forward_ode_with_dv(model::NeuralODE{T,P,M,E,S}, data, ps::NamedTuple; kwargs...) where {T,P,M,E,S} =
    forward_ode_with_dv(model, data, ps.theta; kwargs...)

forward_ode_with_dv(model::NeuralODE{T,P,M,E,S}, population::Population, ps::ComponentVector; kwargs...) where {T,P,M,E,S} = 
    forward_ode_with_dv.((model, ), population, (ps, ); kwargs...)

function forward_ode_with_dv(model::NeuralODE{T,P,M,E,S}, individual::AbstractIndividual, ps::ComponentVector; sensealg = model.sensealg, kwargs...) where {T,P,M,E,S}
    @ignore_derivatives ps.I = zero(eltype(ps))
    return Array(
        forward_ode(model.problem, individual, ps; sensealg, force_dtmin = true, kwargs...)
    )[model.dv_compartment, :]
end

predict_de_parameters(::Union{FixedObjective, Type{FixedObjective}}, model::NeuralODE{T,P,M,E,S}, data, ps, st) where {T,P,M,E,S} = 
    ps.theta, model.node.st

# Convenience function for StatefulLuxLayers that wrap AbstractLuxContainerLayers
function (s::StatefulLuxLayer{Lux.True,<:AbstractLuxContainerLayer,<:Any,<:Any})(layer::Symbol, x, ps)
    y, _ = getproperty(s.model, layer)(x, ps, s.st[layer])    
    return y
end