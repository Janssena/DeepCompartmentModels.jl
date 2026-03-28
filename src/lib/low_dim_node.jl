struct LowDimNODE <: Lux.AbstractLuxContainerLayer{(:u, :t)}
    u
    t
end

"""
    LowDimensionalNeuralODE(ann, node; kwargs...)

Convenience function for constructing low-dimensional Neural-ODE based models.
This simple implementation learns a model according to:
    
    du/dt = node.u(u) + I * node.t(t)

Where the model learns the system dynamics based on the previous state u and 
current time point t. See [bram2023] for more details.

# Arguments:
- `u`: NeuralODE taking `u` as its input.
- `t`: NeuralODE taking `t` as its input.
- `kwargs`: keyword arguments given to the NeuralODE constructor.

[bram2023] Bräm, Dominic Stefan, et al. "Low-dimensional neural ODEs and their application in pharmacokinetics." Journal of Pharmacokinetics and Pharmacodynamics (2023): 1-18.
"""
LowDimensionalNeuralODE(u::Lux.AbstractLuxLayer, t::Lux.AbstractLuxLayer, error::AbstractErrorModel, ::Type{T} = Float32; kwargs...) where T = 
    LowDimensionalNeuralODE(LowDimNODE(u, t), error, T; kwargs...)

LowDimensionalNeuralODE(u::Lux.AbstractLuxLayer, t::Lux.AbstractLuxLayer, ::Type{T} = Float32; kwargs...) where T = 
    LowDimensionalNeuralODE(LowDimNODE(u, t), ImplicitError(), T; kwargs...)

function LowDimensionalNeuralODE(m::LowDimNODE, error::AbstractErrorModel, ::Type{T}=Float32; kwargs...) where T
    dudt(u, p, t; model) = model(:u, u, p.weights.u) + p.I * model(:t, [t], p.weights.t)
    return NeuralODE(dudt, m, error, T; kwargs...)
end

LowDimensionalNeuralODE(m::LowDimNODE, ::Type{T}=Float32; kwargs...) where T = 
    LowDimensionalNeuralODE(m, ImplicitError(), T; kwargs...)

_estimate_num_partials_node(node::LowDimNODE) = 1

Base.show(io::IO, model::NeuralODE{T,D,M,E,S}) where {T,D,M<:Lux.StatefulLuxLayer{<:Any,<:LowDimNODE},E,S} = print(io, "LowDimensionalNeuralODE{$T, $(model.error)}")

