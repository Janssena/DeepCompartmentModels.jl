import Random
import Lux

include("population.jl");
include("objectives.jl");
include("constrain.jl");

"""Write docs"""
struct StandardNeuralNetwork{O<:AbstractObjective,M<:Lux.AbstractExplicitLayer,P,R<:Random.AbstractRNG} <: AbstractModel{O,M,P}
    objective::O
    ann::M
    p::P
    rng::R
    # Constructors
    function StandardNeuralNetwork(ann::M; objective::O=SSE(), rng::R=Random.default_rng()) where {O<:AbstractObjective,M,R<:AbstractRNG}
        if !(objective isa SSE) && !(objective isa LogLikelihood)
            return throw(ErrorException("StandardNeuralNetwork model is not implemented for $O. Use `SSE()` or `LogLikelihood()` instead."))
        end
        !(ann isa Lux.AbstractExplicitLayer) && (ann = Lux.transform(ann))
        p = init_params(rng, objective, ann)
        new{O,typeof(ann),typeof(p),R}(objective, ann, p, rng)
    end

    function StandardNeuralNetwork(ann::M, ps, st; objective::O=SSE(), rng::R=Random.default_rng()) where {O<:AbstractObjective,M,R<:AbstractRNG}
        if !(objective isa SSE) && !(objective isa LogLikelihood)
            return throw(ErrorException("StandardNeuralNetwork model is not implemented for $O. Use `SSE()` or `LogLikelihood()` instead."))
        end
        !(ann isa Lux.AbstractExplicitLayer) && (ann = Lux.transform(ann))
        p = init_params(rng, objective, ps, st)
        new{O,typeof(ann),typeof(p),R}(objective, ann, p, rng)
    end
end

SNN(args...; kwargs...) = StandardNeuralNetwork(args...; kwargs...)

_stack_x_and_t(indv::AbstractIndividual; saveat = indv.t) = vcat(repeat(indv.x, 1, length(saveat)), transpose(saveat))

function forward(model::StandardNeuralNetwork, individual::AbstractIndividual, p::NamedTuple; saveat = individual.t) 
    X = _stack_x_and_t(individual; saveat)
    ŷ, st = model.ann(X, p.weights, p.st)
    return ŷ[1, :], st
end

function forward(model::StandardNeuralNetwork, population::Population, p::NamedTuple)
    X = reduce(hcat, _stack_x_and_t.(population))
    ŷ, st = model.ann(X, p.weights, p.st)
    return ŷ[1, :], st
end

forward_adjoint(model::StandardNeuralNetwork, args...) = forward(model, args...)

Base.show(io::IO, snn::StandardNeuralNetwork{O,M,P,R}) where {O,M,P,R} = print(io, "$(typeof(snn).name.name){$(snn.objective)}")