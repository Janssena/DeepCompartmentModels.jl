abstract type AbstractHybridUDEType <: AbstractUDEType end

"""
    HybridModel(num_latent, encoder, node)

Container coupling a covariate `encoder` (→ `num_latent` baseline parameters) with
a NeuralODE `node` that learns one unknown dynamic term.
"""
struct HybridModel{E,NODE} <: Lux.AbstractLuxContainerLayer{(:encoder, :node)}
    num_latent::Int
    encoder::E
    node::NODE
end

"""
    CustomUDE(name::Symbol)

Tag selecting the user-defined differential equation. Define the dynamics with a
method

    (::UniversalDiffEq{P,T})(model, u, p, t) where {T<:CustomUDE{StaticSymbol{:name}}}

where `p.latents` is the encoder output, `p.node` the NeuralODE parameters, `p.I`
the intervention, and `model([t;;], p.node)` evaluates the NeuralODE at time `t`.
"""
struct CustomUDE{O<:StaticSymbol} <: AbstractHybridUDEType
    ode_fn::O
end
CustomUDE(s::Symbol) = CustomUDE(static(s))
Base.show(io::IO, ude::CustomUDE) = print(io, "CustomUDE{$(dynamic(ude.ode_fn))}()")

setup(rng::Random.AbstractRNG, dcm::DeepCompartmentModel{<:UniversalDiffEq,<:HybridModel}) =
    Lux.setup(rng, dcm.model)

Lux.initialparameters(rng::Random.AbstractRNG, m::HybridModel) = (
    encoder=Lux.initialparameters(rng, m.encoder),
    dynamics=ComponentVector(
        latents=zeros(Float32, m.num_latent),
        node=Lux.initialparameters(rng, m.node),
        I=zero(Float32),
    ),
)

Lux.initialstates(rng::Random.AbstractRNG, m::HybridModel) = (
    encoder=Lux.initialstates(rng, m.encoder),
    dynamics=(latents=NamedTuple(), node=Lux.initialstates(rng, m.node)),
)

# Baseline latents, with (mixed) or without (fixed) multiplicative random effects.
_hybrid_random_effect(ζ::AbstractArray, _, ::NamedTuple{(:theta,)}) = ζ
_hybrid_random_effect(ζ::AbstractArray, _, ::NamedTuple{(:theta, :encoder)}) = ζ
function _hybrid_random_effect(ζ::AbstractVector, ps, st::NamedTuple{(:theta, :phi)})
    η = get_random_effects(ps, st)
    return @. ζ * exp(η)
end

function SciMLBase.solve(
    dcm::DeepCompartmentModel{<:UniversalDiffEq,<:HybridModel},
    individual::AbstractIndividual,
    ps::Union{NamedTuple,ComponentArray},
    st::NamedTuple;
    kwargs...,
)
    ζ, _ = dcm.model.encoder(get_x(individual), ps.theta.encoder, st.theta.encoder)
    z = vec(_hybrid_random_effect(ζ, ps, st))
    ps_dynamic = ComponentVector(
        [z; ps.theta.dynamics.node; zero(ps.theta.dynamics.I)],
        getaxes(ps.theta.dynamics))
    prob = build_problem(dcm.problem, dcm.model, st)
    return solve(prob, individual, ps_dynamic; sensealg=dcm.sensealg, kwargs...)
end

function build_problem(ude::UniversalDiffEq{P}, model::HybridModel, st::NamedTuple) where {P<:SciMLBase.AbstractODEProblem}
    stateful = Lux.StatefulLuxLayer{true}(model.node, nothing, st.theta.dynamics.node)
    dudt(u, p, t; model=stateful) = ude(model, u, p, t)
    return remake(ude.problem, f=dudt)
end

function solve_for_target(
    dcm::DeepCompartmentModel{<:UniversalDiffEq,<:HybridModel},
    individual::AbstractIndividual,
    ps::NamedTuple,
    st::NamedTuple;
    kwargs...,
)
    sol = solve(dcm, individual, ps, st; kwargs...)
    return _take_target(sol, individual, dcm.target)
end

_estimate_typ_parameter_size(
    dcm::DeepCompartmentModel{<:UniversalDiffEq,<:HybridModel}, ::Population, args...) =
    dcm.model.num_latent

# User-facing prediction — the gap the base UniversalDiffEq leaves open.
function predict(
    dcm::DeepCompartmentModel{<:UniversalDiffEq,<:HybridModel}, data, ps, st;
    individual=true, target=true, kwargs...,
)
    return target ? solve_for_target(dcm, data, ps, st; kwargs...) :
           solve(dcm, data, ps, st; kwargs...)
end

"""
    interpret_node(dcm, ps, st; t_dummy=0:360)

Evaluate the NeuralODE `node` on dummy time points to expose the learned
time-varying function (e.g. a fold-change in clearance). Returns `(t, effect)`.
"""
function interpret_node(
    dcm::DeepCompartmentModel{<:UniversalDiffEq,<:HybridModel}, ps, st; t_dummy=0:360)
    grid = permutedims(collect(Float32, t_dummy))
    effect, _ = dcm.model.node(grid, ps.theta.dynamics.node, st.theta.dynamics.node)
    return vec(grid), vec(effect)
end
