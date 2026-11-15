abstract type AbstractUDEType end
abstract type AbstractHybridUDEType <: AbstractUDEType end
struct BasicUDE <: AbstractUDEType end
struct TimeConcatUDE <: AbstractUDEType end

struct UniversalDiffEq{P<:SciMLBase.AbstractDEProblem,T<:AbstractUDEType} <: SciMLBase.AbstractDEProblem
    problem::P
    type::T
    UniversalDiffEq(problem::P, type::T = BasicUDE()) where {P,T} = 
        new{P,T}(problem, type)
end

function UniversalDiffEq(num_partials::Int, ::Type{TU}=Float32; inplace::Bool=false, type::T = BasicUDE()) where {TU,T} 
    problem = ODEProblem{inplace}(_empty_de, zeros(TU, num_partials), (-TU(0.1), one(TU)))
    return UniversalDiffEq(problem, type)
end

_empty_de(args...; kwargs...) = nothing

"""
    build_problem(ude::UniversalDiffEq, model, st) -> AbstractDEProblem

Wraps `model` (with layer state `st`) in a `Lux.StatefulLuxLayer` and returns a copy of
`ude.problem` whose right-hand side calls the `UniversalDiffEq` functor
`ude(model, args...)`. Packages that add new `AbstractUDEType`s can extend this method
for their own `UniversalDiffEq{P,<:MyType}`.

`st` is the Lux state of `model`, i.e. `st.theta` when working with the output of `setup`.
`solve` and `solve_for_target` accept the full state returned by `setup` as well and
select the `theta` entry using `_layer_state`.
"""
function build_problem(ude::UniversalDiffEq, model::Lux.AbstractLuxLayer, st::NamedTuple)
    stateful = Lux.StatefulLuxLayer{true}(model, nothing, st)
    dudt(args...; model = stateful) = ude(model, args...)
    return remake(ude.problem, f = dudt)
end

# For UDEs the neural network is part of the right-hand side, so the number of parameters is
# typically much larger. In a micro benchmark (one-compartment UDE, 16 individuals, Float32) forward
# sensitivities were ~4x faster than adjoints at ~100 parameters, but ~3x slower at ~1,000.
# The break-even is at roughly 250 parameters. The compiled ReverseDiff tape is only valid for
# right-hand sides without value dependent control flow, which holds for Basic/TimeConcatUDE.
const UDE_ADJOINT_MIN_PARAMS = 256

_default_sensealg(::UniversalDiffEq{P,<:Union{BasicUDE,TimeConcatUDE}}, model::Lux.AbstractLuxLayer) where P = 
    Lux.parameterlength(model) >= UDE_ADJOINT_MIN_PARAMS ? GaussAdjoint(; autojacvec = ReverseDiffVJP(false)) : ForwardDiffSensitivity()

# The state returned by `setup` is `(theta = <Lux state>, ...)`, but `build_problem` needs the Lux
# state of the model. Accept both so callers can pass either.
_layer_state(st::NamedTuple) = haskey(st, :theta) ? st.theta : st

# The infusion rate `I` is added to the first state, i.e. du = [I, 0, ...] + model(u, p). Doing this
# in a single fused broadcast avoids the temporaries of `[p.I; zeros(...)] .+ model(...)`.
_add_infusion(du, I) = du .+ ifelse.(eachindex(du) .== 1, I, zero(I))

(::UniversalDiffEq{P,T})(model, u, p, t) where {P,T<:BasicUDE} = _add_infusion(model(u, p), p.I)
(::UniversalDiffEq{P,T})(model, u, p, t) where {P,T<:TimeConcatUDE} = _add_infusion(model(_concat(u, t), p), p.I)

# In-place versions, used when the problem is created with `UniversalDiffEq(n; inplace = true)`.
function (::UniversalDiffEq{P,T})(model, du, u, p, t) where {P,T<:BasicUDE}
    du .= model(u, p)
    du[1] += p.I
    return nothing
end

function (::UniversalDiffEq{P,T})(model, du, u, p, t) where {P,T<:TimeConcatUDE}
    du .= model(_concat(u, t), p)
    du[1] += p.I
    return nothing
end

 # TODO: versions for images
_concat(u::Union{Real, AbstractVector}, t::Real) = [u; t]

function SciMLBase.solve(
        dcm::DeepCompartmentModel{<:UniversalDiffEq}, 
        individual::AbstractIndividual, 
        ps::Union{NamedTuple, ComponentArray},
        st::NamedTuple; 
        kwargs...
    )
    prob = build_problem(dcm.problem, dcm.model, _layer_state(st))
    return solve(prob, individual, ps; kwargs...) 
end

SciMLBase.solve(
        dcm::DeepCompartmentModel{<:UniversalDiffEq}, 
        population::Population, 
        ps::Union{NamedTuple, ComponentArray},
        st::NamedTuple; 
        kwargs...
    ) = solve.((dcm, ), population, (ps, ), (st, ); kwargs...)

solve_for_target(
        dcm::DeepCompartmentModel{<:UniversalDiffEq}, 
        population::Population, 
        ps::Union{NamedTuple, ComponentArray},
        st::NamedTuple; 
        kwargs...
    ) = solve_for_target.((dcm, ), population, (ps, ), (st, ); kwargs...)

function solve_for_target(
        dcm::DeepCompartmentModel{<:UniversalDiffEq}, 
        individual::AbstractIndividual, 
        ps::Union{NamedTuple, ComponentArray},
        st::NamedTuple; 
        sensealg = dcm.sensealg,
        kwargs...
    )
    prob = build_problem(dcm.problem, dcm.model, _layer_state(st))
    sol = solve(prob, individual, ps; sensealg, kwargs...)
    return _take_target(sol, individual, dcm.target)
end

"""
    predict(dcm::DeepCompartmentModel{<:UniversalDiffEq}, data, ps, st; target = true, kwargs...)

Solve the UDE and return the predicted observations (`target = true`) or the full solution(s)
(`target = false`). Unlike the parameter-network based DCM, a UDE has no per-individual
parameters to predict, so the `individual` keyword of the generic `predict` does not apply.
`kwargs` (e.g. `saveat`) are forwarded to `solve`.
"""
function predict(dcm::DeepCompartmentModel{<:UniversalDiffEq}, data, ps, st; target = true, individual = nothing, kwargs...)
    return target ? solve_for_target(dcm, data, ps, st; kwargs...) : solve(dcm, data, ps, st; kwargs...)
end

SciMLBase.solve(prob::SciMLBase.AbstractDEProblem, individual::AbstractIndividual, ps::NamedTuple; kwargs...) = 
    solve(prob, individual, ps.theta; kwargs...)

function Base.summary(io::IO, ude::UniversalDiffEq)
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(io,
        type_color, nameof(typeof(ude)),
        no_color, " with uType ",
        type_color, typeof(ude.problem.u0),
        no_color, " and tType ",
        type_color,
        ude.problem.tspan isa Function ?
        "Unknown" : (ude.problem.tspan === nothing ?
         "Nothing" : typeof(ude.problem.tspan[1])),
        no_color,
        ". In-place: ", type_color, isinplace(ude.problem), no_color
    )
end

function Base.show(io::IO, mime::MIME"text/plain", ude::UniversalDiffEq)
    summary(io, ude)
    println(io)
    print(io, "timespan: ")
    show(io, mime, ude.problem.tspan)
    println(io)
    print(io, "u0: ")
    show(io, mime, ude.problem.u0)
end

Base.show(io::IO, dcm::DeepCompartmentModel{<:UniversalDiffEq}) = 
    print(io, "DeepCompartmentModel{ude = $(dcm.problem.type), error = $(dcm.error)}")
