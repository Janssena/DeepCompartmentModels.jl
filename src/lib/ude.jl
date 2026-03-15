abstract type AbstractUDEType end
struct BasicUDE <: AbstractUDEType end
struct TimeConcatUDE <: AbstractUDEType end

struct UniversalDiffEq{P<:SciMLBase.AbstractDEProblem,T<:AbstractUDEType} <: SciMLBase.AbstractDEProblem
    problem::P
    type::T
    function UniversalDiffEq(problem::P, type::T = BasicUDE()) where {P,T} 
        if isinplace(problem)
            throw(ErrorException("UniversalDiffEq does not work with in-place DEProblems. Define your DE function using out-of-place functionality."))
        end
        return new{P,T}(problem, type)
    end
end

function UniversalDiffEq(num_partials::Int, ::Type{T1}=Float32; type::T = BasicUDE()) where {T1,T} 
    problem = ODEProblem{false}(_empty_de, zeros(T1, num_partials), (-T1(0.1), one(T1)))
    return UniversalDiffEq(problem, type)
end

_empty_de(u, p, t) = nothing

function build_problem(ude::UniversalDiffEq{P}, model::Lux.AbstractLuxLayer, st::NamedTuple) where P<:SciMLBase.AbstractODEProblem
    stateful = Lux.StatefulLuxLayer{true}(model, nothing, st.theta)
    dudt(u, p, t; model = stateful) = ude(model, u, p, t)
    return remake(ude.problem, f = dudt)
end

(::UniversalDiffEq{P,T})(model, u, p, t) where {P,T<:BasicUDE} = [p.I; zeros(eltype(u), length(u) - 1)] .+ model(u, p)
(::UniversalDiffEq{P,T})(model, u, p, t) where {P,T<:TimeConcatUDE} = [p.I; zeros(eltype(u), length(u) - 1)] .+ model(_concat(u, t), p)

 # TODO: versions for images
_concat(u::Union{Real, AbstractVector}, t::Real) = [u; t]

function SciMLBase.solve(
        dcm::DeepCompartmentModel{<:UniversalDiffEq}, 
        individual::AbstractIndividual, 
        ps::Union{NamedTuple, ComponentArray},
        st::NamedTuple; 
        kwargs...
    )
    prob = build_problem(dcm.problem, dcm.model, st)
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
    prob = build_problem(dcm.problem, dcm.model, st)
    sol = solve(prob, individual, ps; sensealg, kwargs...)
    return _take_target(sol, individual, dcm.target)
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

Base.show(io::IO, dcm::DeepCompartmentModel{UniversalDiffEq{P,T},M,E,S}) where {P,T,M,E,S} = 
    print(io, "DeepCompartmentModel{UniversalDiffEq($(nameof(P)), $(nameof(T))), $(dcm.error)}")
