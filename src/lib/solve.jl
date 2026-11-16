SciMLBase.solve(model::AbstractDEModel, individual::AbstractIndividual, z; kwargs...) = 
    SciMLBase.solve(model.problem, individual, z; kwargs...)

"""
    solve(problem, individual, z; solver = Tsit5(), interpolate = false, saveat = get_t(individual), kwargs...)

Extends the solve function from SciMLBase to simplify the solving of differential equations
within the DeepCompartmentModels ecosystem.

# Arguments:
- `problem`: DEProblem to solve.
- `individual`: Individual for which to solve the DE.
- `z`: DE parameters. Are passed to a downstream `construct_p(z, individual)` function to add additional variables used in the DE function.
- `solver = Tsit5()`: Solver to use.
- `interpolate = false`: , 
- `saveat = get_t(individual)`: time points at which to save the solution. Is forced to be empty when interpolate = true.
- `kwargs`: Additional keyword arguments that are passed to the solve call from DifferentialEquations.jl.
"""
function SciMLBase.solve(
        problem::SciMLBase.AbstractDEProblem, 
        individual::AbstractIndividual{T,O}, 
        z::AbstractVecOrMat{<:Real}; 
        solver = Tsit5(),
        interpolate::Bool = false, 
        saveat::AbstractVector{<:Real} = get_t(individual),
        kwargs...
    ) where {T,O<:Nothing}
    prob = _remake_prob(problem, individual, saveat, z)
    callback = _solve_callback(individual.callback, interpolate)
    return SciMLBase.solve(prob, solver;
        saveat = interpolate ? empty(saveat) : saveat, callback, 
        tstops = _get_tstops(individual.callback), kwargs...
    )
end

function SciMLBase.solve(
        problem::SciMLBase.AbstractDEProblem, 
        individual::AbstractIndividual{T,O}, 
        z::AbstractVecOrMat{<:Real}; 
        solver = Tsit5(),
        interpolate::Bool = false, 
        saveat::AbstractVector{<:Real} = get_t(individual),
        kwargs...
    ) where {T,O<:AbstractVector{<:Pair}}

    callback = _solve_callback(individual.callback, interpolate)
    tstops = _get_tstops(individual.callback)
    sols = map(individual.occasions) do occ
        saveat_occ = filter(Base.Fix2(_within_occ, occ), saveat)
        tstops_occ = filter(Base.Fix2(_within_occ, occ), tstops)
        _problem = remake(problem, tspan = (occ.first - T(0.1), problem.tspan[2]))
        prob = _remake_prob(_problem, individual, saveat_occ, z)
        return SciMLBase.solve(prob, solver;
            saveat = interpolate ? empty(saveat_occ) : saveat_occ, callback, 
            tstops = tstops_occ, kwargs...
        )
    end
    return sols
end

_within_occ(t::Real, occ::Pair) = t >= occ.first && t <= occ.second

"""
    _remake_prob(problem, individual, saveat, z)

Internal function that remakes the DEProblem to have the same type as the Individual, takes the u0 from the 
individual if not empty, sets the full DE parameters using `construct_p(z, individual)`, and makes sure 
that the tspan is in support of the maximum of `saveat`.
"""
function _remake_prob(prob::SciMLBase.AbstractDEProblem, individual::AbstractIndividual{T}, saveat, z::AbstractArray) where T
    p = construct_p(z, individual)
    u0 = _get_u0(T.(prob.u0), individual.u0)
    return remake(prob, u0 = T.(u0), tspan = (T(prob.tspan[1]), T(maximum(saveat))), p = p)
end

# When interpolating, the solution has to be saved at the callback events. We return a modified
# copy instead of changing `save_positions` on the callback of the individual, since that object
# is shared between solves (and threads).
_solve_callback(callback, interpolate::Bool) = 
    interpolate ? _with_save_positions(callback, true) : callback

_with_save_positions(callback::DiscreteCallback, value::Bool) = 
    Accessors.@set callback.save_positions = BitArray([value, value])

_with_save_positions(callbacks::CallbackSet, value::Bool) = 
    CallbackSet(callbacks.continuous_callbacks, map(Base.Fix2(_with_save_positions, value), callbacks.discrete_callbacks))

_get_u0(prob_u0::AbstractVector{T}, individual_u0::AbstractVector{T}) where T<:Real = 
    !isempty(individual_u0) ? individual_u0 : prob_u0

_get_tstops(callback::DiscreteCallback) = callback.condition.times

# TODO: What about continuous callbacks?
_get_tstops(callbacks::CallbackSet) = 
    sort(reduce(vcat, map(_get_tstops, callbacks.discrete_callbacks)))


"""
    construct_p(z::AbstractVector, ::AbstractIndividual)

Creates the full parameter vector based on the individual type. The default behaviour is to 
set `p = [z; 0]`, adding a trailing zero to the vector that represents the treatment intervention.
The callback function changes this value in-place and should be present in the DEFunction. This 
function can (and should be) extended for custom Individual types that require additional (unlearnable) 
variables in the DEFunction.

# Arguments:
- `z`: DE parameters.
- `individual`: Individual for which the DE is solved.
"""
construct_p(z::AbstractVector{T}, ::AbstractIndividual) where T = 
    vcat(z, zero(T))

"""
    construct_p(z::AbstractMatrix, ::TimeVariableIndividual)

DE parameters for TimeVariableIndividuals are Matrices and thus a specific function is
required to add zeros to the bottom row of the matrix to correctly set the treament
intervention.
"""
construct_p(z::AbstractMatrix{T}, ::TimeVariableIndividual) where T = 
    vcat(z, zeros(T, 1, size(z, 2)))

"""
    construct_p(z::ComponentArray, ::AbstractIndividual)

If the DE parameters are ComponentArrays, they already contain an Intervention key that is
initialised to zero. This is the case for UniversalDiffEq, since these take ComponentArrays as parameters.
"""
function construct_p(z::ComponentArray{T}, ::AbstractIndividual) where T
    # The callback mutates `p.I` during the solve. Work on a copy so that the parameters of the
    # caller (and those of other individuals solved on other threads) are never modified.
    p = copy(z)
    p.I = zero(T)
    return p
end

"""
    solve_for_target(model::AbstractDEModel, individual::AbstractIndividual, z::AbstractArray{<:Real})

Specific version of the solve call that passes the sensealg to the solve call, only grabs the target 
indices from prediction inside the `sol` object.
"""
function solve_for_target(model::DeepCompartmentModel{P,M,E,T}, individual::AbstractIndividual, z::AbstractArray{<:Real}; sensealg = model.sensealg, kwargs...) where {P,M,E,T<:Union{Int,AbstractVector{Int}}}
    sol = solve(model.problem, individual, z; sensealg, kwargs...)
    # The outputs of an `MOIndividual` are observed at subsets of its times (`dvid`), which only applies
    # when the solution is saved at those times. A custom `saveat` returns every output at every time.
    saveat = get(kwargs, :saveat, nothing)
    isnothing(saveat) || saveat == get_t(individual) || return _take_target(sol, model.target)
    return _take_target(sol, individual, model.target) # old
end

_take_target(sol::SciMLBase.AbstractODESolution, ::AbstractIndividual, target::Int) = _take_target(sol, target)

function _take_target(sol::SciMLBase.AbstractODESolution, individual::MOIndividual, target::AbstractVector{Int}) 
    ŷs = _take_target(sol, target)
    return map(getindex, ŷs, individual.dvid) # TODO: view?
end

_take_target(sol::SciMLBase.AbstractODESolution, target::Int) = Array(sol)[target, :]
function _take_target(sol::SciMLBase.AbstractODESolution, target::AbstractVector{Int}) 
    preds = Array(sol)
    return map(target) do j
        preds[j, :]
    end
end

_take_target(sols::AbstractVector{<:SciMLBase.AbstractODESolution}, target) = 
    map(Base.Fix2(_take_target, target), sols)

# Individuals with occasions are solved once per occasion
_take_target(sols::AbstractVector{<:SciMLBase.AbstractODESolution}, individual::AbstractIndividual, target) = 
    map(sol -> _take_target(sol, individual, target), sols)

solve_for_target(dcm::DeepCompartmentModel{P,M,E,T}, population::Population{<:AbstractIndividual}, z::AbstractMatrix; kwargs...) where {P,M,E,T} = 
    [solve_for_target(dcm, population[i], view(z, :, i); kwargs...) for i in eachindex(population)]

solve_for_target(dcm::DeepCompartmentModel{P,M,E,T}, population::Population{<:TimeVariableIndividual}, z::AbstractVector{<:AbstractMatrix}; kwargs...) where {P,M,E,T} = 
    [solve_for_target(dcm, population[i], z[i]; kwargs...) for i in eachindex(population)]

SciMLBase.solve(dcm::AbstractDEModel, population::Population{<:AbstractIndividual}, z::AbstractMatrix; kwargs...) = 
    [SciMLBase.solve(dcm, population[i], view(z, :, i); kwargs...) for i in eachindex(population)]

SciMLBase.solve(dcm::AbstractDEModel, population::Population{<:TimeVariableIndividual}, z::AbstractVector{<:AbstractMatrix}; kwargs...) = 
    [SciMLBase.solve(dcm, population[i], z[i]; kwargs...) for i in eachindex(population)]