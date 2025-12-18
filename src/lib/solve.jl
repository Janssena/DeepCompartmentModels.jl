SciMLBase.solve(model::AbstractDEModel, individual::AbstractIndividual, z; kwargs...) = 
    SciMLBase.solve(model.problem, individual, z; kwargs...)

"""
    solve(problem, individual, z; solver = Tsit5(), interpolate = false, saveat = get_t(individual), kwargs...)

Extends the solve function from SciMLBase to simplify the solving of differential equations
for the DeepCompartmentModels ecosystem.

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
        individual::AbstractIndividual, 
        z::AbstractVecOrMat{<:Real}; 
        solver = Tsit5(),
        interpolate::Bool = false, 
        saveat::AbstractVector{<:Real} = get_t(individual),
        kwargs...
    )
    prob = _remake_prob(problem, individual, saveat, z)
    interpolate && _set_save_positions!(individual.callback, true)
    sol = SciMLBase.solve(prob, solver;
        saveat = interpolate ? empty(saveat) : saveat, callback = individual.callback, 
        tstops = individual.callback.condition.times,
        kwargs...
    )
    interpolate && _set_save_positions!(individual.callback, false)
    return sol
end

"""
    _remake_prob(problem, individual, saveat, z)

Internal function that remakes the DEProblem to have the same type as the Individual, takes the u0 from the 
individual if not empty, sets the full DE parameters using `construct_p(z, individual)`, and makes sure 
that the tspan is in support of the maximum of `saveat`.
"""
function _remake_prob(prob::SciMLBase.AbstractODEProblem, individual::AbstractIndividual{T}, saveat, z::AbstractArray) where T
    p = construct_p(z, individual)
    u0 = _get_u0(T.(prob.u0), individual.u0)
    return remake(prob, u0 = T.(u0), tspan = (T(prob.tspan[1]), T(maximum(saveat))), p = p)
end

_set_save_positions!(callback::DiscreteCallback, value::Bool) = 
    callback.save_positions .= value

_get_u0(prob_u0::AbstractVector{T}, individual_u0::AbstractVector{T}) where T<:Real = 
    !isempty(individual_u0) ? individual_u0 : prob_u0

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
    @ignore_derivatives z.I = zero(T)
    return z
end

"""
    solve_for_target(model::AbstractDEModel, individual::AbstractIndividual, z::AbstractArray{<:Real})

Specific version of the solve call that passes the sensealg to the solve call, only grabs the target 
indices from prediction inside the `sol` object.
"""
function solve_for_target(model::DeepCompartmentModel{P,M,E,T}, individual::AbstractIndividual, z::AbstractArray{<:Real}; sensealg = model.sensealg, kwargs...) where {P,M,E,T<:Int}
    sol = solve(model.problem, individual, z; sensealg, kwargs...)
    return _take_target(sol, model.target) # old
end

# TODO: version that works with multiple dvs
_take_target(sol::DESolution, target::Int) = Array(sol)[target, :]

solve_for_target(dcm::DeepCompartmentModel{P,M,E,T}, population::Population{<:AbstractIndividual}, z::AbstractMatrix; kwargs...) where {P,M,E,T} = 
    solve_for_target.((dcm, ), population, eachcol(z); kwargs...)

solve_for_target(dcm::DeepCompartmentModel{P,M,E,T}, population::Population{<:TimeVariableIndividual}, z::AbstractVector{<:AbstractMatrix}; kwargs...) where {P,M,E,T} = 
    solve_for_target.((dcm, ), population, z; kwargs...)

SciMLBase.solve(dcm::AbstractDEModel, population::Population{<:AbstractIndividual}, z::AbstractMatrix; kwargs...) = 
    SciMLBase.solve.((dcm, ), population, eachcol(z); kwargs...)

SciMLBase.solve(dcm::AbstractDEModel, population::Population{<:TimeVariableIndividual}, z::AbstractVector{<:AbstractMatrix}; kwargs...) = 
    SciMLBase.solve.((dcm, ), population, z; kwargs...)