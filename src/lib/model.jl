abstract type AbstractModel{T,M,E} end
abstract type AbstractDEModel{T,P,M,E,S<:SciMLBase.AbstractSensitivityAlgorithm} <: AbstractModel{T,M,E} end

abstract type Parameterization end
struct MeanVar <: Parameterization end # Is used with Natural gradients / Bayesian Learning Rule
struct MeanSqrt <: Parameterization end # needs to be constrained during optimization

const DEFAULT_ALG = Tsit5()

# Improve printing of Vector{<:DESolution} (predict output)
function Base.show(io::IO, ::MIME"text/plain", vector::T) where {T<:Vector{<:SciMLBase.AbstractTimeseriesSolution}}
    println(io, "$(length(vector))-element Vector{<:$(eltype(vector).name.name)}")
    for i in eachindex(vector)
        if i > 5
            println(io, "  ", "...")
            break
        else
            println(io, "  ", "$(eltype(vector).name.name){...}")
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", vector::AbstractVector{T}) where {T<:Vector{<:SciMLBase.AbstractTimeseriesSolution}}
    println(io, "$(length(vector))-element Vector{Vector{<:$(eltype(vector[1]).name.name)}}")
    for i in eachindex(vector)
        if i > 5
            println(io, "  ", "...")
            break
        else
            println(io, "  ", "Vector{<:$(eltype(vector[1]).name.name)}")
        end
    end
end

"""
    forward_ode(problem, individual, z; kwargs...)

# Arguments:
- `problem`: DEProblem to solve.
- `individual`: Individual for which to solve the DE.
- `z`: ODE parameters.
- `sensealg`: Sensitivity algorithm to calculate adjoint for calculating gradients, 
- `interpolate = false`: whether , 
- `saveat`: time points at which to save the solution. Empty when interpolate = true.
"""
function forward_ode(
        problem::SciMLBase.AbstractDEProblem, 
        individual::AbstractIndividual, 
        zᵢ::AbstractVecOrMat; 
        solver = DEFAULT_ALG,
        sensealg = nothing, 
        interpolate::Bool = false, 
        saveat::AbstractVector = get_t(individual),
        kwargs...
    )
    prob = _remake_prob(problem, individual, saveat, zᵢ)
    interpolate && (individual.callback.save_positions .= 1)
    sol = solve(prob, solver;
        saveat = interpolate ? empty(saveat) : saveat, callback = individual.callback, 
        tstops = individual.callback.condition.times, sensealg = sensealg,
        kwargs...
    )
    interpolate && (individual.callback.save_positions .= 0)
    return sol
end

forward_ode(model::AbstractDEModel, individual::AbstractIndividual, z::AbstractVecOrMat; kwargs...) = 
    forward_ode(model.problem, individual, z; kwargs...)

# The below is used in objective functions
forward_ode_with_dv(model::AbstractDEModel, individual::AbstractIndividual, z::AbstractVecOrMat; sensealg = model.sensealg, kwargs...) = 
    Array(forward_ode(model.problem, individual, z; sensealg, kwargs...))[model.dv_compartment, :] # old

for op = (:forward_ode, :forward_ode_with_dv)
    # When running on a whole population given the full ODE parameter matrix
    @eval $op(m, population::Population, z::AbstractMatrix{<:Real}; kwargs...) = 
        $op.((m, ), population, eachcol(z); kwargs...)
    # When running on a whole population with multiple samples of the ODE parameters for VI         
    @eval $op(m, population::Population, z::AbstractVector{<:AbstractMatrix{<:Real}}; kwargs...) = 
        $op.((m, ), (population, ), z; kwargs...)
    # When running on a whole population with time-variable covariates (i.e. matrix per individual)
    @eval $op(m, population::Population{T,I}, zₜ::AbstractVector{<:AbstractMatrix{<:Real}}; kwargs...) where {T<:TimeVariable,I} = 
        $op.((m, ), population, zₜ; kwargs...)
    # When running on a whole population with time-variable covariates & mixed-effects
    @eval $op(m, population::Population{T,I}, zₜ::AbstractVector{<:AbstractVector{<:AbstractMatrix{<:Real}}}; kwargs...) where {T<:TimeVariable,I} = 
        $op.((m, ), (population, ), zₜ; kwargs...)
end

_get_u0(prob_u0::AbstractVector{T}, individual_u0::AbstractVector{T}) where T<:Real = 
    all(iszero.(prob_u0)) && !isempty(individual_u0) ? individual_u0 : prob_u0

function _remake_prob(prob, individual, saveat, p)
    u0 = _get_u0(prob.u0, individual.initial)
    return remake(prob, u0 = u0, tspan = (prob.tspan[1], maximum(saveat)), p = p)
end

##### construct p -> add zero for I and t if time-variable

##### Individuals

construct_p(z::AbstractVector{T}, ::AbstractIndividual) where T<:Real = vcat(z, zero(T))

construct_p(ζ::AbstractVector{<:Real}, η::AbstractVector{<:Real}, individual::AbstractIndividual) = 
    construct_p(ζ .* exp.(η), individual)

# This happens when we have the mask * [η₁]' or when eta is time-dependent. In  
# the latter case we need a different function, maybe through a specific individual?
construct_p(ζ::AbstractVector{<:Real}, η::AbstractMatrix{<:Real}, individual::BasicIndividual) = 
    construct_p(ζ .* exp.(dropdims(η; dims = 2)), individual) 

construct_p(z::AbstractMatrix{T}, individual::AbstractIndividual) where T<:Real = 
    vcat(individual.t.x, z, zeros(T, 1, size(z, 2)))

##### Populations

construct_p(z::AbstractMatrix{T}, ::Population) where T<:Real = 
    vcat(z, zeros(T, 1, size(z, 2)))

construct_p(z::AbstractVector{<:AbstractMatrix{T1}}, population::Population{T2,I}) where {T1<:Real,T2<:TimeVariable,I} = 
    construct_p.(z, population)

construct_p(z::AbstractMatrix{T}, η::AbstractMatrix{T}, ::Population) where {T<:Real} = 
    vcat(z .* exp.(η), zeros(T, 1, size(z, 2)))

construct_p(z::AbstractMatrix{T}, η::AbstractVector{<:AbstractMatrix{T}}, population::Population) where T<:Real = 
    construct_p.((z, ), η, (population, ))::Vector{Matrix{T}}

