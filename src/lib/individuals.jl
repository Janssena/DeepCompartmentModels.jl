"""
    abstract type AbstractIndividual{T} end

Supertype for various Individual types.
"""
abstract type AbstractIndividual{T,I,C} end

Base.show(io::IO, indv::AbstractIndividual{T}) where T = 
    print(io, "$(typeof(indv).name.name){$T}(id = $(indv.id), ...)")

Base.copy(individual::AbstractIndividual) = copy(individual, individual.callback)

Base.copy(individual::AbstractIndividual{T,I,C}, cb::Union{DiscreteCallback, CallbackSet}) where {T,I,C} = 
    typeof(individual).name.wrapper{T,I,typeof(cb)}(
        [field == :callback ? cb : deepcopy(getfield(individual, field)) for field in fieldnames(typeof(individual))]...
    )

get_x(indv::AbstractIndividual, key::Symbol=:zeta) = indv.x[key]
get_t(indv::AbstractIndividual) = indv.t
get_y(indv::AbstractIndividual) = indv.y

@non_differentiable get_x(::AbstractIndividual, ::Symbol)
@non_differentiable get_y(::AbstractIndividual)


################################################################################
##########                        Individuals                         ##########
################################################################################

"""
    BasicIndividual{T,I,C}(...)

Struct holding the data for a single subject for standard PK or PD analyses. 
This type of individual has a covariate vector that is assumed static over time, 
and only has a single type of observation of type T (i.e. `x`, `t`, and `y` are 
one-dimensional).
"""
struct BasicIndividual{T,I<:Union{Integer, AbstractString},C} <: AbstractIndividual{T,I,C}
    id::I
    x::@NamedTuple{zeta::Vector{T}, error::Vector{T}}
    t::Vector{T}
    y::Vector{T}
    initial::Vector{T}
    callback::C
end

BasicIndividual(id, x::AbstractVector, t, y, cb, ::Type{T}=Float32; kwargs...) where {T} = 
    BasicIndividual(id, (zeta = x, error = empty(x)), t, y, cb, T; kwargs...)

function BasicIndividual(id::I, x::NamedTuple{(:zeta,:error)}, t::AbstractVector, y::AbstractVector, cb::C, ::Type{T}=Float32; initial::AbstractVector=empty(y)) where {T,I,C}
    length(t) !== length(y) && throw(ErrorException("Length of time points vector does not match length of observations."))
    _callback_type_matches(cb, T) # warn if callback does not match type.
    return BasicIndividual{T,I,C}(
        id, 
        fmap(Base.Fix1(convert, Vector{T}), x),
        map(Base.Fix1(convert, Vector{T}), (t, y, initial))...,
        cb
    )
end

"""
    TimeVariableIndividual{T,I,C}(...)

Struct holding the data for a single subject for standard PK or PD analyses. 
This type of individual has a covariate vector with multiple columns 
representing observed values of the covariates at specific time points. These 
time points are provided in `t` (which is a NamedTuple{(:x,:y)}).
"""
struct TimeVariableIndividual{T,I<:Union{Integer, AbstractString},C} <: AbstractIndividual{T,I,C}
    id::I
    x::@NamedTuple{zeta::Matrix{T}, error::Matrix{T}}
    t::@NamedTuple{zeta::Matrix{T}, error::Matrix{T}, y::Vector{T}}
    y::Vector{T}
    initial::Vector{T}
    callback::C
end

TimeVariableIndividual(id, x::AbstractMatrix, t::NamedTuple{(:zeta, :y)}, y, cb, ::Type{T}=Float32; kwargs...) where T = 
    TimeVariableIndividual(
        id, 
        (zeta = x, error = Matrix{eltype(x)}(undef, 0, 1)), 
        (zeta = isempty(t.zeta) ? zeros(eltype(t.zeta), 1) : t.zeta, error = eltype(x)[], y = t.y, ), 
        y, cb, T; kwargs...
    )

TimeVariableIndividual(
    id, x::NamedTuple{(:zeta,:error)}, 
    t::NamedTuple{(:zeta,:error,:y), <:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector}},
    y, cb, ::Type{T}=Float32; kwargs...) where T = 
    TimeVariableIndividual(
        id, 
        x,
        (zeta = reshape(t.zeta, 1, :), error = isempty(x.error) ? Matrix{eltype(x)}(undef, 0, 1) : reshape(t.error, 1, :), y = t.y, ), 
        y, cb, T; kwargs...
    )


function TimeVariableIndividual(
    id::I, x::NamedTuple{(:zeta,:error)}, 
    t::NamedTuple{(:zeta,:error,:y), <:Tuple{<:AbstractMatrix,<:AbstractMatrix,<:AbstractVector}}, 
    y::AbstractVector, cb::C, ::Type{T}=Float32; 
    initial::AbstractVector=empty(y)) where {T,I,C}

    length(t.zeta) !== size(x.zeta, 2) && throw(ErrorException("Length of time points vector does not match length of covariates used in the PK model."))
    size(x.error, 2) !== size(t.error, 2) && throw(ErrorException("Length of time points vector does not match length of covariates used in the error model."))
    length(t.y) !== length(y) && throw(ErrorException("Length of time points vector does not match length of observations."))
    _callback_type_matches(cb, T)
    return TimeVariableIndividual{T,I,C}(
        id, 
        map(Base.Fix1(convert, Matrix{T}), x),
        map(Base.Fix1(broadcast, T), t),
        map(Base.Fix1(convert, Vector{T}), (y, initial))..., 
        cb
    )
end

_to_timevariable(indv::BasicIndividual) = TimeVariableIndividual(
        indv.id,
        map(Base.Fix2(reshape, (Colon(), 1)), indv.x),
        (zeta = zeros(eltype(indv.t), 1, 1), error = zeros(eltype(indv.t), isempty(indv.x.error) ? 0 : 1, 1), y = indv.t),
        indv.y,
        indv.initial,
        indv.callback
    )

_to_timevariable(indv::TimeVariableIndividual) = indv

get_t(indv::TimeVariableIndividual) = indv.t.y
get_tx(indv::TimeVariableIndividual, key::Symbol=:zeta) = indv.t[key]

"""
```julia
Individual(id, x, t, y, callback, ::Type{T}=Float32; kwargs...)
```

Constructor to create Individuals. This is the primary object holding data in 
DeepCompartmentModels.jl. Each Individual contains information on the covariates 
`x`, time points of interest `t`, and the observations `y`. Individuals also 
contain a callback that stores information on the relevant clinical 
interventions. This function recognizes the appropriate subtype based on its 
arguments.

## Arguments

  - `id <: Union{Integer,AbstractString}`: the id linked to this subject.
  - `x`: Covariate values.
  - `t`: Relevant time points.
  - `y`: Observations.
  - `callback`: Clinical interventions used in the differential equation.

## Keyword Arguments
  
  - `initial`: Initial values to use for the differential equation.

### Examples

If a `Vector` is provided for the covariates, time points, and observations, 
    construct a BasicIndividual.

```jldoctest
julia> Individual("test", rand(3), rand(4), rand(4), CallbackSet())
BasicIndividual{Float32}(id = "test", ...)
```
"""
Individual(id::I, x, t, y, cb::C, ::Type{T}=Float32; kwargs...) where {T,I,C} = 
    _select_indv_type(t, y)(id, x, t, y, cb, T; kwargs...)


################################################################################
##########                          Helpers                           ##########
################################################################################

_select_indv_type(::Any, ::Any) = 
    throw(ErrorException("Could not identify `Individual` sub-type based on arguments. It is possible that the passed data is not yet supported, please reach out for support with this error."))

_select_indv_type(::AbstractVector, ::AbstractVector) = 
    BasicIndividual

_select_indv_type(::NamedTuple, ::AbstractVector) = 
    TimeVariableIndividual

function _callback_type_matches(cb::DiscreteCallback, T)
    affect!_args = map(Base.Fix1(getproperty, cb.affect!), fieldnames(typeof(cb.affect!)))[2:end]
    if !(eltype(cb.condition.times) == T) || !all(map(==(T) ∘ eltype, affect!_args))
        @warn "Types used in the callback function do not match Individual type. This negatively affects performance. Make sure to call the callback generation function with the $T type as the last argument."
        return false
    end
    
    return true
end

function _callback_type_matches(cb::CallbackSet, T) 
    for callback in cb.discrete_callbacks
        matches = _callback_type_matches(callback, T)
        if !matches
            return false
        end
    end

    return true
end
