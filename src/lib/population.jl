abstract type AbstractIndividual{I,X,T,Y,C} end

Base.show(io::IO, indv::I) where {I<:AbstractIndividual} = print(io, "$(I.name.name){id = $(indv.id), ...)")

# TODO: write docs
"""
    BasicIndividual(...)

Struct holding the data for a single subject for standard PK or PD analyses.
"""
struct BasicIndividual{I<:Union{Integer, AbstractString}, X, T, Y<:AbstractVector, C} <: AbstractIndividual{I,X,T,Y,C}
    id::I
    x::X # Vector/Matrix or NamedTuple with multi-component setup, e.g. (cov = ..., error = ...)
    t::T # Vector or NamedTuple
    y::Y # Vector
    callback::C
    initial::Y
    eta::Y
end
# Constructors, TODO: set common type (default = Float32) -> how to do this for the callback?
"""
    BasicIndividual(x, t, y, callback; id, initial, eta)

# Arguments
- `x`: Subject specific covariates. If Matrix is passed, predictions can change over time.
- `t`: Time points of observations. If the subject is time-variable, a NamedTuple (x = [...], y = [...]) should be passed.
- `y::AbstractVector`: Observations. Must be a vector. Only supports a single DV.
- `callback`: Differential equation callback containing treatment interventions.
- `id`: Patient id to store in the Individual instance.
- `initial`: Initial value of the dependent value at t = 0. Default = [].
- `eta`: Random effect estimate for the individual. Default = [].
"""
function BasicIndividual(x::X, t::T, y::Y, callback::C; initial=empty(y), eta=empty(y), id::I = "") where {I,X,T,Y,C} 
    return BasicIndividual{I,X,T,Y,C}(id, x, t, y, callback, initial, eta)
end

"""
    Individual(...)

Alias for constructing a BasicIndividual.
"""
Individual(args...; kwargs...) = BasicIndividual(args...; kwargs...)

"""
    is_timevariable(individual)

Returns whether the individual has time variable effects.
"""
is_timevariable(indv::AbstractIndividual) = is_timevariable(typeof(indv))
is_timevariable(::Type{<:AbstractIndividual{I,X,T,Y,C}}) where {I,X,T,Y,C} = X <: AbstractMatrix && T <: NamedTuple && :x in fieldnames(T)

function make_timevariable(indv::I) where I<:AbstractIndividual
    if is_timevariable(indv) return indv end

    res = NamedTuple()
    for field in fieldnames(I)
        property = getproperty(indv, field)
        if field == :x
            adjusted_property = reshape(property, length(property), 1)
        elseif field == :t
            adjusted_property = (x = zero.(property[1:1]), y = property,)
        else
            adjusted_property = property
        end
        res = merge(res, [field => adjusted_property])
    end

    return Base.typename(I).wrapper(;res...)
end

struct Static end
struct TimeVariable end

"""
    Population(AbstractIndividual[...])

Combines a vector of individuals into a Population. Makes sure all the 
Individuals are of the same type. If any subject has time-dependent effects, all 
Individuals are transformed to the time-variable format.
"""
struct Population{T,I<:AbstractIndividual} <: AbstractArray{I, 1}
    indvs::Vector{I}
    count::Int
    # Constructor
    function Population(indvs::AbstractVector{<:AbstractIndividual})
        type = Static()
        timevar_idxs = is_timevariable.(indvs)
        if any(timevar_idxs)
            type = TimeVariable()
            reference = indvs[findfirst(isequal(1), timevar_idxs)]
            indvs_ = map(make_timevariable, indvs)
        else
            reference = indvs[1]
            indvs_ = convert(Vector{typeof(reference)}, indvs)
        end
        return new{typeof(type), typeof(reference)}(indvs_, length(indvs_))
    end
end

Base.showarg(io::IO, ::Population{T,I}, toplevel) where {T,I} = print(io, "Population{$(T.name.name), $(I.name.name)}")

Base.size(pop::Population) = (pop.count,)
Base.IndexStyle(::Type{<:Population}) = IndexLinear()
Base.getindex(pop::Population, i::Int) = pop.indvs[i]
Base.broadcastable(pop::Population{T,I}) where {T,I<:AbstractIndividual} = pop.indvs
function Base.getproperty(pop::Population{T1,<:AbstractIndividual{I,X,T,Y,C}}, f::Symbol) where {T1,I,X,T,Y,C}
    error_text = "Directly getting $f when it is not an AbstractVector is not implemented. Try `getfield.(population, $f)` instead."
    if f == :x
        xs = getfield.(pop, :x)
        return X <: AbstractVector ? stack(xs) : xs
    elseif f == :y 
        return Y <: AbstractVector ? getfield.(pop, :y) : throw(ErrorException(error_text))
    elseif f == :t
        return T2 <: AbstractVector ? getfield.(pop, :t) : throw(ErrorException(error_text))
    else
        return getfield(pop, f)
    end
end