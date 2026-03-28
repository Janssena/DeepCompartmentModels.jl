
# TODO: Make Population hold a tuple, so that we can mix different individual types
# TODO: Make this a standalone package to isolate development?
"""
    Population{T<:AbstractIndividual} <: AbstractArray{T, 1}

Array holding a collection of Individuals that have the same type.
"""
struct Population{T<:AbstractIndividual} <: AbstractArray{T, 1}
    data::Vector{T}
    count::Int
end

"""
    Population(data::AbstractVector{AbstractIndividual})

Constructor to create a Population. Automatically detects the Individual types 
and attemps to harmonize them if multiple types are passed.

## Arguments

  - `data <: AbstractVector{AbstractIndividual}`: A vector containing the Individuals in the population.
"""
function Population(data::AbstractVector{T}) where {T<:AbstractIndividual} 
    types = typeof.(data)
    if length(unique(types)) !== 1
        # type names are different -> make all TimeVariable
        if length(unique(map(Base.Fix2(getproperty, :name), types))) !== 1
            if any(isa.(data, TimeVariableIndividual))
                @info "Detected a mix of Individual types in `data`. Changed all Individuals to TimeVariableIndividuals."
                data = _to_timevariable.(data)
            end
        end
        # TODO: need to check if resulting types is a Union.
        # Occassions might also be a problem when a mix.

        # Parameter types are different
        if length(unique(map(Base.Fix2(getproperty, :parameters), types))) !== 1
            throw(ErrorException("Parametric types of Individuals do not match. Make sure that the ids, Number type, and callbacks are all of the same type."))
        end
    end

    new_type = only(unique(typeof.(data)))
    return Population{new_type}(Vector{new_type}(data), length(data))
end

# Population(data::AbstractVector{T}) where T<:Union{BasicIndividual, TimeVariableIndividual} = 
#     Population{T}(data, length(data))

"""Initializing an empty Population."""
Population(::Type{T}, dims::Dims) where {T<:AbstractIndividual} = 
    Population{T}(Vector{T}(undef, dims), only(dims))

Base.IndexStyle(::Type{<:Population}) = IndexLinear()
Base.size(pop::Population) = (pop.count, )
Base.similar(::Population, ::Type{T}, dims::Dims) where {T} = Population(T, dims)
Base.getindex(pop::Population, idx::Int) = getindex(pop.data, idx) # No default
Base.setindex!(pop::Population{T}, v::T, idx::Int) where {T} = (pop.data[idx] = v)
Base.showarg(io::IO, ::Population{T}, toplevel) where T = print(io, "Population{$(nameof(T)){$(T.parameters[1])}}")

function get_x(pop::Population{T}, key::Symbol=:zeta) where T<:BasicIndividual
    x = zeros(first(T.parameters), length(get_x(pop[1], key)), pop.count)
    for i in eachindex(pop)
        x[:, i] .= get_x(pop[i], key)
    end
    return x
end

get_x(pop::Population{T}, key::Symbol=:zeta) where T<:TimeVariableIndividual = [get_x(indv, key) for indv in pop]
get_t(pop::Population) = [get_t(indv) for indv in pop]
get_y(pop::Population) = [get_y(indv) for indv in pop]

@non_differentiable get_x(::Population, ::Symbol)
@non_differentiable get_y(::Population)

load() = nothing # Is implemented by extensions
