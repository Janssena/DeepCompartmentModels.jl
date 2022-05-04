import Zygote: ignore

struct Population
    x::Matrix{Float64}
    y::Vector{Vector{Float64}}
    t::Vector{Vector{Float64}}
    callbacks::Vector{DiscreteCallback}
    scale_x::Tuple # It might make more sense to store this in the DCM as well?
    u0::Vector{Float64} # initial concentration
end


"""Constructer that does not require one to pass u0."""
Population(x, y, t, callbacks, scale_x) = 
    Population(x, y, t, callbacks, scale_x, zeros(length(y)))

Base.length(pop::Population) = length(pop.y)


struct Individual
    x::Vector{Float64}
    y::Vector{Float64}
    t::Vector{Float64}
    callback::DiscreteCallback
    u0::Float64
end

"""Indexing for getting Individuals from a Population. Uses Zygote.ignore to prevent indexing issues."""
function (pop::Population)(i::Int64)
    ignore() do
        return Individual(pop.x[i, :], pop.y[i], pop.t[i], pop.callbacks[i], pop.u0[i])
    end
end

(pop::Population)(idxs::Vector{Int64}) =
    Population(pop.x[idxs, :], pop.y[idxs], pop.t[idxs], pop.callbacks[idxs], pop.scale_x, pop.u0[idxs])

(pop::Population)(range::UnitRange{Int64}) =
        Population(pop.x[range, :], pop.y[range], pop.t[range], pop.callbacks[range], pop.scale_x, pop.u0[range])

(pop::Population)(bitvector::BitVector) = pop(collect(1:length(pop))[bitvector])

# Syntactic sugar allowing the use of population[index] and population[1:n]
Base.getindex(pop::Population, indexes) = pop(indexes)
Base.lastindex(pop::Population) = length(pop.y)

# Looping through the individuals would also be cool.
"""Clean printing of Populations and Individuals"""
Base.show(io::IO, pop::Population) = print(io, "Population(...) containing $(length(pop.y)) Individuals")
Base.show(io::IO, indv::Individual) = print(io, "Individual(...)") # Maybe nice to have origin or like #36 out of 197