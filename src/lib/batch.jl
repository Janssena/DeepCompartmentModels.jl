# Mini-batching of populations: assigning individuals to batches, slicing parameters and states to the
# individuals of a batch, and combining the gradients of the batches.

const DEFAULT_BATCHSIZE = 16

"""
    create_batches([rng,] population; batchsize = nothing, nbatch = nothing)
    create_batches([rng,] n::Int; batchsize = nothing, nbatch = nothing)

Randomly divide the individuals of `population` (or the indices `1:n`) into mini-batches and return a
vector of index vectors. Set at most one of `batchsize` and `nbatch`:

- `batchsize`: the maximum number of individuals per batch. The number of batches is `cld(n, batchsize)`.
- `nbatch`: the number of batches.

The individuals are divided as evenly as possible: batch sizes differ by at most one, and never exceed
`batchsize`. Values larger than the number of individuals are reduced to it. If neither keyword is
set, `batchsize = $DEFAULT_BATCHSIZE`.
"""
function create_batches(rng::Random.AbstractRNG, n::Int; batchsize::Union{Nothing,Int} = nothing, nbatch::Union{Nothing,Int} = nothing)
    M = _number_of_batches(n, batchsize, nbatch)
    batches = [collect(i:M:n) for i in 1:M]
    idxs = Random.shuffle(rng, 1:n)
    return map(Base.Fix1(getindex, idxs), batches) # randomize indexes in each batch
end

create_batches(rng::Random.AbstractRNG, population::Population; kwargs...) = create_batches(rng, length(population); kwargs...)
create_batches(x::Union{Int,Population}; kwargs...) = create_batches(Random.default_rng(), x; kwargs...)

function _number_of_batches(n::Int, batchsize::Union{Nothing,Int}, nbatch::Union{Nothing,Int})
    if !isnothing(batchsize) && !isnothing(nbatch)
        throw(ArgumentError("Set either `batchsize` or `nbatch`, not both (got batchsize = $batchsize, nbatch = $nbatch)."))
    elseif !isnothing(nbatch)
        nbatch >= 1 || throw(ArgumentError("`nbatch` should be at least 1, got $nbatch."))
        return min(nbatch, max(n, 1))
    else
        batchsize = something(batchsize, DEFAULT_BATCHSIZE)
        batchsize >= 1 || throw(ArgumentError("`batchsize` should be at least 1, got $batchsize."))
        return cld(max(n, 1), min(batchsize, max(n, 1)))
    end
end

take_batch(ps::NamedTuple, st::NamedTuple, batch; kwargs...) = take_batch(ps, batch; kwargs...), take_batch(st, batch; kwargs...)

take_batch(x, ::Any; kwargs...) = x
take_batch(x::AbstractVector{<:AbstractArray{<:Real}}, i; kwargs...) = x[i]
take_batch(x::AbstractVector{<:AbstractVector{<:AbstractArray{<:Real}}}, i; kwargs...) = x[i]
take_batch(x::AbstractVector{<:Cholesky}, i; kwargs...) = x[i]
take_batch(x::AbstractVector{<:AbstractVector{<:Cholesky}}, i; kwargs...) = x[i]
function take_batch(x::NamedTuple, i; exclude::Union{Nothing, Symbol} = :error)
    keys_ = keys(x)
    values_ = map(keys_) do key # Almost a fmap, but does not recurse into indexes
        if !isnothing(exclude) && key == exclude
            return x[key]
        else
            return take_batch(x[key], i; exclude)
        end
    end
    return NamedTuple{keys_}(values_)
end

# Combine the gradients of individuals or batches into the gradient of the population.
_sum_grads(::Vararg{Nothing}) = nothing
_sum_grads(xs::Vararg) = +(filter(!isnothing, xs)...)
_isleaf_or_vec_of_arrays(::KeyPath, x) = _isleaf_or_vec_of_arrays(x)
_isleaf_or_vec_of_arrays(x) = Functors.isleaf(x)
_isleaf_or_vec_of_arrays(x::AbstractVector{<:AbstractArray{<:Real}}) = true

_sum_gradients(grads) = fmap(_sum_grads, grads...; exclude = _isleaf_or_vec_of_arrays)

# The gradients of the batches (or individuals) are summed to obtain the gradient of the full
# population. That is only correct if the objective is a sum over individuals (SSE, LogLikelihood).
# MSE is a mean over individuals, so the gradient of a batch has to be weighted by its share of the
# population. Without this the summed gradient is `length(population) / length(batch)` times too large.
_batch_weight(::FixedObjective, batch_size::Int, population_size::Int) = true
_batch_weight(::MSE, batch_size::Int, population_size::Int) = batch_size / population_size
