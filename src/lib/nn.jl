"""
    StandardNeuralNetwork(...)

Standard neural network architecture that directly predicts the observations `y`
based on covariates `x`, time point `t`, and dose `d`.
"""
struct StandardNeuralNetwork{T,M<:Lux.AbstractLuxLayer,E<:AbstractErrorModel} <: AbstractModel{T,M,E}
    model::M
    error::E
    StandardNeuralNetwork(model::M, error::E, ::Type{T}=Float32) where {T,M,E} = 
    new{T,M,E}(model, error)
end

StandardNeuralNetwork(model, ::Type{T}=Float32) where T = 
    StandardNeuralNetwork(model, ImplicitError(), T)

"""
    SNN(...)

Alias for StandardNeuralNetwork.
"""
SNN(args...; kwargs...) = StandardNeuralNetwork(args...; kwargs...)

Base.show(io::IO, dcm::StandardNeuralNetwork{T,M,E}) where {T,M,E} = print(io, "StandardNeuralNetwork{$T}")

################################################################################
##########                        Model API                           ##########
################################################################################

function predict(m::StandardNeuralNetwork, data, ps, st)
    X = _stack_x_and_t(data)
    ŷ, st = m.model(X, ps.theta, st.theta)
    return _reshape_to_y(ŷ, data), st
end

_stack_x_and_t(population::Population) = 
    reduce(hcat, _stack_x_and_t.(population))

_stack_x_and_t(indv::AbstractIndividual; saveat = indv.t) = 
    vcat(repeat(indv.x, 1, length(saveat)), transpose(saveat))

_reshape_to_y(ŷ::AbstractMatrix, data) = _reshape_to_y(ŷ[1, :], data)

# This is somewhat wastefull, but helps with LogLikelihood evaluation
function _reshape_to_y(ŷ::AbstractVector{<:Real}, population::Population) 
    lengths = length.(get_y(population))
    cumsum_lengths = cumsum(lengths)
    idxs = [i == 1 ? (1:cumsum_lengths[i]) : (cumsum_lengths[i-1]+1:cumsum_lengths[i]) for i in eachindex(cumsum_lengths)]
    return getindex.((ŷ, ), idxs)
end

##### USING USER SUPPLIED TIMEPOINTS:

function predict(m::StandardNeuralNetwork, data, t::AbstractVector{<:Real}, ps, st)
    X = _stack_x_and_t(data, t)
    ŷ, st = m.model(X, ps.theta, st.theta)
    return _reshape_to_t(ŷ, data, t), st
end

_reshape_to_y(ŷ::AbstractVector{<:Real}, ::AbstractIndividual) = ŷ

_stack_x_and_t(population::Population, t::AbstractVector) = 
    reduce(hcat, _stack_x_and_t.(population; saveat = t))

_stack_x_and_t(individual::AbstractIndividual, t::AbstractVector) = 
    _stack_x_and_t(individual; saveat = t)

_reshape_to_t(ŷ::AbstractMatrix, data, t) = _reshape_to_t(ŷ[1, :], data, t)

_reshape_to_t(ŷ::AbstractVector{<:Real}, population::Population, t) =
    collect(eachcol(reshape(ŷ, length(t), length(population))))

_reshape_to_t(ŷ::AbstractVector{<:Real}, ::AbstractIndividual, t) = ŷ
