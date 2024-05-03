################################################################################
##########                                                            ##########
##########                       Normalize layer                      ##########
##########                                                            ##########
################################################################################

"""
    Normalize(lb, ub)

Performs min-max scaling of the input according to `(x - lb) / (ub - lb)`. The 
length of `lb` and `ub` should match the input vector.
# Arguments:
- `lb`: lower bound, default = zero(ub).
- `ub`: upper bound.
"""
struct Normalize <: Lux.AbstractExplicitLayer
    lb::Vector{Float32}
    ub::Vector{Float32}
    Normalize(lb::T, ub::T) where T<:Vector{<:Real} = new(Float32.(lb), Float32.(ub))
    Normalize(ub::T) where T<:Vector{<:Real} = new(zeros(Float32, length(ub)), Float32.(ub))
    Normalize(lb::Real, ub::Real) = new(Float32[lb], Float32[ub])
    Normalize(ub::Real) = new(Float32[ub])
end

Lux.initialparameters(::Random.AbstractRNG, ::Normalize) = NamedTuple()
Lux.initialstates(::Random.AbstractRNG, l::Normalize) = (lb = l.lb, ub = l.ub)

Lux.parameterlength(::Normalize) = 0
Lux.statelength(l::Normalize) = 2 * length(l.ub)

function (l::Normalize)(x::AbstractArray, ps, st::NamedTuple)
    y = (x .- st.lb) ./ (st.ub - st.lb)
    return y, st
end

################################################################################
##########                                                            ##########
##########                 Add global parameter layer                 ##########
##########                                                            ##########
################################################################################


"""
    AddGlobalParameters(out_dim, loc; init_theta, activation)

Adds learnable global parameters `θ` to the input vector at indexes `loc` to 
create a vector of size `out_dim`.

# Arguments:
- `out_dim::Int`: Length of the resulting output vector.
- `loc::Int`: Indexes of θ in the output vector.
- `init_theta`: Initialization function for `θ` from WeightInitializers.jl. Default = glorot_uniform. 
- `activation`: Activation function to use on `θ`. Default = softplus.

# Examples
```jldoctest
julia> layer = AddGlobalParameters(4, [2, 4])
AddGlobalParameters()  # 2 parameters, plus 16 non-trainable
 
julia> layer([1; 2;;], ps, st)[1] # returns y, st
4×1 Matrix{Float32}:
 1.0
 θ₁
 2.0
 θ₂
```
"""
struct AddGlobalParameters{T, F1, F2} <: Lux.AbstractExplicitLayer
    theta_dim::Int
    out_dim::Int
    locations::Vector{Int}
    init_theta::F1
    activation::F2
end

AddGlobalParameters(out_dim, loc, T=Float32; init_theta=Lux.glorot_uniform, activation=softplus) = AddGlobalParameters{T, typeof(init_theta), typeof(activation)}(length(loc), out_dim, loc, init_theta, activation)

Lux.initialparameters(rng::Random.AbstractRNG, l::AddGlobalParameters) = (theta = l.init_theta(rng, l.theta_dim, 1),)
Lux.initialstates(::Random.AbstractRNG, l::AddGlobalParameters{T,F1,F2}) where {T,F1,F2} = (indicator_theta = indicator(l.out_dim, l.locations, T), indicator_x = indicator(l.out_dim, filter(!∈(l.locations), 1:l.out_dim), T))
Lux.parameterlength(l::AddGlobalParameters) = l.theta_dim
Lux.statelength(l::AddGlobalParameters) = 2 * l.theta_dim * l.out_dim

# the indicators should be in the state!
function (l::AddGlobalParameters)(x::AbstractMatrix, ps, st::NamedTuple)
    if size(st.indicator_x, 2) !== size(x, 1)
        indicator_x = st.indicator_x * st.indicator_x' # Or we simply do not do this, the one might already be in the correct place following the combine function.
    else
        indicator_x = st.indicator_x
    end
    y = indicator_x * x + st.indicator_theta * repeat(l.activation.(ps.theta), 1, size(x, 2))
    return y, st
end


################################################################################
##########                                                            ##########
##########                  Combine parameters layer                  ##########
##########                                                            ##########
################################################################################

"""
    Combine([out_dim], pairs...)

Connects specific branches to specific outputs in a Lux model containing a BranchLayer. 
Output from branches pointing to the same output are combined using a product.

# Arguments:
- `pairs...`: Pairs denoting what branch to connected to what output(s). Pairs take the following format: i => [j] or [j, k, ...]

# Examples
```jldoctest
julia> layer = Combine(1 => [1], 2 => [1], 3 => [2]) # Connects branch 1 to output 1, branch 2 to output 1, etc.
Combine()
```
"""
struct Combine{T1, T2} <: Lux.AbstractExplicitLayer
    out_dim::Int
    pairs::T2
end

function Combine(pairs::Vararg{Pair{Int64, Vector{Int64}}}; T=Float32)
    out_dim = maximum([maximum(pairs[i].second) for i in eachindex(pairs)])
    return Combine{T, typeof(pairs)}(out_dim, pairs)
end

function get_state(l::Combine{T1, T2}) where {T1, T2}
    indicators = Vector{Matrix{T1}}(undef, length(l.pairs))
    negatives = Vector{Vector{T1}}(undef, length(l.pairs))
    for pair in l.pairs
        Iₛ = indicator(l.out_dim, pair.second, T1)
        indicators[pair.first] = Iₛ
        negatives[pair.first] = abs.(vec(sum(Iₛ, dims=2)) .- one(T1))
    end
    return (indicators = indicators, negatives = negatives)
end

Lux.initialparameters(::Random.AbstractRNG, ::Combine) = NamedTuple()
Lux.initialstates(::Random.AbstractRNG, l::Combine) = get_state(l)
Lux.parameterlength(::Combine) = 0
Lux.statelength(l::Combine) = 2 * l.out_dim * length(l.pairs)

function (l::Combine)(x::Tuple, ps, st::NamedTuple) 
    indicators = st.indicators
    negatives = st.negatives
    y = reduce(.*, _combine.([x...], indicators, negatives))
    return y, st
end

_combine(x::AbstractMatrix, indicator::AbstractMatrix, negative::AbstractVector) = indicator * x .+ negative

################################################################################
##########                                                            ##########
##########                    Interpretable layers                    ##########
##########                                                            ##########
################################################################################

"""
    SingleHeadedBranch(covariate_idx, neurons; activation, init_bias).

Convenience function for creating branches in Lux models. Constructs a single 
hidden layer neural network with the specified number of `neurons`. It is advisable
to use at most 2 covariates as input to the model to facilitate interpretation.

# Arguments:
- `covariate_idx::Union{AbstractVector{<:Int}, Int}`: Index of the covariate(s) to use in the branch.
- `neurons::Int`: The amount of neurons in the hidden layer.
- `activation`: Activation function to use in the hidden layer. Default = swish.
- `init_bias`: Initialization function of the bias parameters. Default = ones32. Can help with improving the initial estimates.
"""
function SingleHeadedBranch(covariate_idx::Union{AbstractVector{<:Int}, Int}, neurons::Int; activation = swish, init_bias = Lux.ones32) 
    return make_branch(
        covariate_idx, 
        Lux.Dense(1, neurons, activation), 
        Lux.Dense(neurons, 1, softplus, init_bias=init_bias)
    )
end

# TODO: There is a problem when predicting in a multi-branch model when passing a vector (i.e. when working with a single individual)

"""
    MultiHeadedBranch(covariate_idx, neurons, heads; activation, init_bias).

Convenience function for creating branches in Lux models. Constructs a neural 
network containing a single hidden layer connecting to multiple independent 
layers in parallel specified by `heads`. This way, the covariate effect shares a 
similar base, but can still learn independent effects for each head. All hidden 
layers contain the same number of `neurons`. It is advisable to use at most 2 
covariates as input to the model to facilitate interpretation.

# Arguments:
- `covariate_idx::Union{AbstractVector{<:Int}, Int}`: Index of the covariate(s) to use in the branch.
- `neurons::Int`: The amount of neurons in the hidden layer.
- `heads::Int`: The number of independent heads.
- `activation`: Activation function to use in the hidden layer. Default = swish.
- `init_bias`: Initialization function of the bias parameters. Default = ones32. Can help with improving the initial estimates.
"""
function MultiHeadedBranch(covariate_idx::Union{AbstractVector{<:Int}, Int}, neurons::Int, heads::Int; activation = swish, init_bias = Lux.ones32)
    if heads == 1
        head = Lux.Dense(neurons, 1, softplus, init_bias=init_bias)
    else
        head = Lux.Parallel(vcat, 
            [Lux.Dense(neurons, 1, Lux.softplus, init_bias=init_bias) for _ in 1:heads]...
        )        
    end
    return make_branch(
        covariate_idx, 
        Lux.Dense(1, neurons, activation), 
        head
    )
end

"""
    make_branch(covariate_idx, layers...)

Internal function used to create branches. Allows one to create more specific branch layers.
Appends the `layers` to the following Lux.Chain:

```julia
Chain(
    SelectDim(1, covariate_idx),
    ReshapeLayer((1,)),
    layers...
)
```

# Arguments:
- `covariate_idx::Union{AbstractVector{<:Int}, Int}`: Index of the covariate(s) to use in the branch.
- `layers`: Layers used to build the neural network.
"""
function make_branch(covariate_idx::Union{AbstractVector{<:Int}, Int}, layers...)
    return Lux.Chain(
        Lux.SelectDim(1, covariate_idx),
        Lux.ReshapeLayer((1,)),
        layers...
    )
end


"""
    interpret_branch(model, covariate_idx, anchor; x)

Convenience function for running the interpretation function on models following 
the AbstractModel interface.

# Arguments:
- `model::AbstractModel`: AbstractModel (e.g. DeepCompartmentModel(...)) using an Multi-branch based neural network.
- `covariate_idx`: Index of the covariate in the input vector.
- `anchor`: Unnormalized covariate value to which the output is "anchored", i.e. f(anchor) = 1.
- `x`: Normalized dummy input to the branch. 
"""
interpret_branch(model::AbstractModel, covariate_idx, anchor; kwargs...) = interpret_branch(model.ann, model.p.weights, model.p.st, covariate_idx, anchor; kwargs...)

"""
    interpret_branch(model, covariate_idx, anchor, ps, st; x)

Returns the output of a specific branch based on dummy input. Can be used to 
interpret covariate effects. Returns the unnormalized `x` and the branch output 
`y` divided by the prediction at the location of `anchor`:

```julia
y = f(x) ./ f(anchor)
```

# Arguments:
- `ann`: Lux model.
- `ps`: Model parameters.
- `st`: Model state.
- `covariate_idx`: Index of the covariate in the input vector.
- `anchor`: Unnormalized covariate value to which the output is "anchored", i.e. f(anchor) = 1.
- `x`: Normalized dummy input to the branch. 
"""
function interpret_branch(ann::Lux.AbstractExplicitContainerLayer, ps, st, covariate_idx, anchor; x = 0:0.01:1)
    branch_layer = findall(layer -> typeof(layer) <: BranchLayer, ann.layers)
    if length(branch_layer) > 1
        throw(ErrorException("Multiple BranchLayers detected. The interpret_branch function only accepts model with a single BranchLayer"))
    end

    norm_layer = findfirst(layer -> typeof(layer) <: Normalize, ann.layers)
    if norm_layer == nothing
        throw(ErrorException("No Normalize layer found in this model."))
    end
    norm = ann.layers[norm_layer]
    norm_anchor = (anchor - norm.lb[covariate_idx]) / (norm.ub[covariate_idx] - norm.lb[covariate_idx])
    x_unnorm = x .* (norm.ub[covariate_idx] - norm.lb[covariate_idx]) .+ norm.lb[covariate_idx]
    
    branch = ann.layers[branch_layer][1].layers[covariate_idx]
    dummy = transpose(repeat(x, 1, covariate_idx))
    ps_ = ps[branch_layer][1][covariate_idx]
    st_ = st[branch_layer][1][covariate_idx]

    y, _ = branch(dummy, ps_, st_)
    y_norm, _ = branch(fill(norm_anchor, covariate_idx, 1), ps_, st_)
    return (x_unnorm, y ./ y_norm)
end
