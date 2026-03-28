##### AbstractErrorModel setup

setup(::ImplicitError, ::Any) = nothing

function setup(error::AbstractErrorModel, ::Nothing) 
    if isempty(error.init)
        throw(ErrorException("No explicit initialisation passed for the error parameters. Either recreate the ErrorModel using a suitable `init` argument or pass initialisation through the `init_sigma` keyword of the `setup` function."))
    end
    return (σ = invsoftplus.(error.init), )
end

function setup(error::ErrorModelSet, ::Nothing) 
    if any(isempty.(getproperty.(error.errors, :init)))
        throw(ErrorException("No explicit initialisation passed for all error parameters. Either recreate the ErrorModels using a suitable `init` argument or pass initialisation through the `init_sigma` keyword of the `setup` function."))
    end

    return (σ = [invsoftplus.(e.init) for e in error.errors], )
end

setup(::AbstractErrorModel, init::AbstractVector{<:Real}) = (σ = invsoftplus.(init), )
setup(::ErrorModelSet, inits::AbstractVector{<:AbstractVector{<:Real}}) = (σ = map(Base.Fix1(broadcast, invsoftplus), inits), )

##### Full setup
setup(rng, dcm::DeepCompartmentModel{<:SciMLBase.AbstractDEProblem}) = 
    Lux.setup(rng, dcm.model)

function setup(rng, dcm::DeepCompartmentModel{<:UniversalDiffEq{P,T}}) where {P,T}
    ps, st = Lux.setup(rng, dcm.model)
    return ComponentVector(merge(ps, (I = 0, ))), st
end

"""
    setup(obj::FixedObjective, rng, model, ::Type{T}=Float32; init_sigma, params)

Setup function that initialises model parameters and state.

# Arguments
- `obj::FixedObjective`: Fixed effect based objective function that is used to optimise the model.
- `rng`: Random number generator.
- `model`: Model for which to initialise parameters.
- `T`: Type to use. Default = Float32.

# Keyword arguments
- `init_sigma`: Initial values for the error model. Default = nothing, using the initial parameters in the ErrorModel.
- `params`: Parameterisation to use. Should be one of [MeanSqrt(), MeanVar()]. Default = MeanSqrt().
"""
function setup(::FixedObjective, rng::Random.AbstractRNG, model::AbstractModel, ::Type{T}=Float32; init_sigma = nothing, params::Parameterisation=MeanSqrt()) where T
    ps_theta, st_theta = setup(rng, model)

    ps = (
        theta = ps_theta, 
    )
    if !(model.error isa ImplicitError)
        ps_error = setup(model.error, init_sigma)
        merge(ps, (error = ps_error, ))
    end
    ps_params = _convert_parameters(params, ps)
    st = (theta = st_theta, )
    type_fn = _get_type_fn(T)
    return (ps_params, st) |> type_fn
end

_get_type_fn(T) = T == Float64 ? Lux.f64 : (T == Float32 ? Lux.f32 : Lux.f16)

"""
    setup(obj::MixedObjective, rng, model, population, ::Type{T}=Float32; init_sigma, params)

Setup function that initialises model parameters and state.

# Arguments
- `obj::MixedObjective`: Mixed effect based objective function that is used to optimise the model.
- `rng`: Random number generator.
- `model`: Model for which to initialise parameters.
- `population`: Data used for fitting the model. Used to initialise the number of etas.
- `T`: Type to use. Default = Float32.

# Keyword arguments
- `init_omega`: Initial variance(s) of the diagonals of the omega matrix. Can be either a Real (all random effects have same initial variance) or Vector matching the number of random effects. Default = 0.1.
- `init_sigma`: Initial values for the error model. Default = nothing, which uses the initial value(s) specified in the ErrorModel.
- `params`: Parameterisation to use. Should be one of [MeanSqrt(), MeanVar()]. Default = MeanSqrt().
- `scale`: Optional parameter used to scale the intial estimates of some parameters. For example used when obj = VariationalELBO to scale the initial estimates of the variance of the Variational posteriors.
"""
function setup(
    obj::MixedObjective, rng::Random.AbstractRNG, model::AbstractModel, population::Population, ::Type{T}=Float32; 
    init_omega=0.1, init_sigma = nothing, params::Parameterisation=MeanSqrt(), kwargs...) where T

    ps_theta, st_theta = setup(rng, model)
    Ω = Symmetric(collect(Diagonal(ones(_num_random_effects(obj)) .* init_omega)))
    num_params = _estimate_typ_parameter_size(model, population[1:1], (theta = ps_theta, ), (theta = st_theta, ))
    ps_phi, st_phi = setup_phi(obj, rng, model, population, Ω; num_params, kwargs...)

    ps = (
        theta = ps_theta, 
        error = setup(model.error, init_sigma),
        omega = Ω,
        phi = ps_phi
    )
    st = (
        theta = st_theta, 
        phi = st_phi
    )
    ps_params = _convert_parameters(params, ps)
    type_fn = _get_type_fn(T)
    return (ps_params, st) |> type_fn
end

function setup_phi(obj::VariationalELBO{MF}, rng::Random.AbstractRNG, ::AbstractDEModel, population::Population, Ω::Symmetric; num_params, scale::Real=0.1) where MF
    ps = (
        μ = [zeros(size(Ω, 1)) for _ in eachindex(population)],
        Σ = [scale * copy(Ω) for _ in eachindex(population)],
    )
    if MF == True # i.e. if mean_field = true
        ps = (
            μ = ps.μ,
            σ² = diag.(ps.Σ)
        )
    end

    st = (
        mask = indicator(num_params, obj.idxs),
        epsilon = [randn(rng, size(Ω, 1)) for _ in eachindex(population)],
    )

    return ps, st
end

function indicator(n::Int, a::AbstractVector{Int}, ::Type{T}=Float32) where T
    Iₐ = zeros(T, n, length(a))
    for i in eachindex(a)
        Iₐ[a[i], i] = one(T)
    end
    return Iₐ
end

# TODO: change parameter name as well in the kp
function _convert_parameters(::MeanVar, ps::NamedTuple) 
    ps_new = fmap_with_path(ps) do kp, x
        if isnothing(x)
            return x
        elseif :L in kp
            return Symmetric(x * x')
        elseif :σ in kp && !(:error in kp)
            return softplus.(x).^2
        else
            return x
        end
    end

    return _update_keys(ps_new, :L => :Σ, :σ => :σ²)
end

function _convert_parameters(::MeanSqrt, ps::NamedTuple) 
    ps_new = fmap_with_path(ps) do kp, x
        if isnothing(x)
            return x
        elseif :Σ in kp
            return cholesky(x).L
        elseif :σ² in kp && !(:error in kp)
            return invsoftplus.(sqrt.(x))
        else
            return x
        end
    end

    # TODO: changes the names
    return _update_keys(ps_new, :Σ => :L, :σ² => :σ)
end

_update_keys(x, replacements::Pair...; kwargs...) = x

function _update_keys(nt::NamedTuple, replacements::Pair...; exclude = :error)
    old_keys = keys(nt)

    new_values = map(old_keys) do key
        if key == exclude 
            return nt[key]
        else
            return _update_keys(nt[key], replacements...; exclude)
        end
    end
    new_keys = replace(old_keys, replacements...)
    return NamedTuple{new_keys}(new_values)
end

_estimate_typ_parameter_size(m::AbstractDEModel, population::Population, ps, st) =
    _estimate_typ_parameter_size(first(predict_typ_parameters(m, population, ps, st)))

_estimate_typ_parameter_size(y::AbstractVector{<:Real}) = length(y)
_estimate_typ_parameter_size(y::AbstractMatrix{<:Real}) = size(y, 1)
_estimate_typ_parameter_size(y::AbstractVector{AbstractArray{<:Real}}) = _estimate_typ_parameter_size(first(y))
