import InteractiveUtils: @code_lowered

"""
    DeepCompartmentModel{T,D,M,S}

Model architecture originally described in [janssen2022]. Originally uses a 
neural network to learn the relationship between the covariates and the 
parameters of a system of differential equations, which for example represents 
a compartment model.

# Arguments
- `problem::AbstractDEProblem`: DE problem describing the dynamical system.
- `model`: Model to use for the prediction of DE parameters. The package focusses on the use of neural networks based on Lux.jl.
- `T::Type`: DataType of parameters and predictions.
- `dv_compartment::Int`: The index of the compartment for the prediction of the dependent variable. Default = 1.
- `sensealg`: Sensitivity algorithm to use for the calculation of gradients of the parameters with respect to the DESolution.

\\
[janssen2022] Janssen, Alexander, et al. "Deep compartment models: a deep learning approach for the reliable prediction of time‐series data in pharmacokinetic modeling." CPT: Pharmacometrics & Systems Pharmacology 11.7 (2022): 934-945.
"""
struct DeepCompartmentModel{T,P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer,E,S} <: AbstractDEModel{T,P,M,E,S}
    problem::P
    model::M
    dv_compartment::Int
    error::E
    sensealg::S
    
    function DeepCompartmentModel(
        de::D, 
        model::M,
        error::E,
        ::Type{T} = Float32; 
        dv_compartment::Int = 1, 
        sensealg::S = ForwardDiffSensitivity(; convert_tspan = true)
    ) where {T,D<:SciMLBase.AbstractDEProblem,M,E,S}
        problem = _rebuild_problem_type(de, T)
        return new{T,typeof(problem),M,E,S}(problem, model, dv_compartment, error, sensealg)
    end
end

# TODO: this likely is not sufficient for all DEProblems
_rebuild_problem_type(de::SciMLBase.AbstractDEProblem, T) = 
    (Base.typename(typeof(de)).wrapper)(de.f, T.(de.u0), T.(de.tspan), T[])
    
# Constructors. 
"""
    DeepCompartmentModel(ode_fn, model, error, T=Float32; kwargs...)

Convenience constructor that internally creates an ODEProblem based on the 
passed ode_fn. Attempts to estimate the number of partial differential equations
present in the ode_fn.

# Arguments
- `ode_fn::Function`: Function that describes the dynamical system.
- `model`: Model to use for the prediction of DE parameters. The package focusses on the use of neural networks based on Lux.jl.
- `error::AbstractErrorModel`: Error model to use. Should be one of [ImplicitError, AdditiveError, ProportionalError, CombinedError, CustomError].
- `T::Type`: DataType of parameters and predictions.

# Keyword arguments
- `dv_compartment::Int`: The index of the compartment for the prediction of the dependent variable. Default = 1.
- `sensealg`: Sensitivity algorithm to use for the calculation of gradients of the parameters with respect to the DESolution.
"""
DeepCompartmentModel(ode_fn::Function, model, error::AbstractErrorModel, ::Type{T} = Float32; kwargs...) where T = 
    DeepCompartmentModel(ode_fn, _estimate_num_partials(ode_fn), model, error, T; kwargs...)

DeepCompartmentModel(ode_fn::Function, model, ::Type{T} = Float32; kwargs...) where T = 
    DeepCompartmentModel(ode_fn, _estimate_num_partials(ode_fn), model, ImplicitError(), T; kwargs...)

# Hack, does not work if setindex! is called on other variables or if never called (as is the case in non in-place functions)
function _estimate_num_partials(ode_fn::Function)
    nargs = first(methods(ode_fn)).nargs - 1
    if nargs < 4
        throw(ErrorException("Cannot automatically identify the number of partials in the ODE function. Explicitly provide them using `DCM(ode_fn, num_partials, model)`."))
    end
    lowered_lines = split(string(@code_lowered ode_fn(1:nargs...)), "\n")
    setindex_lines = filter(!=(nothing), match.(r"setindex!.*", lowered_lines))
    matches = map(Base.Fix2(getfield, :match), setindex_lines)
    raw_indexes_set = map(Base.Fix2(getfield, :match), match.(r", \d+", matches))
    indexes_set = [parse(Int, x[3:end]) for x in raw_indexes_set]
    return maximum(indexes_set)
end


"""
DeepCompartmentModel(ode_fn, num_partials, model, error, T=Float32; kwargs...)

Convenience constructor that internally creates an ODEProblem based on the 
passed ode_fn.

# Arguments
- `ode_fn::Function`: Function that describes the dynamical system.
- `num_partials::Int`: The number of partial differential equations present in the ode_fn.
- `model`: Model to use for the prediction of DE parameters. The package focusses on the use of neural networks based on Lux.jl.
- `error::AbstractErrorModel`: Error model to use. Should be one of [ImplicitError, AdditiveError, ProportionalError, CombinedError, CustomError].
- `T::Type`: DataType of parameters and predictions.

# Keyword arguments
- `dv_compartment::Int`: The index of the compartment for the prediction of the dependent variable. Default = 1.
- `sensealg`: Sensitivity algorithm to use for the calculation of gradients of the parameters with respect to the DESolution.
"""
DeepCompartmentModel(ode_fn::Function, num_comp::Int, model, error::AbstractErrorModel, ::Type{T} = Float32; kwargs...) where T = 
    DeepCompartmentModel(ODEProblem(ode_fn, zeros(T, num_comp), (T(-0.1), one(T)), T[]), model, error, T; kwargs...)

DeepCompartmentModel(ode_fn::Function, num_comp::Int, model, ::Type{T} = Float32; kwargs...) where T = 
    DeepCompartmentModel(ODEProblem(ode_fn, zeros(T, num_comp), (T(-0.1), one(T)), T[]), model, ImplicitError(), T; kwargs...)

"""
DCM(args...; kwargs...)

Alias for DeepCompartmentModel(args...; kwargs...)
"""
DCM(args...; kwargs...) = DeepCompartmentModel(args...; kwargs...)

Base.show(io::IO, dcm::DeepCompartmentModel{T,D,M,E,S}) where {T,D,M,E,S} = print(io, "DeepCompartmentModel{$T, $(dcm.problem.f.f), $(dcm.error)}")

################################################################################
##########                        Model API                           ##########
################################################################################

function predict_typ_parameters(dcm::DeepCompartmentModel, container::Union{AbstractIndividual, Population{T,I}}, ps, st) where {T,I}
    ζ, st_θ = dcm.model(get_x(container), ps.theta, st.theta)
    return ζ, merge(st, (theta = st_θ, ))
end

# TODO: TimeVariable version probably gives a vector of ζ and st like this
function predict_typ_parameters(dcm::DeepCompartmentModel, population::Population{T,I}, ps, st) where {T<:TimeVariable,I}
    ζ, st_θ = dcm.model.(get_x(population), (ps.theta,), (st.theta,))
    return ζ, merge(st, (theta = st_θ[end], ))
end

function predict(dcm::DeepCompartmentModel, data, ps_, st; individual::Bool = false, kwargs...)
    ps = constrain(dcm, ps_)
    type = individual ? MixedObjective : FixedObjective # Any MixedObjective will do here
    p, _ = predict_de_parameters(type, dcm, data, ps, st)

    return forward_ode(dcm, data, p; kwargs...)
end


# Arguments
- `model::DeepCompartmentModel`: The model to use to perform the prediction.
- `container::Union{AbstractIndividual, Population}`: A population or Individual to perform the predictions for.
- `p`: Model parameters. Default = model.p.
- `full`: Return the output for all model compartments.
- `interpolate`: Saves additional time points to return a continuous solution of the DE.
- `get_dv`: Directly returns the predictions for the dv compartment.
- `saveat`: Custom time points to save the solution.
- `sensealg`: Sensitivity algorithm to use for gradient calculations.
"""
function forward(model::DeepCompartmentModel, container::Union{AbstractIndividual, Population}, p; get_dv=Val(false), kwargs...) # Everything else → pass z directly
    ζ_, st = predict_typ_parameters(model, container, p)
    ζ = construct_p(ζ_, container)
    return forward_ode(model, container, ζ, get_dv; kwargs...), st
end

forward_adjoint(model::DeepCompartmentModel, args...) = forward(model, args...; get_dv=Val(true), sensealg=ForwardDiffSensitivity(;convert_tspan=true))
# TODO: Adding converted tspan is type unsafe, investigate InterpolatingAdjoint & GaussAdjoint

##### Variational Inference: TODO: Make more general, i.e. something like forward_adjoint_with_sample(...)?

function objective(obj::VariationalELBO{V,A,E,F1,F2}, model::DeepCompartmentModel, population::Population{T,I}, p_::NamedTuple, phi_::NamedTuple) where {V,A,E,F1,F2,T<:Static,I}
    p = constrain(obj, p_)
    phi = constrain_phi(V, phi_)
    ζ, st = predict_typ_parameters(model, population, p)
    eta_mask = @ignore_derivatives indicator(size(ζ, 1), model.objective.idxs)

    ϵ = take_mc_samples(eltype(ζ), model.objective.approx, length(model.objective.idxs), length(population))
    return -sum(elbo.((model,), population, (eta_mask,), eachcol(ζ), (p,), eachcol(phi.mean), phi.L, ϵ))
end

take_mc_samples(::Type{T}, sa::SampleAverage, args...) where T<:Real = sa.samples # very large vector we should reshape?
take_mc_samples(::Type{T}, mc::MonteCarlo, k, n) where T<:Real = eachslice(randn(T, k, mc.n_samples, n), dims=3)

"""These always use the path derivative gradient estimator by Roeder et al."""
# getq(μ, L::AbstractMatrix) = @ignore_derivatives MvNormal(μ, L * L')
# getq(μ, σ::AbstractVector) = @ignore_derivatives MvNormal(μ, σ)
getq(μ, L::AbstractMatrix) = @ignore_derivatives MvNormal(μ, L * L')
getq(μ, σ::AbstractVector) = @ignore_derivatives MvNormal(μ, σ)
getq_and_eta(μ, L::AbstractMatrix, ϵ) = (getq(μ, L), μ .+ L * ϵ)
getq_and_eta(μ, σ::AbstractVector, ϵ) = (getq(μ, σ), μ .+ σ .* ϵ)

function elbo(model, individual::AbstractIndividual, eta_mask, ζ::AbstractVector, p::NamedTuple, μ::AbstractVector, L::AbstractMatrix, ϵ::AbstractMatrix) 
    q, η = getq_and_eta(μ, L, ϵ)
    return mean(elbo.((model,), (individual,), (eta_mask,), (ζ,), (p,), (q,), eachcol(η)))
end

# TODO: Calculate z_ one step before this to reduce inputs?
function elbo(model::DeepCompartmentModel, individual::AbstractIndividual, eta_mask, ζ::AbstractVector, p::NamedTuple, q, η::AbstractVector)
    z_ = ζ .* exp.(eta_mask * η)
    z = construct_p(z_, individual)
    ŷ = forward_ode(model, individual, z, Val(true); sensealg=ForwardDiffSensitivity(;convert_tspan=true))
    Σ = variance(model, p, ŷ)
    y = ignore_derivatives(individual.y)
    return logpdf(MvNormal(ŷ, Σ), y) + logpdf(MvNormal(zero(η), p.omega), η) - logpdf(q, η)
end

Base.show(io::IO, dcm::DeepCompartmentModel) = print(io, "DeepCompartmentModel{$(dcm.problem.f.f), $(dcm.objective)}")
