import InteractiveUtils: @code_lowered

"""
    DeepCompartmentModel{P,M,E,S}

Model architecture originally described in [janssen2022]. Originally uses a 
neural network to learn the relationship between the covariates and the 
parameters of a system of differential equations, which for example represents 
a compartment model.

# Arguments
- `problem::AbstractDEProblem`: DE problem describing the dynamical system.
- `model`: Model to use for the prediction of DE parameters or DE components. The package focusses on the use of neural networks based on Lux.jl.
- `error`: Error model to use for likelihood calculations.
- `target`: The index of the partial derivative for the prediction of the dependent variable. Default = 1.
- `sensealg`: Sensitivity algorithm to use for the calculation of gradients of the parameters with respect to the DESolution.

\\
[janssen2022] Janssen, Alexander, et al. "Deep compartment models: a deep learning approach for the reliable prediction of time‐series data in pharmacokinetic modeling." CPT: Pharmacometrics & Systems Pharmacology 11.7 (2022): 934-945.
"""
struct DeepCompartmentModel{P<:SciMLBase.AbstractDEProblem,M<:Lux.AbstractLuxLayer,E<:AbstractErrorModel,T<:Union{<:Int,AbstractVector{<:Int}}, S<:SciMLBase.AbstractSensitivityAlgorithm} <: AbstractDEModel{P,M,E,S}
    problem::P
    model::M
    error::E
    target::T
    sensealg::S

    DeepCompartmentModel(
        problem::P, model::M, error::E=ImplicitError(); 
        target::T = 1, sensealg::S = ForwardDiffSensitivity()
    ) where {P<:SciMLBase.AbstractDEProblem, M, T, E<:AbstractErrorModel, S<:SciMLBase.AbstractSensitivityAlgorithm} = 
        new{P,M,E,T,S}(problem, model, error, target, sensealg)
end

# Constructors. 
"""
    DeepCompartmentModel(ode_fn, model, error; target, sensealg)

Convenience constructor that internally creates an ODEProblem based on the 
passed ode_fn. Attempts to estimate the number of partial differential equations
present in the ode_fn.

# Arguments
- `ode_fn::Function`: Function that describes the dynamical system.
- `model`: Model to use for the prediction of DE parameters. The package focusses on the use of neural networks based on Lux.jl.
- `error::AbstractErrorModel`: Error model to use. Should be one of [ImplicitError, AdditiveError, ProportionalError, CombinedError, CustomError].

# Keyword arguments
- `target`: The index(es) of the compartment(s) for the prediction of the dependent variable. Default = 1.
- `sensealg`: Sensitivity algorithm to use for the calculation of gradients of the parameters with respect to the DESolution.
"""
DeepCompartmentModel(ode_fn::Function, model, error::AbstractErrorModel=ImplicitError(); kwargs...) = 
    DeepCompartmentModel(ode_fn, _estimate_num_partials(ode_fn), model, error; kwargs...)

# Hack, does not work if setindex! is called on other variables or if never called (as is the case in non in-place functions)
function _estimate_num_partials(ode_fn::Function)
    nargs = first(methods(ode_fn)).nargs - 1
    if nargs < 4 # I.e. if not inplace
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
DeepCompartmentModel(ode_fn, num_partials, model, error=ImplicitError(); target, sensealg)

Convenience constructor that internally creates an ODEProblem based on the 
passed ode_fn.

# Arguments
- `ode_fn::Function`: Function that describes the dynamical system.
- `num_partials::Int`: The number of partial differential equations present in the `ode_fn`.
- `model`: Model to use for the prediction of DE parameters. The package focusses on the use of neural networks based on Lux.jl.
- `error::AbstractErrorModel`: Error model to use. Should be one of [ImplicitError, AdditiveError, ProportionalError, CombinedError, CustomError].

# Keyword arguments
- `target`: The index(es) of the compartment(s) for the prediction of the dependent variable. Default = 1.
- `sensealg`: Sensitivity algorithm to use for the calculation of gradients of the parameters with respect to the DESolution. Default = ForwardDiffSensitivity().
"""
DeepCompartmentModel(ode_fn::Function, num_partials::Int, model, error::AbstractErrorModel=ImplicitError(); kwargs...) = 
    DeepCompartmentModel(ODEProblem(ode_fn, zeros(num_partials), (-0.1, 1.)), model, error; kwargs...)

"""
DCM(args...; kwargs...)

Alias for DeepCompartmentModel(args...; kwargs...)
"""
DCM(args...; kwargs...) = DeepCompartmentModel(args...; kwargs...)

Base.show(io::IO, dcm::DeepCompartmentModel{SciMLBase.AbstractDEProblem}) = 
    print(io, "DeepCompartmentModel{$(dcm.problem.f.f), $(dcm.error)}")

################################################################################
##########                        Model API                           ##########
################################################################################

function predict_typ_parameters(dcm::DeepCompartmentModel, data::D, ps, st) where D<:Union{<:AbstractIndividual, Population{<:AbstractIndividual}}
    ζ, st_θ = dcm.model(get_x(data), ps.theta, st.theta)
    return ζ, Accessors.@set st.theta = st_θ
end

function predict_typ_parameters(dcm::DeepCompartmentModel, population::Population{<:TimeVariableIndividual}, ps, st)
    st_local = deepcopy(st)
    ζ = map(get_x(population)) do xᵢ
        ζᵢ, st_θ = dcm.model(xᵢ, ps.theta, st.theta)
        Accessors.@reset st_local.theta = st_θ
        return ζᵢ
    end
    return ζ, st_local
end

predict_de_parameters(dcm::DeepCompartmentModel, data::D, ps::NamedTuple{(:theta,:error)}, st) where D<:Union{<:AbstractIndividual, Population{<:AbstractIndividual}} = 
    predict_typ_parameters(dcm, data, ps, st)

function predict_de_parameters(dcm::DeepCompartmentModel, data::D, ps::NamedTuple{(:theta,:error,:omega,:phi)}, st) where D<:Union{<:AbstractIndividual, Population{<:AbstractIndividual}}
    ζ, st_theta = predict_typ_parameters(dcm, data, ps, st)
    η = get_random_effects(ps, st)
    z = ζ .* exp.(η)
    return z, Accessors.@set st.theta = st_theta
end


function predict(dcm::AbstractDEModel, data, ps, st; individual = true, target = true, kwargs...)
    if individual
        ps_local = ps
    else
        _keys = filter(!∈([:omega, :phi]), keys(ps))
        ps_local = ps[_keys]
    end

    z, _ = predict_de_parameters(dcm, data, ps_local, st)
    
    if target
        return solve_for_target(dcm, data, z; kwargs...)
    else
        return solve(dcm, data, z; kwargs...)
    end
end