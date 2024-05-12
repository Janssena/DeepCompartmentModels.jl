import DifferentialEquations.SciMLBase: AbstractDEProblem, AbstractODEProblem
import SciMLSensitivity: ForwardDiffSensitivity
import DifferentialEquations: Tsit5
import Random
import Lux

include("population.jl");
include("objectives.jl");
include("constrain.jl");

"""
    DeepCompartmentModel{O,D,M,P,R}

Model architecture originally described in [janssen2022]. Uses a Neural Network 
to learn the relationship between the covariates and the parameters of a system 
of differential equations, for example describing a compartment model.
\\
[janssen2022] Janssen, Alexander, et al. "Deep compartment models: a deep learning approach for the reliable prediction of time‐series data in pharmacokinetic modeling." CPT: Pharmacometrics & Systems Pharmacology 11.7 (2022): 934-945.
"""
struct DeepCompartmentModel{O<:AbstractObjective,D<:AbstractDEProblem,M<:Lux.AbstractExplicitLayer,P,R<:Random.AbstractRNG} <: AbstractDEModel{O,D,M,P}
    objective::O
    problem::D
    ann::M
    p::P
    dv_compartment::Int
    rng::R
end
# Constructors. TODO: Consider using StatefulLuxLayers and removing st from parameter vector.
"""
    DeepCompartmentModel(prob, ann, p; rng, objective, dv_compartment)

# Arguments
- `prob::AbstractDEProblem`: DE problem describing the dynamical system.
- `ann::AbstractExplicitLayer`: Lux model representing the ann.
- `p`: Model parameters, containing all model parameters.
- `rng`: Randomizer used for initialization of the parameters.
- `objective::AbstractObjective`: Objective function to optimize. Currently supports SSE, LogLikelihood, and VariationalELBO (for mixed effects estimation). Default = SSE.
- `dv_compartment::Int`: The index of the compartment for the prediction of the dependent variable. Default = 1.
"""
function DeepCompartmentModel(problem_::D, ann::M, p::P; rng::R=Random.default_rng(), objective::O=SSE(), dv_compartment::Int=1) where {O<:AbstractObjective,D<:AbstractDEProblem,M,P,R}
    !(ann isa Lux.AbstractExplicitLayer) && (ann = Lux.transform(ann))
    if !(problem_ isa AbstractODEProblem)
        println("[info] DeepCompartmentModels.jl is not tested using problems of type $(D). Be wary of any errors.")
    end
    problem = (Base.typename(typeof(problem_)).wrapper)(problem_.f, Float32.(problem_.u0), Float32.(problem_.tspan), Float32[])
    DeepCompartmentModel{O,typeof(problem),M,P,R}(objective, problem, ann, p, dv_compartment, rng)
end
"""
    DeepCompartmentModel(ode_f, num_compartments, args...; kwargs...)

Convenience constructor creating an ODE model based on the user supplied 
`ode_f` function. The number of compartments needs to be supplied in order 
to correctly initialize the ODEProblem.

# Arguments
- `ode_f::AbstractDEProblem`: DE problem describing the dynamical system.
- `num_compartments::Int`: Number of partial differential equations in the ODE.
"""
function DeepCompartmentModel(ode_f::Function, num_compartments::Integer, args...; kwargs...)
    problem = ODEProblem(ode_f, zeros(Float32, num_compartments), (-0.1f0, 1.f0), Float32[])
    return DeepCompartmentModel(problem, args...; kwargs...)
end
"""
    DeepCompartmentModel(problem, ann; kwargs...)

Convenience constructor also initializing the model parameters.

# Arguments
- `prob::AbstractDEProblem`: DE problem describing the dynamical system.
- `ann::AbstractExplicitLayer`: Lux model representing the ann.
"""
function DeepCompartmentModel(problem::D, ann::M; rng=Random.default_rng(), objective=SSE(), kwargs...) where {D<:AbstractDEProblem,M}
    !(ann isa Lux.AbstractExplicitLayer) && (ann = Lux.transform(ann))
    p = init_params(rng, objective, ann)
    DeepCompartmentModel(problem, ann, p; rng, objective, kwargs...)
end
"""
    DeepCompartmentModel(problem, ann, ps, st; kwargs...)

Convenience constructor initializing the remaining model parameters with user 
initialized neural network weights `ps` and state `st`.

# Arguments
- `prob::AbstractDEProblem`: DE problem describing the dynamical system.
- `ann::AbstractExplicitLayer`: Lux model representing the ann.
- `ps`: Initial parameters for the neural network.
- `st`: Initial state for the neural network.
"""
function DeepCompartmentModel(problem::D, ann::M, ps::NamedTuple, st::NamedTuple; rng::R=Random.default_rng(), objective::O=SSE(), kwargs...) where {O<:AbstractObjective,D<:AbstractDEProblem,M,R}
    !(ann isa Lux.AbstractExplicitLayer) && (ann = Lux.transform(ann))
    p = init_params(rng, objective, ps, st)
    DeepCompartmentModel(problem, ann, p; rng, kwargs...)
end

"""
    DCM(args...; kwargs...)

Alias for DeepCompartmentModel(args...; kwargs...)
"""
DCM(args...; kwargs...) = DeepCompartmentModel(args...; kwargs...)


# predict_typ_parameters → simple forward
predict_typ_parameters(model::DeepCompartmentModel, container::Union{AbstractIndividual, Population}, p) = model.ann(container.x, p.weights, p.st)
predict_typ_parameters(model::DeepCompartmentModel, container::Population{T,I}, p) where {T<:TimeVariable,I} = model.ann.(container.x, (p.weights,), (p.st,))
# construct_p (add random effect & add padding: zeros for I and t for timevariable)
construct_p(z::AbstractVector, ::AbstractIndividual) = [z; 0]
construct_p(z::AbstractMatrix, ::Population) = vcat(z, zeros(eltype(z), 1, size(z, 2))) # → run eachcol
construct_p(z::AbstractMatrix, individual::AbstractIndividual) = vcat(individual.t.x, z, zeros(eltype(z), 1, size(z, 2)))
function construct_p(z::AbstractVector{<:AbstractMatrix}, population::Population) 
    ts = getfield.(getfield.(population, :t), :x)
    return vcat.(ts, z, zero.(ts))
end

"""
    forward(model::DeepCompartmentModel, container, p; full, interpolate, get_dv, saveat, sensealg)

Predicts the differential equation parameters and returns the solution.

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


##### Variational Inference:

function objective(::VariationalELBO{V,A,E,F1,F2}, model::DeepCompartmentModel, population::Population{T,I}, p_::NamedTuple, phi_::NamedTuple) where {V,A,E,F1,F2,T<:Static,I}
    p = constrain(p_)
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
    return logpdf(MvNormal(ŷ, Σ), individual.y) + logpdf(MvNormal(zero(η), p.omega), η) - logpdf(q, η)
end

Base.show(io::IO, dcm::DeepCompartmentModel) = print(io, "DeepCompartmentModel{$(dcm.problem.f.f), $(dcm.objective)}")
