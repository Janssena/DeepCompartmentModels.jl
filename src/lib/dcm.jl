import DifferentialEquations.SciMLBase: AbstractDEProblem, AbstractODEProblem
import SciMLSensitivity: ForwardDiffSensitivity
import DifferentialEquations: Tsit5
import Random
import Lux

include("population.jl");
include("objectives.jl");
include("constrain.jl");


"""
Might make sense to write init_params function for objective functions. This way
we can just run init_params(model) -> init_params(ann, objective), which could
also initialize in-place to allow re-training.
"""

"""
An abstract DE model has a common way of solving the ODE. One thing it must have 
that is not enforced here is a dv_compartment variable.
"""
abstract type AbstractDEModel{O,D,M,P} <: AbstractModel{O,M,P} end

struct DeepCompartmentModel{O<:AbstractObjective,D<:AbstractDEProblem,M<:Lux.AbstractExplicitLayer,P,R<:Random.AbstractRNG} <: AbstractDEModel{O,D,M,P}
    objective::O
    problem::D
    ann::M
    p::P
    dv_compartment::Int
    rng::R
    # Constructors
    function DeepCompartmentModel(problem_::D, ann::M, p::P; rng::R=Random.default_rng(), objective::O=SSE(), dv_compartment::Int=1) where {O<:AbstractObjective,D<:AbstractDEProblem,M,P,R}
        !(ann isa Lux.AbstractExplicitLayer) && (ann = Lux.transform(ann))
        if !(problem_ isa AbstractODEProblem)
            println("[info] DeepCompartmentModels.jl is not tested using problems of type $(D). Be wary of any errors.")
        end
        problem = (Base.typename(typeof(problem_)).wrapper)(problem_.f, Float32.(problem_.u0), Float32.(problem_.tspan), Float32[])
        new{O,typeof(problem),M,P,R}(objective, problem, ann, p, dv_compartment, rng)
    end
    # For convenience:
    # Using ODEFunc instead of ODEProblem
    function DeepCompartmentModel(ode_f::Function, num_compartments::Integer, args...; kwargs...)
        problem = ODEProblem(ode_f, zeros(Float32, num_compartments), (-0.1f0, 1.f0), Float32[])
        return DeepCompartmentModel(problem, args...; kwargs...)
    end
    # Auto-init parameters
    function DeepCompartmentModel(problem::D, ann::M; rng=Random.default_rng(), objective=SSE(), kwargs...) where {D<:AbstractDEProblem,M}
        !(ann isa Lux.AbstractExplicitLayer) && (ann = Lux.transform(ann))
        p = init_params(rng, objective, ann)
        DeepCompartmentModel(problem, ann, p; rng, objective, kwargs...)
    end
    # with ps and st
    function DeepCompartmentModel(problem::D, ann::M, ps::NamedTuple, st::NamedTuple; rng::R=Random.default_rng(), objective::O=SSE(), kwargs...) where {O<:AbstractObjective,D<:AbstractDEProblem,M,R}
        !(ann isa Lux.AbstractExplicitLayer) && (ann = Lux.transform(ann))
        p = init_params(rng, objective, ps, st)
        DeepCompartmentModel(problem, ann, p; rng, kwargs...)
    end
end

"""Alias"""
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
# forward_ode → solve ode
forward_ode(model::AbstractDEModel, population::Population, z::AbstractMatrix; kwargs...) = forward_ode.((model,), population, eachcol(z); kwargs...)
forward_ode(model::AbstractDEModel, population::Population, z::AbstractVector{<:AbstractMatrix}; kwargs...) = forward_ode.((model,), population, z; kwargs...)
function forward_ode(model::AbstractDEModel, individual::AbstractIndividual, zᵢ::AbstractVecOrMat; get_dv::Bool=false, sensealg=nothing, full::Bool=false, interpolate::Bool=false, saveat = is_timevariable(individual) ? individual.t.y : individual.t)
    u0 = isempty(individual.initial) ? model.problem.u0 : individual.initial
    saveat_ = interpolate ? empty(saveat) : saveat
    save_idxs = full ? (1:length(u0)) : model.dv_compartment
    prob = remake(model.problem, u0 = u0, tspan = (model.problem.tspan[1], maximum(saveat)), p = zᵢ)
    interpolate && (individual.callback.save_positions .= 1)
    sol = solve(prob, Tsit5(),
        save_idxs = save_idxs, saveat = saveat_, callback=individual.callback, 
        tstops=individual.callback.condition.times, sensealg=sensealg
    )
    interpolate && (individual.callback.save_positions .= 0)
    return get_dv ? sol[model.dv_compartment, :] : sol
end

"""Note: this does not use z."""
function forward(model::DeepCompartmentModel, container::Union{AbstractIndividual, Population}, p; kwargs...) # Everything else → pass z directly
    ζ_, st = predict_typ_parameters(model, container, p)
    ζ = construct_p(ζ_, container)
    return forward_ode(model, container, ζ; kwargs...), st
end


forward_adjoint(model::DeepCompartmentModel, args...) = forward(model, args...; get_dv=true, full=true, sensealg=ForwardDiffSensitivity(;convert_tspan=true))


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

function elbo(model::DeepCompartmentModel, individual::AbstractIndividual, eta_mask, ζ::AbstractVector, p::NamedTuple, q, η::AbstractVector)
    z_ = ζ .* exp.(eta_mask * η)
    z = construct_p(z_, individual)
    ŷ = forward_ode(model, individual, z; get_dv=true, full=true, sensealg=ForwardDiffSensitivity(;convert_tspan=true))
    Σ = variance(model, p, ŷ)
    return logpdf(MvNormal(ŷ, Σ), individual.y) + logpdf(MvNormal(zero(η), p.omega), η) - logpdf(q, η)
end


# """Static Individual"""
# function predict_parameters(model::DeepCompartmentModel, individual::AbstractIndividual{I,X,T,Y,C}, p; kwargs...) where {I,X<:AbstractVector,T<:AbstractVector,Y,C}
#     ζ, st = model.ann(individual.x, p.weights, p.st)
#     return [construct_z(model.objective, ζ); 0], st
# end

# """TimeVariable Individual"""
# function predict_parameters(model::DeepCompartmentModel, individual::AbstractIndividual{I,X,T,Y,C}, p::NamedTuple; kwargs...) where {I,X<:AbstractMatrix,T<:NamedTuple,Y,C}
#     ζ, st = model.ann(individual.x, p.weights, p.st)
#     z_ = construct_z(model.objective, ζ)
#     return vcat(individual.t.x, z_, zero.(individual.t.x)), st
# end

# """Static population"""
# function predict_parameters(model::DeepCompartmentModel, population::Population{T,I}, p::NamedTuple; kwargs...) where {T<:Static,I}
#     ζ, st = model.ann(population.x, p.weights, p.st)
#     z_ = construct_z(model.objective, ζ)
#     return vcat(z_, zeros(eltype(z_), 1, size(z_, 2))), st
# end

# # function forward(model::DeepCompartmentModel, individual::AbstractIndividual{I,X,T,Y,C}, p::NamedTuple; kwargs...) where {I,X<:AbstractVector,T<:AbstractVector,Y,C}
# #     ζ, st = model.ann(individual.x, p.weights, p.st)
# #     z = [construct_z(model.objective, ζ); 0]
# #     return forward(model, individual, z; kwargs...), st
# # end

# # function forward(model::DeepCompartmentModel, individual::AbstractIndividual{I,X,T,Y,C}, p::NamedTuple; kwargs...) where {I,X<:AbstractMatrix,T<:NamedTuple,Y,C}
# #     ζ, st = model.ann(individual.x, p.weights, p.st)
# #     z_ = construct_z(model.objective, ζ)
# #     z = vcat(individual.t.x, z_, zero.(individual.t.x))
# #     return forward(model, individual, z; kwargs...), st
# # end

# function forward(model::DeepCompartmentModel, population::Population{T,I}, p::NamedTuple; kwargs...) where {T<:Static,I}
#     ζ, st = model.ann(population.x, p.weights, p.st)
#     z_ = construct_z(model.objective, ζ)
#     z = vcat(z_, zeros(eltype(z_), 1, size(z_, 2)))
#     return forward.((model,), population, eachcol(z); kwargs...), st
# end

# function forward(model::DeepCompartmentModel, population::Population{T,I}, p::NamedTuple; kwargs...) where {T<:TimeVariable,I}
#     res = model.ann.(getfield.(population, :x), (p.weights,), (p.st,))
#     ζ = first.(res)
#     st = last(res[end])
#     z_ = construct_z.((model.objective,), ζ)
#     ts = getfield.(getfield.(population, :t), :x)
#     z = vcat.(ts, z_, zero.(ts))
#     return forward.((model,), population, z; kwargs...), st
# end

# """Solves the ODE based on the supplied DCM, individual, and parameter vector"""
# function forward(model::AbstractDEModel, individual::AbstractIndividual, zᵢ::AbstractVecOrMat; get_dv::Bool=false, sensealg=nothing, full::Bool=false, interpolate::Bool=false, saveat_ = is_timevariable(individual) ? individual.t.y : individual.t)
#     u0 = isempty(individual.initial) ? model.problem.u0 : individual.initial
#     saveat = interpolate ? empty(saveat_) : saveat_
#     save_idxs = full ? (1:length(u0)) : model.dv_compartment
#     prob = remake(model.problem, u0 = u0, tspan = (model.problem.tspan[1], maximum(saveat_)), p = zᵢ)
#     interpolate && (individual.callback.save_positions .= 1)
#     sol = solve(prob, Tsit5(),
#         save_idxs = save_idxs, saveat = saveat, callback=individual.callback, 
#         tstops=individual.callback.condition.times, sensealg=sensealg
#     )
#     interpolate && (individual.callback.save_positions .= 0)
#     return get_dv ? sol[model.dv_compartment, :] : sol
# end

# """Specific function used in the objective"""
# function forward_adjoint(model::DeepCompartmentModel, container::Union{AbstractIndividual, Population}, p::NamedTuple)
#     sol, st = forward(model, container, p; get_dv=true, full=true, sensealg=ForwardDiffSensitivity(;convert_tspan=true))
#     return sol, st
# end

# get_sol_idx(sol, i) = sol[i, :]
# get_sol_idx(sol::AbstractVector, i) = sol[i, :]

# function forward(model::DeepCompartmentModel, population::Population, p::NamedTuple; kwargs...)
#     ζ, st = model.ann(population.x, p.weights, p.st)
#     return forward.((model,), population, eachcol(ζ); kwargs...), st
# end

# function forward_adjoint(model::DeepCompartmentModel, population::Population, p::NamedTuple; kwargs...)
#     if is_timevariable(population)
#         res = model.ann.(getfield.(population, :x), (p.weights,), (p.st,))
#         ζ = first.(res) # Vector{<:Matrix}
#         st = last(res[end])
#         return forward_adjoint.((model,), population, ζ), st
#     else
#         ζ, st = model.ann(population.x, p.weights, p.st)
#         return forward_adjoint.((model,), population, eachcol(ζ)), st
#     end
#     # return [vec(forward(model, population[i], ζ[:, i]; kwargs...)) for i in eachindex(population)], st
#     # return vec.(forward.((model,), population, eachcol(ζ); kwargs...)), st
# end

# function forward_adjoint(model::DeepCompartmentModel, individual::AbstractIndividual, ζᵢ::AbstractVecOrMat)
#     # TODO: handle time-dependent, i.e. ζᵢ isa AbstractMatrix
#     saveat = is_timevariable(individual) ? individual.t.y : individual.t
#     u0 = @ignore_derivatives isempty(individual.initial) ? model.problem.u0 : individual.initial
#     prob = remake(model.problem, u0 = u0, tspan = (model.problem.tspan[1], maximum(saveat)), p = [ζᵢ; 0])

#     sol = solve(prob, Tsit5(),
#         saveat = saveat, callback=individual.callback, 
#         tstops=individual.callback.condition.times,
#         sensealg=ForwardDiffSensitivity(; convert_tspan=true)
#     )

#     return sol[model.dv_compartment, :]
# end

Base.show(io::IO, dcm::DeepCompartmentModel) = print(io, "DeepCompartmentModel{$(dcm.problem.f.f), $(dcm.objective)}")
