import DifferentialEquations.SciMLBase: AbstractODEProblem
import DifferentialEquations: ODEProblem, remake, solve
import Zygote: pullback, ignore
import Statistics: mean
import Flux

using DiffEqSensitivity

struct DCM
    ode::AbstractODEProblem
    weights::Vector{Float32}
    re::Function
    measurement_compartment::Integer
end

"""
Simplified constructors, destructures the Flux model into weights and re, and 
collects the number of compartments in the ODE model. Follows the assumption 
that the measurements depict the concentration in compartment 1.
"""
DCM(ode::AbstractODEProblem, ann::Flux.Chain; measurement_compartment::Integer=1) = 
    DCM(ode, Flux.destructure(ann)..., measurement_compartment)

"""Even simpler version that creates a ODEProblem based on a passed function."""
DCM(f::Function, ann::Flux.Chain, num_compartments::Integer; measurement_compartment::Integer=1) =
    DCM(ODEProblem(f, zeros(num_compartments), (0., 1.)), Flux.destructure(ann)..., measurement_compartment)

"""Predict PK parameters based on data (and parameters)"""
(dcm::DCM)(𝐱ᵢ::Vector{Float64}, 𝑤::Vector{Float32})::Vector{Float64} = dcm.re(𝑤)(𝐱ᵢ)
(dcm::DCM)(𝐗::Matrix{Float64}, 𝑤::Vector{Float32})::Matrix{Float64} = dcm.re(𝑤)(𝐗')
(dcm::DCM)(x) = dcm(x, dcm.weights)
(dcm::DCM)(population::Population) = dcm(population.x)
(dcm::DCM)(individual::Individual) = dcm(individual.x)

Base.copy(model::DCM, 𝑤::Vector{Float32}) = 
    DCM(model.ode, 𝑤, model.re, model.measurement_compartment)

"""Function from collecting the event times vector from the callback."""
get_tstops(callback::DiscreteCallback) = 
    getfield(callback.condition, first(fieldnames(typeof(callback.condition))))


"""
Function for prediction of concentrations. Accepts three optional arguments:
    * interpolating (Boolean, default = false):
    Switches from saving the predictions for the measurement time points only to 
    an complete interpolated continuous solution. Setting to true is advised 
    when plotting the solution.
    
    * tspan (Tuple, default = nothing):
    Allows to set a custom time span to solve the solution in. Is a tuple of two
    Floats, in the format (starting point, end point). The starting point should 
    be set before the first dose event of interest, otherwise the callback might 
    be missed.

    * full (Boolean, default = false):
    Switches from saving the full solution and only the solution for the 
    measurement compartment. Setting this to true allows one to inspect the 
    concentration in the other compartments over time, and can be usefull for 
    evaluating the appropriateness of the current compartment model.
"""
function predict(model::DCM, indv::Individual; interpolating=false, tspan=nothing, full::Bool=false)
    ζ = model(indv.x)
    u0 = copy(model.ode.u0)
    ignore() do # ignore the array mutation
        u0[model.measurement_compartment] = indv.u0
    end
    tspan = tspan === nothing ? (-.1, maximum(indv.t)) : tspan
    prob = remake(model.ode, u0=u0, tspan=tspan, p=vcat(ζ, 0.))
    save_idxs = full ? (1:length(u0)) : model.measurement_compartment
    if interpolating
        # Sets save_positions to true, so that interpolating works correctly for events
        indv.callback.save_positions .= (true, true)
        sol = solve(prob, save_idxs=save_idxs, tstops=get_tstops(indv.callback), callback=indv.callback)
        indv.callback.save_positions .= (false, false) # Need to reset
    else
        sol = solve(prob, saveat=indv.t, save_idxs=save_idxs, tstops=get_tstops(indv.callback), callback=indv.callback)
    end

    return sol
end


predict(model::DCM, population::Population; interpolating=false, tspan=nothing, full=false) =
    [predict(model, population[i]; interpolating=interpolating, tspan=tspan, full=full) for i in 1:length(population)]



"""Prediction function used for gradient calculation"""
function predict_adjoint(model::DCM, indv::Individual, 𝑤::Vector{Float32})
    ζ = model(indv.x, 𝑤)
    u0 = copy(model.ode.u0)
    ignore() do # ignore the array mutation
        u0[model.measurement_compartment] = indv.u0
    end
    prob = remake(model.ode, u0=u0, tspan=(-.1, maximum(indv.t)), p=vcat(ζ, 0.))
    return solve(
        prob, 
        saveat=indv.t, 
        tstops=get_tstops(indv.callback), 
        callback=indv.callback, 
        sensealg=ForwardDiffSensitivity()
    )[model.measurement_compartment, :]
end

predict_adjoint(model::DCM, population::Population, 𝑤::Vector{Float32}) = 
    [predict_adjoint(model, population[i], 𝑤) for i in 1:length(population)]


"""Main objective function for the DCM. Is currently hardcoded in the fit! function."""
function mse(model::DCM, population::Population, 𝑤::Vector{Float32})
    predictions = predict_adjoint(model, population, 𝑤)
    squared_errors = (vcat(population.y...) - vcat(predictions...)).^2
    return mean(squared_errors)
end

mse(model::DCM, population::Population) = mse(model, population, model.weights)

function mse(model::DCM, individual::Individual, 𝑤::Vector{Float32})
    prediction = predict_adjoint(model, individual, 𝑤)
    squared_error = (individual.y - prediction).^2
    return mean(squared_error)
end


"""
Function to fit a DCM to a population. Has two optional arguments:
    * iterations (Integer, default = 100):
    Sets the number of iterations to train the model for. 
    
    * callback (Function, default = () -> nothing):
    A function that is called before the neural network weights are updated 
    during each iteration. Should be a Function with two parameters:
    (loss, current_iteration). Can be used to monitor loss, or produce 
    diagnostic plots at each iteration.
"""
function fit!(model::DCM, population::Population, optimizer; iterations::Integer=100, callback=(l, e) -> nothing)
    for epoch in 1:iterations
        loss, back = pullback((𝑤) -> mse(model, population, 𝑤), model.weights)
        ∇𝑤 = first(back(1.0))
        callback(loss, epoch)
        Flux.update!(optimizer, model.weights, ∇𝑤)
    end
    nothing
end

"""Very simple callback that prints the train set loss for each epoch."""
monitor_loss(loss, epoch) = println("Epoch $epoch, training loss: $loss")

"""Pretty print the model"""
Base.show(io::IO, model::DCM) = print(io, "DCM($(model.ode.f.f), $(model.re.m))")