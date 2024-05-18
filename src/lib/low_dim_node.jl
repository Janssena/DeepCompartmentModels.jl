"""
    LowDimensionalNODE(ann, node; kwargs...)

Convenience function for constructing low-dimensional Neural-ODE based models.
This simple implementation learns a model according to:
    
    du/dt = node([u; t]) + I / V

Where the model learns the system dynamics based on the previous state u and 
current time point t. Rate of drug infusions I is divided by a subject-specific
volume of distribution estimate V, which is learned by another neural network. 
See [bram2023] for more details.

# Arguments:
- `ann`: Neural network for predicting V according to f(x) = V.
- `node`: Neural-ODE model.
- `kwargs`: keyword arguments given to the HybridDCM constructor.

[bram2023] Bräm, Dominic Stefan, et al. "Low-dimensional neural ODEs and their application in pharmacokinetics." Journal of Pharmacokinetics and Pharmacodynamics (2023): 1-18.
"""
function LowDimensionalNODE(ann, node; kwargs...)
    dudt(u, p, t; model) = model([u; t], p.weights) .+ (p.I / p.z[1])
    return LowDimensionalNODE(dudt, ann, node, 1; kwargs...)
end

"""
    LowDimensionalNODE(dudt, ann, node; kwargs...)

Constructor for describing low-dimensional Neural-ODE based models.
See [bram2023] for examples of ways of defining dudt.

# Arguments:
- `dudt`: Function describing the system of partial differential equations. 
Should take the form of `f(u, p, t; model) = ...`. p.weights contains the 
parameters for the `node`, p.I the drug infusions at time t, and p.z the parameters
produced by the `ann`.
- `ann`: Neural network for predicting V according to f(x) = V.
- `node`: Neural-ODE model.
- `kwargs`: keyword arguments given to the HybridDCM constructor.

[bram2023] Bräm, Dominic Stefan, et al. "Low-dimensional neural ODEs and their application in pharmacokinetics." Journal of Pharmacokinetics and Pharmacodynamics (2023): 1-18.
"""
function LowDimensionalNODE(dudt, ann, node, num_compartments; kwargs...)
    model = BasicHybridModel(ann, node)
    return HybridDCM(dudt, model, num_compartments; kwargs...)
end