# DeepCompartmentModels.jl

```@meta
CurrentModule = DeepCompartmentModels
```

Pharmacokinetic models where a neural network predicts the parameters of a system of ODEs from patient
covariates, written in Julia on top of [Lux.jl](https://lux.csail.mit.edu) and
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).

- Describe the drug's movement through the body with a compartment model, a built-in one or any ODE you write.
- Let a neural network turn covariates such as weight, age or kidney function into the ODE parameters.
- Dosing regimens (boluses, infusions, many doses) are events in the solve, so doses never need to be
  part of the network input.
- Fit by least squares, by likelihood with a residual error model, or with random effects through
  variational inference.
- Get individual predictions with uncertainty, and read off covariate effects with interpretable
  branch networks.

The method is described in [Janssen et al. (2022)](https://doi.org/10.1002/psp4.12808), *Deep compartment
models: a deep learning approach for the reliable prediction of time-series data in pharmacokinetic modeling*,
CPT: Pharmacometrics & Systems Pharmacology 11(7).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Janssena/DeepCompartmentModels.jl")
```

Loading data from a `DataFrame` and the plots in the tutorials need `DataFrames`, `CSV` and `CairoMakie`
in your own environment.

## Example

A one-compartment model with first-order absorption, where a network maps body weight to
`[kₐ, CL, V]`, fitted to the sum of squared errors:

```julia
using DeepCompartmentModels, Random

# `population` is a Population of Individuals, see the tutorial on loading data
ann = Lux.Chain(
    Normalize([100f0]),                   # scale weight to [0, 1]
    Lux.Dense(1, 8, tanh),
    Lux.Dense(8, 3, Lux.softplus),        # kₐ, CL, V > 0
)
model = DCM(one_comp_abs!, ann; target = 2)   # state 2 is the concentration

objective = SSE()
ps, st = setup(objective, Random.default_rng(), model)

opt_state = Optimisers.setup(Optimisers.Adam(0.02f0), ps)
for epoch in 1:200
    ∇ = gradient(objective, model, population, ps, st)
    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
end

predict(model, population, ps, st)
```

The [first tutorial](generated/01-basic-dcm.md) runs this on real data from start to finish.

## Where to go next

| | |
|:--|:--|
| [Tutorials](tutorials/index.md) | worked examples, each of which is a Quarto notebook that is executed when this site is built |
| [Status and limitations](limitations.md) | what does not work yet, and the quirks worth knowing before you start |
| [API reference](api.md) | everything the package exports |

## Concepts

| | what | where |
|:--|:--|:--|
| **Individual** | covariates, observation times, observations and dosing events of one subject | [`Individual`](@ref) |
| **Population** | a vector of individuals | [`Population`](@ref) |
| **Model** | an ODE, a network predicting its parameters, and an error model | [`DCM`](@ref) |
| **Objective** | the quantity to minimise: `SSE`, `MSE`, `LogLikelihood`, `VariationalELBO` | [objectives tutorial](generated/07-objectives.md) |
| **`ps`, `st`** | learnable parameters and constant state, in the Lux convention | [`setup`](@ref) |

Models, objectives and data are plain Julia values, and parameters are nested `NamedTuple`s, so everything
works with [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) and any Lux layer.
