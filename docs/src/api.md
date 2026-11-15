# API reference

```@meta
CurrentModule = DeepCompartmentModels
```

The names below are exported by `using DeepCompartmentModels`. The package also re-exports
[Lux.jl](https://lux.csail.mit.edu), [Optimisers.jl](https://github.com/FluxML/Optimisers.jl),
[Accessors.jl](https://github.com/JuliaObjects/Accessors.jl),
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl),
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) and
[SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl), whose documentation applies to
those names.

Some functions have no docstring yet; they are listed in tables with a one-line description so that
this page still covers the whole interface.

## Data

An [`Individual`](@ref) holds the covariates, observation times, observations and dosing events of one
subject. A [`Population`](@ref) is a vector of individuals of the same type. See the
[data tutorial](generated/05-loading-data.md).

```@docs
Individual
BasicIndividual
TimeVariableIndividual
AbstractIndividual
Population
generate_dosing_callback
```

| function | description |
|:--|:--|
| `get_x(data[, key])` | covariates of an individual, or a matrix with one column per individual for a population |
| `get_t(data)` | observation times |
| `get_y(data)` | observations |
| `get_tx(individual)` | times at which the covariates of a `TimeVariableIndividual` were measured |
| `load(df, covariates)` | build a population from a `DataFrame` (needs `DataFrames`; currently broken, see [Status and limitations](limitations.md)) |
| `MOIndividual` | an individual with several observed outputs (not usable in populations yet, see [Status and limitations](limitations.md)) |

## Models

```@docs
DeepCompartmentModel
DCM
StandardNeuralNetwork
SNN
```

| name | description |
|:--|:--|
| `AbstractModel`, `AbstractDEModel` | supertypes of the models |
| `UniversalDiffEq`, `BasicUDE`, `TimeConcatUDE` | neural network right-hand sides (universal differential equations), see [Status and limitations](limitations.md) |

### Prediction

```@docs
predict
solve_for_target
construct_p
```

| function | description |
|:--|:--|
| `predict_typ_parameters(model, data, ps, st)` | typical ODE parameters `ζ(x)`, one column per individual, and the new state |
| `predict_de_parameters(model, data, ps, st)` | ODE parameters including the random effects when `ps` has them |
| `solve(model, individual, z; saveat, solver, ...)` | solve the ODE of the model for the ODE parameters `z` |

### Compartment models

All are in-place ODE functions `f!(dA, A, p, t)`; the last element of `p` is the infusion rate.

```@docs
unpack
one_comp!
one_comp_abs!
two_comp!
two_comp_abs!
```

## Error models

```@docs
AdditiveError
ProportionalError
CombinedError
CustomError
ErrorModelSet
ImplicitError
```

| function | description |
|:--|:--|
| `AbstractErrorModel` | supertype of the error models |
| `var(error, ŷ, ps, st)` | covariance matrix of the observations given the predictions |
| `make_dist(error, ŷ, ps)` | `MvNormal` distribution of the observations |

## Objectives

An objective is a callable `objective(model, data, ps, st)` that returns the scalar to minimise.

```@docs
VariationalELBO
sse
mse
```

| objective | description |
|:--|:--|
| `SSE()` | sum of squared errors, fixed effects only |
| `MSE()` | mean squared error, fixed effects only |
| `LogLikelihood()` | negative log-likelihood under the error model of the model, fixed effects only |

Building blocks of the ELBO:

| function | description |
|:--|:--|
| `loglikelihood(model, data, ps, st)` | log-likelihood of the observations |
| `logprior(model, ps, st)` | log density of the sampled random effects under `N(0, Ω)` |
| `logq(model, ps, st)` | log density of the sampled random effects under their variational posteriors |
| `logjoint(model, data, ps, st)` | `loglikelihood + logprior` |
| `elbo(model, data, ps, st)` | `logjoint - logq` |
| `kldivergence(model, ps, st)` | KL divergence between the posteriors and the prior |
| `getq(model, ps, st)` | the variational posteriors as `MvNormal` distributions |

## Setting up and training

There is no single `fit` function; training is a loop over [`gradient`](@ref) and an Optimisers.jl
update, as shown in the [first tutorial](generated/01-basic-dcm.md).

```@docs
setup
gradient
```

| function | description |
|:--|:--|
| `take_batch(ps, st, idxs)` | select the entries of `ps` and `st` that belong to a subset of individuals |
| `setup_phi(objective, rng, model, population, Ω; ...)` | initial variational posteriors and sampling state |
| `get_random_effects(ps, st)` | random effects of all individuals as an `n_params × n_individuals` matrix |
| `update_epsilon!(rng, st)` | resample the noise used to draw from the variational posteriors, in place |
| `create_batches(population, batchsize)` | random mini-batches of indices |
| `m_step(objective, rng, model, population, ps, st; epochs, num_samples, verbose)` | update the residual error and `Ω` from the variational posteriors |
| `optimise_omega(ps)` | `Ω` implied by the variational posteriors |
| `optimise_residual_error(...)` | optimise the residual error parameters with the posteriors fixed |
| `MeanSqrt()`, `MeanVar()` | parameterisations of the posterior covariances, the `params` keyword of `setup` |

## Lux layers

```@docs
Normalize
SingleHeadedBranch
MultiHeadedBranch
make_branch
Combine
AddGlobalParameters
interpret_branch
```
