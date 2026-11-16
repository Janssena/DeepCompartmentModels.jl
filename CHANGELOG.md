# Changelog

All notable changes to DeepCompartmentModels.jl are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> This changelog was reconstructed from the git history. Versions 1.2.2 and 1.2.3 were not tagged,
> so their entries are placed according to the `Project.toml` version bumps (`2576c60`, `43f4d22`).

## [Unreleased]

Development on the `feat/new-ude` branch, after 1.2.3.

### Added

- Documentation website (Documenter.jl) with an API reference, a limitations page, and eleven
  tutorials (`tutorials/*.qmd`) covering basic DCMs, mixed effects, universal differential
  equations, covariate effects, loading data, custom ODEs, objectives, end-to-end workflows,
  counterfactual predictions, PKPD models and occasions. Includes a GitHub Actions workflow to build
  and deploy the docs, README badges, and example datasets (theophylline, tobramycin, warfarin).
- Seeded checks in the tutorials (hidden cells that assert that the loss decreases, the predictions are
  sensible and so on), so the documentation build fails when a tutorial stops learning. A scheduled
  workflow (`.github/workflows/timings.yml`, `docs/timings.jl`) records the run time of every tutorial,
  and the gradient tests compare the speed of Enzyme and ForwardDiff relative to each other.
- `value_and_gradient`, which returns the loss together with its gradient. `gradient` now calls it,
  so the two always agree.
- `rng`, `batchsize` and `nbatch` keyword arguments for `fit`, `gradient` and `value_and_gradient`,
  and an `rng` argument for `update_state` and `update_state_epsilon`.
- `build_problem(ude, model, st)` and in-place right-hand sides are now extendable for custom UDE
  types. Both `BasicUDE` and `TimeConcatUDE` support in-place problems.
- The default sensitivity algorithm of a `DeepCompartmentModel` is picked by the new
  `_default_sensealg(problem, model)`. Users can add methods for their own problem types. For a
  `UniversalDiffEq` with many parameters (at least `UDE_ADJOINT_MIN_PARAMS = 512`) it uses an adjoint
  method instead of `ForwardDiffSensitivity()`.
- `_solve_callback` hook, so the interpolation behaviour of callbacks during a solve can be customised.
- An `ad` keyword for `gradient`, `value_and_gradient` and `fit` that selects the automatic differentiation
  backend as an [ADTypes.jl](https://github.com/SciML/ADTypes.jl) object: `AutoEnzyme()` (reverse mode, the
  default), `AutoForwardDiff()` or, after `import Zygote`, `AutoZygote()`. `AutoEnzyme`, `AutoForwardDiff`, `AutoZygote` and
  `DeepCompartmentModels.default_ad()` are provided. Enzyme in forward mode is not supported yet (Enzyme cannot differentiate through the solvers). Zygote is an optional dependency (package extension `DCMZygoteExt`);
  it is only advised for in-place UDEs with large networks, for which Enzyme cannot yet use the adjoint methods.

### Changed

- Batching code (`create_batches`, `take_batch`, combining batch gradients) moved from
  `gradients.jl` and `optimization.jl` to a new `src/lib/batch.jl`. `create_batches` and `take_batch`
  are still exported.
- `fit` and `update_state` were reworked: `fit` sets up parameters and state itself through
  `_setup_for_fit` and uses the new optimisation helpers. `fit` is now exported alongside
  `update_state`.
- The in-place `UniversalDiffEq` check was removed, since in-place UDEs are supported (see above).
- **Gradients are now computed with Enzyme (reverse mode) instead of Zygote**, which is several times faster
  (about 4-5x for the standard models in a micro benchmark). Zygote is no longer a dependency, and the
  `@non_differentiable`/`@ignore_derivatives` annotations were removed. The first gradient call takes longer,
  because Enzyme compiles the code for the objective.
- **Requires OrdinaryDiffEq 7 and SciMLBase 3** (compat was `6.108` and `2.152`). Enzyme 0.13.219 (EnzymeCore 0.8.27)
  does not work with the Enzyme extension of DiffEqBase 6, which OrdinaryDiffEq 6 depends on, whereas DiffEqBase 7 does.
  SciMLBase 3 no longer provides `DESolution`; the package now uses `SciMLBase.AbstractODESolution`.
- Gradients of parameters that are not used are now zero instead of `nothing` (e.g. the infusion rate `I` of a
  `UniversalDiffEq`, for which Zygote returned a meaningless nonzero value).
- In-place `UniversalDiffEq` problems now always default to `ForwardDiffSensitivity()`: Enzyme cannot differentiate
  the adjoint sensitivity algorithms of in-place problems yet.
- `solve`, `solve_for_target` on a `Population` use explicit loops instead of broadcasting over the population.

### Fixed

- `VariationalELBO(idxs; path_deriv = false)` no longer fails with `_Diagonal_pullback(::NoTangent)` (a Zygote
  error).
- `residual_error_value_and_gradient(...; mode = :reverse)` no longer fails with a `MethodError` on a conversion
  from `Float64` to `Vector{Float32}`.
- Dispatch errors in `dcm.jl`, `objectives.jl` and `ude.jl` (`1dff8ed`).
- The gradient of `MSE` is now weighted by the batch's share of the population. Before, summing the
  batch gradients made it `length(population) / length(batch)` times too large when using
  mini-batches.
- Race conditions when solving individuals in parallel. `ComponentVector` parameters are copied
  before the infusion state `I` is set, and callbacks are no longer mutated to switch
  `save_positions` when interpolating.
- Removed an unnecessary `p.I = 0` in `node.jl`.

## [1.2.3] - 2026-08-19

### Added

- Support for in-place `UniversalDiffEq` problems via `UniversalDiffEq(n; inplace = true)`.

## [1.2.2] - 2026-08-04

Not tagged in git.

### Added

- Gradient and optimisation helpers, `DeepCompartmentModel` updates and `UniversalDiffEq` fixes
  from the "latest state" commit (`32581b8`, 2026-06-03).

### Fixed

- `MOIndividual.occasions` is now typed as `Union{Nothing,Vector{Pair}}`.
- Printing a `DeepCompartmentModel` that wraps a `UniversalDiffEq` now shows the UDE type.
- The UDE fix that was applied in `dcm.jl` now lives in `ude.jl`, and a package dependency that was
  no longer needed was removed.

## [1.2.1] - 2026-05-14

### Fixed

- `VariationalELBO` now stores `mean_field`, `path_deriv` and `natural` as fields instead of only as
  type parameters, which fixes the variational ELBO.

## [1.2.0] - 2026-03-28

A large rewrite of the package internals, built up from December 2025 onwards. This release is
**breaking** for code written against 1.1.x.

### Added

- New `Individual` and `Population` implementation (`src/lib/individuals.jl`) with the types
  `BasicIndividual`, `TimeVariableIndividual` and `MOIndividual`, and the accessors `get_t` and
  `get_tx`. The initial state is now called `u0` (previously `initial`).
- Multi-output models and occasions: `MOIndividual` with occasions, `ErrorModelSet` for one error
  model per output, and `solve` that dispatches on occasions.
- `UniversalDiffEq` (`src/lib/ude.jl`) for universal differential equations, with `BasicUDE` and
  `TimeConcatUDE`, as well as `build_problem`.
- New `solve`, `solve_for_target` and `construct_p`, plus new `setup` and `setup_phi` functions,
  gradients (including multi-output gradients) and a VEM procedure (`m_step`, `optimise_omega`,
  `optimise_residual_error`).
- New objectives: `MSE`, `SSE`, `LogLikelihood` and `VariationalELBO`, with the lowercase
  functions `mse`, `sse`, `loglikelihood`, `kldivergence`, `logjoint` and `elbo`.
- New `DeepCompartmentModel` definition, with the error model API (`AbstractErrorModel`,
  `AdditiveError`, `ProportionalError`, `CombinedError`, `CustomError`, `make_dist`, `var`)
  redesigned.
- `Lux.f32`-style precision handling in `setup`.
- `DataFrames` as a weak dependency (`DCMDataFramesExt`), and the `Accessors`, `Static` and
  `ComponentArrays` packages are re-exported.
- Julia compat raised to `1.11`, with updated compat bounds: Lux 1.31, Optimisers 0.4,
  ForwardDiff 1.3, Zygote 0.7, SciMLBase 2.152 and others.
- Test directory restructured (`test/lib/`).

### Changed

- **Breaking:** the neural ODE model types (`NeuralODE`, `LowDimensionalNeuralODE`,
  `AutoEncodingNeuralODE` and their aliases) and the `init_*` initialisers (`init_theta`,
  `init_error`, `init_omega`, `init_phi`) are no longer loaded or exported. Use `UniversalDiffEq`
  and `setup` instead.
- **Breaking:** the objectives and mixed-effects code were restructured. `FO`, `FOCE`, `MeanField`,
  `FullRank`, `optimize_etas`, `constrain`, `forward`, `forward_ode` and `Parameterization` are no
  longer exported (the parameterisation types are now `Parameterisation`, `MeanVar` and `MeanSqrt`).
- The `FiniteDiff`, `Optim` and `PartialFunctions` dependencies were dropped, and `ForwardDiff`,
  `Functors`, `ThreadPools`, `InvertedIndices`, `LogExpFunctions` and `ComponentArrays` are now used
  directly.

## [1.1.1] - 2024-12-06

### Fixed

- Fixes for breaking changes introduced by 1.1.0 in the node-based models (`node.jl`,
  `low_dim_node.jl`, `auto_encoding_node.jl`) and `nn.jl`.

### Changed

- README updated to match the new version, and tests for the DCM added.

## [1.1.0] - 2024-12-05

### Added

- `SciMLBase` added to the compat entries.
- A new, more robust backend (`a2ea2bd`) with `Lux`-based models, new initialisers, mixed-effects
  helpers, node implementations, error models and DCM code.

### Changed

- Individuals no longer carry `eta` (`07f7712`).
- Dependencies simplified and old code and folders removed. Natural gradients are not supported yet.
- Function names in `dcm.jl` now match their definitions, and the DCM docstring was completed.

## [1.0.3] - 2024-07-15

### Fixed

- Another fix for the `DataFrames` package extension.

## [1.0.2] - 2024-07-15

### Added

- `load` function is part of the package.

### Changed

- README updated.

## [1.0.1] - 2024-07-15

### Added

- `DCMDataFramesExt`: a package extension to load a `DataFrame` into a `Population`
  ([#1](https://github.com/Janssena/DeepCompartmentModels.jl/issues/1)).

## [1.0.0] - 2024-05-19

First public release.

### Added

- Deep compartment models (DCM) in Julia, based on the approach of Janssen et al. (2022):
  `DeepCompartmentModel` (`DCM`), plus the node-based models `HybridDCM`, `BasicHybridModel`,
  `LowDimensionalNODE` and `LatentEncoderDecoder`.
- Objectives `SSE`, `LogLikelihood` and `VariationalELBO` (with `MeanField`, `FullRank`,
  `MonteCarlo` and `SampleAverage` options), and `fit!` for optimisation.
- `Population` and `Individual` types, error models (`Additive`, `Proportional`, `Combined`,
  `Custom`), constraints, Lux helpers, the `StandardNeuralNetwork` (`SNN`), built-in compartment
  models and dosing callbacks.
- License, README with citation information, and a `.gitignore`.

[Unreleased]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.2.1...HEAD
[1.2.3]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.2.1...43f4d22
[1.2.2]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.2.1...2576c60
[1.2.1]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Janssena/DeepCompartmentModels.jl/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Janssena/DeepCompartmentModels.jl/releases/tag/v1.0.0
