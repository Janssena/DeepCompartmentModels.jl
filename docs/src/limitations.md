# Status and limitations

```@meta
CurrentModule = DeepCompartmentModels
```

The tutorials only use features that work in the current source, and every example on this site is
executed when the documentation is built. While preparing them, several parts of the interface turned
out not to work. They are listed here so that nobody has to rediscover them. Each entry gives the
symptom and the cause found in the source, to make fixing them quick.

!!! note "Training loop"
    There is no `fit` function. `src/lib/optimization.jl` defines one, but the file is not included by
    `src/DeepCompartmentModels.jl`, and it uses `setup(objective, model)` without an `rng`, which no longer
    exists. `fit` in the package namespace is `Distributions.fit`. The tutorials therefore train with a
    short loop around [`gradient`](@ref) and `Optimisers.update`, which is what `fit` would do anyway.

## Not working at the moment

| feature | symptom | cause |
|:--|:--|:--|
| `load(df, covariates)` | `MethodError: no method matching Individual(::Vector{Float32}, ...)` | `ext/DCMDataFramesExt.jl` calls `Individual(x, t, y, cb; id = ...)`, but the constructor is `Individual(id, x, t, y, cb, T)`. It also requires an `MDV` column. The [data tutorial](generated/05-loading-data.md) shows the equivalent loop. |
| Universal differential equations (`UniversalDiffEq`) | `MethodError: no method matching build_problem(...)` when computing predictions or gradients | `build_problem` is defined with four arguments `(ude, model, ps, st)` but called with three in `src/lib/ude.jl`. `setup` works. |
| Several outputs (`MOIndividual`, `ErrorModelSet`, `target = [2, 3]`) | `MethodError: no method matching get_x(::Population{MOIndividual...})` | `get_x` and the related methods in `src/lib/population.jl` are only defined for `BasicIndividual` and `TimeVariableIndividual` populations. |
| Occasions (`occasions = [0.0 => 23.0, ...]`) | `MethodError: no method matching _take_target(::Vector{ODESolution}, ...)` in `predict` | the solve returns one solution per occasion, and `_take_target` has no method that takes the individual together with a vector of solutions. |
| Time-varying covariates (`TimeVariableIndividual`) | `BoundsError` in `one_comp_abs!` when computing the objective | not investigated in depth. `unpack(p::AbstractMatrix, t)` expects the first row of the parameter matrix to hold time points and drops it, which suggests the matrix built for these individuals does not follow that layout. |
| `SNN` / `StandardNeuralNetwork` | `MethodError: no method matching setup(::Xoshiro, ::StandardNeuralNetwork)` | `setup(rng, model)` is only defined for `DeepCompartmentModel`. |
| `interpret_branch` | `MethodError` | the method is restricted to `Lux.AbstractLuxContainerLayer`, and a `Lux.Chain` is a wrapper layer in current Lux versions. It also refers to `BranchLayer` without the `Lux.` prefix. The [covariate tutorial](generated/04-covariate-effects.md) uses a short helper instead. |

## Rough edges

- `Combine(1 => [1], 2 => [2])`, the form shown in its docstring, fails with a `MethodError`:
  the pairs are passed on as a tuple instead of being splatted. Use the form with the output
  dimension, `Combine(2, 1 => [1], 2 => [2])`.
- `SingleHeadedBranch` and `MultiHeadedBranch` default to `activation = swish`, which is not in scope
  inside the package. Always pass `activation = Lux.swish` (or another activation).
- `softplus` and `tanh` are not re-exported, use `Lux.softplus`. (Julia's own `tanh` is fine.)
- A branch that feeds `k` outputs of `Combine` must have `k` output rows, so use
  `MultiHeadedBranch(idx, neurons, k)`.
- The time span of an `ODEProblem` has to start before the first dose (the default of
  `DCM(ode_fn, ...)` is `-0.1`), otherwise a dose at `t = 0` is silently never given and all
  predictions are zero.
- The ODE solver logs a `dt_epsilon` warning for every solve when the initial state is zero. Setting an
  initial step with `ODEProblem(...; dt = 0.01f0)` avoids thousands of warnings during training.
- `make_etas` is exported but not defined.
- `predict(model, individual, ps, st)` for a single individual fails when `ps` contains random effects for the
  whole population. Slice the parameters with `take_batch` and predict for a one-element population, as in the
  [mixed-effects tutorial](generated/02-mixed-effects.md).
- `target` defaults to `1`. For models with an absorption compartment the observed concentration is
  state 2, so set `target = 2`.
- Many exported functions have no docstring yet (see the [API reference](api.md)).
- The tests in `test/lib` refer to an older interface (`make_etas`, `FullRank()`, `update_state_epsilon`) and do not
  run against the current source.

## Notebooks that are kept but not built

The notebooks in `tutorials/` that are not part of this site (`03-universal-diffeq`, `08-end-to-end`,
`09-counterfactual`, `10-pkpd` and `11-occasions`) were written for an earlier interface and rely on the
features in the first table. They are left in place and can be brought back by listing them in `TUTORIALS`
in `docs/make.jl` once those features work again.
