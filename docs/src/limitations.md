# Status and limitations

```@meta
CurrentModule = DeepCompartmentModels
```

The tutorials only use features that work in the current source, and every example on this site is
executed when the documentation is built. While preparing them, several parts of the interface turned
out not to work. They are listed here so that nobody has to rediscover them. Each entry gives the
symptom and the cause found in the source, to make fixing them quick.

!!! note "Training loop"
    [`fit`](@ref) runs the training loop (setup, [`gradient`](@ref), `Optimisers.update`). It supports the
    fixed effect objectives (`MSE`, `SSE`, `LogLikelihood`) and `VariationalELBO`, see the entries below
    for the exceptions. The tutorials write the loop out explicitly, because that shows what each step
    does; `fit` does the same.

## Not working at the moment

| feature | symptom | cause |
|:--|:--|:--|
| `load(df, covariates)` | `MethodError: no method matching Individual(::Vector{Float32}, ...)` | `ext/DCMDataFramesExt.jl` calls `Individual(x, t, y, cb; id = ...)`, but the constructor is `Individual(id, x, t, y, cb, T)`. It also requires an `MDV` column. The [data tutorial](generated/05-loading-data.md) shows the equivalent loop. |
| In-place `UniversalDiffEq` with an adjoint sensitivity algorithm | `MethodError: no method matching augmented_primal(...)` from `gradient` | Enzyme's rule for the adjoint methods of SciMLSensitivity does not support in-place right-hand sides yet. In-place UDEs therefore default to `ForwardDiffSensitivity()`, which works but is slow for large networks. Out-of-place UDEs are not affected. For a large in-place UDE, `ad = AutoZygote()` (after `import Zygote`; `using Zygote` clashes with our `gradient`) works with the adjoint methods, but writing the network as an out-of-place UDE and using Enzyme is preferred. |
| `VariationalELBO(idxs; natural = true)` | `MethodError: no method matching _value_and_gradient(::VariationalELBO{False, True, True}, ...)` from `gradient` and `fit` | there are gradient methods for `natural = false` only. The objective itself also fails: `kldivergence` calls `getprior`, which is not defined. |
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
- For a `UniversalDiffEq` with at least 512 parameters the default `sensealg` is `GaussAdjoint(; autojacvec = ReverseDiffVJP(false))`,
  which is much faster than forward sensitivities for large networks (5x at ~1,000 and 50x at ~4,000 parameters). Its gradients
  differ from the forward ones by ~2e-4 (relative), and Lux logs a one-time warning about `ReverseDiff.TrackedReal` inputs.
  Pass `sensealg = ForwardDiffSensitivity()` to use forward sensitivities regardless of the size. Do not use
  `ReverseDiffVJP(true)` (compiled tape) with in-place UDEs: it returned a wrong gradient in a test. `InterpolatingAdjoint(; autojacvec = EnzymeVJP())`
  is another 1.5-2.5x faster and gave correct gradients, at the cost of a longer compilation.
- The ODE solver logs a `dt_epsilon` warning for every solve when the initial state is zero. Setting an
  initial step with `ODEProblem(...; dt = 0.01f0)` avoids thousands of warnings during training.
- `make_etas` is exported but not defined.
- `predict(model, individual, ps, st)` for a single individual needs the parameters of that individual only.
  When `ps` holds the random effects of a whole population, select them with `ps_i, st_i = take_batch(ps, st, [i])`
  (an `ArgumentError` explains this otherwise), as in the [mixed-effects tutorial](generated/02-mixed-effects.md).
- `target` defaults to `1`. For models with an absorption compartment the observed concentration is
  state 2, so set `target = 2`.
- Many exported functions have no docstring yet (see the [API reference](api.md)).
- `UniversalDiffEq` models only support the fixed effect objectives (`MSE`, `SSE`, `LogLikelihood`), not
  `VariationalELBO`.
- Five test files in `test/lib` (`objectives`, `initializers`, `mixed_effects`, `model`, `dcm`) refer to an older
  interface (`make_etas`, `init_omega`, `forward_ode`, ...) and do not run against the current source. They are
  skipped by `runtests.jl` unless `DCM_LEGACY_TESTS=true`.
