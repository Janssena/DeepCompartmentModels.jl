import Distributions: fit

default_callback(epoch, loss) = println("Epoch $epoch, loss = $loss")

"""
    fit(obj, model, population, opt; rng, epochs, callback, parallel, batchsize, nbatch, ...)
    fit(obj, model, population, opt, ps, st; rng, epochs, callback, parallel, batchsize, nbatch, ...)
    fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch, ...)

Optimise the parameters of `model` by minimising `obj` on `population` with the optimiser `opt`.
Each epoch is one gradient step on the whole population. `fit` is a thin wrapper around
`setup`, `gradient` and `Optimisers.update`:

```julia
ps, st = setup(obj, rng, model)
opt_state = Optimisers.setup(opt, ps)
for epoch in 1:epochs
    st = update_state(rng, st, obj, model, population, ps) # only does something for VariationalELBO
    ∇ = gradient(obj, model, population, ps, st; parallel, batchsize, nbatch, rng)
    opt_state, ps = Optimisers.update(opt_state, ps, ∇)
end
```

When `ps` and `st` are not provided they are created with `setup` (using `rng`). Passing `ps` and
`st` allows continuing a previous fit. Instead of an optimiser rule (e.g. `Optimisers.Adam(1e-3)`)
a previously returned `opt_state` can be passed.

Supported objectives are the fixed effect objectives (`MSE`, `SSE`, `LogLikelihood`) and
`VariationalELBO` (without `natural = true`). Models with a `UniversalDiffEq` only support the
fixed effect objectives.

## Arguments
- `obj`: Objective, e.g. `SSE()`, `LogLikelihood()`, `VariationalELBO(idxs)`.
- `model`: A `DeepCompartmentModel`.
- `population`: A `Population`.
- `opt`: An `Optimisers.AbstractRule` or an optimiser state returned by an earlier call.
- `ps`, `st`: Parameters and state as returned by `setup`.

## Keyword arguments
- `epochs = 100`: Number of gradient steps.
- `callback`: Called after each step as `callback(epoch, loss)`, where `loss` is the objective evaluated
  at the parameters used for that step. Defaults to printing. Use `callback = (args...) -> nothing` to
  silence it. The loss is a by-product of the gradient computation and costs no extra solve.
- `rng = Random.default_rng()`: Random number generator used by `setup` (when `ps` and `st` are not given)
  and, for `VariationalELBO`, to resample the noise of the variational posteriors every epoch. Pass a
  seeded `rng` to make a fit reproducible.
- `parallel = false`, `batchsize = nothing`, `nbatch = nothing`: Passed to `gradient`. `parallel = :individual`
  solves individuals on separate threads (start Julia with `-t auto`), `parallel = :batch` does so per
  mini-batch of `batchsize` individuals (at most) or in `nbatch` mini-batches; set at most one of them
  (default `batchsize = 16`). The batches are drawn using `rng`.
- Other keywords (e.g. `init_sigma`, `init_omega`, `params`) are passed on to `setup`.

## Returns
`(opt_state, ps, st)`. When interrupted (Ctrl-C) the current state is returned.

## Example
```julia
dcm = DeepCompartmentModel(UniversalDiffEq(1), Lux.Chain(Lux.Dense(1, 16, tanh), Lux.Dense(16, 1)))
opt_state, ps, st = fit(SSE(), dcm, population, Optimisers.Adam(3f-3); epochs = 500)
predict(dcm, population[1], ps, st)
```
"""
function fit(
        obj::AbstractObjective, model::AbstractDEModel, population::Population, opt::Optimisers.AbstractRule;
        rng::Random.AbstractRNG = Random.default_rng(), epochs = 100, callback = default_callback,
        parallel = false, batchsize = nothing, nbatch = nothing, kwargs...
    )
    ps, st = _setup_for_fit(obj, rng, model, population; kwargs...)
    opt_state = Optimisers.setup(opt, ps)
    return _fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch, kwargs...)
end

function fit(
        obj::AbstractObjective, model::AbstractDEModel, population::Population, opt::Optimisers.AbstractRule, ps, st;
        rng::Random.AbstractRNG = Random.default_rng(), epochs = 100, callback = default_callback,
        parallel = false, batchsize = nothing, nbatch = nothing, kwargs...
    )
    opt_state = Optimisers.setup(opt, ps)
    return _fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch, kwargs...)
end

fit(
    obj::AbstractObjective, model::AbstractDEModel, population::Population, opt_state::NamedTuple, ps, st;
    rng::Random.AbstractRNG = Random.default_rng(), epochs = 100, callback = default_callback,
    parallel = false, batchsize = nothing, nbatch = nothing, kwargs...
) = _fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch, kwargs...)

_setup_for_fit(obj::FixedObjective, rng, model, population; init_sigma = nothing, params = MeanSqrt(), kwargs...) =
    setup(obj, rng, model; init_sigma, params)
_setup_for_fit(obj::MixedObjective, rng, model, population; kwargs...) =
    setup(obj, rng, model, population; kwargs...)

function _fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch, kwargs...)
    try
        for epoch in 1:epochs
            st = update_state(rng, st, obj, model, population, ps)
            loss, ∇ = value_and_gradient(obj, model, population, ps, st; parallel, batchsize, nbatch, rng)
            callback(epoch, loss)
            opt_state, ps = Optimisers.update(opt_state, ps, ∇)
            ensure_posdef_phi!(obj, ps)
        end

        return opt_state, ps, st
    catch e
        if e isa InterruptException
            println()
            @info "User interrupted, gracefully exiting optimization..."
            return opt_state, ps, st
        elseif obj isa VariationalELBO && !_phis_are_valid(ps.phi)
            @error "Some of the variational posteriors are not positive definite or have variances close to zero. This can mean that the posterior of one or more random effects is shrinking to zero for some individuals. Reducing the learning rate can help to address this issue."
            return nothing
        else
            rethrow()
        end
    end
end

_phis_are_valid(phi::NamedTuple) = true
_phis_are_valid(phi::NamedTuple{(:μ,:L), <:Any}) =
    all(map(L -> isposdef(L * L') && det(L * L') > 0, phi.L))

# In some cases the diagonals of the Cholesky factors of the variational posteriors become negative.
ensure_posdef_phi!(::AbstractObjective, ps) = nothing
function ensure_posdef_phi!(::VariationalELBO, ps)
    if :L in keys(ps.phi)
        for i in eachindex(ps.phi.L)
            if any(diag(ps.phi.L[i]) .< 0)
                ps.phi.L[i] .= cholesky(ps.phi.L[i] * ps.phi.L[i]').L
            end
        end
    end

    return nothing
end

################################################################################
##########                        update_state                        ##########
################################################################################

"""
    update_state([rng,] st, obj, model, population, ps)

Hook for objectives that need to change the state every epoch. Returns the new state. Only does
something for `VariationalELBO`, where the noise `ϵ` used for the reparameterisation of the variational
posteriors is resampled using `rng` (default `Random.default_rng()`). `st` itself is not modified.
"""
update_state(rng::Random.AbstractRNG, st, ::AbstractObjective, model, population, ps; kwargs...) = st
update_state(rng::Random.AbstractRNG, st, ::VariationalELBO, model, population, ps; kwargs...) = update_state_epsilon(rng, st)
update_state(st, obj::AbstractObjective, model, population, ps; kwargs...) =
    update_state(Random.default_rng(), st, obj, model, population, ps; kwargs...)

"""
    update_state_epsilon([rng,] st)

Resamples each epsilon in `st` (the noise used for the reparameterisation of the variational posteriors) from N(0, I)
using `rng`. Returns a new state, `st` is not modified.
"""
update_state_epsilon(rng::Random.AbstractRNG, st::NamedTuple) = fmapstructure_with_path(
    (path, x) -> :epsilon in path ? randn(rng, eltype(x), size(x)) : x, st; cache = nothing)
update_state_epsilon(st::NamedTuple) = update_state_epsilon(Random.default_rng(), st)
