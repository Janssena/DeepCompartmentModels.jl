################################################################################
##########                             fit                             ##########
################################################################################

default_callback(epoch, loss) = println("Epoch $epoch, loss = $loss")

"""
<<<<<<< Updated upstream
    fit(obj, model, data, opt; epochs, callback, kwargs...)
    fit(obj, model, data, opt, ps, st; epochs, callback, kwargs...)
    fit(obj, model, data, opt_state, ps, st; epochs, callback, kwargs...)
=======
    fit(obj, model, population, opt; rng, T, epochs, callback, parallel, batchsize, nbatch, init_sigma, params, ...)
    fit(obj, model, population, opt, ps, st; rng, epochs, callback, parallel, batchsize, nbatch)
    fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch)
>>>>>>> Stashed changes

Optimise `model` parameters by minimising `obj` on `data` using optimizer `opt`.

When `ps` and `st` are not provided they are initialised via `setup`. When an
`Optimisers.AbstractRule` is provided instead of an `opt_state`, the optimiser
state is initialised from the rule and `ps`.

## Arguments
- `obj`: Objective function (e.g. `SSE()`, `LogLikelihood()`, `VariationalELBO(...)`).
- `model`: An `AbstractModel` (e.g. `DCM`, `SNN`, `UniversalDiffEq`).
- `data`: A `Population` or `AbstractIndividual`.
- `opt`: An `Optimisers.AbstractRule` (e.g. `Adam(1e-3)`) or pre-built optimiser state.
- `ps`: Model parameters returned by `setup`.
- `st`: Model state returned by `setup`.

## Keyword arguments
- `epochs = 100`: Number of gradient steps.
- `callback`: Called after each step as `callback(epoch, loss)`, where `loss` is the objective evaluated
  at the parameters used for that step. Defaults to printing. Use `callback = (args...) -> nothing` to
  silence it. The loss is a by-product of the gradient computation and costs no extra solve.
- `T = Float32`: Element type of the parameters created by `setup` (`Float16`, `Float32` or `Float64`). Only when
  `ps` and `st` are not given, otherwise the types of `ps` are used.
- `rng = Random.default_rng()`: Random number generator used by `setup` (when `ps` and `st` are not given)
  and, for `VariationalELBO`, to resample the noise of the variational posteriors every epoch. Pass a
  seeded `rng` to make a fit reproducible.
- `parallel = false`, `batchsize = nothing`, `nbatch = nothing`: Passed to `gradient`. `parallel = :individual`
  solves individuals on separate threads (start Julia with `-t auto`), `parallel = :batch` does so per
  mini-batch of `batchsize` individuals (at most) or in `nbatch` mini-batches; set at most one of them
  (default `batchsize = 16`). The batches are drawn using `rng`.
- `init_sigma`, `params` and, for `VariationalELBO`, `init_omega` and `scale`: passed on to `setup`, see there.
  Only when `ps` and `st` are not given.

Any other keyword throws an `ArgumentError`.

## Returns
`(opt_state, ps, st)` — the final optimiser state, parameters, and model state.

## Example
```julia
obj = SSE()
model = DCM(two_comp!, ann)
opt = Adam(1e-3)
opt_state, ps, st = fit(obj, model, population, opt; epochs=500)
```
"""
<<<<<<< Updated upstream
function fit(obj::AbstractObjective, model, data, opt::Optimisers.AbstractRule; kwargs...)
    ps, st = setup(obj, model)
    opt_state = Optimisers.setup(opt, ps)
    return fit(obj, model, data, opt_state, ps, st; kwargs...)
end

function fit(obj::AbstractObjective, model, data, opt::Optimisers.AbstractRule, ps, st; kwargs...)
    opt_state = Optimisers.setup(opt, ps)
    return fit(obj, model, data, opt_state, ps, st; kwargs...)
end

function fit(obj::AbstractObjective, model, data, opt_state::NamedTuple, ps, st; epochs = 100, callback = default_callback, kwargs...)
    _pre_compile_gradient(obj, model, data, ps, st; kwargs...)
    return _run_optimization(obj, model, data, opt_state, ps, st; epochs, callback, kwargs...)
end

function _pre_compile_gradient(obj, model, data, ps, st; kwargs...)
    data_batch, ps_batch, st_batch = take_batch([1], data, ps, st)
    printstyled("[ Info:", color=:cyan); print(" Compiling gradient...");
    st_batch = update_state(st_batch, obj, model, data_batch, ps_batch; kwargs...)
    Zygote.withgradient(p -> objective(obj, model, data_batch, p, st_batch), ps_batch)
    println(" Done! Starting optimization...")
    nothing
end

function _run_optimization(obj, model, data, opt_state, ps, st; epochs, callback, kwargs...)
    try 
=======
function fit(
        obj::AbstractObjective, model::AbstractDEModel, population::Population, opt::Optimisers.AbstractRule;
        rng::Random.AbstractRNG = Random.default_rng(), T::Type{<:Real} = Float32, epochs = 100, callback = default_callback,
        parallel = false, batchsize = nothing, nbatch = nothing, kwargs...
    )
    _check_keywords(kwargs, _setup_keywords(obj), "These are the keywords of `setup` that `fit` passes on.")
    ps, st = _setup_for_fit(obj, rng, model, population, T; kwargs...)
    opt_state = Optimisers.setup(opt, ps)
    return _fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch)
end

function fit(
        obj::AbstractObjective, model::AbstractDEModel, population::Population, opt::Optimisers.AbstractRule, ps, st;
        rng::Random.AbstractRNG = Random.default_rng(), epochs = 100, callback = default_callback,
        parallel = false, batchsize = nothing, nbatch = nothing, kwargs...
    )
    _check_keywords(kwargs, (), _PS_GIVEN_HINT)
    opt_state = Optimisers.setup(opt, ps)
    return _fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch)
end

function fit(
        obj::AbstractObjective, model::AbstractDEModel, population::Population, opt_state::NamedTuple, ps, st;
        rng::Random.AbstractRNG = Random.default_rng(), epochs = 100, callback = default_callback,
        parallel = false, batchsize = nothing, nbatch = nothing, kwargs...
    )
    _check_keywords(kwargs, (), _PS_GIVEN_HINT)
    return _fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch)
end

# Keywords of `setup` that `fit` passes on, per type of objective.
_setup_keywords(::FixedObjective) = (:init_sigma, :params)
_setup_keywords(::MixedObjective) = (:init_omega, :init_sigma, :params, :scale)
_setup_keywords(::AbstractObjective) = ()

const _PS_GIVEN_HINT = "Keywords of `setup` (`T`, `init_sigma`, `params`, ...) are only used when `ps` and `st` are not given."

function _check_keywords(kwargs, allowed, hint)
    unknown = filter(!in(allowed), keys(kwargs))
    isempty(unknown) && return nothing
    allowed_str = isempty(allowed) ? "" : " Accepted here: " * join(string.(allowed), ", ") * "."
    throw(ArgumentError("`fit` got unsupported keyword argument(s): " * join(string.(unknown), ", ") * ". " * hint * allowed_str))
end

_setup_for_fit(obj::FixedObjective, rng, model, population, ::Type{T}; init_sigma = nothing, params = MeanSqrt()) where T =
    setup(obj, rng, model, T; init_sigma, params)
_setup_for_fit(obj::MixedObjective, rng, model, population, ::Type{T}; kwargs...) where T =
    setup(obj, rng, model, population, T; kwargs...)

function _fit(obj, model, population, opt_state, ps, st; rng, epochs, callback, parallel, batchsize, nbatch)
    try
>>>>>>> Stashed changes
        for epoch in 1:epochs
            opt_state, st = update_state(opt_state, st, obj, model, data, ps; kwargs...)
            loss, ∇ = Zygote.withgradient(p -> objective(obj, model, data, p, st), ps)
            callback(epoch, loss)
            opt_state, ps = Optimisers.update(opt_state, ps, first(∇))
            ensure_posdef_phi!(obj, ps)
        end

        return opt_state, ps, st
    catch e
        if e isa InterruptException
            println()
            @info "User interrupted, gracefully exiting optimization..."
            return opt_state, ps, st
        else 
            if obj isa VariationalELBO && !_phis_are_valid(constrain(model, ps).phi)
                @error "Some of the Variational posteriors are not positive definite or have variances close to zero. This can mean that the posterior of one or more random-effects are shrinking to zero for some Individuals. Reducing the learning rate can help to address this issue."
                return nothing
            else
                throw(e) # Would be nice to have this send a more user friendly message.
            end
        end
    end
end

_phis_are_valid(phi::NamedTuple) = true
_phis_are_valid(phi::NamedTuple{(:μ,:L), <:Any}) = 
    all(map(L -> isposdef(L * L') && det(L * L') > 0, phi.L))


ensure_posdef_phi!(::AbstractObjective, ps) = nothing
function ensure_posdef_phi!(::VariationalELBO, ps) # In some cases the diagonals of the variational posterior Ls will be negative, this should not be the case.
    if :L in keys(ps.phi)
        for i in eachindex(ps.phi.L)
            if any(diag(ps.phi.L[i]) .< 0)
                ps.phi.L[i] .= cholesky(ps.phi.L[i] * ps.phi.L[i]').L
            end
        end
    end
    
    return nothing
end

# function _run_optimization_stochastic(obj, model, data::Population, opt_state, ps, st; M::Int = Int(round(length(data) * 0.25)), epochs, callback, kwargs...)
#     try 
#         for epoch in 1:epochs
#             idxs = rand(eachindex(data), M)
#             data_batch, ps_batch, st_batch = take_batch(idxs, data, ps, st)
#             opt_state, st_batch = update_state(opt_state, st_batch, obj, model, data, ps; kwargs...)
#             # TODO: translate this gradient into the full gradient
#             loss, ∇ = Zygote.withgradient(p -> objective(obj, model, data_batch, p, st_batch), ps_batch)
#             callback(epoch, loss)
#             opt_state, ps = Optimisers.update(opt_state, ps, first(∇))
#         end

#         return opt_state, ps, st
#     catch e
#         if e isa InterruptException
#             println()
#             @info "User interrupted, gracefully exiting optimization..."
#             return opt_state, ps, st
#         else 
#             throw(e)
#         end
#     end
# end

################################################################################
##########                     Minibatch training                     ##########
################################################################################

function create_batches(M::Int, n::Int)
    batches = [collect(i:M:n) for i in 1:M]
    idxs = Random.shuffle(1:n)
    return map(Base.Fix1(getindex, idxs), batches) # randomize indexes in each batch
end

create_batches(M::Int, population::Population) = create_batches(M, length(population))

function take_batch(idxs::AbstractVector{<:Int}, population::Population, ps, st)
    ps_new, st_new = ps, st

    if :phi in keys(ps) && :phi in keys(st)
        ps_new = merge(ps, (phi = _take_batch_from_phi(idxs, ps.phi), ))
        st_new = merge(st, (phi = _take_batch_from_phi(idxs, st.phi), ))
    end

    return population[idxs], ps_new, st_new
end

# When phi is from the parameters
_take_batch_from_phi(_, phi::@NamedTuple{}) = phi # FO and FOCE
_take_batch_from_phi(idxs, phi::NamedTuple{<:Any,<:Tuple{<:AbstractVector,<:AbstractVector}}) = 
    NamedTuple{keys(phi)}(map(Base.Fix2(getindex, idxs), values(phi))) # Variational Inference

# When phi is from the state
_take_batch_from_phi(idxs, phi::NamedTuple{(:mask,:eta),<:Tuple{<:AbstractVector{<:Real},<:AbstractMatrix}}) = 
    (mask = phi.mask, eta = phi.eta[idxs], )

_take_batch_from_phi(idxs, phi::NamedTuple{(:mask,:eta),<:Tuple{<:AbstractMatrix{<:Real},<:AbstractMatrix}}) = 
    (mask = phi.mask, eta = phi.eta[:, idxs], )

_take_batch_from_phi(idxs, phi::NamedTuple{(:epsilon,:mask),<:Tuple{<:AbstractMatrix{<:Real},<:AbstractMatrix}}) = 
    (epsilon = phi.epsilon[idxs, :], mask = phi.mask, )

_take_batch_from_phi(idxs, phi::NamedTuple{(:epsilon,:mask),<:Tuple{<:AbstractArray{<:Real, 3},<:AbstractMatrix}}) = 
    (epsilon = phi.epsilon[:, idxs, :], mask = phi.mask, )
    
# TODO: fit_batched # Here we thread over mini-batches of data

# TODO: fit_stochastic # Here we take a random subsample of the data and train on only that. Check on how to do this with VI

################################################################################
##########                        update_state                        ##########
################################################################################

update_state(st, ::AbstractObjective, model, data, ps; kwargs...) = st
update_state(opt_state, st, obj::AbstractObjective, model, data, ps; kwargs...) = 
    update_opt_state(opt_state, ps, data), update_state(st, obj, model, data, ps; kwargs...)

# Need to use a population here to make sure the shape of etas is correct for 
# evaluation of the gradient. Also optimize_etas for populations uses constrain 
# internally
function update_state(st, ::FOCE, model, population::Population, ps)
    etas = optimize_etas(model, population, ps, st)
    return _etas_to_state(st, etas)
end

update_state(st, ::VariationalELBO, model, data, ps; num_samples = 1) = 
    update_state_epsilon(st, num_samples)


function update_opt_state(opt_state, ps, data)
    update_opt_state!(opt_state, ps, data)
    return opt_state
end

# TODO: remove when implementing natural gradients
update_opt_state!(opt_state, ps, data) = nothing

update_state(::VariationalELBO, st, num_samples = 1) = update_state_epsilon(st, num_samples)

"""
    update_state_epsilon(rng, st; n)

Sets each epsilon in st to `n` samples from N(0, I).
"""
function update_state_epsilon(st::NamedTuple, num_samples = 1) 
    # TODO: what if epsilon is not the last element in keypath here?
    return fmapstructure_with_path(
        (path, x) -> :epsilon in path ? randn(Random.GLOBAL_RNG, eltype(x), size(x)[begin:end-1]..., num_samples) : x, st; cache = nothing)
        # (path, x) -> :epsilon in path ? _randn_epsilon(x, num_samples) : x, st; cache = nothing)
end