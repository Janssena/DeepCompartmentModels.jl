################################################################################
##########                             fit                             ##########
################################################################################

default_callback(epoch, loss) = println("Epoch $epoch, loss = $loss")

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

create_batches(M::Int, population::Population) = create_batches(M, population.count)

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

update_state(::VariationalELBO, st, num_samples = 1) = update_state_epsilon(st, num_samples)

"""
    update_state_epsilon(rng, st; n)

Sets each epsilon in st to `n` samples from N(0, I).
"""
function update_state_epsilon(st::NamedTuple, num_samples = 1) 
    # TODO: what if epsilon is not the last element in keypath here?
    return Functors.fmapstructure_with_path(
        (path, x) -> :epsilon in path ? randn(Random.GLOBAL_RNG, eltype(x), size(x)[begin:end-1]..., num_samples) : x, st; cache = nothing)
        # (path, x) -> :epsilon in path ? _randn_epsilon(x, num_samples) : x, st; cache = nothing)
end