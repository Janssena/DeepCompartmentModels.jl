struct VAENODE <: Lux.AbstractLuxContainerLayer{(:encoder,:node,:decoder)} # Lu et al
    encoder
    node
    decoder
    num_latent::Int
end


AutoEncodingNeuralODE(encoder::AbstractLuxLayer, node::AbstractLuxLayer, decoder::AbstractLuxLayer, ::Type{T}=Float32; kwargs...) where T =
    AutoEncodingNeuralODE(encoder, node, decoder, ImplicitError(), T; kwargs...)

function AutoEncodingNeuralODE(encoder::AbstractLuxLayer, node::AbstractLuxLayer, decoder::AbstractLuxLayer, error::AbstractErrorModel, ::Type{T}=Float32; num_latent = nothing, kwargs...) where T
    if num_latent == nothing
        num_latent = _estimate_num_partials_node(node)
    end
    model = VAENODE(encoder, node, decoder, num_latent)
    dzdt(z, p, t; model) = model(:node, z, p.weights) + [p.I; zeros(T, num_latent-1)]
    return NeuralODE(dzdt, model, num_latent, error, T)
end

function init_theta(model::NeuralODE{T,P,M,E,S}) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S}
    # Note: model.node here is a StatefulLuxLayer
    ps_, _ = Lux.setup(Random.GLOBAL_RNG, model.node.model)
    ps = (
        encoder = ps_.encoder,
        node = ComponentVector((weights = ps_.node, I = zero(T), )),
        decoder = ps_.decoder
    )
    return ps, model.node.st
end

setup(::FixedObjective, ::NeuralODE{T,P,M,E,S}) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S} =
    throw(ErrorException("`setup(::FixedObjective, ::AutoEncodingNeuralODE)` is not defined. Call `setup(::FixedObjective, ::AutoEncodingNeuralODE, ::Population)` instead."))

function setup(obj::FixedObjective, model::NeuralODE{T,P,M,E,S}, population::Population) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S}
    ps, st = _setup(obj, model)
    return ps, merge(st, (phi = (epsilon = randn(T, model.node.model.num_latent, length(population)), ), ))
end

################################################################################
##########                        Model API                           ##########
################################################################################

function forward(::AbstractObjective, model::NeuralODE{T,P,M,E,S}, population::Population, ps, st; kwargs...) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S}
    z₀, st = predict_de_parameters(FixedObjective, model, population, ps, st)
    return forward_ode_with_dv(model, population, z₀, ps; kwargs...)
end

predict(model::NeuralODE{T,P,M,E,S}, population::Population, ps::NamedTuple, st; kwargs...) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S} = 
    forward(SSE(), model, population, ps, st; kwargs...)

function predict_de_parameters(::Union{FixedObjective, Type{FixedObjective}}, model::NeuralODE{T,P,M,E,S}, population::Population, ps, st) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S}
    num_latent = model.node.model.num_latent
    𝜙 = model.node(:encoder, get_x(population), ps.theta.encoder)
    Z₀ = 𝜙[1:num_latent, :] .+ st.phi.epsilon .* softplus.(𝜙[num_latent+1:end, :])
    return Z₀, _z0_to_state(st, Z₀)
end

_z0_to_state(st, Z₀) = merge(st, (phi = merge(st.phi, (z0 = Z₀, )), ))

forward_ode_with_dv(model::NeuralODE{T,P,M,E,S}, population::Population, z₀::AbstractVector{<:AbstractMatrix}, ps::NamedTuple) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S} = 
    forward_ode_with_dv.((model, ), (population, ), z₀, (ps, ))

forward_ode_with_dv(model::NeuralODE{T,P,M,E,S}, population::Population, z₀::AbstractMatrix, ps::NamedTuple) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S} = 
    forward_ode_with_dv.((model, ), population, eachcol(z₀), (ps, ))

function forward_ode_with_dv(model::NeuralODE{T,P,M,E,S}, individual::AbstractIndividual, z₀::AbstractVector, ps::NamedTuple; sensealg = model.sensealg, kwargs...) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S}
    @ignore_derivatives ps.theta.node.I = zero(eltype(ps.theta.node))
    prob = _set_u0(model.problem, z₀)
    zₜ = Array(
        forward_ode(prob, individual, ps.theta.node; sensealg, force_dtmin = true, kwargs...)
    )
    return vec(model.node(:decoder, zₜ, ps.theta.decoder))
end

_set_u0(prob, u0) = remake(prob, u0 = collect(u0))

Base.show(io::IO, model::NeuralODE{T,P,M,E,S}) where {T,P,M<:StatefulLuxLayer{<:Any,<:VAENODE},E,S} = print(io, "AutoEncodingNeuralODE{$T, $(model.error)}")







# LatentEncoderDecoder(dzdt::Function, model::LatentEncoderDecoder, num_latent::Int; kwargs...) = HybridDCM(dzdt, model, num_latent; dv_compartment=1:num_latent, kwargs...)

# function forward(m::HybridDCM{O,D,M,P,S,R}, population::Population, p::NamedTuple; kwargs...) where {O<:FixedObjective,D,M<:LatentEncoderDecoder,P,S,R}
#     k = m.model.num_latent - 1
#     p_node = _add_I(p.theta.node)
#     𝜙, st = m.model.encoder(get_x(population), p.theta.encoder, m.st.encoder)
#     μ = 𝜙[1:k, :]
#     σ = softplus.(𝜙[k+1:end, :])
#     ϵ = randn(eltype(𝜙), k, size(𝜙, 2))
#     z0 = vcat(zeros(eltype(𝜙), 1, size(𝜙, 2)), μ + ϵ .* σ)
#     zₜ = forward_ode.((m,), population, eachcol(z0), (p_node,); kwargs...)
#     ŷ = first.(m.model.decoder.(zₜ, (p.theta.decoder,), (m.st.decoder,)))
#     return getindex.(ŷ, 1, :), st # decoder outputs matrices, so need to convert to vectors
# end

# function forward(m::HybridDCM{O,D,M,P,S,R}, individual::AbstractIndividual, p::NamedTuple; kwargs...) where {O<:FixedObjective,D,M<:LatentEncoderDecoder,P,S,R}
#     k = m.model.num_latent - 1
#     p_node = _add_I(p.theta.node)
#     𝜙, st_encoder = m.model.encoder(individual.x, p.theta.encoder, m.st.encoder)
#     μ = 𝜙[1:k]
#     σ = softplus.(𝜙[k+1:end])
#     ϵ = randn(eltype(𝜙), k)
#     z0 = [zero(eltype(𝜙)); μ + ϵ .* σ]
#     zₜ = forward_ode(m, individual, z0, p_node; kwargs...)
#     ŷ, st_decoder = m.model.decoder(zₜ, p.theta.decoder, m.st.decoder)
#     return ŷ[1, :], (encoder = st_encoder, decoder = st_decoder) # decoder outputs matrices, so need to convert to vectors
# end


# # TODO: Make version using VariationalELBO
# # function encode_z0(model, population)
# # function solve_and_decode(model, population, z0, p_node, p)
# #     # ...
# # end

# function forward_ode(m::HybridDCM{O,D,M,P,S,R}, individual::AbstractIndividual, z0::AbstractVector, p_node::ComponentArray; kwargs...) where {O,D,M<:LatentEncoderDecoder,P,S,R} 
#     @ignore_derivatives p_node.I = zero(p_node.I)
#     prob = remake(m.problem, u0 = Vector(z0))
#     return forward_ode(prob, individual, p_node, Val(true); dv_idx=m.dv_compartment, kwargs...)
# end

# forward_adjoint(m::HybridDCM{O,D,M,P,S,R}, args...) where {O,D,M<:LatentEncoderDecoder,P,S,R} = forward(m, args...; sensealg = InterpolatingAdjoint(; autojacvec = ReverseDiffVJP()))

# Base.show(io::IO, m::HybridDCM{O,D,M,P,S,R}) where {O,D,M<:LatentEncoderDecoder,P,S,R} = print(io, "LatentEncoderDecoder{num_latent = $(m.model.num_latent), $(m.objective)}")



# # """This version only takes a single sample from the VAE, there should be a specific version for the ELBO."""
# # # function forward(m::CustomLuxModel{O,M,P}, population::Population, p_::ComponentArray, st; kwargs...) where {O<:FixedObjective,M<:LatentEncoderDecoder,P}
# # function forward(m::LatentEncoderDecoder, population::Population, p::NamedTuple, st; kwargs...)
# #     p_node = ComponentVector((theta = p.theta.node, I = 0.f0,)) # Only using a ComponentVector for the NODE theta saves 0.1M allocations
# #     𝜙, _ = m.encoder(population.x, p.theta.encoder, st.encoder)
# #     k = Integer(size(𝜙, 1) / 2)
# #     z₀ = vcat(zero(𝜙[1:1, :]), 𝜙[1:k, :] + randn(eltype(𝜙), k, size(𝜙, 2)) .* softplus.(𝜙[k+1:end, :])) # How are we going to do muliple samples?)
# #     zₜ = forward_ode.((m,), population, eachcol(z₀), (p_node,), (st,); kwargs...)
# #     ŷ = first.(m.decoder.(zₜ, (p.theta.decoder,), (st.decoder,)))
# #     return getindex.(ŷ, 1, :) # decoder outputs matrices, so need to convert to vectors
# # end

# # forward_adjoint(m::LatentEncoderDecoder, args...) = forward(m, args...; sensealg = InterpolatingAdjoint(; autojacvec = ReverseDiffVJP())) 

# # """Ideally this is re-usable for all CustomLuxModels based on NeuralODEs"""
# # # function forward_ode(m::CustomLuxModel{O,M,P}, individual::AbstractIndividual, z0, p, st; get_z = false, saveat = is_timevariable(individual) ? individual.t.y : individual.t, interpolate=false, sensealg=nothing) where {O<:FixedObjective,M<:LatentEncoderDecoder,P}
# # function forward_ode(m, individual::AbstractIndividual, z0, p, st; get_z = true, saveat = is_timevariable(individual) ? individual.t.y : individual.t, interpolate=false, sensealg=nothing)
# #     @ignore_derivatives p.I = zero(p.I)
# #     saveat_ = interpolate ? empty(saveat) : saveat
# #     node = StatefulLuxLayer(m.node, nothing, st.node)
# #     dzdt(z, p, t; model=node) = model(z, p.theta) + vcat(p.I, zeros(eltype(z), length(z) - 1))
    
# #     ff = ODEFunction{false}(dzdt; tgrad = basic_tgrad)
# #     prob = ODEProblem{false}(ff, Vector(z0), (-0.1f0, maximum(saveat)), p)
    
# #     interpolate && (individual.callback.save_positions .= 1)
# #     sol = solve(prob, Tsit5(),
# #         saveat = saveat_, callback=individual.callback, 
# #         tstops=individual.callback.condition.times, sensealg=sensealg
# #     )
# #     interpolate && (individual.callback.save_positions .= 0)
# #     return get_z ? reduce(hcat, sol.u) : sol
# # end