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