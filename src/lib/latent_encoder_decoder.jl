struct LatentEncoderDecoder <: Lux.AbstractExplicitContainerLayer{(:encoder,:node,:decoder)} # Lu et al
    encoder
    node
    decoder
    num_latent::Int
end


function LatentEncoderDecoder(encoder, node, decoder; num_latent = nothing, kwargs...)
    if num_latent == nothing # TODO: make this fool proof
        input_layer = findfirst(l -> l isa Dense, node.layers)
        num_latent = node.layers[input_layer].in_dims
    end
    model = LatentEncoderDecoder(encoder, node, decoder, num_latent)
    dzdt(z, p, t; model) = model(z, p.weights) + [p.I; zeros(Float32, num_latent-1)]
    return LatentEncoderDecoder(dzdt, model, num_latent)
end

LatentEncoderDecoder(dzdt::Function, model::LatentEncoderDecoder, num_latent::Int; kwargs...) = HybridDCM(dzdt, model, num_latent; dv_compartment=1:num_latent, kwargs...)

function forward(m::HybridDCM{O,D,M,P,S,R}, population::Population, p::NamedTuple; kwargs...) where {O<:FixedObjective,D,M<:LatentEncoderDecoder,P,S,R}
    k = m.num_latent - 1
    p_node = _add_I(p.weights.node)
    𝜙, st = m.encoder(get_x(population), p.weights.encoder, m.st.encoder)
    μ = 𝜙[1:k, :]
    σ = softplus.(𝜙[k+1:end, :])
    ϵ = randn(eltype(𝜙), k, size(𝜙, 2))
    z0 = vcat(zeros(eltype(𝜙), 1, size(𝜙, 2)), μ + ϵ .* σ)
    zₜ = forward_ode.((m,), population, eachcol(z0), (p_node,); kwargs...)
    ŷ = first.(m.decoder.(zₜ, (p.weights.decoder,), (m.st.decoder,)))
    return getindex.(ŷ, 1, :), st # decoder outputs matrices, so need to convert to vectors
end

# TODO: Make version using VariationalELBO
# function solve_and_decode(model, population, z0, p_node, p)
#     # ...
# end

function forward_ode(m::HybridDCM{O,D,M,P,S,R}, individual::AbstractIndividual, z0::AbstractVector, p_node::ComponentArray; kwargs...) where {O,D,M<:LatentEncoderDecoder,P,S,R} 
    @ignore_derivatives p_node.I = zero(p_node.I)
    prob = remake(m.problem, u0 = Vector(z0))
    return forward_ode(prob, individual, p_node, Val(true); dv_idx=m.dv_compartment, kwargs...)
end

forward_adjoint(m::HybridDCM{O,D,M,P,S,R}, args...) where {O,D,M<:LatentEncoderDecoder,P,S,R} = forward(m, args...; sensealg = InterpolatingAdjoint(; autojacvec = ReverseDiffVJP()))

Base.show(io::IO, dcm::HybridDCM{O,D,M,P,S,R}) where {O,D,M<:LatentEncoderDecoder,P,S,R} = print(io, "LatentEncoderDecoder{num_latent = $(m.model.num_latent), $(dcm.objective)}")

