basic_tgrad(u,p,t) = zero(u)

# Hybrid models, UniversalDiffEq setup
struct HybridDCM{O,D,M,P,S,R} <: AbstractDEModel{O,D,M,P,S}
    objective::O
    problem::D
    model::M
    p::P
    st::S
    dv_compartment # can be single DV but also multiple
    rng::R
end

function HybridDCM(dudt::Function, model::Lux.AbstractLuxContainerLayer, num_compartments::Int; rng = Random.default_rng(), kwargs...)
    ps, st = Lux.setup(rng, model)
    node = StatefulLuxLayer(model.node, st.node)
    
    dudt_ = dudt$(; model = node)
    ff = ODEFunction{false}(dudt_; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, zeros(Float32, num_compartments), (-0.1f0, 1.f0))

    return HybridDCM(prob, model, ps, st; rng, kwargs...)
end

function HybridDCM(prob::D, model::M, ps, st::S; objective::O=SSE(), dv_compartment=1, rng::R=Random.default_rng()) where {O,D,M,R,S}
    p_, _ = init_params(rng, objective, ps, st)
    weights = merge(p_.weights, (node = ComponentVector(weights = p_.weights.node),))
    p = merge(p_, (weights = weights,))
    return HybridDCM{O,D,M,typeof(p),S,R}(objective, prob, model, p, st, dv_compartment, rng)
end

function _add_z_and_I(xx::ComponentVector, z)
    prev_data = ComponentArrays.getdata(xx)
    prev_axis = first(ComponentArrays.getaxes(xx))
    new_data = vcat(prev_data, z, zero(eltype(xx)))
    k = length(xx)
    new_axis = merge(prev_axis, Axis(z = k+1:k+length(z), I = k+length(z)+1))
    return ComponentVector(new_data, new_axis)
end

function _add_I(xx::ComponentVector)
    prev_data = ComponentArrays.getdata(xx)
    prev_axis = first(ComponentArrays.getaxes(xx))
    new_data = vcat(prev_data, zero(eltype(xx)))
    new_axis = merge(prev_axis, Axis(I = length(xx)+1))
    return ComponentVector(new_data, new_axis)
end

Base.show(io::IO, dcm::HybridDCM{O,D,M,P,S,R}) where {O,D,M,P,S,R} = print(io, "HybridDCM{$M, $(dcm.objective)}")