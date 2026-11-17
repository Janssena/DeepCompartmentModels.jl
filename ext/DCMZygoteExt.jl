module DCMZygoteExt

# Zygote as an opt-in automatic differentiation backend: `ad = AutoZygote()`.
#
# Enzyme (the default) cannot differentiate the adjoint sensitivity algorithms of in-place
# `UniversalDiffEq`s, Zygote can. The package is written to be differentiated by Enzyme, so the
# few places where Zygote needs help are declared here.

import DeepCompartmentModels
import DeepCompartmentModels: AbstractIndividual, Population, AutoZygote
import Zygote
import Zygote.ChainRules: @non_differentiable, rrule, unthunk, NoTangent, ZeroTangent, AbstractZero

function DeepCompartmentModels._ad_value_and_gradient(::AutoZygote, f, ps, args...)
    loss, grads = Zygote.withgradient(p -> f(p, args...), ps)
    return loss, only(grads)
end

# Data accessors and callback utilities are constants.
@non_differentiable DeepCompartmentModels.get_x(::AbstractIndividual, ::Symbol)
@non_differentiable DeepCompartmentModels.get_y(::AbstractIndividual)
@non_differentiable DeepCompartmentModels.get_x(::Population, ::Symbol)
@non_differentiable DeepCompartmentModels.get_y(::Population)
@non_differentiable DeepCompartmentModels._solve_callback(::Any, ::Any)
@non_differentiable DeepCompartmentModels._apply_intervention!(::Any, ::Any)

# `construct_p` copies the parameters of a UDE and sets the infusion rate `I` to zero, which is a
# mutation that Zygote cannot differentiate. The infusion rate is set by the callback and is not a
# parameter: its gradient is zero.
function rrule(::typeof(DeepCompartmentModels.construct_p), z::DeepCompartmentModels.ComponentArray{T}, individual::AbstractIndividual) where T
    p = DeepCompartmentModels.construct_p(z, individual)
    function construct_p_pullback(Δ)
        Δ = unthunk(Δ)
        Δ isa AbstractZero && return NoTangent(), Δ, NoTangent()
        Δz = DeepCompartmentModels.ComponentArray(collect(Δ), DeepCompartmentModels.getaxes(z))
        Δz.I = zero(T)
        return NoTangent(), Δz, NoTangent()
    end
    return p, construct_p_pullback
end

end
