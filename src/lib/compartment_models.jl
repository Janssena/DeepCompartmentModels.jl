# Please feel free to contribute other compartment models to the current list.

"""
    unpack(p::AbstractMatrix, t::Real)

Enables the use of parameters that change over time. Collects the column of the 
parameter matrix that matches the current intergrator time point.
"""
function unpack(p::AbstractMatrix, t::Real)
    view = @view p[1, :]
    index = findlast(<(t), view)
    return @view p[2:end, index === nothing ? 1 : index] # Should be a view
    # Above saves one allocation versus below 🌠
    # index = findlast(<(t), p[1, :])
    # return p[2:end, index === nothing ? 1 : index] # Should be a view
end

"""
    unpack(p::AbstractVector, t::Real)

For Vector-based `p` this function just returns `p`.
"""
unpack(p::AbstractVector, ::Real) = p


"""
    one_comp!(dA, A, p, t)

In-place function for the one-compartment model:
`dA/dt = (I / Vd) - A₁ ⋅ k₁₀`

# Model parameters
- `CL`: Drug clearance estimate.
- `Vd`: Drug volume of distribution estimate.
- `I`: Rate of drug infusion; handled by callback function.
# Inferred parameters
- `k₁₀`: CL / Vd
"""
function one_comp!(dA, A, p, t)
    CL, Vd, I = unpack(p, t)
    k₁₀ = CL / Vd

    dA[1] = (I / Vd) - A[1] * k₁₀
end

"""
    one_comp_abs!(dA, A, p, t)

In-place function for the one-compartment model with an absorption compartment:
```
dA₁/dt = I - kₐ ⋅ A₁
dA₂/dt = kₐ ⋅ A₁ - A₂ ⋅ k₁₀
```

# Model parameters
- `kₐ`: Rate of drug absorption.
- `k₁₀`: Rate of drug elimination.
- `I`: Rate of drug infusion; handled by callback function.
"""
function one_comp_abs!(dA, A, p, t)
    kₐ, k₁₀, I = unpack(p, t)
    
    dA[1] = I - kₐ * A[1]
    dA[2] = kₐ * A[1] - A[2] * k₁₀
end

"""
    two_comp!(dA, A, p, t)

In-place function for the two-compartment model:
```
dA₁/dt = (I / V₁) + A₂ ⋅ k₂₁ - A₁ ⋅ (k₁₀ + k₁₂)
dA₂/dt = A₁ ⋅ k₁₂ - A₂ ⋅ k₂₁
```

# Model Parameters
- `CL`: Drug clearance from the first compartment.
- `V₁`: Central volume of distribution (in A₁).
- `Q`: Inter-compartmental clearance.
- `V₂`: Peripheral volume of distribution (in A₂).
- `I`: Rate of drug infusion; handled by callback function.
# Inferred parameters
- `k₁₀`: CL / V₁
- `k₁₂`: Q / V₁
- `k₂₁`: Q / V₂
"""
function two_comp!(dA, A, p, t)
    CL, V1, Q, V2, I = unpack(p, t)
    
    k₁₀ = CL / V1
    k₁₂ = Q / V1
    k₂₁ = Q / V2

    dA[1] = (I / V1) + A[2] * k₂₁ - A[1] * (k₁₀ + k₁₂)
    dA[2] = A[1] * k₁₂ - A[2] * k₂₁    
end

"""
    two_comp_abs!(dA, A, p, t)

In-place function for the two-compartment model with an absorption compartment:
```
dA₁/dt = I - kₐ ⋅ A₁
dA₂/dt = kₐ ⋅ A₁ + A₃ ⋅ k₃₂ - A₂ ⋅ (k₂₀ + k₂₃)
dA₃/dt = A₂ ⋅ k₂₃ - A₃ ⋅ k₃₂
```

# Model parameters
- `kₐ`: Rate of drug absorption.
- `k₂₀`: Rate of drug elimination.
- `k₂₃`: Rate of drug moving from compartment 2 to 3.
- `k₃₂`: Rate of drug moving from compartment 3 to 2.
- `I`: Rate of drug infusion; handled by callback function.
"""
function two_comp_abs!(dA, A, p, t)
    kₐ, k₂₀, k₂₃, k₃₂, I = unpack(p, t)

    dA[1] = I - kₐ * A[1]
    dA[2] = kₐ * A[1] + A[3] * k₃₂ - A[2] * (k₂₀ + k₂₃)
    dA[3] = A[2] * k₂₃ - A[3] * k₃₂
end
