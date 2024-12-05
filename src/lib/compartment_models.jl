"""
    unpack(p::AbstractVector, t)

For Vector-based `p` this function just returns `p`.
"""
unpack(p::AbstractVector, _) = p

"""
    unpack(p::AbstractMatrix, t)

Enables the use of parameters that change over time. Collects the column of the 
parameter matrix that matches the current intergrator time point.
"""
function unpack(p::AbstractMatrix, t) # TODO: can we set the type of t? Can become Dual
    view = @view p[1, :]
    return _take_col(p, findlast(<(t), view))
    # Above saves one allocation versus below 🌠
    # index = findlast(<(t), p[1, :])
    # return p[2:end, index === nothing ? 1 : index] # Should be a view
end

_take_col(p, ::Nothing) = @view p[2:end, 1]
_take_col(p, index::Int) = @view p[2:end, index]

########## COMPARTMENT MODELS
# Please feel free to contribute other compartment models to the current list.

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
- `CL`: Drug clearance estimate.
- `Vd`: Drug volume of distribution estimate.
- `I`: Rate of drug infusion; handled by callback function.
"""
function one_comp_abs!(dA, A, p, t)
    kₐ, CL, Vd, I = unpack(p, t)
    k₁₀ = CL / Vd
    
    dA[1] = I - kₐ * A[1]
    dA[2] = (kₐ * A[1] / Vd) - A[2] * k₁₀
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
    CL, V₁, Q, V₂, I = unpack(p, t)
    
    k₁₀ = CL / V₁
    k₁₂ = Q / V₁
    k₂₁ = Q / V₂

    dA[1] = (I / V₁) + A[2] * k₂₁ - A[1] * (k₁₀ + k₁₂)
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
- `CL`: Drug clearance from the first compartment (in A₂).
- `V₁`: Central volume of distribution (in A₂).
- `Q`: Inter-compartmental clearance (in A₃).
- `V₂`: Peripheral volume of distribution (in A₃).
- `I`: Rate of drug infusion; handled by callback function.
"""
function two_comp_abs!(dA, A, p, t)
    kₐ, CL, V₁, Q, V₂, I = unpack(p, t)
    k₁₀ = CL / V₁
    k₁₂ = Q / V₁
    k₂₁ = Q / V₂

    dA[1] = I - kₐ * A[1]
    dA[2] = (kₐ * A[1] / V₁) + A[3] * k₂₁ - A[2] * (k₁₀ + k₁₂)
    dA[3] = A[2] * k₁₂ - A[3] * k₂₁
end