# Please feel free to contribute other compartment models to the current list.

function one_comp!(dA, A, p, t)
    CL, Vd, I = p
    k₁₀ = CL / Vd

    dA[1] = (I / Vd) - A[1] * k₁₀
end


function two_comp!(dA, A, p, t)
    CL, V1, Q, V2, I = p
    k₁₀ = CL / V1
    k₁₂ = Q / V1
    k₂₁ = Q / V2

    dA[1] = (I / V1) + A[2] * k₂₁ - A[1] * (k₁₀ + k₁₂)
    dA[2] = A[1] * k₁₂ - A[2] * k₂₁
end


function two_comp_abs!(dA, A, p, t)
    kₐ, k₂₀, k₂₃, k₃₂, I = p

    dA[1] = I - kₐ * A[1]
    dA[2] = kₐ * A[1] + A[3] * k₃₂ - A[2] * (k₂₀ + k₂₃)
    dA[3] = A[2] * k₂₃ - A[3] * k₃₂
end
