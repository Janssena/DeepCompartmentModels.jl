# Please feel free to contribute other compartment models to the current list.

# This function prepares the parameter vector for the current time point.
# When the covariates change over time, the parameter vector contains PK parameters
# for different time points, and this function grabs the correct parameter vector
function unpack(p::AbstractMatrix, t::Real)
    view = @view p[1, :]
    index = findlast(<(t), view)
    return @view p[2:end, index === nothing ? 1 : index] # Should be a view
    # Above saves one allocation versus below 🌠
    # index = findlast(<(t), p[1, :])
    # return p[2:end, index === nothing ? 1 : index] # Should be a view
end

unpack(p::AbstractVector, ::Real) = p

function one_comp!(dA, A, p, t)
    CL, Vd, I = unpack(p, t)
    k₁₀ = CL / Vd

    dA[1] = (I / Vd) - A[1] * k₁₀
end

function one_comp_abs!(dA, A, p, t)
    kₐ, k₁₀, I = unpack(p, t)
    
    dA[1] = I - kₐ * A[1]
    dA[2] = kₐ * A[1] - A[2] * k₁₀
end


function two_comp!(dA, A, p, t)
    CL, V1, Q, V2, I = unpack(p, t)
    
    k₁₀ = CL / V1
    k₁₂ = Q / V1
    k₂₁ = Q / V2

    dA[1] = (I / V1) + A[2] * k₂₁ - A[1] * (k₁₀ + k₁₂)
    dA[2] = A[1] * k₁₂ - A[2] * k₂₁    
end

function two_comp_abs!(dA, A, p, t)
    kₐ, k₂₀, k₂₃, k₃₂, I = unpack(p, t)

    dA[1] = I - kₐ * A[1]
    dA[2] = kₐ * A[1] + A[3] * k₃₂ - A[2] * (k₂₀ + k₂₃)
    dA[3] = A[2] * k₂₃ - A[3] * k₃₂
end
