using LinearAlgebra

N = 10
ω = 0.1
ωs = [0.1, 0.1]
Ω = [0.01 0; 0 0.01]

@testset "_to_init_omega" begin
    res = DeepCompartmentModels._to_init_omega(ω)
    @test res isa Symmetric{typeof(ω), Matrix{typeof(ω)}} && isapprox(first(sqrt.(diag(res))), ω)

    res = DeepCompartmentModels._to_init_omega(ωs)
    @test res isa Symmetric{eltype(ωs), Matrix{eltype(ωs)}} && isapprox(sqrt.(diag(res)), ωs)
    
    res = DeepCompartmentModels._to_init_omega(Ω)
    @test res isa Symmetric{eltype(Ω), Matrix{eltype(Ω)}} && isapprox(res, Ω)
end

@testset "init_omega" begin
    res = DeepCompartmentModels.init_omega(ω, MeanSqrt())
    @test res isa NamedTuple && keys(res) == (:σ, ) && isapprox(only(res.σ), DeepCompartmentModels.softplus_inv(ω))
    @test only(res.σ) isa Real

    res = DeepCompartmentModels.init_omega(ω, MeanVar())
    @test res isa NamedTuple && keys(res) == (:σ², ) && isapprox(sqrt(only(res.σ²)), ω)
    @test only(res.σ²) isa Real

    res = DeepCompartmentModels.init_omega(ωs, MeanSqrt())
    @test res isa NamedTuple && keys(res) == (:L, )
    @test only(res) isa LowerTriangular{eltype(ωs), <:AbstractMatrix{eltype(ωs)}}
    # NOTE: res can be a LowerTriangular{Float, Adjoint{...}} because of _chol_lower
    
    res = DeepCompartmentModels.init_omega(ωs, MeanVar())
    @test res isa NamedTuple && keys(res) == (:Σ, )
    @test only(res) isa Symmetric{eltype(ωs), <:Matrix{eltype(ωs)}}

    res = DeepCompartmentModels.init_omega(Ω, MeanSqrt())
    @test res isa NamedTuple && keys(res) == (:L, )
    @test only(res) isa LowerTriangular{eltype(Ω), <:AbstractMatrix{eltype(Ω)}}
    
    res = DeepCompartmentModels.init_omega(Ω, MeanVar())
    @test res isa NamedTuple && keys(res) == (:Σ, )
    @test only(res) isa Symmetric{eltype(Ω), <:Matrix{eltype(Ω)}}
end

@testset "_init_phi_ps" begin
    # when called with omega and n
    # Univariate omega
    for type in [MeanVar(), MeanSqrt()]
        Ω_ = DeepCompartmentModels.init_omega(ω, type)
        res = DeepCompartmentModels._init_phi_ps(Ω_, N)
        @test res isa NamedTuple{<:Any, <:Tuple{<:AbstractVector, <:AbstractVector}} && keys(res) == (:μ, keys(Ω_)...)
    end
    # Multivariate omega
    for omega_in in [ωs, Ω]
        for type in [MeanVar(), MeanSqrt()]
            Ω_ = DeepCompartmentModels.init_omega(omega_in, type)
            res = DeepCompartmentModels._init_phi_ps(Ω_, N)
            @test res isa NamedTuple && keys(res) == (:μ, keys(Ω_)...)
            @test res.μ isa AbstractVector{<:AbstractVector{<:Real}}
            @test res[keys(Ω_)[1]] isa AbstractVector{<:AbstractMatrix} && !(res[keys(Ω_)[1]][1] === res[keys(Ω_)[1]][2])
        end
    end

    # MeanField and FullRank
    for omega_in in [ωs, Ω]
        for type in [MeanVar(), MeanSqrt()]
            Ω_ = DeepCompartmentModels.init_omega(omega_in, type)
            res_fr = DeepCompartmentModels._init_phi_ps(FullRank, Ω_, N)
            res_mf = DeepCompartmentModels._init_phi_ps(MeanField, Ω_, N)
            @test keys(res_fr) == (:μ, only(keys(Ω_))) 
            @test keys(res_mf) !== (:μ, only(keys(Ω_))) && keys(res_mf) == (:μ, type isa MeanVar ? (:σ²) : (:σ))
            @test isapprox(first(res_mf[2]), type isa MeanVar ? ωs.^2 : ωs)
        end
    end
end

@testset "init_phi" begin
    num_typ = 4
    Ω_ = DeepCompartmentModels.init_omega(Ω, MeanVar())
    Ω_uni = DeepCompartmentModels.init_omega(ω, MeanVar())

    for obj in [FO([1, 2]), FO([1]), FOCE([1]), FOCE([1, 2])]
        Ω__ = length(obj.idxs) == 1 ? Ω_uni : Ω_
        ps_res, st_res = init_phi(obj, Ω__, N, num_typ)
        @test ps_res isa NamedTuple && isempty(ps_res)
        @test st_res isa NamedTuple && keys(st_res) == (:mask, ) && size(st_res.mask) == (num_typ, length(obj.idxs))
    end

    # Variational Inference
    for obj in [VariationalELBO([1], FullRank()), VariationalELBO([1, 2], FullRank()), VariationalELBO([1], MeanField()), VariationalELBO([1, 2], MeanField())]
        Ω__ = length(obj.idxs) == 1 ? Ω_uni : Ω_
        _, res = init_phi(obj, Ω__, N, num_typ)
        @test keys(res) == (:epsilon, :mask)
        if length(obj.idxs) == 1
            @test size(res.epsilon) == (N, 1)
        else
            @test size(res.epsilon) == (length(obj.idxs), N, 1)
        end
    end
end