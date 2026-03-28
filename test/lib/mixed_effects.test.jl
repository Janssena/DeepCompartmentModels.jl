using LinearAlgebra

# setup
N = 10
M = 3
ps_fo = (phi = NamedTuple(), )
ps_vi = (phi = (μ = fill(zeros(2), N), L = fill(LowerTriangular([0.316 0; 0 0.316]), N), ), )
ps_vi_uni = (phi = (μ = zeros(10), σ = fill(0.316, N)), )

mask = Float64[1 0; 0 1; 0 0; 0 0]
mask_uni = Float64[1; 0; 0;;]

st_fo = (phi = (mask = mask, eta = randn(2, N)), )
st_fo_uni = (phi = (mask = mask_uni, eta = randn(N) ), )

st_vi = (phi = (epsilon = randn(2, N), mask = mask, ), )
st_vi_mc = (phi = (epsilon = randn(2, N, M), mask = mask, ), )
st_vi_uni = (phi = (epsilon = randn(N), mask = mask_uni, ), )
st_vi_uni_mc = (phi = (epsilon = randn(N, M), mask = mask_uni, ), )

@testset "make_etas" begin
    # First-order approximation based objectives
    # Multivariate eta in state
    @test size(make_etas(ps_fo.phi, st_fo.phi)) == (2, N)
    @test make_etas(ps_fo.phi, st_fo.phi) == st_fo.phi.eta
    # Univariate eta in state
    @test size(make_etas(ps_fo.phi, st_fo_uni.phi)) == (N, )
    @test make_etas(ps_fo.phi, st_fo_uni.phi) == st_fo_uni.phi.eta
    
    # Variational Inference
    @test size(make_etas(ps_vi.phi, st_vi.phi)) == (2, N)
    @test size(make_etas(ps_vi.phi, st_vi_mc.phi)) == (M, )
    @test make_etas(ps_vi.phi, st_vi_mc.phi) isa AbstractVector{<:AbstractMatrix{<:Real}}
    @test size(make_etas(ps_vi.phi, st_vi_mc.phi)[1]) == (2, N)
    
    
    @test size(make_etas(ps_vi_uni.phi, st_vi_uni.phi)) == (N, )
    @test size(make_etas(ps_vi_uni.phi, st_vi_uni_mc.phi)) == (M, )
    @test make_etas(ps_vi_uni.phi, st_vi_uni_mc.phi) isa AbstractVector{<:AbstractVector{<:Real}}
    @test size(make_etas(ps_vi_uni.phi, st_vi_uni_mc.phi)[1]) == (N, )
end

@testset "_to_random_eff_matrix" begin
    η_pop = make_etas(ps_fo.phi, st_fo.phi)
    η_pop_uni = make_etas(ps_fo.phi, st_fo_uni.phi)

    η_pop_mc = make_etas(ps_vi.phi, st_vi_mc.phi)
    η_pop_uni_mc = make_etas(ps_vi_uni.phi, st_vi_uni_mc.phi)

    @test DeepCompartmentModels._to_random_eff_matrix(st_fo.phi.mask, η_pop) isa AbstractMatrix
    @test size(DeepCompartmentModels._to_random_eff_matrix(st_fo.phi.mask, η_pop)) == (size(mask, 1), N)
    @test DeepCompartmentModels._to_random_eff_matrix(st_fo_uni.phi.mask, η_pop_uni) isa AbstractMatrix
    @test size(DeepCompartmentModels._to_random_eff_matrix(st_fo_uni.phi.mask, η_pop_uni)) == (size(mask_uni, 1), N)
    # Monte Carlo samples
    @test DeepCompartmentModels._to_random_eff_matrix(st_fo.phi.mask, η_pop_mc) isa AbstractVector{<:AbstractMatrix{<:Real}}
    @test size(DeepCompartmentModels._to_random_eff_matrix(st_fo.phi.mask, η_pop_mc)) == (M, )
    @test size(DeepCompartmentModels._to_random_eff_matrix(st_fo.phi.mask, η_pop_mc)[1]) == (size(mask, 1), N)
    @test DeepCompartmentModels._to_random_eff_matrix(st_fo_uni.phi.mask, η_pop_uni_mc) isa AbstractVector{<:AbstractMatrix{<:Real}}
    @test size(DeepCompartmentModels._to_random_eff_matrix(st_fo_uni.phi.mask, η_pop_uni_mc)) == (M, )
    @test size(DeepCompartmentModels._to_random_eff_matrix(st_fo_uni.phi.mask, η_pop_uni_mc)[1]) == (size(mask_uni, 1), N)
end

@testset "get_random_effects" begin
    # First-order approximation based objectives
    res, st_new = get_random_effects(ps_fo, st_fo)
    @test res isa AbstractMatrix && size(res) == (size(mask, 1), N)
    @test st_new isa NamedTuple && :eta in keys(st_new.phi) && st_new.phi.eta isa AbstractMatrix{<:Real}
    # Univariate
    res, st_new = get_random_effects(ps_fo, st_fo_uni)
    @test res isa AbstractMatrix && size(res) == (size(mask_uni, 1), N)
    @test st_new isa NamedTuple && :eta in keys(st_new.phi) && st_new.phi.eta isa AbstractVector{<:Real}

    # Variational Inference
    res, st_new = get_random_effects(ps_vi, st_vi)
    @test res isa AbstractMatrix && size(res) == (size(mask, 1), N)
    @test st_new isa NamedTuple && :eta in keys(st_new.phi) && st_new.phi.eta isa AbstractMatrix{<:Real}
    # Univariate
    res, st_new = get_random_effects(ps_vi_uni, st_vi_uni)
    @test res isa AbstractMatrix && size(res) == (size(mask_uni, 1), N)
    @test st_new isa NamedTuple && :eta in keys(st_new.phi) && st_new.phi.eta isa AbstractVector{<:Real}
    # with Monte Carlo samples
    res, st_new = get_random_effects(ps_vi, st_vi_mc)
    @test res isa AbstractVector{<:AbstractMatrix} && size(res) == (M, ) && size(res[1]) == (size(mask, 1), N)
    @test st_new isa NamedTuple && :eta in keys(st_new.phi) && st_new.phi.eta isa AbstractVector{<:AbstractMatrix{<:Real}}
    # Univariate
    res, st_new = get_random_effects(ps_vi_uni, st_vi_uni_mc)
    @test res isa AbstractVector{<:AbstractMatrix} && size(res) == (M, ) && size(res[1]) == (size(mask_uni, 1), N)
    @test st_new isa NamedTuple && :eta in keys(st_new.phi) && st_new.phi.eta isa AbstractVector{<:AbstractVector{<:Real}}
end

