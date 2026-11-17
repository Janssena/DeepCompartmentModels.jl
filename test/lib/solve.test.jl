using Test
using DeepCompartmentModels

# One-compartment model with two DE parameters (CL, V) and two dosing events.
solve_nn = Lux.Chain(Lux.Dense(1, 8, tanh), Lux.Dense(8, 2, Lux.softplus))
solve_dcm = DeepCompartmentModel(DeepCompartmentModels.one_comp!, solve_nn, AdditiveError([1f0]))
solve_problem = solve_dcm.problem

solve_t = Float32[1, 4, 8, 169, 172, 180]
solve_cb = generate_dosing_callback(Float32[0 100; 168 100], Float32)
solve_occasions = [0f0 => 12f0, 168f0 => 192f0]

solve_indv = Individual(1, Float32[1], solve_t, zeros(Float32, 6), solve_cb, Float32)
solve_indv_occ = Individual(1, Float32[1], solve_t, zeros(Float32, 6), solve_cb, Float32; occasions = solve_occasions)

solve_pop = Population([solve_indv, solve_indv])
solve_pop_occ = Population([solve_indv_occ, solve_indv_occ])

solve_z = Float32[1, 2]
solve_Z = Float32[1 1.5; 2 2.5]
solve_t_dense = Float32[0.5, 1, 2, 3, 169, 170] # in both occasions

@testset "solve" begin
    @testset "individual" begin
        @test solve(solve_problem, solve_indv, solve_z) isa SciMLBase.AbstractODESolution
        @test solve(solve_dcm, solve_indv, solve_z) isa SciMLBase.AbstractODESolution
    end

    @testset "individual with occasions" begin
        sols = solve(solve_problem, solve_indv_occ, solve_z)
        @test sols isa Vector{<:SciMLBase.AbstractODESolution}
        @test length(sols) == length(solve_occasions) # one solution per occasion
    end

    @testset "population" begin
        sols = solve(solve_dcm, solve_pop, solve_Z)
        @test sols isa Vector{<:SciMLBase.AbstractODESolution}
        @test length(sols) == 2

        sols_occ = solve(solve_dcm, solve_pop_occ, solve_Z)
        @test sols_occ isa Vector{<:Vector{<:SciMLBase.AbstractODESolution}}
        @test length.(sols_occ) == [2, 2]
    end

    @testset "keyword arguments" begin
        # Interpolating
        @test solve(solve_problem, solve_indv, solve_z; interpolate = true) isa SciMLBase.AbstractODESolution
        @test solve(solve_problem, solve_indv_occ, solve_z; interpolate = true) isa Vector
        @test solve(solve_dcm, solve_pop, solve_Z; interpolate = true) isa Vector
        # Changing saveat
        @test solve(solve_problem, solve_indv, solve_z; saveat = solve_t_dense) isa SciMLBase.AbstractODESolution
        @test solve(solve_problem, solve_indv_occ, solve_z; saveat = solve_t) isa Vector
        # Explicit solver and solver options
        @test solve(solve_problem, solve_indv, solve_z; solver = Tsit5(), abstol = 1f-6) isa SciMLBase.AbstractODESolution
    end
end

@testset "solve_for_target" begin
    solve_for_target = DeepCompartmentModels.solve_for_target

    @testset "individual" begin
        ŷ = solve_for_target(solve_dcm, solve_indv, solve_z)
        @test ŷ isa Vector{Float32}
        @test length(ŷ) == length(solve_t)
    end

    @testset "individual with occasions" begin
        ŷ = solve_for_target(solve_dcm, solve_indv_occ, solve_z)
        @test ŷ isa Vector{Vector{Float32}}
        @test length.(ŷ) == [3, 3] # observations within each occasion
    end

    @testset "population" begin
        ŷ = solve_for_target(solve_dcm, solve_pop, solve_Z)
        @test ŷ isa Vector{Vector{Float32}}
        @test length(ŷ) == 2

        ŷ_occ = solve_for_target(solve_dcm, solve_pop_occ, solve_Z)
        @test ŷ_occ isa Vector{Vector{Vector{Float32}}}
        @test length(ŷ_occ) == 2
    end

    @testset "custom saveat" begin
        ŷ = solve_for_target(solve_dcm, solve_indv, solve_z; saveat = solve_t_dense)
        @test length(ŷ) == length(solve_t_dense) # every output at every time
        @test solve_for_target(solve_dcm, solve_indv_occ, solve_z; saveat = solve_t_dense) isa Vector
        @test solve_for_target(solve_dcm, solve_pop, solve_Z; saveat = solve_t_dense) isa Vector
    end
end

# Type stability. The dosing callback of an individual is erased into a `CallbackSet{Vector{Any},Vector{Any}}`
# by DiffEqBase >= 7 (Julia >= 1.12) when the ODEFunction is `AutoSpecialize` (the default of `ODEProblem`), which makes 
# `solve` return `Any`. `_remake_prob` therefore switches ODE problems to `FullSpecialize`.
@testset "type stability" begin
    solve_for_target = DeepCompartmentModels.solve_for_target

    @testset "solve" begin
        @test (@inferred solve(solve_problem, solve_indv, solve_z)) isa SciMLBase.AbstractODESolution
        @test (@inferred solve(solve_dcm, solve_indv, solve_z)) isa SciMLBase.AbstractODESolution
        @test (@inferred solve(solve_problem, solve_indv_occ, solve_z)) isa Vector{<:SciMLBase.AbstractODESolution}
        @test (@inferred solve(solve_dcm, solve_pop, solve_Z)) isa Vector{<:SciMLBase.AbstractODESolution}
        @test (@inferred solve(solve_dcm, solve_pop_occ, solve_Z)) isa Vector{<:Vector{<:SciMLBase.AbstractODESolution}}
    end

    @testset "solve_for_target" begin
        @test (@inferred solve_for_target(solve_dcm, solve_indv, solve_z)) isa Vector{Float32}
        @test (@inferred solve_for_target(solve_dcm, solve_indv_occ, solve_z)) isa Vector{Vector{Float32}}
        @test (@inferred solve_for_target(solve_dcm, solve_pop, solve_Z)) isa Vector{Vector{Float32}}
        @test (@inferred solve_for_target(solve_dcm, solve_pop_occ, solve_Z)) isa Vector{Vector{Vector{Float32}}}
    end

    @testset "keyword arguments" begin
        @test (@inferred solve(solve_problem, solve_indv, solve_z; interpolate = true)) isa SciMLBase.AbstractODESolution
        @test (@inferred solve(solve_problem, solve_indv_occ, solve_z; interpolate = true)) isa Vector
        @test (@inferred solve(solve_problem, solve_indv, solve_z; saveat = solve_t_dense)) isa SciMLBase.AbstractODESolution
        @test (@inferred solve(solve_problem, solve_indv, solve_z; solver = Tsit5(), abstol = 1f-6)) isa SciMLBase.AbstractODESolution
        @test (@inferred solve_for_target(solve_dcm, solve_indv, solve_z; saveat = solve_t_dense)) isa Vector{Float32}
        @test (@inferred solve_for_target(solve_dcm, solve_pop, solve_Z; saveat = solve_t_dense)) isa Vector{Vector{Float32}}
    end
end

@testset "FullSpecialize" begin
    remake_prob = DeepCompartmentModels._remake_prob
    auto_problem = ODEProblem(DeepCompartmentModels.one_comp!, zeros(1), (-0.1, 1.))
    @test SciMLBase.specialization(auto_problem.f) === SciMLBase.AutoSpecialize

    # The problem of the user is switched to FullSpecialize, without being changed itself
    prob = remake_prob(auto_problem, solve_indv, solve_t, solve_z)
    @test SciMLBase.specialization(prob.f) === SciMLBase.FullSpecialize
    @test SciMLBase.specialization(auto_problem.f) === SciMLBase.AutoSpecialize
    @test SciMLBase.isinplace(prob) == SciMLBase.isinplace(auto_problem)

    # u0, tspan and p are still set from the individual
    @test eltype(prob.u0) == Float32
    @test prob.tspan == (-0.1f0, maximum(solve_t))
    @test prob.p == [solve_z; 0f0]
    @test remake_prob(auto_problem, solve_indv, solve_t, solve_z; t0 = 5f0).tspan[1] == 5f0

    # The other fields of the ODEFunction are kept
    mass_matrix = [1.0 0.0; 0.0 0.0]
    dae_like = ODEProblem(ODEFunction(DeepCompartmentModels.one_comp!; mass_matrix), zeros(2), (-0.1, 1.))
    @test remake_prob(dae_like, solve_indv, solve_t, solve_z).f.mass_matrix == mass_matrix

    # A problem that is already FullSpecialize is used as is
    full_problem = ODEProblem{true,SciMLBase.FullSpecialize}(DeepCompartmentModels.one_comp!, zeros(1), (-0.1, 1.))
    @test remake_prob(full_problem, solve_indv, solve_t, solve_z).f === full_problem.f
end
