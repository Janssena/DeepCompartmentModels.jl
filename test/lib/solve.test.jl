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

# `@inferred` fails the test if the return type of the call cannot be inferred.
@testset "solve" begin
    @testset "individual" begin
        @test (@inferred solve(solve_problem, solve_indv, solve_z)) isa SciMLBase.AbstractODESolution
        @test (@inferred solve(solve_dcm, solve_indv, solve_z)) isa SciMLBase.AbstractODESolution
    end

    @testset "individual with occasions" begin
        sols = @inferred solve(solve_problem, solve_indv_occ, solve_z)
        @test sols isa Vector{<:SciMLBase.AbstractODESolution}
        @test length(sols) == length(solve_occasions) # one solution per occasion
    end

    @testset "population" begin
        sols = @inferred solve(solve_dcm, solve_pop, solve_Z)
        @test sols isa Vector{<:SciMLBase.AbstractODESolution}
        @test length(sols) == 2

        sols_occ = @inferred solve(solve_dcm, solve_pop_occ, solve_Z)
        @test sols_occ isa Vector{<:Vector{<:SciMLBase.AbstractODESolution}}
        @test length.(sols_occ) == [2, 2]
    end

    @testset "keyword arguments keep the return type inferrable" begin
        # Interpolating
        @test (@inferred solve(solve_problem, solve_indv, solve_z; interpolate = true)) isa SciMLBase.AbstractODESolution
        @test (@inferred solve(solve_problem, solve_indv_occ, solve_z; interpolate = true)) isa Vector
        @test (@inferred solve(solve_dcm, solve_pop, solve_Z; interpolate = true)) isa Vector
        # Changing saveat
        @test (@inferred solve(solve_problem, solve_indv, solve_z; saveat = solve_t_dense)) isa SciMLBase.AbstractODESolution
        @test (@inferred solve(solve_problem, solve_indv_occ, solve_z; saveat = solve_t)) isa Vector
        # Explicit solver and solver options
        @test (@inferred solve(solve_problem, solve_indv, solve_z; solver = Tsit5(), abstol = 1f-6)) isa SciMLBase.AbstractODESolution
    end
end

@testset "solve_for_target" begin
    solve_for_target = DeepCompartmentModels.solve_for_target

    @testset "individual" begin
        ŷ = @inferred solve_for_target(solve_dcm, solve_indv, solve_z)
        @test ŷ isa Vector{Float32}
        @test length(ŷ) == length(solve_t)
    end

    @testset "individual with occasions" begin
        ŷ = @inferred solve_for_target(solve_dcm, solve_indv_occ, solve_z)
        @test ŷ isa Vector{Vector{Float32}}
        @test length.(ŷ) == [3, 3] # observations within each occasion
    end

    @testset "population" begin
        ŷ = @inferred solve_for_target(solve_dcm, solve_pop, solve_Z)
        @test ŷ isa Vector{Vector{Float32}}
        @test length(ŷ) == 2

        ŷ_occ = @inferred solve_for_target(solve_dcm, solve_pop_occ, solve_Z)
        @test ŷ_occ isa Vector{Vector{Vector{Float32}}}
        @test length(ŷ_occ) == 2
    end

    @testset "custom saveat" begin
        ŷ = @inferred solve_for_target(solve_dcm, solve_indv, solve_z; saveat = solve_t_dense)
        @test length(ŷ) == length(solve_t_dense) # every output at every time
        @test (@inferred solve_for_target(solve_dcm, solve_indv_occ, solve_z; saveat = solve_t_dense)) isa Vector
        @test (@inferred solve_for_target(solve_dcm, solve_pop, solve_Z; saveat = solve_t_dense)) isa Vector
    end
end
