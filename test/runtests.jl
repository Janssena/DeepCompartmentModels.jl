print("\tLoading local DeepCompartmentModels package...")
include("../src/DeepCompartmentModels.jl");
println(" Done!")

using .DeepCompartmentModels
using Test

@info "Starting tests..."

begin
    @testset "Populations" begin
        include("population.test.jl")
    end
    
    @testset "Objectives" begin
        include("objectives.test.jl")
    end

    @testset "Initializers" begin
        include("initializers.test.jl")
    end

    @testset "Mixed effect estimation" begin
        include("mixed_effects.test.jl")
    end

    @testset "Model" begin
        include("model.test.jl")
    end

    @testset "DCM" begin
        include("dcm.test.jl")
    end
end