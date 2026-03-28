import Core.Compiler: return_type, isconcretetype

using Test
using DataFrames
@info "Loading local DeepCompartmentModels package..."
using DeepCompartmentModels
println("Done!")

@info "Starting tests..."

begin
    # TODO: Test generate_dosing_callback before individuals
    @testset "Populations and Individuals" begin
        include("lib/population.test.jl")
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