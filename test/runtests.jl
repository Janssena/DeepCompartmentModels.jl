import Core.Compiler: return_type, isconcretetype

using Test
using DataFrames
@info "Loading local DeepCompartmentModels package..."
using DeepCompartmentModels
println("Done!")
include("lib/testutils.jl")

@info "Starting tests..."

begin
    # TODO: Test generate_dosing_callback before individuals
    @testset "Populations and Individuals" begin
        include("lib/population.test.jl")
    end
    
    @testset "Objectives" begin
        include("lib/objectives.test.jl")
    end

    @testset "Initializers" begin
        include("lib/initializers.test.jl")
    end

    @testset "Mixed effect estimation" begin
        include("lib/mixed_effects.test.jl")
    end

    @testset "Model" begin
        include("lib/model.test.jl")
    end

    @testset "DCM" begin
        include("lib/dcm.test.jl")
    end
end
