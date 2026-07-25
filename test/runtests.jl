import Core.Compiler: return_type, isconcretetype

using Test
using DataFrames
using Serialization
@info "Loading local DeepCompartmentModels package..."
using DeepCompartmentModels
println("Done!")
include("lib/testutils.jl")

@info "Starting tests..."

begin
    @testset "Public API" begin
        include("lib/api.test.jl")
    end

    # TODO: Test generate_dosing_callback before individuals
    @testset "Populations and Individuals" begin
        include("lib/population.test.jl")
    end
    
    @testset "Objectives" begin
        include("lib/objectives.test.jl")
    end

    @testset "Error models" begin
        include("lib/error_models.test.jl")
    end

    @testset "Initializers" begin
        include("lib/initializers.test.jl")
    end

    @testset "Structural parameters" begin
        include("lib/structural.test.jl")
    end

    @testset "Mixed effect estimation" begin
        include("lib/mixed_effects.test.jl")
    end

    @testset "Multi-output models" begin
        include("lib/multi_output.test.jl")
    end

    @testset "Model" begin
        include("lib/model.test.jl")
    end

    @testset "DCM" begin
        include("lib/dcm.test.jl")
    end

    @testset "Solving and dosing callbacks" begin
        include("lib/solve.test.jl")
    end

    @testset "Fit" begin
        include("lib/fit.test.jl")
    end

    @testset "Uncertainty" begin
        include("lib/uncertainty.test.jl")
    end

    @testset "Diagnostics" begin
        include("lib/diagnostics.test.jl")
    end
end
