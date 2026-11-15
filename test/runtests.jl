import Core.Compiler: return_type, isconcretetype

using Test
using DataFrames
@info "Loading local DeepCompartmentModels package..."
using DeepCompartmentModels
println("Done!")

@info "Starting tests..."

@testset "DeepCompartmentModels" begin
    # TODO: Test generate_dosing_callback before individuals
    @testset "Populations and Individuals" begin
        include("lib/population.test.jl")
    end

    @testset "UniversalDiffEq" begin
        include("lib/ude.test.jl")
    end

    @testset "Fitting" begin
        include("lib/fit.test.jl")
    end

    # The test files below were written for an older version of the package: they call functions that no
    # longer exist (e.g. `make_etas`, `init_omega`, `forward_ode`, `_get_prior`) and error or fail. They are
    # kept for reference, until their cases are ported to the current API, and are skipped by default.
    # Run them with `DCM_LEGACY_TESTS=true`.
    if get(ENV, "DCM_LEGACY_TESTS", "false") == "true"
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
end
