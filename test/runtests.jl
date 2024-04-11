using Test

@time begin
    @time @testset "Population" begin
        include("population.test.jl")
    end
end