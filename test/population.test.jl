import DifferentialEquations: CallbackSet

include("../src/lib/population.jl");

individual = Individual(rand(10), rand(5), rand(5), CallbackSet(); id = "Standard")
individual2 = Individual((cov = rand(10), error = rand(2)), rand(5), rand(5), CallbackSet(); id = "NamedTuple X")
individual3 = Individual(rand(3, 10), (x = rand(3), y = rand(5)), rand(5), CallbackSet(); id = "Time-variable X")

@test is_timevariable(individual) == false
@test is_timevariable(individual2) == false
@test is_timevariable(individual3) == true

@test is_timevariable(make_timevariable(individual)) == true

N = 10
X_DIM = 6
TY_DIM = 4
indvs = AbstractIndividual[Individual(rand(X_DIM), rand(TY_DIM), rand(TY_DIM), CallbackSet(); id = i) for i in 1:N]
indvs_time_var = AbstractIndividual[Individual(rand(X_DIM, 2), (x = [0., 10.], y = rand(TY_DIM)), rand(TY_DIM), CallbackSet(); id = i) for i in 1:N]
population = Population(indvs)
population_time_var = Population(indvs_time_var)

@test length(population) == 10
@test is_timevariable(population) == false
@test typeof(population.x) <: AbstractMatrix
@test size(population.x) == (X_DIM, N)
@test typeof(population.t) <: AbstractVector{<:AbstractVector}
@test size(population.t) == (N,)
@test typeof(population.y) <: AbstractVector{<:AbstractVector}
@test size(population.y) == (N,)

@test length(population_time_var) == 10
@test is_timevariable(population_time_var) == true
@test_throws ErrorException population_time_var.x
@test_throws ErrorException population_time_var.t
@test typeof(population_time_var.y) <: AbstractVector{<:AbstractVector}
@test size(population_time_var.y) == (N,)
