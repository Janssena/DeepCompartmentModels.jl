import Core.Compiler: return_type
import Zygote

# setup
z = [0.15, 3., 0.10, 1.0, 0.]
problem = ODEProblem(two_comp!, zeros(2), (-0.1, 48.))
callback = generate_dosing_callback([0. 1000. 60_000. 1/60]; S1=1/10)
individual = Individual(Float64[], [4., 24., 45.], [0.5, 0.2, 0.03], callback; id = "test")

@testset "forward_ode" begin
    arg_types = Tuple{typeof(problem), typeof(individual), typeof(z)}
    forward_ode_without_solver(args...) = forward_ode(args...; solver = nothing)
    forward_ode_interpolating(args...) = forward_ode(args...; interpolate = true)
    forward_ode_changing_saveat(args...) = forward_ode(args...; saveat = [1, 2, 3])
    forward_ode_sensealg(args...) = forward_ode(args...; sensealg = ForwardDiffSensitivity(; convert_tspan = true))

    # Returns a DESolution
    @test forward_ode(problem, individual, z) isa DESolution
    # If we do not pass a solver, the return type cannot be inferred.
    @test !isconcretetype(return_type(forward_ode_without_solver, arg_types))
    # The default solver (Tsit5) results in concrete return type
    @test isconcretetype(return_type(forward_ode, arg_types))
    # Setting interpolate = true does not break the type stability
    @test isconcretetype(return_type(forward_ode_interpolating, arg_types))
    # Changing saveat does not break type stability
    @test isconcretetype(return_type(forward_ode_changing_saveat, arg_types))
    # Setting the sensealg does not affect the 
    @test isconcretetype(return_type(forward_ode_sensealg, arg_types))
end

@testset "forward_ode_with_dv" begin
    model = DeepCompartmentModel(problem, Lux.Chain(), AdditiveError(), Float64)
    arg_types = Tuple{typeof(model), typeof(individual), typeof(z)}
    
    # Returns a Vector{<:Number}
    @test forward_ode_with_dv(model, individual, z) isa Vector{Float64}
    
    @test isconcretetype(return_type(forward_ode_with_dv, arg_types))
end

