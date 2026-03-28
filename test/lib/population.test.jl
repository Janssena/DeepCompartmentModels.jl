# DEFAULTS
p₁ = 4 # covariate dim for zeta
p₂ = 2 # covariate dim for error model
k = 5 # number of time variable measurements
d = 8 # observation dim
m = 3 # ode dim

@testset "BasicIndividual" begin 
    # test types and dimensions
    for T in [Float16, Float32, Float64]
        indv_basic = BasicIndividual("basic", (zeta = rand(p₁), error = rand(p₂)), rand(d), rand(d), CallbackSet(), T; u0 = zeros(m))
        
        @test typeof(indv_basic).parameters[1] == T
        @test indv_basic.x isa @NamedTuple{zeta::Vector{T},error::Vector{T}}
        @test eltype(indv_basic.t) == eltype(indv_basic.y) == eltype(indv_basic.u0) == T
        
        @test size(indv_basic.x.zeta) == (p₁, )
        @test size(indv_basic.x.error) == (p₂, )
        @test size(indv_basic.t) == (d, )
        @test size(indv_basic.y) == (d, )
        @test size(indv_basic.u0) == (m, )
    end
    # test Individual constructor
    x = rand(p₁)
    t = rand(d)
    y = rand(d)
    u0 = rand(m)
    indv_basic1 = BasicIndividual("basic_1", x, t, y, CallbackSet(); u0)
    indv_basic2 = Individual("basic_2", x, t, y, CallbackSet(); u0)

    @test indv_basic2 isa BasicIndividual
    @test typeof(indv_basic1) == typeof(indv_basic2)
    @test isempty(indv_basic1.x.error) && isempty(indv_basic2.x.error) # not passing a x for error results in empty vector
    @test all([getfield(indv_basic1, field) == getfield(indv_basic2, field) for field in filter(Base.Fix2(!==, :id), fieldnames(typeof(indv_basic1)))])

    # test get functions
    @test get_x(indv_basic1) == indv_basic1.x.zeta # defaults to zeta
    @test isconcretetype(return_type(get_x, Tuple{typeof(indv_basic1)}))
    @test get_x(indv_basic1, :zeta) == indv_basic1.x.zeta
    @test get_x(indv_basic1, :error) == indv_basic1.x.error
    @test get_t(indv_basic1) == indv_basic1.t
    @test isconcretetype(return_type(get_t, Tuple{typeof(indv_basic1)}))
    @test get_y(indv_basic1) == indv_basic1.y
    @test isconcretetype(return_type(get_y, Tuple{typeof(indv_basic1)}))

    # Test transformation into TimeVariableIndividual
    indv_basic = BasicIndividual("basic", (zeta = rand(p₁), error = rand(p₂)), rand(d), rand(d), CallbackSet(); u0 = zeros(m))
    indv_time_var = DeepCompartmentModels._to_timevariable(indv_basic)
    T = typeof(indv_time_var).parameters[1]
    @test indv_time_var isa TimeVariableIndividual
    @test T == typeof(indv_basic).parameters[1]
    @test indv_time_var.x isa @NamedTuple{zeta::Matrix{T}, error::Matrix{T}}
    @test size(indv_time_var.x.zeta) == (p₁, 1)
    @test size(indv_time_var.x.error) == (p₂, 1)
    @test indv_time_var.t isa @NamedTuple{zeta::Matrix{T}, error::Matrix{T}, y::Vector{T}}
    @test size(indv_time_var.t.zeta) == (1, 1)
    @test size(indv_time_var.t.error) == (1, 1)
    @test size(indv_time_var.t.y) == (d, )
    @test size(indv_time_var.y) == (d, )
    @test size(indv_time_var.u0) == (m, )
    
    # Copying individual, without changing callback
    indv_basic_copy = copy(indv_basic)
    @test typeof(indv_basic) == typeof(indv_basic_copy)
    for field in filter(Base.Fix2(!∈, [:id, :callback]), fieldnames(typeof(indv_basic)))
        @test getfield(indv_basic_copy, field) == getfield(indv_basic, field) && !(getfield(indv_basic_copy, field) === getfield(indv_basic, field))
    end
    
    # When changing the callback:
    new_cb = DiscreteCallback(() -> nothing, () -> nothing)
    indv_basic_cb_copy = copy(indv_basic, new_cb)
    @test typeof(indv_basic) !== typeof(indv_basic_cb_copy)
    for field in filter(Base.Fix2(!∈, [:id, :callback]), fieldnames(typeof(indv_basic)))
        @test getfield(indv_basic_cb_copy, field) == getfield(indv_basic, field) && !(getfield(indv_basic_cb_copy, field) === getfield(indv_basic, field))
    end
    @test typeof(indv_basic_cb_copy).parameters[end] == typeof(new_cb)
    @test indv_basic_cb_copy.callback == new_cb
end

@testset "TimeVariableIndividual" begin 
    # test types and dimensions
    for T in [Float16, Float32, Float64]
        indv_time_var = TimeVariableIndividual("time_var", (zeta = rand(p₁, k), error = rand(p₂, k)), (zeta = rand(1, k), error = rand(1, k), y = rand(d)), rand(d), CallbackSet(), T; u0 = zeros(m))
        indv_time_var_vecs = TimeVariableIndividual("time_var_vecs", (zeta = rand(p₁, k), error = rand(p₂, k)), (zeta = rand(k), error = rand(k), y = rand(d)), rand(d), CallbackSet(), T; u0 = zeros(m))
        
        for indv in [indv_time_var, indv_time_var_vecs]
            @test typeof(indv).parameters[1] == T
            @test indv.x isa @NamedTuple{zeta::Matrix{T},error::Matrix{T}}
            @test indv.t isa @NamedTuple{zeta::Matrix{T},error::Matrix{T},y::Vector{T}}
            @test eltype(indv.t.zeta) == eltype(indv.t.error) == eltype(indv.t.y) == eltype(indv.y) == eltype(indv.u0) == T
            @test size(indv.x.zeta) == (p₁, k)
            @test size(indv.x.error) == (p₂, k)
            @test size(indv.t.zeta) == (1, k)
            @test size(indv.t.error) == (1, k)
            @test size(indv.t.y) == (d, )
            @test size(indv.y) == (d, )
            @test size(indv.u0) == (m, )
        end
    end
    # test Individual constructor without error covariates
    x = rand(p₁, k)
    t = (zeta = rand(k), y = rand(d))
    y = rand(d)
    u0 = rand(m)
    indv_time_var1 = TimeVariableIndividual("time_var_1", x, t, y, CallbackSet(); u0)
    indv_time_var2 = Individual("time_var_2", x, t, y, CallbackSet(); u0)

    @test indv_time_var2 isa TimeVariableIndividual
    @test typeof(indv_time_var1) == typeof(indv_time_var2)
    @test isempty(indv_time_var1.x.error) && isempty(indv_time_var2.x.error) # not passing a x for error results in empty vector
    @test all([getfield(indv_time_var1, field) == getfield(indv_time_var2, field) for field in filter(Base.Fix2(!==, :id), fieldnames(typeof(indv_time_var1)))])

    # test get functions
    @test get_x(indv_time_var1) == indv_time_var1.x.zeta # defaults to zeta
    @test isconcretetype(return_type(get_x, Tuple{typeof(indv_time_var1)}))
    @test get_x(indv_time_var1, :zeta) == indv_time_var1.x.zeta
    @test get_x(indv_time_var1, :error) == indv_time_var1.x.error
    @test get_t(indv_time_var1) == indv_time_var1.t.y
    @test isconcretetype(return_type(get_t, Tuple{typeof(indv_time_var1)}))
    @test get_y(indv_time_var1) == indv_time_var1.y
    @test isconcretetype(return_type(get_y, Tuple{typeof(indv_time_var1)}))

    # Test transformation into TimeVariableIndividual
    @test indv_time_var1 === DeepCompartmentModels._to_timevariable(indv_time_var1)
    
    # Copying individual, without changing callback
    indv_time_var_copy = copy(indv_time_var1)
    @test typeof(indv_time_var1) == typeof(indv_time_var_copy)
    for field in filter(Base.Fix2(!∈, [:id, :callback]), fieldnames(typeof(indv_time_var1)))
        @test getfield(indv_time_var_copy, field) == getfield(indv_time_var1, field) && !(getfield(indv_time_var_copy, field) === getfield(indv_time_var1, field))
    end
    
    # When changing the callback:
    new_cb = DiscreteCallback(() -> nothing, () -> nothing)
    indv_time_var_cb_copy = copy(indv_time_var1, new_cb)
    @test typeof(indv_time_var1) !== typeof(indv_time_var_cb_copy)
    for field in filter(Base.Fix2(!∈, [:id, :callback]), fieldnames(typeof(indv_time_var1)))
        @test getfield(indv_time_var_cb_copy, field) == getfield(indv_time_var1, field) && !(getfield(indv_time_var_cb_copy, field) === getfield(indv_time_var1, field))
    end
    @test typeof(indv_time_var_cb_copy).parameters[end] == typeof(new_cb)
    @test indv_time_var_cb_copy.callback == new_cb
end

@testset "_select_indv_type" begin 
    @test DeepCompartmentModels._select_indv_type(Vector(), Vector()) <: BasicIndividual
    @test DeepCompartmentModels._select_indv_type((zeta = nothing, error = nothing, y = nothing), Vector()) <: TimeVariableIndividual
    @test_throws ErrorException DeepCompartmentModels._select_indv_type(nothing, nothing)
end

@testset "_callback_type_matches" begin
    # DiscreteCallback
    𝐈 = Float64[0 1 1 1;]
    cb = generate_dosing_callback(𝐈)
    cb_64 = generate_dosing_callback(𝐈, Float64)
    @test (@test_logs (:warn,) DeepCompartmentModels._callback_type_matches(cb, Float64)) == false # calls @warn
    @test (@test_nowarn DeepCompartmentModels._callback_type_matches(cb, Float32)) == true # does not call @warn

    @test (@test_nowarn DeepCompartmentModels._callback_type_matches(cb_64, Float64)) == true # does not call @warn
    @test (@test_logs (:warn,) DeepCompartmentModels._callback_type_matches(cb_64, Float32)) == false # calls @warn

    # CallbackSet
    cb_set = CallbackSet(cb, cb_64)
    cb_set_64 = CallbackSet(cb_64, cb_64)
    @test (@test_logs (:warn,) DeepCompartmentModels._callback_type_matches(cb_set, Float64)) == false # calls @warn
    @test (@test_logs (:warn,) DeepCompartmentModels._callback_type_matches(cb_set, Float32)) == false # calls @warn
    
    @test (@test_nowarn DeepCompartmentModels._callback_type_matches(cb_set_64, Float64)) == true # does not call @warn
    @test (@test_logs (:warn,) DeepCompartmentModels._callback_type_matches(cb_set_64, Float32)) == false # calls @warn once
end

@testset "Population" begin
    # all the same type
    indv_basic = BasicIndividual("basic", rand(p₁), rand(d), rand(d), CallbackSet(); u0 = zeros(m))
    indv_basic_with_error = BasicIndividual("basic", (zeta = rand(p₁), error = rand(p₂)), rand(d), rand(d), CallbackSet(); u0 = zeros(m))
    indv_time_var = TimeVariableIndividual("time_var", rand(p₁, k), (zeta = rand(k), y = rand(d)), rand(d), CallbackSet(); u0 = zeros(m))
    indv_time_var_with_error = TimeVariableIndividual("time_var_with_error", (zeta = rand(p₁, k), error = rand(p₂, k)), (zeta = rand(k), error = rand(k), y = rand(d)), rand(d), CallbackSet(); u0 = zeros(m))

    indv_basic_alt_type = BasicIndividual(1241551, rand(p₁), rand(d), rand(d), CallbackSet(), Float64; u0 = zeros(m))
    indv_basic_alt_id = BasicIndividual(1241551, rand(p₁), rand(d), rand(d), CallbackSet(); u0 = zeros(m))
    indv_basic_alt_cb = copy(indv_basic, DiscreteCallback(() -> nothing, () -> nothing))
    
    population = Population([indv_basic, copy(indv_basic)])
    population_with_error = Population([indv_basic_with_error, copy(indv_basic_with_error)])
    @test eltype(population) == typeof(indv_basic)
    @test population.count == 2
    @test get_x(population) isa Matrix{eltype(population).parameters[1]}
    @test isempty(get_x(population, :error))
    @test isconcretetype(return_type(get_x, Tuple{typeof(population)}))
    @test isconcretetype(return_type(get_x, Tuple{typeof(population), Symbol}))
    @test !isempty(get_x(population_with_error, :error))
    @test get_x(population_with_error, :error) isa Matrix{eltype(population).parameters[1]}
    @test get_t(population) isa Vector{Vector{eltype(population).parameters[1]}}
    @test isconcretetype(return_type(get_t, Tuple{typeof(population)}))
    @test get_y(population) isa Vector{Vector{eltype(population).parameters[1]}}
    @test isconcretetype(return_type(get_y, Tuple{typeof(population)}))

    population_time_var = Population([indv_basic, indv_time_var])
    @test_logs (:info,) Population([indv_basic, indv_time_var]);
    @test eltype(population_time_var) == typeof(indv_time_var)
    @test population_time_var.count == 2
    @test get_x(population_time_var) isa Vector{Matrix{eltype(population_time_var).parameters[1]}}
    @test all(isempty.(get_x(population_time_var, :error)))
    @test isconcretetype(return_type(get_x, Tuple{typeof(population_time_var)}))
    @test isconcretetype(return_type(get_x, Tuple{typeof(population_time_var), Symbol}))
    @test get_t(population_time_var) isa Vector{Vector{eltype(population_time_var).parameters[1]}}
    @test isconcretetype(return_type(get_t, Tuple{typeof(population_time_var)}))
    @test get_y(population_time_var) isa Vector{Vector{eltype(population_time_var).parameters[1]}}
    @test isconcretetype(return_type(get_y, Tuple{typeof(population_time_var)}))

    @test_throws ErrorException Population([indv_basic, indv_basic_alt_type])
    @test_throws ErrorException Population([indv_basic, indv_basic_alt_id])
    @test_throws ErrorException Population([indv_basic, indv_basic_alt_cb]) 
end