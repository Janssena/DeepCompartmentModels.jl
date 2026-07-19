import Random

function toy_dcm(::Type{T} = Float32; error = nothing) where {T}
    network = Lux.Chain(
        Normalize(T[100.0]),
        Lux.Dense(1, 8, Lux.swish),
        Lux.Dense(8, 3, Lux.softplus),
        Lux.WrappedFunction(x -> T[1.5, 3.0, 30.0] .* x),
    )
    return error === nothing ? DCM(one_comp_abs!, network; target = 2) :
           DCM(one_comp_abs!, network, error; target = 2)
end

function toy_population(::Type{T} = Float32; n = 6, seed = 1) where {T}
    rng = Random.MersenneTwister(seed)
    individuals = map(1:n) do i
        callback = generate_dosing_callback(T[0.0 100.0 1000.0 0.1], T)
        times = T[0.5, 1.0, 2.0, 4.0, 8.0]
        observations = T.(max.(2 .* exp.(-0.25 .* times) .+
                              0.05 .* randn(rng, length(times)), 0.01))
        covariates = T[70.0 + 5 * randn(rng)]
        Individual("id$i", covariates, times, observations, callback, T)
    end
    return Population(individuals)
end
