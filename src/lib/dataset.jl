import Zygote: ignore
import Random: shuffle!
import CSV

using DataFrames

"""
Convenience function to load the dataset and generate the corresponding 
population for the data. This function in this current form does not correctly 
handle every NONMEM-compatible dataset. It is thus advised to extend this 
function for your personal use case.
"""
function load(file::String, covariates::Vector{Symbol}; DVID=nothing, has_OCC=false, S1::Float64=1.)
    df = DataFrame(CSV.File(file))
    if !("RATE" in names(df)) || !("DURATION" in names(df))
        throw(ErrorException("Dataset does not contain RATE and/or DURATION columns. These columns should be defined for ALL doses. Bolus doses should be converted into infusions by for example using a short duration (i.e. one minute or 1/60, with a corresponding RATE of DOSE * 60)."))
    end

    # Gets the baseline for the current ID if it exists.
    get_baseline(group) = "BASE" in names(df) ? first(group.BASE) : 0.
    # Gets the rows based on MDV and DVID (if it exists).
    get_rows(group; MDV::Integer) = DVID !== nothing ? (group.MDV .== MDV) .& (group.DVID .== DVID) : group.MDV .== MDV
    
    # The coalesce function replaces all missing values by 999. It is however 
    # advised to explicitly handle missing values in the data set before using 
    # the load function.
    df_group = groupby(coalesce.(df, 999), has_OCC ? (:ID, :OCC) : :ID)
    x = Matrix(DataFrame([group[1, covariates] for group in df_group]))
    # Note that we subtract the baseline from the DV column:
    t_y = DataFrame([group[get_rows(group; MDV=0), [:TIME, :DV]] .- [0. get_baseline(group)] for group in df_group])
    y = t_y.DV
    t = t_y.TIME
    I = [Matrix(group[get_rows(group; MDV=1), [:TIME, :DOSE, :RATE, :DURATION]]) for group in df_group]
    callbacks = _generate_dosing_callbacks(I; S1=S1)

    # Technically this might not be the cleanest approach; since we are 
    # normalizing over the entire dataset instead of just the train set, there 
    # is some information leakage to the test set. 
    x̃, scale_x = normalize(x)

    return Population(x̃, y, t, callbacks, scale_x)
end

"""
Sometimes, covariates cannot be defined using Symbols. For example, when using 
spaces or some special characters, using the :Symbol synthax might not work. 
This functions allows for the use of strings, and just converts them to Symbols.
"""
load(file, covariates::Vector{String}) = load(file, Symbol.(covariates))


"""
Matrix operated version for getting the dosing rate over time following a matrix 
Iᵢ. The function constructs a k x d event matrix for the k timepoints and d 
doses/events in Iᵢ. Each dose is an event and affects two timepoints (i.e. the 
infusion start and end). We thus collect for each timepoint k the rate change 
resulting from the events. We can sum over all events to get the net effect at 
each timepoint and take the cumulative sum of that vector to get the rate over 
time.
"""
function _get_rate_over_t(Iᵢ::Matrix{Float64})::Matrix{Float64}
    times, _, rates, durations = eachcol(Iᵢ)
    timepoints = unique(vcat(times, times + durations))
    sort!(timepoints)
    k = length(timepoints)
    events = length(rates)
    event_matrix = zeros(k, events)
    for i in 1:events
        infusion_start = findfirst(t -> t == times[i], timepoints)
        infusion_end = findfirst(t -> t == times[i] + durations[i], timepoints)
        
        if infusion_start === nothing || infusion_end === nothing
            throw(Exception("Cannot find infusion start or end in the dosing timepoints for event $i"))
        end
        
        event_matrix[infusion_start, i] = rates[i]
        event_matrix[infusion_end, i] = -rates[i]
    end
    rate_over_t = cumsum(sum(event_matrix, dims=2), dims=1)
    # set all rates close to zero actually to 0. (results from inexact Float operations).
    rate_over_t[isapprox.(rate_over_t, 0; atol=1e-5)] .= 0. 
    return hcat(timepoints, rate_over_t)
end


"""
Generates a discrete callback for the differential equation solver for each 
individuals Iᵢ. Works by collecting the rate over time and stopping if the 
current time is in dosing event timepoints. Then it sets the I parameter of the 
diffeq to the corresponding rate.
"""
function _generate_dosing_callbacks(I::Vector{Matrix{Float64}}; S1::Float64=1.)
    n = length(I)
    callbacks = Vector{DiscreteCallback}(undef, n)
    for i in 1:n
        times_rates = _get_rate_over_t(I[i]) .* [1. S1]
        times = times_rates[:, 1]
        rates = times_rates[:, 2]
        condition(u, t, p)::Bool = t ∈ times
        affect!(integrator) = integrator.p[end] = rates[findfirst(t -> t == integrator.t, times)] # Here we assume that only a single event happens at each t, which is reasonable.
        callbacks[i] = DiscreteCallback(condition, affect!; save_positions=(false, false)) # Setting save_positions breaks interpolation. We address this in the predict function when interpolating=true
    end
    
    return callbacks
end

normalize(df::DataFrame) = normalize(Matrix(df))
normalize(x, scale_x::Tuple) = normalize(x, scale_x[1], scale_x[2])
normalize(x, min, max) = (x .- min) ./ (max - min)

"""Convenience function which determines the min and max."""
function normalize(x)
    min = minimum(x, dims=1)
    max = maximum(x, dims=1)
    return normalize(x, min, max), (min, max)
end

"""Inverse of the normalization function"""
normalize⁻¹(x̃, min, max) = (max - min) .* x̃ .+ min
normalize⁻¹(x̃, scale_x::Tuple) = normalize⁻¹(x̃, scale_x[1], scale_x[2])
normalize⁻¹(pop::Population) = normalize⁻¹(pop.x, pop.scale_x)

"""Easier to write aliases"""
normalize_inv(x̃, scale_x::Tuple) = normalize⁻¹(x̃, scale_x)
normalize_inv(population::Population) = normalize⁻¹(population)


"""Creates a train test split from data set size."""
function create_split(n::Integer; ratio::Float64=0.7)
    if (ratio >= 1.) || ratio <= 0.
        throw(ErrorException("Ratio should be between 0 and 1. Setting it to 1 just returns the population"))
        return
    end

    idxs = collect(1:n)
    split_idx = Integer(round(n * ratio, RoundUp))
    shuffle!(idxs)
    train = idxs[1:split_idx]
    test = idxs[split_idx+1:end]
    @assert length(train) + length(test) == n

    return train, test
end

"""Creates a train test split from a population."""
function create_split(population::Population; ratio::Float64=0.7)
    n = length(population)
    train, test = create_split(n; ratio=ratio) 
    return population[train], population[test]
end

