function _get_rate_over_t(Iᵢ::AbstractMatrix{T}) where T
    times, doses, rates, durations = eachcol(Iᵢ)
    # Set any rates of 0 to have taken one minute
    durations[rates .== 0] .= T(1/60)
    rates[rates .== 0] = doses[rates .== 0] .* T(60)
    timepoints = unique(vcat(times, times + durations))
    sort!(timepoints)
    k = length(timepoints)
    events = length(rates)
    event_matrix = zeros(T, k, events)
    for i in 1:events
        infusion_start = findfirst(isequal(times[i]), timepoints)
        infusion_end = findfirst(isequal(times[i] + durations[i]), timepoints)
        
        if infusion_start === nothing || infusion_end === nothing
            throw(Exception("Cannot find infusion start or end in the dosing timepoints for event $i"))
        end
        
        event_matrix[infusion_start, i] = rates[i]
        event_matrix[infusion_end, i] = -rates[i]
    end
    rate_over_t = cumsum(sum(event_matrix, dims=2), dims=1)
    # set all rates close to zero actually to 0. (results from inexact Float operations).
    rate_over_t[isapprox.(rate_over_t, 0; atol=1e-5)] .= zero(T)
    return hcat(timepoints, rate_over_t)
end

"""
    generate_dosing_callback(I; S1)

Returns a DiscreteCallback implementing the dosing events in intervention matrix `I`.

# Arguments
- `I`: Matrix with rows containing events with time, dose, rate, and duration columns.
- `S1`: Scaling factor for the doses. Used to get the dose in the same unit as model parameters. Default = 1.
"""
function generate_dosing_callback(I::AbstractMatrix; S1=1)
    times_rates = _get_rate_over_t(Float32.(I)) .* Float32[1 S1]
    times = times_rates[:, 1]
    rates = times_rates[:, 2]
    
    function condition(u, t, p) 
        return t ∈ times
    end
    function affect!(integrator)
        # Here we assume that only a single event happens at each t, which is reasonable.
        if !(integrator.t ∈ times) return end
    
        rate = rates[findfirst(isequal(integrator.t), times)]
        @ignore_derivatives integrator.p[end, :] .= rate
    end
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end