function _get_rate_over_t(a::AbstractMatrix{T}; bolus_duration = 1/60) where {T<:Real}
    times, doses, rates, durations = eachcol(a)
    # Set any rates of 0 to have taken one minute
    durations[rates .== 0] .= T(bolus_duration)
    rates[rates .== 0] = doses[rates .== 0] .* T(1/bolus_duration)
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

_remove_missing_type(x::AbstractArray{T}) where {T} = 
    convert(AbstractArray{nonmissingtype(T)}, x)

"""
    generate_dosing_callback(a; S1)

Returns a DiscreteCallback implementing the dosing events in a matrix of interventions `a`.

# Arguments
- `a`: Matrix with rows containing events with time, dose, (rate, and duration) columns.
- `S1`: Scaling factor for the doses. Used to get the dose in the same unit as model parameters. Default = 1.
- `bolus_duration`: Sets the virtual infusion duration for bolus doses. Default = 1/60 (i.e. one minute when t is in hours).
"""
function generate_dosing_callback(A::AbstractMatrix, ::Type{T}=Float32; S1=1, kwargs...) where T
    if size(A, 2) == 2 # only times and amts -> assume bolus doses
        A = hcat(A, zero(A))
    end
    
    times_rates = T.(_get_rate_over_t(_remove_missing_type(A); kwargs...) .* [1 S1])
    times = times_rates[:, 1]
    rates = times_rates[:, 2]
    
    condition(u, t, p; times=times) = t ∈ times

    function affect!(integrator; rates=rates, times=times)
        # Here we assume that only a single event happens at each t, which is reasonable.
        rate_idx = findfirst(isequal(integrator.t), times)
        if !isnothing(rate_idx)
            rate = rates[rate_idx]
            _apply_intervention!(integrator.p, rate)
        end
        
        nothing
    end
    
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

_apply_intervention!(p::AbstractVector, rate) = p[end] .= rate
_apply_intervention!(p::AbstractMatrix, rate) = p[end, :] .= rate
_apply_intervention!(p::ComponentArray, rate) = p.I .= rate # TODO: this should become `A` in time.

@non_differentiable _apply_intervention!(::Any, ::Any)