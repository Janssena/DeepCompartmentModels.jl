module DCMDataFramesExt

import DeepCompartmentModels: DeepCompartmentModels, Population, AbstractIndividual, Individual, generate_dosing_callback
import DataFrames: DataFrame, groupby

"""
    load(df::DataFrame, covariates; S1 = 1)

Function to load a population from a DataFrame object. Automatically detects 
column names and loads the data accordingly. Reports on any ambiguities with 
respect to the column names.

# Arguments
- `df::DataFrame`: DataFrame containing the data.
- `covariates`: A vector of strings or symbols detailing the covariates to take.
"""
function DeepCompartmentModels.load(df::DataFrame, covariates; S1=1)
    colnames = find_colnames(df)
    df, colnames = handle_missing_rate_duration(df::DataFrame, colnames)
    df_group = groupby(df, colnames[:id])

    indvs = Vector{AbstractIndividual}(undef, length(df_group))
    for (i, group) in enumerate(df_group)
        x = Vector{Float32}(group[1, covariates])
        ty = group[group[:, colnames.mdv] .== 0, [colnames.time, colnames.dv]]
        𝐈 = Matrix{Float32}(group[group[:, colnames.mdv] .== 1, [colnames.time, colnames.amt, colnames.rate, colnames.duration]])
        callback_ = generate_dosing_callback(𝐈; S1=Float32(S1))
        indvs[i] = Individual(x, Float32.(ty[:, colnames.time]), Float32.(ty[:, colnames.dv]), callback_; id = first(group[:, colnames.id]))
    end
    
    return Population(indvs)
end

function find_colnames(df::DataFrame)
    columns = names(df)
    result = (
        id = match.(r"^(?!dvid$)(patient.*|subject.*|.*id)$"i, columns),
        time = match.(r"^time$"i, columns),
        dv = match.(r"^dv$"i, columns),
        mdv = match.(r"^mdv$"i, columns),
        amt = match.(r"^(amt|dose)$"i, columns),
        duration = match.(r"^dur(ation)?$"i, columns),
        rate = match.(r"^rate$"i, columns),
    )
    return check_colnames(result)
end

function check_colnames(x::NamedTuple)
    y = (id = "",)
    for key in keys(x)
        matched_values = filter(value -> value !== nothing, x[key])
        if length(matched_values) > 1 
            return throw(ArgumentError("Cannot identify '$(key)' column, the following column names are ambigious: '$(join(map(a -> getfield(a, :match), matched_values), "', '"))'. Change the column name(s) you do not want to use as '$(key)' and try again."))
        elseif isempty(matched_values) # duration and rate column can be missing.
            if !(key ∈ [:duration, :rate])
                return throw(ArgumentError("Cannot identify '$(key)' column, check if the column is present and try again."))
            else
                continue
            end
        else
            y = merge(y, [key => only(matched_values).match])
        end
    end
    return y
end

function handle_missing_rate_duration(df::DataFrame, colnames)
    df_ = copy(df)
    if !(:rate in keys(colnames))
        if :duration in keys(colnames)
            @info "Did not find a rate column. Attemping to fill in missing data from '$(colnames.duration)' column."
            df_[!, :RATE] = df_[:, colnames.amt] ./ df_[:, colnames.duration]
        else
            @info "Did not find a rate column. Assumed all doses are bolus doses with duration of one minute."
            df_[!, :RATE] = df_[:, colnames.amt] .* 60.
        end
        colnames = merge(colnames, (rate = "RATE",))
    end
    
    if !(:duration in keys(colnames))
        if :rate in keys(colnames)
            @info "Did not find a duration column. Attemping to fill in missing data from '$(colnames.rate)' column."
            df_[!, :DURATION] = df_[:, colnames.amt] ./ df_[:, colnames.rate]
        else
            @info "Did not find a duration column. Assumed all doses are bolus doses with duration of one minute."
            df_[!, :DURATION] = df_[:, colnames.amt].^0 .* 1/60
        end
        colnames = merge(colnames, (duration = "DURATION",))
    end

    return df_, colnames
end

end
