import Random
import CSV

using DataFrames

function load_theophylline(::Type{T}=Float32) where T
    file = joinpath("data", "theophylline.csv")
    df = DataFrame(CSV.File(file))

    df_group = groupby(df, :ID)
    indvs = Vector{AbstractIndividual}(undef, length(df_group))
    for (i, group) in enumerate(df_group)
        x = group[1:1, "WEIGHT"]
        ty = group[.!ismissing.(group.DV), [:TIME, :DV]]
        # AMTs in data set are likely in AMT⋅1e2 mg
        I = hcat(Matrix(group[1:1, [:TIME, :AMT]]) .* [1 100], [group.AMT[1] * 100 * 60 1/60])
        cb = generate_dosing_callback(I, T) 
        indvs[i] = Individual(group.ID[1], x, ty.TIME, ty.DV, cb, T)
    end

    return Population(indvs)
end

function load_warfarin(::Type{T}=Float32) where T
    file = joinpath("data", "warfarin.csv")
    df = DataFrame(CSV.File(file))
    filter!(row -> row.DVID == 1, df)

    df_group = groupby(df, :ID)
    indvs = Vector{AbstractIndividual}(undef, length(df_group))
    for (i, group) in enumerate(df_group)
        x = Vector{T}(group[1, ["WEIGHT", "AGE", "SEX"]])
        x[3, :] .= T.(x[3, :] .== 0) # set less dominant group to 1
        ty = group[.!ismissing.(group.DV), [:TIME, :DV]]
        I = Matrix{T}(group[group.MDV .== 1, [:TIME, :DOSE, :RATE, :DURATION]])
        cb = generate_dosing_callback(I, T) 
        indvs[i] = Individual(group.ID[1], x, ty.TIME, ty.DV, cb, T)
    end

    return Population(indvs)
end

# Tobramycin follows a two-compartment model structure
function load_tobramycin(::Type{T}=Float32) where T
    duration = 40 / 60 # Doses are infusion from 20 -- 60 min
    file = joinpath("data", "tobramycin.csv")
    df = DataFrame(CSV.File(file))

    df_group = groupby(df, :ID)
    indvs = Vector{AbstractIndividual}(undef, length(df_group))
    for (i, group) in enumerate(df_group)
        x = Vector{T}(group[1, ["WT", "AGE", "SEX", "CLCR"]])
        ty = group[iszero.(group.EVID), [:TIME, :CP]]
        doses = Matrix{T}(group[group.EVID .== 1, [:TIME, :DOSE]])
        I = hcat(doses, (1 / duration) .* doses[:, 2], fill(T(duration), size(doses, 1)))
        cb = generate_dosing_callback(I, T) 
        indvs[i] = Individual(group.ID[1], x, ty.TIME, ty.CP, cb, T)
    end

    return Population(indvs)
end

function load_remifentanil(::Type{T}=Float32) where T
    file = joinpath("data", "reminfentanil.csv")
    df = DataFrame(CSV.File(file; missingstring="."))
    df[!, :SEX] = (df.SEX .== "M") .+ 0
    
    df_group = groupby(df, :ID)
    indvs = Vector{AbstractIndividual}(undef, length(df_group))
    for (i, group) in enumerate(df_group)
        x = Vector{T}(group[1, ["LBM", "AGE", "SEX"]])
        ty = group[.!ismissing.(group.DV), [:TIME, :DV]]
        I = hcat(Matrix{T}(group[.!ismissing.(group.AMT), [:TIME, :AMT, :RATE]]), T.(group.AMT[.!ismissing.(group.AMT)] ./ group.RATE[.!ismissing.(group.AMT)]))
        cb = generate_dosing_callback(I, T) 
        indvs[i] = Individual(group.ID[1], x, ty.TIME, ty.DV, cb, T)
    end

    return Population(indvs)
end


function handle_quin_intervals(df; steady_state_in = 15)
    filter!(row -> !(ismissing(row.conc) && ismissing(row.dose)), df)

    df_aug = DataFrame()
    for group in groupby(df, :ID)
        group_aug = copy(group)
        interval_idxs = findall(!ismissing, group.interval)

        if !isnothing(interval_idxs)
            for idx in interval_idxs
                interval_length = group.interval[idx]
                if idx == 1
                    steps = collect(1:steady_state_in) .* interval_length
                else
                    diff = group.time[idx] - group.time[idx-1]
                    if diff > steady_state_in * interval_length
                        steps = collect(1:steady_state_in) .* interval_length
                    else
                        steps = interval_length:interval_length:(diff - 0.1)
                    end
                end
                times = sort(collect(group.time[idx] .- steps))
                
                new_rows = DataFrame(fill(copy(group[idx, :]), length(times)))
                new_rows.time .= times
                new_rows.interval .= missing
                append!(group_aug, new_rows)
                sort!(group_aug, :time)
            end
        end
        group_aug[!, :time] = group_aug.time .- group_aug.time[1]
        append!(df_aug, group_aug)
    end

    return df_aug
end

# TODO: select specific OCC.
function load_quinidine(::Type{T}=Float32) where T
    file = joinpath("data", "quinidine.csv")
    df = DataFrame(CSV.File(file; missingstring="NA"))
    df_aug = handle_quin_intervals(df)
    df_aug[!, :Smokes] = (df_aug.Smoke .== "yes") .+ 0
    df_aug[!, :Alcohol] = map(value -> value == "current" ? 1 : (value == "former" ? 2 : 0), df_aug.Ethanol)
    df_aug[!, :occ] .= 1
    
    # only use data from OCC 1
    df_occ = DataFrame()
    for group in groupby(df_aug, :ID)
        for i in 2:nrow(group)
            diff = group.time[i] - group.time[i-1]
            if diff > 100
                group[i:end, :occ] .+= 1
            end 
        end
        subgroups = groupby(group, :occ)
        subgroup_to_keep = argmax([sum(.!ismissing.(subgroup.conc)) for subgroup in subgroups])
        subgroup = subgroups[subgroup_to_keep]
        subgroup[!, :time] = subgroup.time .- subgroup.time[1]
        last_dv = findlast(!ismissing, subgroup.conc)
        append!(df_occ, subgroup[1:last_dv, :])
    end
    df_occ[!, :Low_crcl] .= 0
    df_group = filter(group -> !all(ismissing.(group.conc)), groupby(df_occ, [:ID, :occ]))
    
    # Patient is set to have low CrCL if CrCL < 50 at the time of most of the concentation measurements
    for group in df_group
        crcl = group[.!ismissing.(group.conc), :Creatinine]
        group.Low_crcl .= (sum(crcl .== "< 50") > 0.5 * length(crcl)) + 0
    end

    covariates = ["Weight", "Age", "Smokes", "Alcohol", "Low_crcl", "glyco"]
    indvs = Vector{AbstractIndividual}(undef, length(df_group))
    for (i, group) in enumerate(df_group)
        x = Vector{T}(group[1, covariates])
        x[findfirst(==("glyco"), covariates)] = mean(group[.!ismissing.(group.conc), :glyco])
        ty = group[.!ismissing.(group.conc), [:time, :conc]]
        doses = Matrix{T}(group[.!ismissing.(group.dose), [:time, :dose]])
        I = hcat(doses, 60 .* doses[:, 2], fill(T(1/60), size(doses, 1)))
        cb = generate_dosing_callback(I, T)

        # There are no occasions
        # occasions = map(unique(group.occ)) do occ_id
        #     return findfirst(==(occ_id), group.occ) => findlast(==(occ_id), group.occ)
        # end

        indvs[i] = Individual(group.ID[1], x, ty.time, ty.conc, cb, T)
    end

    return Population(indvs)
end