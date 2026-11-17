# Runs every tutorial in `tutorials/` in a fresh Julia process and records how long each one takes.
#
#     julia --project=docs docs/timings.jl [output.csv]
#
# Each tutorial is run as a plain script (all `{julia}` cells that are not `#| eval: false`), so the
# timing includes compilation, as a reader running the notebook would experience it. The seeded checks
# that sit in the hidden cells of the tutorials run as well, so a tutorial that stops learning fails
# here too. Set `DCM_TUTORIALS=01,02` to time a subset and `DCM_TUTORIAL_TIMEOUT` (seconds, default
# 1200) to change the limit per tutorial.
#
# The result is written as CSV (default `tutorial_timings.csv`) and, when running on GitHub Actions,
# appended to the job summary. The exit code is non-zero if any tutorial fails or times out.

using Dates

const TUTORIAL_DIR = normpath(joinpath(@__DIR__, "..", "tutorials"))
const TIMEOUT = parse(Float64, get(ENV, "DCM_TUTORIAL_TIMEOUT", "1200"))

"""Concatenate the executed `{julia}` cells of a notebook into one script."""
function extract_cells(src::AbstractString)
    text = read(src, String)
    cells = [m.captures[1] for m in eachmatch(r"(?s)```\{julia\}\n(.*?)```", text)]
    filter!(c -> !occursin("#| eval: false", c), cells)
    return replace(join(cells, "\n"), "@__DIR__" => repr(TUTORIAL_DIR))
end

function time_tutorial(file)
    script = joinpath(mktempdir(), replace(file, ".qmd" => ".jl"))
    write(script, extract_cells(joinpath(TUTORIAL_DIR, file)))
    project = Base.active_project()
    cmd = `$(Base.julia_cmd()) --project=$project --threads=auto $script`
    log = IOBuffer()
    t0 = time()
    proc = run(pipeline(ignorestatus(cmd); stdout = log, stderr = log); wait = false)
    status = if timedwait(() -> !process_running(proc), TIMEOUT) == :timed_out
        kill(proc)
        "timeout"
    else
        success(proc) ? "ok" : "failed"
    end
    elapsed = time() - t0
    status == "ok" || print(stderr, "\n--- output of $file ---\n", String(take!(log)), "\n")
    return (; file, seconds = elapsed, status)
end

function main()
    out = isempty(ARGS) ? "tutorial_timings.csv" : first(ARGS)
    selected = filter(!isempty, split(get(ENV, "DCM_TUTORIALS", ""), ','))
    files = sort(filter(f -> endswith(f, ".qmd"), readdir(TUTORIAL_DIR)))
    isempty(selected) || filter!(f -> any(s -> startswith(f, s), selected), files)

    results = map(files) do file
        println("running $file ...")
        r = time_tutorial(file)
        println("  $(r.status) in $(round(r.seconds; digits = 1)) s")
        r
    end

    open(out, "w") do io
        println(io, "tutorial,seconds,status")
        for r in results
            println(io, r.file, ",", round(r.seconds; digits = 1), ",", r.status)
        end
    end

    if haskey(ENV, "GITHUB_STEP_SUMMARY")
        open(ENV["GITHUB_STEP_SUMMARY"], "a") do io
            println(io, "### Tutorial timings ($(Dates.format(now(UTC), "yyyy-mm-dd HH:MM")) UTC, Julia $VERSION)\n")
            println(io, "| tutorial | seconds | status |\n|:--|--:|:--|")
            for r in results
                println(io, "| ", r.file, " | ", round(r.seconds; digits = 1), " | ", r.status, " |")
            end
            println(io, "| **total** | **", round(sum(r.seconds for r in results); digits = 1), "** | |")
        end
    end

    all(r -> r.status == "ok", results) || exit(1)
end

main()
