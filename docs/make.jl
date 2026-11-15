using Documenter
using DeepCompartmentModels
using DataFrames # loads the DCMDataFramesExt extension (`load`)

include("qmd.jl")

# The tutorials are Quarto notebooks in `tutorials/` (see `qmd.jl`). The ones listed
# here run against the current API and are built into the site; the remaining
# notebooks in `tutorials/` are kept but excluded until the features they use work
# again (see docs/src/limitations.md). Set `DCM_TUTORIALS=01,02` to build a subset.
const TUTORIALS = [
    "01-basic-dcm.qmd",
    "05-loading-data.qmd",
    "02-mixed-effects.qmd",
    "07-objectives.qmd",
    "03-universal-diffeq.qmd",
    "06-custom-ode.qmd",
    "04-covariate-effects.qmd",
]

selected = filter(!isempty, split(get(ENV, "DCM_TUTORIALS", ""), ','))
tutorials = filter(TUTORIALS) do f
    isempty(selected) || any(s -> startswith(f, s), selected)
end

generated = joinpath(@__DIR__, "src", "generated")
rm(generated; force = true, recursive = true)
tutorial_pages = [
    String(qmd_to_md(joinpath(@__DIR__, "..", "tutorials", f), joinpath(generated, replace(f, ".qmd" => ".md")))) =>
        "generated/" * replace(f, ".qmd" => ".md")
    for f in tutorials
]

makedocs(;
    modules = [DeepCompartmentModels],
    sitename = "DeepCompartmentModels.jl",
    authors = "Janssen",
    checkdocs = :none,
    doctest = false,   # several docstring examples are not runnable yet
    format = Documenter.HTML(;
        canonical = "https://janssena.github.io/DeepCompartmentModels.jl",
        edit_link = "main",
        prettyurls = get(ENV, "CI", "false") == "true",
        size_threshold = nothing,
        example_size_threshold = nothing,
        size_threshold_warn = nothing,
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["Overview" => "tutorials/index.md"; tutorial_pages],
        "Status and limitations" => "limitations.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/Janssena/DeepCompartmentModels.jl.git",
    devbranch = "main",
    push_preview = true,
)
