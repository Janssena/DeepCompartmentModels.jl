# Converts the Quarto notebooks in `tutorials/` into Documenter pages, so that the
# `.qmd` files stay the single source of truth: they render standalone with
# Quarto, and on the docs site every `{julia}` cell is executed by Documenter as
# an `@example` block at build time.
#
# Supported subset of Quarto:
#   - YAML front matter: `title` becomes the page heading, `subtitle` a lead paragraph
#   - ```{julia} cells, with `#| eval: false` (show only), `#| echo: false` (run, hide
#     the code) and `#| output: false` (run, hide the output)
#   - `::: {.callout-note}` ... `:::` callouts (note, tip, warning, important, caution)
#   - `joinpath(@__DIR__, "..", "data", ...)` inside cells becomes
#     `joinpath(pkgdir(DeepCompartmentModels), "data", ...)`; any other `@__DIR__` is
#     replaced by the absolute `tutorials/` directory
#   - links to other `*.qmd` files are rewritten to the generated `*.md` pages

const CALLOUTS = Dict(
    "note" => "note", "tip" => "tip", "warning" => "warning",
    "important" => "warning", "caution" => "warning",
)

"""
    qmd_to_md(src, dest)

Write the Documenter version of the Quarto notebook `src` to `dest`. Returns the
page title.
"""
function qmd_to_md(src::AbstractString, dest::AbstractString)
    lines = readlines(src)
    page = replace(first(splitext(basename(src))), r"[^A-Za-z0-9]" => "")
    srcdir = dirname(abspath(src))

    title, subtitle = "", ""
    i = 1
    if !isempty(lines) && strip(lines[1]) == "---"
        i = 2
        while i <= length(lines) && strip(lines[i]) != "---"
            m = match(r"^(title|subtitle):\s*\"?(.*?)\"?\s*$", lines[i])
            if m !== nothing
                m.captures[1] == "title" ? (title = m.captures[2]) : (subtitle = m.captures[2])
            end
            i += 1
        end
        i += 1
    end

    out = String["```@meta\nEditURL = \"$(relpath(abspath(src), dirname(abspath(dest))))\"\n```\n", "# $title\n"]
    isempty(subtitle) || push!(out, "*$subtitle*\n")

    callout = nothing   # admonition type while inside a callout
    n_callout = 0
    while i <= length(lines)
        line = lines[i]
        if (m = match(r"^```\{julia\}\s*$", line)) !== nothing
            i += 1
            body = String[]
            while i <= length(lines) && !startswith(lines[i], "```")
                push!(body, lines[i]); i += 1
            end
            opts = Dict{String,String}()
            filter!(body) do l
                o = match(r"^#\|\s*([\w-]+):\s*(\S+)\s*$", l)
                o === nothing && return true
                opts[o.captures[1]] = o.captures[2]
                false
            end
            code = join(body, "\n")
            # `joinpath(@__DIR__, "..", "data", ...)` in a notebook becomes a path relative to the package
            code = replace(code, "joinpath(@__DIR__, \"..\", \"data\"" => "joinpath(pkgdir(DeepCompartmentModels), \"data\"")
            code = replace(code, "@__DIR__" => repr(srcdir))
            fence = if get(opts, "eval", "true") == "false"
                "```julia"
            elseif get(opts, "echo", "true") == "false"
                "```@setup $page"
            else
                "```@example $page"
            end
            get(opts, "output", "true") == "false" && fence == "```@example $page" && (code *= "\nnothing # hide")
            block = fence * "\n" * code * "\n```\n"
            push!(out, callout === nothing ? block : _indent(block))
        elseif (m = match(r"^:::\s*\{\.callout-(\w+)[^}]*\}\s*$", line)) !== nothing
            callout = get(CALLOUTS, m.captures[1], "note")
            n_callout += 1
            push!(out, "!!! $callout\n")
        elseif strip(line) == ":::" && callout !== nothing
            callout = nothing
            push!(out, "")
        else
            line = replace(line, r"\]\(([^)#]*)\.qmd(#[^)]*)?\)" => s"](\1.md\2)")
            push!(out, callout === nothing ? line : _indent(line))
        end
        i += 1
    end

    mkpath(dirname(dest))
    write(dest, join(out, "\n") * "\n")
    return title
end

_indent(s::AbstractString) = join(("    " * l for l in split(s, '\n')), "\n")
