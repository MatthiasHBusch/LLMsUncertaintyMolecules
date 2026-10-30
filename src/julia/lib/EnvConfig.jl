#=
EnvConfig.jl — minimal .env loader (no package dependency).

Credentials must never live in tracked source. `LLMs.jl` therefore reads its API
key from the environment, and this loader populates the environment from a local
`.env` file in the repository root that is git-ignored.

Format (one per line, `#` comments and blank lines ignored):

    OPENROUTER_API_KEY=<your-openrouter-key>

Values are taken literally to the end of the line; surrounding single or double
quotes are stripped. An existing environment variable is NOT overwritten unless
`override=true`, so a real environment (CI, a shell export) always wins over the
file.
=#

"""
    load_dotenv(path=joinpath(@__DIR__, ".env"); override=false, quiet=false)

Load `KEY=VALUE` pairs from `path` into `ENV`. Returns the vector of names that
were set. Missing file is not an error (returns an empty vector) — that is the
normal case on a machine that supplies credentials through the real environment.
"""
function load_dotenv(path::AbstractString=joinpath(@__DIR__, ".env");
                     override::Bool=false, quiet::Bool=false)
    names = String[]
    if !isfile(path)
        quiet || @info "load_dotenv: no .env at $path (using the ambient environment)"
        return names
    end
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        i = findfirst('=', s)
        i === nothing && continue
        name = strip(s[1:prevind(s, i)])
        startswith(name, "export ") && (name = strip(name[8:end]))
        value = strip(s[nextind(s, i):end])
        isempty(name) && continue
        if length(value) >= 2 && ((value[1] == '"' && value[end] == '"') ||
                                  (value[1] == '\'' && value[end] == '\''))
            value = value[2:prevind(value, lastindex(value))]
        end
        if override || !haskey(ENV, name)
            ENV[name] = value
            push!(names, name)
        end
    end
    return names
end

"""
    require_env(name) -> String

Fetch a required credential, failing loudly (and without echoing the value) if it
is absent. Use at the point of construction so a missing key is obvious rather
than surfacing as a confusing 401 later.
"""
function require_env(name::AbstractString)
    v = get(ENV, name, "")
    isempty(v) && error("Missing credential $name. Set the environment variable $name or put the line " *
                        "$name=<value> into " * normpath(joinpath(@__DIR__, "..", "..", "..", ".env")) *
                        " (repository root, git-ignored).")
    return v
end
