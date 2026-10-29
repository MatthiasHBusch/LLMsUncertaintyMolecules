#
# Shared definitions for the v2 Julia runners (Run_ICL.jl, Run_ZeroShot_TestSet.jl).
#
# Built on the in-house library in the directory $LLM_JULIA_LIBS (LLMUtils.jl via LLMs.jl, which
# also holds the model handles; FileWritingHelpers.jl; not part of this repository), exactly as the
# memorization paper's Run_ZeroShot.jl and the transfer-learning paper's Run_*_Experiments.jl.
#
# One addition over the library: `ask_gpt_full` returns the FULL response (text plus usage:
# prompt / cached / completion / reasoning tokens, cost, finish_reason, provider, served model)
# instead of the text alone, so every call's token counts are stored with the result
# (decision 2026-09-08). It reuses the library's request construction (`get_url`, `get_body`
# with the ("role", text, "ephemeral") cache-control convention and the reasoning-effort
# handling, `get_header`), its cost-limit check and its global usage accounting, so LLMUsage.json
# keeps tracking the spend as for every other project.
#
# Model registry (since 2026-10-08): src/registry/models_v2.json (`load_models`, `PANEL` = its
# in_panel entries). For pinned entries (allow_fallbacks = false) `build_request_body` adds
# provider.allow_fallbacks = false to the library's body (the library itself is not changed).
#
using CSV, DataFrames, JSON, Dates, Printf, HTTP, Random, Statistics
# The in-house library is not part of this repository: LLM_JULIA_LIBS names its directory.
const LLM_LIBS = get(ENV, "LLM_JULIA_LIBS") do
    error("set LLM_JULIA_LIBS to the directory holding LLMs.jl, LLMUtils.jl, FileWritingHelpers.jl")
end
include(joinpath(LLM_LIBS, "LLMs.jl"))                   # -> LLMUtils.jl, key_openrouter, model handles
include(joinpath(LLM_LIBS, "FileWritingHelpers.jl"))

const JL_DIR = @__DIR__
const ROOT = normpath(joinpath(JL_DIR, "..", ".."))
const DATA_DIR = joinpath(ROOT, "data")
const RESULTS_DIR = joinpath(ROOT, "results")
const MEMO45_DIR = joinpath(ROOT, "src", "memo45")

# OpenRouter key: the environment variable OPENROUTER_API_KEY, else the line OPENROUTER_API_KEY=... of the
# git-ignored .env file in the repository root, replaces the library's key_openrouter. Never printed.
function repo_env_value(name)
    v = strip(get(ENV, name, ""))
    isempty(v) || return String(v)
    f = joinpath(ROOT, ".env")
    isfile(f) || return ""
    for line in eachline(f)
        s = strip(line)
        (isempty(s) || startswith(s, "#") || !occursin("=", s)) && continue
        k, val = split(s, "="; limit=2)
        strip(replace(k, r"^export\s+" => "")) == name || continue
        val = strip(val)
        if length(val) >= 2 && val[1] == val[end] && val[1] in ('"', '\'')
            val = val[2:end-1]
        end
        return String(val)
    end
    return ""
end
let k = repo_env_value("OPENROUTER_API_KEY")
    isempty(k) || (global key_openrouter = k)
end

# ------------------------------------------------------------------ datasets
# Prompt words of blinding level 1 (real property name, real values, real SMILES), in the
# wording of the v1 io prompt. Mirrors src/common.py::DATASETS (the Python copy is the
# reference for the prompt-parity check, src/check_prompt_parity.py).
# ESOL: the file also carries the ESOL-*predicted* solubility; the label names the MEASURED value.
const DATASETS = Dict(
    "lipophilicity" => Dict(
        "name" => "Lipophilicity (MoleculeNet, AZ/ChEMBL)",
        "expertise" => "physical chemistry and lipophilicity prediction",
        "task" => "lipophilicity (octanol/water distribution coefficient logD at pH 7.4)",
        "medium" => "", "label" => "logD (pH 7.4)",
        "predict" => "octanol/water distribution coefficient logD at pH 7.4"),
    "bace" => Dict(
        "name" => "BACE (MoleculeNet)",
        "expertise" => "medicinal chemistry and BACE-1 inhibitor potency prediction",
        "task" => "BACE-1 inhibitory potency (pIC50)", "medium" => "", "label" => "BACE-1 pIC50",
        "predict" => "BACE-1 inhibitory potency (pIC50)"),
    "antiviral" => Dict(
        "name" => "Antiviral potency (ASAP/Polaris 2025)",
        "expertise" => "medicinal chemistry and SARS-CoV-2 main protease (Mpro) inhibitor potency prediction",
        "task" => "SARS-CoV-2 main protease (Mpro) inhibitory potency (pIC50)", "medium" => "",
        "label" => "SARS-CoV-2 Mpro pIC50", "predict" => "SARS-CoV-2 main protease (Mpro) inhibitory potency (pIC50)"),
    "esol" => Dict(
        "name" => "ESOL / Delaney",
        "expertise" => "physical chemistry and solubility prediction",
        "task" => "measured aqueous solubility logS (log10 of the solubility in mol/L)", "medium" => "",
        "label" => "measured logS (log10 mol/L)",
        "predict" => "measured aqueous solubility logS (log10 of the solubility in mol/L)"),
)
const DATASET_KEYS = ["lipophilicity", "bace", "antiviral", "esol"]

# ------------------------------------------------------------------ models
const MODELS_V2_FILE = joinpath(ROOT, "src", "registry", "models_v2.json")

"Registry entries of src/registry/models_v2.json in file order (= panel order)."
registry_entries() = JSON.parsefile(MODELS_V2_FILE)["models"]

"v2 model registry (tag -> entry: endpoint, providers, service tier, allow_fallbacks, prices, ...)."
load_models() = Dict(e["tag"] => e for e in registry_entries())

"Model registry of the memorization paper (tag -> endpoint, providers, service tier, prices)."
function load_models45()
    entries = JSON.parsefile(joinpath(MEMO45_DIR, "models_45.json"))["models"]
    return Dict(e["tag"] => e for e in entries)
end

# Main panel (decision 2026-10-08): the in_panel entries of models_v2.json, in registry order.
const PANEL = [String(e["tag"]) for e in registry_entries() if get(e, "in_panel", false)]

"Every enabled registry tag (panel and reserve), in registry order."
enabled_tags() = [String(e["tag"]) for e in registry_entries() if get(e, "enabled", true)]

"""
    resolve_tags(spec)

Model selection of the runners (ICL_MODELS / PROBE_MODELS / ZS_MODELS): `all` = every enabled
registry tag, `panel` = PANEL, otherwise a comma list of tags (`all` / `panel` may be mixed with
tags, duplicates are dropped, order kept). Unknown tags raise an error.
"""
function resolve_tags(spec::AbstractString)
    out = String[]
    for x in split(spec, ",")
        x = strip(x)
        isempty(x) && continue
        add = x == "all" ? enabled_tags() : x == "panel" ? PANEL : [String(x)]
        append!(out, add)
    end
    out = unique(out)
    known = Set(String(e["tag"]) for e in registry_entries())
    for t in out
        t in known || error("unknown model tag '$t' (see src/registry/models_v2.json)")
    end
    return out
end

# Routing (endpoint, provider order, service tier) -> allow_fallbacks, registered by `llm_access`.
# The library's LLMAccessOpenRouter has no field for it; keeping it here leaves the library and
# every function that takes an `LLMAccess` (ask_threaded_full, ...) unchanged.
const ALLOW_FALLBACKS = Dict{Tuple{String,Vector{String},String},Bool}()
_route_key(a) = (String(a.model), String[String(p) for p in a.providers_order], String(a.service_tier))
fallbacks_allowed(a::LLMAccess) = hasproperty(a, :providers_order) ? get(ALLOW_FALLBACKS, _route_key(a), true) : true

"OpenRouter access object for a registry tag, with the key from LLMs.jl (never stored here)."
function llm_access(tag::AbstractString; models=load_models())
    haskey(models, tag) || error("unknown model tag '$tag' (see src/registry/models_v2.json)")
    md = models[tag]
    provs = String[String(p) for p in get(md, "providers", [])]
    acc = LLMAccessOpenRouter(key_openrouter, md["endpoint"], provs, get(md, "service_tier", ""))
    ALLOW_FALLBACKS[_route_key(acc)] = Bool(get(md, "allow_fallbacks", isempty(provs)))
    return acc
end

"Zero-shot prompt words of the memorization paper's dataset registry."
function load_datasets45()
    entries = JSON.parsefile(joinpath(MEMO45_DIR, "datasets_45.json"))["datasets"]
    return Dict(e["key"] => e for e in entries)
end

# ------------------------------------------------------------------ data & splits
function load_dataset(key::AbstractString)
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "$(key).csv")))
    df.mol_id = string.(df.mol_id)
    return df
end

"""
    split_path(key; draw=1)

data/splits/<key>_v2.json for draw 1 (the main split), data/splits/<key>_v2_draw<d>.json for the
additional context draws d = 2, 3 (src/make_splits.py --draws 3: same test_ids, sizes and training
pool, independently permuted pool order). Mirrors src/common.py::split_path.
"""
function split_path(key::AbstractString; draw::Integer=1)
    draw >= 1 || error("draw must be >= 1, got $draw")
    return joinpath(DATA_DIR, "splits", draw == 1 ? "$(key)_v2.json" : "$(key)_v2_draw$(draw).json")
end

"v2 split (src/make_splits.py): test_ids, train_order_ids, sizes, n_train_pool, provisional; `draw` selects the context draw."
function load_split(key::AbstractString; draw::Integer=1)
    sp = JSON.parsefile(split_path(key; draw=draw))
    Int(get(sp, "draw", 1)) == draw || error("$(split_path(key; draw=draw)): stored draw $(get(sp, "draw", 1)) != requested $draw")
    return sp
end

"Test DataFrame and the first `size` training (smiles, value) pairs of the split order of context draw `draw`."
function cell_data(key::AbstractString, size::Int; draw::Integer=1)
    df = load_dataset(key)
    sp = load_split(key; draw=draw)
    size <= sp["n_train_pool"] || error("$key: size $size exceeds the training pool ($(sp["n_train_pool"]))")
    byid = Dict(r.mol_id => r for r in eachrow(df))
    train = [(string(byid[m].smiles), byid[m].value) for m in sp["train_order_ids"][1:size]]
    test = DataFrame([byid[m] for m in sp["test_ids"]])
    return test, train
end

"'min' | 'max' | 'all' | comma list -> valid ladder sizes of the dataset."
function resolve_sizes(key::AbstractString, spec::AbstractString)
    ladder = Int.(load_split(key)["sizes"])
    out = Int[]
    for s in split(spec, ",")
        s = strip(s)
        if s == "min"; push!(out, ladder[1])
        elseif s == "max"; push!(out, ladder[end])
        elseif s == "all"; append!(out, ladder)
        else
            n = parse(Int, s)
            n <= ladder[end] && push!(out, n)     # a rung above the pool does not exist here
        end
    end
    return sort(unique(out))
end

# ------------------------------------------------------------------ parsing
"Last number in the reply, scientific notation first (== 45/Run_ZeroShot.jl::parse_number)."
function parse_number(s::AbstractString)
    txt = replace(String(s), "\u2212" => "-", "\u2013" => "-", "," => "")
    m = match(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+", txt)
    if m !== nothing
        v = tryparse(Float64, m.match)
        v === nothing || return v
    end
    return search_for_last_number_in_string(String(txt))
end

function _normalise_json(s::AbstractString)
    t = replace(String(s), "\u2212" => "-")
    t = replace(t, r"//[^\n]*" => "")
    t = replace(t, r"/\*.*?\*/"s => "")
    t = replace(t, r",\s*([}\]])" => s"\1")
    t = replace(t, r"\bNaN\b" => "null", r"\bInfinity\b" => "null")
    return t
end

"Last fenced ```json block (fallback: greedy {...}); == v1 analyze_uc.extract_json_block."
function extract_json_block(text::AbstractString)
    isempty(text) && return nothing
    cands = [m.captures[1] for m in eachmatch(r"```json\s*(.*?)\s*```"si, text)]
    reverse!(cands)
    if isempty(cands)
        m = match(r"\{[\s\S]*\}"s, text)
        m === nothing || push!(cands, m.match)
    end
    for c in cands
        for variant in (c, _normalise_json(c), replace(_normalise_json(c), "'" => "\""))
            try
                obj = JSON.parse(variant)
                obj isa AbstractDict && return obj
            catch
            end
        end
    end
    return nothing
end

function coerce_float(x)
    x === nothing && return nothing
    v = x isa Number ? Float64(x) : tryparse(Float64, replace(strip(string(x)), "\u2212" => "-"))
    (v === nothing || !isfinite(v)) && return nothing
    return v
end

"prediction / uncertainty / parse_ok from an io response."
function parse_response(text::AbstractString)
    obj = extract_json_block(text)
    pred = obj === nothing ? nothing : coerce_float(get(obj, "prediction", nothing))
    unc = obj === nothing ? nothing : coerce_float(get(obj, "uncertainty", nothing))
    unc = unc === nothing ? nothing : abs(unc)
    return Dict("prediction" => pred, "uncertainty" => unc,
                "parse_ok" => pred !== nothing && unc !== nothing)
end

# ------------------------------------------------------------------ API call with full usage
struct UnsupportedRequest <: Exception
    msg::String
end

"""
    build_request_body(llmaccess, conversation; max_tokens, reasoning_effort, temperature, top_p, allow_fallbacks)

The JSON body `ask_gpt_full` sends: the library's `get_body` (messages with cache_control,
service_tier, provider.order, reasoning.effort, usage.include), plus provider.allow_fallbacks =
false when the registry entry pins its providers. The library body is parsed, patched and
re-serialised; `provider.order` as set by the library is kept.
"""
function build_request_body(llmaccess::LLMAccess, conversation::Vector{Tuple};
                            max_tokens::Int=32000, reasoning_effort=missing,
                            temperature=1.0, top_p=1.0, allow_fallbacks::Bool=fallbacks_allowed(llmaccess))
    body = get_body(llmaccess, conversation; temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    allow_fallbacks && return body
    j = JSON.parse(body)
    prov = get(j, "provider", nothing)
    prov isa AbstractDict || (prov = Dict{String,Any}())
    prov["allow_fallbacks"] = false
    j["provider"] = prov
    return JSON.json(j)
end

"""
    ask_gpt_full(llmaccess, conversation; max_tokens, reasoning_effort, temperature, top_p, retries)

Like LLMUtils.ask_gpt (same URL / body / header / cost-limit check / usage accounting), but
returns a Dict with the answer text AND the per-call usage. Sampling parameters default to
the provider default (temperature = top_p = 1.0 are omitted from the body by `get_body`).
The body comes from `build_request_body` (library body + provider.allow_fallbacks = false
for pinned registry entries).
HTTP 400 (e.g. an effort level the endpoint rejects) and HTTP 404 (no endpoint for a pinned
provider without fallback) raise `UnsupportedRequest` at once, also when reported in the body;
other failures are retried with a 10 s pause; 429 honours Retry-After when present.
"""
function ask_gpt_full(llmaccess::LLMAccess, conversation::Vector{Tuple};
                      max_tokens::Int=32000, reasoning_effort=missing,
                      temperature=1.0, top_p=1.0, retries::Int=5, timeout::Int=900,
                      allow_fallbacks::Bool=fallbacks_allowed(llmaccess))
    url = get_url(llmaccess, "/chat/completions")
    body = build_request_body(llmaccess, conversation; temperature=temperature, top_p=top_p,
                              max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                              allow_fallbacks=allow_fallbacks)
    header = get_header(llmaccess)
    check_llm_cost_limit(llmaccess.model, estimate_tokens(conversation), max_tokens) ||
        error("LLMUtils cost limit reached -- see LLMUtilsData/LLMCosts.json")
    attempts = retries + 1
    while true
        attempts -= 1
        t0 = time()
        try
            resp = HTTP.request("POST", url, headers=header, body=body, readtimeout=timeout)
            j = JSON.parse(String(resp.body))
            if !haskey(j, "choices")
                err = get(j, "error", nothing)
                code = err isa AbstractDict ? get(err, "code", nothing) : nothing
                code in (400, 404) && throw(UnsupportedRequest(first("HTTP $(code): $(JSON.json(err))", 400)))
                attempts > 0 || error("upstream error without choices: $(get(j, "error", j))")
                println("  upstream error (no choices): $(get(j, "error", j)) -- retrying in 12 s")
                sleep(12); continue
            end
            update_llm_usage_from_response(llmaccess.model, j)
            u = get(j, "usage", Dict())
            cdet = something(get(u, "completion_tokens_details", Dict()), Dict())
            pdet = something(get(u, "prompt_tokens_details", Dict()), Dict())
            ch = j["choices"][1]
            txt = something(get(ch["message"], "content", ""), "")
            return Dict{String,Any}(
                "text" => txt,
                "finish" => something(get(ch, "finish_reason", ""), ""),
                "prompt_tokens" => Int(something(get(u, "prompt_tokens", 0), 0)),
                "cached_tokens" => Int(something(get(pdet, "cached_tokens", 0), 0)),
                "cache_write_tokens" => Int(something(get(pdet, "cache_write_tokens", 0), 0)),
                "completion_tokens" => Int(something(get(u, "completion_tokens", 0), 0)),
                "reasoning_tokens" => Int(something(get(cdet, "reasoning_tokens", 0), 0)),
                "cost" => Float64(something(get(u, "cost", 0.0), 0.0)),
                "provider" => string(get(j, "provider", "")),
                "model_served" => string(get(j, "model", "")),
                "latency_s" => round(time() - t0, digits=2))
        catch e
            if e isa HTTP.ExceptionRequest.StatusError
                # 400: request rejected (e.g. an effort level); 404: no endpoint can serve it (pinned
                # provider without fallback, unknown model) -- neither is fixed by retrying
                e.status in (400, 404) && throw(UnsupportedRequest(first("HTTP $(e.status): $(String(e.response.body))", 400)))
                if e.status == 429 && attempts > 0     # bounded by `retries`, as in LLMUtils.ask_gpt
                    m = match(r"Retry-After: (\d+)"i, string(e))
                    wait = m === nothing ? 10 : parse(Int, m.captures[1])
                    println("  rate limit -- waiting $wait s"); sleep(wait); continue
                end
            end
            attempts > 0 || rethrow(e)
            println("  error: $(first(sprint(showerror, e), 200)) -- retrying in 10 s")
            sleep(10)
        end
    end
end

"""
    ask_threaded_full(llmaccess, conversations, on_result; num_threads, kwargs...)

Green-thread pool as in LLMUtils.ask_gpt_threaded. `on_result(idx, result_or_nothing)` is
called under a lock as each answer arrives, so the caller can append to disk immediately
(resumable mid-cell). Stops the whole cell on `UnsupportedRequest` or when `stop[]` is set.
"""
function ask_threaded_full(llmaccess::LLMAccess, conversations::Vector{Vector{Tuple}}, on_result::Function;
                           num_threads::Int=8, stop::Ref{Bool}=Ref(false), kwargs...)
    n = length(conversations)
    n == 0 && return
    lk = ReentrantLock()
    ch = Channel{Int}(min(num_threads, n))
    unsupported = Ref{Union{Nothing,String}}(nothing)
    workers = [@async begin
        for idx in ch
            (stop[] || unsupported[] !== nothing) && continue
            res = nothing
            try
                res = ask_gpt_full(llmaccess, conversations[idx]; kwargs...)
            catch e
                if e isa UnsupportedRequest
                    unsupported[] = e.msg
                else
                    lock(lk) do; println("  call $idx failed: $(first(sprint(showerror, e), 200))"); end
                end
            end
            lock(lk) do; on_result(idx, res); end
        end
    end for _ in 1:min(num_threads, n)]
    for i in 1:n; put!(ch, i); end
    close(ch)
    foreach(wait, workers)
    return unsupported[]
end

# ------------------------------------------------------------------ result files
function jsonl_append(path::AbstractString, rec::AbstractDict)
    open(path, "a") do io
        println(io, JSON.json(rec))
    end
end

"(mol_id, repeat) pairs already stored in a JSONL cell file."
function existing_pairs(path::AbstractString)
    done = Set{Tuple{String,Int}}()
    isfile(path) || return done
    for line in eachline(path)
        isempty(strip(line)) && continue
        try
            r = JSON.parse(line)
            push!(done, (string(r["mol_id"]), Int(r["repeat"])))
        catch
        end
    end
    return done
end

"Cumulative OpenRouter spend recorded by the shared library, in USD (as in 45/Run_ZeroShot.jl)."
function current_usage()
    f = joinpath(LLM_LIBS, "LLMUtilsData", "LLMUsage.json")
    isfile(f) || return 0.0
    try
        return Float64(get(JSON.parsefile(f), "total_cost_usd", 0.0))
    catch
        return 0.0
    end
end

envget(name, default) = get(ENV, name, default)

"Comma list from an environment variable: entries stripped, empty entries dropped."
envlist(name, default) = String[strip(x) for x in split(envget(name, default), ",") if !isempty(strip(x))]

const KNOWN_LEVELS = ["none", "minimal", "low", "medium", "high"]

"Warn (do not drop) effort levels outside none/minimal/low/medium/high and the max_tokens:N form."
function warn_unknown_levels(levels)
    for l in levels
        (l in KNOWN_LEVELS || startswith(l, "max_tokens:")) ||
            @warn "unknown reasoning level '$l' -- passed to the endpoint as reasoning.effort unchanged"
    end
    return levels
end
envflag(name) = get(ENV, name, "0") == "1"
