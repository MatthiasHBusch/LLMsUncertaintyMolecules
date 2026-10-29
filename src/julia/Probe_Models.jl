#
# Effort-dosability probe: does the requested reasoning effort actually change what a model
# emits? (Lesson of the memorization paper: the requested effort is not the dose -- Kimi K2.6,
# GLM 5 / 5.2, Qwen3.5 and DeepSeek V4 Pro ignored the budget; Grok 4.5 and Gemini 3.5 Flash
# answered HTTP 400 to `none`. OpenRouter lists `reasoning_effort` for every candidate; that
# listing is what this probe measures.)
#
# One call per (model, level, repeat) of the zero-shot prompt (Prompts_ZS.jl, the prompt of
# Run_ZeroShot_TestSet.jl) on the FIRST molecule of the <dataset>_<rule> test set. Calls of one
# model run sequentially; models run concurrently. Request = exactly the runners' request
# (Common.jl::ask_gpt_full: registry routing, service tier, provider.allow_fallbacks).
#
# Environment
#   PROBE_MODELS      comma list of registry tags, or `all` (= every enabled entry of src/registry/models_v2.json,
#                     the default) / `panel` (= the in_panel entries); see Common.jl::resolve_tags
#   PROBE_LEVELS      comma list of effort levels   (default none,minimal,low,medium,high; unknown names
#                     are warned about and sent unchanged, but left out of the verdict)
#   PROBE_REPEATS     calls per (model, level)      (default 1; 2 RECOMMENDED for the panel -- one call per
#                     level cannot separate a dose effect from call-to-call noise. The summary uses the
#                     median over repeats.)
#   PROBE_MAX_TOKENS  completion cap                (default 32000)
#   PROBE_DATASET     dataset of the probe molecule (default lipophilicity)
#   PROBE_RULE        test-set rule                 (default v2test150)
#   PROBE_RETRIES     retries per call on errors other than HTTP 400 / 404 (default 1)
#   PROBE_BUDGET      hard ceiling in USD           (default 3.0). A call starts only if spent + the
#                     reserved worst case of every call in flight + its own worst case stays within it;
#                     otherwise it waits for in-flight calls, and is skipped (record status
#                     "skipped_budget") once even an idle start would exceed the ceiling. Worst case =
#                     2 x (prompt + PROBE_MAX_TOKENS at registry price): the factor 2 covers flex not being
#                     applied, DeepSeek peak hours, 1.1x regional hosts and a billed retry.
#   PROBE_DRYRUN      "1": print plan + prompt, no API calls
#   PROBE_CHECK_BODY  "1": print the request bodies (ask_gpt_full's code path) for PROBE_BODY_MODELS
#                     (default kimik3,terra,gem38flash) on a 3-record ICL conversation; no API calls
#
# Output (never overwritten: _2, _3, ... suffix, same for both files)
#   results/meta/effort_probe_<yyyy-mm-dd>.jsonl  one record per call, appended as it arrives
#   results/meta/effort_probe_<yyyy-mm-dd>.json   at the end: {"meta", "calls", "summary": {tag: {levels_supported,
#                                                 median_dose_by_level, monotone_verdict, partial,
#                                                 suggested_reasoning_levels, cost_ratio, ...}}}
#   plus a printed model x level table of emitted reasoning tokens ([400] = rejected) with the ratio
#   billed cost / cost at registry prices ("[price x2.0?]" when > 1.6: flex or the pinned price not applied).
#   The suggested reasoning_levels are NOT written into the registry -- enter them by hand.
#
# ALWAYS dry-run first:   PROBE_DRYRUN=1 julia src/julia/Probe_Models.jl
#
include(joinpath(@__DIR__, "Common.jl"))
include(joinpath(@__DIR__, "Prompts_ZS.jl"))
include(joinpath(@__DIR__, "Prompts_ICL.jl"))

const LEVEL_ORDER = KNOWN_LEVELS                 # none, minimal, low, medium, high
const SPENT = Ref(0.0)
const INFLIGHT = Ref(0.0)      # worst-case USD reserved by calls in flight
const WORST_FACTOR = 2.0
const FLAT_RATIO = 1.5         # max dose < 1.5 x min dose over a set of levels -> "the same amount"
const PRICE_FLAG = 1.6         # billed / expected cost above this -> "[price x2.0?]"

# ------------------------------------------------------------------ one call
"Cost at the registry prices (USD) for the token counts of one answered call."
function expected_cost(md, prompt, cached, completion)
    p_cr = md["price_cache_read"] === nothing ? md["price_in"] : md["price_cache_read"]
    return ((prompt - cached) * md["price_in"] + cached * p_cr + completion * md["price_out"]) / 1e6
end

function base_record(tag, md, level, rep, max_tokens, truth)
    return Dict{String,Any}("tag" => tag, "endpoint" => md["endpoint"], "level" => level, "repeat" => rep,
                            "providers" => md["providers"], "service_tier" => md["service_tier"],
                            "allow_fallbacks" => md["allow_fallbacks"], "max_tokens" => max_tokens,
                            "truth" => truth, "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
                            "status" => nothing, "http_status" => nothing, "error" => nothing,
                            "reasoning_tokens" => nothing, "completion_tokens" => nothing, "prompt_tokens" => nothing,
                            "cached_tokens" => nothing, "cost" => 0.0, "expected_cost" => nothing, "cost_ratio" => nothing,
                            "finish" => nothing, "provider" => nothing, "model_served" => nothing,
                            "latency_s" => nothing, "text" => nothing, "value" => nothing)
end

function probe_call(tag, md, level, rep, conv, max_tokens, retries, truth)
    rec = base_record(tag, md, level, rep, max_tokens, truth)
    t0 = time()
    try
        r = ask_gpt_full(llm_access(tag), conv; max_tokens=max_tokens, reasoning_effort=level, retries=retries)
        txt = strip(r["text"])
        v = parse_number(txt)
        exp = expected_cost(md, r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"])
        merge!(rec, Dict("status" => "ok", "http_status" => 200,
                         "reasoning_tokens" => r["reasoning_tokens"], "completion_tokens" => r["completion_tokens"],
                         "prompt_tokens" => r["prompt_tokens"], "cached_tokens" => r["cached_tokens"],
                         "cost" => r["cost"], "expected_cost" => exp,
                         "cost_ratio" => exp > 0 ? round(r["cost"] / exp, digits=3) : nothing,
                         "finish" => r["finish"], "provider" => r["provider"],
                         "model_served" => r["model_served"], "latency_s" => r["latency_s"],
                         "text" => first(txt, 200),
                         "value" => (v isa Number && isfinite(v)) ? Float64(v) : nothing))
    catch e
        if e isa UnsupportedRequest
            m = match(r"^HTTP (\d+)", e.msg)
            status = m === nothing ? 400 : parse(Int, m.captures[1])
            # 400 = this request (level) rejected; 404 = no endpoint for the pinned routing at all
            kind = status == 404 ? "no_endpoint" : "unsupported"
        else
            status = e isa HTTP.ExceptionRequest.StatusError ? Int(e.status) : nothing
            kind = "error"
        end
        merge!(rec, Dict("status" => kind, "http_status" => status,
                         "error" => first(e isa UnsupportedRequest ? e.msg : sprint(showerror, e), 600),
                         "latency_s" => round(time() - t0, digits=2)))
    end
    return rec
end

# ------------------------------------------------------------------ verdicts
_med(v) = isempty(v) ? nothing : median(v)
_flat(vals) = length(vals) >= 2 && (maximum(vals) == 0 || maximum(vals) < FLAT_RATIO * max(minimum(vals), 1))

"Requested levels in ladder order; unknown names after the known ones, in request order."
ordered_levels(levels) = vcat([l for l in LEVEL_ORDER if l in levels], [l for l in levels if !(l in LEVEL_ORDER)])

"""
Per model: supported levels, median dose per level (over repeats), verdict, suggested
`reasoning_levels`. Dose = emitted reasoning_tokens; if an endpoint never reports any,
completion_tokens are used instead (flagged in `dose_measure`). Verdicts:
  flat         the levels other than `none` emit the same amount (max < 1.5 x min), and so does none
  on/off only  none emits (almost) nothing, all other levels the same amount (the 45 failure mode)
  monotone     dose rises along none|minimal < low < high and never falls along the answered levels
  ... "partial" is appended when a requested level was not answered (budget, error, 404).
"""
function summarise(calls, tags, levels)
    out = Dict{String,Any}()
    lv_run = ordered_levels(levels)
    for t in tags
        cs = [c for c in calls if c["tag"] == t]
        ok = [c for c in cs if c["status"] == "ok"]
        use_completion = !isempty(ok) && all(c -> c["reasoning_tokens"] == 0, ok)
        dose(c) = use_completion ? c["completion_tokens"] : c["reasoning_tokens"]
        supported = [l for l in lv_run if any(c -> c["level"] == l && c["status"] == "ok", cs)]
        rejected = [l for l in lv_run if !(l in supported) && any(c -> c["level"] == l && c["status"] == "unsupported", cs)]
        untested = [l for l in lv_run if !(l in supported) && !(l in rejected)]
        med = Dict(l => _med([dose(c) for c in ok if c["level"] == l]) for l in supported)
        known = [l for l in supported if l in LEVEL_ORDER]
        others = [l for l in known if l != "none"]
        bottom = "none" in known ? "none" : "minimal"      # endpoints rejecting none: minimal is the floor
        core = [l for l in (bottom, "low", "high") if l in known]
        core_inc = length(core) >= 2 && all(med[core[i]] < med[core[i + 1]] for i in 1:length(core) - 1)
        full_nondec = all(med[known[i]] <= med[known[i + 1]] for i in 1:length(known) - 1)
        others_flat = _flat([med[l] for l in others])
        # "off" levels: a leading run of known levels (none, then minimal) whose dose is (almost) zero
        # compared with the remaining levels; the 45 failure mode is off-levels + a flat remainder.
        off = String[]
        for l in known
            rest = [med[m] for m in known if !(m in off) && m != l]
            (isempty(rest) || !(med[l] * FLAT_RATIO < max(minimum(rest), 1))) && break
            push!(off, l)
        end
        on = [l for l in known if !(l in off)]
        none_off = !isempty(off) && _flat([med[l] for l in on])
        verdict = length(core) < 2 ? "insufficient (fewer than two of none|minimal / low / high answered)" :
                  none_off ? "on/off only ($(join(off, "/")) switch reasoning off; all other levels emit the same amount)" :
                  others_flat || (length(others) < 2 && _flat([med[l] for l in known])) ?
                      "flat (effort ignored: max dose < $(FLAT_RATIO) x min dose)" :
                  core_inc && full_nondec ? "monotone" :
                  core_inc ? "monotone on $(join(core, "<")), inversions among the other levels" : "non-monotone"
        partial = !isempty(untested)
        partial && (verdict *= "  [partial: $(join(untested, ", ")) not tested]")
        # suggestion: greedy strictly increasing subsequence; flat -> low only; on/off -> none + low;
        # `low` (the main experiment's level) is always kept when its call succeeded
        sugg = String[]
        if startswith(verdict, "flat")
        elseif startswith(verdict, "on/off")
            push!(sugg, off[1])
        else
            for l in known
                (isempty(sugg) || med[l] > med[sugg[end]]) && push!(sugg, l)
            end
        end
        "low" in supported && !("low" in sugg) && push!(sugg, "low")
        sugg = [l for l in LEVEL_ORDER if l in sugg]
        !("low" in supported) && (verdict *= "  [rerun low: no answer at the main level]")
        ratios = [c["cost_ratio"] for c in ok if c["cost_ratio"] !== nothing]
        out[t] = Dict("levels_requested" => lv_run, "levels_supported" => supported, "levels_rejected" => rejected,
                      "levels_untested" => untested, "partial" => partial,
                      "dose_measure" => use_completion ? "completion_tokens (no reasoning_tokens reported)" : "reasoning_tokens",
                      "median_dose_by_level" => med, "monotone_chain" => join(core, "<"), "monotone_chain_increasing" => core_inc,
                      "nondecreasing_all_supported" => full_nondec, "monotone_verdict" => verdict,
                      "suggested_reasoning_levels" => sugg,
                      "cost_ratio_median" => isempty(ratios) ? nothing : median(ratios),
                      "cost_ratio_max" => isempty(ratios) ? nothing : maximum(ratios),
                      "price_flag" => !isempty(ratios) && maximum(ratios) > PRICE_FLAG,
                      "providers_served" => unique([c["provider"] for c in ok]),
                      "n_skipped_budget" => count(c -> c["status"] == "skipped_budget", cs),
                      "cost" => sum((c["cost"] for c in cs); init=0.0))
    end
    return out
end

function cell_text(c)
    c["status"] == "ok" && return string(c["reasoning_tokens"])
    c["status"] == "unsupported" && return "[400]"
    c["status"] == "no_endpoint" && return "[404]"
    c["status"] == "skipped_budget" && return "skip"
    return c["http_status"] === nothing ? "err" : "e$(c["http_status"])"
end

function print_table(calls, tags, levels, summary)
    lv = ordered_levels(levels)
    @printf("\n%-11s %s  %s\n", "reasoning", join([lpad(l, 12) for l in lv], ""), "  cost/list")
    for t in tags
        cells = String[]
        for l in lv
            cs = sort([c for c in calls if c["tag"] == t && c["level"] == l], by=c -> c["repeat"])
            push!(cells, lpad(isempty(cs) ? "-" : join(cell_text.(cs), "/"), 12))
        end
        sm = summary[t]
        r = sm["cost_ratio_median"] === nothing ? "     -" : @sprintf("%6.2f", sm["cost_ratio_median"])
        @printf("%-11s %s  %s%s%s\n", t, join(cells, ""), r, sm["price_flag"] ? "  [price x2.0?]" : "",
                sm["partial"] ? "  partial" : "")
    end
    println("(emitted reasoning tokens per call, repeats separated by /; [400] = level rejected by the endpoint;")
    println(" [404] = no endpoint for the pinned routing; skip = budget; eNNN / err = other failure; - = not run;")
    println(" cost/list = median billed cost / cost at registry prices; [price x2.0?] = some call > $(PRICE_FLAG)x,")
    println(" i.e. flex or the pinned provider's price was probably not applied)")
end

"effort_probe_<date>[_k] base path such that neither the .json nor the .jsonl exists yet."
function out_base()
    base0 = joinpath(RESULTS_DIR, "meta", "effort_probe_$(Dates.format(today(), "yyyy-mm-dd"))")
    base = base0; i = 2
    while isfile(base * ".json") || isfile(base * ".jsonl")
        base = base0 * "_$(i)"; i += 1
    end
    return base
end

# ------------------------------------------------------------------ request-body check
function check_body()
    tags = envlist("PROBE_BODY_MODELS", "kimik3,terra,gem38flash")
    key = "lipophilicity"
    test, train = cell_data(key, 30)
    conv = build_conversation(key, train[1:3], string(test[1, :smiles]))
    for t in tags
        acc = llm_access(t)
        body = build_request_body(acc, conv; max_tokens=32000, reasoning_effort="low")   # == ask_gpt_full's call
        println("=" ^ 78, "\n", t, "  (", acc.model, ")  allow_fallbacks registered = ", fallbacks_allowed(acc))
        JSON.print(stdout, JSON.parse(body), 2)
        println()
    end
    println("CHECK BODY: no API calls made.")
end

# ------------------------------------------------------------------ main
function main()
    models = load_models()
    if envflag("PROBE_CHECK_BODY")
        check_body(); return
    end
    tags = resolve_tags(envget("PROBE_MODELS", "all"))
    levels = warn_unknown_levels(envlist("PROBE_LEVELS", join(LEVEL_ORDER, ",")))
    repeats = parse(Int, envget("PROBE_REPEATS", "1"))
    max_tokens = parse(Int, envget("PROBE_MAX_TOKENS", "32000"))
    key = envget("PROBE_DATASET", "lipophilicity")
    rule = envget("PROBE_RULE", "v2test150")
    retries = parse(Int, envget("PROBE_RETRIES", "1"))
    budget = parse(Float64, envget("PROBE_BUDGET", "3.0"))
    for t in tags
        haskey(models, t) || error("unknown tag '$t' (src/registry/models_v2.json)")
    end

    row = load_test_set(key, rule)[1, :]
    s, u = zs_prompts(load_datasets45()[key], strip(string(row.smiles)))
    conv = Tuple[("system", s), ("user", u)]
    p_est = estimate_tokens(conv)
    worst(t) = WORST_FACTOR * (p_est * models[t]["price_in"] + max_tokens * models[t]["price_out"]) / 1e6
    n_calls = length(tags) * length(levels) * repeats

    println("=" ^ 78)
    println("effort-dosability probe   $(length(tags)) models x $(length(levels)) levels x $repeats repeat(s) = $n_calls calls",
            repeats < 2 ? "   (PROBE_REPEATS=2 recommended for the panel)" : "")
    println("levels     ", join(levels, ", "), "   max_tokens $max_tokens   sampling: provider default")
    println("molecule   $key/$rule #1  $(row.mol_id)  truth $(row.value)   (~$p_est prompt tokens)")
    println("budget     \$$budget hard ceiling (spent + reserved worst case of in-flight calls + next call's worst case;")
    println("           worst case = $(WORST_FACTOR) x (prompt + max_tokens at registry price))")
    println("-" ^ 78)
    @printf("%-11s %-30s %-28s %-5s %-6s %9s\n", "tag", "endpoint", "providers", "tier", "fallb", "worst/call")
    for t in tags
        md = models[t]
        @printf("%-11s %-30s %-28s %-5s %-6s %9s%s\n", t, md["endpoint"], join(md["providers"], ","),
                md["service_tier"], md["allow_fallbacks"], @sprintf("\$%.3f", worst(t)),
                worst(t) > budget / 2 ? "  <- skipped once spent > \$$(round(budget - worst(t), digits=2))" : "")
    end
    @printf("sum of worst cases: \$%.2f -- the ceiling holds regardless (reservation); expensive calls are skipped near it\n",
            sum(worst(t) for t in tags) * length(levels) * repeats)
    println("=" ^ 78)

    if envflag("PROBE_DRYRUN")
        println("\n--- SYSTEM ---\n$s\n--- USER ---\n$u\n\nDRY RUN: no API calls made.")
        return
    end

    base = out_base()
    mkpath(dirname(base))
    jsonl = base * ".jsonl"
    calls = Vector{Dict{String,Any}}()
    lk = ReentrantLock()
    t0 = time()
    @sync for t in tags
        @async for l in levels, rep in 1:repeats
            w = worst(t)
            decision = :wait
            while decision == :wait
                decision = lock(lk) do
                    SPENT[] + w > budget && return :skip
                    SPENT[] + INFLIGHT[] + w > budget && return :wait
                    INFLIGHT[] += w
                    return :go
                end
                decision == :wait && sleep(2)
            end
            if decision == :skip
                lock(lk) do
                    rec = base_record(t, models[t], l, rep, max_tokens, Float64(row.value))
                    rec["status"] = "skipped_budget"
                    rec["error"] = @sprintf("spent \$%.3f + worst case \$%.3f > budget \$%.2f", SPENT[], w, budget)
                    push!(calls, rec)
                    jsonl_append(jsonl, rec)
                    println("  [budget] $t $l #$rep skipped: $(rec["error"])")
                end
                continue
            end
            rec = try
                probe_call(t, models[t], l, rep, conv, max_tokens, retries, Float64(row.value))
            catch e        # never let a task die holding its reservation (the others would wait forever)
                r = base_record(t, models[t], l, rep, max_tokens, Float64(row.value))
                r["status"] = "error"; r["error"] = first(sprint(showerror, e), 600)
                r
            end
            lock(lk) do
                push!(calls, rec)
                jsonl_append(jsonl, rec)
                INFLIGHT[] -= w
                SPENT[] += rec["cost"]
                if rec["status"] == "ok"
                    @printf("  %-10s %-8s #%d reasoning=%6d completion=%6d prompt=%5d cached=%5d \$%.5f (x%s list) [%s] value=%s via %s %.1fs\n",
                            t, l, rep, rec["reasoning_tokens"], rec["completion_tokens"], rec["prompt_tokens"],
                            rec["cached_tokens"], rec["cost"], rec["cost_ratio"], rec["finish"], rec["value"],
                            rec["provider"], rec["latency_s"])
                else
                    @printf("  %-10s %-8s #%d %s (HTTP %s): %s\n", t, l, rep, uppercase(rec["status"]),
                            rec["http_status"], first(replace(rec["error"], "\n" => " "), 200))
                end
            end
        end
    end

    summary = summarise(calls, tags, levels)
    print_table(calls, tags, levels, summary)
    println("\nverdicts and suggested `reasoning_levels` (median over repeats; NOT written to the registry):")
    for t in tags
        sm = summary[t]
        @printf("  %-11s supported %-38s %s\n", t, string(sm["levels_supported"]), sm["monotone_verdict"])
        println("              suggested \"reasoning_levels\": ", JSON.json(sm["suggested_reasoning_levels"]),
                sm["dose_measure"] == "reasoning_tokens" ? "" : "   [dose = completion tokens]")
    end
    meta = Dict("date" => string(today()), "registry" => relpath(MODELS_V2_FILE, ROOT), "tags" => tags,
                "levels" => levels, "repeats" => repeats, "max_tokens" => max_tokens, "dataset" => key,
                "rule" => rule, "mol_id" => row.mol_id, "truth" => Float64(row.value), "smiles" => string(row.smiles),
                "prompt" => Dict("system" => s, "user" => u), "budget" => budget, "worst_factor" => WORST_FACTOR,
                "spent" => SPENT[], "wall_s" => round(time() - t0, digits=1), "calls_jsonl" => basename(jsonl))
    path = base * ".json"
    write(path, JSON.json(Dict("meta" => meta, "calls" => calls, "summary" => summary), 1))
    @printf("\nwritten %s (+ .jsonl)   spent \$%.4f of \$%.2f (library total now \$%.2f)\n", path, SPENT[], budget, current_usage())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
