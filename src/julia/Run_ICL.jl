#
# v2 main experiment: LLM as in-context regressor with self-reported uncertainty
# (io protocol, blinding level 1, separate 150-molecule test set, nested training blocks).
#
# Cells: dataset x model x in-context size x reasoning effort. Every test molecule is queried
# ICL_REPEATS times (independent calls) -> UC1 (self-reported sigma) and UC3 (spread over
# repeats). Main experiment: effort "low", all ladder sizes. Reasoning ladder: several
# efforts at the smallest and largest size only (ICL_REASONING=none,minimal,low,medium,high
# ICL_SIZES=min,max). Sampling at the provider default (temperature = top_p = 1 -> omitted).
#
# Environment
#   ICL_DATASETS   comma list of dataset keys          (default lipophilicity,bace,antiviral,esol)
#   ICL_MODELS     comma list of registry tags, `all` (every enabled entry) or `panel`
#                                                      (default panel = in_panel entries of src/registry/models_v2.json)
#   ICL_SIZES      all | min,max | comma list          (default all)
#   ICL_REASONING  effort level(s), comma list         (default low)
#   ICL_REPEATS    calls per test molecule and cell    (default 1, decision 2026-10-08: R = 1 first, extend to 3
#                  later). Repeats are indexed 0..R-1 and the restart skips stored (mol_id, repeat) pairs,
#                  so a later run with ICL_REPEATS=3 on the same cells only adds repeats 1 and 2.
#   ICL_MAX_TOKENS completion cap                      (default 32000)
#   ICL_THREADS    concurrent requests                 (default 8)
#   ICL_BUDGET     hard ceiling in USD for this run    (default 5.0; also for ICL_PROBE)
#   ICL_DRYRUN     "1": print plan + example prompt, no API calls
#   ICL_PROBE      "1": two identical calls per model (sequential within a model, models concurrent) on the
#                  first test molecule of the first dataset at the first size of ICL_SIZES (use ICL_SIZES=max),
#                  effort = first ICL_REASONING level. The second call shows the prompt-cache hit. Budget:
#                  a call starts only if spent + reserved worst case of the calls in flight + its own worst
#                  case <= ICL_BUDGET (worst case = 2 x (3 x chars/4-estimated prompt at max(price_in, price_cache_write)
#                  + ICL_MAX_TOKENS at price_out)); a call that cannot start even when idle is recorded as
#                  "skipped_budget". ICL_PROBE_RETRIES (default 1) retries per call on errors other than 400/404.
#                  Output: results/meta/icl_probe_<yyyy-mm-dd>[_k].jsonl (one record per call, as it arrives:
#                  tag, call, prompt/cached/cache_write/completion/reasoning tokens, cost, expected_cost at
#                  registry prices, cost_ratio, finish, provider, model_served, latency_s, parse_ok,
#                  prediction, uncertainty, text, ...) + .json (meta + calls) at the end. Never overwritten.
#                  ICL_PROBE_PLAN=1 (with ICL_PROBE=1): print the probe plan and worst cases only, no API calls.
#   ICL_DUMP       path: write the example conversation as JSON (prompt-parity check)
#   ICL_DRAW       context draw (default 1 = the main split data/splits/<ds>_v2.json); 2, 3 = the
#                  additional nested context draws data/splits/<ds>_v2_draw<d>.json (same test set and
#                  training pool, independently permuted pool order; src/make_splits.py --draws 3).
#   ICL_PROBE_PLAN "1" without ICL_PROBE: print the run plan (cells, pending calls per model after the
#                  resume check, first training molecules of each dataset) and exit -- no API calls.
#
# Output: results/icl/<dataset>__<tag>__N<size>__<effort>.jsonl (draw 1) or
#         results/icl/<dataset>__<tag>__N<size>__<effort>__d<draw>.jsonl (draw >= 2), one JSON record per call
#         (dataset, model, endpoint, size, reasoning, draw, repeat, mol_id, truth, text, prediction,
#         uncertainty, parse_ok, prompt_tokens, cached_tokens, completion_tokens,
#         reasoning_tokens, cost, finish, provider, model_served, latency_s, timestamp).
#         Appended as answers arrive; a restart skips (mol_id, repeat) pairs already stored in the
#         same file. Records written before 2026-10-13 have no "draw" field (= draw 1).
# Analysis: python src/analyze_icl.py
#
# ALWAYS dry-run first:   ICL_DRYRUN=1 julia src/julia/Run_ICL.jl
#
include(joinpath(@__DIR__, "Common.jl"))
include(joinpath(@__DIR__, "Prompts_ICL.jl"))

const OUT_DIR = joinpath(RESULTS_DIR, "icl")
const SPENT = Ref(0.0)

"Result file of a cell: draw 1 keeps the original name, draw d >= 2 gets the suffix __d<d>."
cell_path(key, tag, size, effort; draw::Integer=1) =
    joinpath(OUT_DIR, "$(key)__$(tag)__N$(size)__$(effort)" * (draw == 1 ? "" : "__d$(draw)") * ".jsonl")

"Calls still missing in a cell file for `repeats` repeats of the 150 test molecules."
function pending_calls(key, tag, size, effort; repeats::Int, draw::Integer=1)
    done = existing_pairs(cell_path(key, tag, size, effort; draw=draw))
    ids = [string(m) for m in load_split(key; draw=draw)["test_ids"]]
    return count(!((m, r) in done) for m in ids for r in 0:repeats-1)
end

function run_cell(key::AbstractString, tag::AbstractString, size::Int, effort::AbstractString;
                  repeats::Int, max_tokens::Int, threads::Int, budget::Float64, models, draw::Int=1)
    test, train = cell_data(key, size; draw=draw)
    path = cell_path(key, tag, size, effort; draw=draw)
    dlab = draw == 1 ? "" : "/d$draw"
    done = existing_pairs(path)
    jobs = [(row, rep) for row in eachrow(test) for rep in 0:repeats-1 if !((row.mol_id, rep) in done)]
    if isempty(jobs)
        println("  [skip] $key/$tag/N$size/$effort$dlab: complete ($(length(done)) calls)")
        return
    end
    println("  [run ] $key/$tag/N$size/$effort$dlab: $(length(jobs)) calls ($(length(done)) already on disk)")
    llm = llm_access(tag; models=models)
    convs = Vector{Tuple}[build_conversation(key, train, string(row.smiles)) for (row, _) in jobs]
    t0 = time(); before = SPENT[]; n_ok = Ref(0); n_fail = Ref(0); stop = Ref(false)
    function on_result(idx, res)
        if res === nothing
            n_fail[] += 1
            return
        end
        row, rep = jobs[idx]
        rec = Dict{String,Any}("dataset" => key, "model" => tag, "endpoint" => models[tag]["endpoint"],
                               "size" => size, "reasoning" => effort, "draw" => draw, "repeat" => rep,
                               "mol_id" => row.mol_id, "truth" => Float64(row.value),
                               "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
        merge!(rec, res, parse_response(res["text"]))
        jsonl_append(path, rec)
        SPENT[] += res["cost"]; n_ok[] += 1
        SPENT[] >= budget && (stop[] = true)
        n_ok[] % 50 == 0 && @printf("       .. %s/%s/N%d %d/%d  \$%.3f\n", key, tag, size, n_ok[], length(jobs), SPENT[] - before)
    end
    unsupported = ask_threaded_full(llm, convs, on_result; num_threads=threads, stop=stop,
                                    max_tokens=max_tokens, reasoning_effort=effort)
    unsupported === nothing || println("  [--] $key/$tag/$effort: endpoint rejected the request: $unsupported")
    @printf("  [ok ] %-14s %-10s N%-5d %-8s d%d %d/%d stored, %d failed  \$%.3f  %.0f s\n",
            key, tag, size, effort, draw, n_ok[], length(jobs), n_fail[], SPENT[] - before, time() - t0)
end

# ------------------------------------------------------------------ ICL probe (ICL_PROBE=1)
const PROBE_WORST_FACTOR = 2.0     # flex not applied, peak-hour prices, regional 1.1x, a billed retry
const PROBE_PROMPT_FACTOR = 3.0    # estimate_tokens (chars / 4) undercounts SMILES: N=1000 lipophilicity ~21k by chars/4, ~47k tiktoken, ~1.3x more on Claude

"Cost at the registry prices (USD) for the token counts of one answered call (cache read / write split)."
function expected_cost(md, prompt, cached, cache_write, completion)
    p_in = md["price_in"]
    p_cr = md["price_cache_read"] === nothing ? p_in : md["price_cache_read"]
    p_cw = md["price_cache_write"] === nothing ? p_in : md["price_cache_write"]
    plain = max(prompt - cached - cache_write, 0)
    return (plain * p_in + cached * p_cr + cache_write * p_cw + completion * md["price_out"]) / 1e6
end

"icl_probe_<date>[_k] base path such that neither the .json nor the .jsonl exists yet."
function probe_out_base()
    base0 = joinpath(RESULTS_DIR, "meta", "icl_probe_$(Dates.format(today(), "yyyy-mm-dd"))")
    base = base0; i = 2
    while isfile(base * ".json") || isfile(base * ".jsonl")
        base = base0 * "_$(i)"; i += 1
    end
    return base
end

function run_probe(key, tags, sizes_spec, effort; max_tokens, budget, models, draw::Int=1)
    n = resolve_sizes(key, sizes_spec)[1]
    test, train = cell_data(key, n; draw=draw)
    row = test[1, :]
    conv = build_conversation(key, train, string(row.smiles))
    p_est = estimate_tokens(conv)
    retries = parse(Int, envget("ICL_PROBE_RETRIES", "1"))
    function worst(t)
        md = models[t]
        p_hi = max(md["price_in"], something(md["price_cache_write"], 0.0))
        return PROBE_WORST_FACTOR * (PROBE_PROMPT_FACTOR * p_est * p_hi + max_tokens * md["price_out"]) / 1e6
    end
    println("ICL PROBE  $key N=$n  molecule $(row.mol_id) (truth $(row.value))  effort $effort  max_tokens $max_tokens")
    println("           ~$p_est prompt tokens (chars/4)   2 calls x $(length(tags)) models   budget \$$budget hard ceiling")
    for t in tags
        @printf("  %-12s worst case per call \$%.3f\n", t, worst(t))
    end
    @printf("  sum of worst cases (one call per model in flight): \$%.2f\n", sum(worst(t) for t in tags))
    if envflag("ICL_PROBE_PLAN")
        println("ICL_PROBE_PLAN: no API calls made.")
        return
    end
    base = probe_out_base()
    mkpath(dirname(base))
    jsonl = base * ".jsonl"
    calls = Vector{Dict{String,Any}}()
    lk = ReentrantLock()
    inflight = Ref(0.0)
    lib0 = current_usage()
    t0 = time()
    function record(t, call)
        md = models[t]
        return Dict{String,Any}("tag" => t, "endpoint" => md["endpoint"], "providers" => md["providers"],
            "service_tier" => md["service_tier"], "allow_fallbacks" => md["allow_fallbacks"], "dataset" => key,
            "size" => n, "reasoning" => effort, "call" => call, "mol_id" => row.mol_id, "truth" => Float64(row.value),
            "max_tokens" => max_tokens, "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            "status" => nothing, "error" => nothing, "prompt_tokens" => nothing, "cached_tokens" => nothing,
            "cache_write_tokens" => nothing, "completion_tokens" => nothing, "reasoning_tokens" => nothing,
            "cost" => 0.0, "expected_cost" => nothing, "cost_ratio" => nothing, "finish" => nothing,
            "provider" => nothing, "model_served" => nothing, "latency_s" => nothing, "parse_ok" => false,
            "prediction" => nothing, "uncertainty" => nothing, "text" => nothing)
    end
    @sync for t in tags
        @async begin
            md = models[t]
            w = worst(t)
            for call in 1:2
                decision = :wait
                while decision == :wait
                    decision = lock(lk) do
                        SPENT[] + w > budget && return :skip
                        SPENT[] + inflight[] + w > budget && return :wait
                        inflight[] += w
                        return :go
                    end
                    decision == :wait && sleep(2)
                end
                rec = record(t, call)
                if decision == :skip
                    rec["status"] = "skipped_budget"
                    rec["error"] = @sprintf("spent \$%.3f + worst case \$%.3f > budget \$%.2f", SPENT[], w, budget)
                    lock(lk) do
                        push!(calls, rec); jsonl_append(jsonl, rec)
                        println("  [budget] $t call$call skipped: $(rec["error"])")
                    end
                    break
                end
                t1 = time()
                try
                    r = ask_gpt_full(llm_access(t; models=models), conv; max_tokens=max_tokens,
                                     reasoning_effort=effort, retries=retries)
                    p = parse_response(r["text"])
                    exp = expected_cost(md, r["prompt_tokens"], r["cached_tokens"], r["cache_write_tokens"], r["completion_tokens"])
                    merge!(rec, r, p)
                    rec["status"] = "ok"
                    rec["expected_cost"] = exp
                    rec["cost_ratio"] = exp > 0 ? round(r["cost"] / exp, digits=3) : nothing
                catch e
                    rec["status"] = e isa UnsupportedRequest ? (startswith(e.msg, "HTTP 404") ? "no_endpoint" : "unsupported") : "error"
                    rec["error"] = first(e isa UnsupportedRequest ? e.msg : sprint(showerror, e), 600)
                    rec["latency_s"] = round(time() - t1, digits=2)
                end
                lock(lk) do
                    inflight[] -= w
                    SPENT[] += rec["cost"]
                    push!(calls, rec); jsonl_append(jsonl, rec)
                    if rec["status"] == "ok"
                        @printf("  %-12s N%d call%d: prompt %d (cached %d, write %d)  completion %d (reasoning %d)  \$%.4f (x%s list)  parse_ok=%s pred=%s unc=%s truth=%s [%s] via %s (%s) %.1fs\n",
                                t, n, call, rec["prompt_tokens"], rec["cached_tokens"], rec["cache_write_tokens"],
                                rec["completion_tokens"], rec["reasoning_tokens"], rec["cost"], rec["cost_ratio"],
                                rec["parse_ok"], rec["prediction"], rec["uncertainty"], row.value, rec["finish"],
                                rec["provider"], rec["model_served"], rec["latency_s"])
                    else
                        println("  $t call$call $(uppercase(rec["status"])): ", first(replace(rec["error"], "\n" => " "), 300))
                    end
                end
                rec["status"] in ("no_endpoint", "unsupported") && break    # call 2 would fail the same way
                SPENT[] >= budget && break                                  # ceiling reached: no further call
            end
        end
    end
    meta = Dict("date" => string(today()), "registry" => relpath(MODELS_V2_FILE, ROOT), "dataset" => key, "size" => n,
                "reasoning" => effort, "max_tokens" => max_tokens, "mol_id" => row.mol_id, "truth" => Float64(row.value),
                "smiles" => string(row.smiles), "prompt_est_tokens" => p_est, "tags" => tags, "budget" => budget,
                "worst_factor" => PROBE_WORST_FACTOR, "spent" => SPENT[], "library_total_before" => lib0,
                "library_total_after" => current_usage(), "wall_s" => round(time() - t0, digits=1),
                "calls_jsonl" => basename(jsonl))
    write(base * ".json", JSON.json(Dict("meta" => meta, "calls" => calls), 1))
    @printf("probe spent \$%.4f of \$%.2f   written %s (+ .json)   library total now \$%.2f\n",
            SPENT[], budget, jsonl, current_usage())
end

function main()
    keys = envlist("ICL_DATASETS", join(DATASET_KEYS, ","))
    tags = resolve_tags(envget("ICL_MODELS", "panel"))
    sizes_spec = envget("ICL_SIZES", "all")
    efforts = warn_unknown_levels(envlist("ICL_REASONING", "low"))
    repeats = parse(Int, envget("ICL_REPEATS", "1"))
    max_tokens = parse(Int, envget("ICL_MAX_TOKENS", "32000"))
    threads = parse(Int, envget("ICL_THREADS", "8"))
    budget = parse(Float64, envget("ICL_BUDGET", "5.0"))
    draw = parse(Int, envget("ICL_DRAW", "1"))
    draw >= 1 || error("ICL_DRAW must be >= 1")
    for k in keys
        isfile(split_path(k; draw=draw)) || error("no split for $k draw $draw: $(split_path(k; draw=draw)) (python src/make_splits.py --draws 3)")
    end
    models = load_models()

    plan = [(k, t, n, e) for k in keys for t in tags for n in resolve_sizes(k, sizes_spec) for e in efforts]
    println("=" ^ 78)
    println("datasets  ", join(keys, ", "))
    println("models    ", join(["$t ($(models[t]["endpoint"]))" for t in tags], ", "))
    println("sizes     ", sizes_spec, " -> ", join(["$k: $(resolve_sizes(k, sizes_spec))" for k in keys], "; "))
    println("reasoning ", join(efforts, ", "), "   repeats $repeats   max_tokens $max_tokens   sampling: provider default")
    println("draw      $draw  (", draw == 1 ? "main split data/splits/<ds>_v2.json; files <ds>__<tag>__N<size>__<effort>.jsonl" :
            "data/splits/<ds>_v2_draw$(draw).json; files <ds>__<tag>__N<size>__<effort>__d$(draw).jsonl", ")")
    for k in keys
        sp = load_split(k; draw=draw)
        println("  $k first training molecules (draw $draw, seed_draw $(get(sp, "seed_draw", sp["seed"]))): ",
                join(first(sp["train_order_ids"], 5), ", "))
    end
    println("$(length(plan)) cells, up to $(150 * repeats * length(plan)) calls, budget ceiling \$$budget")
    println("=" ^ 78)

    if envflag("ICL_PROBE_PLAN") && !envflag("ICL_PROBE")
        # plan only: pending calls per model after the resume check (no API calls, nothing written)
        pend = Dict(t => 0 for t in tags)
        for (k, t, n, e) in plan
            pend[t] += pending_calls(k, t, n, e; repeats=repeats, draw=draw)
        end
        for t in tags
            @printf("  %-12s %d cells  %6d pending calls\n", t, count(p -> p[2] == t, plan), pend[t])
        end
        println("total pending calls: $(sum(values(pend))) of $(150 * repeats * length(plan))")
        println("ICL_PROBE_PLAN: run plan only, no API calls made.")
        return
    end

    if envflag("ICL_DRYRUN") || haskey(ENV, "ICL_DUMP")
        k, t, n, e = plan[1]
        test, train = cell_data(k, n; draw=draw)
        conv = build_conversation(k, train, string(test[1, :smiles]))
        if haskey(ENV, "ICL_DUMP")
            open(ENV["ICL_DUMP"], "w") do io
                JSON.print(io, [Dict("role" => c[1], "content" => c[2], "cache" => length(c) > 2 ? c[3] : nothing) for c in conv])
            end
            println("conversation written to $(ENV["ICL_DUMP"])")
        end
        blk = split(conv[2][2], "\n")
        println("\n--- SYSTEM ---\n", conv[1][2])
        println("--- USER 1 (training block, ephemeral; $(length(train)) records, ~$(estimate_tokens(Tuple[conv[2]])) tokens) ---")
        println(join(blk[1:min(9, end)], "\n"), "\n...\n", join(blk[max(1, end - 3):end], "\n"))
        println("--- USER 2 (task) ---\n", conv[3][2])
        println("first cell: $k/$t/N$n/$e draw $draw -> $(relpath(cell_path(k, t, n, e; draw=draw), ROOT))")
        println("DRY RUN: no API calls made.")
        return
    end

    mkpath(OUT_DIR)
    if envflag("ICL_PROBE")
        run_probe(keys[1], tags, sizes_spec, efforts[1]; max_tokens=max_tokens, budget=budget, models=models, draw=draw)
        return
    end

    for (k, t, n, e) in plan
        if SPENT[] >= budget
            @printf("BUDGET \$%.2f reached (\$%.2f) -- stopping.\n", budget, SPENT[]); break
        end
        run_cell(k, t, n, e; repeats=repeats, max_tokens=max_tokens, threads=threads, budget=budget, models=models, draw=draw)
    end
    @printf("\nDONE. spent \$%.2f of \$%.2f (library total now \$%.2f)\n", SPENT[], budget, current_usage())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
