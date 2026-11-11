#
# Zero-shot memorization test on the v2 test sets -- the memorization paper's probe
# (45/src/Run_ZeroShot.jl / run_budget_sweep.py prompt with the "3 significant figures"
# sentence), reasoning controlled by EFFORT LEVEL (decision 2026-09-08), token counts per call.
#
# Environment
#   ZS_DATASETS    comma list of dataset keys          (default lipophilicity,bace,antiviral,esol)
#   ZS_MODELS      comma list of registry tags, `all` (every enabled entry) or `panel`   (default panel = in_panel entries of src/registry/models_v2.json)
#   ZS_REASONING   effort level(s), comma list         (default low = the main experiment's level)
#   ZS_ITERS       repeats per molecule                (default 1, as in the 45 map)
#   ZS_MAX_TOKENS  completion cap                      (default 32000)
#   ZS_THREADS     concurrent requests                 (default 12)
#   ZS_BUDGET      hard ceiling in USD                 (default 5.0)
#   ZS_RULE        test-set rule                       (default v2test150)
#   ZS_DRYRUN      "1": print plan + prompt, no API calls
#   ZS_PROBE       "1": one call per model x level on one molecule
#   ZS_FORCE       "1": re-query cells that already have a result file
#
# Output: results/contamination/zeroshot/<dataset>__<tag>__<effort>__<rule>.json
#         {"meta": {...}, "calls": [{mol_id, truth, value, text, reasoning_tokens, completion,
#          prompt_tokens, cached_tokens, cost, finish, truncated, provider, model_served}, ...]}
#         -- the schema src/analyze_zeroshot_testset.py scores.
#
# ALWAYS dry-run first:   ZS_DRYRUN=1 julia src/julia/Run_ZeroShot_TestSet.jl
#
include(joinpath(@__DIR__, "Common.jl"))
include(joinpath(@__DIR__, "Prompts_ZS.jl"))      # zs_prompts, zs_conversation, load_test_set

const OUT_DIR = joinpath(RESULTS_DIR, "contamination", "zeroshot")
const SPENT = Ref(0.0)

function run_cell(key::AbstractString, tag::AbstractString, effort::AbstractString;
                  iters::Int, max_tokens::Int, threads::Int, budget::Float64, rule::AbstractString,
                  force::Bool, models, dsets45)
    out_file = joinpath(OUT_DIR, "$(key)__$(tag)__$(effort)__$(rule).json")
    if isfile(out_file) && !force
        println("  [skip] $(basename(out_file)) exists"); return
    end
    test = load_test_set(key, rule)
    ds = dsets45[key]
    jobs = [(row, i) for row in eachrow(test) for i in 1:iters]
    convs = Vector{Tuple}[zs_conversation(ds, strip(string(row.smiles))) for (row, _) in jobs]
    llm = llm_access(tag; models=models)
    calls = Vector{Dict{String,Any}}()
    t0 = time(); before = SPENT[]; stop = Ref(false)
    function on_result(idx, res)
        res === nothing && return
        row, _ = jobs[idx]
        txt = strip(res["text"])
        v = parse_number(txt)
        rec = Dict{String,Any}("mol_id" => row.mol_id, "truth" => Float64(row.value),
                               "text" => first(txt, 200),
                               "value" => (v isa Number && isfinite(v)) ? Float64(v) : nothing,
                               "reasoning_tokens" => res["reasoning_tokens"], "completion" => res["completion_tokens"],
                               "prompt_tokens" => res["prompt_tokens"], "cached_tokens" => res["cached_tokens"],
                               "cost" => res["cost"], "finish" => res["finish"],
                               "truncated" => res["finish"] == "length",
                               "provider" => res["provider"], "model_served" => res["model_served"])
        push!(calls, rec)
        SPENT[] += res["cost"]
        SPENT[] >= budget && (stop[] = true)
    end
    # sampling pinned to temperature = top_p = 1 as in the 45 map (omitted from the body)
    unsupported = ask_threaded_full(llm, convs, on_result; num_threads=threads, stop=stop,
                                    max_tokens=max_tokens, reasoning_effort=effort)
    mkpath(OUT_DIR)
    if unsupported !== nothing && isempty(calls)
        println("  [--] $key/$tag/$effort: $unsupported")
        write(out_file, JSON.json(Dict("meta" => Dict("dataset" => key, "tag" => tag, "level" => effort,
                                                        "unsupported" => true, "error" => unsupported), "calls" => []), 1))
        return
    end
    vals = [c["value"] for c in calls]
    n_ok = count(v -> v isa Number && isfinite(v), vals)
    n_tr = count(c -> c["truncated"], calls)
    rt = sort([c["reasoning_tokens"] for c in calls])
    meta = Dict("dataset" => key, "tag" => tag, "endpoint" => models[tag]["endpoint"],
                "arm" => "effort_$(effort)", "reasoning" => effort, "max_tokens" => max_tokens,
                "rule" => rule, "n_mol" => nrow(test), "iters" => iters,
                "prompt_precision" => "3 significant figures", "temperature" => 1.0, "top_p" => 1.0,
                "cost" => SPENT[] - before, "n_parsed" => n_ok, "n_truncated" => n_tr,
                "median_reasoning_tokens" => isempty(rt) ? 0 : rt[div(length(rt) + 1, 2)],
                "mean_reasoning_tokens" => isempty(rt) ? 0.0 : mean(rt),
                "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
    write(out_file, JSON.json(Dict("meta" => meta, "calls" => calls), 1))
    warn = (n_tr > 0.10 * length(jobs) || n_ok < 0.95 * length(jobs)) ? "  [WARN] truncation-limited" : ""
    @printf("  [ok ] %-14s %-10s %-8s %d/%d parsed  trunc %d  median reasoning %5d tok  \$%.4f  %.0f s%s\n",
            key, tag, effort, n_ok, length(jobs), n_tr, meta["median_reasoning_tokens"], SPENT[] - before, time() - t0, warn)
end

function main()
    keys = envlist("ZS_DATASETS", join(DATASET_KEYS, ","))
    tags = resolve_tags(envget("ZS_MODELS", "panel"))
    efforts = warn_unknown_levels(envlist("ZS_REASONING", "low"))
    iters = parse(Int, envget("ZS_ITERS", "1"))
    max_tokens = parse(Int, envget("ZS_MAX_TOKENS", "32000"))
    threads = parse(Int, envget("ZS_THREADS", "12"))
    budget = parse(Float64, envget("ZS_BUDGET", "5.0"))
    rule = envget("ZS_RULE", "v2test150")
    models = load_models(); dsets45 = load_datasets45()

    n_mol = sum(nrow(load_test_set(k, rule)) for k in keys)
    println("=" ^ 78)
    println("reasoning effort ", join(efforts, ", "), "  max_tokens $max_tokens  rule $rule  iters $iters")
    println("models   ", join(["$t ($(models[t]["endpoint"]))" for t in tags], ", "))
    println("datasets ", join(["$k($(nrow(load_test_set(k, rule))))" for k in keys], ", "))
    println("$(length(keys) * length(tags) * length(efforts)) cells, $(n_mol * length(tags) * length(efforts) * iters) calls, budget ceiling \$$budget")
    println("=" ^ 78)
    if envflag("ZS_DRYRUN")
        s, u = zs_prompts(dsets45[keys[1]], strip(string(load_test_set(keys[1], rule)[1, :smiles])))
        println("\n--- SYSTEM ---\n$s\n--- USER ---\n$u\n\nDRY RUN: no API calls made.")
        return
    end
    if envflag("ZS_PROBE")
        row = load_test_set(keys[1], rule)[1, :]
        s, u = zs_prompts(dsets45[keys[1]], strip(string(row.smiles)))
        for t in tags, e in efforts
            try
                r = ask_gpt_full(llm_access(t; models=models), Tuple[("system", s), ("user", u)];
                                 max_tokens=max_tokens, reasoning_effort=e)
                SPENT[] += r["cost"]
                @printf("  %-10s %-8s reasoning=%6d completion=%6d \$%.5f [%s] answer=%s truth=%s via %s\n",
                        t, e, r["reasoning_tokens"], r["completion_tokens"], r["cost"], r["finish"],
                        repr(first(strip(r["text"]), 30)), row.value, r["provider"])
            catch err
                println("  $t $e: ", first(sprint(showerror, err), 300))
            end
        end
        @printf("probe spent \$%.4f\n", SPENT[])
        return
    end
    for t in tags, e in efforts, k in keys
        if SPENT[] >= budget
            @printf("BUDGET \$%.2f reached (\$%.2f) -- stopping.\n", budget, SPENT[]); return
        end
        run_cell(k, t, e; iters=iters, max_tokens=max_tokens, threads=threads, budget=budget, rule=rule,
                 force=envflag("ZS_FORCE"), models=models, dsets45=dsets45)
    end
    @printf("\nDONE. spent \$%.2f of \$%.2f (library total now \$%.2f)\n", SPENT[], budget, current_usage())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
