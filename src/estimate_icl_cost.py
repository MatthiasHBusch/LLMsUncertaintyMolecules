"""
Cost estimate for the v2 ICL experiments (main ladder + reasoning sub-experiment) from the
prices of the v2 registry (src/registry/models_v2.json, USD per Mtok, fetched 2026-10-08 from the
OpenRouter /endpoints of the pinned / flex endpoint) and explicit token assumptions.
Replace the assumptions with measured numbers from `ICL_PROBE=1 julia src/julia/Run_ICL.jl`
before committing.

Assumptions (override on the command line):
  tokens per in-context record   ~55 for every dataset (v1 measured 44-54; v2 with tiktoken:
                                 lipophilicity 47, BACE 64, antiviral 52, ESOL 35 -- more on Claude
                                 tokenizers, BACE N = 1000 ~80-87k). The prompt of a cell is
                                 fixed + 55 * N; all stay below the long-context surcharges of the
                                 panel (Haiku 5.5 above 100k, Grok above 200k, GPT-5.6 above 272k).
  fixed prompt tokens            ~450 (system + task + JSON contract)
  cached input                   at N >= 120 a share --cache-frac (default 0.8; v1 measured 70-89 %)
                                 of the prompt tokens is billed at the registry's price_cache_read
                                 (USD/Mtok of that endpoint), the rest at price_in. N < 120: no
                                 cache. Cache-write premiums (Anthropic 1.25x on the uncached share)
                                 are NOT modelled. Whether an endpoint actually serves cache hits
                                 (e.g. Kimi K3 at Moonshot lists no implicit caching) is measured by
                                 the probe, not assumed here.
  output tokens per call         main level `low`: --out-main (default 1200; v1 measured
                                 685-5356 across providers at "medium"); ladder levels
                                 none 150, minimal 400, low 1200, medium 2500, high 6000
  DeepSeek V4.1 Flash            off-peak base price; weekday peak windows (01-04, 06-10 UTC) cost 2x.

    python src/estimate_icl_cost.py --out-main 1200 --main-level low --repeats 3 --cache-frac 0.8
    python src/estimate_icl_cost.py --repeats 2 --csv results/icl_cost_estimate_R2.csv

Measured mode (2026-10-08): `--measured results/meta/probe_summary.csv` (written by
src/summarize_probes.py from the effort probe and the ICL probe) overrides, PER MODEL, the main
level's output tokens (`icl_out_tokens`: mean completion tokens incl. reasoning of the ICL-probe
calls at N = 1000, effort low), the cache share (`icl_cache_share_call2`: cached / prompt tokens
of the second, identical call) and the tokens per record (`tok_factor`: measured prompt tokens /
48,685 o200k tokens of the same N = 1000 lipophilicity prompt, applied to --tok-per-record). A
model without a usable measurement keeps the command-line assumptions (marked "assumed").
Cache-write premiums (Anthropic 1.25x, once per cell and cache lifetime) are still not modelled.
`--models all` = every tag of the registry; `--models measured` = every tag in the CSV.

    python src/estimate_icl_cost.py --measured results/meta/probe_summary.csv --models measured --repeats 1
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import ALL_KEYS, PANEL, RESULTS, load_registry, load_split  # noqa: E402

OUT_BY_LEVEL = {"none": 150, "minimal": 400, "low": 1200, "medium": 2500, "high": 6000}
CACHE_MIN_N = 120
TIKTOKEN_LIPO_N1000 = 48685      # o200k tokens of the lipophilicity N = 1000 probe prompt (3 messages)


def load_measured(path):
    """{tag: dict(out, cache_frac, tok_factor)} from probe_summary.csv; NaN / missing -> None."""
    import math
    M = pd.read_csv(path)
    out = {}
    for _, r in M.iterrows():
        def val(col):
            v = r.get(col)
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        out[str(r["tag"])] = dict(out=val("icl_out_tokens"), cache_frac=val("icl_cache_share_call2"),
                                  tok_factor=val("tok_factor"))
    return out


def main_cost(md, repeats, out_tokens, cache_frac, tok_per_rec=55, fixed=450, n_test=150):
    """USD of the main experiment (all datasets x ladder sizes, one level) for one model."""
    total = 0.0
    for key in ALL_KEYS:
        for n in load_split(key)["sizes"]:
            total += cell_cost(md, n, n_test, repeats, out_tokens, tok_per_rec, fixed, cache_frac)[0]
    return total


def cell_cost(md, n_train, n_test, repeats, out_tokens, tok_per_rec, fixed, cache_frac):
    """USD, calls, prompt tokens per call for one cell."""
    prompt = fixed + tok_per_rec * n_train
    calls = n_test * repeats
    p_in = md["price_in"]
    p_cr = md.get("price_cache_read")
    p_cr = p_in if p_cr is None else p_cr
    if n_train >= CACHE_MIN_N:
        in_usd = prompt * ((1 - cache_frac) * p_in + cache_frac * p_cr)
    else:
        in_usd = prompt * p_in
    return calls * (in_usd + out_tokens * md["price_out"]) / 1e6, calls, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--out-main", type=int, default=1200, help="output tokens per call at the main level")
    ap.add_argument("--main-level", default="low")
    ap.add_argument("--tok-per-record", type=int, default=55)
    ap.add_argument("--fixed", type=int, default=450)
    ap.add_argument("--cache-frac", type=float, default=0.8, help="share of prompt tokens served from cache")
    ap.add_argument("--models", default=",".join(PANEL), help="comma list, 'all' (registry) or 'measured' (CSV)")
    ap.add_argument("--measured", default=None, help="probe_summary.csv: per-model output tokens / cache share / tok factor")
    ap.add_argument("--csv", default=os.path.join(RESULTS, "icl_cost_estimate.csv"))
    args = ap.parse_args()
    models = load_registry()
    out_by_level = dict(OUT_BY_LEVEL)
    out_by_level[args.main_level] = args.out_main
    main_level = args.main_level
    measured = load_measured(args.measured) if args.measured else {}
    if args.models == "all":
        tags = list(models)
    elif args.models == "measured":
        tags = [t for t in models if t in measured]
    else:
        tags = args.models.split(",")
    # per-model parameters: measured where available, else the command-line assumptions
    par = {}
    for t in tags:
        m = measured.get(t, {})
        par[t] = dict(out=m.get("out") if m.get("out") is not None else out_by_level[main_level],
                      cache_frac=m.get("cache_frac") if m.get("cache_frac") is not None else args.cache_frac,
                      tok=args.tok_per_record * (m.get("tok_factor") or 1.0),
                      source="measured" if m.get("out") is not None else "assumed")
    if measured:
        print("per-model parameters (main level):")
        for t in tags:
            print(f"  {t:13s} out {par[t]['out']:7.0f} tok/call  cache share {par[t]['cache_frac']:.2f}  "
                  f"tok/record {par[t]['tok']:5.1f}  [{par[t]['source']}]")
        print()
    rows = []
    for key in ALL_KEYS:
        sizes = load_split(key)["sizes"]
        for t in tags:
            md = models[t]
            for n in sizes:                                    # main experiment
                c, calls, prompt = cell_cost(md, n, args.n_test, args.repeats, par[t]["out"],
                                             par[t]["tok"], args.fixed, par[t]["cache_frac"])
                rows.append(dict(experiment="main", dataset=key, model=t, size=n, level=main_level,
                                 repeats=args.repeats, calls=calls, prompt_tokens=prompt, usd=c))
            for n in (sizes[0], sizes[-1]):                    # reasoning ladder, min & max size
                for lv, out in out_by_level.items():
                    if lv == main_level:
                        continue                               # already covered by the main run
                    c, calls, prompt = cell_cost(md, n, args.n_test, args.repeats, out,
                                                 par[t]["tok"], args.fixed, par[t]["cache_frac"])
                    rows.append(dict(experiment="reasoning_ladder", dataset=key, model=t, size=n,
                                     level=lv, repeats=args.repeats, calls=calls, prompt_tokens=prompt, usd=c))
    D = pd.DataFrame(rows)
    D.to_csv(args.csv, index=False)
    pd.set_option("display.width", 200)
    print(f"assumptions: repeats {args.repeats}, {args.n_test} test molecules, {args.tok_per_record} tok/record, "
          f"main level '{main_level}' with {args.out_main} output tok/call, cache share {args.cache_frac} at N >= "
          f"{CACHE_MIN_N} billed at price_cache_read, ladder output tok {out_by_level}")
    print(f"largest prompt per call: {D.prompt_tokens.max():,} tokens"
          + ("  [WARN] above 100k: Haiku 5.5 long-context surcharge applies" if D.prompt_tokens.max() > 100_000 else ""))
    print("prices (USD/Mtok in / cache read / out): " + "; ".join(
        f"{t} {models[t]['price_in']:g}/{models[t]['price_cache_read'] if models[t]['price_cache_read'] is not None else '-'}"
        f"/{models[t]['price_out']:g}" for t in tags) + "\n")
    print(f"MAIN EXPERIMENT ({main_level}), USD per model x dataset:")
    M = D[D.experiment == "main"]
    print(M.pivot_table(index="model", columns="dataset", values="usd", aggfunc="sum", margins=True)
          .reindex(tags + ["All"]).round(0).to_string())
    print("\nREASONING LADDER (other levels at min & max size), USD per model x level:")
    L = D[D.experiment == "reasoning_ladder"]
    print(L.pivot_table(index="model", columns="level", values="usd", aggfunc="sum", margins=True)
          .reindex(tags + ["All"]).round(0).to_string())
    S = pd.DataFrame({"main": M.groupby("model").usd.sum(), "ladder": L.groupby("model").usd.sum()}).reindex(tags)
    S["total"] = S.main + S.ladder
    if measured:
        S["params"] = [par[t]["source"] for t in S.index]
        S = S.sort_values("main")
        S.loc["All"] = [S.main.sum(), S.ladder.sum(), S.total.sum(), ""]
        S[["main", "ladder", "total"]] = S[["main", "ladder", "total"]].astype(float)
    else:
        S.loc["All"] = S.sum()
    print(f"\nPER MODEL (R = {args.repeats}), USD:")
    print(S.round(0).to_string())
    print(f"\nTOTAL calls: main {int(M.calls.sum())}, ladder {int(L.calls.sum())}; "
          f"USD main {M.usd.sum():.0f}, ladder {L.usd.sum():.0f}  -> {args.csv}")


if __name__ == "__main__":
    main()
