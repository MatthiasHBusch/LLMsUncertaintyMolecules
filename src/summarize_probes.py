"""
Summary of the two pre-run probes of 2026-10-08 (no API calls):

  effort-dosability probe  src/julia/Probe_Models.jl  -> results/meta/effort_probe_<date>[_k].json(l)
  ICL probe (N = max)      src/julia/Run_ICL.jl, ICL_PROBE=1 ICL_SIZES=max
                                                     -> results/meta/icl_probe_<date>[_k].jsonl

Several files per probe are merged (batches, retries): per model the LATEST file in which the
model has an answered call at `low` is used (falling back to the latest file that contains it), so
a retry of one model replaces its failed first attempt and nothing else.

Per model: dosability verdict and suggested reasoning_levels (effort probe), median emitted
reasoning tokens at `low` (effort probe, zero-shot prompt), and from the ICL probe (lipophilicity,
N = 1000, effort low, two identical calls): prompt / cached / completion / reasoning tokens, cache
share of call 2 (cached / prompt), cost per call, parse_ok, finish, provider and model served,
billed / list cost ratio; then the projected main-run cost (4 datasets x 6 sizes x 150 test
molecules, effort low) for R = 1 and R = 3 with that model's MEASURED output tokens, cache share
and tokens-per-record factor (estimate_icl_cost.main_cost; same model as
`estimate_icl_cost.py --measured results/meta/probe_summary.csv`).

    python src/summarize_probes.py [--date 2026-10-08]
-> results/meta/probe_summary.csv + printed table ranked by projected cost
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, load_registry  # noqa: E402
from estimate_icl_cost import TIKTOKEN_LIPO_N1000, main_cost  # noqa: E402

META = os.path.join(RESULTS, "meta")
PRICE_FLAG_HI = 1.6       # billed / list above this: flex or the pinned price probably not applied
PRICE_FLAG_LO = 0.6       # billed / list below this: list price or token accounting off


def _suffix_key(path, stem):
    """effort_probe_<date>.json -> 1, _2 -> 2, ... (file order of the runs)."""
    m = re.search(re.escape(stem) + r"(?:_(\d+))?\.jsonl?$", os.path.basename(path))
    return int(m.group(1)) if m and m.group(1) else 1


def load_jsonl(path):
    out = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def pick_latest(files_calls, tag, ok_test):
    """(file, calls of tag) from the latest file with an ok call per ok_test, else the latest with the tag."""
    with_tag = [(f, [c for c in cs if c.get("tag") == tag]) for f, cs in files_calls]
    with_tag = [(f, cs) for f, cs in with_tag if cs]
    if not with_tag:
        return None, []
    good = [(f, cs) for f, cs in with_tag if any(ok_test(c) for c in cs)]
    return (good or with_tag)[-1]


def norm_model(s):
    s = (s or "").lower().strip()
    s = re.sub(r"-(20\d{6}|\d{4})$", "", s)          # date / snapshot suffix
    return s


def fmt(v, spec):
    width = int(re.match(r"(\d*)", spec).group(1) or 0)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-".rjust(width)
    return format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-08")
    ap.add_argument("--out", default=os.path.join(META, "probe_summary.csv"))
    args = ap.parse_args()
    reg = load_registry()

    # ---------------------------------------------------------------- effort probe
    stem_e = f"effort_probe_{args.date}"
    e_files = sorted(set(glob.glob(os.path.join(META, stem_e + "*.jsonl"))), key=lambda p: _suffix_key(p, stem_e))
    e_calls, e_summ = [], {}
    for f in e_files:
        e_calls.append((f, load_jsonl(f)))
        js = f[:-1]                                       # .jsonl -> .json
        e_summ[f] = json.load(open(js, encoding="utf8"))["summary"] if os.path.exists(js) else {}

    # ---------------------------------------------------------------- ICL probe
    stem_i = f"icl_probe_{args.date}"
    i_files = sorted(glob.glob(os.path.join(META, stem_i + "*.jsonl")), key=lambda p: _suffix_key(p, stem_i))
    i_calls = [(f, load_jsonl(f)) for f in i_files]

    tags = [t for t in reg if any(c.get("tag") == t for _, cs in e_calls + i_calls for c in cs)]
    rows = []
    for t in tags:
        md = reg[t]
        r = dict(tag=t, name=md["name"], endpoint=md["endpoint"], in_panel=md.get("in_panel", False),
                 tier=md.get("service_tier") or "std", providers_pinned=",".join(md.get("providers") or []))
        notes = []
        # effort probe
        ef, ecs = pick_latest(e_calls, t, lambda c: c.get("level") == "low" and c.get("status") == "ok")
        sm = e_summ.get(ef, {}).get(t, {}) if ef else {}
        r["effort_file"] = os.path.basename(ef) if ef else None
        r["verdict"] = sm.get("monotone_verdict")
        r["levels_supported"] = "/".join(sm.get("levels_supported", [])) or None
        r["levels_rejected"] = "/".join(sm.get("levels_rejected", [])) or None
        r["suggested_levels"] = "/".join(sm.get("suggested_reasoning_levels", [])) or None
        r["dose_measure"] = sm.get("dose_measure")
        low_ok = [c for c in ecs if c.get("level") == "low" and c.get("status") == "ok"]
        r["zs_reasoning_low_median"] = statistics.median([c["reasoning_tokens"] for c in low_ok]) if low_ok else None
        r["zs_completion_low_median"] = statistics.median([c["completion_tokens"] for c in low_ok]) if low_ok else None
        r["effort_cost"] = sum(c.get("cost") or 0.0 for c in ecs)
        r["effort_cost_ratio_max"] = sm.get("cost_ratio_max")
        bad = sorted({c["status"] for c in ecs if c.get("status") not in ("ok", "unsupported")})
        r["effort_failures"] = ",".join(bad) or None
        if not ecs:
            notes.append("not in effort probe")
        elif not low_ok:
            err = next((c.get("error") or "" for c in ecs if c.get("status") not in ("ok", "unsupported")), "")
            if "Guardrails" in err or "training violation" in err:
                notes.append("effort probe: no endpoint (account guardrail: provider trains on paid prompts)")
            elif "Flex processing is temporarily unavailable" in err:
                notes.append("effort probe: flex unavailable (502)")
            else:
                notes.append("effort probe: no answer at low")
        cf = [c for c in ecs if c.get("finish") == "content_filter"]
        if cf:
            notes.append(f"effort probe: {len(cf)} call(s) finish=content_filter (" + ",".join(sorted({c['level'] for c in cf})) + ")")
        trunc = [c for c in ecs if c.get("finish") == "length"]
        if trunc:
            notes.append(f"effort probe: {len(trunc)} call(s) truncated at max_tokens (" + ",".join(sorted({c['level'] for c in trunc})) + ")")
        if r["levels_rejected"]:
            notes.append(f"rejects {r['levels_rejected']}")
        if r["verdict"] and (r["verdict"].startswith("flat") or r["verdict"].startswith("on/off")):
            notes.append("effort not dosable (" + r["verdict"].split(" (")[0] + ")")

        # ICL probe
        # an HTTP-200 answer with finish = content_filter (0 tokens, no text) is a refusal, not an answer
        answered = lambda c: c.get("status") == "ok" and c.get("finish") != "content_filter"   # noqa: E731
        icf, ics = pick_latest(i_calls, t, answered)
        ok = [c for c in ics if answered(c)]
        refused = [c for c in ics if c.get("status") == "ok" and c.get("finish") == "content_filter"]
        r["icl_refused"] = len(refused)
        c1 = next((c for c in ok if c.get("call") == 1), None)
        c2 = next((c for c in ok if c.get("call") == 2), None)
        r["icl_file"] = os.path.basename(icf) if icf else None
        r["icl_calls_ok"] = f"{len(ok)}/{len(ics)}" if ics else None
        for k, c in (("1", c1), ("2", c2)):
            for fld in ("prompt_tokens", "cached_tokens", "cache_write_tokens", "completion_tokens",
                        "reasoning_tokens", "cost", "latency_s", "finish", "prediction", "uncertainty"):
                r[f"icl{k}_{fld}"] = c.get(fld) if c else None
        r["icl_cache_share_call2"] = (c2["cached_tokens"] / c2["prompt_tokens"]) if c2 and c2.get("prompt_tokens") else None
        r["icl_out_tokens"] = statistics.mean([c["completion_tokens"] for c in ok]) if ok else None
        r["icl_reasoning_tokens"] = statistics.mean([c["reasoning_tokens"] for c in ok]) if ok else None
        r["icl_cost_per_call"] = statistics.mean([c["cost"] for c in ok]) if ok else None
        r["tok_factor"] = (c1 or c2)["prompt_tokens"] / TIKTOKEN_LIPO_N1000 if (c1 or c2) else None
        r["icl_parse_ok"] = f"{sum(bool(c.get('parse_ok')) for c in ok)}/{len(ok)}" if ok else None
        r["icl_finish"] = "/".join(str(c.get("finish")) for c in ok) if ok else None
        r["provider_served"] = ",".join(sorted({str(c.get("provider")) for c in ok + low_ok if c.get("provider")})) or None
        r["model_served"] = ",".join(sorted({str(c.get("model_served")) for c in ok if c.get("model_served")})) or None
        ratios = [c["cost_ratio"] for c in ok if c.get("cost_ratio") is not None]
        r["icl_cost_ratio_max"] = max(ratios) if ratios else None
        r["icl_cost_ratio_min"] = min(ratios) if ratios else None
        n_refused_all = sum(1 for _, cs in i_calls for c in cs
                            if c.get("tag") == t and c.get("status") == "ok" and c.get("finish") == "content_filter")
        if ics and not ok and n_refused_all:
            n_all = sum(1 for _, cs in i_calls for c in cs if c.get("tag") == t)
            notes.append(f"ICL probe REFUSED: finish=content_filter, 0 tokens, on {n_refused_all}/{n_all} calls")
        elif ics and not ok:
            err = next((c.get("error") or "" for c in ics), "")
            st = ",".join(sorted({c.get("status") for c in ics}))
            if "Guardrails" in err or "training violation" in err:
                notes.append("ICL probe: no endpoint (account guardrail)")
            elif "Flex processing is temporarily unavailable" in err:
                notes.append("ICL probe: flex unavailable (502)")
            elif "429" in err:
                notes.append("ICL probe: rate limited (429)")
            else:
                notes.append(f"ICL probe failed ({st})")
        elif not ics:
            notes.append("not in ICL probe")
        if ok and len(ok) < len(ics):
            bad_i = [c for c in ics if not answered(c)]
            why = ",".join(sorted({("429" if "429" in (c.get("error") or "") else
                                    "flex 502" if "Flex processing" in (c.get("error") or "") else
                                    c.get("finish") if c.get("status") == "ok" else c.get("status")) for c in bad_i}))
            notes.append(f"ICL probe: {len(bad_i)} of {len(ics)} calls without answer ({why})")
        if c1 is not None and c2 is not None and (c1.get("cached_tokens") or 0) > 0.5 * c1["prompt_tokens"]:
            notes.append("call 1 already reports cached tokens (= cache-write tokens; provider accounting)")
        if ok and sum(bool(c.get("parse_ok")) for c in ok) < len(ok):
            notes.append("ICL parse failure")
        if ok and any(c.get("finish") == "length" for c in ok):
            notes.append("ICL truncated (finish=length)")
        if c2 is not None and (r["icl_cache_share_call2"] or 0) < 0.05:
            notes.append("no cache hit on call 2")
        # price flag
        flags = []
        for lab, v in (("effort", r["effort_cost_ratio_max"]), ("icl", r["icl_cost_ratio_max"])):
            if v is not None and v > PRICE_FLAG_HI:
                flags.append(f"{lab} x{v:.2f}")
        if r["icl_cost_ratio_min"] is not None and r["icl_cost_ratio_min"] < PRICE_FLAG_LO:
            flags.append(f"icl x{r['icl_cost_ratio_min']:.2f}")
        r["price_flag"] = "; ".join(flags) or None
        if flags:
            notes.append("price flag " + "; ".join(flags))
        provs = {c.get("provider") for c in ok + low_ok if c.get("provider")}
        if len(provs) > 1:
            notes.append("several providers served: " + ",".join(sorted(provs)))
        if r["model_served"]:
            for ms in r["model_served"].split(","):
                if not norm_model(ms).startswith(norm_model(md["endpoint"])) and \
                        not norm_model(md["endpoint"]).startswith(norm_model(ms)):
                    notes.append(f"served model {ms} != {md['endpoint']}")
        # projection with the measured parameters
        if r["icl_out_tokens"] is not None:
            cf = r["icl_cache_share_call2"] if r["icl_cache_share_call2"] is not None else 0.0
            tok = 55 * (r["tok_factor"] or 1.0)
            r["proj_main_R1"] = main_cost(md, 1, r["icl_out_tokens"], cf, tok)
            r["proj_main_R3"] = main_cost(md, 3, r["icl_out_tokens"], cf, tok)
        else:
            r["proj_main_R1"] = r["proj_main_R3"] = None
        r["notes"] = "; ".join(notes) or None
        rows.append(r)

    D = pd.DataFrame(rows)
    D = D.sort_values("proj_main_R1", na_position="last").reset_index(drop=True)
    D.to_csv(args.out, index=False)

    # ---------------------------------------------------------------- printed table
    e_spent = sum(c.get("cost") or 0.0 for _, cs in e_calls for c in cs)
    i_spent = sum(c.get("cost") or 0.0 for _, cs in i_calls for c in cs)
    print(f"effort probe files: {', '.join(os.path.basename(f) for f in e_files) or '-'}   spent ${e_spent:.4f}")
    print(f"ICL probe files:    {', '.join(os.path.basename(f) for f in i_files) or '-'}   spent ${i_spent:.4f}")
    print(f"total probe spend ${e_spent + i_spent:.4f}\n")
    hdr = (f"{'tag':13s} {'P':1s} {'verdict':12s} {'levels':26s} {'rsn@low':>7s} {'ICLout':>7s} {'ICLrsn':>7s} "
           f"{'cache2':>6s} {'$/call1':>8s} {'$/call2':>8s} {'parse':>5s} {'R1 $':>7s} {'R3 $':>7s}  provider / served")
    print(hdr)
    print("-" * len(hdr))
    for _, r in D.iterrows():
        v = (r["verdict"] or "-").split(" (")[0].split("  [")[0]
        print(f"{r['tag']:13s} {'*' if r['in_panel'] else ' ':1s} {v[:12]:12s} {(r['suggested_levels'] or '-')[:26]:26s} "
              f"{fmt(r['zs_reasoning_low_median'], '7.0f')} {fmt(r['icl_out_tokens'], '7.0f')} {fmt(r['icl_reasoning_tokens'], '7.0f')} "
              f"{fmt(r['icl_cache_share_call2'], '6.2f')} {fmt(r['icl1_cost'], '8.4f')} {fmt(r['icl2_cost'], '8.4f')} "
              f"{(r['icl_parse_ok'] or '-'):>5s} {fmt(r['proj_main_R1'], '7.0f')} {fmt(r['proj_main_R3'], '7.0f')}  "
              f"{r['provider_served'] or '-'} / {r['model_served'] or '-'}")
    print("\n(P * = panel; verdict / levels = effort probe (suggested reasoning_levels); rsn@low = median reasoning tokens")
    print(" at low, zero-shot prompt; ICLout / ICLrsn = mean completion / reasoning tokens of the ICL-probe calls at N=1000,")
    print(" low; cache2 = cached / prompt tokens of call 2; R1 / R3 = projected main run, USD, measured parameters)")
    print("\nnotes:")
    for _, r in D.iterrows():
        if r["notes"]:
            print(f"  {r['tag']:13s} {r['notes']}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
