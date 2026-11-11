"""Numbers quoted in docs/REPORT_2026-10-13.md that no other results table holds.

No API calls; reads only finished results and writes results/icl/report_extras.json.

    python src/report_extras.py

Sections of the JSON (all LLM cells: reasoning `low`; draw 1 unless stated):
  costs            calls / USD per run (main run, draws 2+3 raw records, ladder, zero-shot, probes)
  table31          REPORT table 3.1 (draw-1 rule): RMSE at N = 30 in draw 1 with [min-max] over the 3 context
                   draws, and at the top rung (draw 1); LLMs from cells.csv, baselines (all 8) from
                   baselines/summary.csv averaged over the seeds within a draw
  gains            relative RMSE change N = 30 -> top on the clean sets (draw 1 -> draw 1, and with draw 2 or
                   3 as the N = 30 starting point)
  paired_top       each LLM vs TabPFN (seed 0) at the top rung on identical molecules: paired Wilcoxon
                   signed-rank test on |err_LLM| - |err_TabPFN|, mean |err| difference, RMSE difference
                   with a paired bootstrap 95 % CI (1000 resamples over molecules, default_rng(0), molecules
                   in the split's test_ids order, so the CIs equal calibration_bootstrap.csv and
                   bootstrap_cells.csv)
  pred_corr        Pearson correlation of the per-molecule prediction vectors (mean of the R = 3
                   repeats) at N = 30 vs the top rung, per model and dataset
  sonnet_text      regex counts in Sonnet 5.5's response texts at the top rung (all parsed repeats):
                   share of calls naming "AqSolDB" / containing "recall", and the accuracy of those calls
  r1_vs_r3         RMSE from repeat 0 alone (R = 1) minus RMSE of the 3-repeat mean (R = 3), per cell
  sigma_distinct   number of distinct raw self-reported sigma values per cell (single calls)
  esol_subsets     RMSE recalled / never recalled (from cells.csv)
  draws            SD of RMSE over the 3 context draws: medians per dataset x N (LLM vs baselines),
                   per-model range at N = 30, learning-curve inversions (N = 60 worse than N = 30)
  selective        REPORT table 3.6 (draw-1 rule): RMSE / RMSE50 / gain50 (gain50 = 1 - RMSE50 / RMSE) at
                   N = 30 / 60 / 120 in draw 1 with [min-max] and the per-draw values, and at the top rung
                   (draw 1); LLMs (`low`) and 7 baselines, baselines averaged over the seeds within a draw
                   (RMSE, RMSE50 and gain50 each averaged, as in fig3)
  gp_bounds        GP ML-II fits at N = 30 (seed 0, draws 1-3): which hyperparameter sits on a bound
  gemini_zero_reasoning  share of Gemini 3.8 Flash main-run calls (draw 1, `low`) with 0 reasoning tokens
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, load_split  # noqa: E402

ICL = os.path.join(RESULTS, "icl")
BASE = os.path.join(RESULTS, "baselines")
TOP = {"lipophilicity": 1000, "bace": 1000, "antiviral": 844, "esol": 976}
DS = ["lipophilicity", "bace", "antiviral", "esol"]
CLEAN = DS[:3]
LLMS = ["sonnet55", "haiku55", "gpt61sol", "gpt6astra", "gem38flash", "gpt6luna", "glm53flash"]
BASELINES = ["tabpfn", "gp_tanimoto", "rf_desc", "chemprop_ens", "mlp_ens", "rf", "knn_tanimoto"]
BOOT_B, BOOT_SEED = 1000, 0


def r(x, k=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def read_jsonl(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows)


def records(ds, model, size, effort="low", draw=1, dedup=True):
    suffix = "" if draw == 1 else f"__d{draw}"
    R = read_jsonl(os.path.join(ICL, f"{ds}__{model}__N{size}__{effort}{suffix}.jsonl"))
    if dedup:  # as analyze_icl.py: a gap-fill may append a (mol_id, repeat) pair twice
        R = R.drop_duplicates(["mol_id", "repeat"], keep="last")
    return R


# ---------------------------------------------------------------- costs
def costs(C):
    out = {}
    main = C[(C.reasoning == "low") & (C.draw == 1)]
    out["main_run"] = dict(calls=int(main.n_calls.sum()), usd=r(main.cost_usd.sum(), 2),
                           astra_usd=r(main[main.model == "gpt6astra"].cost_usd.sum(), 2))
    n = usd = 0
    for f in glob.glob(os.path.join(ICL, "*__low__d[23].jsonl")):
        R = read_jsonl(f)
        n += len(R)
        usd += R.cost.fillna(0).sum()
    dr = C[(C.reasoning == "low") & (C.draw > 1)]
    out["draws_2_3"] = dict(raw_records=n, usd_raw=r(usd, 2), calls_after_dedup=int(dr.n_calls.sum()),
                            usd_after_dedup=r(dr.cost_usd.sum(), 2),
                            duplicates_dropped=int(dr.n_duplicates_dropped.sum()))
    lad = C[C.reasoning != "low"]
    out["ladder"] = dict(calls=int(lad.n_calls.sum()), usd=r(lad.cost_usd.sum(), 2), cells=len(lad),
                         min_calls_per_cell=int(lad.n_calls.min()),
                         gemini_high_usd=r(lad[(lad.model == "gem38flash") & (lad.reasoning == "high")].cost_usd.sum(), 2))
    Z = pd.read_csv(os.path.join(RESULTS, "contamination", "zeroshot", "cells.csv"))
    out["zeroshot"] = dict(calls=int(Z.n_calls.sum()), usd=r(Z.cost_usd.sum(), 2), cells=len(Z),
                           astra_calls=int(Z[Z.model == "gpt6astra"].n_calls.sum()))
    n = usd = 0
    models = set()
    for f in glob.glob(os.path.join(RESULTS, "meta", "*probe*.jsonl")):
        for rec in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip()):
            n += 1
            v = rec.get("cost", rec.get("cost_usd", 0))
            usd += v if isinstance(v, (int, float)) else 0
            models.add(rec.get("model") or rec.get("tag"))
    out["probes"] = dict(calls=n, usd=r(usd, 2), n_models=len(models - {None}))
    out["total_usd"] = r(sum(out[k]["usd"] for k in ("main_run", "ladder", "zeroshot", "probes"))
                         + out["draws_2_3"]["usd_raw"], 2)
    return out


# ---------------------------------------------------------------- table 3.1 and gains
def seed_mean_rmse():
    """Baseline RMSE averaged over the seeds within a draw: {(method, dataset, size, draw): RMSE}."""
    B = pd.read_csv(os.path.join(BASE, "summary.csv"))
    return B.groupby(["method", "dataset", "size", "draw"]).RMSE.mean().to_dict()


def table31(C):
    L = C[C.reasoning == "low"].set_index(["model", "dataset", "size", "draw"]).RMSE.to_dict()
    BM = seed_mean_rmse()
    out = {}
    for ds in DS:
        for m in LLMS + BASELINES + ["trivial"]:
            src = L if m in LLMS else BM
            d30 = [src[(m, ds, 30, d)] for d in (1, 2, 3)]
            out[f"{ds}|{m}"] = dict(N30_draw1=r(d30[0], 6), N30_min=r(min(d30), 6), N30_max=r(max(d30), 6),
                                    N30_draws=[r(x, 6) for x in d30], top_draw1=r(src[(m, ds, TOP[ds], 1)], 6))
    return out


def gains(T):
    g1, g23 = [], []
    for ds in CLEAN:
        for m in LLMS:
            t = T[f"{ds}|{m}"]
            g1.append((1 - t["top_draw1"] / t["N30_draw1"], ds, m))
            g23 += [(1 - t["top_draw1"] / t["N30_draws"][d - 1], ds, m, d) for d in (2, 3)]
    f = lambda g: dict(min=r(min(g)[0], 3), min_cell=min(g)[1:], max=r(max(g)[0], 3), max_cell=max(g)[1:])
    return dict(draw1=f(g1), draw2or3_N30=f(g23))


# ---------------------------------------------------------------- paired tests at the top rung
def permol(ds):
    P = pd.read_csv(os.path.join(ICL, f"{ds}_permolecule.csv"), dtype={"mol_id": str})
    return P[(P.reasoning == "low") & (P.draw == 1)]


def paired_top():
    out = {}
    for ds in DS:
        P = permol(ds)
        tab = pd.read_csv(os.path.join(BASE, f"tabpfn__{ds}__N{TOP[ds]}.csv"), dtype={"mol_id": str})[["mol_id", "truth", "prediction"]]
        order = {m: i for i, m in enumerate(str(x) for x in load_split(ds)["test_ids"])}
        for m in LLMS:
            A = P[(P.model == m) & (P["size"] == TOP[ds])][["mol_id", "truth", "pred"]]
            J = A.merge(tab, on="mol_id", suffixes=("", "_tab"))
            J = J.assign(_o=J.mol_id.map(order)).sort_values("_o").reset_index(drop=True)   # split order
            assert J._o.notna().all() and J.mol_id.is_unique
            assert np.allclose(J.truth, J.truth_tab)
            ea, eb = np.abs(J.pred - J.truth).values, np.abs(J.prediction - J.truth).values
            n = len(J)
            w = stats.wilcoxon(ea - eb)
            rm = lambda e: np.sqrt(np.mean(e ** 2))
            idx = np.random.default_rng(BOOT_SEED).integers(0, n, (BOOT_B, n))
            d = np.array([rm(ea[i]) - rm(eb[i]) for i in idx])
            dm = np.array([ea[i].mean() - eb[i].mean() for i in idx])
            out[f"{ds}|{m}"] = dict(
                n=n, rmse_llm=r(rm(ea)), rmse_tabpfn=r(rm(eb)), rmse_diff=r(rm(ea) - rm(eb)),
                rmse_diff_ci=[r(np.percentile(d, 2.5)), r(np.percentile(d, 97.5))],
                mae_diff=r(ea.mean() - eb.mean()),
                mae_diff_ci=[r(np.percentile(dm, 2.5)), r(np.percentile(dm, 97.5))],
                wilcoxon_p=float(f"{w.pvalue:.3g}"))
    return out


def pred_corr():
    out = {}
    for ds in DS:
        P = permol(ds)
        for m in LLMS:
            a = P[(P.model == m) & (P["size"] == 30)].set_index("mol_id").pred
            b = P[(P.model == m) & (P["size"] == TOP[ds])].set_index("mol_id").pred
            j = pd.concat([a, b], axis=1, join="inner")
            out[f"{ds}|{m}"] = r(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1], 3)
    rng = lambda keys: [min(out[k] for k in keys), max(out[k] for k in keys)]
    out["range_clean"] = rng([k for k in out if k.split("|")[0] in CLEAN])
    out["range_esol"] = rng([k for k in out if k.startswith("esol|")])
    return out


# ---------------------------------------------------------------- Sonnet response texts
def sonnet_text():
    """Per size (draw 1, all parsed repeats) and pooled over the six sizes; key 'top' = top rung."""
    out = {}
    pats = {"AqSolDB": re.compile(r"aqsoldb", re.I), "recall": re.compile(r"recall", re.I)}
    for ds in ("esol", "lipophilicity"):
        allR = []
        for n in (30, 60, 120, 250, 500, TOP[ds]):
            R = records(ds, "sonnet55", n)
            R = R[R.parse_ok == True].copy()  # noqa: E712
            R["abs_err"] = (R.prediction - R.truth).abs()
            R["size"] = n
            allR.append(R)
        allR = pd.concat(allR)
        o = {}
        for key, R in [(str(n), g) for n, g in allR.groupby("size")] + [("pooled", allR)]:
            oo = dict(n_calls=len(R))
            for name, p in pats.items():
                hit = R.text.fillna("").str.contains(p)
                oo[name] = dict(share=r(hit.mean(), 3), n=int(hit.sum()),
                                mae_with=r(R[hit].abs_err.mean(), 3) if hit.any() else None,
                                mae_without=r(R[~hit].abs_err.mean(), 3) if (~hit).any() else None)
            o["top" if key == str(TOP[ds]) else key] = oo
        out[ds] = o
    return out


# ---------------------------------------------------------------- R = 1 vs R = 3, sigma granularity
def r1_vs_r3(C):
    main = C[(C.reasoning == "low") & (C.draw == 1)].set_index(["dataset", "model", "size"]).RMSE
    diffs, distinct = {}, {}
    for (ds, m, n), rmse3 in main.items():
        R = records(ds, m, n)
        R = R[R.parse_ok == True]  # noqa: E712
        r0 = R[R.repeat == 0]
        diffs[f"{ds}|{m}|{n}"] = float(np.sqrt(((r0.prediction - r0.truth) ** 2).mean()) - rmse3)
        distinct[f"{ds}|{m}|{n}"] = int(R.uncertainty.round(6).nunique())
    d = np.array(list(diffs.values()))
    per_model = {m: [min(v for k, v in distinct.items() if k.split("|")[1] == m),
                     max(v for k, v in distinct.items() if k.split("|")[1] == m)] for m in LLMS}
    return (dict(n_cells=len(d), min=r(d.min(), 3), max=r(d.max(), 3), median=r(np.median(d), 3),
                 share_R3_better=r((d > 0).mean(), 3), q10_q90=[r(np.quantile(d, .1), 3), r(np.quantile(d, .9), 3)]),
            dict(per_cell_min=min(distinct.values()), per_cell_max=max(distinct.values()), per_model=per_model))


def esol_subsets(C):
    E = C[(C.reasoning == "low") & (C.draw == 1) & (C.dataset == "esol")]
    out = dict(n_recalled=int(E.n_recalled.iloc[0]), n_never_recalled=int(E.n_never_recalled.iloc[0]),
               n_unscorable=int(E.n_unscorable_zeroshot.iloc[0]))
    for m in LLMS:
        x = E[E.model == m].set_index("size")
        out[m] = dict(top=[r(x.loc[TOP["esol"], "RMSE_recalled"], 3), r(x.loc[TOP["esol"], "RMSE_never_recalled"], 3)],
                      N30=[r(x.loc[30, "RMSE_recalled"], 3), r(x.loc[30, "RMSE_never_recalled"], 3)],
                      ratio_range=[r((x.RMSE_never_recalled / x.RMSE_recalled).min(), 2),
                                   r((x.RMSE_never_recalled / x.RMSE_recalled).max(), 2)])
    return out


# ---------------------------------------------------------------- context draws
def draws(DC):
    SD = DC.groupby(["dataset", "family", "model", "size"]).RMSE.std(ddof=1).reset_index()
    med = SD.groupby(["dataset", "size", "family"]).RMSE.median().unstack("family")
    out = dict(median_sd={f"{ds}|{n}": dict(LLM=r(row.LLM, 3), baseline=r(row.baseline, 3))
                          for (ds, n), row in med.iterrows()})
    out["median_sd_range"] = dict(LLM=[r(med.LLM.min(), 3), r(med.LLM.max(), 3)],
                                  baseline=[r(med.baseline.min(), 3), r(med.baseline.max(), 3)],
                                  cells_LLM_lower=int((med.LLM < med.baseline).sum()), cells=len(med))
    s30 = SD[SD["size"] == 30].groupby("model").RMSE
    out["sd_N30_per_model"] = {m: [r(a, 3), r(b, 3)] for m, a, b in zip(s30.min().index, s30.min(), s30.max())}
    inv = {}
    for d in (1, 2, 3):
        X = DC[(DC.family == "LLM") & (DC.draw == d)].pivot_table(index=["dataset", "model"], columns="size", values="RMSE")
        inv[d] = {ds: int(v) for ds, v in (X[60] > X[30]).groupby("dataset").sum().items()}
    out["inversions_N60_gt_N30"] = inv
    return out


# ---------------------------------------------------------------- selective prediction
def selective():
    S = pd.read_csv(os.path.join(ICL, "selective_prediction.csv"))
    L = S[(S.family == "LLM") & (S.reasoning == "low")]
    Bs = S[S.family == "baseline"].groupby(["ds", "model", "size", "draw"], as_index=False)[["RMSE", "RMSE50", "gain50"]].mean()
    X = pd.concat([L[["ds", "model", "size", "draw", "RMSE", "RMSE50", "gain50"]], Bs]).set_index(["ds", "model", "size", "draw"])
    out = {}
    for ds in DS:
        for m in LLMS + BASELINES:
            o = {}
            for n in (30, 60, 120):
                for st in ("RMSE", "RMSE50", "gain50"):
                    v = [float(X.loc[(ds, m, n, d), st]) for d in (1, 2, 3)]
                    o[f"N{n}_{st}"] = r(v[0], 4)
                    o[f"N{n}_{st}_range"] = [r(min(v), 4), r(max(v), 4)]
                    o[f"N{n}_{st}_draws"] = [r(x, 4) for x in v]
            for st in ("RMSE", "RMSE50", "gain50"):
                o[f"top_{st}"] = r(float(X.loc[(ds, m, TOP[ds], 1), st]), 4)
            out[f"{ds}|{m}"] = o
    return out


# ---------------------------------------------------------------- GP bounds
def gp_bounds():
    out = {}
    for f in sorted(glob.glob(os.path.join(BASE, "gp_tanimoto__*__N30*.meta.json"))):
        M = json.load(open(f))
        if M.get("seed", 0) != 0:
            continue
        v = M["gp_var_y"]
        a_rel, s2_rel = M["gp_amplitude"] / v, M["gp_noise_var"] / v
        hit = []
        if np.isclose(a_rel, 1e-3, rtol=1e-3):
            hit.append("amplitude lower (trivial predictor)")
        if np.isclose(a_rel, 1e2, rtol=1e-3):
            hit.append("amplitude upper")
        if np.isclose(s2_rel, 1e-4, rtol=1e-3):
            hit.append("noise lower (interpolating)")
        if np.isclose(s2_rel, 1e1, rtol=1e-3):
            hit.append("noise upper")
        out[f"{M['dataset']}|d{M['draw']}"] = dict(a_over_var=float(f"{a_rel:.4g}"), s2_over_var=float(f"{s2_rel:.4g}"), bound=hit)
    out["n_cells"] = len([k for k in out if "|" in k])
    out["n_on_bound"] = sum(1 for k, v in out.items() if isinstance(v, dict) and v["bound"])
    out["n_trivial"] = sum(1 for k, v in out.items() if isinstance(v, dict) and any("trivial" in b for b in v["bound"]))
    return out


def gemini_zero_reasoning():
    tok = pd.concat([read_jsonl(f).reasoning_tokens
                     for f in glob.glob(os.path.join(ICL, "*__gem38flash__N*__low.jsonl"))])
    return dict(n_records=len(tok), share_zero=r((tok.fillna(0) == 0).mean(), 3))


def main():
    C = pd.read_csv(os.path.join(ICL, "cells.csv"))
    DC = pd.read_csv(os.path.join(ICL, "draws_cells.csv"))
    T = table31(C)
    r13, sig = r1_vs_r3(C)
    out = dict(costs=costs(C), table31=T, gains=gains(T), paired_top=paired_top(), pred_corr=pred_corr(),
               sonnet_text=sonnet_text(), r1_vs_r3=r13, sigma_distinct=sig, esol_subsets=esol_subsets(C),
               draws=draws(DC), selective=selective(), gp_bounds=gp_bounds(),
               gemini_zero_reasoning=gemini_zero_reasoning())
    path = os.path.join(ICL, "report_extras.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
