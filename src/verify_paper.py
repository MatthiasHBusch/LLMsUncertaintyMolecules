"""Recompute every number that paper/manuscript.tex (Methods, Results, Discussion) and paper/appendix.tex
state, and print it in the order the paper states it, next to the value the paper prints.

Every value comes from the final tables in results/ and from data/splits (nothing is refitted, no API
call is made, nothing is written). A number that exists only in a design document or in code is
printed as DOC-ONLY or CODE so that it can be checked by hand. Claim ids (2.1.3, 3.2.6, E.4, ...) are
the rows of docs/CLAIMS.md.

    python src/verify_paper.py              # everything
    python src/verify_paper.py --sec 3.2    # one subsection (2.1 ... 2.6, 3.1 ... 3.4, 4, A ... H)
    python src/verify_paper.py --sec 3      # one section

Conventions of the paper (Sections 2.3-2.5): language models at reasoning `low`; values are context
draw 1; ranges [a-b] are min-max over the three context draws at N <= 120; baselines are averaged over
the seeds of a draw, except in the paired bootstrap (seed 0); the ensembles (C.2) have seed-0 files only, and
the ESOL recall split (3.4.3-3.4.4) still reads calibration_cells.csv, whose TabPFN rows are seed 0.
Review round 2026-10-28: the calibration analysis left the paper (rows 2.5.4, 2.5.5, 3.3.6-3.3.8, D.2 and
the old E.1-E.5 removed); Appendix E is now "Repeat spread and similarity to the context" (E.1-E.4).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
ICL = os.path.join(RES, "icl")
BASE = os.path.join(RES, "baselines")
SPLITS = os.path.join(ROOT, "data", "splits")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASETS = ["lipophilicity", "bace", "antiviral", "esol"]
CLEAN = ["lipophilicity", "bace", "antiviral"]
TOP = {"lipophilicity": 1000, "bace": 1000, "antiviral": 844, "esol": 976}
LLMS = ["haiku55", "sonnet55", "gpt61sol", "gpt6luna", "gpt6astra", "gem38flash", "glm53flash"]
NAMES = {"haiku55": "Haiku 5.5", "sonnet55": "Sonnet 5.5", "gpt61sol": "GPT-6.1 Sol", "gpt6luna": "GPT-6 Luna",
         "gpt6astra": "GPT-6 Astra", "gem38flash": "Gemini 3.8 Flash", "glm53flash": "GLM 5.3 Flash",
         "tabpfn": "TabPFN", "gp_tanimoto": "GP", "rf_desc": "RF (descriptors)", "rf": "RF (Morgan)",
         "chemprop_ens": "Chemprop ens.", "mlp_ens": "MLP ens.", "knn_tanimoto": "kNN", "trivial": "trivial"}
BASELINES = ["tabpfn", "gp_tanimoto", "rf_desc", "rf", "chemprop_ens", "mlp_ens", "knn_tanimoto", "trivial"]
DISPLAY = ["tabpfn", "gp_tanimoto", "rf_desc", "chemprop_ens"]

LOG = []          # (id, status, text) for the summary


# ----------------------------------------------------------------------------------------------- output
def hdr(s):
    print(f"\n{'=' * 100}\n{s}\n{'=' * 100}")


def chk(cid, what, stated, computed, src=""):
    """stated and computed are the strings as the paper prints them; equal -> OK."""
    ok = str(stated).strip() == str(computed).strip()
    st = "OK" if ok else "DIFF"
    LOG.append((cid, st, what))
    print(f"[{cid:7s}] {st:4s} {what}\n           paper: {stated}\n           data:  {computed}"
          + (f"\n           src:   {src}" if src else ""))


def info(cid, what, computed, src=""):
    """a number printed for the reader (comment-only or context), not compared."""
    LOG.append((cid, "INFO", what))
    print(f"[{cid:7s}] INFO {what}: {computed}" + (f"\n           src:   {src}" if src else ""))


def doc(cid, what, stated, where):
    LOG.append((cid, "DOC-ONLY", what))
    print(f"[{cid:7s}] DOC-ONLY {what}: {stated}\n           only in: {where}")


def code(cid, what, stated, computed, where):
    ok = str(stated).strip() == str(computed).strip()
    LOG.append((cid, "CODE" if ok else "DIFF", what))
    print(f"[{cid:7s}] {'CODE' if ok else 'DIFF'} {what}\n           paper: {stated}\n           code:  {computed}"
          f"\n           src:   {where}")


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def sgn(x, d=2):
    return f"{x:+.{d}f}"


def ci(est, lo, hi, d=2):
    return f"{sgn(est, d)} [{sgn(lo, d)}, {sgn(hi, d)}]"


def rng(vals, d=2):
    """draw-1 value and [min-max] over draws, as the paper prints it."""
    v1, lo, hi = vals[0], min(vals), max(vals)
    return f"{v1:.{d}f} [{lo:.{d}f}-{hi:.{d}f}]"


def grep_const(path, name):
    m = re.search(rf"^{name}\s*=\s*([^#\n]+)", open(os.path.join(ROOT, path), encoding="utf-8").read(), re.M)
    return m.group(1).strip() if m else "not found"


# ----------------------------------------------------------------------------------------------- data
_cache = {}


def cells():
    if "cells" not in _cache:
        _cache["cells"] = pd.read_csv(os.path.join(ICL, "cells.csv"))
    return _cache["cells"]


def low(draw=None):
    c = cells()
    c = c[c.reasoning == "low"]
    return c if draw is None else c[c.draw == draw]


def base_summary():
    if "base" not in _cache:
        _cache["base"] = pd.read_csv(os.path.join(BASE, "summary.csv"))
    return _cache["base"]


def base_mean():
    """baselines averaged over the seeds of a draw (rule of the figures and tables)."""
    b = base_summary()
    return b.groupby(["dataset", "method", "size", "draw"], as_index=False).mean(numeric_only=True)


def sel():
    if "sel" not in _cache:
        s = pd.read_csv(os.path.join(ICL, "selective_prediction.csv")).rename(columns={"ds": "dataset"})
        _cache["sel"] = s
    return _cache["sel"]


def sel_llm(draw=1, reasoning="low"):
    s = sel()
    s = s[(s.family == "LLM") & (s.reasoning == reasoning)]
    return s if draw is None else s[s.draw == draw]


def sel_base(draw=1):
    s = sel()
    s = s[s.family == "baseline"].groupby(["dataset", "model", "size", "draw"], as_index=False).mean(numeric_only=True)
    return s if draw is None else s[s.draw == draw]


def boot():
    if "boot" not in _cache:
        _cache["boot"] = pd.read_csv(os.path.join(ICL, "bootstrap_cells.csv"))
    return _cache["boot"]


def bdiff(stat, baseline, ds, size, draw):
    b = boot()
    return b[(b.stat == stat) & (b.baseline == baseline) & (b.dataset == ds) & (b["size"] == size) & (b.draw == draw)]


def bsingle(stat, ds, size, draw):
    b = boot()
    return b[b.baseline.isna() & (b.stat == stat) & (b.dataset == ds) & (b["size"] == size) & (b.draw == draw)]


def cal():
    if "cal" not in _cache:
        _cache["cal"] = pd.read_csv(os.path.join(ICL, "calibration_cells.csv"))
    return _cache["cal"]


def calboot():
    return pd.read_csv(os.path.join(ICL, "calibration_bootstrap.csv"))


def ladder():
    if "lad" not in _cache:
        l = pd.read_csv(os.path.join(ICL, "ladder_cells.csv"))
        l = l[l.model != "gpt6astra"]          # only `low` in the file; not in the ladder (Section 2.6)
        s = sel()
        s = s[(s.family == "LLM") & (s.draw == 1)][["dataset", "model", "size", "reasoning", "rho"]]
        s = s.rename(columns={"reasoning": "effort"})
        _cache["lad"] = l.merge(s, on=["dataset", "model", "size", "effort"], how="left")
    return _cache["lad"]


def esol_subsets():
    """ESOL, language models at `low`, draw 1, from results/icl/esol_permolecule.csv: per (model, size, subset)
    n, RMSE, mean stated sigma and the RMS of e / sigma (sigma = uc1 floored at 0.01); subsets recalled /
    never_recalled (zero-shot flag True / False; unscorable labels in neither) and all."""
    if "esol_sub" not in _cache:
        P = pd.read_csv(os.path.join(ICL, "esol_permolecule.csv"), dtype={"mol_id": str})
        P = P[(P.reasoning == "low") & (P.draw == 1)]
        fl = P.recalled_zeroshot.astype(str).str.lower()
        rows = []
        for sub, mask in (("recalled", fl == "true"), ("never_recalled", fl == "false"), ("all", fl == fl)):
            for (m, n), g in P[mask].groupby(["model", "size"]):
                e = (g.pred - g.truth).values
                sig = np.clip(g.uc1.values, 0.01, None)
                rows.append(dict(model=m, size=int(n), subset=sub, n=len(g), RMSE=float(np.sqrt(np.mean(e ** 2))),
                                 mean_sigma=float(np.mean(sig)), rmsz=float(np.sqrt(np.mean((e / sig) ** 2)))))
        _cache["esol_sub"] = pd.DataFrame(rows)
    return _cache["esol_sub"]


def r2(x):
    """two decimals as the paper prints a correlation (no negative zero)."""
    t = f"{x:.2f}"
    return "0.00" if t == "-0.00" else t


def split(ds, draw=1):
    f = f"{ds}_v2.json" if draw == 1 else f"{ds}_v2_draw{draw}.json"
    return json.load(open(os.path.join(SPLITS, f), encoding="utf-8"))


def zeroshot():
    return pd.read_csv(os.path.join(RES, "contamination", "zeroshot", "cells.csv"))


def extras():
    return json.load(open(os.path.join(ICL, "report_extras.json"), encoding="utf-8"))


def rmse_q_rows(q):
    """RMSE of the q most certain molecules per cell (exact tie rule), the quantity of Fig. A3 / Section 3.3."""
    key = f"rq{q}"
    if key not in _cache:
        import make_figs_paper as mp
        _cache[key] = mp._rmse_q_rows(q)
    return _cache[key]


# =============================================================================================== METHODS
def sec2_1():
    hdr("2.1 Datasets and contamination control")
    S = {ds: split(ds) for ds in DATASETS}
    chk("2.1.1", "dataset sizes lipophilicity / BACE / antiviral / ESOL", "4,200 / 1,513 / 1,006 / 1,128",
        " / ".join(f"{S[d]['n_total']:,}" for d in DATASETS), "data/splits/<ds>_v2.json: n_total")
    chk("2.1.2", "corpus filter excluded 20 of the first 170 lipophilicity candidates, none on BACE / antiviral (Appendix B)",
        "20 of 170; 0; 0",
        f"{S['lipophilicity']['excluded_present']} of {S['lipophilicity']['positions_checked']}; "
        f"{S['bace']['excluded_present']}; {S['antiviral']['excluded_present']}", "splits: excluded_present, positions_checked")
    chk("2.1.3", "test set size", "150", ",".join(sorted({str(len(S[d]['test_ids'])) for d in DATASETS})), "splits: test_ids")
    chk("2.1.4", "training pools after dedup (Table 1a, Table A2)", "4,042 / 1,355 / 844 / 976",
        " / ".join(f"{S[d]['n_train_pool']:,}" for d in DATASETS), "splits: n_train_pool")
    chk("2.1.4b", "pool molecules removed by the InChIKey-14 rule (Table A2)", "8 / 8 / 12 / 2",
        " / ".join(str(S[d]["n_removed_dedup"]) for d in DATASETS), "splits: n_removed_dedup")
    cs = pd.read_csv(os.path.join(RES, "contamination", "corpus", "summary_by_subset.csv"))
    chk("2.1.4c", "open pre-training corpora searched", "7", str(cs["index"].nunique()), "corpus/summary_by_subset.csv: index")
    t = cs[cs.subset == "v2test150"]
    tc = t[t.dataset.isin(CLEAN)]
    mx = tc[["pct_present_verbatim", "pct_present_smiles", "pct_present_inchikey", "pct_row_cooccur"]].max().max()
    n = int(tc.groupby("dataset").n_mol.first().sum())
    chk("2.1.5", "clean test molecules present in any corpus in any form (max % over 7 indexes and 4 forms)",
        "0 % of 450", f"{mx:g} % of {n}", "corpus/summary_by_subset.csv, subset v2test150")
    e = t[t.dataset == "esol"].set_index("index")
    best = e.pct_present_smiles.idxmax()
    chk("2.1.6", "ESOL: a third of the test molecules occur as SMILES in Dolma (Appendix D)",
        "32.4 % in v4_dolma-v1_7_llama", f"{e.pct_present_smiles.max():.1f} % in {best}", "summary_by_subset.csv pct_present_smiles")
    z = zeroshot()
    zc = z[z.dataset.isin(CLEAN)]
    ze = z[z.dataset == "esol"]
    chk("2.1.7", "zero-shot screen: clean cells on the clean datasets; ESOL flagged", "21 of 21 clean; ESOL 6 of 7",
        f"{int((zc.verdict == 'clean').sum())} of {len(zc)} clean; ESOL {int((ze.verdict == 'contaminated').sum())} of {len(ze)}",
        "results/contamination/zeroshot/cells.csv: verdict")
    chk("2.1.8", "lipophilicity labels with three significant figures", "81",
        ",".join(map(str, sorted(z[z.dataset == "lipophilicity"].n_usable.unique()))), "zeroshot/cells.csv: n_usable")
    dates = {}
    for f in glob.glob(os.path.join(RES, "contamination", "zeroshot", "*__low__v2test150.json")):
        m = json.load(open(f, encoding="utf-8"))["meta"]
        dates.setdefault(m["tag"], set()).add(m["timestamp"][:10])
    other = sorted({d for k, v in dates.items() if k != "gpt6astra" for d in v})
    chk("2.1.9", "zero-shot screen dates (all models / GPT-6 Astra)", "2026-10-09 / 2026-10-13",
        f"{','.join(other)} / {','.join(sorted(dates.get('gpt6astra', [])))}", "zeroshot/*.json: meta.timestamp")
    # ESOL unit rule (decision 2026-10-15, label-sign plausibility rule): converted counts in cells.csv must
    # equal the positive raw answers on molecules with a negative label
    npos, npos_poslab = {}, 0
    for f in glob.glob(os.path.join(RES, "contamination", "zeroshot", "*__low__v2test150.json")):
        d = json.load(open(f, encoding="utf-8"))
        npos[(d["meta"]["dataset"], d["meta"]["tag"])] = sum(1 for c in d["calls"]
                                                             if c["value"] is not None and c["value"] > 0
                                                             and (d["meta"]["dataset"] != "esol" or c["truth"] < 0))
        if d["meta"]["dataset"] == "esol":
            npos_poslab += sum(1 for c in d["calls"] if c["value"] is not None and c["value"] > 0 and c["truth"] >= 0)
    zi = z.set_index(["dataset", "model"]).n_unit_converted
    same = all(int(zi[k]) == (v if k[0] == "esol" else 0) for k, v in npos.items())
    ec = ze.set_index("model").n_unit_converted
    oth = ec.drop(["sonnet55", "haiku55"])
    chk("2.1.10", "ESOL answers converted from mol/L (positive answer, negative label): Sonnet 5.5 / Haiku 5.5; other models; clean datasets",
        "41 / 30; 0 to 5; 0",
        f"{int(ec['sonnet55'])} / {int(ec['haiku55'])}; {int(oth.min())} to {int(oth.max())}; {int(zc.n_unit_converted.sum())}",
        f"zeroshot/cells.csv n_unit_converted; equals the positive raw answers on negative labels of the *.json (ESOL) / 0 (clean): {same}; "
        f"positive answers on ESOL labels >= 0 kept as stated: {npos_poslab}; "
        f"positive raw answers on the clean datasets (no rule): {sum(v for k, v in npos.items() if k[0] != 'esol')}")


def sec2_2():
    hdr("2.2 In-context protocol and model panel")
    doc("2.2.1", "labels printed with two decimals; 1-sigma definition '68 %'; JSON keys", "two decimals",
        "src/icl_common.py (verbatim prompt in Appendix A)")
    code("2.2.2", "stated sigma floored at 0.01 for scoring", "0.01", grep_const("src/analyze_icl.py", "SIGMA_FLOOR"),
         "src/analyze_icl.py: SIGMA_FLOOR")
    chk("2.2.4", "seven panel models in the main run", "7", str(low(1).model.nunique()), "cells.csv")
    reg = json.load(open(os.path.join(ROOT, "src", "registry", "models_v2.json"), encoding="utf-8"))["models"]
    pinned = [m for m in reg if m["tag"] in LLMS]
    chk("2.2.5", "panel models pinned without fallback", "7 of 7",
        f"{sum((not m['allow_fallbacks']) and bool(m['providers']) for m in pinned)} of {len(pinned)}",
        "src/registry/models_v2.json: providers, allow_fallbacks")
    chk("2.2.6", "flex tier for the OpenAI and Google models", "gem38flash,gpt61sol,gpt6astra,gpt6luna",
        ",".join(sorted(m["tag"] for m in pinned if m.get("service_tier") == "flex")), "models_v2.json: service_tier")
    code("2.2.7", "completion cap", "32000", re.search(r"max_tokens::Int=(\d+)", open(os.path.join(ROOT, "src", "julia", "Common.jl"),
         encoding="utf-8").read()).group(1), "src/julia/Common.jl: max_tokens default")
    doc("2.2.8", "temperature and top-p not sent (provider default)", "not sent", "docs/EXPERIMENT_DESIGN_v2.md section 4")
    out = {}
    for f in ["meta/diagnostics/icl_probe_2026-10-08_3.jsonl", "meta/diagnostics/icl_probe_2026-10-08_4.jsonl",
              "meta/icl_probe_2026-10-08.jsonl", "meta/icl_probe_2026-10-08_2.jsonl"]:
        for line in open(os.path.join(RES, f), encoding="utf-8"):
            r = json.loads(line)
            if r.get("tag") == "opus55":
                k = int(r["size"])
                out.setdefault(k, []).append(f"{r.get('finish')}/{'parsed' if r.get('parse_ok') else 'no answer'}")
    chk("2.2.9", "Claude Opus 5.5: N = 30 answered; content filter in every call at N = 250 and 1,000 (also Appendix B)",
        "30: 2 stop/parsed; 250: 2 content_filter/no answer; 1000: 4 content_filter/no answer",
        "; ".join(f"{k}: " + ", ".join(f"{v.count(x)} {x}" for x in sorted(set(v))) for k, v in sorted(out.items())),
        "results/meta/(diagnostics/)icl_probe_2026-10-08*.jsonl: finish, parse_ok")


def sec2_3():
    hdr("2.3 Context sizes, repeats and context draws")
    sz = {ds: sorted({int(n) for n in low(1)[low(1).dataset == ds]["size"].unique()}) for ds in DATASETS}
    chk("2.3.1", "context sizes 30, 60, 120, 250, 500, largest (1,000 / 1,000 / 844 / 976)",
        "30,60,120,250,500 + 1000/1000/844/976",
        ",".join(map(str, sz["lipophilicity"][:5])) + " + " + "/".join(str(sz[d][-1]) for d in DATASETS), "cells.csv size")
    c = low()
    reps = c[c.draw == 1].n_calls / c[c.draw == 1].n_mol
    info("2.3.2", "calls per molecule in the main run (R = 3)", f"median {reps.median():.2f}, min {reps.min():.2f}")
    ov = pd.read_csv(os.path.join(RES, "meta", "context_draw_overlap.csv"))
    i = ov.overlap_first120.idxmax()
    chk("2.3.3", "first 120 molecules of the draws share at most 20 (Appendix B, Table A2)",
        "20 (expected 17.1)", f"{ov.overlap_first120.max()} (expected {ov.loc[i, 'expected_first120']:.1f})",
        "results/meta/context_draw_overlap.csv")
    m1, m23 = c[c.draw == 1], c[c.draw > 1]
    chk("2.3.4", "cells of the main run / of the context draws (Appendix E)", "168 / 168", f"{len(m1)} / {len(m23)}", "cells.csv reasoning low")
    chk("2.3.5", "main run: every cell answers 150 molecules; parse rate >= 0.995 (Appendix A); no truncation",
        "150; >= 0.995; 0", f"{int(m1.n_mol.min())}; >= {np.floor(m1.parse_rate.min() * 1000) / 1000:.3f}; {m1.trunc_rate.max():g}",
        "cells.csv n_mol, parse_rate (min %.4f), trunc_rate" % m1.parse_rate.min())
    info("2.3.5b", "context draws: parse rate min; cells with 149 of 150 molecules",
         f"{m23.parse_rate.min():.4f}; {int((m23.n_mol < 150).sum())}")


def sec2_4():
    hdr("2.4 Classical baselines")
    b = base_summary()
    chk("2.4.1", "eight classical methods", "8", str(b.method.nunique()), "baselines/summary.csv")
    nf = [json.load(open(f, encoding="utf-8"))["n_features"] for f in glob.glob(os.path.join(BASE, "tabpfn__*.meta.json"))]
    chk("2.4.2", "TabPFN descriptors per cell after the filter (Table A3)", "159 to 201", f"{min(nf)} to {max(nf)}", "tabpfn__*.meta.json: n_features")
    sd_def = {json.load(open(f, encoding="utf-8")).get("sigma_definition") for f in glob.glob(os.path.join(BASE, "tabpfn__*.meta.json"))}
    chk("2.4.2b", "TabPFN sigma = half the 16-84 percentile distance", "(q84 - q16) / 2 of the predictive distribution",
        " | ".join(sorted(sd_def)), "tabpfn__*.meta.json: sigma_definition")
    mj = lambda m: json.load(open(sorted(glob.glob(os.path.join(BASE, f"{m}__*.meta.json")))[0], encoding="utf-8"))
    chk("2.4.3", "random forests: trees; Morgan radius / bits (Table A3)", "200; morgan_r2_2048",
        f"{mj('rf_desc')['n_trees']}; {mj('rf')['features']}", "rf*__*.meta.json")
    chk("2.4.4", "GP: marginal likelihood maximised from 14 starts (Appendix C)", "14", str(len(mj("gp_tanimoto")["gp_starts"])), "gp_tanimoto__*.meta.json")
    chk("2.4.5", "kNN k (Table A3)", "5", str(mj("knn_tanimoto")["k"]), "knn_tanimoto__*.meta.json")
    chk("2.4.6", "Chemprop / MLP ensembles of five members", "5 / 5",
        f"{mj('chemprop_ens')['n_members']} / {len(mj('mlp_ens')['member_seeds'])}", "*_ens__*.meta.json")
    seeds = b.groupby(["method", "draw"]).seed.apply(lambda s: ",".join(map(str, sorted(s.unique())))).unstack()
    three = sorted(m for m in seeds.index if seeds.loc[m, 1] == "0,1,2")
    three23 = sorted(m for m in seeds.index if seeds.loc[m].get(2) == "0,1,2")
    chk("2.4.7", "three seeds in draw 1 for TabPFN, both RF, GP, kNN; draws 2-3 GP, kNN (Table A3)",
        "gp_tanimoto,knn_tanimoto,rf,rf_desc,tabpfn | gp_tanimoto,knn_tanimoto",
        f"{','.join(three)} | {','.join(three23)}", "summary.csv seed by method and draw")
    v = {json.load(open(f, encoding="utf-8")).get("tabpfn") for f in glob.glob(os.path.join(BASE, "tabpfn__*.meta.json"))}
    w = {json.load(open(f, encoding="utf-8")).get("chemprop") for f in glob.glob(os.path.join(BASE, "chemprop_ens__*.meta.json"))}
    chk("2.4.9", "library versions tabpfn / chemprop (Appendix C, H, data availability)", "9.1.0 / 2.0.0", f"{','.join(sorted(v))} / {','.join(sorted(w))}",
        "*.meta.json")
    doc("2.4.10", "TabPFN v3.5 trained on synthetic tasks only (vendor statement)", "synthetic only", "docs/EXPERIMENT_DESIGN_v2.md section 7")
    doc("2.4.11", "leakage self-check (arbitrary labels leave predictions bit-identical; Appendix C)", "bit-identical", "src/baselines/selfcheck.py (run it)")


def sec2_5():
    hdr("2.5 Metrics and statistics")
    code("2.5.1", "ties determined after rounding sigma to 10 decimals", "10", grep_const("src/selective_prediction.py", "TIE_DECIMALS"),
         "src/selective_prediction.py: TIE_DECIMALS")
    s = sel_llm(1)
    chk("2.5.3", "median distinct stated sigmas per main-run cell", "30", f"{s.n_sigma_distinct.median():.0f}",
        f"selective_prediction.csv n_sigma_distinct, low, draw 1, {len(s)} cells")
    e = low(1)
    e = e[e.dataset == "esol"]
    chk("2.5.6", "ESOL recalled / never recalled / unscorable (Section 3.4, Appendix D)", "63 / 50 / 37",
        f"{int(e.n_recalled.iloc[0])} / {int(e.n_never_recalled.iloc[0])} / {int(e.n_unscorable_zeroshot.iloc[0])}", "cells.csv")
    code("2.6.1", "bootstrap resamples and seed", "1000, 0", grep_const("src/bootstrap_cells.py", "B, BOOT_SEED"), "src/bootstrap_cells.py")
    b = boot()
    chk("2.6.2", "differences against TabPFN and the GP", "gp_tanimoto,tabpfn", ",".join(sorted(b.baseline.dropna().unique())), "bootstrap_cells.csv")
    chk("2.6.3", "draws in the bootstrap at N <= 120 (0 = combined) / at the largest N",
        "0,1,2,3 / 1", f"{','.join(map(str, sorted(b[b['size'] <= 120].draw.unique())))} / "
        f"{','.join(map(str, sorted(b[b['size'] > 120].draw.unique())))}", "bootstrap_cells.csv")


def sec2_6():
    hdr("2.6 Reasoning ladder")
    l = pd.read_csv(os.path.join(ICL, "ladder_cells.csv"))
    chk("2.7.1", "ladder models (GPT-6 Astra not in it)", "6",
        str(l[l.effort != "low"].model.nunique()), "ladder_cells.csv effort != low")
    has_none = sorted(l[l.effort == "none"].model.unique())
    no_none = sorted(set(l[l.effort != "low"].model) - set(has_none))
    chk("2.7.2", "effort none rejected by four of the six ladder models; accepted by Haiku, Luna",
        "gem38flash,glm53flash,gpt61sol,sonnet55 | gpt6luna,haiku55", f"{','.join(no_none)} | {','.join(has_none)}", "ladder_cells.csv")
    c = cells()
    lad = c[c.reasoning != "low"]
    chk("2.7.3", "ladder cells beyond the main run (Appendix G)", "160", str(len(lad)), "cells.csv reasoning != low")
    m1 = low(1)
    main_cost = m1.cost_usd.sum()
    edge = lambda d: d[(d["size"] == 30) | (d["size"] == d.dataset.map(TOP))]
    astra = edge(m1[m1.model == "gpt6astra"]).cost_usd.sum()
    ratio = {}
    for m in ("gpt6luna", "gpt61sol"):
        other = lad[(lad.model == m) & (lad.reasoning != "none")].cost_usd.sum()   # Astra: minimal, medium, high
        ratio[m] = other / edge(m1[m1.model == m]).cost_usd.sum()
    proj = (astra * min(ratio.values()), astra * max(ratio.values()))
    chk("2.7.4", "an Astra ladder would have cost at least as much as the main run of all seven models (Appendix G: 4.8 / 5.9 x; USD 315-393; main run 311.02)",
        "yes; 4.8 / 6.0; 315-393; 311.02",
        f"{'yes' if proj[0] >= main_cost else 'no'}; {ratio['gpt6luna']:.1f} / {ratio['gpt61sol']:.1f}; {proj[0]:.0f}-{proj[1]:.0f}; {main_cost:.2f}",
        f"cells.csv cost_usd: Astra step-low cells USD {astra:.2f}")
    gh = l[(l.model == "gem38flash") & (l.effort == "high") & l.dataset.isin(CLEAN)]
    chk("3.1.12", "Gemini high on the clean datasets: truncated calls; molecules with a parsed answer (Section 2.6, 3.1, Fig. 2)",
        "15 to 35 %; 126 to 150",
        f"{100 * gh.trunc_rate.min():.0f} to {100 * gh.trunc_rate.max():.0f} %; {gh.n_mol.min()} to {gh.n_mol.max()}", "ladder_cells.csv")


# =============================================================================================== RESULTS
def sec3_1():
    hdr("3.1 Q1: accuracy over context size and reasoning effort")
    c = low()
    d1 = c[c.draw == 1]
    g = []
    for (ds, m), x in d1[d1.dataset.isin(CLEAN)].groupby(["dataset", "model"]):
        x = x.set_index("size").RMSE
        g.append((1 - x[TOP[ds]] / x[30], ds, m))
    g.sort()
    chk("3.1.1", "RMSE falls from N = 30 to the largest N by 8 to 37 % (draw 1, clean)", "8 to 37",
        f"{100 * g[0][0]:.0f} to {100 * g[-1][0]:.0f}", f"cells.csv; min {g[0][1]}/{g[0][2]}, max {g[-1][1]}/{g[-1][2]}")
    inv = []
    for dr in (1, 2, 3):
        p = c[(c.dataset == "antiviral") & (c.draw == dr)].pivot_table(index="model", columns="size", values="RMSE")
        inv.append(int((p[60] > p[30]).sum()))
    chk("3.1.2", "antiviral: models with RMSE(60) > RMSE(30) in draws 1 / 2 / 3", "7 / 0 / 0", " / ".join(map(str, inv)), "cells.csv")
    bm = base_mean()
    wins, best = 0, set()
    for ds in DATASETS:
        for dr in (1, 2, 3):
            L = c[(c.dataset == ds) & (c["size"] == 30) & (c.draw == dr)]
            B = bm[(bm.dataset == ds) & (bm["size"] == 30) & (bm.draw == dr)]
            r = L.loc[L.RMSE.idxmin()]
            best.add(r.model)
            wins += int(r.RMSE < B.RMSE.min())
    chk("3.1.3", "N = 30: Sonnet is the most accurate LLM and below every baseline, every dataset, every draw",
        "sonnet55; 12 of 12", f"{','.join(sorted(best))}; {wins} of 12", "cells.csv, summary.csv (8 methods, seed mean)")
    son = [c[(c.dataset == "lipophilicity") & (c.model == "sonnet55") & (c["size"] == 30) & (c.draw == d)].RMSE.iloc[0] for d in (1, 2, 3)]
    tab = [bm[(bm.dataset == "lipophilicity") & (bm.method == "tabpfn") & (bm["size"] == 30) & (bm.draw == d)].RMSE.iloc[0] for d in (1, 2, 3)]
    chk("3.1.4", "lipophilicity N = 30: Sonnet vs TabPFN", "0.60 [0.60-0.60] vs 1.17 [1.05-1.17]",
        f"{rng(son)} vs {rng(tab)}", "cells.csv; summary.csv seed mean")
    out, only = [], []
    for ds in CLEAN:
        x = bdiff("RMSE", "tabpfn", ds, 30, 0)
        out.append(str(int((x.hi < 0).sum())))
        if ds == "antiviral":
            only = sorted(x[x.hi < 0].model)
    chk("3.1.5", "N = 30 combined, RMSE diff to TabPFN excludes 0 (lipo / BACE / antiviral); antiviral only Sonnet",
        "6 / 3 / 1; sonnet55", f"{' / '.join(out)}; {','.join(only)}", "bootstrap_cells.csv RMSE tabpfn 30 draw 0")
    cb = calboot().set_index(["dataset", "metric"])
    r = cb.loc[("lipophilicity", "RMSE")]
    chk("3.1.6", "largest N lipophilicity Sonnet - TabPFN RMSE", "-0.14 [-0.21, -0.07]", ci(r["diff"], r.ci_lo, r.ci_hi, 2),
        "calibration_bootstrap.csv")
    r = cb.loc[("bace", "RMSE")]
    chk("3.1.7", "largest N BACE: not distinguishable", "CI includes 0", "CI includes 0" if not r.excludes_zero else "excludes 0",
        f"{ci(r['diff'], r.ci_lo, r.ci_hi, 3)}")
    r1, r2 = cb.loc[("antiviral", "RMSE")], cb.loc[("antiviral", "mean_abs_err")]
    chk("3.1.8", "largest N antiviral: RMSE diff excludes 0, MAE diff does not", "True / False",
        f"{bool(r1.excludes_zero)} / {bool(r2.excludes_zero)}",
        f"RMSE {ci(r1['diff'], r1.ci_lo, r1.ci_hi, 3)}, MAE {ci(r2['diff'], r2.ci_lo, r2.ci_hi, 3)}")
    # ranking at the largest N; Luna and GLM behind TabPFN
    top3 = []
    for ds in CLEAN:
        x = d1[(d1.dataset == ds) & (d1["size"] == TOP[ds])].set_index("model").RMSE.sort_values()
        top3.append(set(x.index[:3]))
    both = all({"sonnet55", "gpt6astra"} <= t for t in top3)
    behind = {}
    for ds in CLEAN:
        tb = bm[(bm.dataset == ds) & (bm.method == "tabpfn") & (bm.draw == 1)].set_index("size").RMSE
        first = []
        for m in ("gpt6luna", "glm53flash"):
            x = d1[(d1.dataset == ds) & (d1.model == m)].set_index("size").RMSE
            n_behind = [n for n in sorted(x.index) if x[n] > tb[n]]
            # first N from which the model stays behind TabPFN at every larger N
            ns = sorted(x.index)
            start = next((n for k, n in enumerate(ns) if all(x[q] > tb[q] for q in ns[k:])), None)
            first.append(start)
        behind[ds] = first
    nb, nt = 0, 0
    for ds in ("bace", "antiviral"):
        tb = bm[(bm.dataset == ds) & (bm.method == "tabpfn") & (bm.draw == 1)].set_index("size").RMSE
        for m in ("gpt6luna", "glm53flash"):
            x = d1[(d1.dataset == ds) & (d1.model == m)].set_index("size").RMSE
            nb += int((x > tb.reindex(x.index)).sum())
            nt += len(x)
    chk("3.1.9", "largest N: Sonnet and Astra among the three most accurate on each clean set; Luna / GLM less accurate than TabPFN in most BACE + antiviral cells; on lipophilicity from N = 500",
        "True; most (22 of 24); [500, 500]",
        f"{both}; {'most' if nb > nt / 2 else 'not most'} ({nb} of {nt}); {behind['lipophilicity']}",
        "cells.csv draw 1 vs summary.csv seed mean; top-3 sets " + "; ".join(f"{ds}: {sorted(t)}" for ds, t in zip(CLEAN, top3))
        + f"; first N behind at every larger N: bace {behind['bace']}, antiviral {behind['antiviral']}")
    l = ladder()
    son_t = l[l.model == "sonnet55"].median_reasoning_tokens
    hk = l[l.model == "haiku55"]
    hk0 = hk[hk.effort.isin(["none", "minimal", "low"])].median_reasoning_tokens.max()
    hk1 = hk[hk.effort.isin(["medium", "high"])].median_reasoning_tokens.min()
    gm = l[(l.model == "gem38flash") & (l.effort == "high")].median_reasoning_tokens.max()
    chk("3.1.10", "Sonnet 0 tokens at every level; Haiku 0 below medium, > 0 from medium; Gemini high max",
        "0; 0 / > 0; 30,717", f"{son_t.max():.0f}; {hk0:.0f} / {'> 0' if hk1 > 0 else '0'}; {gm:,.0f}", "ladder_cells.csv")
    clean = l[l.dataset.isin(CLEAN)]
    P = clean.pivot_table(index=["model", "dataset", "size"], columns="effort", values=["RMSE", "rho"])
    d = (P[("RMSE", "high")] - P[("RMSE", "low")]).dropna()
    lu = l[(l.model == "gpt6luna") & (l.dataset == "esol") & (l["size"] == 976)].set_index("effort").RMSE
    chk("3.1.11", "RMSE high - low over the 36 clean cells; no shared direction; Luna ESOL largest N low -> none",
        "36: -0.12 to +0.06; both signs; 0.56 -> 0.81",
        f"{len(d)}: {d.min():+.2f} to {d.max():+.2f}; {'both signs' if (d < 0).any() and (d > 0).any() else 'one sign'}; {f2(lu['low'])} -> {f2(lu['none'])}",
        f"ladder_cells.csv; min at {d.idxmin()}, max at {d.idxmax()}, median {d.median():+.3f}; lower at high in {(d < 0).sum()} of {len(d)}")
    es = d1[(d1.dataset == "esol") & (d1.model == "sonnet55")].set_index("size").RMSE
    tb = bm[(bm.dataset == "esol") & (bm.method == "tabpfn") & (bm.draw == 1)].set_index("size").RMSE
    chk("3.1.13", "ESOL flat: Sonnet N = 30 -> 976; TabPFN", "0.30 -> 0.29; 0.97 -> 0.51",
        f"{f2(es[30])} -> {f2(es[976])}; {f2(tb[30])} -> {f2(tb[976])}", "cells.csv; summary.csv seed mean")


def sec3_2():
    hdr("3.2 Q2: which uncertainty flags the molecules a model gets wrong")
    s = sel_llm(1)
    sc = s[s.dataset.isin(CLEAN)]
    sb = sel_base(1)
    tb = sb[(sb.model == "tabpfn") & sb.dataset.isin(CLEAN)]
    chk("3.2.1", "median rho over 126 clean LLM cells vs 18 clean TabPFN cells", "126: 0.31 vs 18: 0.36",
        f"{len(sc)}: {f2(sc.rho.median())} vs {len(tb)}: {f2(tb.rho.median())}", "selective_prediction.csv rho, draw 1")
    i = sc.rho.idxmax()
    chk("3.2.2", "no LLM cell on the clean datasets exceeds 0.46", "0.46", f2(sc.rho.max()), f"{sc.loc[i, 'dataset']} {sc.loc[i, 'model']} N = {sc.loc[i, 'size']}")
    son = [sel_llm(d)[(sel_llm(d).dataset == "lipophilicity") & (sel_llm(d).model == "sonnet55") & (sel_llm(d)["size"] == 30)].rho.iloc[0] for d in (1, 2, 3)]
    tab = [sel_base(d)[(sel_base(d).dataset == "lipophilicity") & (sel_base(d).model == "tabpfn") & (sel_base(d)["size"] == 30)].rho.iloc[0] for d in (1, 2, 3)]
    x = bdiff("rho", "tabpfn", "lipophilicity", 30, 0)
    L = s[(s.dataset == "lipophilicity") & (s["size"] == 30)]
    B = sb[(sb.dataset == "lipophilicity") & (sb["size"] == 30)]
    chk("3.2.3", "lipophilicity N = 30: Sonnet rho vs TabPFN; combined diff excludes 0 for; LLMs above every baseline (draw 1)",
        "0.36 [0.36-0.40] vs 0.13 [0.10-0.15]; gpt6astra,sonnet55; 5 of 7",
        f"{rng(son)} vs {rng(tab)}; {','.join(sorted(x[x.lo > 0].model))}; {int((L.rho > B.rho.max()).sum())} of 7",
        "selective_prediction.csv; bootstrap_cells.csv rho tabpfn lipophilicity 30 draw 0: " +
        ", ".join(f"{m} {ci(r.est, r.lo, r.hi)}" for m, r in x.set_index('model').loc[['sonnet55', 'gpt6astra']].iterrows()))
    av = bdiff("rho", "tabpfn", "antiviral", 30, 0)
    bt = bdiff("rho", "tabpfn", "bace", 30, 0)
    bg = bdiff("rho", "gp_tanimoto", "bace", 30, 0)
    chk("3.2.4", "N = 30 combined: BACE LLMs better than TabPFN / GP (lo > 0); antiviral TabPFN better than (hi < 0)",
        "0 / 0; 4", f"{int((bt.lo > 0).sum())} / {int((bg.lo > 0).sum())}; {int((av.hi < 0).sum())}", "bootstrap_cells.csv rho")
    tl = tb[tb.dataset == "lipophilicity"].set_index("size").rho
    rise = []
    for ds in CLEAN:
        med = sc[sc.dataset == ds].groupby("size").rho.median()
        rise.append(med[TOP[ds]] - med[30])
    chk("3.2.5", "TabPFN lipophilicity rho N = 30 -> 1,000; LLM median rho at the largest N not above that at N = 30 by more than 0.05 (each clean set)",
        "0.13 -> 0.38; True", f"{f2(tl[30])} -> {f2(tl[1000])}; {bool(max(rise) <= 0.05)}",
        "selective_prediction.csv tabpfn seed mean; LLM median rise " + ", ".join(f"{ds} {r:+.2f}" for ds, r in zip(CLEAN, rise)))
    tv, lm = [], []
    for ds in ("lipophilicity", "antiviral"):
        t = tb[tb.dataset == ds].set_index("size").rho
        med = sc[sc.dataset == ds].groupby("size").rho.median()
        for n in (250, 500, TOP[ds]):
            tv.append(t[n])
            lm.append(med[n])
    tbb = tb[tb.dataset == "bace"].set_index("size").rho
    medb = sc[sc.dataset == "bace"].groupby("size").rho.median()
    bl = [medb[n] for n in (250, 500, 1000)]
    bt_ = [tbb[n] for n in (250, 500, 1000)]
    chk("3.2.6", "N >= 250: TabPFN rho lipophilicity + antiviral range vs LLM medians; BACE LLM medians vs TabPFN",
        "0.37-0.44 vs 0.26-0.38; 0.31-0.34 vs 0.25-0.27",
        f"{min(tv):.2f}-{max(tv):.2f} vs {min(lm):.2f}-{max(lm):.2f}; {min(bl):.2f}-{max(bl):.2f} vs {min(bt_):.2f}-{max(bt_):.2f}",
        f"selective_prediction.csv draw 1; TabPFN above the LLM median in {sum(t > m for t, m in zip(tv, lm))} of 6 lipophilicity/antiviral cells, "
        f"LLM median above TabPFN in {sum(m > t for t, m in zip(bt_, bl))} of 3 BACE cells")
    hi_, lo_ = [], []
    for ds in CLEAN:
        x = bdiff("rho", "tabpfn", ds, TOP[ds], 1)
        hi_ += [f"{ds}/{m}" for m in x[x.hi < 0].model]
        lo_ += [f"{ds}/{m}" for m in x[x.lo > 0].model]
    chk("3.2.7", "largest N: rho diff to TabPFN excludes 0 in TabPFN's favour only for Haiku on lipophilicity; in an LLM's favour never",
        "lipophilicity/haiku55; none", f"{','.join(hi_) or 'none'}; {','.join(lo_) or 'none'}", "bootstrap_cells.csv rho tabpfn, largest N, draw 1")
    K = cal()
    u1 = K[(K.signal == "UC1") & (K.subset == "all")].set_index(["dataset", "model", "size"])
    u3 = K[(K.signal == "UC3") & (K.subset == "all")].set_index(["dataset", "model", "size"])
    better = (u3.rho_sigma_err > u1.rho_sigma_err)
    nclean = int(better.loc[CLEAN].sum())
    chk("3.2.8", "UC3 exactly zero for up to a third of a cell; ranks better than UC1 in 8 of the 126 clean cells",
        "0.33; 8 of 126", f"{f2(u3.share_sigma_zero.max())}; {nclean} of {len(better.loc[CLEAN])}", "calibration_cells.csv UC3")
    l = ladder()
    P = l[l.dataset.isin(CLEAN)].pivot_table(index=["model", "dataset", "size"], columns="effort", values=["rho"])
    r = (P[("rho", "high")] - P[("rho", "low")]).dropna()
    chk("3.2.9", "rho high - low over 36 clean ladder cells: median", "36: +0.02", f"{len(r)}: {r.median():+.2f}",
        f"range {r.min():+.2f} to {r.max():+.2f}; up in {(r > 0).sum()} of {len(r)}")


def sec3_3():
    hdr("3.3 Q3: using the most certain predictions without measuring")
    s = sel_llm(1)
    sb = sel_base(1)
    n, ok = 0, 0
    for ds in CLEAN:
        for dr in (1, 2, 3):
            Ld = sel_llm(dr)
            Ld = Ld[(Ld.dataset == ds) & (Ld["size"] == 30)]
            Bd = sel_base(dr)
            Bd = Bd[(Bd.dataset == ds) & (Bd["size"] == 30)]
            n += 1
            ok += int(Ld.RMSE50.min() < Bd.RMSE50.min())
    son = [sel_llm(d)[(sel_llm(d).dataset == "lipophilicity") & (sel_llm(d).model == "sonnet55") & (sel_llm(d)["size"] == 30)].RMSE50.iloc[0] for d in (1, 2, 3)]
    tab = [sel_base(d)[(sel_base(d).dataset == "lipophilicity") & (sel_base(d).model == "tabpfn") & (sel_base(d)["size"] == 30)].RMSE50.iloc[0] for d in (1, 2, 3)]
    gap = {}
    for ds in CLEAN:
        Ld = sel_llm(1)
        Bd = sel_base(1)
        gap[ds] = Ld[(Ld.dataset == ds) & (Ld["size"] == 30)].RMSE50.min() - Bd[(Bd.dataset == ds) & (Bd["size"] == 30)].RMSE50.min()
    chk("3.3.1", "N = 30 draw 1: best LLM minus best baseline RMSE50 (clearly ahead only on lipophilicity; BACE and antiviral within 0.06); below every baseline in every clean dataset and draw; lipophilicity Sonnet vs TabPFN",
        "-0.71; -0.01; -0.06; 9 of 9; 0.44 [0.42-0.44] vs 1.15 [0.99-1.15]",
        f"{gap['lipophilicity']:.2f}; {gap['bace']:.2f}; {gap['antiviral']:.2f}; {ok} of {n}; {rng(son)} vs {rng(tab)}",
        "selective_prediction.csv (baselines seed mean)")
    res = {ds: bdiff("gain50", "tabpfn", ds, 30, 0) for ds in CLEAN}
    lp = res["lipophilicity"]
    g1 = s[(s.dataset == "lipophilicity") & (s["size"] == 30)].set_index("model").gain50
    tg = sb[(sb.dataset == "lipophilicity") & (sb.model == "tabpfn") & (sb["size"] == 30)].gain50.iloc[0]
    mx = max(res["bace"].est.abs().max(), res["antiviral"].est.abs().max())
    g = s[(s.dataset == "antiviral") & (s["size"] >= 60)].groupby("size").gain50.median()
    gt = sb[(sb.dataset == "antiviral") & (sb.model == "tabpfn") & (sb["size"] >= 60)].set_index("size").gain50
    gl = s[(s.dataset == "lipophilicity") & (s["size"] <= 120)].groupby("size").gain50.median()
    gb = sb[(sb.dataset == "lipophilicity") & (sb["size"] <= 120) & (sb.model != "trivial")].groupby("size").gain50.median()
    chk("3.3.7", "gain50 draw 1: antiviral N >= 60 TabPFN range; LLM median per N range; lipophilicity N <= 120 LLM median >= baseline median at every N",
        "0.33-0.51; 0.16-0.32; yes",
        f"{gt.min():.2f}-{gt.max():.2f}; {g.min():.2f}-{g.max():.2f}; {'yes' if (gl.values >= gb.reindex(gl.index).values).all() else 'no'}",
        "selective_prediction.csv (baselines seed mean)")
    chk("3.3.2", "N = 30 combined gain50 diff to TabPFN excludes 0 only on lipophilicity (models); Sonnet / Haiku gain50 about 0.3 vs TabPFN 0.02; BACE / antiviral within 0.06",
        "gem38flash,haiku55,sonnet55; 0 / 0; 0.27 / 0.31 vs 0.02; <= 0.06",
        f"{','.join(sorted(lp[lp.lo > 0].model))}; {int((res['bace'].lo > 0).sum() + (res['bace'].hi < 0).sum())} / "
        f"{int((res['antiviral'].lo > 0).sum() + (res['antiviral'].hi < 0).sum())}; {f2(g1['sonnet55'])} / {f2(g1['haiku55'])} vs {f2(tg)}; "
        f"{'<= 0.06' if mx <= 0.06 else f'{mx:.3f}'}",
        f"bootstrap_cells.csv gain50 tabpfn 30 draw 0; selective_prediction.csv draw 1; max |diff| bace/antiviral {mx:.3f}")
    Lb = s[(s.dataset == "bace") & (s["size"] == 30)].set_index("model").RMSE50
    Bb = sb[(sb.dataset == "bace") & (sb["size"] == 30)].set_index("model").RMSE50
    d1x = bdiff("RMSE50", "tabpfn", "bace", 30, 1)
    d0x = bdiff("RMSE50", "tabpfn", "bace", 30, 0)
    chk("3.3.2b", "BACE draw 1: best LLM vs TabPFN RMSE50; excludes 0 in draw 1 / combined",
        "0.77 vs 0.78; 0 / 2", f"{f2(Lb.min())} vs {f2(Bb['tabpfn'])}; {int((d1x.hi < 0).sum())} / {int((d0x.hi < 0).sum())}",
        f"selective_prediction.csv (best {Lb.idxmin()}); bootstrap_cells.csv RMSE50 tabpfn bace 30")
    one = bsingle("RMSE50", "antiviral", 120, 0).set_index("model").est
    llm = one.reindex(LLMS)
    dd = bdiff("RMSE50", "tabpfn", "antiviral", 120, 0)
    t = bdiff("RMSE50", "tabpfn", "antiviral", 844, 1).set_index("model")
    top = bsingle("RMSE50", "antiviral", 844, 1).set_index("model").est
    top["tabpfn"] = sb[(sb.dataset == "antiviral") & (sb.model == "tabpfn") & (sb["size"] == 844)].RMSE50.iloc[0]   # seed mean, as the text
    chk("3.3.3", "antiviral N = 120 combined: TabPFN RMSE50 lower than every LLM; excludes 0 for; largest N TabPFN vs best LLM; CI of that difference includes 0",
        "yes; 4; 0.31 vs 0.32; includes 0",
        f"{'yes' if one['tabpfn'] < llm.min() else 'no'}; {int((dd.lo > 0).sum())}; {f2(top['tabpfn'])} vs {f2(top.reindex(LLMS).min())}; "
        f"{'includes 0' if t.loc[top.reindex(LLMS).idxmin(), 'lo'] <= 0 <= t.loc[top.reindex(LLMS).idxmin(), 'hi'] else 'excludes 0'}",
        f"bootstrap_cells.csv: N = 120 combined TabPFN {one['tabpfn']:.3f}, LLMs {llm.min():.3f}-{llm.max():.3f}, excl. {','.join(sorted(dd[dd.lo > 0].model))}; "
        f"largest N TabPFN seed mean {top['tabpfn']:.3f} (seed 0 {bsingle('RMSE50', 'antiviral', 844, 1).set_index('model').est['tabpfn']:.3f}); "
        f"largest N best LLM {top.reindex(LLMS).idxmin()} {ci(t.loc[top.reindex(LLMS).idxmin(), 'est'], t.loc[top.reindex(LLMS).idxmin(), 'lo'], t.loc[top.reindex(LLMS).idxmin(), 'hi'])}")
    every = []
    for ds in ("lipophilicity", "bace"):
        tbp = sb[(sb.dataset == ds) & (sb.model == "tabpfn")].set_index("size").RMSE50
        lm = s[s.dataset == ds].groupby("size").RMSE50.min()
        every.append(bool((lm < tbp.reindex(lm.index)).all()))
    al = s[(s.dataset == "lipophilicity") & (s["size"] == 1000) & (s.model == "gpt6astra")].RMSE50.iloc[0]
    tl = sb[(sb.dataset == "lipophilicity") & (sb["size"] == 1000) & (sb.model == "tabpfn")].RMSE50.iloc[0]
    ab = s[(s.dataset == "bace") & (s["size"] == 1000) & (s.model == "gpt6astra")].RMSE50.iloc[0]
    tbv = sb[(sb.dataset == "bace") & (sb["size"] == 1000) & (sb.model == "tabpfn")].RMSE50.iloc[0]
    xl = bdiff("RMSE50", "tabpfn", "lipophilicity", 1000, 1).set_index("model").loc["gpt6astra"]
    xb = bdiff("RMSE50", "tabpfn", "bace", 1000, 1).set_index("model").loc["gpt6astra"]
    chk("3.3.4", "lipophilicity / BACE: best LLM RMSE50 below TabPFN at every N; largest N Astra vs TabPFN; lipophilicity excludes 0, BACE includes 0",
        "True / True; 0.34 vs 0.43; 0.46 vs 0.61; excludes / includes",
        f"{every[0]} / {every[1]}; {f2(al)} vs {f2(tl)}; {f2(ab)} vs {f2(tbv)}; {'excludes' if xl.hi < 0 else 'includes'} / {'excludes' if xb.hi < 0 else 'includes'}",
        f"selective_prediction.csv; bootstrap_cells.csv RMSE50 tabpfn 1000 draw 1: lipophilicity {ci(xl.est, xl.lo, xl.hi)}, bace {ci(xb.est, xb.lo, xb.hi)}")
    rl = s[s.dataset == "lipophilicity"].pivot(index="model", columns="size", values="RMSE")
    rb = sb[sb.dataset == "lipophilicity"].pivot(index="model", columns="size", values="RMSE")
    il = 1 - rl[1000] / rl[30]
    ib = 1 - rb[1000] / rb[30]
    chk("3.1.14", "lipophilicity draw 1, 1 - RMSE(1000)/RMSE(30): LLM range; TabPFN and Chemprop range",
        "0.18-0.37; 0.46-0.48", f"{il.min():.2f}-{il.max():.2f}; {ib[['tabpfn', 'chemprop_ens']].min():.2f}-{ib[['tabpfn', 'chemprop_ens']].max():.2f}",
        "selective_prediction.csv RMSE (baselines seed mean)")
    se = sel_llm(1)
    chk("3.4.15", "rho draw 1: ESOL LLM median vs clean LLM median", "0.45; 0.31",
        f"{se[se.dataset == 'esol'].rho.median():.2f}; {se[se.dataset.isin(CLEAN)].rho.median():.2f}", "selective_prediction.csv")
    R = rmse_q_rows(0.2)
    R1 = R[R.draw == 1]
    a120 = R1[(R1.dataset == "antiviral") & (R1["size"] == 120)].set_index("model").val
    a500 = R1[(R1.dataset == "antiviral") & (R1["size"] == 500)].set_index("model").val
    ok2 = []
    for ds in ("lipophilicity", "bace"):
        for nn in sorted(R1[R1.dataset == ds]["size"].unique()):
            x = R1[(R1.dataset == ds) & (R1["size"] == nn)].set_index("model").val
            ok2.append(bool(x.reindex(LLMS).min() <= x["tabpfn"]))
    chk("3.3.5", "q = 0.2, antiviral N = 120: TabPFN vs LLM range; lipophilicity / BACE best LLM <= TabPFN at every N",
        "0.27 vs 0.43-0.54; True", f"{f2(a120['tabpfn'])} vs {a120.reindex(LLMS).min():.2f}-{a120.reindex(LLMS).max():.2f}; {all(ok2)}",
        f"make_figs_paper._rmse_q_rows(0.2), draw 1; antiviral N = 500 TabPFN {a500['tabpfn']:.2f} vs {a500.reindex(LLMS).min():.2f}-{a500.reindex(LLMS).max():.2f}")
    # 3.3.6-3.3.8 (calibration: s, held-out PICP90 / PIT-ECE, TabPFN ECE after scaling) removed with the
    # calibration analysis (review round 2026-10-28)


def sec3_4():
    hdr("3.4 Contamination")
    e = low(1)
    e = e[e.dataset == "esol"]
    x = e.pivot_table(index="model", columns="size", values="RMSE")
    drop = (x[30] - x[976])
    chk("3.4.1", "ESOL: no LLM lowers its RMSE by more than 0.08 between N = 30 and 976", "0.08", f2(drop.max()),
        f"cells.csv esol draw 1; max at {drop.idxmax()} ({x.loc[drop.idxmax(), 30]:.2f} -> {x.loc[drop.idxmax(), 976]:.2f})")
    chk("3.4.2", "recalled / never recalled", "63 / 50", f"{int(e.n_recalled.iloc[0])} / {int(e.n_never_recalled.iloc[0])}", "cells.csv")
    K = cal()
    S = K[(K.dataset == "esol") & K.signal.isin(["UC1", "sigma"]) & (K["size"] == 976)]
    R = S.pivot_table(index="model", columns="subset", values="RMSE")
    ratio = R.never_recalled / R.recalled
    z = zeroshot()
    z = z[z.dataset == "esol"].set_index("model")
    top4 = list(z.hit3.sort_values(ascending=False).index[:4])
    chk("3.4.3", "ratio never / recalled RMSE at N = 976 for the 4 highest-hit3 models",
        "gem38flash,gpt61sol,gpt6astra,sonnet55 1.9 to 2.4",
        f"{','.join(sorted(top4))} {ratio[top4].min():.1f} to {ratio[top4].max():.1f}",
        "calibration_cells.csv; zeroshot/cells.csv hit3 " + ", ".join(f"{m} {z.hit3[m]:.1f}" for m in top4))
    chk("3.4.4", "TabPFN and GLM 5.3 Flash (not flagged on ESOL): factor", "clean; 1.3; 1.3",
        f"{z.verdict['glm53flash']}; {ratio['tabpfn']:.1f}; {ratio['glm53flash']:.1f}",
        f"TabPFN {ratio['tabpfn']:.2f}, GLM {ratio['glm53flash']:.2f}")
    Q = esol_subsets()
    Sv = Q[Q["size"] == 976].pivot_table(index="model", columns="subset", values="rmsz")
    Sc = S.pivot_table(index="model", columns="subset", values="s")
    chk("3.4.5", "Sonnet N = 976 RMS of e / sigma (sigma floored at 0.01) recalled / never recalled", "0.42 / 1.27",
        f"{f2(Sv.loc['sonnet55', 'recalled'])} / {f2(Sv.loc['sonnet55', 'never_recalled'])}",
        f"esol_permolecule.csv uc1, draw 1 (n = {int(Q[(Q['size'] == 976) & (Q.model == 'sonnet55') & (Q.subset == 'recalled')].n.iloc[0])} / "
        f"{int(Q[(Q['size'] == 976) & (Q.model == 'sonnet55') & (Q.subset == 'never_recalled')].n.iloc[0])}); "
        f"= calibration_cells.csv s ({Sc.loc['sonnet55', 'recalled']:.4f} / {Sc.loc['sonnet55', 'never_recalled']:.4f})")
    # Fig. 5 (fig5_contamination) and the sentences of Section 3.4 that quote it
    z = zeroshot()
    zc = z[z.dataset != "esol"]
    chk("3.4.11", "zero-shot Spearman rho with the labels on the clean datasets (Fig. 5a)", "0.09 to 0.83",
        f"{zc.spearman.min():.2f} to {zc.spearman.max():.2f}",
        f"zeroshot/cells.csv spearman; min {zc.loc[zc.spearman.idxmin(), 'model']} {zc.loc[zc.spearman.idxmin(), 'dataset']}, "
        f"max {zc.loc[zc.spearman.idxmax(), 'model']} {zc.loc[zc.spearman.idxmax(), 'dataset']}")
    ze = z[(z.dataset == "esol") & (z.spearman >= 0.9)]
    chk("3.4.12", "ESOL: models with a zero-shot rho of at least 0.9; their range (Fig. 5a)", "7 of 7; 0.91 to 0.98",
        f"{len(ze)} of {int((z.dataset == 'esol').sum())}; {ze.spearman.min():.2f} to {ze.spearman.max():.2f}",
        "zeroshot/cells.csv spearman: " + ", ".join(sorted(ze.model)))
    nu = z.groupby("dataset").n_usable.first()
    chk("3.4.13", "labels with three significant figures scored by the screen (Fig. 5 caption)", "81 / 143 / 136 / 113",
        " / ".join(str(int(nu[d])) for d in ["lipophilicity", "bace", "antiviral", "esol"]), "zeroshot/cells.csv n_usable")
    cs = pd.read_csv(os.path.join(RES, "contamination", "corpus", "summary_by_subset.csv"))
    ne = cs[(cs.dataset == "esol") & (cs.subset == "v2test150")].n_mol
    chk("3.4.14", "ESOL test molecules with a searchable string (Fig. 5b caption)", "136", f"{int(ne.min())}",
        "corpus/summary_by_subset.csv n_mol, esol v2test150 (equal over the 7 indexes: " + str(ne.nunique() == 1) + ")")


def sec4():
    hdr("4 Discussion")
    m1 = low(1)
    top = m1[m1["size"] == m1.dataset.map(TOP)]
    lo_, hi_ = top.cost_usd.min(), top.cost_usd.max()
    chk("4.1", "cost of a cell at the largest N differs by two orders of magnitude between panel models", "yes",
        "yes" if hi_ / lo_ >= 100 else f"no (factor {hi_ / lo_:.0f})",
        f"cells.csv cost_usd, draw 1, largest N: USD {lo_:.2f} ({top.loc[top.cost_usd.idxmin(), 'model']}) to {hi_:.2f} ({top.loc[top.cost_usd.idxmax(), 'model']}), factor {hi_ / lo_:.0f}")
    bb = boot()
    rr = bb[bb.baseline.isna() & (bb.stat == "rho") & bb.model.isin(LLMS) & bb.dataset.isin(CLEAN) & (bb.draw == 1)]
    hw = (rr.hi - rr.lo) / 2
    chk("4.2", "bootstrap interval of a single rho about +-0.15 (Limitations, Appendix F)", "0.15", f2(hw.median()),
        f"bootstrap_cells.csv rho single, LLM, clean, draw 1 (n = {len(rr)}, {hw.min():.2f}-{hw.max():.2f})")


# =============================================================================================== APPENDIX
def secC():
    hdr("C Classical baselines")
    rows = []
    for f in glob.glob(os.path.join(BASE, "gp_tanimoto__*__N30*.meta.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("seed", 0) != 0:
            continue
        a, s2 = d["gp_amplitude"] / d["gp_var_y"], d["gp_noise_var"] / d["gp_var_y"]
        rows.append("noise lower" if np.isclose(s2, 1e-4, rtol=1e-3) else ("amplitude lower" if np.isclose(a, 1e-3, rtol=1e-3) else "interior"))
    chk("C.1", "GP ML-II at N = 30 on a bound in all 12 cells: noise lower / amplitude lower", "12: 8 / 4",
        f"{len(rows)}: {rows.count('noise lower')} / {rows.count('amplitude lower')}", "gp_tanimoto__*__N30*.meta.json, seed 0, draws 1-3")
    from selective_prediction import BASE_RE
    ens = {"chemprop_ens": [], "mlp_ens": []}
    for f in sorted(os.listdir(BASE)):
        mt = BASE_RE.match(f)
        if mt and mt["method"] in ens and int(mt["draw"] or 1) == 1 and int(mt["seed"] or 0) == 0:
            D = pd.read_csv(os.path.join(BASE, f))
            ens[mt["method"]].append(float(np.sqrt(np.mean(((D.prediction - D.truth) / np.clip(D.uncertainty, 0.01, None)) ** 2))))
    fz = lambda x: f"{x:.1f}"
    chk("C.2", "ensembles: RMS of e / member spread (floored at 0.01) over the draw-1 cells: Chemprop; MLP",
        "3.8-26.8; 8.9-31.8",
        f"{fz(min(ens['chemprop_ens']))}-{fz(max(ens['chemprop_ens']))}; {fz(min(ens['mlp_ens']))}-{fz(max(ens['mlp_ens']))}",
        f"results/baselines/<chemprop_ens|mlp_ens>__<ds>__N<size>.csv, draw 1, seed 0 ({len(ens['chemprop_ens'])} / {len(ens['mlp_ens'])} cells)")
    w30 = [json.load(open(f, encoding="utf-8"))["wall_s"] for f in glob.glob(os.path.join(BASE, "tabpfn__*__N30*.meta.json"))]
    wall = [json.load(open(f, encoding="utf-8"))["wall_s"] for f in glob.glob(os.path.join(BASE, "tabpfn__*.meta.json"))]
    chk("C.3", "TabPFN fit wall time at N = 30; maximum", "9 to 17 s; 98 s", f"{min(w30):.0f} to {max(w30):.0f} s; {max(wall):.0f} s", "tabpfn__*.meta.json wall_s")
    doc("C.4", "217 RDKit 2D descriptors; constant / mostly missing columns dropped, median imputation", "217", "docs/EXPERIMENT_DESIGN_v2.md section 7; src/baselines")


def secD():
    hdr("D Contamination")
    cs = pd.read_csv(os.path.join(RES, "contamination", "corpus", "summary_by_subset.csv"))
    n_all = int(cs[cs.subset == "all"].groupby("dataset").n_mol.first().sum())
    chk("D.0", "searchable molecules (strings of at least 8 characters) of 7,847", "7,753", f"{n_all:,}", "summary_by_subset.csv n_mol, subset all")
    K = cal()
    E = K[(K.dataset == "esol") & K.signal.isin(["UC1", "sigma"])]
    P = E.pivot_table(index=["model", "size"], columns="subset", values="RMSE")
    ea = low(None)
    ea = ea[ea.dataset == "esol"]
    Q = esol_subsets()
    Qs = Q.pivot_table(index=["model", "size"], columns="subset", values="mean_sigma")
    Qz = Q.pivot_table(index=["model", "size"], columns="subset", values="rmsz")
    chk("D.1", "ESOL: RMSE recalled < never recalled in every cell (suite: LLMs + TabPFN + RF; cells.csv all draws); "
        "mean stated sigma recalled < never in the 42 LLM cells; RMS of e / sigma recalled < never in the 42 LLM cells",
        "54 of 54; 84 of 84; 42 of 42; 42 of 42",
        f"{int((P.recalled < P.never_recalled).sum())} of {len(P)}; {int((ea.RMSE_recalled < ea.RMSE_never_recalled).sum())} of {len(ea)}; "
        f"{int((Qs.recalled < Qs.never_recalled).sum())} of {len(Qs)}; {int((Qz.recalled < Qz.never_recalled).sum())} of {len(Qz)}",
        "calibration_cells.csv RMSE; cells.csv RMSE_recalled / _never_recalled; esol_permolecule.csv uc1 (floored at 0.01), draw 1")
    # D.2 (ESOL held-out PICP68 after scaling) removed with the calibration analysis (review round 2026-10-28)
    x = low(1)
    x = x[x.dataset == "esol"].set_index(["model", "size"])
    chk("D.3", "GLM ESOL RMSE N = 30 -> 976", "0.74 -> 0.73", f"{f2(x.loc[('glm53flash', 30), 'RMSE'])} -> {f2(x.loc[('glm53flash', 976), 'RMSE'])}", "cells.csv")
    z = zeroshot()
    chk("D.4", "zero-shot screen: all replies parsed, none truncated", "1.0; 0", f"{z.parse_rate.min():.1f}; {int(z.n_truncated.sum())}", "zeroshot/cells.csv")
    zo = pd.read_csv(os.path.join(RES, "contamination", "zeroshot", "_pre_unitfix_2026-10-15", "cells.csv"))
    zo = zo[zo.dataset == "esol"].set_index("model")
    zn = z[z.dataset == "esol"].set_index("model")
    pm = lambda d: pd.read_csv(os.path.join(RES, "contamination", "zeroshot", d, "esol_v2test150_permolecule.csv")).set_index("mol_id")
    po, pn = pm("_pre_unitfix_2026-10-15"), pm("")
    so, sn = po.n_models_hit3, pn.n_models_hit3.reindex(po.index)
    to_never = int(((so > 0) & (sn == 0)).sum())
    to_rec = int(((so == 0) & (sn > 0)).sum())
    lost = sum(int(max(zo[c][m] - zn[c][m], 0)) for m in zn.index for c in ("m1", "m2", "m3"))
    chk("D.5", "unit rule: Sonnet / Haiku hit3 without -> with; Haiku regime; Spearman without -> with; matches lost; recalled -> never / never -> recalled",
        "31.0 -> 37.2; 8.8 -> 10.6; partial -> heavy; 0.27 / 0.51 -> 0.94 / 0.94; 0; 0 / 1",
        f"{zo.hit3['sonnet55']:.1f} -> {zn.hit3['sonnet55']:.1f}; {zo.hit3['haiku55']:.1f} -> {zn.hit3['haiku55']:.1f}; "
        f"{zo.regime['haiku55']} -> {zn.regime['haiku55']}; "
        f"{f2(zo.spearman['sonnet55'])} / {f2(zo.spearman['haiku55'])} -> {f2(zn.spearman['sonnet55'])} / {f2(zn.spearman['haiku55'])}; "
        f"{lost}; {to_never} / {to_rec}",
        "zeroshot/_pre_unitfix_2026-10-15/ vs zeroshot/ cells.csv and esol_v2test150_permolecule.csv n_models_hit3; "
        f"ESOL test labels > 0: {int((pn.truth > 0).sum())}")
    ze = z[z.dataset == "esol"].set_index("model")
    heavy = sorted(ze.index[ze.regime == "heavy"])
    partial = sorted(ze.index[ze.regime == "partial"])
    chk("D.6", "ESOL regimes: heavy Sonnet, GPT-6.1 Sol, GPT-6 Astra, Gemini, Haiku; partial GPT-6 Luna",
        "heavy gem38flash,gpt61sol,gpt6astra,haiku55,sonnet55; partial gpt6luna",
        f"heavy {','.join(heavy)}; partial {','.join(partial)}", "zeroshot/cells.csv regime")


def secE():
    hdr("E Repeat spread and similarity to the context")
    K = cal()
    u1 = K[(K.signal == "UC1") & (K.subset == "all")].set_index(["dataset", "model", "size"])
    u3 = K[(K.signal == "UC3") & (K.subset == "all")].set_index(["dataset", "model", "size"])
    better = u3.rho_sigma_err > u1.rho_sigma_err
    so = K[(K.dataset == "esol") & (K.model == "sonnet55") & (K["size"] == 976)].set_index(["signal", "subset"])
    chk("E.1", "UC3 exactly zero share range; UC3 ranks better than UC1: clean cells; ESOL cells; Sonnet ESOL N = 976 rho UC3 vs UC1; identical repeats recalled / never",
        "0.7-33 %; 8 of 126; 24 of 42; 0.65 vs 0.40; 38 / 18",
        f"{100 * u3.share_sigma_zero.min():.1f}-{100 * u3.share_sigma_zero.max():.0f} %; "
        f"{int(better.loc[CLEAN].sum())} of {len(better.loc[CLEAN])}; {int(better.loc['esol'].sum())} of {len(better.loc['esol'])}; "
        f"{f2(so.loc[('UC3', 'all'), 'rho_sigma_err'])} vs {f2(so.loc[('UC1', 'all'), 'rho_sigma_err'])}; "
        f"{100 * so.loc[('UC3', 'recalled'), 'share_sigma_zero']:.0f} / {100 * so.loc[('UC3', 'never_recalled'), 'share_sigma_zero']:.0f}",
        "calibration_cells.csv rho_sigma_err, share_sigma_zero (draw 1)")
    X = pd.read_csv(os.path.join(ICL, "mixture_sigma.csv"))
    X1 = X[X.draw == 1]
    xc, xe = X1[X1.dataset.isin(CLEAN)], X1[X1.dataset == "esol"]
    dm = X1.rho_mix - X1.rho_uc1
    chk("E.2", "mixture sigma vs stated sigma: median rho_mix - rho_uc1 clean; clean cells with rho_mix > rho_uc1; ESOL cells; median ESOL",
        "0.00; 68 of 126; 42 of 42; 0.03",
        f"{dm[X1.dataset.isin(CLEAN)].median():.2f}; {int((xc.rho_mix > xc.rho_uc1).sum())} of {len(xc)}; "
        f"{int((xe.rho_mix > xe.rho_uc1).sum())} of {len(xe)}; {dm[X1.dataset == 'esol'].median():.2f}",
        "mixture_sigma.csv draw 1")
    Sm = pd.read_csv(os.path.join(ICL, "context_similarity.csv"))
    S1 = Sm[(Sm.draw == 1) & Sm.dataset.isin(CLEAN)]
    s30 = S1[S1["size"] == 30]
    stop = S1[S1["size"] == S1.dataset.map(TOP)]
    rg = lambda x: f"{r2(x.min())} and {r2(x.max())}"
    L30, Ltop = s30[s30.model.isin(LLMS)], stop[stop.model.isin(LLMS)]
    B30, Btop = s30[s30.model.isin(DISPLAY)], stop[stop.model.isin(DISPLAY)]
    Lb = pd.concat([L30, Ltop])
    Bb = pd.concat([B30, Btop])
    chk("E.3", "rho(maxsim, |e|), clean, draw 1: LLM median N = 30; LLM median largest N; LLM range over both sizes; four baselines range over both sizes",
        "-0.21; -0.12; -0.31 and 0.01; -0.37 and 0.15",
        f"{L30.rho_sim_abserr.median():.2f}; {Ltop.rho_sim_abserr.median():.2f}; {rg(Lb.rho_sim_abserr)}; {rg(Bb.rho_sim_abserr)}",
        f"context_similarity.csv rho_sim_abserr ({len(L30)} / {len(Ltop)} LLM, {len(B30)} / {len(Btop)} baseline cells: "
        f"{', '.join(DISPLAY)}); LLM median {L30.rho_sim_abserr.median():+.2f} -> {Ltop.rho_sim_abserr.median():+.2f}")
    both = pd.concat([s30, stop])
    tb, gp = both[both.model == "tabpfn"], both[both.model == "gp_tanimoto"]
    med = lambda d, m: f"{d[d.model == m].rho_sim_sigma.median():.2f}"
    chk("E.4", "rho(maxsim, sigma), clean, draw 1, medians over 7 LLMs x 3 datasets / over the 3 datasets for a baseline: LLMs N = 30 / largest N; TabPFN N = 30 / largest N; GP N = 30 / largest N",
        "-0.56 / -0.49; -0.54 / -0.28; -0.97 / -0.89",
        f"{L30.rho_sim_sigma.median():.2f} / {Ltop.rho_sim_sigma.median():.2f}; {med(s30, 'tabpfn')} / {med(stop, 'tabpfn')}; "
        f"{med(s30, 'gp_tanimoto')} / {med(stop, 'gp_tanimoto')}",
        f"context_similarity.csv rho_sim_sigma ({len(tb)} TabPFN, {len(gp)} GP cells); LLM median "
        f"{L30.rho_sim_sigma.median():+.2f} -> {Ltop.rho_sim_sigma.median():+.2f}")


def secF():
    hdr("F Selective prediction and context draws")
    from scipy.stats import spearmanr
    s = sel_llm(1)
    sb = sel_base(1)
    mins = []
    for ds in CLEAN:
        for nn in sorted(s[s.dataset == ds]["size"].unique()):
            x = pd.concat([s[(s.dataset == ds) & (s["size"] == nn)][["model", "RMSE", "RMSE80"]],
                           sb[(sb.dataset == ds) & (sb["size"] == nn) & sb.model.isin(DISPLAY)][["model", "RMSE", "RMSE80"]]])
            mins.append(spearmanr(x.RMSE, x.RMSE80).correlation)
    chk("F.1", "q = 0.8: rank correlation between the models' RMSE and RMSE80 over the 11 plotted models, minimum over the 18 clean cells",
        "0.69", f2(min(mins)), "selective_prediction.csv draw 1")
    left, above = [], []
    for ds in CLEAN:
        Ld, Bd = s[(s.dataset == ds) & (s["size"] == 30)], sb[(sb.dataset == ds) & (sb["size"] == 30)]
        left.append(int(Ld.RMSE.min() < Bd.RMSE.min()))
        if Ld.gain50.max() > Bd.gain50.max():
            above.append(ds)
    chk("F.2", "Fig. A5: most accurate LLMs left of every baseline (3 clean sets); LLMs highest in gain50 only on",
        "3; lipophilicity", f"{sum(left)}; {','.join(above)}", "selective_prediction.csv N = 30 draw 1")
    a = sel_llm(None)
    w = a.RMSE50_max - a.RMSE50_min
    chk("F.3", "tie-order range of RMSE50: cells with a split group; median width; max in a draw-1 cell",
        "286 of 336; 0.03; 0.23", f"{int((w > 0).sum())} of {len(w)}; {f2(w.median())}; {f2(w[a.draw == 1].max())}",
        f"selective_prediction.csv; max any draw {w.max():.2f}")
    ex = extras()["draws"]["median_sd_range"]
    chk("F.4", "median SD of the RMSE over draws lower for the LLMs in 10 of 12 dataset x N", "10 of 12",
        f"{ex['cells_LLM_lower']} of {ex['cells']}", "report_extras.json draws (cells.csv, summary.csv)")
    a30 = a[(a["size"] == 30)].groupby(["dataset", "model"]).gain50.agg(lambda v: v.max() - v.min())
    chk("F.5", "gain50 range over draws at N = 30 up to (LLM)", "0.22", f2(a30.max()), f"selective_prediction.csv; max at {a30.idxmax()}")


def secG():
    hdr("G Reasoning ladder")
    l = pd.read_csv(os.path.join(ICL, "ladder_cells.csv"))
    o = l[~((l.model == "gem38flash") & (l.effort == "high"))]
    chk("G.1", "outside Gemini high: at most two of the 450 calls of a cell truncated", "2", f"{(o.trunc_rate * 450).max():.0f}", "ladder_cells.csv trunc_rate")
    chk("G.2", "Gemini 3.8 Flash main-run calls with zero reasoning tokens", "18 %", f"{100 * extras()['gemini_zero_reasoning']['share_zero']:.0f} %", "report_extras.json")
    c = cells()
    lad = c[c.reasoning != "low"]
    by = lad.groupby("reasoning").cost_usd.sum()
    gh = lad[(lad.model == "gem38flash") & (lad.reasoning == "high")].cost_usd.sum()
    chk("G.3", "ladder steps cost USD none / minimal / medium / high; Gemini high", "3.01 / 71.62 / 78.65 / 241.68; 171.70",
        f"{by['none']:.2f} / {by['minimal']:.2f} / {by['medium']:.2f} / {by['high']:.2f}; {gh:.2f}", "cells.csv cost_usd")
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "meta", "effort_probe_2026-10-0*.jsonl"))):
        for line in open(f, encoding="utf8"):
            rows.append(json.loads(line))
    P = pd.DataFrame(rows)
    P = P[P.tag.isin(LLMS)]
    acc = {m: set(P[(P.tag == m) & (P.status == "ok")].level) | set(l[l.model == m].effort) for m in LLMS}
    rej = sorted(m for m in LLMS if "none" not in acc[m])
    chk("G.5", "panel endpoints that reject the effort none (probe or ladder, as Table 1b)", "5 of 7", f"{len(rej)} of 7",
        "results/meta/effort_probe_*.jsonl status, ladder_cells.csv: " + ",".join(rej))
    # G.4 (range of Table A9) removed with that table (2026-10-14); the maximum is checked in 3.1.10


def secH():
    hdr("H Operational findings")
    c = cells()
    few = c[c.n_calls < 450]
    chk("H.1", "cells with fewer than 450 calls: all OpenAI; minimum calls", "openai only; 445",
        f"{'openai only' if set(few.model) <= {'gpt61sol', 'gpt6luna', 'gpt6astra'} else sorted(set(few.model))}; {int(few.n_calls.min())}",
        f"cells.csv n_calls; {len(few)} cells; every cell answers all molecules: {bool((c[c.reasoning == 'low'].n_mol == 150).all())} (main run and draws)")
    m1 = low(1)
    cs = m1[m1["size"] >= 120].cache_share
    chk("H.2", "cached share of prompt tokens, main-run cells with N >= 120", "0.91 to 1.00", f"{cs.min():.2f} to {cs.max():.2f}", "cells.csv cache_share")
    sd = extras()["sigma_distinct"]
    chk("H.3", "distinct standard deviations among the 450 replies of a main-run cell", "7 to 37", f"{sd['per_cell_min']} to {sd['per_cell_max']}", "report_extras.json sigma_distinct")
    rc = 0
    for f in glob.glob(os.path.join(ICL, "*__d[23].jsonl")):
        with open(f, encoding="utf-8") as fh:
            for _ in fh:
                rc += 1
    dups = rc - int(c[(c.reasoning == "low") & (c.draw > 1)].n_calls.sum())
    chk("H.4", "duplicated records in the context draws", "1,613", f"{dups:,}", "raw *__d[23].jsonl minus cells.csv n_calls")
    ts = {}
    pat = re.compile(r'"timestamp":"(\d{4}-\d\d-\d\d)')
    for f in glob.glob(os.path.join(ICL, "*.jsonl")):
        b = os.path.basename(f)
        run = "draws" if re.search(r"__d[23]\.jsonl$", b) else ("main" if "__low" in b else "ladder")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                m = pat.search(line)
                if m:
                    ts.setdefault(run, set()).add(m.group(1))
    out = {k: f"{min(v)}..{max(v)}" for k, v in ts.items()}
    chk("H.5", "main run 2026-10-09..13; ladder 2026-10-12..13; draws 2026-10-13",
        "main 2026-10-09..2026-10-13; ladder 2026-10-12..2026-10-13; draws 2026-10-13..2026-10-13",
        f"main {out.get('main')}; ladder {out.get('ladder')}; draws {out.get('draws')}", "results/icl/*.jsonl timestamp")
    doc("H.6", "prices and endpoint lists fetched 2026-10-08", "2026-10-08", "src/registry/models_v2.json prices_fetched")
    # H.7 (totals of Table A11) removed with that table (2026-10-14); the per-cell calls and cost are in cells.csv


SECTIONS = [("2.1", sec2_1), ("2.2", sec2_2), ("2.3", sec2_3), ("2.4", sec2_4), ("2.5", sec2_5), ("2.6", sec2_6),
            ("3.1", sec3_1), ("3.2", sec3_2), ("3.3", sec3_3), ("3.4", sec3_4), ("4", sec4),
            ("C", secC), ("D", secD), ("E", secE), ("F", secF), ("G", secG), ("H", secH)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sec", default=None, help="2, 3, 2.1 ... 3.4, 4, C ... H")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    for k, f in SECTIONS:
        if a.sec is None or k == a.sec or k.split(".")[0] == a.sec:
            f()
    hdr("SUMMARY")
    for st in ("OK", "CODE", "INFO", "DOC-ONLY", "DIFF"):
        ids = [c for c, s, _ in LOG if s == st]
        print(f"  {st:8s} {len(ids):3d}  {' '.join(ids) if st in ('DIFF', 'DOC-ONLY', 'CODE') else ''}")


if __name__ == "__main__":
    main()
