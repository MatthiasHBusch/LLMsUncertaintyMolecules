"""
Export every table of docs/PAPER_OUTLINE.md section E as LaTeX (booktabs; longtable where a table
is longer than a page) into tables/*.tex. Reads finished result CSVs only; no API calls,
nothing under results/ or data/ is written.

    python src/export_tables.py            -> tables/*.tex (list printed at the end)

Every output file is a complete float (`table`) or `longtable` with caption and \\label{tab:...};
the manuscript only \\input{}s it. Captions open with a plain title sentence (STYLE rule 9), no
bold or italics; markers are plain symbols (dagger, double dagger).

Rules (decision of the project lead, PAPER_OUTLINE.md H.4):
  * values are context draw 1 at every N; where a table is about N <= 120 the three context draws
    are given as a min-max range in brackets next to the draw-1 value (never averaged);
  * baselines are averaged over their seeds within a draw first (draw 1: seeds 0-2 for TabPFN,
    GP, kNN and both RF; draws 2-3: seeds 0-2 for GP and kNN; seed 0 elsewhere); the calibration
    suite (calibration_cells.csv: the repository tables t3 / a_calibration / a_uc3 and, in the paper,
    the ESOL subset table c_d2_esol_subsets) and the paired bootstrap (bootstrap_cells.csv) use seed 0, as
    their source scripts do, and the captions say so;
  * one shared model / baseline order and display-name list (LLM_ORDER, BASE_ORDER below).

Sources: results/icl/cells.csv, draws_cells.csv (cross-check only), selective_prediction.csv,
bootstrap_cells.csv, calibration_cells.csv, calibration_bootstrap.csv, ladder_cells.csv;
results/baselines/summary.csv and *.meta.json; results/contamination/zeroshot/cells.csv,
corpus/summary_by_subset.csv, corpus/headers.csv, train_pool_dedup.csv;
results/meta/effort_probe_*.jsonl, icl_probe_*.jsonl, diagnostics/*.jsonl, context_draw_overlap.csv;
src/registry/models_v2.json; data/splits/<ds>_v2.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "tables")

# ---------------------------------------------------------------- shared order and names
LLM_ORDER = ["haiku55", "sonnet55", "gpt61sol", "gpt6luna", "gpt6astra", "gem38flash", "glm53flash"]
LLM_NAMES = {"haiku55": "Haiku 5.5", "sonnet55": "Sonnet 5.5", "gpt61sol": "GPT-6.1 Sol",
             "gpt6luna": "GPT-6 Luna", "gpt6astra": "GPT-6 Astra", "gem38flash": "Gemini 3.8 Flash",
             "glm53flash": "GLM 5.3 Flash"}
LADDER_ORDER = [m for m in LLM_ORDER if m != "gpt6astra"]       # declared six-model sub-experiment
BASE_ORDER = ["tabpfn", "gp_tanimoto", "rf_desc", "rf", "chemprop_ens", "mlp_ens", "knn_tanimoto", "trivial"]
BASE_NAMES = {"tabpfn": "TabPFN v3.5", "gp_tanimoto": "GP (Tanimoto)", "rf_desc": "RF (descriptors)",
              "rf": "RF (Morgan)", "chemprop_ens": "Chemprop ensemble", "mlp_ens": "MLP ensemble",
              "knn_tanimoto": "kNN (Tanimoto)", "trivial": "trivial"}
NAMES = {**LLM_NAMES, **BASE_NAMES}
ORDER = LLM_ORDER + BASE_ORDER
DS = ["lipophilicity", "bace", "antiviral", "esol"]
CLEAN = DS[:3]
DS_NAMES = {"lipophilicity": "Lipophilicity", "bace": "BACE", "antiviral": "Antiviral", "esol": "ESOL"}
TOP = {"lipophilicity": 1000, "bace": 1000, "antiviral": 844, "esol": 976}
SIZES = {ds: [30, 60, 120, 250, 500, TOP[ds]] for ds in DS}
SMALL = [30, 60, 120]
EFFORTS = ["none", "minimal", "low", "medium", "high"]
INDEX_NAMES = {"v4_olmo-2-0325-32b-instruct_llama": "OLMo-2 32B Instruct training data",
               "v4_dolma-v1_7_llama": "Dolma v1.7", "v4_dclm-baseline_llama": "DCLM-baseline",
               "v4_olmo-mix-1124_llama": "OLMo-2 mix 1124", "v4_piletrain_llama": "The Pile",
               "v4_rpj_llama_s4": "RedPajama", "v4_c4train_llama": "C4"}
INDEX_ORDER = list(INDEX_NAMES)
DAG, DDAG = r"$^{\dagger}$", r"$^{\ddagger}$"

WRITTEN = []


# ---------------------------------------------------------------- formatting
def num(x, d=2, plus=False):
    """Number with a typographic minus; '--' for NaN."""
    if x is None or pd.isna(x) or not np.isfinite(x):
        return "--"
    # the sign of a value that rounds to zero is kept (a CI bound printed as -0.00 lies below 0)
    s = f"{x:+.{d}f}" if plus else f"{x:.{d}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def rng(lo, hi, d=2):
    """min-max range in brackets; en dash for non-negative bounds, 'to' otherwise."""
    if pd.isna(lo) or pd.isna(hi):
        return ""
    sep = "--" if lo >= 0 and hi >= 0 else " to "
    return f"[{num(lo, d)}{sep}{num(hi, d)}]"


def ci(lo, hi, d=2):
    return f"[{num(lo, d, plus=True)}, {num(hi, d, plus=True)}]"


def intc(x):
    return "--" if pd.isna(x) else f"{int(round(x)):,}"


def tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#").replace("$", r"\$"))


def write(name, text, label):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.tex")
    with open(path, "w", encoding="utf8", newline="\n") as f:
        f.write("% generated by src/export_tables.py -- do not edit by hand\n" + text.rstrip() + "\n")
    WRITTEN.append((os.path.relpath(path, ROOT), label))


def float_table(name, label, caption, colspec, header, body, size=r"\footnotesize", tabcolsep=None,
                resize=False, placement="tbp", star=False):
    env = "table*" if star else "table"
    lines = [rf"\begin{{{env}}}[{placement}]", r"\centering", size]
    if tabcolsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    tab = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *body, r"\bottomrule",
           r"\end{tabular}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines += tab
        lines.append("}")
    else:
        lines += tab
    lines.append(rf"\end{{{env}}}")
    write(name, "\n".join(lines), label)


def long_table(name, label, caption, colspec, header, body, size=r"\scriptsize", tabcolsep="3pt"):
    ncol = len([c for c in colspec.replace("@{}", "") if c in "lcrp"])
    lines = [r"\begingroup", size, rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}", r"\setlength{\LTcapwidth}{\textwidth}",
             rf"\begin{{longtable}}{{{colspec}}}",
             rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", *header, r"\midrule", r"\endfirsthead",
             rf"\caption[]{{(continued)}}\\", r"\toprule", *header, r"\midrule", r"\endhead",
             rf"\midrule\multicolumn{{{ncol}}}{{r}}{{continued on the next page}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot", *body, r"\end{longtable}", r"\endgroup"]
    write(name, "\n".join(lines), label)


def row(*cells):
    return " & ".join(cells) + r" \\"


# ---------------------------------------------------------------- data
def rd(*p):
    return pd.read_csv(os.path.join(RES, *p))


def load():
    D = {}
    C = rd("icl", "cells.csv")
    D["cells_all"] = C
    D["cells"] = C[C.reasoning == "low"].copy()
    S = rd("icl", "selective_prediction.csv")
    D["sel"] = S
    B = rd("baselines", "summary.csv")
    D["base_seeds"] = B
    D["boot"] = rd("icl", "bootstrap_cells.csv")
    D["cal"] = rd("icl", "calibration_cells.csv")
    D["calboot"] = rd("icl", "calibration_bootstrap.csv")
    D["ladder"] = rd("icl", "ladder_cells.csv")
    D["zs"] = rd("contamination", "zeroshot", "cells.csv")
    D["corpus"] = rd("contamination", "corpus", "summary_by_subset.csv")
    D["headers"] = rd("contamination", "corpus", "headers.csv")
    D["dedup"] = rd("contamination", "train_pool_dedup.csv")
    D["overlap"] = rd("meta", "context_draw_overlap.csv")
    D["draws_cells"] = rd("icl", "draws_cells.csv")
    D["registry"] = {m["tag"]: m for m in json.load(open(os.path.join(ROOT, "src", "registry", "models_v2.json"),
                                                           encoding="utf8"))["models"]}
    D["splits"] = {ds: json.load(open(os.path.join(ROOT, "data", "splits", f"{ds}_v2.json"), encoding="utf8"))
                   for ds in DS}
    return D


def unified(D):
    """One row per (dataset, model, size, draw): LLMs at reasoning `low` (cells.csv + selective_prediction.csv),
    baselines averaged over their seeds within a draw (summary.csv + selective_prediction.csv).
    `rho` = Spearman(sigma, |err|) with average ranks on sigma rounded to 10 decimals, from
    selective_prediction.csv: the definition of the Methods and the point estimate of bootstrap_cells.csv,
    whose markers the tables print. cells.csv / summary.csv `rho_UC1_err` rank the unrounded repeat means,
    whose last-bit noise splits ties (differences up to 0.009); it is kept only for the cross-check."""
    C = D["cells"]
    L = C[["dataset", "model", "size", "draw", "RMSE", "MAE", "spearman_pred", "rho_UC1_err", "PICP68", "PICP90",
           "NLL", "mean_UC1", "median_reasoning_tokens", "coverage", "parse_rate", "n_mol"]].copy()
    L["family"] = "LLM"
    S = D["sel"]
    SL = S[(S.family == "LLM") & (S.reasoning == "low")][["ds", "model", "size", "draw", "RMSE50", "gain50",
                                                             "RMSE80", "gain80", "RMSE", "n_sigma_distinct", "rho"]]
    SL = SL.rename(columns={"ds": "dataset", "RMSE": "RMSE_sel"})
    L = L.merge(SL, on=["dataset", "model", "size", "draw"], how="left")
    assert (L.RMSE - L.RMSE_sel).abs().max() < 1e-9, "selective_prediction.csv out of date"
    B = D["base_seeds"]
    cols = ["RMSE", "MAE", "spearman_pred", "rho_UC1_err", "PICP68", "PICP90", "NLL", "mean_UC1"]
    BB = B.groupby(["dataset", "method", "size", "draw"])[cols].mean().reset_index().rename(columns={"method": "model"})
    BB["n_seeds"] = B.groupby(["dataset", "method", "size", "draw"]).size().values
    SB = S[S.family == "baseline"].groupby(["ds", "model", "size", "draw"])[["RMSE50", "gain50", "RMSE80", "gain80",
                                                                            "rho"]]
    SB = SB.mean().reset_index().rename(columns={"ds": "dataset"})
    BB = BB.merge(SB, on=["dataset", "model", "size", "draw"], how="left")
    BB["family"] = "baseline"
    U = pd.concat([L, BB], ignore_index=True)
    return U


def get(U, ds, m, n, d, col):
    r = U[(U.dataset == ds) & (U.model == m) & (U["size"] == n) & (U.draw == d)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def with_range(U, ds, m, n, col, d=2):
    v1 = get(U, ds, m, n, 1, col)
    vals = [get(U, ds, m, n, k, col) for k in (1, 2, 3)]
    if any(pd.isna(v) for v in vals):
        return num(v1, d)
    return f"{num(v1, d)} {rng(min(vals), max(vals), d)}"


def boot_flag(D, ds, m, n, stat, draw=0, base="tabpfn"):
    """dagger if the paired-bootstrap CI of LLM - baseline excludes 0 in the LLM's favour, double dagger
    if in the baseline's favour (RMSE / RMSE50: negative favours the LLM; rho / gain50: positive)."""
    Bt = D["boot"]
    r = Bt[(Bt.dataset == ds) & (Bt.model == m) & (Bt.baseline == base) & (Bt["size"] == n) & (Bt.draw == draw)
           & (Bt.stat == stat)]
    if not len(r):
        return ""
    lo, hi = float(r.lo.iloc[0]), float(r.hi.iloc[0])
    better_low = stat in ("RMSE", "RMSE50")
    if hi < 0:
        return DAG if better_low else DDAG
    if lo > 0:
        return DDAG if better_low else DAG
    return ""


# ---------------------------------------------------------------- T1 datasets and panel
def effort_probe_levels():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "meta", "effort_probe_2026-10-0*.jsonl"))):
        for line in open(f, encoding="utf8"):
            rows.append(json.loads(line))
    P = pd.DataFrame(rows)
    return P[P.tag.isin(LLM_ORDER)]


def t1_panel(D, U):
    sp, Z, K = D["splits"], D["zs"], D["corpus"]
    prop = {"lipophilicity": r"logD at pH 7.4", "bace": r"BACE-1 pIC50", "antiviral": r"SARS-CoV-2 Mpro pIC50",
            "esol": r"logS (log$_{10}$ mol/L)"}
    src = {"lipophilicity": r"\citep{MoleculeNet}", "bace": r"\citep{Subramanian2016}",
           "antiviral": r"\citep{asap_polaris_openadmet_2025}", "esol": r"\citep{Delaney2004}"}
    body_a = []
    for ds in DS:
        s = sp[ds]
        k = K[(K.dataset == ds) & (K.subset == "v2test150")]
        pres = k.pct_present_smiles.max()
        z = Z[Z.dataset == ds]
        nflag = int((z.verdict != "clean").sum())
        body_a.append(row(DS_NAMES[ds] + " " + src[ds], prop[ds], str(s["n_test"]), f"{s['n_train_pool']:,}",
                          f"{TOP[ds]:,}", f"{num(pres, 1)}\\,\\%", f"{nflag} of {len(z)}"))
    hdr_a = [row("dataset", "property", "test", "pool", "largest $N$", "in corpus", "flagged")]
    # (b) models: what the requested effort does (accepted levels, emitted tokens) and the ESOL screen; endpoint,
    # tier and prices are in Table A1 only (project lead, 2026-10-14: each piece of information once)
    P = effort_probe_levels()
    Lad = D["ladder"]
    C = D["cells"]
    tk = _ladder_token_table(D)
    body_b = []
    for m in LLM_ORDER:
        ok = set(P[(P.tag == m) & (P.status == "ok")].level) | set(Lad[Lad.model == m].effort)
        g = C[(C.model == m) & (C.draw == 1)]
        lo_t, hi_t = g.median_reasoning_tokens.min(), g.median_reasoning_tokens.max()
        tok = intc(lo_t) if lo_t == hi_t else f"{intc(lo_t)}--{intc(hi_t)}"
        ladder = LADDER_SHORT[_dosability(tk, m)] if m in LADDER_ORDER else "no ladder"
        z = Z[(Z.dataset == "esol") & (Z.model == m)].iloc[0]
        out = "clean" if z.verdict == "clean" else f"flagged, {z.regime}" if isinstance(z.regime, str) else "flagged"
        body_b.append(row(LLM_NAMES[m], "accepted" if "none" in ok else "rejected", tok, ladder, num(z.hit3, 1), out))
    hdr_b = [r" & & \multicolumn{2}{c}{reasoning tokens per call} & \multicolumn{2}{c}{ESOL zero-shot screen}\\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
             row("model", "effort none", "at low", "along the ladder", r"hit$_3$ (\%)", "outcome")]
    cap = (r"Datasets and model panel. (a) The four datasets: test-set size, deduplicated training pool, largest "
           r"context size $N$, share of the test molecules found in any of the seven searched pre-training corpora "
           r"as a SMILES string (in corpus; maximum over the corpora, Fig.~\ref{fig:contamination}b), and the "
           r"number of panel models the zero-shot screen flags (flagged). (b) The seven language models: whether the "
           r"endpoint accepts the reasoning effort none (every endpoint accepts minimal to high), the median emitted "
           r"reasoning tokens per call at the effort low (range over the 24 main-run cells) and how they change "
           r"along the reasoning ladder (Fig.~\ref{fig:ladder}), and on the ESOL test set the share of the molecules "
           r"whose published value the model reproduced to three significant figures (hit$_3$; answers given in "
           r"mol/L converted to logS, Section~\ref{sec:data}) with the outcome of "
           r"the zero-shot screen and its regime. Endpoints, service tiers and prices: Table~\ref{tab:b_panel}. "
           r"Endpoints as of 2026-10-08, zero-shot outcomes as of 2026-10-09 (GPT-6 Astra 2026-10-13).")
    lines = [r"\begin{table}[tbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             rf"\caption{{{cap}}}", r"\label{tab:panel}",
             r"\begin{tabular}{@{}lllrrrr@{}}", r"\multicolumn{7}{@{}l}{(a) Datasets}\\", r"\toprule",
             *hdr_a, r"\midrule", *body_a, r"\bottomrule", r"\end{tabular}", r"\par\medskip",
             r"\begin{tabular}{@{}llrlrl@{}}",
             r"\multicolumn{6}{@{}l}{(b) Language models}\\", r"\toprule", *hdr_b, r"\midrule", *body_b,
             r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("t1_panel", "\n".join(lines), "tab:panel")


# ---------------------------------------------------------------- T2 small-N summary
def t2_smallN(D, U, n=30, name="t2_smallN", label="tab:smallN", main=True):
    body = []
    for ds in CLEAN:
        body.append(rf"\multicolumn{{5}}{{@{{}}l}}{{{DS_NAMES[ds]}}}\\")
        for m in ORDER:
            cells = [NAMES[m]]
            for col, stat in (("RMSE", "RMSE"), ("rho", "rho"), ("RMSE50", "RMSE50"), ("gain50", "gain50")):
                v = with_range(U, ds, m, n, col)
                if m in LLM_ORDER:
                    v += boot_flag(D, ds, m, n, stat)
                cells.append(v)
            body.append(row(*cells))
        if ds != CLEAN[-1]:
            body.append(r"\addlinespace")
    hdr = [row("method", "RMSE", r"$\rho(\sigma,|e|)$", r"RMSE$_{50}$", r"gain$_{50}$")]
    cap = (f"Accuracy and selection at {n} in-context molecules on the three clean datasets. Root-mean-square "
           r"error on the 150 test molecules, Spearman rank correlation $\rho$ between the stated standard "
           r"deviation and the absolute error, RMSE$_{50}$ of the half of the test molecules with the smallest "
           r"stated standard deviation (molecules tied at the cut selected at random; square root of the expected "
           r"mean squared error) and gain$_{50}$ = "
           r"1 $-$ RMSE$_{50}$/RMSE. Each entry is context draw 1 with the range over the three context draws in "
           r"brackets; language models at effort low, baselines averaged over their seeds within a draw. "
           r"$\dagger$ / $\ddagger$: the paired-bootstrap 95\,\% interval of the difference to TabPFN v3.5 "
           r"(combined over the three draws, seed 0) excludes 0 in favour of the language model / of TabPFN.")
    if not main:
        cap += " ESOL and the larger context sizes are in the per-cell tables."
    long_table(name, label, cap, r"@{}lllll@{}", hdr, body, size=r"\footnotesize", tabcolsep="5pt") \
        if not main else float_table(name, label, cap, r"@{}lllll@{}", hdr, body, size=r"\scriptsize",
                                     tabcolsep="5pt", placement="p")


# ---------------------------------------------------------------- T3 calibration
def t3_calibration(D):
    K = D["cal"]
    U1 = K[(K.subset == "all") & K.signal.isin(["UC1", "sigma"])]
    U3 = K[(K.subset == "all") & (K.signal == "UC3")]
    body = []
    for m in LLM_ORDER + ["tabpfn", "rf_desc"]:
        g = U1[(U1.model == m) & U1.dataset.isin(CLEAN)]
        s_med, s_lo, s_hi = g.s.median(), g.s.min(), g.s.max()
        n_over = int((g.s > 1).sum())
        p90 = f"{num(g.PICP90_test.median())} $\\to$ {num(g.PICP90_scaled.median())}"
        ece = f"{num(g.ECE_test.median(), 3)} $\\to$ {num(g.ECE_scaled.median(), 3)}"
        if m in LLM_ORDER:
            a = U1[U1.model == m].set_index(["dataset", "size"]).NLL_scaled
            b = U3[U3.model == m].set_index(["dataset", "size"]).NLL_scaled
            worse = f"{int((b > a.reindex(b.index)).sum())} of {len(b)}"
        else:
            worse = "--"
        E = K[(K.dataset == "esol") & (K.model == m) & (K["size"] == TOP["esol"]) & K.signal.isin(["UC1", "sigma"])]
        er = E[E.subset == "recalled"].s
        en = E[E.subset == "never_recalled"].s
        esol = f"{num(er.iloc[0])} / {num(en.iloc[0])}" if len(er) and len(en) else "--"
        body.append(row(NAMES[m], f"{num(s_med)} {rng(s_lo, s_hi)}", f"{n_over} of {len(g)}", p90, ece, worse, esol))
    hdr = [row("model", r"$s$ median [range]", r"$s>1$", r"PICP$_{90}$ raw $\to$ scaled",
               r"ECE raw $\to$ scaled", r"UC3 worse NLL", r"ESOL $s$ rec.\,/\,never")]
    cap = (r"Calibration of the stated standard deviation and its recalibration. Variance-scaling factor $s$ "
           r"($s>1$: the stated standard deviation is too narrow) over the 18 clean cells per model (3 datasets "
           r"$\times$ 6 context sizes, context draw 1), number of those cells with $s>1$, median held-out "
           r"PICP$_{90}$ and PIT-ECE before and after variance scaling fitted on 75 calibration molecules and "
           r"scored on the other 75, the number of the model's 24 cells in which the repeat spread (UC3) has "
           r"the higher held-out negative log-likelihood after scaling than the stated standard deviation "
           r"(UC1), and $s$ on the recalled / never-recalled ESOL molecules at 976 in-context molecules. "
           r"Language models at effort low; TabPFN v3.5 and RF (descriptors) seed 0.")
    float_table("t3_calibration", "tab:calibration", cap, r"@{}lllllll@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="4pt", resize=True, placement="tbp")


# ---------------------------------------------------------------- appendix tables
def a_registry(D):
    body = []
    for m in LLM_ORDER:
        r = D["registry"][m]
        prices = f"{r['price_in']:g} / {r['price_cache_read']:g} / {r['price_out']:g}"
        body.append(row(LLM_NAMES[m], r["vendor"], r["release"], r"\texttt{" + r["endpoint"] + "}",
                        r["price_endpoint"].replace("_", r"\_"), "no" if not r["allow_fallbacks"] else "yes",
                        prices, r"\texttt{" + r["open_weights"] + "}" if r["open_weights"] else "--"))
    hdr = [row("model", "vendor", "listed", "OpenRouter id", "endpoint reached", "fallback",
               r"USD/Mtok in / cache / out", "open weights")]
    cap = ("Model registry. Vendor, OpenRouter listing date, endpoint id, the provider endpoint the requests "
           "are pinned to (flex tier where it exists), whether OpenRouter may fall back to another host, the "
           "price per million input, cached-input and output tokens at that endpoint, and the open-weights "
           "repository where one exists. Prices and listing dates fetched from OpenRouter on 2026-10-08 "
           r"(\texttt{src/registry/models\_v2.json}).")
    float_table("a01_registry", "tab:registry", cap, r"@{}lllllll l@{}".replace(" ", ""), hdr, body,
                size=r"\footnotesize", tabcolsep="4pt", resize=True, placement="htbp")


def a_probe(D):
    P = effort_probe_levels()
    body = []
    for m in LLM_ORDER:
        cells = [LLM_NAMES[m]]
        for e in EFFORTS:
            g = P[(P.tag == m) & (P.level == e)]
            ok = g[g.status == "ok"]
            if len(ok):
                cells.append(intc(ok.reasoning_tokens.median()))
            elif (g.status == "unsupported").any():
                cells.append("rejected")
            elif len(g):
                cells.append("no answer")
            else:
                cells.append("--")
        body.append(row(*cells))
    hdr = [row("model", *[r"\texttt{" + e + "}" for e in EFFORTS])]
    cap = (r"Effort probe. Median emitted reasoning tokens of the answered calls (two per level and model; "
           r"GPT-6.1 Sol four, because of a retry) on one zero-shot prompt, per requested effort level; "
           r"``rejected'' = HTTP 400 (reasoning cannot be disabled); ``no answer'' = every call failed with "
           r"a flex-tier outage (HTTP 502). Probe of 2026-10-08 (\texttt{results/meta/effort\_probe\_*.jsonl}).")
    float_table("a02_effort_probe", "tab:probe", cap, r"@{}lrrrrr@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="6pt", placement="htbp")


def a_corpus(D):
    K = D["corpus"]
    body = []
    for ds in DS:
        body.append(rf"\multicolumn{{8}}{{@{{}}l}}{{{DS_NAMES[ds]}}}\\")
        for ix in INDEX_ORDER:
            t = K[(K.dataset == ds) & (K["index"] == ix) & (K.subset == "v2test150")].iloc[0]
            a = K[(K.dataset == ds) & (K["index"] == ix) & (K.subset == "all")].iloc[0]
            nm = lambda v: "--" if pd.isna(v) else num(v, 1)
            body.append(row(INDEX_NAMES[ix], nm(t.pct_present_smiles), nm(t.pct_present_inchikey),
                            nm(t.pct_present_name), nm(a.pct_present_verbatim), nm(a.pct_present_smiles),
                            nm(a.pct_present_inchikey), nm(a.pct_row_cooccur)))
        body.append(r"\addlinespace")
    body = body[:-1]
    hdr = [r" & \multicolumn{3}{c}{test set (150)} & \multicolumn{4}{c}{whole dataset}\\",
           r"\cmidrule(lr){2-4}\cmidrule(l){5-8}",
           row("corpus", "SMILES", "InChIKey", "name", "verbatim", "SMILES", "InChIKey", "co-occ.")]
    n_all = {ds: int(K[(K.dataset == ds) & (K.subset == "all")].n_mol.iloc[0]) for ds in DS}
    n_esol_t = int(K[(K.dataset == "esol") & (K.subset == "v2test150")].n_mol.iloc[0])
    cap = (r"Corpus search. Per cent of molecules present at least once in each of the seven infini-gram "
           r"indexes, for the 150 test molecules and for the whole dataset: any SMILES form (verbatim, RDKit "
           r"canonical isomeric or non-isomeric), InChIKey, name (ESOL only), verbatim SMILES, and the SMILES "
           r"together with its label value in one document (a lower bound). Strings shorter than 8 characters "
           rf"are not searched (searchable: ESOL {n_esol_t} of 150 test and {n_all['esol']:,} of 1,128 molecules; "
           rf"the other datasets in full). \texttt{{results/contamination/corpus/summary\_by\_subset.csv}}.")
    long_table("a03_corpus", "tab:corpus", cap, r"@{}lrrrrrrr@{}", hdr, body, size=r"\footnotesize",
               tabcolsep="5pt")
    # header strings
    H = D["headers"]
    body = []
    for ds in DS:
        for h in H[H.dataset == ds].header.unique():
            g = H[(H.dataset == ds) & (H.header == h)].set_index("index")["count"]
            body.append(row(DS_NAMES[ds], r"\texttt{" + tex_escape(h) + "}",
                            *[intc(g.get(ix, np.nan)) for ix in INDEX_ORDER]))
    short = ["OLMo-2 32B", "Dolma", "DCLM", "OLMo mix", "Pile", "RedPajama", "C4"]
    hdr = [row("dataset", "string", *short)]
    cap = (r"Dataset header and file-name strings in the corpora. Number of documents containing the string, "
           r"per index (\texttt{results/contamination/corpus/headers.csv}).")
    float_table("a04_corpus_headers", "tab:headers", cap, r"@{}llrrrrrrr@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="4pt", resize=True, placement="htbp")


def a_zeroshot(D):
    Z = D["zs"]
    body = []
    for ds in DS:
        for m in LLM_ORDER:
            z = Z[(Z.dataset == ds) & (Z.model == m)].iloc[0]
            q = lambda v: "--" if pd.isna(v) else (f"{v:.2f}" if v >= 0.01 else f"{v:.0e}".replace("e-0", "e-")
                                                    .replace("e-", r"$\times10^{-") + "}$")
            outcome = "flagged" + (f" ({z.regime})" if isinstance(z.regime, str) else "") if z.verdict != "clean" \
                else "clean"
            body.append(row(DS_NAMES[ds], LLM_NAMES[m], str(int(z.n_usable)), str(int(z.m1)), str(int(z.m2)),
                            str(int(z.m3)), num(z.hit3, 1), num(z.R12, 1), num(z.floor_R12, 1), q(z.q_R12),
                            q(z.q_R23), num(z.spearman), outcome))
        if ds != DS[-1]:
            body.append(r"\addlinespace")
    hdr = [row("dataset", "model", r"$n_\mathrm{usable}$", "$m_1$", "$m_2$", "$m_3$", r"hit$_3$ \%",
               r"$R_{12}$ \%", r"floor \%", r"$q_{12}$", r"$q_{23}$", r"$\rho_\mathrm{pred}$", "outcome")]
    cap = (r"Zero-shot screen. Protocol of \citet{Busch2026Memorization}: one zero-shot call per test molecule "
           r"at effort low, nested matches $m_1 \ge m_2 \ge m_3$ at one to three significant figures among the "
           r"$n_\mathrm{usable}$ labels with at least three significant figures, three-figure hit rate, "
           r"retention $R_{12}=m_2/m_1$ with its label-only floor, Benjamini--Hochberg $q$-values of the two "
           r"retention tests (``--'' = fewer than 15 preceding matches, untestable), and the Spearman "
           r"correlation between zero-shot prediction and label. ESOL only: a positive answer to a molecule with a "
           r"negative label is read as a solubility in mol/L and converted to logS before scoring (converted answers: "
           + ", ".join(f"{LLM_NAMES[m]} {int(Z[(Z.dataset == 'esol') & (Z.model == m)].n_unit_converted.iloc[0])}"
                       for m in LLM_ORDER) + r"). "
           r"Calls of 2026-10-09 (GPT-6 Astra 2026-10-13); "
           r"\texttt{results/contamination/zeroshot/cells.csv}.")
    long_table("a05_zeroshot", "tab:zeroshot", cap, r"@{}llrrrrrrrrrrl@{}", hdr, body, size=r"\scriptsize",
               tabcolsep="3.5pt")


def a_dedup(D):
    X = D["dedup"]
    body = []
    for _, r in X.iterrows():
        body.append(row(DS_NAMES[r.dataset], r"\texttt{" + r.inchikey14 + "}", "yes" if r.same_smiles else "no",
                        "yes" if r.same_label else "no", num(r.train_value), num(r.test_value),
                        str(int(r.pre_dedup_pool_position))))
    hdr = [row("dataset", "InChIKey-14", "same SMILES", "same label", "pool value", "test value", "position")]
    counts = X.groupby("dataset").size()
    cap = (r"Training-pool deduplication. Pool molecules whose InChIKey connectivity block (first 14 characters) "
           r"equals that of a test molecule, removed before any context was drawn: "
           + ", ".join(f"{DS_NAMES[d]} {counts.get(d, 0)}" for d in DS)
           + r". Whether the SMILES strings and labels of the pair are identical, both labels, and the position "
             r"of the removed molecule in the seeded pool permutation; molecule identifiers and SMILES strings "
             r"are in \texttt{results/contamination/train\_pool\_dedup.csv}.")
    long_table("a06_dedup", "tab:dedup", cap, r"@{}llccrrr@{}", hdr, body, size=r"\footnotesize", tabcolsep="6pt")


def a_overlap(D):
    O = D["overlap"]
    body = []
    for _, r in O.iterrows():
        body.append(row(DS_NAMES[r.dataset], f"{int(r.draw_a)} vs {int(r.draw_b)}", f"{int(r.n_train_pool):,}",
                        f"{int(r.overlap_first30)} ({num(r.expected_first30, 1)})",
                        f"{int(r.overlap_first60)} ({num(r.expected_first60, 1)})",
                        f"{int(r.overlap_first120)} ({num(r.expected_first120, 1)})"))
    hdr = [row("dataset", "draws", "pool", "first 30", "first 60", "first 120")]
    cap = (r"Overlap of the context draws. Number of molecules shared by the first $N$ molecules of two "
           r"context draws, with the expectation $N^2/\text{pool}$ for independent permutations in parentheses "
           r"(\texttt{results/meta/context\_draw\_overlap.csv}).")
    float_table("a07_draw_overlap", "tab:overlap", cap, r"@{}llrrrr@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="6pt", placement="htbp")


def a_main_per_dataset(D, U):
    for ds in DS:
        body = []
        for m in ORDER:
            for n in SIZES[ds]:
                g = lambda c, d=2: num(get(U, ds, m, n, 1, c), d)
                tok = intc(get(U, ds, m, n, 1, "median_reasoning_tokens")) if m in LLM_ORDER else "--"
                body.append(row(NAMES[m] if n == SIZES[ds][0] else "", f"{n:,}", g("RMSE"), g("MAE"),
                                g("spearman_pred"), g("rho"), g("PICP68"), g("PICP90"), g("NLL"),
                                g("RMSE50"), g("gain50"), tok))
            body.append(r"\addlinespace[2pt]")
        body = body[:-1]
        hdr = [row("method", "$N$", "RMSE", "MAE", r"$\rho_\mathrm{pred}$", r"$\rho(\sigma,|e|)$", r"PICP$_{68}$",
                   r"PICP$_{90}$", "NLL", r"RMSE$_{50}$", r"gain$_{50}$", "rsn tok")]
        cap = (f"Main run, {DS_NAMES[ds]}, per cell. Context draw 1 at every context size $N$: root-mean-square "
               r"and mean absolute error, Spearman correlation between prediction and label ($\rho_\mathrm{pred}$) "
               r"and between stated standard deviation and absolute error, Gaussian coverage of $\pm1\sigma$ and "
               r"$\pm1.645\sigma$, Gaussian negative log-likelihood, RMSE$_{50}$, gain$_{50}$ and the median "
               r"emitted reasoning tokens per call. Language models at effort low with $R=3$ calls per molecule "
               r"(prediction and $\sigma$ = repeat means); baselines averaged over their seeds "
               r"(\texttt{results/icl/cells.csv}, \texttt{selective\_prediction.csv}, "
               r"\texttt{results/baselines/summary.csv}).")
        long_table(f"a1{DS.index(ds)}_main_{ds}", f"tab:main_{ds}", cap, r"@{}lrrrrrrrrrrr@{}", hdr, body,
                   size=r"\scriptsize", tabcolsep="3.5pt")


def a_draws(D, U):
    body = []
    for ds in DS:
        body.append(rf"\multicolumn{{15}}{{@{{}}l}}{{{DS_NAMES[ds]}}}\\")
        for m in ORDER:
            for n in SMALL:
                cells = [NAMES[m] if n == 30 else "", str(n)]
                rm = [get(U, ds, m, n, d, "RMSE") for d in (1, 2, 3)]
                for col in ("RMSE", "rho", "PICP68", "RMSE50"):
                    cells += [num(get(U, ds, m, n, d, col)) for d in (1, 2, 3)]
                cells.append(num(np.std(rm, ddof=1)) if not any(pd.isna(rm)) else "--")
                body.append(row(*cells))
        body.append(r"\addlinespace")
    body = body[:-1]
    hdr = [r" & & \multicolumn{3}{c}{RMSE} & \multicolumn{3}{c}{$\rho(\sigma,|e|)$} & "
           r"\multicolumn{3}{c}{PICP$_{68}$} & \multicolumn{3}{c}{RMSE$_{50}$} & SD\\",
           r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
           row("method", "$N$", *(["d1", "d2", "d3"] * 4), "RMSE")]
    cap = (r"Context draws. RMSE, $\rho(\sigma,|e|)$, PICP$_{68}$ and RMSE$_{50}$ in each of the three nested "
           r"context draws (d1 = the main run) at $N$ = 30, 60 and 120, and the standard deviation of the RMSE "
           r"over the draws. Language models at effort low; baselines averaged over their seeds within a draw "
           r"(\texttt{results/icl/cells.csv}, \texttt{selective\_prediction.csv}, "
           r"\texttt{results/baselines/summary.csv}).")
    long_table("a20_draws", "tab:draws", cap, r"@{}lr" + "r" * 13 + "@{}", hdr, body, size=r"\scriptsize",
               tabcolsep="2.6pt")


def a_selective(D):
    S = D["sel"]
    S = S[(S.family == "LLM") & (S.reasoning == "low") & (S.draw == 1)]
    body = []
    for ds in DS:
        body.append(rf"\multicolumn{{10}}{{@{{}}l}}{{{DS_NAMES[ds]}}}\\")
        for m in LLM_ORDER:
            for n in SIZES[ds]:
                r = S[(S.ds == ds) & (S.model == m) & (S["size"] == n)].iloc[0]
                body.append(row(LLM_NAMES[m] if n == 30 else "", f"{n:,}", str(int(r.n_sigma_distinct)),
                                num(r.RMSE50), rng(r.RMSE50_min, r.RMSE50_max), num(r.gain50),
                                num(r.RMSE80), rng(r.RMSE80_min, r.RMSE80_max), num(r.gain80), num(r.rho)))
        body.append(r"\addlinespace")
    body = body[:-1]
    hdr = [row("model", "$N$", r"$n_\sigma$", r"RMSE$_{50}$", "tie range", r"gain$_{50}$", r"RMSE$_{80}$",
               "tie range", r"gain$_{80}$", r"$\rho(\sigma,|e|)$")]
    cap = (r"Selective prediction with ties. Number of distinct stated standard deviations among the 150 test "
           r"molecules ($n_\sigma$), RMSE of the 50\,\% and 80\,\% of the molecules with the smallest stated "
           r"standard deviation (square root of the expected mean squared error when the molecules tied at the "
           r"cut are selected uniformly at random), the exact range of that RMSE over all tie orders, the gains "
           r"$1-\mathrm{RMSE}_F/\mathrm{RMSE}$ and $\rho(\sigma,|e|)$ with average ranks on the rounded "
           r"$\sigma$. Main run, context draw 1, effort low (\texttt{results/icl/selective\_prediction.csv}).")
    long_table("a21_selective", "tab:selective", cap, r"@{}lrrrlrrlrr@{}", hdr, body, size=r"\scriptsize",
               tabcolsep="3.5pt")


def a_calibration(D):
    K = D["cal"]
    for ds in DS:
        body = []
        A = K[(K.dataset == ds) & (K.subset == "all") & K.signal.isin(["UC1", "sigma"])]
        for m in LLM_ORDER + ["tabpfn", "rf_desc"]:
            for n in SIZES[ds]:
                r = A[(A.model == m) & (A["size"] == n)]
                if not len(r):
                    continue
                r = r.iloc[0]
                body.append(row(NAMES[m] if n == 30 else "", f"{n:,}", num(r.ECE, 3), num(r.ENCE), num(r.s),
                                num(r.s_robust), num(r.PICP68), num(r.PICP90), num(r.NLL),
                                num(r.PICP90_test), num(r.PICP90_scaled), num(r.ECE_test, 3), num(r.ECE_scaled, 3),
                                num(r.ECE_isotonic, 3), num(r.conformal90_width), num(r.conformal90n_width)))
            body.append(r"\addlinespace[2pt]")
        body = body[:-1]
        hdr = [r" & & \multicolumn{7}{c}{full test set (150)} & \multicolumn{5}{c}{held-out half (75)} & "
               r"\multicolumn{2}{c}{conformal 90\,\%}\\",
               r"\cmidrule(lr){3-9}\cmidrule(lr){10-14}\cmidrule(l){15-16}",
               row("signal", "$N$", "ECE", "ENCE", "$s$", r"$s_\mathrm{rob}$", r"PICP$_{68}$", r"PICP$_{90}$",
                   "NLL", r"PICP$_{90}$", r"scaled", "ECE", "scaled", "isot.", "width", "norm."),
               ]
        cap = (f"Calibration suite, {DS_NAMES[ds]}. Stated standard deviation (UC1) of each language model at "
               r"effort low and the $\sigma$ of TabPFN v3.5 and RF (descriptors), seed 0, context draw 1. Full "
               r"test set: PIT-ECE (21-point grid), ENCE (10 bins), variance-scaling factor $s$, its median-based "
               r"version $s_\mathrm{rob}$, Gaussian coverage and NLL. Held-out half: PICP$_{90}$ and PIT-ECE of "
               r"the raw $\sigma$, after variance scaling fitted on the calibration half, and after isotonic "
               r"recalibration. Conformal: mean width of the split-conformal 90\,\% interval on the absolute "
               r"residual and of the normalised version on $|e|/\sigma$ "
               r"(\texttt{results/icl/calibration\_cells.csv}).")
        long_table(f"a3{DS.index(ds)}_calibration_{ds}", f"tab:cal_{ds}", cap, r"@{}lr" + "r" * 14 + "@{}", hdr,
                   body, size=r"\scriptsize", tabcolsep="2.4pt")


def a_uc3(D):
    K = D["cal"]
    S = D["sel"]
    S = S[(S.family == "LLM") & (S.reasoning == "low") & (S.draw == 1)]
    body = []
    for ds in DS:
        body.append(rf"\multicolumn{{9}}{{@{{}}l}}{{{DS_NAMES[ds]}}}\\")
        for m in LLM_ORDER:
            for n in (30, TOP[ds]):
                a = K[(K.dataset == ds) & (K.model == m) & (K["size"] == n) & (K.subset == "all")]
                u1, u3 = a[a.signal == "UC1"].iloc[0], a[a.signal == "UC3"].iloc[0]
                rho1 = S[(S.ds == ds) & (S.model == m) & (S["size"] == n)].rho.iloc[0]   # as in every other table
                body.append(row(LLM_NAMES[m] if n == 30 else "", f"{n:,}", num(100 * u3.share_sigma_zero, 1),
                                num(rho1), num(u3.rho_sigma_err), num(u1.s), num(u3.s, 1),
                                num(u1.NLL_scaled), num(u3.NLL_scaled)))
        body.append(r"\addlinespace")
    body = body[:-1]
    hdr = [r" & & & \multicolumn{2}{c}{$\rho(\sigma,|e|)$} & \multicolumn{2}{c}{$s$} & "
           r"\multicolumn{2}{c}{NLL scaled}\\", r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
           row("model", "$N$", r"UC3 $=0$ \%", "UC1", "UC3", "UC1", "UC3", "UC1", "UC3")]
    cap = (r"Repeat spread against stated standard deviation. Share of test molecules whose three repeat "
           r"predictions are identical (UC3 = 0), and for UC1 and UC3 the rank correlation with the absolute "
           r"error, the variance-scaling factor $s$ and the held-out negative log-likelihood after variance "
           r"scaling, at the smallest and the largest context size. Main run, context draw 1, effort low "
           r"(\texttt{results/icl/calibration\_cells.csv}).")
    long_table("a40_uc3", "tab:uc3", cap, r"@{}lrrrrrrrr@{}", hdr, body, size=r"\scriptsize", tabcolsep="4pt")


def a_bootstrap(D):
    Bt = D["boot"]

    def cell(ds, m, n, draw, stat, base):
        r = Bt[(Bt.dataset == ds) & (Bt.model == m) & (Bt.baseline == base) & (Bt["size"] == n) & (Bt.draw == draw)
               & (Bt.stat == stat)]
        if not len(r):
            return "--"
        r = r.iloc[0]
        return f"{num(r.est, 2, plus=True)} {ci(r.lo, r.hi)}{boot_flag(D, ds, m, n, stat, draw, base)}"

    for base, name, label in (("tabpfn", "a50_bootstrap_tabpfn", "tab:boot_tabpfn"),
                              ("gp_tanimoto", "a51_bootstrap_gp", "tab:boot_gp")):
        body = []
        for ds in DS:
            body.append(rf"\multicolumn{{6}}{{@{{}}l}}{{{DS_NAMES[ds]}}}\\")
            settings = [(30, 0), (60, 0), (120, 0), (TOP[ds], 1)]
            for m in LLM_ORDER:
                for n, draw in settings:
                    lab = f"{n:,}" + (" c" if draw == 0 else " d1")
                    body.append(row(LLM_NAMES[m] if n == 30 else "", lab,
                                    *[cell(ds, m, n, draw, st, base) for st in ("RMSE", "rho", "RMSE50", "gain50")]))
            body.append(r"\addlinespace")
        body = body[:-1]
        hdr = [row("model", "$N$", r"$\Delta$RMSE", r"$\Delta\rho(\sigma,|e|)$", r"$\Delta$RMSE$_{50}$",
                   r"$\Delta$gain$_{50}$")]
        bn = BASE_NAMES[base]
        cap = (rf"Paired bootstrap against {bn}. Difference language model minus {bn} (seed 0) with the "
               r"percentile 95\,\% interval from 1,000 resamples of the 150 test molecules (one index matrix per "
               r"dataset, \texttt{default\_rng(0)}). ``c'' = combined over the three context draws (the "
               r"difference is averaged over the draws within each resample), ``d1'' = context draw 1 at the "
               r"largest context size. $\dagger$ / $\ddagger$: interval excludes 0 in favour of the language "
               r"model / of the baseline (negative $\Delta$RMSE and $\Delta$RMSE$_{50}$, positive $\Delta\rho$ "
               r"and $\Delta$gain$_{50}$ favour the language model). \texttt{results/icl/bootstrap\_cells.csv}.")
        long_table(name, label, cap, r"@{}lrllll@{}", hdr, body, size=r"\scriptsize", tabcolsep="3pt")
    # Sonnet vs TabPFN, top rung (calibration_bootstrap.csv)
    CB = D["calboot"]
    body = []
    for ds in DS:
        for met, lab in (("RMSE", "RMSE"), ("mean_abs_err", "MAE"), ("ECE", "PIT-ECE")):
            r = CB[(CB.dataset == ds) & (CB.metric == met)].iloc[0]
            d = 3 if met == "ECE" else 3
            body.append(row(DS_NAMES[ds] if met == "RMSE" else "", f"{int(r['size']):,}", lab, num(r.sonnet, d),
                            num(r.tabpfn, d), f"{num(r['diff'], d, plus=True)} {ci(r.ci_lo, r.ci_hi, d)}"))
    hdr = [row("dataset", "$N$", "metric", "Sonnet 5.5", "TabPFN v3.5", r"difference [95\,\% CI]")]
    cap = (r"Sonnet 5.5 against TabPFN v3.5 at the largest context size. Root-mean-square error, mean absolute "
           r"error and PIT-ECE with the paired-bootstrap difference (same resampling as the tables above; "
           r"TabPFN seed 0, context draw 1; \texttt{results/icl/calibration\_bootstrap.csv}).")
    float_table("a52_bootstrap_top", "tab:boot_top", cap, r"@{}lrlrrl@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="5pt", placement="htbp")


def a_esol_subsets(D):
    K = D["cal"]
    E = K[(K.dataset == "esol") & K.signal.isin(["UC1", "sigma"]) & K.subset.isin(["recalled", "never_recalled"])]
    body = []
    for m in LLM_ORDER + ["tabpfn", "rf_desc"]:
        for n in SIZES["esol"]:
            r1 = E[(E.model == m) & (E["size"] == n) & (E.subset == "recalled")]
            r0 = E[(E.model == m) & (E["size"] == n) & (E.subset == "never_recalled")]
            if not len(r1):
                continue
            r1, r0 = r1.iloc[0], r0.iloc[0]
            body.append(row(NAMES[m] if n == 30 else "", f"{n:,}", num(r1.RMSE), num(r0.RMSE), num(r1.PICP68),
                            num(r0.PICP68), num(r1.s), num(r0.s), num(r1.ECE, 3), num(r0.ECE, 3)))
        body.append(r"\addlinespace[2pt]")
    body = body[:-1]
    n1 = int(E[E.subset == "recalled"].n.iloc[0])
    n0 = int(E[E.subset == "never_recalled"].n.iloc[0])
    hdr = [r" & & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{PICP$_{68}$} & \multicolumn{2}{c}{$s$} & "
           r"\multicolumn{2}{c}{ECE}\\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
           row("signal", "$N$", *(["rec.", "never"] * 4))]
    cap = (rf"ESOL by recall. Test molecules whose value at least one panel model reproduced to three "
           rf"significant figures in the zero-shot screen (rec., {n1} molecules) and those no model reproduced "
           rf"(never, {n0} molecules); the remaining {150 - n1 - n0} labels have fewer than three significant "
           r"figures and are in neither subset. Language models at effort low, TabPFN v3.5 and RF "
           r"(descriptors) seed 0, context draw 1 (\texttt{results/icl/calibration\_cells.csv}).")
    long_table("a60_esol_subsets", "tab:esol_subsets", cap, r"@{}lrrrrrrrrr@{}", hdr, body, size=r"\scriptsize",
               tabcolsep="4pt")


def a_ladder(D):
    L = D["ladder"]
    S = D["sel"]
    S = S[(S.family == "LLM") & (S.draw == 1)].rename(columns={"ds": "dataset", "reasoning": "effort"})
    L = L.merge(S[["dataset", "model", "size", "effort", "rho"]], on=["dataset", "model", "size", "effort"],
                how="left", validate="one_to_one")
    assert L.rho.notna().all(), "selective_prediction.csv lacks ladder cells"
    body = []
    for ds in DS:
        body.append(rf"\multicolumn{{13}}{{@{{}}l}}{{{DS_NAMES[ds]}}}\\")
        for m in LADDER_ORDER:
            first = True
            for n in (30, TOP[ds]):
                for e in EFFORTS:
                    r = L[(L.dataset == ds) & (L.model == m) & (L["size"] == n) & (L.effort == e)]
                    if not len(r):
                        continue
                    r = r.iloc[0]
                    body.append(row(LLM_NAMES[m] if first else "", f"{n:,}", r"\texttt{" + e + "}", num(r.RMSE),
                                    num(r.spearman_pred), num(r.rho), num(r.PICP68), num(r.PICP90),
                                    num(r.NLL), intc(r.median_reasoning_tokens), intc(r.median_completion_tokens),
                                    str(int(r.n_mol)), num(r.trunc_rate)))
                    first = False
            body.append(r"\addlinespace[2pt]")
        body[-1] = r"\addlinespace"
    body = body[:-1]
    hdr = [row("model", "$N$", "effort", "RMSE", r"$\rho_\mathrm{pred}$", r"$\rho(\sigma,|e|)$", r"PICP$_{68}$",
               r"PICP$_{90}$", "NLL", "rsn tok", "compl tok", r"$n_\mathrm{mol}$", "trunc")]
    cap = (r"Reasoning ladder. Six models, requested efforts from \texttt{none} to \texttt{high} (levels an "
           r"endpoint rejects are absent), smallest and largest context size, $R=3$ calls per molecule, context "
           r"draw 1; \texttt{low} is the main run. Median emitted reasoning and completion tokens per call, "
           r"number of molecules with at least one parsed answer and share of calls truncated at the "
           r"32,000-token completion cap (\texttt{results/icl/ladder\_cells.csv}).")
    long_table("a70_ladder", "tab:ladder", cap, r"@{}lrlrrrrrrrrrr@{}", hdr, body, size=r"\scriptsize",
               tabcolsep="3pt")


def a_basefits(D):
    rows = []
    for f in glob.glob(os.path.join(RES, "baselines", "*.meta.json")):
        d = json.load(open(f, encoding="utf8"))
        d.pop("train_ids", None), d.pop("test_ids", None)
        d["draw"] = d.get("draw", 1)
        rows.append(d)
    M = pd.DataFrame(rows)
    M = M[M.seed == 0]
    body = []
    for ds in DS:
        for n in SIZES[ds]:
            def one(meth, draw=1):
                r = M[(M.dataset == ds) & (M["size"] == n) & (M.method == meth) & (M.draw == draw)]
                return r.iloc[0] if len(r) else None
            gp, tp, mlp = one("gp_tanimoto"), one("tabpfn"), one("mlp_ens")

            def gpcell(g):
                if g is None:
                    return "--", "--", ""
                a, s2 = g.gp_amplitude / g.gp_var_y, g.gp_noise_var / g.gp_var_y
                flag = []
                if np.isclose(a, 1e-3, rtol=1e-3):
                    flag.append("$a$ lower")
                if np.isclose(s2, 1e-4, rtol=1e-3):
                    flag.append(r"$s^2$ lower")
                if np.isclose(a, 1e2, rtol=1e-3):
                    flag.append("$a$ upper")
                if np.isclose(s2, 10, rtol=1e-3):
                    flag.append(r"$s^2$ upper")
                return f"{a:.3g}".replace("e-0", r"e$-$").replace("e+0", "e"), f"{s2:.3g}".replace("e-0", r"e$-$"), \
                    ", ".join(flag)
            ga, gs, gf = gpcell(gp)
            bounds_draws = []
            if n in SMALL:
                for dd in (2, 3):
                    g2 = one("gp_tanimoto", dd)
                    bounds_draws.append(gpcell(g2)[2] or "interior")
            ep = f"{min(mlp.epochs)}--{max(mlp.epochs)}" if mlp is not None else "--"
            body.append(row(DS_NAMES[ds] if n == 30 else "", f"{n:,}", ga, gs, gf or "interior",
                            " / ".join(bounds_draws) if bounds_draws else "--", ep,
                            str(int(tp.n_features)) if tp is not None else "--"))
        body.append(r"\addlinespace")
    body = body[:-1]
    hdr = [r" & & \multicolumn{4}{c}{GP (Tanimoto), ML-II} & MLP & TabPFN\\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){7-7}\cmidrule(l){8-8}",
           row("dataset", "$N$", r"$a/\mathrm{var}(y)$", r"$s^2/\mathrm{var}(y)$", "bound (d1)", "bound (d2 / d3)",
               "epochs", "features")]
    cap = (r"Fitted baseline settings. GP: amplitude $a$ and noise variance $s^2$ selected by maximising the log "
           r"marginal likelihood (14 starts, bounds $a\in[10^{-3},10^{2}]\,\mathrm{var}(y)$ and "
           r"$s^2\in[10^{-4},10^{1}]\,\mathrm{var}(y)$), in units of the training-label variance, and the bound "
           r"the optimum lies on in context draw 1 and in draws 2 and 3 ($s^2$ lower = interpolating solution; "
           r"$a$ lower = noise-only solution, i.e. the trivial predictor). MLP ensemble: range of training epochs "
           r"over the five members (Chemprop: 50 epochs everywhere). TabPFN v3.5: number of RDKit descriptors "
           r"after the per-cell filter. Seed 0 (\texttt{results/baselines/*.meta.json}).")
    long_table("a80_baseline_fits", "tab:basefits", cap, r"@{}lrrrllrr@{}", hdr, body, size=r"\scriptsize",
               tabcolsep="4pt")


def a_cost(D):
    C = D["cells_all"]
    Z = D["zs"]
    body = []
    runs = [("main run", (C.reasoning == "low") & (C.draw == 1)),
            ("context draws 2 and 3", (C.reasoning == "low") & (C.draw > 1)),
            ("reasoning ladder (efforts other than low)", C.reasoning != "low")]
    tot_calls, tot_usd = 0, 0.0
    for lab, mask in runs:
        g = C[mask]
        body.append(rf"\multicolumn{{4}}{{@{{}}l}}{{{lab}}}\\")
        for m in LLM_ORDER:
            h = g[g.model == m]
            if not len(h):
                continue
            cs = h[h["size"] >= 120].cache_share.median()
            body.append(row(LLM_NAMES[m], f"{int(h.n_calls.sum()):,}", num(h.cost_usd.sum()), num(cs)))
        tot_calls += int(g.n_calls.sum())
        tot_usd += g.cost_usd.sum()
        if lab.startswith("context draws"):
            # duplicated (molecule, repeat) records: billed, stored in the raw files, dropped by analyze_icl.py
            rc, ru = 0, 0.0
            for f in glob.glob(os.path.join(RES, "icl", "*__d[23].jsonl")):
                for line in open(f, encoding="utf8"):
                    rc += 1
                    ru += json.loads(line).get("cost") or 0
            dup_c, dup_u = rc - int(g.n_calls.sum()), ru - g.cost_usd.sum()
            body.append(row("duplicated records (billed, not analysed)", f"{dup_c:,}", num(dup_u), "--"))
            tot_calls += dup_c
            tot_usd += dup_u
            body.append(row(r"\quad total", f"{int(g.n_calls.sum()) + dup_c:,}", num(g.cost_usd.sum() + dup_u), ""))
        else:
            body.append(row(r"\quad total", f"{int(g.n_calls.sum()):,}", num(g.cost_usd.sum()), ""))
        body.append(r"\addlinespace")
    body.append(r"\multicolumn{4}{@{}l}{zero-shot screen}\\")
    for m in LLM_ORDER:
        h = Z[Z.model == m]
        body.append(row(LLM_NAMES[m], f"{int(h.n_calls.sum()):,}", num(h.cost_usd.sum()), "--"))
    body.append(row(r"\quad total", f"{int(Z.n_calls.sum()):,}", num(Z.cost_usd.sum()), ""))
    tot_calls += int(Z.n_calls.sum())
    tot_usd += Z.cost_usd.sum()
    body.append(r"\addlinespace")
    # probes
    pc, pu = 0, 0.0
    for f in sorted(glob.glob(os.path.join(RES, "meta", "*probe_2026-10-0*.jsonl"))):
        for line in open(f, encoding="utf8"):
            r = json.loads(line)
            pc += 1
            pu += r.get("cost") or 0
    dc, du = 0, 0.0
    for f in sorted(glob.glob(os.path.join(RES, "meta", "diagnostics", "*.jsonl"))):
        for line in open(f, encoding="utf8"):
            r = json.loads(line)
            dc += 1
            du += r.get("cost") or 0
    body.append(row("effort and ICL probes (all registry models)", f"{pc:,}", num(pu), "--"))
    body.append(row("context-size diagnostic, Claude Opus 5.5", f"{dc:,}", num(du), "--"))
    tot_calls += pc + dc
    tot_usd += pu + du
    body.append(r"\midrule")
    body.append(row("all paid calls", f"{tot_calls:,}", num(tot_usd), ""))
    hdr = [row("run and model", "calls", "USD", "cache share")]
    cap = (r"Calls and cost. Billed cost (OpenRouter \texttt{usage.cost}) and number of stored calls per run and "
           r"model (stored, deduplicated calls; the duplicated records of the context draws are listed separately); cache share = median share of prompt "
           r"tokens served from the prompt cache over the cells with $N \ge 120$. The probes cover every "
           r"registry model tested before the panel was fixed (\texttt{results/icl/cells.csv}, "
           r"\texttt{results/contamination/zeroshot/cells.csv}, \texttt{results/meta/*probe*.jsonl}).")
    long_table("a90_cost", "tab:cost", cap, r"@{}lrrr@{}", hdr, body, size=r"\footnotesize", tabcolsep="6pt")


# ================================================================ compact tables of the restructured manuscript
# (docs/PAPER_OUTLINE_v3.md, 2026-10-14): one main-text summary table (c_t2_summary) and the appendix tables
# B-H. File names c_*.tex; the per-cell tables above stay in place for the repository.
CLEAN_NAMES = {"lipophilicity": "Lipophilicity", "bace": "BACE", "antiviral": "Antiviral"}


def _best(U, ds, n, col, lowest=True):
    """tag of the best language model at (ds, n) in context draw 1."""
    g = U[(U.dataset == ds) & (U["size"] == n) & (U.draw == 1) & U.model.isin(LLM_ORDER)].set_index("model")[col]
    return g.idxmin() if lowest else g.idxmax()


def c_t2_summary(D, U):
    """Table 2 of the main text: one row per clean dataset x N (30, largest); best language model and TabPFN
    for RMSE, rho(sigma,|e|) and RMSE50; draw 1 [range over draws at N = 30]; bootstrap marker of the best
    language model against TabPFN (combined over draws at N = 30, draw 1 at the largest N)."""
    body = []
    for ds in CLEAN:
        for n in (30, TOP[ds]):
            cells = [CLEAN_NAMES[ds], f"{n:,}"]
            draw = 0 if n == 30 else 1
            for col, stat, lowest in (("RMSE", "RMSE", True), ("rho", "rho", False), ("RMSE50", "RMSE50", True)):
                m = _best(U, ds, n, col, lowest)
                v = with_range(U, ds, m, n, col) + boot_flag(D, ds, m, n, stat, draw)
                cells.append(f"{LLM_NAMES[m]} {v}")
                cells.append(with_range(U, ds, "tabpfn", n, col))
            body.append(row(*cells))
        if ds != CLEAN[-1]:
            body.append(r"\addlinespace")
    hdr = [r" & & \multicolumn{2}{c}{RMSE (Q1)} & \multicolumn{2}{c}{$\rho(\sigma,|e|)$ (Q2)} & "
           r"\multicolumn{2}{c}{RMSE$_{50}$ (Q3)}\\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
           row("dataset", "$N$", "best language model", "TabPFN", "best language model", "TabPFN",
               "best language model", "TabPFN")]
    cap = (r"Best language model against TabPFN at the smallest and the largest context size on the three clean "
           r"datasets. Root-mean-square error on the 150 test molecules (Q1), Spearman rank correlation between "
           r"the stated standard deviation and the absolute error (Q2) and RMSE$_{50}$, the RMSE of the half of "
           r"the test molecules with the smallest stated standard deviation (Q3); in each column the language "
           r"model that is best on that quantity in context draw 1. Values are context draw 1, at $N=30$ with "
           r"the range over the three context draws in brackets; language models at effort low, TabPFN v3.5 "
           r"averaged over its seeds within a draw. $\dagger$ / $\ddagger$: the paired-bootstrap 95\,\% interval "
           r"of the difference between the named language model and TabPFN (seed 0; combined over the three "
           r"draws at $N=30$, draw 1 at the largest $N$) excludes 0 in favour of the language model / of TabPFN. "
           r"Every model and context size: Appendix~\ref{app:F} and the repository.")
    float_table("c_t2_summary", "tab:summary", cap, r"@{}llllllll@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="4pt", resize=True, placement="tbp")


def _ladder_token_table(D):
    """median emitted reasoning tokens per model and effort (min-max-median over the 8 ladder cells of a model)."""
    L = D["ladder"]
    L = L[L.model.isin(LADDER_ORDER)]
    return L.pivot_table(index="model", columns="effort", values="median_reasoning_tokens", aggfunc=["min", "max", "median"])


def _dosability(tk, m):
    """plain verdict from the median emitted tokens per level."""
    med = {e: tk[("median", e)].get(m, np.nan) for e in EFFORTS if ("median", e) in tk.columns}
    med = {e: v for e, v in med.items() if not pd.isna(v)}
    if all(v == 0 for v in med.values()):
        return "no reasoning tokens at any level"
    nz = [e for e in EFFORTS if e in med and med[e] > 0]
    if nz[0] not in ("none", "minimal"):
        return f"tokens from \\texttt{{{nz[0]}}} on"
    lo = med.get("minimal", med.get("low"))
    if med["medium"] <= 1.3 * lo and med["high"] >= 2 * lo:
        return r"rises only at \texttt{high}"
    if med["high"] >= 2 * lo and med["medium"] > lo:
        return "rises with the level"
    return "flat"


LADDER_SHORT = {"no reasoning tokens at any level": "none at any effort",
                r"tokens from \texttt{medium} on": "from medium on", r"tokens from \texttt{high} on": "only at high",
                r"rises only at \texttt{high}": "rise only at high", "rises with the level": "rise with the effort",
                "flat": "flat"}


def c_b_panel(D, U):
    """Appendix B: (1) panel and endpoints with prices and the dosability verdict; (2) datasets, splits,
    deduplication and draw overlap."""
    body = []
    for m in LLM_ORDER:
        r = D["registry"][m]
        tier = "flex" if r["service_tier"] == "flex" else "standard"
        body.append(row(LLM_NAMES[m], r["vendor"], r["price_endpoint"].replace("_", r"\_"), tier,
                        f"{r['price_in']:g}", f"{r['price_cache_read']:g}", f"{r['price_out']:g}"))
    hdr = [r" & & & & \multicolumn{3}{c}{USD per million tokens}\\", r"\cmidrule(l){5-7}",
           row("model", "vendor", "provider endpoint", "tier", "input", "cached", "output")]
    cap = (r"Endpoints and prices. The provider endpoint on OpenRouter to which the requests of each language model "
           r"are pinned, with fallback to other hosts disabled, its service tier (flex: the provider's flex tier, "
           r"billed at half the standard price), and the price per million input, cached-input and output tokens "
           r"at that endpoint. The reasoning efforts an endpoint accepts and the emitted reasoning tokens are in "
           r"Table~\ref{tab:panel}b. Prices fetched from OpenRouter on 2026-10-08 "
           r"(\texttt{src/registry/models\_v2.json}, which also gives the OpenRouter ids).")
    float_table("c_b1_panel", "tab:b_panel", cap, r"@{}llllrrr@{}", hdr, body, size=r"\small",
                tabcolsep="5pt", placement="tbp")
    # datasets
    sp, O, X = D["splits"], D["overlap"], D["dedup"]
    body = []
    for ds in DS:
        s = sp[ds]
        o = O[O.dataset == ds]
        nd = int((X.dataset == ds).sum())
        body.append(row(DS_NAMES[ds], f"{s['n_total']:,}", f"{s['excluded_present']} of {s['positions_checked']}",
                        str(s["n_test"]), str(nd), f"{s['n_train_pool']:,}",
                        f"{int(o.overlap_first120.max())} ({num(o.expected_first120.iloc[0], 1)})"))
    hdr = [row("dataset", "molecules", "corpus filter", "test", "dedup.", "pool", "draw overlap (exp.)")]
    cap = (r"Datasets and splits. Number of molecules; test candidates rejected by the corpus filter (present in a "
           r"searched corpus) among the positions checked; test-set size; pool molecules removed because their "
           r"InChIKey connectivity block equals that of a test molecule (dedup.); training pool after removal; and "
           r"the largest number of molecules shared by the first 120 molecules of two context draws, with the "
           r"expectation for independent permutations in parentheses. Largest context size, corpus presence and "
           r"screen outcome: Table~\ref{tab:panel}a and Fig.~\ref{fig:contamination} "
           r"(\texttt{data/splits/*\_v2*.json}, \texttt{results/meta/context\_draw\_overlap.csv}).")
    float_table("c_b2_datasets", "tab:b_datasets", cap, r"@{}lrrrrrr@{}", hdr, body, size=r"\small",
                tabcolsep="6pt", placement="tbp")


def c_c_baselines(D):
    """Appendix C: configuration per classical method."""
    def meta(m):
        return json.load(open(sorted(glob.glob(os.path.join(RES, "baselines", f"{m}__lipophilicity__N30.meta.json")))[0],
                              encoding="utf8"))
    B = D["base_seeds"]
    seeds = B.groupby(["method", "draw"]).seed.apply(lambda s: ",".join(map(str, sorted(s.unique())))).unstack()
    nf = [json.load(open(f, encoding="utf8"))["n_features"] for f in glob.glob(os.path.join(RES, "baselines", "tabpfn__*.meta.json"))]
    feat = {"tabpfn": f"RDKit 2D descriptors ({min(nf)}--{max(nf)} after the filter)",
            "gp_tanimoto": "Morgan r = 2, 2,048 bits", "rf_desc": "RDKit 2D descriptors (as TabPFN)",
            "rf": "Morgan r = 2, 2,048 bits", "chemprop_ens": "molecular graph (chemprop 2 featurizer)",
            "mlp_ens": "Morgan r = 2, 2,048 bits", "knn_tanimoto": "Morgan r = 2, 2,048 bits", "trivial": "--"}
    hp = {"tabpfn": f"v3.5, tabpfn {meta('tabpfn')['tabpfn']}, default estimators, CPU",
          "gp_tanimoto": r"$k=a\,T+s^2\delta$, ML-II from " + str(len(meta("gp_tanimoto")["gp_starts"])) + " starts",
          "rf_desc": f"{meta('rf_desc')['n_trees']} trees, scikit-learn defaults",
          "rf": f"{meta('rf')['n_trees']} trees, scikit-learn defaults",
          "chemprop_ens": f"{meta('chemprop_ens')['n_members']} members, chemprop {meta('chemprop_ens')['chemprop']} defaults, {meta('chemprop_ens')['max_epochs']} epochs",
          "mlp_ens": f"{len(meta('mlp_ens')['member_seeds'])} members, 2 $\\times$ 256 ReLU, dropout {meta('mlp_ens')['dropout']}, Adam {meta('mlp_ens')['lr']:g}",
          "knn_tanimoto": f"$k={meta('knn_tanimoto')['k']}$, Tanimoto-weighted mean", "trivial": "training mean"}
    sig = {"tabpfn": r"$(q_{84}-q_{16})/2$ of the predictive distribution", "gp_tanimoto": "posterior predictive SD incl. noise",
           "rf_desc": "SD over the trees", "rf": "SD over the trees", "chemprop_ens": "SD over the members",
           "mlp_ens": "SD over the members", "knn_tanimoto": "SD of the 5 neighbour labels (floor 0.01)",
           "trivial": "training SD"}
    body = []
    for m in BASE_ORDER:
        s1 = seeds.loc[m, 1] if m in seeds.index else "--"
        s23 = seeds.loc[m].get(2, "--") if m in seeds.index else "--"
        body.append(row(BASE_NAMES[m], feat[m], hp[m], sig[m], f"{s1} / {s23}"))
    hdr = [row("method", "features", "settings", r"$\sigma$", "seeds")]
    cap = (r"Classical baselines. Input features, fixed settings, the definition of the uncertainty $\sigma$ "
           r"(SD: standard deviation) and the seeds run in context draw 1 / in draws 2 and 3 (values are averaged "
           r"over the seeds of a draw; "
           r"the paired bootstrap uses seed 0). Every method is trained on exactly the "
           r"in-context molecules of a cell and predicts the same 150 test molecules "
           r"(\texttt{results/baselines/*.meta.json}, \texttt{summary.csv}).")
    pc = r">{\raggedright\arraybackslash}p"
    float_table("c_c_baselines", "tab:c_baselines", cap,
                rf"@{{}}{pc}{{2.0cm}}{pc}{{3.8cm}}{pc}{{3.9cm}}{pc}{{3.1cm}}l@{{}}", hdr, body, size=r"\small",
                tabcolsep="4pt", placement="tbp")


def c_d_contamination(D):
    """Appendix D: (1) corpus presence and zero-shot outcome per dataset x model; (2) ESOL recalled /
    never-recalled decomposition at the largest N."""
    Z, K = D["zs"], D["corpus"]
    body = []
    pres = []
    for ds in DS:
        k = K[(K.dataset == ds) & (K.subset == "v2test150")]
        pres.append(f"{num(k.pct_present_smiles.max(), 1)}\\,\\%")
    body.append(row("test molecules in a corpus (SMILES, max over 7 indexes)", *pres))
    body.append(r"\midrule")
    for m in LLM_ORDER:
        cells = [LLM_NAMES[m]]
        for ds in DS:
            z = Z[(Z.dataset == ds) & (Z.model == m)].iloc[0]
            out = "clean" if z.verdict == "clean" else f"flagged ({z.regime})" if isinstance(z.regime, str) else "flagged"
            cells.append(f"{num(z.hit3, 1)}\\,\\% {out}")
        body.append(row(*cells))
    hdr = [row("", *[DS_NAMES[d] for d in DS])]
    nus = {ds: int(Z[Z.dataset == ds].n_usable.iloc[0]) for ds in DS}
    cap = (r"Contamination screen per dataset and model. First row: share of the 150 test molecules that occur in "
           r"any of the seven searched pre-training corpora as a SMILES string (maximum over the corpora; the "
           r"InChIKey form gives the same result on the clean datasets). Model rows: three-significant-figure hit "
           r"rate (hit$_3$, share of the labels with at least three significant figures that the model reproduced "
           r"zero-shot; usable labels " + ", ".join(f"{DS_NAMES[d]} {nus[d]}" for d in DS) + r") and the outcome "
           r"of the retention test with Benjamini--Hochberg correction after \citet{Busch2026Memorization}, with "
           r"the regime of a flagged cell. Calls of 2026-10-09 (GPT-6 Astra 2026-10-13) at effort low "
           r"(\texttt{results/contamination/corpus/summary\_by\_subset.csv}, \texttt{zeroshot/cells.csv}).")
    float_table("c_d1_contamination", "tab:d_contamination", cap, r"@{}lllll@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="4pt", resize=True, placement="tbp")
    # ESOL subsets at the largest N
    Kc = D["cal"]
    E = Kc[(Kc.dataset == "esol") & Kc.signal.isin(["UC1", "sigma"]) & (Kc["size"] == TOP["esol"])]
    body = []
    for m in LLM_ORDER + ["tabpfn"]:
        r1 = E[(E.model == m) & (E.subset == "recalled")].iloc[0]
        r0 = E[(E.model == m) & (E.subset == "never_recalled")].iloc[0]
        ra = E[(E.model == m) & (E.subset == "all")].iloc[0]
        z = Z[(Z.dataset == "esol") & (Z.model == m)]
        hit = f"{num(z.hit3.iloc[0], 1)}" if len(z) else "--"
        # s of calibration_cells.csv = sqrt(mean(e^2 / sigma^2)), sigma floored at 0.01 = the RMS of e / sigma
        body.append(row(NAMES[m], hit, num(ra.RMSE), num(r1.RMSE), num(r0.RMSE), num(r0.RMSE / r1.RMSE, 1),
                        num(ra.s), num(r1.s), num(r0.s)))
    n1 = int(E[E.subset == "recalled"].n.iloc[0])
    n0 = int(E[E.subset == "never_recalled"].n.iloc[0])
    hdr = [r" & & \multicolumn{4}{c}{RMSE} & \multicolumn{3}{c}{RMS $e/\sigma$}\\",
           r"\cmidrule(lr){3-6}\cmidrule(l){7-9}",
           row("model", r"hit$_3$ \%", "all", "rec.", "never", "ratio", "all", "rec.", "never")]
    cap = (rf"ESOL by recall at 976 in-context molecules. Test molecules whose value at least one panel model "
           rf"reproduced to three significant figures in the zero-shot screen (rec., {n1} molecules) against those no "
           rf"model reproduced (never, {n0}); the remaining {150 - n1 - n0} labels have fewer than three significant "
           r"figures and are in neither subset. Root-mean-square error with the ratio never / recalled, and root "
           r"mean square of the error in units of the stated $\sigma$, $(\frac1n\sum_i e_i^2/\sigma_i^2)^{1/2}$ with "
           r"$\sigma$ floored at 0.01, on all 150 molecules and on the two subsets; language models at effort low, "
           r"TabPFN v3.5 seed 0, context draw 1 (\texttt{results/icl/calibration\_cells.csv}, columns RMSE and \texttt{s}).")
    float_table("c_d2_esol_subsets", "tab:d_esol", cap, r"@{}lrrrrrrrr@{}", hdr, body, size=r"\small",
                tabcolsep="4pt", placement="tbp")


def c_e_calibration(D):
    """Appendix E: (1) calibration per model over the 18 clean cells; (2) UC3 against UC1."""
    K = D["cal"]
    U1 = K[(K.subset == "all") & K.signal.isin(["UC1", "sigma"]) & K.dataset.isin(CLEAN)]
    body = []
    for m in LLM_ORDER + ["tabpfn", "rf_desc"]:
        g = U1[U1.model == m]
        body.append(row(NAMES[m], num(g.s.median()), f"{num(g.s.min())}--{num(g.s.max())}", str(int((g.s > 1).sum())),
                        num(g.PICP68.median()), num(g.PICP90.median()),
                        num(g.PICP90_test.median()), num(g.PICP90_scaled.median()),
                        num(g.ECE_test.median(), 3), num(g.ECE_scaled.median(), 3), num(g.ECE_isotonic.median(), 3)))
    hdr = [r" & \multicolumn{3}{c}{$s$} & \multicolumn{2}{c}{all 150} & \multicolumn{5}{c}{held-out 75}\\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){7-11}",
           r" & & & & & & \multicolumn{2}{c}{PICP$_{90}$} & \multicolumn{3}{c}{PIT-ECE}\\",
           r"\cmidrule(lr){7-8}\cmidrule(l){9-11}",
           row("model", "median", "range", "$>1$", r"PICP$_{68}$", r"PICP$_{90}$", "raw", "scaled", "raw", "scaled",
               "isotonic")]
    cap = (r"Calibration of the stated standard deviation on the clean datasets. Per model over its 18 clean cells "
           r"(3 datasets $\times$ 6 context sizes, context draw 1): median and range of the variance-scaling factor "
           r"$s$ ($s>1$: $\sigma$ too narrow) with the number of the 18 cells above 1, median coverage of $\pm1\sigma$ and $\pm1.645\sigma$ on all "
           r"150 molecules, and on the held-out half of 75 molecules the median PICP$_{90}$ and PIT-ECE before and "
           r"after variance scaling fitted on the other 75, and the PIT-ECE after isotonic recalibration. "
           r"Language models at effort low; TabPFN v3.5 and RF (descriptors) seed 0 "
           r"(\texttt{results/icl/calibration\_cells.csv}).")
    float_table("c_e1_calibration", "tab:e_calibration", cap, r"@{}lrlrrrrrrrr@{}", hdr, body, size=r"\small",
                tabcolsep="3.5pt", placement="tbp")
    u1 = K[(K.signal == "UC1") & (K.subset == "all")].set_index(["dataset", "model", "size"])
    u3 = K[(K.signal == "UC3") & (K.subset == "all")].set_index(["dataset", "model", "size"])
    body = []
    for m in LLM_ORDER:
        a = u1.xs(m, level=1)
        b = u3.xs(m, level=1)
        ac, bc = a.loc[CLEAN], b.loc[CLEAN]
        body.append(row(LLM_NAMES[m], num(100 * b.share_sigma_zero.median(), 1),
                        f"{num(100 * b.share_sigma_zero.min(), 1)}--{num(100 * b.share_sigma_zero.max(), 1)}",
                        f"{num(bc.s.median(), 1)}", str(int((b.NLL_scaled > a.NLL_scaled).sum())),
                        num(ac.rho_sigma_err.median()), num(bc.rho_sigma_err.median()),
                        str(int((bc.rho_sigma_err > ac.rho_sigma_err).sum())),
                        num(a.loc['esol'].rho_sigma_err.median()), num(b.loc['esol'].rho_sigma_err.median()),
                        str(int((b.loc['esol'].rho_sigma_err > a.loc['esol'].rho_sigma_err).sum()))))
    hdr = [r" & \multicolumn{2}{c}{UC3 $=0$ (\%)} & & & \multicolumn{3}{c}{clean (18 cells)} & "
           r"\multicolumn{3}{c}{ESOL (6 cells)}\\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
           row("model", "median", "range", r"$s$", "NLL", r"$\rho_{\mathrm{UC1}}$", r"$\rho_{\mathrm{UC3}}$", "UC3",
               r"$\rho_{\mathrm{UC1}}$", r"$\rho_{\mathrm{UC3}}$", "UC3")]
    cap = (r"Repeat spread (UC3) against the stated standard deviation (UC1). Per model over its 24 main-run cells: "
           r"share of test molecules whose three repeat predictions are identical (UC3 $=0$, median and range), the "
           r"median variance-scaling factor $s$ of UC3 on the clean cells, the number of the 24 cells in which UC3 "
           r"has the higher held-out negative log-likelihood than UC1 after variance scaling (NLL), and on the clean "
           r"datasets and on ESOL the median rank correlation $\rho(\sigma,|e|)$ of UC1 and of UC3 and the number of "
           r"cells in which UC3 ranks the errors better (UC3). Context draw 1, effort low "
           r"(\texttt{results/icl/calibration\_cells.csv}).")
    float_table("c_e2_uc3", "tab:e_uc3", cap, r"@{}lrlrrrrrrrr@{}", hdr, body, size=r"\small",
                tabcolsep="4pt", placement="tbp")


def c_f_bootstrap(D):
    """Appendix F: N = 30, LLM - TabPFN in RMSE / rho / RMSE50 / gain50 with CI, combined over draws."""
    Bt = D["boot"]

    def cell(ds, m, stat):
        r = Bt[(Bt.dataset == ds) & (Bt.model == m) & (Bt.baseline == "tabpfn") & (Bt["size"] == 30) & (Bt.draw == 0)
               & (Bt.stat == stat)]
        if not len(r):
            return "--"
        r = r.iloc[0]
        return f"{num(r.est, 2, plus=True)} {ci(r.lo, r.hi)}{boot_flag(D, ds, m, 30, stat, 0)}"
    # one block per statistic, the three clean datasets as columns (fits the text width at \small)
    stats = [("RMSE", r"$\Delta$RMSE"), ("rho", r"$\Delta\rho(\sigma,|e|)$"), ("RMSE50", r"$\Delta$RMSE$_{50}$"),
             ("gain50", r"$\Delta$gain$_{50}$")]
    body = []
    for k, (st, lab) in enumerate(stats):
        body.append(rf"\multicolumn{{4}}{{@{{}}l}}{{{lab}}}\\")
        for m in LLM_ORDER:
            body.append(row(LLM_NAMES[m], *[cell(ds, m, st) for ds in CLEAN]))
        if k < len(stats) - 1:
            body.append(r"\addlinespace")
    hdr = [row("model", *[CLEAN_NAMES[ds] for ds in CLEAN])]
    cap = (r"Paired bootstrap at 30 in-context molecules, language model minus TabPFN v3.5. Difference with the "
           r"percentile 95\,\% interval from 1,000 resamples of the 150 test molecules, computed in each of the "
           r"three context draws on the same resampled molecules and averaged (TabPFN seed 0). "
           r"$\dagger$ / $\ddagger$: the interval excludes 0 in favour of the language model / of TabPFN. The "
           r"intervals at $N$ = 60, 120 and the largest $N$, and against the GP, are in the repository "
           r"(\texttt{results/icl/bootstrap\_cells.csv}).")
    float_table("c_f_bootstrap", "tab:f_bootstrap", cap, r"@{}llll@{}", hdr, body, size=r"\small",
                tabcolsep="9pt", placement="tbp")


def c_g_ladder(D):
    """Appendix G: (1) median emitted tokens per model x effort x N with the truncation share;
    (2) change of RMSE, rho and PICP68 from low to high per model."""
    L = D["ladder"]
    L = L[L.model.isin(LADDER_ORDER)]
    body = []
    for m in LADDER_ORDER:
        for which in ("small", "large"):
            cells = [LLM_NAMES[m] if which == "small" else "", "30" if which == "small" else "largest"]
            tr = []
            for e in EFFORTS:
                g = L[(L.model == m) & (L.effort == e)]
                g = g[g["size"] == 30] if which == "small" else g[g["size"] == g.dataset.map(TOP)]
                if not len(g):
                    cells.append("--")
                    continue
                lo, hi = g.median_reasoning_tokens.min(), g.median_reasoning_tokens.max()
                cells.append(intc(lo) if lo == hi else f"{intc(lo)}--{intc(hi)}")
                tr.append(g.trunc_rate.max())
            cells.append(f"{100 * max(tr):.0f}\\,\\%" if max(tr) >= 0.005 else "0")
            body.append(row(*cells))
        body.append(r"\addlinespace[2pt]")
    body = body[:-1]
    hdr = [row("model", "$N$", *[r"\texttt{" + e + "}" for e in EFFORTS], "truncated")]
    cap = (r"Emitted reasoning tokens along the ladder. Median reasoning tokens per call at each requested effort, "
           r"range over the four datasets, at $N=30$ and at the largest context size; ``--'' = the endpoint "
           r"rejects the level; last column: the largest share of calls cut at the 32,000-token completion cap "
           r"over the levels and datasets of the row. $R=3$ calls per molecule, context draw 1 "
           r"(\texttt{results/icl/ladder\_cells.csv}).")
    float_table("c_g1_ladder_tokens", "tab:g_tokens", cap, r"@{}llrrrrrr@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="4pt", resize=True, placement="tbp")
    S = D["sel"]
    S = S[(S.family == "LLM") & (S.draw == 1)].rename(columns={"ds": "dataset", "reasoning": "effort"})
    Lr = L.merge(S[["dataset", "model", "size", "effort", "rho"]], on=["dataset", "model", "size", "effort"], how="left")
    Lc = Lr[Lr.dataset.isin(CLEAN)]
    P = Lc.pivot_table(index=["model", "dataset", "size"], columns="effort", values=["RMSE", "rho", "PICP68"])
    body = []
    for m in LADDER_ORDER:
        p = P.xs(m, level=0)
        d = (p[("RMSE", "high")] - p[("RMSE", "low")]).dropna()
        r = (p[("rho", "high")] - p[("rho", "low")]).dropna()
        c = (p[("PICP68", "high")] - p[("PICP68", "low")]).dropna()
        body.append(row(LLM_NAMES[m], f"{num(d.min(), 2, True)} / {num(d.median(), 2, True)} / {num(d.max(), 2, True)}",
                        f"{int((d < 0).sum())} of {len(d)}",
                        f"{num(r.min(), 2, True)} / {num(r.median(), 2, True)} / {num(r.max(), 2, True)}",
                        f"{int((r > 0).sum())} of {len(r)}",
                        f"{num(c.min(), 2, True)} / {num(c.median(), 2, True)} / {num(c.max(), 2, True)}"))
    hdr = [row("model", r"$\Delta$RMSE min / median / max", "lower", r"$\Delta\rho(\sigma,|e|)$ min / median / max",
               "higher", r"$\Delta$PICP$_{68}$ min / median / max")]
    cap = (r"Effect of the requested effort on accuracy and uncertainty. Change from the effort low to high over the "
           r"six clean cells of a model (three datasets at $N=30$ and at the largest $N$): root-mean-square error "
           r"with the number of cells in which it falls, rank correlation between stated standard deviation and "
           r"absolute error with the number of cells in which it rises, and coverage of $\pm1\sigma$. Gemini 3.8 "
           r"Flash at high rests on the molecules with a parsed answer (Table~\ref{tab:g_tokens}) "
           r"(\texttt{results/icl/ladder\_cells.csv}, \texttt{selective\_prediction.csv}).")
    float_table("c_g2_ladder_delta", "tab:g_delta", cap, r"@{}llllll@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="4pt", resize=True, placement="tbp")


def c_h_operations(D):
    """Appendix H: per run: cells, calls, cost, cells with declined calls, truncated calls, cache share."""
    C = D["cells_all"]
    Z = D["zs"]
    runs = [("main run (effort low, draw 1)", (C.reasoning == "low") & (C.draw == 1)),
            ("context draws 2 and 3", (C.reasoning == "low") & (C.draw > 1)),
            ("reasoning ladder (other efforts)", C.reasoning != "low")]
    body = []
    for lab, mask in runs:
        g = C[mask]
        cs = g[g["size"] >= 120].cache_share
        body.append(row(lab, str(len(g)), f"{int(g.n_calls.sum()):,}", num(g.cost_usd.sum()),
                        f"{int((g.n_calls < 450).sum())} (min {int(g.n_calls.min())})",
                        f"{int((g.trunc_rate > 0).sum())} (max {100 * g.trunc_rate.max():.0f}\\,\\%)",
                        f"{num(cs.min())}--{num(cs.max())}"))
    body.append(row("zero-shot screen", str(len(Z)), f"{int(Z.n_calls.sum()):,}", num(Z.cost_usd.sum()), "0",
                    f"{int((Z.n_truncated > 0).sum())}", "--"))
    tot_c = int(C.n_calls.sum() + Z.n_calls.sum())
    tot_u = C.cost_usd.sum() + Z.cost_usd.sum()
    body.append(r"\midrule")
    body.append(row("all analysed calls", str(len(C) + len(Z)), f"{tot_c:,}", num(tot_u), "", "", ""))
    hdr = [row("run", "cells", "calls", "USD", "cells $<450$ calls", "cells truncated", "cache share")]
    cap = (r"Calls, cost and operational findings per run. Stored and analysed calls and billed cost "
           r"(OpenRouter \texttt{usage.cost}); cells with fewer than the 450 calls of a complete cell, where an "
           r"endpoint declined single prompts that gap-fill runs did not recover (all of them OpenAI cells, every "
           r"test molecule still answered); cells with at least one call cut at the completion cap, with the largest "
           r"share; range of the median share of prompt tokens served from the prompt cache over the cells with "
           r"$N\ge120$. The duplicated records of the context draws (billed, not analysed) and the probes before "
           r"the panel was fixed are in the repository (\texttt{results/icl/cells.csv}, "
           r"\texttt{results/contamination/zeroshot/cells.csv}).")
    float_table("c_h_operations", "tab:h_operations", cap, r"@{}lrrrlll@{}", hdr, body, size=r"\footnotesize",
                tabcolsep="4pt", resize=True, placement="tbp")


def compact_tables(D, U):
    c_t2_summary(D, U)
    c_b_panel(D, U)
    c_c_baselines(D)
    c_d_contamination(D)
    # c_e_calibration (tables c_e1_calibration, c_e2_uc3) left the paper with the calibration analysis (review
    # round 2026-10-28); the function stays for reference; t3_calibration / a_calibration / a_uc3 are still
    # written for the repository
    c_f_bootstrap(D)
    c_g_ladder(D)
    c_h_operations(D)


# ---------------------------------------------------------------- checks
def cross_check(D, U):
    """The unified frame must reproduce draws_cells.csv for the LLMs (RMSE, rho, PICP68, RMSE50)."""
    X = D["draws_cells"]
    X = X[X.family == "LLM"]
    bad = 0
    for _, r in X.iterrows():
        for c in ("RMSE", "rho_UC1_err", "PICP68", "RMSE50"):
            v = get(U, r.dataset, r.model, r["size"], r.draw, c)
            if abs(v - r[c]) > 1e-9:
                bad += 1
    print(f"check: unified frame vs draws_cells.csv (LLM rows, 4 metrics): {bad} mismatches of {4 * len(X)}")
    d = (U.rho - U.rho_UC1_err).abs()
    print(f"check: tie-aware rho (tables) vs rho_UC1_err of cells.csv / summary.csv: max |diff| {d.max():.4f}")
    assert d.max() < 0.01
    # baseline seed rule: TabPFN lipophilicity N = 30 draw 1 must be the seed mean
    v = get(U, "lipophilicity", "tabpfn", 30, 1, "RMSE")
    B = D["base_seeds"]
    m = B[(B.dataset == "lipophilicity") & (B.method == "tabpfn") & (B["size"] == 30) & (B.draw == 1)].RMSE.mean()
    assert abs(v - m) < 1e-12


def main():
    D = load()
    U = unified(D)
    cross_check(D, U)
    t1_panel(D, U)
    t2_smallN(D, U, 30, "t2_smallN", "tab:smallN", main=True)
    t2_smallN(D, U, 60, "a22_smallN60", "tab:smallN60", main=False)
    t2_smallN(D, U, 120, "a23_smallN120", "tab:smallN120", main=False)
    t3_calibration(D)
    a_registry(D)
    a_probe(D)
    a_corpus(D)
    a_zeroshot(D)
    a_dedup(D)
    a_overlap(D)
    a_main_per_dataset(D, U)
    a_draws(D, U)
    a_selective(D)
    a_calibration(D)
    a_uc3(D)
    a_bootstrap(D)
    a_esol_subsets(D)
    a_ladder(D)
    a_basefits(D)
    a_cost(D)
    compact_tables(D, U)
    print(f"\n{len(WRITTEN)} tables written to {os.path.relpath(OUT, ROOT)}:")
    for p, lab in WRITTEN:
        print(f"  {p:45s} {lab}")


if __name__ == "__main__":
    sys.exit(main())
