"""
Paper figures of the v2 uncertainty study (project 44) from the result tables. Read-only on
results/ and data/; writes PDF + PNG (200 dpi), figs/README.md and figs/figs_meta.json.

    python src/make_figs.py                       # all figures -> figs/
    python src/make_figs.py --only fig1,fig3      # a subset (README entries of the others are kept)
    python src/make_figs.py --out figs_draft      # another output folder
    python src/make_figs.py --draw-mean --out figs_drawmean   # old draw rule, for comparison

Figures
  fig1  fig1_rmse_vs_n               RMSE vs N (4 datasets), 7 LLMs + the 4 display baselines.
  fig2  fig2_uncertainty_usefulness  (a) rho(sigma, |err|) vs N, (b) RMSE of the 50 % most certain vs N.
  fig3  fig3_joint_smallN            RMSE vs gain50 = 1 - RMSE50 / RMSE at N = 30, iso-RMSE50 lines.
  fig4  fig4_calibration             (a) raw PICP68 vs N, (b) variance-scaling factor s vs N.
  fig5  fig5_esol_recall             ESOL: RMSE / RMS of e/sigma (sigma floored at 0.01) on zero-shot-recalled vs
                                     never-recalled molecules (and all 150).
  fig6  fig6a_ladder_smallN / fig6b_ladder_largeN   reasoning ladder, x = emitted reasoning tokens.
  Appendix (A-numbered floats in the manuscript):
  figA1 figA1_risk_coverage          RMSE of the most certain fraction q = 0.1-1.0 at N = 30 (+ oracle, random).
  figA2 figA2_draws                  learning curves per context draw at N <= 120 (7 LLMs + TabPFN).
  figA3 figA3_rho_heatmap            rho(sigma, |err|) model x N, draw 1, bootstrap marks vs TabPFN.
  figA4 figA4_cost                   USD per cell (LLMs) / CPU wall time (baselines) vs RMSE, N = 30 and largest N.
  figA5 figA5_sigma_values           distinct stated sigma values per cell vs N + the sigmas of one cell.

Data rules (shared by all figures)
  * LLM cells: results/icl/cells.csv, reasoning == "low". Cells with coverage < MIN_COVERAGE
    (a run still in progress) are dropped and listed.
  * Context draws (default): every line / marker is the DRAW-1 value at every N (one nested
    context chain, no mixing of means and single draws). At N in DRAW_SIZES a bar spans the
    RANGE (min-max) over the context draws present (>= 2). --draw-mean restores the old rule
    (mean over draws at N in DRAW_SIZES, bar = SD, draw 1 above).
  * Baselines: results/baselines/summary.csv, first averaged over seeds within a draw, then the
    same draw rule as the LLMs. One display subset everywhere (DISPLAY_BASES); fig4 TabPFN only;
    fig3 adds the remaining baselines in light grey. All 8 baselines stay in the tables.
  * rho(sigma, |err|) everywhere (fig2a, fig6, figA3) = selective_prediction.csv `rho` (sigma rounded to 10
    decimals before ranking, the value of the tables), not cells.csv rho_UC1_err (unrounded; differs by <= 0.009).
  * Selective prediction: results/icl/selective_prediction.csv (tie-aware RMSE50 = exact sqrt of the expected MSE over tie orders, gain50);
    LLM rows are cross-checked against cells.csv (same n_mol and RMSE, coverage >= MIN_COVERAGE),
    rows that disagree (a stale table) are replaced by an in-memory recomputation from the
    per-molecule tables; the check result is printed (no-op when the table is current).
No numbers are hard-coded: re-run after the tables are regenerated.
"""
import argparse
import datetime as _dt
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from common import DATASET_KEYS, PANEL, ROOT  # noqa: E402
except Exception as exc:  # registry unreadable: fall back to the known panel
    print(f"WARNING: could not import common.py ({exc}); using built-in panel / dataset list")
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DATASET_KEYS = ["lipophilicity", "bace", "antiviral", "esol"]
    PANEL = ["haiku55", "sonnet55", "gpt61sol", "gem38flash", "gpt6luna", "gpt6astra", "glm53flash"]

ICL = os.path.join(ROOT, "results", "icl")
BASE = os.path.join(ROOT, "results", "baselines")
P_CELLS = os.path.join(ICL, "cells.csv")
P_BASE = os.path.join(BASE, "summary.csv")
P_LADDER = os.path.join(ICL, "ladder_cells.csv")
P_CALIB = os.path.join(ICL, "calibration_cells.csv")
P_SEL = os.path.join(ICL, "selective_prediction.csv")
P_DRAWS = os.path.join(ICL, "draws_cells.csv")
P_ESOL_PM = os.path.join(ICL, "esol_permolecule.csv")

REASONING = "low"
DRAW_SIZES = (30, 60, 120)
JOINT_N = 30                 # context size of fig3
MIN_COVERAGE = 0.95
N_TEST = 150
Z68 = 1.0           # PICP68 = |y - mu| <= 1 sigma, as in src/analyze_icl.py
SIGMA_FLOOR = 0.01
ESOL_CAP = 1.45     # ESOL panels of fig1 / fig2b: y cut at this multiple of the highest LLM value (incl. bars)
DRAW_MEAN = False   # set by --draw-mean

# ---------------------------------------------------------------------------------- style
DS_NAMES = {"lipophilicity": "Lipophilicity (logD)", "bace": "BACE (pIC50)",
            "antiviral": "Antiviral Mpro (pIC50)", "esol": "ESOL (logS, contaminated)"}
CONTAMINATED = {"esol"}
MODEL_NAMES = {"haiku55": "Haiku 5.5", "sonnet55": "Sonnet 5.5", "gpt61sol": "GPT-6.1 Sol",
               "gem38flash": "Gemini 3.8 Flash", "gpt6luna": "GPT-6 Luna", "gpt6astra": "GPT-6 Astra",
               "glm53flash": "GLM 5.3 Flash"}
# Okabe-Ito hues (yellow replaced by Tol's wine #882255); order checked with the dataviz
# palette validator: all adjacent pairs >= 9.6 Delta-E under deuteranopia. Markers are a
# second encoding so identity never rests on colour alone.
MODEL_COLORS = {"haiku55": "#0072B2", "sonnet55": "#D55E00", "gpt61sol": "#009E73",
                "gem38flash": "#882255", "gpt6luna": "#E69F00", "gpt6astra": "#56B4E9",
                "glm53flash": "#CC79A7"}
MODEL_MARKERS = {"haiku55": "o", "sonnet55": "s", "gpt61sol": "D", "gem38flash": "^",
                 "gpt6luna": "v", "gpt6astra": "P", "glm53flash": "X"}
BASE_NAMES = {"tabpfn": "TabPFN", "gp_tanimoto": "GP (Tanimoto)", "rf_desc": "RF (descriptors)",
              "chemprop_ens": "Chemprop ens.", "mlp_ens": "MLP ens.", "rf": "RF (Morgan)",
              "knn_tanimoto": "kNN (Tanimoto)", "trivial": "trivial"}
BASE_SHORT = {"tabpfn": "TabPFN", "gp_tanimoto": "GP", "rf_desc": "RF desc.", "chemprop_ens": "Chemprop"}
# the ONE baseline display subset used in fig1 / fig2 / fig3 (fig4: TabPFN only)
DISPLAY_BASES = ["tabpfn", "gp_tanimoto", "rf_desc", "chemprop_ens"]
OTHER_BASES = ["rf", "knn_tanimoto", "mlp_ens", "trivial"]      # fig3 only, light grey, no legend entry each
BASE_STYLE = {"tabpfn": dict(color="black", ls="--", marker="s", mfc="white"),
              "gp_tanimoto": dict(color="#555555", ls=(0, (4, 2)), marker="o", mfc="white"),
              "rf_desc": dict(color="#7a7a7a", ls=":", marker="^", mfc="white"),
              "chemprop_ens": dict(color="#a8a8a8", ls="-.", marker="v", mfc="white")}
OTHER_COLOR = "#c8c8c8"
LEVELS = ["none", "minimal", "low", "medium", "high"]
LEVEL_MARKERS = {"none": "x", "minimal": "v", "low": "o", "medium": "s", "high": "^"}

# FS: font scale (decision 2026-10-14, revised: figure text about caption size, ~9 pt ticks/legend, ~9.5 pt titles, when the
# figure is included at the full text width of 6.3 in); FW: figure width in inches = the text width.
FS = float(os.environ.get("FIG_FONT_SCALE", "1.1"))
FW = 6.3
plt.rcParams.update({
    "font.size": 8 * FS, "axes.titlesize": 8.5 * FS, "axes.labelsize": 8 * FS, "xtick.labelsize": 7.5 * FS,
    "ytick.labelsize": 7.5 * FS, "legend.fontsize": 7.5 * FS, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "lines.linewidth": 1.2, "lines.markersize": 3.5 * FS ** 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    # typeface of the manuscript (\usepackage{times}): Times for text, STIX (a Times design) for math
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "STIXGeneral",
                                           "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
# off-scale baselines of a cut axis are drawn as arrows only; the value labels next to them were removed
# (project lead, 2026-10-14: the coloured arrows suffice)
CLIP_LABELS = False

NOTES = []          # data gaps / warnings of this run (printed and written to the README)
FIG_META = {}       # numbers found while drawing, used in the README captions


def note(msg):
    NOTES.append(msg)
    print("NOTE:", msg)


def check(msg):
    """a passed consistency check: printed and listed in the README, not a warning"""
    NOTES.append("check: " + msg)
    print("CHECK:", msg)


def read_csv(path, what):
    if not os.path.exists(path):
        note(f"{what}: {os.path.relpath(path, ROOT)} missing")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        note(f"{what}: could not read {os.path.relpath(path, ROOT)} ({exc})")
        return None


def need(df, cols, what):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        note(f"{what}: columns {miss} missing")
        return False
    return True


def draw_rule(df):
    """Keep draw 1 at every N, draws >= 2 only at DRAW_SIZES."""
    if "draw" not in df.columns:
        df = df.assign(draw=1)
    df = df.copy()
    df["draw"] = df["draw"].fillna(1).astype(int)
    return df[(df.draw == 1) | df["size"].isin(DRAW_SIZES)]


# ---------------------------------------------------------------------------------- loaders
_cache = {}


def llm_cells(all_reasoning=False):
    """cells.csv, complete cells only (coverage >= MIN_COVERAGE), reasoning `low`."""
    key = ("cells", all_reasoning)
    if key in _cache:
        return _cache[key]
    C = read_csv(P_CELLS, "cells")
    if C is not None:
        if "draw" not in C.columns:
            C["draw"] = 1
        if not all_reasoning:
            C = C[C.reasoning == REASONING]
        if "coverage" in C.columns:
            bad = C[C.coverage < MIN_COVERAGE]
            if not all_reasoning:
                if len(bad):
                    note("cells.csv: left out (coverage < %.2f, run in progress?): " % MIN_COVERAGE + ", ".join(
                        f"{r.dataset}/{r.model}/N{int(r['size'])}/d{int(r.draw)} ({int(r.n_mol)} mol)"
                        for _, r in bad.iterrows()))
                else:
                    check(f"cells.csv: all {len(C)} `{REASONING}` cells have coverage >= {MIN_COVERAGE:.2f}")
            C = C[C.coverage >= MIN_COVERAGE]
        C = draw_rule(C)
        if not all_reasoning:
            # the bars at N in DRAW_SIZES rest on the draws PRESENT, which may differ between models
            nd = C[C["size"].isin(DRAW_SIZES)].groupby(["dataset", "model", "size"]).draw.nunique()
            if nd.nunique() > 1:
                parts = []
                for ds in DATASET_KEYS:
                    if ds not in nd.index.get_level_values(0):
                        continue
                    t = nd.loc[ds].unstack("size")
                    parts.append(f"{ds} (N {'/'.join(str(int(c)) for c in t.columns)}): " + ", ".join(
                        f"{m} {'/'.join(str(int(v)) if v == v else '-' for v in t.loc[m])}"
                        for m in PANEL if m in t.index))
                note("context draws per LLM point at N <= %d differ between models (bars rest on these "
                     "counts): " % max(DRAW_SIZES) + "; ".join(parts))
            elif len(nd):
                check(f"cells.csv: every LLM point at N = {'/'.join(map(str, DRAW_SIZES))} has "
                      f"{int(nd.iloc[0])} context draws")
        unknown = sorted(set(C.model) - set(MODEL_NAMES))
        if unknown:
            note(f"cells.csv: models without display name/colour, not plotted: {unknown}")
    _cache[key] = C
    return C


def base_cells():
    """summary.csv averaged over seeds within (dataset, method, size, draw)."""
    if "base" in _cache:
        return _cache["base"]
    B = read_csv(P_BASE, "baselines")
    if B is not None:
        if "draw" not in B.columns:
            B["draw"] = 1
        num = [c for c in B.columns if c not in ("dataset", "method", "size", "draw", "seed", "env")
               and pd.api.types.is_numeric_dtype(B[c])]
        B = B.groupby(["dataset", "method", "size", "draw"], as_index=False)[num].mean()
        B = draw_rule(B).rename(columns={"method": "model"})
    _cache["base"] = B
    return B


def selective():
    """RMSE / RMSE50 / gain50 per (dataset, model, size, draw): LLMs (checked against cells.csv)
    + baselines (mean over seeds within a draw); stale or missing LLM cells recomputed in memory."""
    if "sel" in _cache:
        return _cache["sel"]
    S = read_csv(P_SEL, "selective prediction")
    out = None
    cols = ["dataset", "model", "size", "draw", "RMSE", "RMSE50", "gain50", "rho"]
    if S is not None and need(S, ["ds", "model", "size", "RMSE", "RMSE50", "rho", "family"], "selective_prediction.csv"):
        S = S.rename(columns={"ds": "dataset"})
        if "draw" not in S.columns:
            S["draw"] = 1
        S["draw"] = S["draw"].fillna(1).astype(int)
        if "gain50" not in S.columns:
            S["gain50"] = 1 - S.RMSE50 / S.RMSE
        C = llm_cells()
        L = S[(S.family == "LLM") & (S.get("reasoning", REASONING) == REASONING)]
        n_llm_rows = len(L)
        stale = orphan = L.iloc[0:0]
        if C is not None:
            ref = C[["dataset", "model", "size", "draw", "n_mol", "RMSE"]]
            L = L.merge(ref, on=["dataset", "model", "size", "draw"], how="left", suffixes=("", "_cells"))
            stale = L[L.n_mol_cells.notna() & ((L.n_mol != L.n_mol_cells) | ((L.RMSE - L.RMSE_cells).abs() > 1e-6))]
            orphan = L[L.n_mol_cells.isna()]
            if len(stale):
                note("selective_prediction.csv: stale rows (n_mol/RMSE differ from cells.csv), replaced by the in-memory recomputation below: " + ", ".join(
                    f"{r.dataset}/{r.model}/N{int(r['size'])}/d{int(r.draw)} ({int(r.n_mol)} vs {int(r.n_mol_cells)} mol)"
                    for _, r in stale.iterrows()))
            if len(orphan):
                note(f"selective_prediction.csv: {len(orphan)} LLM rows without a complete cell in cells.csv, left out")
            L = L[L.n_mol_cells.notna() & ~L.index.isin(stale.index)]
        L = L[cols]
        Bs = S[S.family == "baseline"].groupby(["dataset", "model", "size", "draw"], as_index=False)[
            ["RMSE", "RMSE50", "gain50", "rho"]].mean()
        out = pd.concat([L, Bs], ignore_index=True)
        # LLM cells complete in cells.csv but absent here (stale or older table): compute RMSE50 in memory
        # from results/icl/<ds>_permolecule.csv with the same function (selective_prediction.rmse_most_certain);
        # last resort draws_cells.csv. Nothing is written.
        missing = []
        if C is not None:
            have = set(map(tuple, out[["dataset", "model", "size", "draw"]].values))
            missing = [k for k in map(tuple, C[["dataset", "model", "size", "draw", "n_mol"]].values)
                       if k[:4] not in have]
            filled, src_pm, src_dc = [], 0, 0
            if missing:
                try:
                    from selective_prediction import rmse_most_certain
                except Exception as exc:
                    rmse_most_certain = None
                    note(f"could not import selective_prediction.rmse_most_certain ({exc})")
                PM, D = {}, False
                for k in missing:
                    ds, m, n, d, nmol = k
                    if rmse_most_certain is not None and ds not in PM:
                        p = os.path.join(ICL, f"{ds}_permolecule.csv")
                        PM[ds] = pd.read_csv(p, dtype={"mol_id": str}) if os.path.exists(p) else None
                        if PM[ds] is not None and "draw" not in PM[ds].columns:
                            PM[ds]["draw"] = 1
                    P = PM.get(ds)
                    if P is not None:
                        g = P[(P.model == m) & (P["size"] == n) & (P.draw == d) & (P.reasoning == REASONING)]
                        g = g[np.isfinite(g.pred) & np.isfinite(g.uc1)]
                        if len(g) == int(nmol):
                            e = (g.pred - g.truth).values
                            rmse = float(np.sqrt(np.mean(e ** 2)))
                            r50 = rmse_most_certain(e, g.uc1.values)
                            sg = np.round(g.uc1.values, 10)      # tie rule of selective_prediction.py
                            rho = pd.Series(sg).corr(pd.Series(np.abs(e)), method="spearman") if np.ptp(sg) > 0 else np.nan
                            filled.append(dict(dataset=ds, model=m, size=n, draw=d, RMSE=rmse, RMSE50=r50,
                                               gain50=1 - r50 / rmse, rho=rho))
                            src_pm += 1
                            continue
                    if D is False:
                        D = read_csv(P_DRAWS, "draws_cells")
                    if D is not None and D is not False and "RMSE50" in D.columns:
                        r = D[(D.dataset == ds) & (D.model == m) & (D["size"] == n) & (D.draw == d) & (D.n_mol == nmol)]
                        if len(r):
                            row = r[["dataset", "model", "size", "draw", "RMSE", "RMSE50"]].iloc[0].to_dict()
                            row["gain50"] = 1 - row["RMSE50"] / row["RMSE"]
                            filled.append(row)
                            src_dc += 1
            if filled:
                out = pd.concat([out, pd.DataFrame(filled)], ignore_index=True)
            if missing:
                still = len(missing) - src_pm - src_dc
                note(f"selective_prediction.csv lacks {len(missing)} complete LLM cells (stale or older than cells.csv); "
                     f"RMSE50 computed in memory from the per-molecule tables for {src_pm}, taken from draws_cells.csv "
                     f"for {src_dc}, unavailable for {still}. Re-run src/selective_prediction.py to refresh the table.")
        if not len(stale) and not len(orphan) and not missing:
            check(f"selective_prediction.csv vs cells.csv: {n_llm_rows} LLM rows (`{REASONING}`) all match "
                  f"(n_mol, RMSE), 0 stale, 0 orphan, 0 missing -> the in-memory recomputation is a no-op")
        if C is not None:
            # rho in the figures = the tie-rounded rho of the tables (selective_prediction.csv `rho`, sigma rounded to
            # 10 decimals); cells.csv rho_UC1_err ranks the unrounded sigma, so last-bit float differences split ties
            dd = out.merge(C[["dataset", "model", "size", "draw", "rho_UC1_err"]], on=["dataset", "model", "size", "draw"])
            dr = (dd.rho - dd.rho_UC1_err).abs()
            if len(dr):
                r = dd.loc[dr.idxmax()]
                check(f"rho in all figures from selective_prediction.csv `rho` (sigma rounded to 10 decimals, as in the "
                      f"tables); vs cells.csv rho_UC1_err: {int((dr > 1e-9).sum())} of {len(dd)} LLM cells differ, max "
                      f"|diff| {dr.max():.4f} ({r.dataset}/{r.model}/N{int(r['size'])}/d{int(r.draw)})")
        out = draw_rule(out)
    _cache["sel"] = out
    return out


def aggregate(df, col):
    """per (dataset, model, size): `val` = plotted value, `lo`/`hi` = bar ends (NaN: no bar).
    Default: val = draw 1, bar = min-max over the draws present (>= 2 draws).
    --draw-mean: val = mean over the draws present, bar = mean +/- SD (ddof = 1)."""
    keys = ["dataset", "model", "size"]
    d = df.dropna(subset=[col])
    g = d.groupby(keys)[col]
    A = g.agg(["count", "min", "max", "mean"]).reset_index()
    A["sd"] = g.std(ddof=1).values
    d1 = d[d.draw == 1].groupby(keys)[col].first().rename("d1").reset_index()
    A = A.merge(d1, on=keys, how="left")
    if DRAW_MEAN:
        A["val"], A["lo"], A["hi"] = A["mean"], A["mean"] - A["sd"], A["mean"] + A["sd"]
    else:
        nod1 = A[A.d1.isna()]
        if len(nod1):
            note(f"{col}: no draw-1 value for " + ", ".join(
                f"{r.dataset}/{r.model}/N{int(r['size'])}" for _, r in nod1.iterrows()) + " (point left out)")
        A = A[A.d1.notna()].copy()
        A["val"], A["lo"], A["hi"] = A["d1"], A["min"], A["max"]
    A.loc[A["count"] < 2, ["lo", "hi"]] = np.nan
    return A


def draw_txt(short=False):
    if DRAW_MEAN:
        return ("at N <= %d mean over the context draws present, bar = SD over draws (>= 2 draws); larger N "
                "draw 1" % max(DRAW_SIZES)) if short else (
            "Points at N = %s are means over the context draws present, bars their SD (>= 2 draws); larger N "
            "use draw 1 only (old rule, --draw-mean)." % "/".join(map(str, DRAW_SIZES)))
    return ("lines through context draw 1 at every N, bar at N = %s = range (min-max) over the three context "
            "draws" % "/".join(map(str, DRAW_SIZES))) if short else (
        "Every line and marker is context draw 1 at every N (one nested context chain); at N = %s the vertical "
        "bar spans the range (min-max) over the three context draws (LLMs and baselines alike; baselines are "
        "averaged over seeds within a draw first)." % "/".join(map(str, DRAW_SIZES)))


# ---------------------------------------------------------------------------------- helpers
def series_list(llms_present, bases):
    s = [(m, "llm") for m in PANEL if m in llms_present and m in MODEL_NAMES]
    return s + [(b, "base") for b in bases]


def style_of(m, kind):
    if kind == "llm":
        return dict(color=MODEL_COLORS[m], ls="-", marker=MODEL_MARKERS[m], mfc=MODEL_COLORS[m], lw=1.2, z=3)
    if kind == "other":
        return dict(color=OTHER_COLOR, ls="-", marker="o", mfc="white", lw=0.8, z=1)
    st = dict(BASE_STYLE.get(m, dict(color="grey", ls="--", marker="o", mfc="white")))
    st.update(lw=1.0, z=2)
    return st


def plot_curve(ax, A, ds, m, kind, k, nser, err=True, label=None, lw=None, ms=None):
    """line through `val`; bars lo-hi where present. Returns (x, y) of the plotted points."""
    a = A[(A.dataset == ds) & (A.model == m)].sort_values("size")
    if a.empty:
        return None
    st = style_of(m, kind)
    dodge = 1.0 + 0.018 * (k - (nser - 1) / 2)          # small multiplicative x-offset on the log axis
    x = a["size"].values * dodge
    y = a["val"].values
    ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"], mec=st["color"],
            lw=lw or st["lw"], ms=ms or 3.5, mew=0.7, zorder=st["z"], label=label)
    if err:
        lo, hi = a["lo"].values, a["hi"].values
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.any():
            ax.errorbar(x[ok], y[ok], yerr=[np.clip(y[ok] - lo[ok], 0, None), np.clip(hi[ok] - y[ok], 0, None)],
                        fmt="none", ecolor=st["color"], elinewidth=0.6, capsize=1.2, zorder=st["z"])
    return x, y


def cap_axis(ax, curves, A, ds):
    """ESOL panels: cut the y-axis at ESOL_CAP x the highest LLM value incl. bars (rounded to 0.1); baseline
    points above are drawn as an upward arrow on the top edge (no value label unless CLIP_LABELS)."""
    a = A[(A.dataset == ds) & A.model.isin(MODEL_NAMES)]
    llm_top = np.nanmax(np.r_[a["val"].values, a["hi"].values]) if len(a) else 0
    top = max(np.round(ESOL_CAP * llm_top, 1), 1.02 * llm_top)
    bottom = min(ax.get_ylim()[0], 0.9 * np.nanmin(a["val"].values) if len(a) else 0)
    ax.set_ylim(bottom, top)
    for m, kind, xy in curves:
        if xy is None or kind == "llm":
            continue
        x, y = xy
        c = y > top
        if not c.any():
            continue
        st = style_of(m, kind)
        ytri = top - 0.025 * (top - bottom)              # just inside the top edge (title stays free)
        # an arrow glyph, not "^": the triangle is the data marker of RF (descriptors) and would read as data
        ax.plot(x[c], np.full(c.sum(), ytri), ls="", marker=r"$\uparrow$", ms=7 * FS ** 0.5, color=st["color"], mec=st["color"],
                mfc=st["color"], mew=0.3, zorder=6, clip_on=False)   # unclipped: the arrow head may pass the top edge
        i0 = np.argmax(c)
        col = "#444444" if st["color"] in (OTHER_COLOR, "#a8a8a8") else st["color"]
        # placed after tight_layout (place_clip_labels, called by save) where it overlaps the fewest points
        if CLIP_LABELS:
            ax._clip_labels = getattr(ax, "_clip_labels", []) + [(f"{BASE_SHORT.get(m, m)} {y[i0]:.2f}", (x[i0], ytri), col)]
    return float(top)


def place_clip_labels(fig):
    """annotate the first off-scale point of each clipped baseline, at the candidate offset whose text box
    covers the fewest plotted vertices / bar segments and no earlier label (leader line when moved away)."""
    pend_axes = [ax for ax in fig.axes if getattr(ax, "_clip_labels", None)]
    if not pend_axes:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in pend_axes:
        pts = [ax.transData.transform(ln.get_xydata()) for ln in ax.lines if len(ln.get_xydata())]
        t = np.linspace(0, 1, 10)[:, None]
        for coll in ax.collections:
            for seg in getattr(coll, "get_segments", lambda: [])():
                seg = np.asarray(seg)
                for i in range(len(seg) - 1):
                    pts.append(ax.transData.transform(seg[i] + (seg[i + 1] - seg[i]) * t))
        P = np.vstack(pts) if pts else np.zeros((0, 2))
        P = P[np.isfinite(P).all(1)]
        abox = ax.get_window_extent(r)
        placed, last_dy = [], 0
        for txt, xy, col in ax._clip_labels:      # stacked downwards in series order, so leaders never cross
            best = None
            for dy in [v for v in (-8, -16, -24, -32, -40) if v < last_dy] or [last_dy - 8]:
                for dx in (4, 10, 18, 28):
                    a = ax.annotate(txt, xy=xy, xytext=(dx, dy), textcoords="offset points", fontsize=6 * FS,
                                    ha="left", va="center")
                    bb = a.get_window_extent(r).expanded(1.08, 1.25)
                    a.remove()
                    n = int(((P[:, 0] > bb.x0) & (P[:, 0] < bb.x1) & (P[:, 1] > bb.y0) & (P[:, 1] < bb.y1)).sum())
                    n += 1000 * sum(bb.overlaps(q) for q in placed)
                    n += 10000 * (not (abox.x0 <= bb.x0 and bb.x1 <= abox.x1 and abox.y0 <= bb.y0))
                    score = n + 0.01 * (dx + abs(dy))
                    if best is None or score < best[0]:
                        best = (score, dx, dy, bb)
            _, dx, dy, bb = best
            last_dy = dy
            far = dx > 10 or dy < -16
            ax.annotate(txt, xy=xy, xytext=(dx, dy), textcoords="offset points", fontsize=6 * FS, color=col, ha="left",
                        va="center", zorder=7, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=col, shrinkA=0, shrinkB=2.5) if far else None)
            placed.append(bb)
        ax._clip_labels = []


def thin_sizes(sizes):
    """tick labels of a narrow panel: every other N from the smallest, and the largest N in place of its
    neighbour (all ticks stay; only the labels are thinned so that none overlap)."""
    sizes = list(sizes)
    keep = set(sizes[0::2])
    if (len(sizes) - 1) % 2:          # largest N not in the every-other set: show it instead of its neighbour
        keep.discard(sizes[-2])
        keep.add(sizes[-1])
    return keep


def size_axis(ax, sizes, narrow=False):
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    show = thin_sizes(sizes) if narrow else set(sizes)
    ax.set_xticklabels([str(int(s)) if s in show else "" for s in sizes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(min(sizes) / 1.25, max(sizes) * 1.25)


def wrap_name(ds):
    """dataset name on two lines for narrow panels: before the bracket, or inside it when the bracket is long
    ("ESOL (logS," / "contaminated)"), so that no title is wider than its panel."""
    t = DS_NAMES.get(ds, ds)
    head, _, br = t.partition(" (")
    if len(br) > 14 and ", " in br:
        a, _, b = br.partition(", ")
        return f"{head} ({a},\n{b}"
    return t.replace(" (", "\n(")


def ds_title(ax, ds, wrap=False):
    t = DS_NAMES.get(ds, ds)
    ax.set_title(wrap_name(ds) if wrap else t, color="#777777" if ds in CONTAMINATED else "black")


def row_major(handles, ncol):
    """matplotlib fills legends column by column; reorder (with blank padding) so rows read left to right."""
    nrow = int(np.ceil(len(handles) / ncol))
    pad = handles + [Line2D([], [], ls="", label=" ")] * (nrow * ncol - len(handles))
    return [pad[r * ncol + c] for c in range(ncol) for r in range(nrow)]


def series_legend(fig, series, y=-0.005, extra=None, line=True):
    """LLMs in the first row, baselines (+ extra handles) in their own row below."""
    h = legend_handles(series, line=line)
    nl = sum(1 for _, k in series if k == "llm")
    h += extra or []
    ncol = max(nl, len(h) - nl, 1)
    if ncol > 7 or FS > 0.95:      # text-width figures (FS ~1.1): 4 legend columns so the legend fits the 6.3 in width
        ncol, rows = 4, h
    else:
        rows = (h[:nl] + [Line2D([], [], ls="", label=" ")] * (ncol - nl)) if nl else []
        rows += h[nl:]
    fig.legend(handles=row_major(rows, ncol), loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), columnspacing=1.0, handlelength=2.4, handletextpad=0.5)


def legend_handles(series, line=True):
    h = []
    for m, kind in series:
        st = style_of(m, kind)
        h.append(Line2D([], [], color=st["color"], ls=st["ls"] if line else "", marker=st["marker"], mfc=st["mfc"],
                        mec=st["color"], lw=st["lw"], ms=4 * FS ** 0.5 if line else 5,
                        label=MODEL_NAMES.get(m, m) if kind == "llm" else BASE_NAMES.get(m, m)))
    return h


def save(fig, out, name):
    place_clip_labels(fig)
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("->", os.path.join(out, name + ".{pdf,png}"))


def freeze_layout(fig):
    """run the figure's constrained layout once and fix the axes positions, so that the x-label / legend
    block placed below by bottom_block() and the clip labels of save() do not move the panels."""
    fig.canvas.draw()
    fig.set_layout_engine("none")


def bottom_block(fig, axes, xlabel=None, handles=None, ncol=4, gap=3.0, **kw):
    """after freeze_layout(): a common x-label centred under the plot areas of `axes` and a legend below it,
    each `gap` pt below the lowest text above, so they never overlap and leave no dead band. Everything may
    extend below the canvas; savefig(bbox="tight") takes the true extent. The legend is shifted inward if it
    would pass a figure edge (the PDF stays as wide as the figure)."""
    r = fig.canvas.get_renderer()
    axes = [a for a in np.ravel(axes) if a.get_visible()]
    W, H, pt = fig.bbox.width, fig.bbox.height, fig.dpi / 72
    y = min(a.get_tightbbox(r).y0 for a in axes)
    xc = (min(a.get_position().x0 for a in axes) + max(a.get_position().x1 for a in axes)) / 2
    if xlabel:
        t = fig.text(xc, (y - gap * pt) / H, xlabel, ha="center", va="top", fontsize=plt.rcParams["axes.labelsize"])
        y = t.get_window_extent(r).y0
    if not handles:
        return None
    kw = {**dict(frameon=False, columnspacing=1.0, handlelength=2.0, handletextpad=0.5, borderaxespad=0.0,
                 borderpad=0.0), **kw}
    # several legends stacked (handles = list of handle lists, ncol = list) or one
    groups = handles if isinstance(handles[0], (list, tuple)) else [handles]
    ncols = ncol if isinstance(ncol, (list, tuple)) else [ncol] * len(groups)
    legs = []
    for hs, nc in zip(groups, ncols):
        yl = (y - gap * pt) / H
        leg = fig.legend(handles=hs, loc="upper center", ncol=nc, bbox_to_anchor=(xc, yl), **kw)
        bb = leg.get_window_extent(r)
        shift = -bb.x0 if bb.x0 < 0 else (W - bb.x1 if bb.x1 > W else 0)
        if shift:
            leg.set_bbox_to_anchor((xc + shift / W, yl), transform=fig.transFigure)
        y = leg.get_window_extent(r).y0 + (gap - 1.0) * pt     # stacked legends: 1 pt apart
        legs.append(leg)
    return legs[0] if len(legs) == 1 else legs


def panel_label(ax, txt):
    ax.text(-0.02, 1.02, txt, transform=ax.transAxes, fontweight="bold", fontsize=9 * FS, ha="right", va="bottom")


# ---------------------------------------------------------------------------------- fig 1
def fig1(out):
    C, B = llm_cells(), base_cells()
    if C is None:
        note("fig1 skipped (no cells.csv)")
        return
    frames = [C[["dataset", "model", "size", "draw", "RMSE"]]]
    if B is not None:
        frames.append(B[B.model.isin(DISPLAY_BASES)][["dataset", "model", "size", "draw", "RMSE"]])
    A = aggregate(pd.concat(frames, ignore_index=True), "RMSE")
    series = series_list(set(C.model), [b for b in DISPLAY_BASES if b in set(A.model)])
    fig, axs = plt.subplots(2, 2, figsize=(FW, 6.36))
    for ax, ds in zip(axs.flat, DATASET_KEYS):
        a = A[A.dataset == ds]
        if a.empty:
            ax.set_visible(False)
            continue
        curves = [(m, kind, plot_curve(ax, A, ds, m, kind, k, len(series))) for k, (m, kind) in enumerate(series)]
        size_axis(ax, sorted(a["size"].unique()))
        if ds in CONTAMINATED:
            FIG_META.setdefault("esol_cut", {})["fig1"] = cap_axis(ax, curves, A, ds)
        ds_title(ax, ds)
        ax.set_ylabel("RMSE")
        ax.set_xlabel("context size N")
    series_legend(fig, series, -0.005)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, out, "fig1_rmse_vs_n")


# ---------------------------------------------------------------------------------- fig 2
def fig2(out):
    C, S = llm_cells(), selective()
    if C is None or S is None:
        note("fig2 skipped (no cells.csv or selective_prediction.csv)")
        return
    # row a: the tie-rounded rho of selective_prediction.csv (= the tables), not cells.csv rho_UC1_err
    S = S[S.model.isin(set(MODEL_NAMES) | set(DISPLAY_BASES))]
    Ar = aggregate(S, "rho")
    As = aggregate(S, "RMSE50")
    FIG_META["fig2_rho"] = Ar
    series = series_list(set(C.model), [b for b in DISPLAY_BASES if b in set(Ar.model)])
    fig, axs = plt.subplots(2, 4, figsize=(FW, 5.52))
    for j, ds in enumerate(DATASET_KEYS):
        ax = axs[0, j]
        a = Ar[Ar.dataset == ds]
        if not a.empty:
            for k, (m, kind) in enumerate(series):
                plot_curve(ax, Ar, ds, m, kind, k, len(series))
            size_axis(ax, sorted(a["size"].unique()), narrow=True)
            ax.axhline(0, color="black", lw=0.6, zorder=1)
        ds_title(ax, ds, wrap=True)
        ax = axs[1, j]
        if As is not None and not As[As.dataset == ds].empty:
            curves = [(m, kind, plot_curve(ax, As, ds, m, kind, k, len(series))) for k, (m, kind) in enumerate(series)]
            size_axis(ax, sorted(As[As.dataset == ds]["size"].unique()), narrow=True)
            if ds in CONTAMINATED:
                FIG_META.setdefault("esol_cut", {})["fig2"] = cap_axis(ax, curves, As, ds)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("context size N")
    axs[0, 0].set_ylabel(r"$\rho_s(\sigma, |\mathrm{err}|)$")
    axs[1, 0].set_ylabel("RMSE of 50 % most certain")
    panel_label(axs[0, 0], "a")
    panel_label(axs[1, 0], "b")
    series_legend(fig, series, -0.01)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.6, h_pad=0.8)
    save(fig, out, "fig2_uncertainty_usefulness")


# ---------------------------------------------------------------------------------- fig 3 (joint)
def _iso_levels(lo, hi, max_lines=6):
    for step in (0.1, 0.2, 0.25, 0.5, 1.0):
        lv = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
        if len(lv) <= max_lines:
            return [round(v, 2) for v in lv if v > 0]
    return []


def fig3(out):
    S = selective()
    if S is None:
        note("fig3 skipped (no selective_prediction.csv)")
        return
    S = S[(S["size"] == JOINT_N) & S.model.isin(set(MODEL_NAMES) | set(DISPLAY_BASES) | set(OTHER_BASES))]
    Ax, Ay = aggregate(S, "RMSE"), aggregate(S, "gain50")
    keys = ["dataset", "model", "size"]
    J = Ax[keys + ["val", "lo", "hi"]].merge(Ay[keys + ["val", "lo", "hi"]], on=keys, suffixes=("_x", "_y"))
    llms = [m for m in PANEL if m in set(J.model) and m in MODEL_NAMES]
    disp = [b for b in DISPLAY_BASES if b in set(J.model)]
    others = [b for b in OTHER_BASES if b in set(J.model)]
    dss = [d for d in DATASET_KEYS if d in set(J.dataset)]
    # 2 x 2 at the text width, flat panels, y-label on the left column, one x-label and the legend below
    fig, axs = plt.subplots(2, 2, figsize=(FW, 4.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
    summary = []
    for ax, ds in zip(axs.flat, DATASET_KEYS):
        if ds not in dss:
            ax.set_visible(False)
            continue
        d = J[J.dataset == ds]
        order = [(m, "other") for m in others] + [(m, "base") for m in disp] + [(m, "llm") for m in llms]
        for m, kind in order:
            r = d[d.model == m]
            if r.empty:
                continue
            r = r.iloc[0]
            st = style_of(m, kind)
            x, y = r.val_x, r.val_y
            xe = [[max(x - r.lo_x, 0)], [max(r.hi_x - x, 0)]] if np.isfinite(r.lo_x) else None
            ye = [[max(y - r.lo_y, 0)], [max(r.hi_y - y, 0)]] if np.isfinite(r.lo_y) else None
            ms = 4.0 if kind == "other" else 5.5
            ax.errorbar([x], [y], xerr=xe, yerr=ye, fmt=st["marker"], color=st["color"], mfc=st["mfc"],
                        mec=st["color"], ms=ms, mew=0.9 if kind != "llm" else 0.5, elinewidth=0.6 if kind != "other" else 0.5,
                        capsize=0, alpha=1.0, zorder={"other": 2, "base": 3, "llm": 4}[kind])
        # axis limits from the points incl. bars, then the iso-RMSE50 lines over the visible region
        xs = np.r_[d.val_x, d.lo_x.dropna(), d.hi_x.dropna()]
        ys = np.r_[d.val_y, d.lo_y.dropna(), d.hi_y.dropna()]
        xpad, ypad = 0.05 * (xs.max() - xs.min()), 0.08 * (ys.max() - ys.min())
        xl, yl = (xs.min() - xpad, xs.max() + xpad), (min(ys.min() - ypad, -0.02), ys.max() + ypad)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.axhline(0, color="black", lw=0.5, zorder=1)
        lv = _iso_levels(xl[0] * (1 - yl[1]), xl[1] * (1 - yl[0]))
        xx = np.linspace(xl[0], xl[1], 200)
        nlab = 0
        for i, c in enumerate(lv):
            yy = 1 - c / xx
            ax.plot(xx, yy, color="#9a9a9a", lw=0.6, ls=(0, (4, 3)), zorder=0.5)
            # label where the line leaves the panel at the top, else at the right edge
            yt = yl[1] - 0.04 * (yl[1] - yl[0])
            xt = c / (1 - yt)
            if xl[0] + 0.05 * (xl[1] - xl[0]) < xt < xl[1] - 0.03 * (xl[1] - xl[0]):
                pos, ha, va = (xt, yt), "center", "center"
            else:
                xr = xl[1] - 0.01 * (xl[1] - xl[0])
                yr = 1 - c / xr
                if not (yl[0] < yr < yl[1]):
                    continue
                pos, ha, va = (xr, yr), "right", "center"
            ax.text(*pos, f"{c:g}", fontsize=6 * FS,   # level only; the legend names the lines (const. RMSE50)
                    color="#6f6f6f", ha=ha, va=va, zorder=5,   # above the bars: a bar passing a level label stays faintly visible
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
            nlab += 1
        ds_title(ax, ds)
        if ax in axs[:, 0]:
            ax.set_ylabel(r"gain50 = 1 $-$ RMSE50 / RMSE")
        # reading for the report: best LLM / baseline in x (low RMSE) and in y (high gain50)
        for fam, ms_ in (("LLM", llms), ("baseline", disp + others)):
            f = d[d.model.isin(ms_)]
            if len(f):
                bx, by = f.loc[f.val_x.idxmin()], f.loc[f.val_y.idxmax()]
                summary.append(f"{ds:13s} {fam:8s} best RMSE {bx.model} {bx.val_x:.2f} [{bx.lo_x:.2f}-{bx.hi_x:.2f}]; "
                               f"best gain50 {by.model} {by.val_y:.2f} [{by.lo_y:.2f}-{by.hi_y:.2f}]")
    print(f"fig3 values (N = {JOINT_N}; {'draw mean [mean +/- SD]' if DRAW_MEAN else 'draw 1 [min-max over draws]'}):")
    for s in summary:
        print("   ", s)
    pd.set_option("display.width", 200)
    tab = J.assign(RMSE50=(1 - J.val_y) * J.val_x).pivot_table(index="model", columns="dataset",
                                                                  values=["val_x", "val_y", "RMSE50"]).round(2)
    print(tab.to_string())
    series = series_list(set(llms), disp)
    extra = []
    if others:
        extra.append(Line2D([], [], ls="", marker="o", mfc="white", mec=OTHER_COLOR, color=OTHER_COLOR, ms=4 * FS ** 0.5,
                            label="other baselines"))
    extra.append(Line2D([], [], color="#9a9a9a", lw=0.6, ls=(0, (4, 3)), label="const. RMSE50"))
    freeze_layout(fig)
    bottom_block(fig, axs, f"RMSE (all test molecules), N = {JOINT_N}",
                 row_major(legend_handles(series, line=False) + extra, 5), ncol=5, handlelength=1.6,
                 columnspacing=0.9, handletextpad=0.3)
    FIG_META["fig3_others"] = [BASE_NAMES.get(b, b) for b in others]
    save(fig, out, "fig3_joint_smallN")


# ---------------------------------------------------------------------------------- fig 4 (calibration)
FIG4_BASES = ["tabpfn"]


def fig4(out):
    C, B = llm_cells(), base_cells()
    K = read_csv(P_CALIB, "calibration_cells")
    if C is None:
        note("fig4 skipped (no cells.csv)")
        return
    frames = [C[["dataset", "model", "size", "draw", "PICP68"]]]
    if B is not None:
        frames.append(B[B.model.isin(FIG4_BASES)][["dataset", "model", "size", "draw", "PICP68"]])
    Ap = aggregate(pd.concat(frames, ignore_index=True), "PICP68")
    As = None
    if K is not None and need(K, ["dataset", "model", "size", "s"], "calibration_cells.csv"):
        k = K.copy()
        if "subset" in k.columns:
            k = k[k.subset == "all"]
        if "signal" in k.columns:
            k = k[((k.family == "LLM") & (k.signal == "UC1")) | ((k.family != "LLM") & k.model.isin(FIG4_BASES))]
        if "draw" in k.columns:
            k = k[k.draw == 1]
        k = k.assign(draw=1)
        As = aggregate(k[k.model.isin(set(MODEL_NAMES) | set(FIG4_BASES))], "s")
        miss = sorted(set(C.model) - set(As.model))
        if miss:
            note(f"calibration_cells.csv has no rows for {miss} (fig4b shows the models present)")
        else:
            check(f"calibration_cells.csv: s present for all {len(set(C.model))} LLMs incl. GPT-6 Astra "
                  f"({len(As[As.model == 'gpt6astra'])} Astra cells)")
    series = series_list(set(C.model), [b for b in FIG4_BASES if b in set(Ap.model)])
    # 2 x 4 at the text width; y shared per row (inner tick labels dropped), thinned N labels, one x-label and
    # the legend below the bottom row (bottom_block)
    fig, axs = plt.subplots(2, 4, figsize=(FW, 3.6), sharey="row", layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.05)
    for j, ds in enumerate(DATASET_KEYS):
        ax = axs[0, j]
        a = Ap[Ap.dataset == ds]
        if not a.empty:
            ax.axhline(0.6827, color="black", lw=0.6, ls="-", zorder=1)
            for kk, (m, kind) in enumerate(series):
                plot_curve(ax, Ap, ds, m, kind, kk, len(series))
            size_axis(ax, sorted(a["size"].unique()), narrow=True)
        ds_title(ax, ds, wrap=True)
        ax = axs[1, j]
        if As is not None and not As[As.dataset == ds].empty:
            ax.axhline(1.0, color="black", lw=0.6, zorder=1)
            for kk, (m, kind) in enumerate(series):
                plot_curve(ax, As, ds, m, kind, kk, len(series), err=False)
            ax.set_yscale("log")
            size_axis(ax, sorted(As[As.dataset == ds]["size"].unique()), narrow=True)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    # common y-limits per row so the four datasets compare directly
    lo = min(a.get_ylim()[0] for a in axs[0]); hi = max(a.get_ylim()[1] for a in axs[0])
    for a in axs[0]:
        a.set_ylim(lo, hi)
    vis = [a for a in axs[1] if a.get_yscale() == "log"]
    if vis:
        lo = min(a.get_ylim()[0] for a in vis); hi = max(a.get_ylim()[1] for a in vis)
        for a in vis:
            a.set_ylim(lo, hi)
            a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 3, 4]))
            a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axs[0, 0].set_ylabel("PICP68 (raw)")
    axs[1, 0].set_ylabel(r"variance scale $s$" + chr(10) + "(>1: over-confident)")
    panel_label(axs[0, 0], "a")
    panel_label(axs[1, 0], "b")
    freeze_layout(fig)
    bottom_block(fig, axs, "context size N", row_major(legend_handles(series), 4), ncol=4)
    save(fig, out, "fig4_calibration")


# ---------------------------------------------------------------------------------- fig 5 (ESOL recall)
def _rms_z(e, sig):
    """root mean square of e / sigma, sigma floored at SIGMA_FLOOR (= the variance-scaling factor s)."""
    sig = np.clip(np.asarray(sig, float), SIGMA_FLOOR, None)
    return float(np.sqrt(np.mean((np.asarray(e, float) / sig) ** 2)))


def _tabpfn_esol_subsets(sizes, flags):
    """TabPFN (draw 1, mean over seeds) RMSE / RMS of e/sigma on recalled / never-recalled / all ESOL molecules,
    from the per-molecule baseline files and the zero-shot recall flags of esol_permolecule.csv."""
    rows = []
    for n in sizes:
        per_seed = []
        for suf in ["", "__s1", "__s2", "__s3", "__s4"]:
            f = os.path.join(BASE, f"tabpfn__esol__N{int(n)}{suf}.csv")
            if not os.path.exists(f):
                continue
            D = pd.read_csv(f, dtype={"mol_id": str})
            D["flag"] = D.mol_id.map(flags)
            for sub, g in (("recalled", D[D.flag == True]), ("never", D[D.flag == False]), ("all", D)):  # noqa: E712
                if len(g):
                    e = (g.prediction - g.truth).values
                    per_seed.append(dict(size=n, subset=sub, RMSE=np.sqrt(np.mean(e ** 2)),
                                         RMSZ=_rms_z(e, g.uncertainty.values), n=len(g)))
        if per_seed:
            P = pd.DataFrame(per_seed).groupby(["size", "subset"], as_index=False).mean()
            rows.append(P)
    return pd.concat(rows, ignore_index=True) if rows else None


def _llm_esol_rmsz(P, sizes, flags):
    """LLMs (draw 1, reasoning low): RMS of e/sigma (sigma = uc1 floored at SIGMA_FLOOR) on recalled /
    never-recalled / all ESOL molecules, from esol_permolecule.csv. Returns {(model, size, subset): (value, n)}."""
    out = {}
    g0 = P[(P.reasoning == REASONING) & (P.draw == 1) & P["size"].isin(sizes)].copy()
    g0["flag"] = g0.mol_id.map(flags)
    for (m, n), g in g0.groupby(["model", "size"]):
        for sub, x in (("recalled", g[g.flag == True]), ("never", g[g.flag == False]), ("all", g)):  # noqa: E712
            if len(x):
                out[(m, int(n), sub)] = (_rms_z((x.pred - x.truth).values, x.uc1.values), len(x))
    return out


def fig5(out):
    C = llm_cells()
    cols = ["RMSE_recalled", "RMSE_never_recalled", "n_recalled"]
    if C is None or not need(C, cols, "cells.csv (fig5)"):
        note("fig5 skipped")
        return
    E = C[(C.dataset == "esol") & (C.draw == 1)]
    if E.empty:
        note("fig5 skipped (no ESOL cells)")
        return
    nmin, nmax = int(E["size"].min()), int(E["size"].max())
    E = E[E["size"].isin([nmin, nmax])]
    models = [m for m in PANEL if m in set(E.model) and m in MODEL_NAMES]
    T, Z = None, {}
    if os.path.exists(P_ESOL_PM):
        P = pd.read_csv(P_ESOL_PM, dtype={"mol_id": str})
        if "draw" not in P.columns:
            P["draw"] = 1
        fl = P.dropna(subset=["recalled_zeroshot"]).drop_duplicates("mol_id")
        flags = dict(zip(fl.mol_id, fl.recalled_zeroshot.astype(str).str.lower().map({"true": True, "false": False})))
        T = _tabpfn_esol_subsets([nmin, nmax], flags)
        Z = _llm_esol_rmsz(P, [nmin, nmax], flags)
        nz = sorted({(k[2], v[1]) for k, v in Z.items()})
        if len(nz) == 3:
            check("fig5: RMS of e/sigma on " + ", ".join(f"{k} n = {v}" for k, v in nz) + " for every LLM cell")
        else:
            note(f"fig5: subset sizes of the e/sigma panel differ between LLM cells: {nz}")
    else:
        note("fig5: esol_permolecule.csv missing, right panel without LLMs")
        if T is not None:
            na = T[T.subset == "all"]["n"]
            if len(na) and (na != N_TEST).any():
                note(f"fig5: TabPFN 'all' tick rests on {sorted(set(na.astype(int)))} molecules, not {N_TEST}")
    rows = [(m, "llm") for m in models] + ([("tabpfn", "base")] if T is not None else [])
    # 1 x 2 at the text width, model rows shared; the subset note as a two-line title, legend in two rows
    # (markers / context sizes) directly below the x-labels (bottom_block)
    fig, axs = plt.subplots(1, 2, figsize=(FW, 2.75), sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04)
    off = {nmin: -0.17, nmax: 0.17}
    for ax, metric in zip(axs, ["RMSE", "RMSZ"]):
        for i, (m, kind) in enumerate(rows):
            col = MODEL_COLORS[m] if kind == "llm" else "black"
            for n in (nmin, nmax):
                if kind == "llm" and metric == "RMSZ":
                    if (m, n, "recalled") not in Z or (m, n, "never") not in Z:
                        continue
                    vr, vn = Z[(m, n, "recalled")][0], Z[(m, n, "never")][0]
                    va = Z.get((m, n, "all"), (np.nan,))[0]
                elif kind == "llm":
                    r = E[(E.model == m) & (E["size"] == n)]
                    if r.empty:
                        continue
                    vr, vn = r[f"{metric}_recalled"].iloc[0], r[f"{metric}_never_recalled"].iloc[0]
                    va = r[metric].iloc[0]
                else:
                    t = T[T["size"] == n].set_index("subset")
                    if not {"recalled", "never"} <= set(t.index):
                        continue
                    vr, vn = t.loc["recalled", metric], t.loc["never", metric]
                    va = t.loc["all", metric] if "all" in t.index else np.nan
                y = i + off[n]
                a = 1.0 if n == nmax else 0.55
                ax.plot([vr, vn], [y, y], color=col, lw=1.0, alpha=a, zorder=2)
                ax.plot(vr, y, "o", color=col, ms=4.5 * FS ** 0.5, alpha=a, zorder=3)
                ax.plot(vn, y, "o", mfc="white", mec=col, mew=1.0, ms=4.5 * FS ** 0.5, alpha=a, zorder=3)
                if np.isfinite(va):
                    ax.plot(va, y, "|", color=col, ms=6 * FS ** 0.5, mew=1.0, alpha=a, zorder=3)
        if metric == "RMSZ":
            ax.axvline(1.0, color="black", lw=0.6, zorder=1)
            ax.set_xlabel("RMS of e / σ")
        else:
            ax.set_xlabel("RMSE (logS)")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(rows)))
    axs[0].set_yticklabels([MODEL_NAMES.get(m, BASE_NAMES.get(m, m)) for m, _ in rows])
    axs[0].invert_yaxis()
    # n annotation: the recalled flag is pooled over the panel (recalled by >= 1 panel model), so the
    # subsets are the same for every model; checked here
    cnt = [c for c in ("n_recalled", "n_never_recalled", "n_unscorable_zeroshot") if c in E]
    nr = E.groupby("model")[cnt].first() if cnt else None
    if nr is not None and len(nr.drop_duplicates()) == 1 and "n_never_recalled" in nr:
        FIG_META["fig5_n"] = {c: int(nr[c].iloc[0]) for c in cnt}
        check("fig5: " + ", ".join(f"{c} = {v}" for c, v in FIG_META["fig5_n"].items()) +
              f" identical for all {len(nr)} models")
        sub = (f"recalled zero-shot by ≥1 panel model (n = {int(nr.n_recalled.iloc[0])}) vs never recalled "
               f"(n = {int(nr.n_never_recalled.iloc[0])})")
        if "n_unscorable_zeroshot" in nr:
            sub += f"; {int(nr.n_unscorable_zeroshot.iloc[0])} unscorable in neither; identical for every model"
    else:
        note("fig5: recall subset sizes differ between models (annotated per row)")
        sub = "n recalled / never recalled per model at the right"
        if nr is not None and "n_never_recalled" in nr:
            for i, (m, _) in enumerate(rows):
                if m in nr.index:
                    axs[1].text(1.01, i, f"{int(nr.loc[m, 'n_recalled'])}/{int(nr.loc[m, 'n_never_recalled'])}",
                                transform=axs[1].get_yaxis_transform(), fontsize=6.5 * FS, va="center")
    h = [Line2D([], [], color="grey", marker="o", ls="", ms=4.5 * FS ** 0.5, label="recalled"),
         Line2D([], [], color="grey", marker="o", mfc="white", ls="", ms=4.5 * FS ** 0.5, label="never recalled"),
         Line2D([], [], color="grey", marker="|", ls="", ms=6 * FS ** 0.5, mew=1.0, label=f"all {N_TEST} test molecules"),
         Line2D([], [], color="grey", lw=1.0, alpha=0.55, label=f"N = {nmin} (upper, faint)"),
         Line2D([], [], color="grey", lw=1.0, label=f"N = {nmax} (lower)")]
    fig.suptitle(chr(10).join(__import__("textwrap").wrap("ESOL: " + sub, 100)), fontsize=7.5 * FS)
    freeze_layout(fig)
    bottom_block(fig, axs, None, h, ncol=5, handlelength=1.4, columnspacing=0.7, handletextpad=0.4)
    FIG_META["fig5_N"] = (nmin, nmax)
    save(fig, out, "fig5_esol_recall")


# ---------------------------------------------------------------------------------- fig 6 (ladder)
def fig6(out):
    L = read_csv(P_LADDER, "ladder_cells")
    if L is None or not need(L, ["dataset", "model", "size", "effort", "RMSE", "rho_UC1_err", "PICP68",
                                 "median_reasoning_tokens"], "ladder_cells.csv"):
        note("fig6 skipped")
        return
    # rho = the tie-rounded rho of selective_prediction.csv (draw 1, reasoning = effort), as in the tables
    Sp = read_csv(P_SEL, "selective prediction (fig6 rho)")
    if Sp is not None and need(Sp, ["ds", "model", "size", "reasoning", "rho", "n_mol"], "selective_prediction.csv (fig6)"):
        Sp = Sp[(Sp.family == "LLM") & (Sp.draw.fillna(1).astype(int) == 1)].rename(
            columns={"ds": "dataset", "reasoning": "effort", "rho": "rho_sel", "n_mol": "n_mol_sel"})
        L = L.merge(Sp[["dataset", "model", "size", "effort", "rho_sel", "n_mol_sel"]],
                    on=["dataset", "model", "size", "effort"], how="left")
        bad = L.rho_sel.isna() | (("n_mol" in L) and (L.n_mol != L.n_mol_sel))
        if bad.any():
            note(f"fig6: {int(bad.sum())} ladder cells without a matching selective_prediction.csv row, "
                 "ladder_cells.csv rho_UC1_err kept for them")
        dmax = (L.rho_sel - L.rho_UC1_err).abs().max()
        L["rho_UC1_err"] = np.where(bad, L.rho_UC1_err, L.rho_sel)
        check(f"fig6: rho from selective_prediction.csv for {int((~bad).sum())} of {len(L)} ladder cells "
              f"(max |diff| to ladder_cells.csv rho_UC1_err {dmax:.4f})")
    pos = L.loc[L.median_reasoning_tokens > 0, "median_reasoning_tokens"]
    minpos = pos.min() if len(pos) else 100.0
    x0 = 10 ** (np.floor(np.log10(minpos)) - 0.35)            # pseudo-position of "0 tokens"
    xbreak = np.sqrt(x0 * minpos)
    xmax = L.median_reasoning_tokens.max() * 2.0
    variants = []
    sizes_per_ds = L.groupby("dataset")["size"].agg(["min", "max"])
    variants.append(("fig6a_ladder_smallN", "min"))
    if (sizes_per_ds["max"] != sizes_per_ds["min"]).any():
        variants.append(("fig6b_ladder_largeN", "max"))
    # ladder models only: a model with a single effort level (GPT-6 Astra: just `low`, copied from the main run)
    # is not part of the ladder sub-experiment and would appear as a lone point plus a legend entry
    nlev = L.groupby("model").effort.nunique()
    models = [m for m in PANEL if m in set(L.model) and m in MODEL_NAMES and nlev.get(m, 0) >= 2]
    left = sorted(set(L.model) - set(models))
    FIG_META["fig6_models"] = (len(models), [MODEL_NAMES.get(m, m) for m in left])
    if left:
        note(f"fig6: left out (only one effort level in ladder_cells.csv, not in the ladder): {left}")
    for name, which in variants:
        fig, axs = plt.subplots(3, 4, figsize=(FW, 7.20), sharex=True)
        for j, ds in enumerate(DATASET_KEYS):
            if ds not in sizes_per_ds.index:
                for ax in axs[:, j]:
                    ax.set_visible(False)
                continue
            n = sizes_per_ds.loc[ds, which]
            d = L[(L.dataset == ds) & (L["size"] == n)]
            axs[0, j].set_title(DS_NAMES.get(ds, ds).replace(" (", "\n(") + f"\nN = {int(n)}",
                                color="#777777" if ds in CONTAMINATED else "black")
            for i, metric in enumerate(["RMSE", "rho_UC1_err", "PICP68"]):
                ax = axs[i, j]
                for k, m in enumerate(models):
                    g = d[d.model == m].copy()
                    if g.empty:
                        continue
                    g["lv"] = g.effort.map({lv: q for q, lv in enumerate(LEVELS)})
                    g = g.sort_values("lv")
                    jit = x0 * 1.13 ** (k - (len(models) - 1) / 2)      # separate the models stacked at 0 tokens
                    x = np.where(g.median_reasoning_tokens > 0, g.median_reasoning_tokens, jit)
                    ax.plot(x, g[metric], color=MODEL_COLORS[m], lw=0.9, zorder=2)
                    for xi, yi, lv in zip(x, g[metric], g.effort):
                        ax.plot(xi, yi, LEVEL_MARKERS.get(lv, "o"), color=MODEL_COLORS[m], ms=4 * FS ** 0.5,
                                mfc=MODEL_COLORS[m] if lv != "none" else "none", mew=1.0, zorder=3)
                if metric == "PICP68":
                    ax.axhline(0.6827, color="black", lw=0.6, zorder=1)
                if metric == "rho_UC1_err":
                    ax.axhline(0, color="black", lw=0.6, zorder=1)
                ax.set_xscale("log")
                ax.set_xlim(x0 / 1.8, xmax)
                ax.axvline(xbreak, color="#bbbbbb", lw=0.6, ls=(0, (2, 2)), zorder=0)
        for ax in axs[-1]:
            ticks = [x0] + [10 ** e for e in range(int(np.ceil(np.log10(minpos))), int(np.log10(xmax)) + 1)]
            ax.set_xticks(ticks)
            ax.set_xticklabels(["0"] + [f"$10^{{{int(np.log10(t))}}}$" for t in ticks[1:]])
            # minor ticks only right of the axis break (left of it there is just the "0" position)
            minor = [f * 10 ** e for e in range(int(np.floor(np.log10(xbreak))), int(np.log10(xmax)) + 1)
                     for f in range(2, 10) if xbreak < f * 10 ** e < xmax]
            ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(minor))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for i, lab in enumerate(["RMSE", r"$\rho_s(\sigma, |\mathrm{err}|)$", "PICP68"]):
            axs[i, 0].set_ylabel(lab)
            lo = min(a.get_ylim()[0] for a in axs[i] if a.get_visible())
            hi = max(a.get_ylim()[1] for a in axs[i] if a.get_visible())
            if i > 0:
                for a in axs[i]:
                    a.set_ylim(lo, hi)
        h = [Line2D([], [], color=MODEL_COLORS[m], lw=1.2, label=MODEL_NAMES[m]) for m in models]
        h += [Line2D([], [], color="grey", marker=LEVEL_MARKERS[lv], ls="", ms=4 * FS ** 0.5,
                     mfc="grey" if lv != "none" else "none", label=f"effort {lv}") for lv in LEVELS]
        fig.legend(handles=h, loc="lower center", ncol=6 if FS <= 1.15 else 4, frameon=False, bbox_to_anchor=(0.5, -0.005),
                   columnspacing=1.1)
        fig.supxlabel("median emitted reasoning tokens per call (0 = no reasoning tokens)", fontsize=8 * FS, y=0.085)
        fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=0.5, h_pad=0.6)
        for ax in axs[-1]:
            if ax.get_visible():   # axis-break marks between the "0" position and the log range
                ax.text(xbreak, 0, "//", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=7 * FS, color="black", backgroundcolor="white", clip_on=False, zorder=5)
        save(fig, out, name)


# ---------------------------------------------------------------------------------- appendix figures (A1-A5)
def _permolecule(ds):
    """results/icl/<ds>_permolecule.csv (cached), reasoning REASONING only, with a draw column."""
    key = ("pm", ds)
    if key not in _cache:
        P = read_csv(os.path.join(ICL, f"{ds}_permolecule.csv"), f"{ds}_permolecule")
        if P is not None:
            if "draw" not in P.columns:
                P["draw"] = 1
            P = P[P.reasoning == REASONING]
        _cache[key] = P
    return _cache[key]


def _llm_err_sig(ds, m, n, draw=1):
    """signed error and UC1 of one complete LLM cell (None if absent or incomplete)."""
    P = _permolecule(ds)
    if P is None:
        return None
    g = P[(P.model == m) & (P["size"] == n) & (P.draw == draw)]
    g = g[np.isfinite(g.pred) & np.isfinite(g.uc1)]
    if len(g) < MIN_COVERAGE * N_TEST:
        return None
    return (g.pred - g.truth).values, g.uc1.values


def _base_files(method, ds, n, draw=1):
    """per-seed per-cell baseline files of one (method, dataset, N, draw), parsed with the file rule of
    src/selective_prediction.py (BASE_RE)."""
    from selective_prediction import BASE_RE
    out = []
    for f in sorted(os.listdir(BASE)):
        mt = BASE_RE.match(f)
        if mt and mt["method"] == method and mt["ds"] == ds and int(mt["size"]) == int(n) and int(mt["draw"] or 1) == draw:
            out.append(os.path.join(BASE, f))
    return out


RC_N = 30
RC_LLMS = ["sonnet55", "haiku55", "gpt6astra"]
RC_BASES = ["tabpfn", "gp_tanimoto"]
RC_REF = ["sonnet55", "tabpfn"]          # oracle / random drawn for these two
RC_Q = tuple(np.round(np.arange(0.1, 1.01, 0.1), 2))


def figA1(out):
    """risk-coverage curves at N = RC_N, draw 1."""
    try:
        from selective_prediction import risk_coverage
    except Exception as exc:
        note(f"figA1 skipped (could not import selective_prediction.risk_coverage: {exc})")
        return
    # 2 x 2 at the text width, q axis shared (labels on the bottom row), y-label on the left column, one x-label
    # and the legend below (bottom_block): curves in two rows, the long reference entry on its own row
    fig, axs = plt.subplots(2, 2, figsize=(FW, 3.7), sharex=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
    summary = []
    for ax, ds in zip(axs.flat, DATASET_KEYS):
        curves = {}
        for m in PANEL:
            es = _llm_err_sig(ds, m, RC_N)
            if es is not None:
                curves[m] = risk_coverage(*es, fracs=RC_Q)
        missing = [m for m in PANEL if m not in curves]
        if missing:
            note(f"figA1 {ds}: no complete N = {RC_N} draw-1 cell for {missing} (left out of the panel median)")
        for b in RC_BASES:
            fs = _base_files(b, ds, RC_N)
            per = []
            for f in fs:
                D = pd.read_csv(f)
                per.append(risk_coverage(D.prediction - D.truth, D.uncertainty, fracs=RC_Q))
            if per:      # mean over the seeds of draw 1 (rule of all figures)
                curves[b] = pd.concat(per).groupby("q", as_index=False).mean(numeric_only=True)
        if not curves:
            ax.set_visible(False)
            continue
        q = np.array(RC_Q)
        llm_present = [m for m in PANEL if m in curves]
        med = np.median(np.vstack([curves[m].RMSE_q.values for m in llm_present]), axis=0)
        # references first (behind), dotted
        for m in RC_REF:
            if m not in curves:
                continue
            col = MODEL_COLORS.get(m, "black")
            ax.plot(q, curves[m].oracle, color=col, ls=":", lw=1.0, zorder=1)
            ax.plot(q, curves[m].random, color=col, ls=":", lw=1.0, zorder=1)
        ax.plot(q, med, color="#8c8c8c", lw=2.6, alpha=0.55, zorder=2, solid_capstyle="round")
        for b in RC_BASES:
            if b in curves:
                st = style_of(b, "base")
                ax.plot(q, curves[b].RMSE_q, color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"],
                        mec=st["color"], lw=1.0, ms=3.2 * FS ** 0.5, mew=0.7, zorder=3)
        for m in RC_LLMS:
            if m in curves:
                ax.plot(q, curves[m].RMSE_q, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], lw=1.2, ms=3.2 * FS ** 0.5, zorder=4)
        ax.set_xticks(q)
        ax.set_xticklabels([f"{v:.1f}" for v in q])
        ax.set_xlim(0.06, 1.04)
        ax.set_ylim(bottom=0)
        if ds in CONTAMINATED:      # cut above the LLMs and TabPFN; off-scale baselines marked by arrows at the top edge
            keep = [curves[m].RMSE_q.max() for m in llm_present] + ([curves["tabpfn"].RMSE_q.max()] if "tabpfn" in curves else [])
            top = 1.15 * max(keep)
            ax.set_ylim(0, top)
            for b in RC_BASES:
                if b in curves and curves[b].RMSE_q.min() > top:
                    v = curves[b].RMSE_q
                    st = style_of(b, "base")      # arrows at the top edge, as cap_axis (no value label)
                    ax.plot(q, np.full(len(q), top * 0.975), ls="", marker=r"$\uparrow$", ms=7 * FS ** 0.5,
                            color=st["color"], mec=st["color"], mfc=st["color"], mew=0.3, zorder=6, clip_on=False)
                    FIG_META["figA1_cut"] = (round(top, 2), BASE_NAMES[b], round(v.min(), 2), round(v.max(), 2))
        ds_title(ax, ds)
        if ax in axs[:, 0]:
            ax.set_ylabel("RMSE of kept molecules")
        for m in RC_LLMS + ["median"] + RC_BASES:
            v = med if m == "median" else (curves[m].RMSE_q.values if m in curves else None)
            if v is not None:
                summary.append(f"{ds:13s} {m:12s} q=0.1 {v[0]:.2f}  q=0.5 {v[4]:.2f}  q=1 {v[-1]:.2f}  "
                               f"(1 - RMSE_0.2/RMSE = {1 - v[1] / v[-1]:.2f})")
        for m in RC_REF:
            if m in curves:
                summary.append(f"{ds:13s} {m:12s} oracle q=0.1 {curves[m].oracle.iloc[0]:.2f} q=0.5 {curves[m].oracle.iloc[4]:.2f}")
    print(f"figA1 values (N = {RC_N}, draw 1; baselines seed mean):")
    for t in summary:
        print("   ", t)
    h = [Line2D([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], lw=1.2, ms=4 * FS ** 0.5, label=MODEL_NAMES[m]) for m in RC_LLMS]
    h.append(Line2D([], [], color="#8c8c8c", lw=2.6, alpha=0.55, label="panel median (7 LLMs)"))
    h += legend_handles([(b, "base") for b in RC_BASES])
    h += [Line2D([], [], color="grey", ls=":", lw=1.0,
                 label="oracle (by |err|) and random (= RMSE), for Sonnet 5.5 and TabPFN in their colours")]
    freeze_layout(fig)
    bottom_block(fig, axs, f"fraction q kept (smallest stated sigma first), N = {RC_N}",
                 [row_major(h[:-1], 3), h[-1:]], ncol=[3, 1], handlelength=2.4)
    save(fig, out, "figA1_risk_coverage")


def figA2(out):
    """learning curves per context draw at N in DRAW_SIZES."""
    C, B = llm_cells(), base_cells()
    if C is None:
        note("figA2 skipped (no cells.csv)")
        return
    frames = [C[C["size"].isin(DRAW_SIZES)][["dataset", "model", "size", "draw", "RMSE"]]]
    if B is not None:
        frames.append(B[(B.model == "tabpfn") & B["size"].isin(DRAW_SIZES)][["dataset", "model", "size", "draw", "RMSE"]])
    D = pd.concat(frames, ignore_index=True)
    series = series_list(set(C.model), ["tabpfn"] if B is not None else [])
    # three context sizes only: one row of four panels at the text width, y-label on the first panel, one
    # x-label and the legend below (bottom_block)
    fig, axs = plt.subplots(1, 4, figsize=(FW, 2.55), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    rep = []
    for ax, ds in zip(axs.flat, DATASET_KEYS):
        d = D[D.dataset == ds]
        if d.empty:
            ax.set_visible(False)
            continue
        for k, (m, kind) in enumerate(series):
            st = style_of(m, kind)
            dodge = 1.0 + 0.022 * (k - (len(series) - 1) / 2)
            for dr in sorted(d.draw.unique()):
                g = d[(d.model == m) & (d.draw == dr)].sort_values("size")
                if g.empty:
                    continue
                x = g["size"].values * dodge
                if dr == 1:
                    ax.plot(x, g.RMSE, color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"], mec=st["color"],
                            lw=1.8, ms=4 * FS ** 0.5, mew=0.7, zorder=4)
                else:
                    ax.plot(x, g.RMSE, color=st["color"], ls=st["ls"], lw=0.7, alpha=0.7, marker=".", ms=2.5 * FS ** 0.5, zorder=3)
        size_axis(ax, list(DRAW_SIZES))
        ax.set_xlim(DRAW_SIZES[0] / 1.15, DRAW_SIZES[-1] * 1.15)
        ds_title(ax, ds, wrap=True)
        if ax is axs.flat[0]:
            ax.set_ylabel("RMSE")
        # reading: draw spread vs learning-curve step, curve shape per draw
        L = d[d.model.isin(MODEL_NAMES)]
        rng = L.groupby(["model", "size"]).RMSE.agg(lambda v: v.max() - v.min())
        w = L.pivot_table(index=["model", "draw"], columns="size", values="RMSE")
        step = (w[DRAW_SIZES[0]] - w[DRAW_SIZES[-1]])
        inv = {dr: int(((w.xs(dr, level="draw").diff(axis=1) > 0).any(axis=1)).sum()) for dr in sorted(L.draw.unique())}
        rep.append(f"{ds:13s} LLM draw range median {rng.median():.3f} (max {rng.max():.3f}); draw-1 step N{DRAW_SIZES[0]}->"
                   f"N{DRAW_SIZES[-1]} median {step.xs(1, level='draw').median():.3f}; LLMs with a non-monotone curve "
                   f"per draw {inv}")
        FIG_META.setdefault("figA2", {})[ds] = dict(range_med=float(rng.median()), range_max=float(rng.max()),
                                                     step_med=float(step.xs(1, level="draw").median()), inv=inv)
    print("figA2 values (RMSE, LLMs):")
    for t in rep:
        print("   ", t)
    extra = [Line2D([], [], color="grey", lw=1.8, marker="o", ms=4 * FS ** 0.5, label="context draw 1"),
             Line2D([], [], color="grey", lw=0.7, alpha=0.7, marker=".", ms=2.5 * FS ** 0.5, label="draws 2 and 3")]
    h = legend_handles(series) + extra
    freeze_layout(fig)
    bottom_block(fig, axs, "context size N", row_major(h, 5), ncol=5, handlelength=2.0, columnspacing=0.8,
                 handletextpad=0.4)
    save(fig, out, "figA2_draws")


def figA3(out):
    """rho heatmap model x N per dataset, draw 1, with bootstrap markers vs TabPFN."""
    S = selective()
    if S is None:
        note("figA3 skipped (no selective_prediction.csv)")
        return
    S = S[S.draw == 1]
    rows = [m for m in PANEL if m in set(S.model) and m in MODEL_NAMES] + [b for b in DISPLAY_BASES if b in set(S.model)]
    Bt = read_csv(os.path.join(ICL, "bootstrap_cells.csv"), "bootstrap_cells")
    if Bt is not None:
        Bt = Bt[(Bt.stat == "rho") & (Bt.baseline == "tabpfn") & (Bt.draw == 1)]
    vals = S[S.model.isin(rows)].rho.abs()
    vmax = float(np.ceil(vals.max() * 10) / 10) if len(vals) else 1.0
    cmap = plt.get_cmap("RdBu_r")
    norm = matplotlib.colors.Normalize(-vmax, vmax)
    fig, axs = plt.subplots(2, 2, figsize=(FW, 8.88))
    nsig = {}
    for ax, ds in zip(axs.flat, DATASET_KEYS):
        d = S[(S.dataset == ds) & S.model.isin(rows)]
        if d.empty:
            ax.set_visible(False)
            continue
        sizes = sorted(d["size"].unique())
        M = d.pivot_table(index="model", columns="size", values="rho").reindex(index=rows, columns=sizes)
        ax.imshow(M.values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        for i, m in enumerate(rows):
            for j, n in enumerate(sizes):
                v = M.iloc[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "-", ha="center", va="center", fontsize=6.5 * FS, color="#777777")
                    continue
                ax.text(j, i, f"{v:.2f}".replace("0.", ".", 1) if v >= 0 else f"{v:.2f}".replace("-0.", "−.", 1),
                        ha="center", va="center", fontsize=6.5 * FS, color="white" if abs(v) > 0.55 * vmax else "black")
                if Bt is not None and m in MODEL_NAMES:
                    b = Bt[(Bt.dataset == ds) & (Bt.model == m) & (Bt["size"] == n)]
                    if len(b) and (b.lo.iloc[0] > 0 or b.hi.iloc[0] < 0):
                        pos = b.lo.iloc[0] > 0
                        ax.plot(j + 0.36, i - 0.3, "o", ms=2.6 * FS ** 0.5, mew=0.7, mec="black", mfc="black" if pos else "white",
                                zorder=5)
                        nsig.setdefault(ds, {"+": 0, "-": 0})["+" if pos else "-"] += 1
        ax.axhline(sum(m in MODEL_NAMES for m in rows) - 0.5, color="white", lw=2.0)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(int(n)) for n in sizes])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([MODEL_NAMES.get(m, BASE_NAMES.get(m, m)) for m in rows])
        ax.tick_params(length=0)
        ax.grid(False)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ds_title(ax, ds)
        ax.set_xlabel("context size N")
        if Bt is not None:
            have = sorted(Bt[Bt.dataset == ds]["size"].unique())
            FIG_META.setdefault("figA3_boot_sizes", {})[ds] = [int(v) for v in have]
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0, h_pad=1.2)
    cax = fig.add_axes([0.08, 0.04, 0.36, 0.012])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(r"$\rho_s(\sigma, |\mathrm{err}|)$, context draw 1", fontsize=7.5 * FS)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)
    fig.text(0.52, 0.045, "●  95 % bootstrap CI of ρ(LLM) − ρ(TabPFN) above 0\n"
             "○  95 % bootstrap CI of ρ(LLM) − ρ(TabPFN) below 0\n"
             "no mark: CI includes 0, or no bootstrap (N = 250, 500)", fontsize=6.8 * FS, ha="left", va="center",
             linespacing=1.5)
    FIG_META["figA3_nsig"] = nsig
    FIG_META["figA3_vmax"] = vmax
    print("figA3: cells with CI(rho_LLM - rho_TabPFN) excluding 0, per dataset (+ LLM higher / - lower):", nsig)
    save(fig, out, "figA3_rho_heatmap")


def figA4(out):
    """USD per cell vs RMSE (LLMs) and CPU wall time vs RMSE (baselines), N = 30 and the largest N."""
    C, B = llm_cells(), base_cells()
    if C is None or not need(C, ["cost_usd"], "cells.csv (figA4)"):
        note("figA4 skipped")
        return
    C1 = C[C.draw == 1]
    has_wall = B is not None and "wall_s" in B.columns
    if has_wall:
        B1 = B[(B.draw == 1) & B.model.isin(DISPLAY_BASES)]
    # 2 x 4 at the text width (LLM cost row taller than the baseline wall-time row), y-labels on the first
    # column, decade labels thinned on the wall-time axes, legend below (bottom_block)
    fig = plt.figure(figsize=(FW, 3.9 if has_wall else 2.4), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0] if has_wall else [1, 0.01])
    all_axes = []
    rep = []
    for j, ds in enumerate(DATASET_KEYS):
        d = C1[C1.dataset == ds]
        if d.empty:
            continue
        nmin, nmax = int(d["size"].min()), int(d["size"].max())
        ax = fig.add_subplot(gs[0, j])
        all_axes.append(ax)
        for m in PANEL:
            g = d[d.model == m].set_index("size")
            if m not in MODEL_NAMES or not {nmin, nmax} <= set(g.index):
                continue
            x, y = g.loc[[nmin, nmax], "cost_usd"].values, g.loc[[nmin, nmax], "RMSE"].values
            col = MODEL_COLORS[m]
            ax.plot(x, y, color=col, lw=0.8, alpha=0.7, zorder=2)
            ax.plot(x[0], y[0], MODEL_MARKERS[m], mfc="white", mec=col, mew=1.0, ms=4.5 * FS ** 0.5, zorder=3)
            ax.plot(x[1], y[1], MODEL_MARKERS[m], color=col, mec=col, ms=4.5 * FS ** 0.5, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ds_title(ax, ds, wrap=True)
        ax.set_xlabel("USD per cell")
        if j == 0:
            ax.set_ylabel("RMSE (LLMs)")
        lo_, hi_ = d[d["size"] == nmin], d[d["size"] == nmax]
        rep.append(f"{ds:13s} USD N{nmin}: {lo_.cost_usd.min():.2f}-{lo_.cost_usd.max():.2f}  N{nmax}: "
                   f"{hi_.cost_usd.min():.2f}-{hi_.cost_usd.max():.2f}; best RMSE N{nmax} {hi_.loc[hi_.RMSE.idxmin(), 'model']} "
                   f"{hi_.RMSE.min():.2f} at ${hi_.loc[hi_.RMSE.idxmin(), 'cost_usd']:.2f}")
        if has_wall:
            ax2 = fig.add_subplot(gs[1, j])
            all_axes.append(ax2)
            b = B1[B1.dataset == ds]
            for bm in DISPLAY_BASES:
                g = b[b.model == bm].set_index("size")
                if not {nmin, nmax} <= set(g.index):
                    continue
                st = style_of(bm, "base")
                x, y = g.loc[[nmin, nmax], "wall_s"].values, g.loc[[nmin, nmax], "RMSE"].values
                ax2.plot(x, y, color=st["color"], ls=st["ls"], lw=0.8, zorder=2)
                ax2.plot(x[0], y[0], st["marker"], mfc="white", mec=st["color"], mew=1.0, ms=4.5 * FS ** 0.5, zorder=3)
                ax2.plot(x[1], y[1], st["marker"], color=st["color"], mec=st["color"], ms=4.5 * FS ** 0.5, zorder=3)
            ax2.set_xscale("log")
            ax2.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
            # every other decade labelled (1, 100, 10^4 s), so the labels of the narrow panels do not touch
            ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:g}" if round(np.log10(v)) % 2 == 0 else ""))
            ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax2.set_xlabel("CPU wall time (s)")
            if j == 0:
                ax2.set_ylabel("RMSE (baselines)")
            rep.append(f"{ds:13s} baseline wall s N{nmin}: " + ", ".join(
                f"{bm} {b[(b.model == bm) & (b['size'] == nmin)].wall_s.mean():.1f}" for bm in DISPLAY_BASES) +
                f"; N{nmax}: " + ", ".join(f"{bm} {b[(b.model == bm) & (b['size'] == nmax)].wall_s.mean():.1f}" for bm in DISPLAY_BASES))
    if not has_wall:
        note("figA4: summary.csv has no wall_s; baselines left out")
    print("figA4 values (draw 1):")
    for t in rep:
        print("   ", t)
    h = legend_handles([(m, "llm") for m in PANEL if m in set(C1.model) and m in MODEL_NAMES], line=False)
    if has_wall:
        h += legend_handles([(b, "base") for b in DISPLAY_BASES], line=False)
    h += [Line2D([], [], ls="", marker="o", mfc="white", mec="grey", mew=1.0, ms=4.5 * FS ** 0.5, label=f"N = {JOINT_N} (open)"),
          Line2D([], [], ls="", marker="o", color="grey", ms=4.5 * FS ** 0.5, label="largest N (filled)")]
    freeze_layout(fig)
    bottom_block(fig, all_axes, None, row_major(h, 5), ncol=5, handlelength=1.2, columnspacing=0.9, handletextpad=0.3)
    save(fig, out, "figA4_cost")


SIGMA_EXAMPLE = ("lipophilicity", "sonnet55", 30)


def figA5(out):
    """distinct stated sigma values per cell vs N, and the sigma values of one cell."""
    Sp = read_csv(P_SEL, "selective prediction (figA5)")
    if Sp is None or not need(Sp, ["n_sigma_distinct", "n_mol"], "selective_prediction.csv (figA5)"):
        note("figA5 skipped")
        return
    Sp = Sp.rename(columns={"ds": "dataset"})
    Sp["draw"] = Sp.draw.fillna(1).astype(int)
    L = Sp[(Sp.family == "LLM") & (Sp.reasoning == REASONING) & (Sp.draw == 1) & (Sp.n_mol >= MIN_COVERAGE * N_TEST)]
    Bm = Sp[(Sp.family == "baseline") & (Sp.draw == 1) & Sp.model.isin(DISPLAY_BASES)]
    fig = plt.figure(figsize=(FW, 6.48))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85])
    top = 0
    for j, ds in enumerate(DATASET_KEYS):
        ax = fig.add_subplot(gs[0, j])
        d = L[L.dataset == ds]
        for k, m in enumerate([m for m in PANEL if m in MODEL_NAMES]):
            g = d[d.model == m].sort_values("size")
            if g.empty:
                continue
            dodge = 1.0 + 0.02 * (k - 3)
            ax.plot(g["size"] * dodge, g.n_sigma_distinct, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], ms=3.2 * FS ** 0.5, lw=1.0)
        top = max(top, d.n_sigma_distinct.max())
        size_axis(ax, sorted(d["size"].unique()), narrow=True)
        ds_title(ax, ds, wrap=True)
        ax.set_xlabel("context size N")
        if j == 0:
            ax.set_ylabel("distinct sigma values\nper cell (of ~150)")
    for ax in fig.axes:
        ax.set_ylim(0, top * 1.08)
    bmin = Bm.groupby("model").n_sigma_distinct.min()
    FIG_META["figA5_base_min"] = {b: int(v) for b, v in bmin.items()}
    FIG_META["figA5_llm"] = L.groupby("model").n_sigma_distinct.agg(["min", "median", "max"]).astype(float).round(1).to_dict("index")
    print("figA5: distinct sigma values, LLM cells (draw 1, low):", FIG_META["figA5_llm"])
    print("figA5: display baselines, minimum distinct sigma values over draw-1 cells:", FIG_META["figA5_base_min"])
    # histogram of one representative cell: counts per distinct sigma value (ties visible directly)
    ds, m, n = SIGMA_EXAMPLE
    ax = fig.add_subplot(gs[1, :])
    es = _llm_err_sig(ds, m, n)
    if es is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        err, sig = es
        sg = np.round(sig, 10)
        u, cnt = np.unique(sg, return_counts=True)
        col = MODEL_COLORS[m]
        ax.vlines(u, 0, cnt, color=col, lw=1.6)
        ax.plot(u, cnt, "o", color=col, ms=2.8 * FS ** 0.5)
        k = int(0.5 * len(sg))
        t = np.sort(sg, kind="stable")[k - 1]
        nb, nt = int((sg < t).sum()), int((sg == t).sum())
        ax.axvline(t, color="black", lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.text(t, cnt.max() * 1.03, f"  cut for the 50 % most certain (k = {k} of {len(sg)}):\n  {nb} molecules below, "
                f"{k - nb} of the {nt} at sigma = {t:.3g} kept", fontsize=6.8 * FS, va="bottom", ha="left")
        ax.set_ylim(0, cnt.max() * 1.3)
        ax.set_xlabel(f"stated sigma (mean over R = 3 repeats), {MODEL_NAMES[m]}, {DS_NAMES[ds].split(' (')[0].lower()}, "
                      f"N = {n}, context draw 1")
        ax.set_ylabel("molecules")
        tied = int(cnt[cnt >= 2].sum())
        FIG_META["figA5_example"] = dict(n=len(sg), distinct=len(u), largest=int(cnt.max()),
                                         largest_at=float(u[cnt.argmax()]), tied=tied, cut=float(t), nb=nb, nt=nt, k=k)
        print("figA5 example cell:", FIG_META["figA5_example"])
    h = legend_handles([(mm, "llm") for mm in PANEL if mm in MODEL_NAMES])
    fig.legend(handles=row_major(h, 4), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005),
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=1.2, w_pad=0.6)
    panel_label(fig.axes[0], "a")
    panel_label(ax, "b")
    save(fig, out, "figA5_sigma_values")


# ---------------------------------------------------------------------------------- README
def _bases_txt(bases):
    names = [BASE_NAMES[b] for b in bases]
    return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]


def readme_entry(key):
    """(files, data, caption) of one figure; captions follow the draw rule of this run."""
    D = _bases_txt(DISPLAY_BASES)
    cut = FIG_META.get("esol_cut", {})
    cuts = lambda k: f"{cut[k]:g}" if k in cut else f"{ESOL_CAP:g} x the highest LLM value"  # noqa: E731
    clip = lambda k: (f"ESOL y-axis cut at {cuts(k)} so the LLM differences stay readable: baseline points above are drawn "
            "as an upward arrow at the top edge (no data marker, no value label).")
    if key == "fig1":
        return ("fig1_rmse_vs_n",
                "cells.csv (reasoning low), baselines/summary.csv (seed mean per draw)",
                f"Test RMSE vs number of in-context molecules N for the seven LLMs (reasoning `low`) and the baselines "
                f"{D} (all eight baselines in the tables). {draw_txt()} {clip('fig1')}")
    if key == "fig2":
        return ("fig2_uncertainty_usefulness",
                "selective_prediction.csv (rho = Spearman with sigma rounded to 10 decimals, the value of the tables, "
                "and RMSE50; LLM rows checked against cells.csv, stale or missing LLM cells would be recomputed in memory "
                "from icl/<ds>_permolecule.csv; baselines averaged over seeds within a draw)",
                "(a) Spearman rank correlation between the reported sigma and the absolute error vs N; (b) RMSE of the "
                "50 % most certain test molecules by sigma (equal sigmas at the cut broken at random; exact square root of the expected MSE); LLMs vs "
                f"{D} in both rows. {draw_txt()} (b) {clip('fig2')}")
    if key == "fig3":
        oth = FIG_META.get("fig3_others") or [BASE_NAMES[b] for b in OTHER_BASES]
        pt = ("mean over the context draws, bars = SD over draws" if DRAW_MEAN else
              "context draw 1, bars = range (min-max) over the three context draws in x and in y")
        return ("fig3_joint_smallN",
                f"selective_prediction.csv at N = {JOINT_N} (RMSE, RMSE50, gain50; LLM rows checked against cells.csv; "
                "baselines: RMSE and gain50 averaged over seeds within a draw, so a baseline point lies only "
                "approximately on its iso-RMSE50 line)",
                f"Accuracy and ranking jointly at N = {JOINT_N} in-context molecules: x = test RMSE on all 150 molecules, "
                "y = gain50 = 1 - RMSE50 / RMSE, the relative error reduction when only the 50 % of molecules with the "
                "smallest stated sigma are kept (tie-aware). LLMs filled (model colours), baselines open: "
                f"{D} in their display styles, {', '.join(oth)} in light grey. Marker = {pt}. Dashed lines: constant "
                "RMSE50 = (1 - gain50) x RMSE (value labelled); a point further left and higher has a lower RMSE50. "
                "ESOL (contaminated) in the fourth panel, greyed title.")
    if key == "fig4":
        return ("fig4_calibration",
                "cells.csv + baselines/summary.csv (PICP68); calibration_cells.csv (s, draw 1, signal UC1 / TabPFN "
                "sigma of seed 0)",
                "(a) Raw coverage of the +/- 1 sigma interval vs N (line: nominal 0.683); (b) variance-scaling factor s "
                "that makes the z-scores unit-variance (s > 1: over-confident), log scale, context draw 1 only (no "
                f"bars). Baseline shown: TabPFN only (all baselines in the tables). (a) {draw_txt()}")
    if key == "fig5":
        n = FIG_META.get("fig5_n", {})
        nmin, nmax = FIG_META.get("fig5_N", ("smallest", "largest"))
        cnt = (f"n_recalled = {n.get('n_recalled')}, never recalled = {n.get('n_never_recalled')}, unscorable = "
               f"{n.get('n_unscorable_zeroshot')}" if n else "the subset sizes")
        return ("fig5_esol_recall",
                "cells.csv (ESOL, reasoning low, draw 1: RMSE _recalled/_never_recalled); RMS of e/sigma (sigma = uc1 "
                "floored at 0.01) from esol_permolecule.csv; TabPFN from baselines/tabpfn__esol__N*.csv (draw 1, seed "
                "mean) with the recall flags of esol_permolecule.csv",
                "ESOL: RMSE and root mean square of e / sigma (vertical line at 1: error equal to the stated sigma) on "
                "test molecules whose value was reproduced zero-shot by >= 1 panel model "
                f"(filled) vs never reproduced (open), at N = {nmin} (faint) and N = {nmax}; tick = all {N_TEST} test "
                "molecules (includes the labels not scorable at 3 significant figures, which are in neither subset), "
                f"for every LLM and for TabPFN (seed mean), the no-memory control. {cnt} are the same for every model, "
                "because the flag is pooled (recalled by at least one panel model).")
    if key == "fig6":
        nm, left = FIG_META.get("fig6_models", (None, []))
        sub = (f"{nm}-model sub-experiment" if nm else "sub-experiment") + (
            f"; {', '.join(left)} not in the ladder" if left else "")
        return ("fig6a_ladder_smallN / fig6b_ladder_largeN",
                "ladder_cells.csv (R = 3 per effort level; `low` from the main run); rho from selective_prediction.csv "
                "(draw 1, reasoning = effort; tie-rounded, as in the tables)",
                f"Reasoning ladder ({sub}): RMSE, rho(sigma,|err|) and "
                "PICP68 vs the median number of emitted reasoning tokens, one line per model through its effort levels none -> high (marker = level); 0 tokens plotted left of "
                "the axis break (models jittered there). Gemini `high` cells rest on 126-150 parsed molecules "
                "(truncations).")
    if key == "figA1":
        return ("figA1_risk_coverage",
                f"icl/<ds>_permolecule.csv (N = {RC_N}, reasoning low, draw 1); baselines/<method>__<ds>__N{RC_N}[__s<seed>].csv "
                "(draw 1, mean over seeds); selective_prediction.risk_coverage (tie rule of the RMSE50 column)",
                f"Error of the most certain molecules as a function of how many are kept. For each dataset at N = {RC_N} "
                "in-context molecules (context draw 1), the root-mean-square error of the fraction q of test molecules "
                "with the smallest stated sigma, for q = 0.1 to 1.0; equal sigmas at the cut are broken at random "
                "(square root of the expected MSE, the rule of RMSE50). Shown are Claude Sonnet 5.5, Claude Haiku 5.5, "
                "GPT-6 Astra, the pointwise median of the seven language models (thick grey), TabPFN and GP (Tanimoto) "
                "(mean over seeds). Dotted references for Sonnet 5.5 and TabPFN in their colours: the oracle that keeps "
                "the molecules with the smallest absolute error, and random selection, whose expected error equals the "
                "RMSE at every q (flat line). A useful sigma bends the curve from the random line towards the oracle; "
                "the value at q = 1 is the RMSE of the cell." + (
                    f" ESOL axis cut at {FIG_META['figA1_cut'][0]:g}: {FIG_META['figA1_cut'][1]} lies at "
                    f"{FIG_META['figA1_cut'][2]:.2f}-{FIG_META['figA1_cut'][3]:.2f} for every q and is marked by arrows at the top "
                    "edge." if "figA1_cut" in FIG_META else ""))
    if key == "figA2":
        return ("figA2_draws",
                f"cells.csv (reasoning low, draws 1-3 at N = {'/'.join(map(str, DRAW_SIZES))}); baselines/summary.csv "
                "(TabPFN, mean over seeds within a draw)",
                "Learning curves of the three context draws. Test RMSE at N = %s for the seven language models and "
                "TabPFN, one line per context draw: draw 1 (the nested chain of all other figures) bold, draws 2 and 3 "
                "thin; models slightly offset in N. The spread between the thin lines is the context-draw noise against "
                "which the steps of a single learning curve must be read." % "/".join(map(str, DRAW_SIZES)))
    if key == "figA3":
        bs = FIG_META.get("figA3_boot_sizes", {})
        top = ", ".join(f"{DS_NAMES.get(d, d).split(' (')[0]} {max(bs[d])}" for d in DATASET_KEYS if d in bs) if bs else "per dataset"
        return ("figA3_rho_heatmap",
                "selective_prediction.csv (rho, draw 1; baselines mean over seeds); bootstrap_cells.csv (stat rho, "
                "baseline tabpfn, draw 1; seed-0 baseline files)",
                "Rank correlation between stated sigma and absolute error for every model and context size. Spearman "
                "rho(sigma, |err|) at context draw 1 for the seven language models (effort low) and the four display "
                "baselines (mean over seeds), value printed in each cell; sigma rounded to 10 decimals before ranking, "
                "as in the tables. A dot marks a language-model cell where the 95 % paired bootstrap interval of "
                "rho(LLM) - rho(TabPFN) excludes zero (filled: the language model ranks better, open: worse; 1000 "
                "resamples of the 150 test molecules, TabPFN seed 0). The bootstrap exists only at N = 30, 60, 120 and "
                f"at the largest N ({top}), so cells at N = 250 and 500 carry no mark whatever their value; baseline rows "
                "carry none.")
    if key == "figA4":
        return ("figA4_cost",
                "cells.csv (cost_usd, RMSE; reasoning low, draw 1); baselines/summary.csv (wall_s, RMSE; draw 1, mean "
                "over seeds)",
                f"Cost and error of each model at the smallest and the largest context. Top: OpenRouter cost in USD of "
                f"one cell (150 test molecules x R = 3 calls, effort low, context draw 1; log scale) against the test "
                f"RMSE, open marker N = {JOINT_N}, filled marker largest N, joined per model. Bottom: the four display "
                "baselines, CPU wall time per cell in seconds as logged by the baseline run (summary.csv wall_s, mean over "
                "seeds; log scale) "
                "against RMSE; seconds of local CPU and dollars of API calls are not on one axis.")
    if key == "figA5":
        e = FIG_META.get("figA5_example", {})
        bmin = FIG_META.get("figA5_base_min", {})
        ex = (f"; in this cell {e['distinct']} distinct values for {e['n']} molecules, the most frequent value "
              f"({e['largest_at']:.3g}) shared by {e['largest']} molecules, {e['tied']} molecules share their sigma with "
              f"at least one other, and the 50 % cut falls inside a group of {e['nt']} equal values") if e else ""
        bt = (", ".join(f"{BASE_NAMES.get(b, b)} >= {v}" for b, v in bmin.items())) if bmin else ""
        return ("figA5_sigma_values",
                "selective_prediction.csv (n_sigma_distinct, sigma rounded to 10 decimals; reasoning low, draw 1); "
                "icl/lipophilicity_permolecule.csv (uc1)",
                "The stated standard deviations take few distinct values. (a) Number of distinct sigma values among "
                "the about 150 test molecules of a cell (sigma = mean of the R = 3 stated values, rounded to 10 "
                f"decimals), per language model against N, context draw 1. For comparison the display baselines give "
                f"almost one value per molecule ({bt} of 150). (b) The stated sigmas of one cell, Claude Sonnet 5.5 on "
                f"lipophilicity at N = 30: number of molecules per distinct value{ex}. Ties of this size are why RMSE50 "
                "needs a tie rule.")
    raise KeyError(key)


META_FILE = "figs_meta.json"


def write_readme(out, done, fig_notes, common_notes):
    """update only the entries of the figures regenerated in this run (figs_meta.json keeps the others)."""
    ts = f"{_dt.datetime.now():%Y-%m-%d %H:%M}"
    mode = "draw-mean (--draw-mean)" if DRAW_MEAN else "draw-1 lines + min-max bars"
    p = os.path.join(out, META_FILE)
    meta = {}
    if os.path.exists(p):
        try:
            with open(p, encoding="utf8") as f:
                meta = json.load(f)
        except Exception as exc:
            print(f"WARNING: could not read {p} ({exc}); README rebuilt from this run only")
    figs = {k: v for k, v in meta.get("figures", {}).items() if k in FIGS}
    for k in done:
        files, src, cap = readme_entry(k)
        figs[k] = dict(files=files, data=src, caption=cap, generated=ts, draw_mode=mode, notes=fig_notes.get(k, []))
    meta = dict(figures=figs, common=dict(generated=ts, draw_mode=mode, notes=common_notes))
    with open(p, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    lines = ["# Paper figures", "",
             "Generated by `python src/make_figs.py`; re-run after the tables in `results/` are regenerated "
             "(`--only` updates just the named entries; per-entry state in `figs_meta.json`). PDF (vector) + PNG "
             "(200 dpi). Tables are relative to `results/`.", "",
             "| figure | generated | data | caption |", "|---|---|---|---|"]
    for k in FIGS:
        if k in figs:
            e = figs[k]
            lines.append(f"| `{e['files']}` | {e['generated']} ({e['draw_mode']}) | {e['data']} | {e['caption']} |")
    modes = {e["draw_mode"] for e in figs.values()}
    lines += ["", "Common rules: LLM cells with coverage < %.2f are left out; context draws 2-3 exist only at N = "
              "30 / 60 / 120. " % MIN_COVERAGE + draw_txt() +
              " Baseline display subset everywhere: %s (fig4: TabPFN only; fig3 adds the other baselines in light "
              "grey); all eight baselines are in the tables. Colours / markers per model are fixed across all figures "
              "(`MODEL_COLORS`, `MODEL_MARKERS` in the script)." % _bases_txt(DISPLAY_BASES), ""]
    if len(modes) > 1:
        lines += ["**Warning:** the entries above were generated with different draw rules: " + ", ".join(sorted(modes)), ""]
    lines += [f"## Data notes and checks (shared loaders: run {ts}; per figure: its own run)", ""]
    lines += [f"- {n}" for n in common_notes] or ["- none"]
    for k in FIGS:
        if k in figs and figs[k]["notes"]:
            lines += [f"- {figs[k]['files']} ({figs[k]['generated']}): {n}" for n in figs[k]["notes"]]
    lines.append("")
    with open(os.path.join(out, "README.md"), "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    print("->", os.path.join(out, "README.md"), "and", META_FILE)


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6,
        "figA1": figA1, "figA2": figA2, "figA3": figA3, "figA4": figA4, "figA5": figA5}


def main():
    global DRAW_MEAN
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated subset of " + ",".join(FIGS))
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    ap.add_argument("--draw-mean", action="store_true",
                    help="old draw rule: mean over draws at N <= 120 (bar = SD), draw 1 above (comparison only)")
    a = ap.parse_args()
    DRAW_MEAN = a.draw_mean
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    todo = [s.strip() for s in a.only.split(",") if s.strip()] or list(FIGS)
    for t in todo:
        if t not in FIGS:
            sys.exit(f"unknown figure {t!r}; choose from {list(FIGS)}")
    os.makedirs(out, exist_ok=True)
    print("draw rule:", "draw mean +/- SD (--draw-mean)" if DRAW_MEAN else "draw-1 lines, min-max bars over draws")
    # shared loaders and their checks first, so their notes are not attributed to one figure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        llm_cells(), base_cells(), selective()
    common = list(NOTES)
    fig_notes = {}
    for t in todo:
        n0 = len(NOTES)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                FIGS[t](out)
        except Exception as exc:          # one broken figure must not stop the others
            import traceback
            traceback.print_exc()
            note(f"{t} failed: {exc!r}")
        fig_notes[t] = NOTES[n0:]
    write_readme(out, todo, fig_notes, common)


if __name__ == "__main__":
    main()
