"""
Figures of the restructured manuscript (decision of the project lead, 2026-10-14): the paper is
organised around three questions, each carried by one learning-curve figure of the same layout
(2 x 2 datasets, x = context size N, 7 LLMs + 4 classical baselines, lines through context draw 1,
bars = min-max over the three context draws at N <= 120):

  Q1  accuracy over context sizes and reasoning levels   -> fig1_rmse_vs_n (data of make_figs.fig1, paper layout)
                                                            fig2_ladder_rmse (RMSE vs emitted reasoning tokens)
  Q2  which model for active learning (measurement selection) -> fig3_rho_vs_n  (rho(sigma, |err|) vs N)
  Q3  using predictions without measuring (selective prediction) -> fig4_rmse50_vs_n (RMSE of the 50 % most
                                                            certain molecules vs N)
  contamination (Section 3.4) -> fig5_contamination (zero-shot rho, corpus presence, retention R12 vs floor)
  appendix: figA_rmseq_vs_n (the same for q = 0.2 and 0.8), plus the existing fig5_esol_recall,
            figA1_risk_coverage, fig3_joint_smallN, figA2_draws, figA4_cost from make_figs.py
            (fig4_calibration left the paper set with the calibration analysis, 2026-10-28; it is still drawn
            by make_figs.py into figs/); figA_similarity comes from src/similarity_context.py

All loaders, styles and the draw rule are imported from src/make_figs.py (nothing recomputed differently).
Output: figs/paper/*.{pdf,png}

    python src/make_figs_paper.py            # all
    python src/make_figs_paper.py --only fig3_rho_vs_n,figA_rmseq_vs_n
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import make_figs as mf                                   # noqa: E402
from make_figs import (FS, FW, DATASET_KEYS, PANEL, MODEL_NAMES, MODEL_COLORS, DISPLAY_BASES, DS_NAMES, CONTAMINATED,
                       LEVELS, LEVEL_MARKERS, aggregate, freeze_layout, bottom_block, row_major, legend_handles, plot_curve, series_list, size_axis, ds_title,
                       series_legend, save, cap_axis, llm_cells, base_cells, selective, read_csv, need,
                       P_LADDER, P_SEL, _llm_err_sig, _base_files, draw_rule, note, FIG_META)
from selective_prediction import risk_coverage          # noqa: E402

OUT = os.path.join(mf.ROOT, "figs", "paper")


# ---------------------------------------------------------------------------- generic 2 x 2 learning-curve figure
def curve_fig(A, col_label, name, ref=None, cap_esol=True, sharey=False, H=3.9):
    """A: output of aggregate() (dataset, model, size, val, lo, hi). One panel per dataset, 2 x 2 at the full
    text width (flat panels, every N labelled), y-label on the left column, one x-label and the legend
    (4 columns, LLMs first) directly below the bottom row. sharey: one y-range for all four panels."""
    llms = set(A.model) & set(MODEL_NAMES)
    series = series_list(llms, [b for b in DISPLAY_BASES if b in set(A.model)])
    fig, axs = plt.subplots(2, 2, figsize=(FW, H), layout="constrained", sharey=sharey)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.07 if sharey else 0.02, hspace=0.04)
    for ax, ds in zip(axs.flat, DATASET_KEYS):
        a = A[A.dataset == ds]
        if a.empty:
            ax.set_visible(False)
            continue
        curves = [(m, kind, plot_curve(ax, A, ds, m, kind, k, len(series))) for k, (m, kind) in enumerate(series)]
        size_axis(ax, sorted(a["size"].unique()))
        if ref is not None:
            ax.axhline(ref, color="black", lw=0.6, zorder=1)
        if cap_esol and ds in CONTAMINATED:
            FIG_META.setdefault("esol_cut", {})[name] = cap_axis(ax, curves, A, ds)
        ds_title(ax, ds)
    for ax in axs[:, 0]:
        ax.set_ylabel(col_label)
    freeze_layout(fig)
    bottom_block(fig, axs, "context size N", row_major(legend_handles(series), 4), ncol=4)
    save(fig, OUT, name)


def fig1_rmse_vs_n():
    """paper layout of make_figs.fig1 (same data and draw rule)."""
    C, B = mf.llm_cells(), mf.base_cells()
    if C is None:
        note("fig1_rmse_vs_n skipped (no cells.csv)")
        return
    frames = [C[["dataset", "model", "size", "draw", "RMSE"]]]
    if B is not None:
        frames.append(B[B.model.isin(DISPLAY_BASES)][["dataset", "model", "size", "draw", "RMSE"]])
    A = aggregate(pd.concat(frames, ignore_index=True), "RMSE")
    curve_fig(A, "RMSE", "fig1_rmse_vs_n")


def fig3_rho_vs_n():
    S = selective()
    if S is None:
        note("fig3_rho_vs_n skipped")
        return
    A = aggregate(S[S.model.isin(list(MODEL_NAMES) + DISPLAY_BASES)][["dataset", "model", "size", "draw", "rho"]], "rho")
    curve_fig(A, r"$\rho_s(\sigma, |\mathrm{err}|)$", "fig3_rho_vs_n", ref=0.0, cap_esol=False, sharey=True)


def fig4_rmse50_vs_n():
    S = selective()
    if S is None:
        note("fig4_rmse50_vs_n skipped")
        return
    A = aggregate(S[S.model.isin(list(MODEL_NAMES) + DISPLAY_BASES)][["dataset", "model", "size", "draw", "RMSE50"]], "RMSE50")
    curve_fig(A, r"RMSE$_{50}$ (50 % most certain)", "fig4_rmse50_vs_n")


# ---------------------------------------------------------------------------- RMSE of the q most certain, q = 0.2 / 0.8
def _rmse_q_rows(q):
    """(dataset, model, size, draw, RMSE_q) for every complete LLM cell and every display baseline cell
    (baselines: mean over seeds within a draw), exact tie rule of selective_prediction.risk_coverage."""
    C = llm_cells()
    rows = []
    for _, r in C[["dataset", "model", "size", "draw"]].drop_duplicates().iterrows():
        es = _llm_err_sig(r.dataset, r.model, int(r["size"]), int(r.draw))
        if es is None:
            continue
        rc = risk_coverage(es[0], es[1], fracs=(q,))
        rows.append(dict(dataset=r.dataset, model=r.model, size=int(r["size"]), draw=int(r.draw),
                         val=float(rc.RMSE_q.iloc[0])))
    B = base_cells()
    for _, r in B[B.model.isin(DISPLAY_BASES)][["dataset", "model", "size", "draw"]].drop_duplicates().iterrows():
        vals = []
        for f in _base_files(r.model, r.dataset, int(r["size"]), int(r.draw)):
            D = pd.read_csv(f)
            e = (D.prediction - D.truth).values
            vals.append(float(risk_coverage(e, D.uncertainty.values, fracs=(q,)).RMSE_q.iloc[0]))
        if vals:
            rows.append(dict(dataset=r.dataset, model=r.model, size=int(r["size"]), draw=int(r.draw),
                             val=float(np.mean(vals))))
    return pd.DataFrame(rows)


def figA_rmseq_vs_n(qs=(0.2, 0.8)):
    """rows q, columns datasets (2 x 4 at the text width, thinned N labels); legend below."""
    fig, axs = plt.subplots(len(qs), 4, figsize=(FW, 1.75 * len(qs)), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.05)
    series = None
    for i, q in enumerate(qs):
        R = draw_rule(_rmse_q_rows(q))
        A = aggregate(R.rename(columns={"val": "v"}), "v")
        if series is None:
            series = series_list(set(A.model) & set(MODEL_NAMES), [b for b in DISPLAY_BASES if b in set(A.model)])
        for j, ds in enumerate(DATASET_KEYS):
            ax = axs[i, j]
            a = A[A.dataset == ds]
            if a.empty:
                ax.set_visible(False)
                continue
            curves = [(m, kind, plot_curve(ax, A, ds, m, kind, k, len(series))) for k, (m, kind) in enumerate(series)]
            size_axis(ax, sorted(a["size"].unique()), narrow=True)
            if ds in CONTAMINATED:
                cap_axis(ax, curves, A, ds)
            if i == 0:
                ds_title(ax, ds, wrap=True)
            if j == 0:
                ax.set_ylabel(f"RMSE of the {int(q * 100)} %" + chr(10) + "most certain")
    freeze_layout(fig)
    bottom_block(fig, axs, "context size N", row_major(legend_handles(series), 4), ncol=4)
    save(fig, OUT, "figA_rmseq_vs_n")


# ---------------------------------------------------------------------------- reasoning ladder, RMSE only
def fig2_ladder_rmse():
    L = read_csv(P_LADDER, "ladder_cells")
    if L is None or not need(L, ["dataset", "model", "size", "effort", "RMSE", "median_reasoning_tokens"], "ladder_cells.csv"):
        note("fig2_ladder_rmse skipped")
        return
    pos = L.loc[L.median_reasoning_tokens > 0, "median_reasoning_tokens"]
    minpos = pos.min() if len(pos) else 100.0
    x0 = 10 ** (np.floor(np.log10(minpos)) - 0.35)
    xbreak = np.sqrt(x0 * minpos)
    xmax = L.median_reasoning_tokens.max() * 2.0
    sizes_per_ds = L.groupby("dataset")["size"].agg(["min", "max"])
    nlev = L.groupby("model").effort.nunique()
    models = [m for m in PANEL if m in set(L.model) and m in MODEL_NAMES and nlev.get(m, 0) >= 2]
    # 2 x 4 at the text width, shared token axis (labels on the bottom row only); the row's context size sits in
    # the y-label (top) and in the panel title (bottom, the largest N differs per dataset); x-label and legend
    # follow the bottom row at fixed distances (bottom_block)
    fig, axs = plt.subplots(2, 4, figsize=(FW, 3.3), sharex=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
    for j, ds in enumerate(DATASET_KEYS):
        for i, which in enumerate(["min", "max"]):
            ax = axs[i, j]
            if ds not in sizes_per_ds.index:
                ax.set_visible(False)
                continue
            n = sizes_per_ds.loc[ds, which]
            d = L[(L.dataset == ds) & (L["size"] == n)]
            for k, m in enumerate(models):
                g = d[d.model == m].copy()
                if g.empty:
                    continue
                g["lv"] = g.effort.map({lv: q for q, lv in enumerate(LEVELS)})
                g = g.sort_values("lv")
                jit = x0 * 1.13 ** (k - (len(models) - 1) / 2)
                x = np.where(g.median_reasoning_tokens > 0, g.median_reasoning_tokens, jit)
                ax.plot(x, g.RMSE, color=MODEL_COLORS[m], lw=0.9, zorder=2)
                for xi, yi, lv in zip(x, g.RMSE, g.effort):
                    ax.plot(xi, yi, LEVEL_MARKERS.get(lv, "o"), color=MODEL_COLORS[m], ms=4 * FS ** 0.5,
                            mfc=MODEL_COLORS[m] if lv != "none" else "none", mew=1.0, zorder=3)
            ax.set_xscale("log")
            ax.set_xlim(x0 / 1.8, xmax)
            ax.axvline(xbreak, color="#bbbbbb", lw=0.6, ls=(0, (2, 2)), zorder=0)
            title = mf.wrap_name(ds) if i == 0 else f"N = {int(n)}"
            ax.set_title(title, color="#777777" if ds in CONTAMINATED else "black",
                         fontsize=plt.rcParams["axes.titlesize"] if i == 0 else 8 * FS)
    axs[0, 0].set_ylabel(f"RMSE, N = {int(sizes_per_ds['min'].min())}" if sizes_per_ds["min"].nunique() == 1
                         else "RMSE, smallest N")
    axs[1, 0].set_ylabel("RMSE, largest N")
    for ax in axs[-1]:
        ticks = [x0] + [10 ** e for e in range(int(np.ceil(np.log10(minpos))), int(np.log10(xmax)) + 1)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0"] + [f"$10^{{{int(np.log10(t))}}}$" if i % 2 == 0 else "" for i, t in enumerate(ticks[1:])])
        minor = [f * 10 ** e for e in range(int(np.floor(np.log10(xbreak))), int(np.log10(xmax)) + 1)
                 for f in range(2, 10) if xbreak < f * 10 ** e < xmax]
        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(minor))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    freeze_layout(fig)
    for ax in axs[-1]:
        if ax.get_visible():
            ax.text(xbreak, 0, "//", transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=7 * FS, color="black", backgroundcolor="white", clip_on=False, zorder=5)
    h = [Line2D([], [], color=MODEL_COLORS[m], lw=1.2, label=MODEL_NAMES[m]) for m in models]
    h += [Line2D([], [], color="grey", marker=LEVEL_MARKERS[lv], ls="", ms=4 * FS ** 0.5,
                 mfc="grey" if lv != "none" else "none", label=f"effort {lv}") for lv in LEVELS]
    nc = max(len(models), len(LEVELS))
    h = h[:len(models)] + [Line2D([], [], ls="", label=" ")] * (nc - len(models)) + h[len(models):]
    bottom_block(fig, axs, "median emitted reasoning tokens per call (0 = no reasoning tokens)", row_major(h, nc),
                 ncol=nc, gap=1.5, handlelength=1.4, columnspacing=0.7, handletextpad=0.4)
    save(fig, OUT, "fig2_ladder_rmse")


# ---------------------------------------------------------------------------- contamination (Section 3.4)
P_ZS = os.path.join(mf.ROOT, "results", "contamination", "zeroshot", "cells.csv")
P_CORPUS = os.path.join(mf.ROOT, "results", "contamination", "corpus", "summary_by_subset.csv")
DS_SHORT = {"lipophilicity": "Lipo.", "bace": "BACE", "antiviral": "Antivir.", "esol": "ESOL"}


def _ds_axis(ax, keys):
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([DS_SHORT[d] for d in keys])
    for lab, d in zip(ax.get_xticklabels(), keys):
        if d in CONTAMINATED:
            lab.set_color("#777777")
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.grid(axis="x", visible=False)


def fig5_contamination():
    """three panels at the text width: (a) Spearman rho of the zero-shot predictions with the labels per model x
    dataset; (b) share of the test molecules present in the seven searched corpora per dataset and string form
    (bar = maximum over the corpora, dots = the single corpora); (c) retention R12 of the zero-shot screen per
    model x dataset with the label-only floor (filled marker = cell flagged by the screen, open = clean).
    Sources: results/contamination/zeroshot/cells.csv, results/contamination/corpus/summary_by_subset.csv
    (subset v2test150)."""
    Z = read_csv(P_ZS, "zeroshot cells")
    K = read_csv(P_CORPUS, "corpus summary")
    if Z is None or K is None:
        note("fig5_contamination skipped")
        return
    keys = [d for d in DATASET_KEYS if d in set(Z.dataset)]
    models = [m for m in PANEL if m in set(Z.model) and m in MODEL_NAMES]
    fig, axs = plt.subplots(1, 3, figsize=(FW, 2.05), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    off = np.linspace(-0.3, 0.3, len(models))
    ms = 4 * FS ** 0.5

    def dots(ax, col):
        for k, m in enumerate(models):
            for j, ds in enumerate(keys):
                r = Z[(Z.dataset == ds) & (Z.model == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                c = MODEL_COLORS[m]
                ax.plot(j + off[k], r[col], ls="", marker=mf.MODEL_MARKERS[m], ms=ms, mec=c, mew=0.8,
                        mfc=c if r.verdict != "clean" else "white", zorder=3)

    # (a) zero-shot rank correlation with the labels
    ax = axs[0]
    dots(ax, "spearman")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(r"Spearman $\rho$ (zero-shot vs label)")
    _ds_axis(ax, keys)
    # (b) corpus presence per string form
    ax = axs[1]
    k = K[(K.subset == "v2test150") & K.dataset.isin(keys)]
    forms = [("pct_present_smiles", "SMILES", "#4d4d4d"), ("pct_present_inchikey", "InChIKey", "#b3b3b3")]
    w = 0.34
    for i, (col, lab, c) in enumerate(forms):
        dx = (i - 0.5) * w
        for j, ds in enumerate(keys):
            v = k[k.dataset == ds][col].astype(float)
            mx = float(v.max())
            ax.bar(j + dx, mx, width=w, color=c, edgecolor=c, lw=0.5, zorder=2, label=lab if j == 0 else None)
            ax.plot(np.full(len(v), j + dx), v.values, ls="", marker="o", ms=2.2, mfc="white", mec="black", mew=0.5,
                    zorder=3)
            if mx == 0:
                ax.text(j + dx, 0.6, "0", ha="center", va="bottom", fontsize=plt.rcParams["xtick.labelsize"])
    ax.set_ylim(0, 45)
    ax.set_ylabel("test molecules in a corpus (%)")
    _ds_axis(ax, keys)
    ax.legend(loc="upper left", frameon=False, handlelength=1.0, handletextpad=0.4, borderaxespad=0.2)
    # (c) retention R12 against the floor from the labels alone
    ax = axs[2]
    for j, ds in enumerate(keys):
        f = float(Z[Z.dataset == ds].floor_R12.iloc[0])
        ax.plot([j - 0.42, j + 0.42], [f, f], color="black", lw=1.0, zorder=2)
    dots(ax, "R12")
    ax.set_ylim(0, 1.05 * max(80.0, float(Z.R12.max())))
    ax.set_ylabel(r"retention $R_{12}$ (%)")
    _ds_axis(ax, keys)
    for ax, t in zip(axs, "abc"):
        ax.set_title(f"({t})", loc="left", fontsize=plt.rcParams["axes.titlesize"])
    FIG_META["fig5_contamination"] = dict(
        rho_clean=(float(Z[Z.dataset != "esol"].spearman.min()), float(Z[Z.dataset != "esol"].spearman.max())))
    freeze_layout(fig)
    h = [Line2D([], [], ls="", marker=mf.MODEL_MARKERS[m], ms=ms, mec=MODEL_COLORS[m], mfc=MODEL_COLORS[m], mew=0.8,
                label=MODEL_NAMES[m]) for m in models]
    h += [Line2D([], [], ls="", marker="o", ms=ms, mec="grey", mfc="white", mew=0.8, label="clean cell"),
          Line2D([], [], ls="", marker="o", ms=ms, mec="grey", mfc="grey", mew=0.8, label="flagged cell"),
          Line2D([], [], color="black", lw=1.0, label=r"floor of $R_{12}$")]
    bottom_block(fig, axs, None, row_major(h, 5), ncol=5, gap=2.0, columnspacing=0.9, handlelength=1.2)
    save(fig, OUT, "fig5_contamination")


# ---------------------------------------------------------------------------- appendix figures reused from make_figs
def reuse_appendix():
    for fn in (mf.fig3, mf.fig5, mf.figA1, mf.figA2, mf.figA4):
        fn(OUT)


FIGS = {"fig1_rmse_vs_n": fig1_rmse_vs_n, "fig2_ladder_rmse": fig2_ladder_rmse, "fig3_rho_vs_n": fig3_rho_vs_n,
        "fig4_rmse50_vs_n": fig4_rmse50_vs_n, "fig5_contamination": fig5_contamination,
        "figA_rmseq_vs_n": figA_rmseq_vs_n, "appendix_reuse": reuse_appendix}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    which = [s.strip() for s in a.only.split(",") if s.strip()] or list(FIGS)
    os.makedirs(OUT, exist_ok=True)
    for k in which:
        FIGS[k]()
    if mf.NOTES:
        print("\nNOTES:")
        for n in mf.NOTES:
            print(" -", n)


if __name__ == "__main__":
    main()
