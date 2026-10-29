"""
Where does the error sit, and does the stated sigma find it?  (question of the project lead, 2026-10-21:
the LLMs lose more against TabPFN in selective prediction than in RMSE although their rank correlation
is similar -- is their error distributed more evenly?)

Decomposition of the selective-prediction gain at the half cut:
    gain50 = 1 - RMSE50 / RMSE
    oracle gain  = 1 - RMSE_oracle50 / RMSE      (keep the 75 molecules with the smallest |e|: the error
                                                   CONCENTRATION; a perfect ranking cannot do better)
    efficiency   = gain50 / oracle gain          (how much of the achievable gain the stated sigma realises)
plus the Gini coefficient of the squared errors (0 = all molecules contribute equally, 1 = one molecule
carries everything) and the share of the total squared error that sits in the 10 % least certain
molecules (sigma) against the same share for the 10 % largest errors (oracle).

Outputs (figs/paper):
  fig_error_concentration.{pdf,png}  2 rows x 4 datasets vs N: oracle gain (top), efficiency (bottom);
                                      lines = context draw 1, bars = min-max over the three draws at N <= 120
  fig_error_capture.{pdf,png}         cumulative share of the squared error removed when discarding molecules
                                      in order of decreasing sigma (solid) or of decreasing |e| (oracle, dotted)
                                      at N = 30 and the largest N, Sonnet 5.5 / GPT-6 Astra / TabPFN / GP
  results/icl/error_concentration.csv  the per-cell numbers
"""
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
from make_figs import (DATASET_KEYS, PANEL, MODEL_NAMES, MODEL_COLORS, DISPLAY_BASES, DS_NAMES, CONTAMINATED,
                       BASE_NAMES, aggregate, plot_curve, series_list, size_axis, series_legend, save, draw_rule,
                       style_of, FS, FW, _llm_err_sig, _base_files, llm_cells, base_cells)
from selective_prediction import selective_rmse          # noqa: E402

OUT = os.path.join(mf.ROOT, "figs", "paper")
CSV = os.path.join(mf.ROOT, "results", "icl", "error_concentration.csv")
CAPTURE_LLMS = ["sonnet55", "gpt6astra"]
CAPTURE_BASES = ["tabpfn", "gp_tanimoto"]


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def capture_curve(e, order_key):
    """share of the total squared error removed after discarding the first f molecules of `order_key`
    (descending); returns (f grid, share) with f = k/n for k = 0..n."""
    se = e ** 2
    idx = np.argsort(-order_key, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(se[idx])]) / se.sum()
    return np.arange(len(e) + 1) / len(e), cum


def cell_stats(e, sig):
    se = e ** 2
    k = len(e) // 2
    rmse = float(np.sqrt(se.mean()))
    oracle50 = float(np.sqrt(np.sort(se)[:k].mean()))
    r50 = selective_rmse(e, sig, 0.5)["mean"]
    og = 1 - oracle50 / rmse
    g = 1 - r50 / rmse
    k10 = max(1, int(round(0.1 * len(e))))
    # share of squared error in the 10 % least certain molecules (ties at the cut: expected value)
    order = np.argsort(-np.round(sig, 10), kind="stable")
    cap10 = float(se[order[:k10]].sum() / se.sum())
    orac10 = float(np.sort(se)[::-1][:k10].sum() / se.sum())
    return dict(RMSE=rmse, RMSE50=r50, gain50=g, oracle_gain=og, efficiency=g / og if og > 0 else np.nan,
                gini_e2=gini(se), capture10=cap10, oracle10=orac10, n_sigma_distinct=len(np.unique(np.round(sig, 10))))


def compute():
    rows = []
    C = llm_cells()
    for _, r in C[["dataset", "model", "size", "draw"]].drop_duplicates().iterrows():
        es = _llm_err_sig(r.dataset, r.model, int(r["size"]), int(r.draw))
        if es is None:
            continue
        rows.append(dict(dataset=r.dataset, model=r.model, size=int(r["size"]), draw=int(r.draw), **cell_stats(*es)))
    B = base_cells()
    for _, r in B[B.model.isin(DISPLAY_BASES)][["dataset", "model", "size", "draw"]].drop_duplicates().iterrows():
        vals = []
        for f in _base_files(r.model, r.dataset, int(r["size"]), int(r.draw)):
            D = pd.read_csv(f)
            vals.append(cell_stats((D.prediction - D.truth).values, D.uncertainty.values))
        if vals:
            rows.append(dict(dataset=r.dataset, model=r.model, size=int(r["size"]), draw=int(r.draw),
                             **pd.DataFrame(vals).mean().to_dict()))
    R = pd.DataFrame(rows)
    R.to_csv(CSV, index=False)
    return R


def fig_concentration(R):
    R = draw_rule(R)
    series = series_list(set(R.model) & set(MODEL_NAMES), [b for b in DISPLAY_BASES if b in set(R.model)])
    fig, axs = plt.subplots(2, 4, figsize=(FW, 4.3), sharex="col")
    for i, (col, lab) in enumerate([("oracle_gain", "oracle gain (error concentration)"),
                                    ("efficiency", "efficiency = gain$_{50}$ / oracle gain")]):
        A = aggregate(R.rename(columns={col: "v"}), "v")
        for j, ds in enumerate(DATASET_KEYS):
            ax = axs[i, j]
            a = A[A.dataset == ds]
            if a.empty:
                ax.set_visible(False)
                continue
            for k, (m, kind) in enumerate(series):
                plot_curve(ax, A, ds, m, kind, k, len(series))
            size_axis(ax, sorted(a["size"].unique()), narrow=True)
            if i == 0:
                ax.set_title(DS_NAMES.get(ds, ds).replace(" (", "\n("), color="#777777" if ds in CONTAMINATED else "black")
            if j == 0:
                ax.set_ylabel(lab)
            if i == 1:
                ax.set_xlabel("context size N")
    series_legend(fig, series, -0.005)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=0.6, h_pad=0.5)
    save(fig, OUT, "fig_error_concentration")


def fig_capture(R):
    fig, axs = plt.subplots(2, 4, figsize=(FW, 4.6), sharex=True, sharey=True)
    sizes_per_ds = R.groupby("dataset")["size"].agg(["min", "max"])
    for j, ds in enumerate(DATASET_KEYS):
        for i, which in enumerate(["min", "max"]):
            ax = axs[i, j]
            n = int(sizes_per_ds.loc[ds, which])
            ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8, zorder=1)              # random order
            for m in CAPTURE_LLMS:
                es = _llm_err_sig(ds, m, n, 1)
                if es is None:
                    continue
                e, sg = es
                f, c = capture_curve(e, np.round(sg, 10))
                ax.plot(f, c, color=MODEL_COLORS[m], lw=1.4, zorder=3)
                f, c = capture_curve(e, np.abs(e))
                ax.plot(f, c, color=MODEL_COLORS[m], lw=0.9, ls=":", zorder=2)
            for b in CAPTURE_BASES:
                fs = _base_files(b, ds, n, 1)
                if not fs:
                    continue
                D = pd.read_csv(fs[0])                                              # seed 0
                e, sg = (D.prediction - D.truth).values, D.uncertainty.values
                st = style_of(b, "base")
                f, c = capture_curve(e, sg)
                ax.plot(f, c, color=st["color"], ls=st["ls"], lw=1.4, zorder=3)
                f, c = capture_curve(e, np.abs(e))
                ax.plot(f, c, color=st["color"], ls=":", lw=0.9, zorder=2)
            ax.axvline(0.5, color="#dddddd", lw=0.6, zorder=0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title((DS_NAMES.get(ds, ds).replace(" (", "\n(") + "\n" if i == 0 else "") + f"N = {n}",
                         color="#777777" if ds in CONTAMINATED else "black", fontsize=8.5 * FS)
            if j == 0:
                ax.set_ylabel("share of squared error removed")
            if i == 1:
                ax.set_xlabel("fraction discarded")
    h = [Line2D([], [], color=MODEL_COLORS[m], lw=1.4, label=MODEL_NAMES[m]) for m in CAPTURE_LLMS]
    h += [Line2D([], [], color=style_of(b, "base")["color"], ls=style_of(b, "base")["ls"], lw=1.4, label=BASE_NAMES[b])
          for b in CAPTURE_BASES]
    h += [Line2D([], [], color="grey", lw=1.4, label="discard by stated sigma (solid)"),
          Line2D([], [], color="grey", lw=0.9, ls=":", label="discard by |error| (oracle, dotted)"),
          Line2D([], [], color="#bbbbbb", lw=0.8, label="random order")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.5, h_pad=0.5)
    save(fig, OUT, "fig_error_capture")


def summary(R):
    pd.set_option("display.width", 220)
    d1 = R[R.draw == 1]
    for ds in DATASET_KEYS:
        t = d1[d1.dataset == ds]
        print("==", ds)
        for n in [30, 120, t["size"].max()]:
            u = t[t["size"] == n].set_index("model")
            llm = u.loc[[m for m in PANEL if m in u.index]]
            tb = u.loc["tabpfn"]
            print(f"  N={n:4d} oracle gain LLM med {llm.oracle_gain.median():.2f} TabPFN {tb.oracle_gain:.2f} | "
                  f"Gini LLM {llm.gini_e2.median():.2f} TabPFN {tb.gini_e2:.2f} | efficiency LLM {llm.efficiency.median():.2f} "
                  f"TabPFN {tb.efficiency:.2f} | 10 % least certain carry LLM {llm.capture10.median():.2f} TabPFN {tb.capture10:.2f} "
                  f"of the error (oracle {llm.oracle10.median():.2f} / {tb.oracle10:.2f}) | distinct sigma LLM {llm.n_sigma_distinct.median():.0f} TabPFN {tb.n_sigma_distinct:.0f}")


if __name__ == "__main__":
    R = compute()
    summary(R)
    fig_concentration(R)
    fig_capture(R)
