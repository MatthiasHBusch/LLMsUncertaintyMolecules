"""
Similarity of a test molecule to its context against the error and the stated sigma (review round of
2026-10-28, Appendix E "Repeat spread and similarity to the context", Fig. figA_similarity).

For every test molecule of a cell: maxsim = the largest Tanimoto similarity of its Morgan fingerprint
(RDKit, radius 2, 2,048 bits, rdFingerprintGenerator; bit-identical to the fingerprints the RF / GP / kNN /
MLP baselines use, results/baselines/cache/*morgan_r2_2048*) to any of the N context molecules of the cell's
context draw, i.e. the first N of train_order_ids of data/splits/<ds>_v2[_draw<k>].json. SMILES by mol_id
from data/<ds>.csv. maxsim depends on (dataset, draw, N, molecule) only, not on the model.

Per cell (dataset, model, size, draw):
  n               test molecules of the cell
  median_maxsim   median of maxsim over the cell's molecules
  rho_sim_abserr  Spearman rho(maxsim, |e|), e = prediction - truth
  rho_sim_sigma   Spearman rho(maxsim, sigma)
                  in both, each input (maxsim, |e|, sigma) is rounded to 10 decimals before ranking
                  (selective_prediction.TIE_DECIMALS), average ranks for ties; NaN if an input is constant
Cells:
  LLMs       every complete `low` cell of results/icl/cells.csv (make_figs.llm_cells: coverage >= 0.95;
             draw 1 at every N, draws 2-3 at N <= 120), per molecule from results/icl/<ds>_permolecule.csv
             (pred = mean over the parsed repeats, sigma = uc1 = their mean stated sigma, unfloored; the floor
             of 0.01 does not change a rank correlation except by merging values below it, and no stated
             sigma is below it)
  baselines  make_figs.DISPLAY_BASES (TabPFN, GP, RF descriptors, Chemprop ensemble), every per-cell file of
             the cell (make_figs._base_files, file rule selective_prediction.BASE_RE); rho computed per seed
             file and averaged over the seeds of the draw (rule of the figures and tables)
Outputs:
  results/icl/context_similarity.csv            dataset, model, size, draw, n, median_maxsim, rho_sim_abserr,
                                                rho_sim_sigma (LLM and baseline cells)
  results/icl/context_similarity_permolecule.csv  dataset, draw, size, mol_id, maxsim (one row per dataset x
                                                draw x size x test molecule)
  figs/paper/figA_similarity.{pdf,png}          2 rows x 4 datasets vs N: rho(maxsim, |e|) (top),
                                                rho(maxsim, sigma) (bottom); 7 LLMs + the 4 display baselines;
                                                lines = context draw 1, bars = min-max over the three draws at
                                                N <= 120 (make_figs.aggregate), zero line; style of
                                                make_figs_paper (FW, FS, Times)
No API call; read-only on results/ except the two CSVs above.

    python src/similarity_context.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figs as mf                                   # noqa: E402
from make_figs import (FW, DATASET_KEYS, MODEL_NAMES, DISPLAY_BASES, CONTAMINATED, aggregate, plot_curve,  # noqa: E402
                       series_list, size_axis, ds_title, freeze_layout, bottom_block, row_major, legend_handles,
                       save, draw_rule, llm_cells, base_cells, _permolecule, _base_files, N_TEST, MIN_COVERAGE)
from selective_prediction import TIE_DECIMALS              # noqa: E402

ROOT = mf.ROOT
OUT_FIG = os.path.join(ROOT, "figs", "paper")
CSV = os.path.join(ROOT, "results", "icl", "context_similarity.csv")
CSV_PM = os.path.join(ROOT, "results", "icl", "context_similarity_permolecule.csv")
FP_RADIUS, FP_NBITS = 2, 2048


def fingerprints(ds):
    """mol_id -> Morgan bit vector (bool numpy array) of every molecule of data/<ds>.csv."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(os.path.join(ROOT, "data", f"{ds}.csv"), dtype={"mol_id": str})
    assert df.mol_id.is_unique, ds
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)
    fp = {}
    for mid, smi in zip(df.mol_id, df.smiles.astype(str)):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, (ds, mid)
        fp[mid] = gen.GetFingerprintAsNumPy(mol).astype(bool)
    return fp


def split(ds, draw):
    f = f"{ds}_v2.json" if draw == 1 else f"{ds}_v2_draw{draw}.json"
    p = os.path.join(ROOT, "data", "splits", f)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def maxsim_table(ds, draw, sizes, fp):
    """DataFrame (size, mol_id, maxsim): largest Tanimoto similarity of every test molecule to the first N
    of train_order_ids, for N in sizes (prefix maximum over one test x pool similarity matrix)."""
    sp = split(ds, draw)
    test = [str(m) for m in sp["test_ids"]]
    order = [str(m) for m in sp["train_order_ids"]]
    nmax = max(sizes)
    assert nmax <= len(order), (ds, draw, nmax, len(order))
    T = np.array([fp[m] for m in test], dtype=np.float64)
    P = np.array([fp[m] for m in order[:nmax]], dtype=np.float64)
    inter = T @ P.T
    union = T.sum(1)[:, None] + P.sum(1)[None, :] - inter
    S = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    cm = np.maximum.accumulate(S, axis=1)                    # cm[:, k] = max over the first k + 1 context molecules
    rows = []
    for n in sorted(sizes):
        for mid, v in zip(test, cm[:, n - 1]):
            rows.append(dict(dataset=ds, draw=draw, size=int(n), mol_id=mid, maxsim=float(v)))
    return pd.DataFrame(rows)


def spearman(x, y):
    """Spearman rho with average ranks, both inputs rounded to TIE_DECIMALS; NaN if either is constant."""
    x, y = np.round(np.asarray(x, float), TIE_DECIMALS), np.round(np.asarray(y, float), TIE_DECIMALS)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def cell_row(ds, model, n, draw, mol_id, e, sig, MS):
    ms = MS.reindex(pd.Index(mol_id).astype(str)).values
    assert np.isfinite(ms).all(), (ds, model, n, draw, "test molecule without maxsim")
    return dict(n=len(e), median_maxsim=float(np.median(ms)), rho_sim_abserr=spearman(ms, np.abs(e)),
                rho_sim_sigma=spearman(ms, sig))


def compute():
    C = llm_cells()
    B = base_cells()
    B = B[B.model.isin(DISPLAY_BASES)]
    keys = pd.concat([C[["dataset", "size", "draw"]], B[["dataset", "size", "draw"]]]).drop_duplicates()
    pm_frames, rows = [], []
    for ds in DATASET_KEYS:
        fp = fingerprints(ds)
        for draw in sorted(keys[keys.dataset == ds].draw.unique()):
            sizes = sorted(int(s) for s in keys[(keys.dataset == ds) & (keys.draw == draw)]["size"].unique())
            M = maxsim_table(ds, int(draw), sizes, fp)
            pm_frames.append(M)
            for n in sizes:
                MS = M[M["size"] == n].set_index("mol_id").maxsim
                # LLM cells
                for _, r in C[(C.dataset == ds) & (C.draw == draw) & (C["size"] == n)].iterrows():
                    P = _permolecule(ds)
                    g = P[(P.model == r.model) & (P["size"] == n) & (P.draw == draw)]
                    g = g[np.isfinite(g.pred) & np.isfinite(g.uc1)]
                    if len(g) < MIN_COVERAGE * N_TEST:
                        continue
                    rows.append(dict(dataset=ds, model=r.model, size=n, draw=int(draw),
                                     **cell_row(ds, r.model, n, draw, g.mol_id.astype(str).values,
                                                (g.pred - g.truth).values, g.uc1.values, MS)))
                # display baselines: per seed file, then the mean over the seeds of the draw
                for b in DISPLAY_BASES:
                    if B[(B.dataset == ds) & (B.model == b) & (B.draw == draw) & (B["size"] == n)].empty:
                        continue
                    vals = []
                    for f in _base_files(b, ds, n, int(draw)):
                        D = pd.read_csv(f, dtype={"mol_id": str})
                        vals.append(cell_row(ds, b, n, draw, D.mol_id.values, (D.prediction - D.truth).values,
                                             D.uncertainty.values, MS))
                    if vals:
                        v = pd.DataFrame(vals).mean().to_dict()
                        v["n"] = int(round(v["n"]))
                        rows.append(dict(dataset=ds, model=b, size=n, draw=int(draw), **v))
    R = pd.DataFrame(rows)[["dataset", "model", "size", "draw", "n", "median_maxsim", "rho_sim_abserr", "rho_sim_sigma"]]
    R = R.sort_values(["dataset", "model", "size", "draw"]).reset_index(drop=True)
    PM = pd.concat(pm_frames, ignore_index=True)[["dataset", "draw", "size", "mol_id", "maxsim"]]
    R.to_csv(CSV, index=False)
    PM.to_csv(CSV_PM, index=False)
    print(f"-> {os.path.relpath(CSV, ROOT)} ({len(R)} cells), {os.path.relpath(CSV_PM, ROOT)} ({len(PM)} rows)")
    return R, PM


def figure(R):
    R = draw_rule(R)
    rowspec = [("rho_sim_abserr", r"$\rho_s(\mathrm{maxsim}, |\mathrm{err}|)$"),
               ("rho_sim_sigma", r"$\rho_s(\mathrm{maxsim}, \sigma)$")]
    fig, axs = plt.subplots(2, 4, figsize=(FW, 3.5), layout="constrained", sharey="row")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.05)
    series = series_list(set(R.model) & set(MODEL_NAMES), [b for b in DISPLAY_BASES if b in set(R.model)])
    for i, (col, lab) in enumerate(rowspec):
        A = aggregate(R.rename(columns={col: "v"}), "v")
        for j, ds in enumerate(DATASET_KEYS):
            ax = axs[i, j]
            a = A[A.dataset == ds]
            if a.empty:
                ax.set_visible(False)
                continue
            for k, (m, kind) in enumerate(series):
                plot_curve(ax, A, ds, m, kind, k, len(series))
            ax.axhline(0.0, color="black", lw=0.6, zorder=1)
            size_axis(ax, sorted(a["size"].unique()), narrow=True)
            if i == 0:
                ds_title(ax, ds, wrap=True)
            if j == 0:
                ax.set_ylabel(lab)
    freeze_layout(fig)
    bottom_block(fig, axs, "context size N", row_major(legend_handles(series), 4), ncol=4)
    save(fig, OUT_FIG, "figA_similarity")


def summary(R, PM):
    d1 = R[R.draw == 1]
    top = PM[PM.draw == 1].groupby("dataset")["size"].max()
    print("\nmedian maxsim per dataset (draw 1, all 150 test molecules):")
    for ds in DATASET_KEYS:
        x = PM[(PM.dataset == ds) & (PM.draw == 1)]
        print(f"  {ds:14s} N=30 {x[x['size'] == 30].maxsim.median():.3f}   N={top[ds]} {x[x['size'] == top[ds]].maxsim.median():.3f}")
    for col in ("rho_sim_abserr", "rho_sim_sigma"):
        print(f"\n{col}, draw 1 (LLM range over the 7 models; baselines)")
        for ds in DATASET_KEYS:
            for n in (30, int(top[ds])):
                t = d1[(d1.dataset == ds) & (d1["size"] == n)].set_index("model")[col]
                llm = t.reindex([m for m in MODEL_NAMES if m in t.index])
                print(f"  {ds:14s} N={n:4d} LLM {llm.min():+.3f} .. {llm.max():+.3f} (median {llm.median():+.3f}) | " +
                      " ".join(f"{b} {t.get(b, np.nan):+.3f}" for b in DISPLAY_BASES))


if __name__ == "__main__":
    R, PM = compute()
    summary(R, PM)
    figure(R)
    if mf.NOTES:
        print("\nNOTES:")
        for n in mf.NOTES:
            print(" -", n)
