"""
Paired bootstrap CIs per cell for RMSE, rho, RMSE50 and gain50, single signals and LLM - baseline
differences (REPORT 4.4 / PAPER_OUTLINE H.1, H.3). No API calls; reads finished results only.

    python src/bootstrap_cells.py          -> results/icl/bootstrap_cells.csv

Cells: reasoning `low`; N = 30 / 60 / 120 for context draws 1, 2, 3, and the top rung (draw 1);
4 datasets. Signals: the 7 panel LLMs (sigma = UC1, the self-reported 1-sigma averaged over the
R = 3 repeats; prediction = repeat mean) and 4 baselines (tabpfn, gp_tanimoto, rf_desc,
chemprop_ens; seed 0, the per-cell files results/baselines/<method>__<ds>__N<size>[__d<draw>].csv).

Statistics per signal (all on |err| = |pred - truth| and sigma rounded to 10 decimals, as in
src/selective_prediction.py):
  RMSE    sqrt(mean err^2)
  rho     Spearman(sigma, |err|), average ranks for ties (computed as Pearson of the ranks)
  RMSE50  RMSE of the floor(n/2) smallest-sigma molecules with random tie-breaking at the cut, exact:
          sqrt((SSE_below + (r / m) * SSE_tied) / k), the square root of the expected MSE (same rule
          as selective_prediction.selective_rmse, since 2026-10-14; the point estimate calls
          rmse_most_certain itself, inside the bootstrap the same closed form is evaluated per
          resample as a grouped sum, vectorised over the resamples)
  gain50  1 - RMSE50 / RMSE

Bootstrap: B = 1000 resamples of the 150 test molecules with replacement, numpy default_rng(0),
one index matrix per dataset shared by every cell, model and draw of that dataset (so all
differences are paired on identical resampled molecules); percentile 95 % CI (2.5 / 97.5).
Molecules are indexed in the order of the split's test_ids (data/splits/<ds>_v2.json, identical for
all draws), as in src/analyze_calibration.py::bootstrap and src/report_extras.py::paired_top, so
the same seed gives the same resamples and the top-rung LLM - TabPFN RMSE CIs coincide with
results/icl/calibration_bootstrap.csv and report_extras.json (paired_top).
Duplicated molecules in a resample share their sigma, i.e. they are ties and enter the RMSE50 tie
rule like any other tie. Point estimate = statistic on the full sample.

Differences (LLM minus baseline, baseline in {tabpfn, gp_tanimoto}): per resample, statistic of
the LLM minus statistic of the baseline on the same molecules; p_neg = share of resamples with
difference < 0 (NaN resamples, e.g. constant sigma, excluded). For RMSE / RMSE50 a negative
difference favours the LLM; for rho and gain50 a POSITIVE one does.

Combined over draws (draw = 0, N <= 120): molecules are resampled once; for each resample the
statistic (or difference) is computed in each of the three draws on the same molecules and averaged
over the draws. The point estimate is the mean of the three full-sample values. So draw-to-draw
variation is averaged out and the CI reflects test-molecule sampling only.

Missing molecules: two antiviral cells (GPT-6.1 Sol and GPT-6 Luna, N = 60, draw 3) have 149 of 150
molecules; they are dropped from that cell's statistics (NaN-aware), and for differences the
baseline is restricted to the same 149 molecules. n_mol reports the molecules used (minimum over
draws for draw = 0).

Output columns: dataset, model, baseline (empty for single-signal rows), size, draw (0 = combined
over draws 1-3), stat, est, lo, hi, p_neg (differences only), n_mol.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, load_split  # noqa: E402
from selective_prediction import TIE_DECIMALS, rmse_most_certain  # noqa: E402

ICL = os.path.join(RESULTS, "icl")
BASE = os.path.join(RESULTS, "baselines")
OUT = os.path.join(ICL, "bootstrap_cells.csv")
DS = ["lipophilicity", "bace", "antiviral", "esol"]
TOP = {"lipophilicity": 1000, "bace": 1000, "antiviral": 844, "esol": 976}
LLMS = ["sonnet55", "haiku55", "gpt61sol", "gpt6astra", "gem38flash", "gpt6luna", "glm53flash"]
BASELINES = ["tabpfn", "gp_tanimoto", "rf_desc", "chemprop_ens"]
REFS = ["tabpfn", "gp_tanimoto"]
SMALL = (30, 60, 120)
STATS = ["RMSE", "rho", "RMSE50", "gain50"]
B, BOOT_SEED = 1000, 0


# ------------------------------------------------------------------------------ data
def load_cells(ds):
    """{(model, size, draw): (abs_err, sigma)} aligned to one molecule order (NaN = missing)."""
    P = pd.read_csv(os.path.join(ICL, f"{ds}_permolecule.csv"), dtype={"mol_id": str})
    P = P[P.reasoning == "low"]
    mols = np.array([str(m) for m in load_split(ds)["test_ids"]])   # split order (see docstring)
    assert len(set(mols)) == len(mols) and set(P.mol_id) <= set(mols), ds
    pos = pd.Series(np.arange(len(mols)), index=mols)
    cells, src_order = {}, {}
    keys = [(n, d) for n in SMALL for d in (1, 2, 3)] + [(TOP[ds], 1)]

    def put(key, mol_id, err, sig):
        e, s = np.full(len(mols), np.nan), np.full(len(mols), np.nan)
        i = pos[mol_id].values
        e[i], s[i] = np.abs(np.asarray(err, float)), np.round(np.asarray(sig, float), TIE_DECIMALS)
        bad = ~(np.isfinite(e) & np.isfinite(s))
        e[bad] = s[bad] = np.nan
        cells[key] = (e, s)
        src_order[key] = i      # row order of the source file (point estimates, see full_stats)

    for m in LLMS:
        for n, d in keys:
            g = P[(P.model == m) & (P["size"] == n) & (P.draw == d)]
            assert g.mol_id.is_unique and len(g) >= 140, (ds, m, n, d, len(g))
            put((m, n, d), g.mol_id, g.pred - g.truth, g.uc1)
    for m in BASELINES:
        for n, d in keys:
            f = os.path.join(BASE, f"{m}__{ds}__N{n}" + ("" if d == 1 else f"__d{d}") + ".csv")
            D = pd.read_csv(f, dtype={"mol_id": str})
            assert len(D) == len(mols) and set(D.mol_id) == set(mols) and int(D.seed.iloc[0]) == 0, f
            put((m, n, d), D.mol_id, D.prediction - D.truth, D.uncertainty)
    return mols, cells, keys, src_order


# ------------------------------------------------------------------------------ statistics
def full_stats(e, s, order=None):
    """Full-sample statistics. RMSE50 is exact and row-order independent; `order` (the source file's
    row order) is still applied so the inputs are literally those of selective_prediction.py."""
    if order is not None:
        e, s = e[order], s[order]
    ok = np.isfinite(e)
    e, s = e[ok], s[ok]
    rmse = float(np.sqrt(np.mean(e ** 2)))
    rho = stats.spearmanr(s, e).correlation if np.ptp(s) > 0 else np.nan
    r50 = rmse_most_certain(e, s, 0.5)
    return dict(RMSE=rmse, rho=float(rho), RMSE50=r50, gain50=1 - r50 / rmse), int(ok.sum())


def row_ranks(X):
    """Average ranks along axis 1; NaN entries stay NaN and do not affect the others."""
    Xi = np.where(np.isnan(X), np.inf, X)
    R = stats.rankdata(Xi, method="average", axis=1)
    R[np.isnan(X)] = np.nan
    return R


def row_pearson(A, Bm):
    A = A - np.nanmean(A, axis=1, keepdims=True)
    Bm = Bm - np.nanmean(Bm, axis=1, keepdims=True)
    num = np.nansum(A * Bm, axis=1)
    den = np.sqrt(np.nansum(A ** 2, axis=1) * np.nansum(Bm ** 2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def boot_rmse50(E2, S):
    """Exact RMSE50 per row (NaN = missing): k = floor(n_valid / 2) slots, t = k-th smallest sigma,
    sqrt((sum_{S<t} E2 + (r / m) * sum_{S==t} E2) / k) with m = #{S == t}, r = k - #{S < t}
    (square root of the expected MSE under uniformly random tie-breaking, as selective_rmse)."""
    valid = np.isfinite(S)
    k = np.maximum(1, (valid.sum(1) * 0.5).astype(int))          # int(0.5 * n) as selective_rmse
    Si = np.where(valid, S, np.inf)
    t = np.take_along_axis(np.sort(Si, axis=1), (k - 1)[:, None], axis=1)[:, 0]
    below, tie = Si < t[:, None], Si == t[:, None]
    E2z = np.where(valid, E2, 0.0)
    s_below = np.where(below, E2z, 0.0).sum(1)
    s_tie = np.where(tie, E2z, 0.0).sum(1)
    r = k - below.sum(1)
    m = tie.sum(1)
    return np.sqrt((s_below + r / m * s_tie) / k)


def boot_stats(e, s, idx):
    E, S = e[idx], s[idx]
    E2 = E ** 2
    rmse = np.sqrt(np.nanmean(E2, axis=1))
    rho = row_pearson(row_ranks(S), row_ranks(E))
    r50 = boot_rmse50(E2, S)
    return dict(RMSE=rmse, rho=rho, RMSE50=r50, gain50=1 - r50 / rmse)


# ------------------------------------------------------------------------------ main
def summarise(est, bvals, diff):
    v = bvals[np.isfinite(bvals)]
    lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
    return dict(est=est, lo=float(lo), hi=float(hi), p_neg=float((v < 0).mean()) if diff and len(v) else np.nan)


def main():
    t0 = time.time()
    rows = []
    sp = pd.read_csv(os.path.join(ICL, "selective_prediction.csv"))
    sp = sp[((sp.family == "LLM") & (sp.reasoning == "low")) |
            ((sp.family == "baseline") & (pd.to_numeric(sp.seed, errors="coerce") == 0))]
    max_dev = 0.0
    for ds in DS:
        mols, cells, keys, src_order = load_cells(ds)
        n = len(mols)
        idx = np.random.default_rng(BOOT_SEED).integers(0, n, (B, n))
        res = {}      # (model, size, draw) -> (est dict, boot dict, n_mol)
        for (m, nn, d), (e, s) in cells.items():
            est, nm = full_stats(e, s, src_order[(m, nn, d)])
            res[(m, nn, d)] = (est, boot_stats(e, s, idx), nm)
            # cross-check the point estimates against selective_prediction.csv
            ref = sp[(sp.ds == ds) & (sp.model == m) & (sp["size"] == nn) & (sp.draw == d)]
            assert len(ref) == 1, (ds, m, nn, d, len(ref))
            for st in STATS:
                max_dev = max(max_dev, abs(est[st] - float(ref[st].iloc[0])))
        print(f"  {ds}: {len(cells)} single cells done ({time.time() - t0:.0f} s)", flush=True)
        diffs = {}    # (llm, ref, size, draw) -> (est dict, boot dict, n_mol)
        for m in LLMS:
            for b in REFS:
                for nn, d in keys:
                    eA, sA = cells[(m, nn, d)]
                    miss = ~np.isfinite(eA)
                    estA, bootA, nm = res[(m, nn, d)]
                    if miss.any():    # restrict the baseline to the LLM's molecules
                        eB, sB = cells[(b, nn, d)]
                        eB, sB = np.where(miss, np.nan, eB), np.where(miss, np.nan, sB)
                        estB, _ = full_stats(eB, sB, src_order[(b, nn, d)])
                        bootB = boot_stats(eB, sB, idx)
                    else:
                        estB, bootB, _ = res[(b, nn, d)]
                    diffs[(m, b, nn, d)] = ({st: estA[st] - estB[st] for st in STATS},
                                            {st: bootA[st] - bootB[st] for st in STATS}, nm)

        def emit(model, base, nn, d, est, boot, nm):
            for st in STATS:
                rows.append(dict(dataset=ds, model=model, baseline=base, size=nn, draw=d, stat=st,
                                 **summarise(est[st], boot[st], base != ""), n_mol=nm))

        for (m, nn, d), (est, boot, nm) in res.items():
            emit(m, "", nn, d, est, boot, nm)
        for (m, b, nn, d), (est, boot, nm) in diffs.items():
            emit(m, b, nn, d, est, boot, nm)
        for nn in SMALL:          # combined over draws: average per resample over the three draws
            for m in LLMS + BASELINES:
                trip = [res[(m, nn, d)] for d in (1, 2, 3)]
                emit(m, "", nn, 0, {st: np.mean([x[0][st] for x in trip]) for st in STATS},
                     {st: np.mean([x[1][st] for x in trip], axis=0) for st in STATS}, min(x[2] for x in trip))
            for m in LLMS:
                for b in REFS:
                    trip = [diffs[(m, b, nn, d)] for d in (1, 2, 3)]
                    emit(m, b, nn, 0, {st: np.mean([x[0][st] for x in trip]) for st in STATS},
                         {st: np.mean([x[1][st] for x in trip], axis=0) for st in STATS}, min(x[2] for x in trip))
        print(f"  {ds}: done ({time.time() - t0:.0f} s)", flush=True)
    R = pd.DataFrame(rows)
    R["dataset"] = pd.Categorical(R.dataset, DS, ordered=True)
    R = R.sort_values(["dataset", "size", "draw", "baseline", "model", "stat"], kind="stable").reset_index(drop=True)
    R.to_csv(OUT, index=False)
    print(f"{len(R)} rows -> {OUT}; point estimates vs selective_prediction.csv max |dev| = {max_dev:.2e}; "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
