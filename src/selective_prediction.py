"""
Selective prediction ("RMSE of the most certain fraction") with proper tie handling.

Definition (exact since 2026-10-14; before: Monte-Carlo estimate over 200 tie orders). For one
cell (n test molecules with absolute error e_i and 1-sigma s_i), keep the k = floor(frac * n)
molecules with the smallest sigma (k >= 1; k = 75 / 120 for n = 150 and frac = 0.5 / 0.8) and
report the RMSE over them. Self-reported LLM sigmas are coarse (median ~30 distinct values per
150 molecules), so several molecules often share the sigma at the cut and "the k most certain" is
not unique; numpy's default argsort then picks an arbitrary subset of the tied group, and the
value moved by up to ~0.17 with the row order. Here the tied group is filled uniformly at random:
let t be the sigma of the k-th smallest value, the molecules with sigma < t are always selected,
and r of the m molecules with sigma == t fill the remaining slots (r = k - #{sigma < t}). Each tied
molecule is selected with probability r / m, so the EXPECTED mean squared error of the selected
set is exactly

    MSE_frac = ( SSE_below + (r / m) * SSE_tied ) / k,     SSE = sum of e_i^2 over the group,

and RMSE_frac := sqrt(MSE_frac), i.e. the square root of the expected MSE over random
tie-breaking -- NOT the expectation of the RMSE over tie orders (by Jensen the latter is <= this
value; the two coincide when the cut does not split a tied group, r = m). The value is
deterministic and independent of the input row order. The exact range over all tie orders is
also reported: _min (the r smallest squared errors of the tied group are kept) and _max (the r
largest).

Ties are detected on sigma rounded to 10 decimals: LLM UC1 is the mean over R = 3 repeats, and
float summation leaves spurious last-bit differences (0.36666666666666664 vs 0.3666666666666667)
in most cells, which would otherwise be treated as distinct values.

Inputs
  LLMs: results/icl/<ds>_permolecule.csv (src/analyze_icl.py; pred, truth, uc1 = self-reported
        sigma, mean over the parsed repeats), every (model, reasoning, draw, size) present.
  Baselines: every per-cell file results/baselines/<method>__<ds>__N<size>[__s<seed>][__d<draw>].csv
        (prediction, truth, uncertainty = the method's 1-sigma), every method / seed / draw.
Output: results/icl/selective_prediction.csv, one row per cell, columns
  ds, model, size, RMSE, rho, RMSE80, RMSE50, gain80, gain50,      (as before)
  draw, reasoning, seed, n_ties_groups,                            (new keys / tie count)
  family (LLM | baseline), n_mol, n_sigma_distinct, RMSE80_min, RMSE80_max, RMSE50_min, RMSE50_max
where rho = Spearman(sigma, |err|) (average ranks for ties), gainF = 1 - RMSE_F / RMSE,
n_ties_groups = number of distinct sigma values shared by >= 2 molecules, reasoning = "" and
seed = the baseline seed for baselines, seed = "" for LLMs (their R = 3 repeats are pooled).
`src/report_icl.py --draws` imports `rmse_most_certain` from here, so its RMSE50 and this CSV agree.
`risk_coverage` (the same tie rule on a grid of fractions q, with oracle and random) is used by
`src/make_figs.py` (figA1); it does not change this CSV.

    python src/selective_prediction.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASET_KEYS, RESULTS  # noqa: E402

FRACS = (0.8, 0.5)
TIE_DECIMALS = 10
OUT = os.path.join(RESULTS, "icl", "selective_prediction.csv")
BASE_RE = re.compile(r"^(?P<method>[a-z0-9_]+)__(?P<ds>[a-z]+)__N(?P<size>\d+)(?:__s(?P<seed>\d+))?(?:__d(?P<draw>\d+))?\.csv$")


def _clean(err, sig):
    err, sig = np.abs(np.asarray(err, float)), np.asarray(sig, float)
    ok = np.isfinite(err) & np.isfinite(sig)
    return err[ok], np.round(sig[ok], TIE_DECIMALS)


def selective_rmse(err, sig, frac=0.5, *_, **__):
    """RMSE of the floor(frac * n) smallest-sigma molecules, ties at the cut broken uniformly at
    random: value = sqrt(E[MSE]) = sqrt((SSE_below + (r / m) * SSE_tied) / k), the square root of the
    expected MSE (not the expected RMSE). Exact and row-order independent. Extra positional /
    keyword arguments (the former n_perm, seed) are accepted and ignored.
    Returns dict(mean = value, min, max over all tie orders, exact = True if the cut does not split a
    tied group); the key `mean` is kept for compatibility."""
    err, sig = _clean(err, sig)
    return _selective_k(err, sig, max(1, int(frac * len(err))))


def _selective_k(err, sig, k):
    """core of selective_rmse for cleaned |err|, rounded sigma and an explicit k (1 <= k <= n)."""
    se = err ** 2
    t = np.sort(sig, kind="stable")[k - 1]          # sigma value at the cut
    below, tie = sig < t, sig == t
    r = k - int(below.sum())                        # slots filled from the tied group
    s_below = float(se[below].sum())
    g_se = se[tie]
    m = len(g_se)
    v = float(np.sqrt((s_below + (r / m) * g_se.sum()) / k))
    if r == m:                                      # cut falls between groups: no randomness
        return dict(mean=v, min=v, max=v, exact=True)
    srt = np.sort(g_se)
    lo = float(np.sqrt((s_below + srt[:r].sum()) / k))
    hi = float(np.sqrt((s_below + srt[::-1][:r].sum()) / k))
    return dict(mean=v, min=lo, max=hi, exact=False)


def risk_coverage(err, sig, fracs=tuple(np.round(np.arange(0.1, 1.01, 0.1), 2))):
    """Risk-coverage curve: for each fraction q, the RMSE of the k = floor(q * n) smallest-sigma molecules
    with the same exact tie rule as selective_rmse (sqrt of the expected MSE over random tie-breaking at
    the cut), plus the oracle (the k smallest |err|, unique) and random selection (expected MSE of a
    uniformly random subset of size k = the full MSE, so random = RMSE at every q). k is computed as
    floor(q * n + 1e-9) so that q = 0.3, 0.7 give 45 / 105 of 150 (int(q * n) would give 44 / 104 by float
    rounding); for q = 0.5 / 0.8 and n = 150 this equals the k of selective_rmse. err may be signed.
    Returns a DataFrame with columns q, k, RMSE_q, RMSE_q_min, RMSE_q_max, oracle, random."""
    err, sig = _clean(err, sig)
    n = len(err)
    se_sorted = np.sort(err ** 2)
    rmse = float(np.sqrt(se_sorted.mean()))
    rows = []
    for q in fracs:
        k = int(min(n, max(1, np.floor(q * n + 1e-9))))
        r = _selective_k(err, sig, k)
        rows.append(dict(q=float(q), k=k, RMSE_q=r["mean"], RMSE_q_min=r["min"], RMSE_q_max=r["max"],
                         oracle=float(np.sqrt(se_sorted[:k].mean())), random=rmse))
    return pd.DataFrame(rows)


def rmse_most_certain(err, sig, frac=0.5, *_, **__):
    """RMSE of the `frac` most certain molecules (smallest sigma), exact square root of the expected
    MSE over random tie-breaking at the cut (see selective_rmse). err may be signed."""
    return selective_rmse(err, sig, frac)["mean"]


def tie_stats(sig):
    _, sig = _clean(np.zeros(len(sig)), sig)
    _, counts = np.unique(sig, return_counts=True)
    return int(len(counts)), int(np.sum(counts >= 2))


def cell_row(err, sig, **keys):
    err, sig = _clean(err, sig)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    row = dict(ds=keys.pop("ds"), model=keys.pop("model"), size=int(keys.pop("size")), RMSE=rmse,
               rho=stats.spearmanr(sig, err).correlation if np.ptp(sig) > 0 else np.nan)
    sel = {f: selective_rmse(err, sig, f) for f in FRACS}
    for f in FRACS:
        row[f"RMSE{int(round(100 * f))}"] = sel[f]["mean"]
    for f in FRACS:
        row[f"gain{int(round(100 * f))}"] = 1 - sel[f]["mean"] / rmse
    n_distinct, n_groups = tie_stats(sig)
    row.update(draw=int(keys.pop("draw")), reasoning=keys.pop("reasoning"), seed=keys.pop("seed"),
               n_ties_groups=n_groups, family=keys.pop("family"), n_mol=len(err), n_sigma_distinct=n_distinct)
    for f in FRACS:
        p = int(round(100 * f))
        row[f"RMSE{p}_min"], row[f"RMSE{p}_max"] = sel[f]["min"], sel[f]["max"]
    return row


def llm_rows():
    rows = []
    for ds in DATASET_KEYS:
        path = os.path.join(RESULTS, "icl", f"{ds}_permolecule.csv")
        if not os.path.exists(path):
            continue
        P = pd.read_csv(path, dtype={"mol_id": str})
        if "draw" not in P.columns:
            P["draw"] = 1
        for (m, rsn, d, n), g in P.groupby(["model", "reasoning", "draw", "size"], sort=True):
            rows.append(cell_row(g.pred - g.truth, g.uc1, ds=ds, model=m, size=n, draw=d, reasoning=rsn,
                                 seed="", family="LLM"))
    return rows


def baseline_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "baselines", "*__*__N*.csv"))):
        mt = BASE_RE.match(os.path.basename(f))
        if not mt:
            continue
        D = pd.read_csv(f, dtype={"mol_id": str})
        seed, draw = int(mt["seed"] or 0), int(mt["draw"] or 1)
        assert int(D.seed.iloc[0]) == seed and mt["method"] == D.method.iloc[0], f
        rows.append(cell_row(D.prediction - D.truth, D.uncertainty, ds=mt["ds"], model=mt["method"],
                             size=int(mt["size"]), draw=draw, reasoning="", seed=seed, family="baseline"))
    return rows


def main():
    S = pd.DataFrame(llm_rows() + baseline_rows())
    S["ds"] = pd.Categorical(S.ds, DATASET_KEYS, ordered=True)
    S = S.sort_values(["family", "ds", "model", "reasoning", "draw", "seed", "size"],
                      key=lambda c: c.astype(str) if c.name in ("seed",) else c).reset_index(drop=True)
    S.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    print(f"{len(S)} cells -> {OUT}  (LLM {int((S.family == 'LLM').sum())}, baseline "
          f"{int((S.family == 'baseline').sum())})")
    for fam, g in S.groupby("family"):
        rng50 = g.RMSE50_max - g.RMSE50_min
        print(f"  {fam}: RMSE50 depends on the tie order in {int((rng50 > 0).sum())} of {len(g)} cells; "
              f"range max {rng50.max():.3f} ({g.loc[rng50.idxmax(), 'model']}), median {rng50.median():.3f}; "
              f"median distinct sigmas {g.n_sigma_distinct.median():.0f}, tied groups {g.n_ties_groups.median():.0f}")
    show = S[(S.draw == 1) & ((S.family == "baseline") & (S.seed.astype(str) == "0") |
                              (S.family == "LLM") & (S.reasoning == "low"))]
    for n_ in sorted(show["size"].unique()):
        if n_ not in (30, 120) and n_ < 844:
            continue
        p = show[show["size"] == n_].pivot_table(index="model", columns="ds", values="RMSE50", observed=False)
        print(f"\nRMSE50 (sqrt of expected MSE over tie orders), draw 1, N = {n_}:")
        print(p.round(3).to_string())


if __name__ == "__main__":
    main()
