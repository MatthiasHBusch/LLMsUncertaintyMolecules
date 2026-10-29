"""
Per-cell metrics of the classical baselines (results/baselines/<method>__<ds>__N<size>[__s<seed>].csv),
with the SAME metric names as src/analyze_icl.py so the two tables can be concatenated:

  RMSE, MAE, spearman_pred = Spearman(truth, prediction)
  rho_UC1_err = Spearman(uncertainty, |error|)   (for a baseline, "UC1" is its 1-sigma column:
                tabpfn (q84-q16)/2, rf / rf_desc tree std, trivial training std, gp_tanimoto GP
                predictive SD incl. noise, mlp_ens / chemprop_ens member SD, knn_tanimoto SD of the
                5 neighbour labels (floor 0.01); see run_baselines.py)
Column correspondence with results/icl/cells.csv: method <-> model, seed <-> reasoning (the
remaining key columns dataset / size are shared).
  mean_UC1, NLL, CRPS, PICP68, MPIW68, PICP90, MPIW90  (Gaussian N(prediction, sigma^2);
                analyze_icl.gaussian_metrics, imported)
plus the v1 calibration suite (src/v1reuse/uq_methods_eval.py, imported, not edited):
  ECE, miscal_area (quantile_calibration), Winkler_90 (interval_metrics), ENCE / sigma_CV (local
  copy of uq.ence with a stable sigma sort -- see ence_stable),
  recalibration on a 50/50 split (recalibrate_cell; its module RNG is re-seeded to 42 before
  every cell so each cell's split is reproducible and independent of the processing order):
  var_scale_factor, ece_varscale, picp90_varscale, picp90_conformal, mpiw90_conformal.
For tabpfn additionally the raw-quantile intervals (no Gaussian assumption):
  PICP68_q / MPIW68_q from [q16, q84], PICP90_q / MPIW90_q from [q05, q95].

Context draws (2026-10-13): cells <stem>__d<d>.csv (run_baselines.py --draw d) get draw = d,
all other files draw = 1 (column `draw` after `seed`). Rows are sorted by draw first, so the
draw-1 rows keep exactly their previous content and order.

Outputs: results/baselines/summary.csv (one row per method x dataset x size x seed x draw), with an
`env` column (python / numpy / scipy / sklearn versions of the producing environment).
Runs in the base env (no tabpfn needed).

    python src/baselines/analyze_baselines.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "v1reuse"))
from common import DATASET_KEYS, RESULTS, load_split  # noqa: E402
from analyze_icl import gaussian_metrics  # noqa: E402
import uq_methods_eval as uq  # noqa: E402

OUT = os.path.join(RESULTS, "baselines")
# new methods (2026-10-13) are appended, so the rows of the original four keep their relative order
METHOD_ORDER = ["tabpfn", "rf_desc", "rf", "trivial", "gp_tanimoto", "mlp_ens", "chemprop_ens", "knn_tanimoto"]
DRAW_RE = re.compile(r"__d(\d+)\.csv$")


def draw_of(path):
    """Context draw of a cell file: `..__d<k>.csv` -> k, no suffix -> 1 (main split)."""
    m = DRAW_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 1


def ence_stable(pred, y, sigma, n_bins=10):
    """ENCE (Levi et al. 2022) exactly as v1reuse/uq_methods_eval.ence, with ONE deliberate
    deviation: the sigma sort is `kind="stable"`. v1 uses the default quicksort, whose order of
    TIED sigmas (frequent for tree std / trivial sigma) differs between numpy builds, so v1's ENCE
    of the same cell differed between the base and uc44 environments. Stable sorting keeps ties in
    test-set order and makes the binning environment-independent. Returns (ENCE, sigma_CV)."""
    pred, y, sigma = uq._clean(pred, y, sigma)
    if len(y) < n_bins:
        return np.nan, np.nan
    order = np.argsort(sigma, kind="stable")
    terms = []
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        rmv = np.sqrt(np.mean(sigma[b] ** 2))
        rmse = np.sqrt(np.mean((y[b] - pred[b]) ** 2))
        if rmv > 0:
            terms.append(abs(rmv - rmse) / rmv)
    mu_sigma = float(np.mean(sigma))
    cv = float(np.std(sigma) / mu_sigma) if mu_sigma > 0 else np.nan
    return (float(np.mean(terms)) if terms else np.nan), cv


def cell_metrics(D, draw=1):
    y, mu, sig = D.truth.to_numpy(float), D.prediction.to_numpy(float), D.uncertainty.to_numpy(float)
    err = np.abs(mu - y)
    r = D.iloc[0]
    c = dict(dataset=r.dataset, method=r.method, size=int(r["size"]), seed=int(r.seed), draw=int(draw), n_mol=len(D),
             RMSE=float(np.sqrt(np.mean((mu - y) ** 2))), MAE=float(np.mean(err)),
             spearman_pred=stats.spearmanr(y, mu).correlation,
             rho_UC1_err=stats.spearmanr(sig, err).correlation if np.ptp(sig) > 0 else np.nan,
             mean_UC1=float(np.mean(sig)),
             **gaussian_metrics(y, mu, sig))
    qc = uq.quantile_calibration(mu, y, sig)
    c["ECE"], c["miscal_area"] = (qc[2], qc[3]) if qc else (np.nan, np.nan)
    c["ENCE"], c["sigma_CV"] = ence_stable(mu, y, sig)   # not uq.ence: see ence_stable
    c["Winkler_90"] = uq.interval_metrics(mu, y, sig).get("Winkler_90", np.nan)
    uq.RNG = np.random.default_rng(uq.SEED)
    rc = uq.recalibrate_cell(mu, y, sig) or {}
    for k in ["var_scale_factor", "ece_varscale", "picp90_varscale", "picp90_conformal", "mpiw90_conformal"]:
        c[k] = rc.get(k, np.nan)
    if D.q16.notna().all():
        for p, lo, hi in [(68, "q16", "q84"), (90, "q05", "q95")]:
            L, H = D[lo].to_numpy(float), D[hi].to_numpy(float)
            c[f"PICP{p}_q"] = float(np.mean((y >= L) & (y <= H)))
            c[f"MPIW{p}_q"] = float(np.mean(H - L))
    c["wall_s"] = float(D.wall_s.iloc[0])
    return c


def check_cell(D, path):
    """Cell integrity: 150 finite predictions on exactly the split's test_ids."""
    r = D.iloc[0]
    test_ids = [str(m) for m in load_split(r.dataset)["test_ids"]]
    assert len(D) == len(test_ids) and set(D.mol_id.astype(str)) == set(test_ids), f"{path}: test ids"
    assert np.isfinite(D.prediction).all() and np.isfinite(D.uncertainty).all(), f"{path}: non-finite"


def main():
    files = sorted(f for f in glob.glob(os.path.join(OUT, "*__*__N*.csv")))
    if not files:
        sys.exit("no baseline cells yet: run src/baselines/run_baselines.py")
    rows = []
    for f in files:
        D = pd.read_csv(f, dtype={"mol_id": str})
        check_cell(D, f)
        draw = draw_of(f)
        if "draw" in D.columns:
            assert (D["draw"] == draw).all(), f"{f}: draw column != file-name draw {draw}"
        rows.append(cell_metrics(D, draw))
    S = pd.DataFrame(rows)
    import scipy
    import sklearn
    S["env"] = (f"python {sys.version.split()[0]} numpy {np.__version__} scipy {scipy.__version__} "
                f"sklearn {sklearn.__version__}")
    S["dataset"] = pd.Categorical(S.dataset, [d for d in DATASET_KEYS if d in set(S.dataset)], ordered=True)
    S["method"] = pd.Categorical(S.method, [m for m in METHOD_ORDER if m in set(S.method)], ordered=True)
    S = S.sort_values(["draw", "dataset", "method", "size", "seed"]).reset_index(drop=True)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = ["dataset", "method", "size", "RMSE", "spearman_pred", "rho_UC1_err", "PICP68", "PICP90",
            "PICP68_q", "PICP90_q", "mean_UC1", "wall_s"]
    s0 = S[(S.seed == 0) & (S.draw == 1)]
    print(s0[[c for c in cols if c in S.columns]].round(3).to_string(index=False))
    if (S.seed != 0).any():
        g = S[S.draw == 1].groupby(["dataset", "method", "size"], observed=True)
        agg = g.agg(n_seeds=("seed", "size"), RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                    rho_mean=("rho_UC1_err", "mean"), rho_sd=("rho_UC1_err", "std")).reset_index()
        agg = agg[agg.n_seeds > 1]
        print("\nseed spread (cells with > 1 seed):")
        print(agg.round(3).to_string(index=False))
    if (S.draw != 1).any():
        sd = S[S.seed == 0]
        keys = sd[sd.draw != 1][["dataset", "method", "size"]].drop_duplicates()
        sd = sd.merge(keys, on=["dataset", "method", "size"])
        p = sd.pivot_table(index=["dataset", "method", "size"], columns="draw", values="RMSE", observed=True)
        p.columns = [f"RMSE_d{c}" for c in p.columns]
        p["mean"], p["sd"] = p.mean(axis=1), p.std(axis=1, ddof=1)
        print("\ncontext-draw spread (seed 0, RMSE per draw):")
        print(p.round(3).to_string())
    print(f"\n{len(S)} cells -> {os.path.join(OUT, 'summary.csv')}")


if __name__ == "__main__":
    main()
