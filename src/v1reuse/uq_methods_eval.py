"""
uq_methods_eval.py
==================

Extended uncertainty-quantification (UQ) evaluation on the EXISTING experiment
data (no new LLM calls). Implements the diagnostics named in the UQ research plan
so the report's "is this method useful here?" verdicts are empirically grounded:

  * Regression CALIBRATION: quantile/PIT calibration curve, Expected Calibration
    Error (ECE_reg), miscalibration area, Expected Normalized Calibration Error
    (ENCE, Levi et al. 2022), sharpness.
  * PREDICTION INTERVALS: coverage (PICP) and Mean Prediction Interval Width
    (MPIW) at nominal 50/68/90/95%, plus the Winkler/interval score at 90%.
  * FORWARD CALIBRATION (recalibration): scalar variance scaling, isotonic
    recalibration (Kuleshov et al. 2018), and split-conformal intervals
    (Angelopoulos & Bates 2021), each fit on a held-out calibration split and
    scored on the complementary test split (50/50, seed 42).
  * GLOBAL SENSITIVITY / SHAPLEY: RDKit molecular descriptors -> an RF surrogate
    for per-molecule "hardness" (|error|) and for the model's self-uncertainty;
    permutation importance + exact Shapley values (coalition enumeration) rank
    which descriptors drive error and uncertainty (applicability-domain view).

Every UQ signal is treated as a Gaussian predictive distribution N(pred, sigma^2):
  * LLM  UC1  -> sigma = UC1_self_reported            (verbalized 1-sigma)
  * LLM  UC3  -> sigma = sqrt(UC3_iter_var)           (R=5 iteration spread)
  * baselines -> sigma = UC_baseline                  (ensemble / tree std)
scored against the matching residuals on the same scale.

Inputs (already in results/):
  uc_per_sample.csv, baseline_per_sample.csv, data/*.csv
Outputs (results/, figures/):
  uq_calibration_summary.csv, uq_recalibration.csv, uq_sensitivity.csv
  figures/uq_reliability.png, figures/uq_coverage_width.png, figures/uq_sensitivity.png

Usage:
  <mace_env python> src/uq_methods_eval.py
"""
from __future__ import annotations

import json
import warnings
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# numpy 2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
RESULTS_DIR = HERE.parent / "results"
FIG_DIR = HERE.parent / "figures"
SEED = 42
RNG = np.random.default_rng(SEED)

# Nominal central coverages and their two-sided Gaussian z-multipliers.
LEVELS = [0.50, 0.68, 0.90, 0.95]
ZMULT = {c: float(norm.ppf(0.5 + c / 2.0)) for c in LEVELS}

DATASETS = {
    "delaney":       {"csv": "delaney-processed.csv", "smiles_col": "smiles",
                      "target_col": "measured log solubility in mols per litre", "name_col": "Compound ID"},
    "lipophilicity": {"csv": "Lipophilicity.csv", "smiles_col": "smiles", "target_col": "exp", "name_col": None},
    "qm7":           {"csv": "qm7.csv", "smiles_col": "smiles", "target_col": "u0_atom", "name_col": None},
}


# ---------------------------------------------------------------------------
# Core UQ metrics (Gaussian predictive model)
# ---------------------------------------------------------------------------

def _clean(pred, y, sigma):
    pred, y, sigma = map(lambda a: np.asarray(a, float), (pred, y, sigma))
    m = np.isfinite(pred) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    return pred[m], y[m], sigma[m]


def quantile_calibration(pred, y, sigma, n_grid=20):
    """Kuleshov et al. 2018 probabilistic calibration.

    For each nominal probability p, the observed frequency is the fraction of
    points whose true value falls at or below the predicted p-quantile, i.e.
    mean( PIT <= p ) with PIT = Phi((y - pred)/sigma).
    Returns (p_grid, observed, ece, miscal_area).
    """
    pred, y, sigma = _clean(pred, y, sigma)
    if len(y) < 5:
        return None
    pit = norm.cdf((y - pred) / sigma)
    p_grid = np.linspace(0.0, 1.0, n_grid + 1)
    observed = np.array([np.mean(pit <= p) for p in p_grid])
    ece = float(np.mean(np.abs(observed - p_grid)))          # mean |gap|
    miscal_area = float(_trapz(np.abs(observed - p_grid), p_grid))  # area between curves
    return p_grid, observed, ece, miscal_area


def ence(pred, y, sigma, n_bins=10):
    """Expected Normalized Calibration Error (Levi et al. 2022).

    Bin by predicted sigma; in each bin compare the root-mean-variance
    RMV = sqrt(mean(sigma^2)) to the empirical RMSE. ENCE = mean |RMV-RMSE|/RMV.
    Also returns the coefficient of variation of sigma (sharpness dispersion).
    """
    pred, y, sigma = _clean(pred, y, sigma)
    if len(y) < n_bins:
        return np.nan, np.nan
    order = np.argsort(sigma)
    bins = np.array_split(order, n_bins)
    terms = []
    for b in bins:
        if len(b) == 0:
            continue
        rmv = np.sqrt(np.mean(sigma[b] ** 2))
        rmse = np.sqrt(np.mean((y[b] - pred[b]) ** 2))
        if rmv > 0:
            terms.append(abs(rmv - rmse) / rmv)
    mu_sigma = float(np.mean(sigma))
    cv = float(np.std(sigma) / mu_sigma) if mu_sigma > 0 else np.nan
    return (float(np.mean(terms)) if terms else np.nan), cv


def interval_metrics(pred, y, sigma):
    """PICP and MPIW at each nominal central coverage, plus the 90% Winkler score."""
    pred, y, sigma = _clean(pred, y, sigma)
    out = {}
    if len(y) < 5:
        return out
    for c in LEVELS:
        z = ZMULT[c]
        lo, hi = pred - z * sigma, pred + z * sigma
        covered = (y >= lo) & (y <= hi)
        out[f"PICP_{int(c*100)}"] = float(np.mean(covered))
        out[f"MPIW_{int(c*100)}"] = float(np.mean(hi - lo))
    # Winkler / interval score at 90% (lower is better): width + 2/alpha * miss distance
    c = 0.90
    alpha = 1 - c
    z = ZMULT[c]
    lo, hi = pred - z * sigma, pred + z * sigma
    width = hi - lo
    below = y < lo
    above = y > hi
    score = width + (2 / alpha) * (lo - y) * below + (2 / alpha) * (y - hi) * above
    out["Winkler_90"] = float(np.mean(score))
    out["mean_sigma"] = float(np.mean(sigma))
    out["rmse"] = float(np.sqrt(np.mean((y - pred) ** 2)))
    # Proper scores (Gaussian predictive N(pred, sigma^2)); lower is better.
    out["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - pred) ** 2 / (2 * sigma ** 2)))
    w = (y - pred) / sigma
    out["crps"] = float(np.mean(sigma * (w * (2 * norm.cdf(w) - 1) + 2 * norm.pdf(w) - 1 / np.sqrt(np.pi))))
    return out


# ---------------------------------------------------------------------------
# Forward calibration / recalibration
# ---------------------------------------------------------------------------

def variance_scaling_factor(pred, y, sigma):
    """Scalar s so that s*sigma is calibrated in the RMSE sense:
    s = sqrt(mean( ((y-pred)/sigma)^2 ))  (matches mean predicted variance to MSE)."""
    pred, y, sigma = _clean(pred, y, sigma)
    z = (y - pred) / sigma
    return float(np.sqrt(np.mean(z ** 2)))


def recalibrate_cell(pred, y, sigma):
    """Fit recalibration on a 50/50 calibration split, score on the held-out split.

    Returns before/after ECE and 90% PICP/MPIW for:
      * raw
      * variance scaling (single scalar)
      * isotonic recalibration of the predictive CDF (Kuleshov 2018)
      * split conformal (absolute-residual, distribution-free 90% interval)
    """
    pred, y, sigma = _clean(pred, y, sigma)
    n = len(y)
    if n < 20:
        return None
    idx = RNG.permutation(n)
    half = n // 2
    cal, tst = idx[:half], idx[half:]
    pc, yc, sc = pred[cal], y[cal], sigma[cal]
    pt, yt, st = pred[tst], y[tst], sigma[tst]

    def ece_of(p, yy, s):
        r = quantile_calibration(p, yy, s)
        return r[2] if r else np.nan

    def picp_mpiw90(p, yy, s):
        z = ZMULT[0.90]
        lo, hi = p - z * s, p + z * s
        return float(np.mean((yy >= lo) & (yy <= hi))), float(np.mean(hi - lo))

    res = {"n_test": len(tst)}
    res["ece_raw"], res["ece_raw_full"] = ece_of(pt, yt, st), ece_of(pred, y, sigma)
    res["picp90_raw"], res["mpiw90_raw"] = picp_mpiw90(pt, yt, st)

    # (1) variance scaling
    s_scale = variance_scaling_factor(pc, yc, sc)
    res["var_scale_factor"] = s_scale
    res["ece_varscale"] = ece_of(pt, yt, st * s_scale)
    res["picp90_varscale"], res["mpiw90_varscale"] = picp_mpiw90(pt, yt, st * s_scale)

    # (2) isotonic recalibration of predictive CDF (Kuleshov 2018)
    pit_c = norm.cdf((yc - pc) / sc)
    p_grid = np.linspace(0, 1, 21)
    emp = np.array([np.mean(pit_c <= p) for p in p_grid])
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_grid, emp)
    pit_t = norm.cdf((yt - pt) / st)
    cal_p = iso.predict(pit_t)          # calibrated probabilities of the test residuals
    # calibrated ECE: how uniform are the calibrated PIT values
    res["ece_isotonic"] = float(np.mean(np.abs(
        np.array([np.mean(cal_p <= p) for p in p_grid]) - p_grid)))

    # (3) split conformal 90% interval (absolute residual, marginal guarantee)
    scores = np.abs(yc - pc)
    q = np.quantile(scores, min(1.0, np.ceil((len(cal) + 1) * 0.90) / len(cal)), method="higher")
    lo, hi = pt - q, pt + q
    res["picp90_conformal"] = float(np.mean((yt >= lo) & (yt <= hi)))
    res["mpiw90_conformal"] = float(2 * q)
    return res


# ---------------------------------------------------------------------------
# Molecular descriptors + sensitivity / Shapley
# ---------------------------------------------------------------------------

DESCRIPTORS = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "HeavyAtomCount", "FractionCSP3", "RingCount",
]


def descriptor_matrix(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
    getters = {
        "MolWt": Descriptors.MolWt, "MolLogP": Crippen.MolLogP, "TPSA": rdMolDescriptors.CalcTPSA,
        "NumHDonors": Lipinski.NumHDonors, "NumHAcceptors": Lipinski.NumHAcceptors,
        "NumRotatableBonds": Lipinski.NumRotatableBonds, "NumAromaticRings": Lipinski.NumAromaticRings,
        "HeavyAtomCount": Descriptors.HeavyAtomCount, "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "RingCount": rdMolDescriptors.CalcNumRings,
    }
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            rows.append([np.nan] * len(DESCRIPTORS))
        else:
            rows.append([float(getters[d](mol)) for d in DESCRIPTORS])
    return np.array(rows, float)


def exact_shapley(model, X, background):
    """Exact Shapley values for a fitted model over DESCRIPTORS features, using a
    mean-imputation background (KernelSHAP with the coalition weighting reduced to
    the exact enumeration; feasible for ~10 features -> 2^10 coalitions).
    Returns mean(|phi|) per feature over rows of X."""
    d = X.shape[1]
    feats = list(range(d))
    # Precompute value function v(S) = model on X with features not in S set to background,
    # averaged over rows. We evaluate per-row then average |phi|.
    phis = np.zeros((len(X), d))
    # enumerate all subsets once as bitmasks
    subsets = []
    for r in range(d + 1):
        for c in combinations(feats, r):
            subsets.append(frozenset(c))
    # cache predictions of each coalition for the whole batch
    def v(S):
        Xs = np.tile(background, (len(X), 1))
        for j in S:
            Xs[:, j] = X[:, j]
        return model.predict(Xs)
    cache = {S: v(S) for S in subsets}
    for j in feats:
        for S in subsets:
            if j in S:
                continue
            Sj = S | {j}
            w = 1.0 / (d * comb(d - 1, len(S)))
            phis[:, j] += w * (cache[Sj] - cache[S])
    return np.mean(np.abs(phis), axis=0)


def sensitivity_for_target(X, target, label):
    m = np.isfinite(target) & np.all(np.isfinite(X), axis=1)
    Xf, yf = X[m], target[m]
    if len(yf) < 30:
        return []
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xf, yf)
    r2_oob = rf.score(Xf, yf)
    perm = permutation_importance(rf, Xf, yf, n_repeats=20, random_state=SEED, n_jobs=-1)
    bg = Xf.mean(axis=0)
    shap = exact_shapley(rf, Xf, bg)
    rows = []
    for i, d in enumerate(DESCRIPTORS):
        rows.append({"target": label, "descriptor": d, "n": int(len(yf)), "rf_r2_insample": float(r2_oob),
                     "perm_importance": float(perm.importances_mean[i]),
                     "shapley_mean_abs": float(shap[i])})
    return rows


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def build_signal_table():
    """One long table of (dataset, source, signal, train_size) -> arrays via a
    generator of per-cell (pred, y, sigma). Returns list of dicts describing cells."""
    cells = []
    # ---- LLM cells (UC1 verbalized sigma, UC3 iteration-spread sigma) ----
    up = pd.read_csv(RESULTS_DIR / "uc_per_sample.csv")
    for (ds, model, size), g in up.groupby(["dataset", "model", "train_size"]):
        pred = g["pred_mean"].to_numpy(float)
        y = g["target_value"].to_numpy(float)
        cells.append(dict(dataset=ds, source=model.split("/")[-1], family="LLM",
                          signal="UC1_verbalized", train_size=int(size),
                          pred=pred, y=y, sigma=g["UC1_self_reported"].to_numpy(float)))
        cells.append(dict(dataset=ds, source=model.split("/")[-1], family="LLM",
                          signal="UC3_iter_spread", train_size=int(size),
                          pred=pred, y=y, sigma=np.sqrt(g["UC3_iter_var"].to_numpy(float))))
    # ---- classical baseline cells (ensemble / tree std) ----
    bp = pd.read_csv(RESULTS_DIR / "baseline_per_sample.csv")
    for (ds, model, size), g in bp.groupby(["dataset", "model", "train_size"]):
        cells.append(dict(dataset=ds, source=model, family="classical",
                          signal="ensemble_std", train_size=int(size),
                          pred=g["pred_mean"].to_numpy(float), y=g["target_value"].to_numpy(float),
                          sigma=g["UC_baseline"].to_numpy(float)))
    return cells


def run_calibration_and_intervals(cells):
    rows, recal_rows, reliability = [], [], {}
    for c in cells:
        base = {k: c[k] for k in ("dataset", "source", "family", "signal", "train_size")}
        qc = quantile_calibration(c["pred"], c["y"], c["sigma"])
        en, cv = ence(c["pred"], c["y"], c["sigma"])
        im = interval_metrics(c["pred"], c["y"], c["sigma"])
        rho = np.nan
        pp, yy, ss = _clean(c["pred"], c["y"], c["sigma"])
        if len(yy) >= 5 and np.unique(ss).size > 1:
            rho = float(spearmanr(ss, np.abs(pp - yy)).statistic)
        rec = dict(base, n=len(yy), spearman_sig_vs_abserr=rho,
                   ece_reg=qc[2] if qc else np.nan, miscal_area=qc[3] if qc else np.nan,
                   ence=en, sigma_cv=cv, **im)
        rows.append(rec)
        # keep reliability curve for a few headline cells
        if qc and c["train_size"] in (60, 1000):
            reliability[(c["dataset"], c["source"], c["signal"], c["train_size"])] = (qc[0], qc[1])
        # recalibration on the larger cells
        rc = recalibrate_cell(c["pred"], c["y"], c["sigma"])
        if rc:
            recal_rows.append(dict(base, **rc))
    return pd.DataFrame(rows), pd.DataFrame(recal_rows), reliability


def run_sensitivity():
    """Attribute per-molecule |error| and self-uncertainty to descriptors, for the
    headline LLM (gemini-3.5-flash) and the Chemprop ensemble, at N=1000."""
    up = pd.read_csv(RESULTS_DIR / "uc_per_sample.csv")
    bp = pd.read_csv(RESULTS_DIR / "baseline_per_sample.csv")
    all_rows = []
    for ds, cfg in DATASETS.items():
        df = pd.read_csv(DATA_DIR / cfg["csv"])
        key_col = cfg["name_col"] or cfg["smiles_col"]
        id2smi = {str(k).strip(): str(s) for k, s in zip(df[key_col], df[cfg["smiles_col"]])}

        # LLM headline cell
        g = up[(up.dataset == ds) & (up.model == "google/gemini-3.5-flash") & (up.train_size == 1000)]
        if len(g):
            smi = [id2smi.get(str(s).strip(), s) for s in g["sample_id"]]
            X = descriptor_matrix(smi)
            for tgt, lab in ((g["abs_error"].to_numpy(float), f"{ds}:LLM_abs_error"),
                             (g["UC1_self_reported"].to_numpy(float), f"{ds}:LLM_UC1")):
                all_rows += sensitivity_for_target(X, tgt, lab)

        # Chemprop ensemble cell (its UC and error) at N=1000
        gb = bp[(bp.dataset == ds) & (bp.model == "chemprop_ensemble") & (bp.train_size == 1000)]
        if len(gb):
            smi = [id2smi.get(str(s).strip(), s) for s in gb["sample_id"]]
            X = descriptor_matrix(smi)
            for tgt, lab in ((gb["abs_error"].to_numpy(float), f"{ds}:chemprop_abs_error"),
                             (gb["UC_baseline"].to_numpy(float), f"{ds}:chemprop_UC")):
                all_rows += sensitivity_for_target(X, tgt, lab)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(cal_df, reliability, sens_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) reliability diagram: headline LLM UC1 vs Chemprop, N=1000, per dataset
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, ds in zip(axes, ["delaney", "lipophilicity", "qm7"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        for (d, src, sig, sz), (pg, obs) in reliability.items():
            if d != ds or sz != 1000:
                continue
            if (src == "gemini-3.5-flash" and sig == "UC1_verbalized"):
                ax.plot(pg, obs, "-o", ms=3, label="LLM UC1 (Gemini 3.5)")
            elif src == "chemprop_ensemble":
                ax.plot(pg, obs, "-s", ms=3, label="Chemprop ens.")
        ax.set_title(ds); ax.set_xlabel("nominal"); ax.set_ylabel("observed")
        ax.legend(fontsize=7)
    fig.suptitle("Regression calibration (reliability) at N=1000")
    fig.tight_layout(); fig.savefig(FIG_DIR / "uq_reliability.png", dpi=140); plt.close(fig)

    # 2) coverage-width trade-off: PICP90 vs MPIW90 across signals, N=1000
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sub = cal_df[cal_df.train_size == 1000]
    for ax, ds in zip(axes, ["delaney", "lipophilicity", "qm7"]):
        d = sub[sub.dataset == ds]
        for _, r in d.iterrows():
            marker = "o" if r.family == "LLM" else "s"
            ax.scatter(r["MPIW_90"], r["PICP_90"], marker=marker, s=40)
            ax.annotate(f"{r.source[:8]}/{r.signal[:4]}", (r["MPIW_90"], r["PICP_90"]), fontsize=6)
        ax.axhline(0.90, color="grey", ls=":", lw=1)
        ax.set_title(ds); ax.set_xlabel("MPIW@90 (width)"); ax.set_ylabel("PICP@90 (coverage)")
    fig.suptitle("Coverage-width trade-off @90% (dotted = nominal 0.90)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "uq_coverage_width.png", dpi=140); plt.close(fig)

    # 3) sensitivity: Shapley of descriptors on LLM |error| per dataset
    if len(sens_df):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, ds in zip(axes, ["delaney", "lipophilicity", "qm7"]):
            d = sens_df[sens_df.target == f"{ds}:LLM_abs_error"].sort_values("shapley_mean_abs")
            if len(d):
                ax.barh(d.descriptor, d.shapley_mean_abs, color="steelblue")
            ax.set_title(f"{ds}: descriptors -> LLM |error|"); ax.set_xlabel("mean |Shapley|")
        fig.tight_layout(); fig.savefig(FIG_DIR / "uq_sensitivity.png", dpi=140); plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    print("[1/3] calibration + intervals + recalibration ...")
    cells = build_signal_table()
    cal_df, recal_df, reliability = run_calibration_and_intervals(cells)
    cal_df.to_csv(RESULTS_DIR / "uq_calibration_summary.csv", index=False)
    recal_df.to_csv(RESULTS_DIR / "uq_recalibration.csv", index=False)
    print(f"      {len(cal_df)} cells scored; {len(recal_df)} recalibrated.")
    print("[2/3] global sensitivity / Shapley on descriptors ...")
    sens_df = run_sensitivity()
    sens_df.to_csv(RESULTS_DIR / "uq_sensitivity.csv", index=False)
    print(f"      {len(sens_df)} descriptor-attribution rows.")
    print("[3/3] figures ...")
    make_figures(cal_df, reliability, sens_df)
    print("done. wrote uq_calibration_summary.csv, uq_recalibration.csv, uq_sensitivity.csv + figures/uq_*.png")

    # console highlight: LLM UC1 vs Chemprop calibration at N=1000
    hi = cal_df[(cal_df.train_size == 1000) &
                (cal_df.signal.isin(["UC1_verbalized", "ensemble_std"]))]
    cols = ["dataset", "source", "signal", "ece_reg", "miscal_area", "ence", "PICP_90", "MPIW_90", "Winkler_90"]
    print("\n=== calibration @N=1000 (headline) ===")
    print(hi[hi.source.isin(["gemini-3.5-flash", "chemprop_ensemble"])][cols].to_string(index=False))


if __name__ == "__main__":
    main()
