"""
Second-stage uncertainty analysis of the ICL main run (reasoning `low`, R = 3) and of the classical
baselines on the identical cells, with the v1 calibration suite (src/v1reuse/uq_methods_eval.py,
imported, not edited).

Inputs (read-only):
  results/icl/<ds>_permolecule.csv      (src/analyze_icl.py; rows with reasoning == "low" only)
  results/baselines/<method>__<ds>__N<size>.csv   (seed 0; tabpfn and rf_desc)
Signals (every signal is scored as a Gaussian N(prediction, sigma^2)):
  UC1    LLM self-reported sigma (mean over the repeats)            -> permolecule `uc1`
  UC3    LLM repeat spread, sigma = sqrt(variance of the R = 3 repeat predictions, ddof = 1)
         (sigma = 0 where all repeats are identical: counted as n_sigma_zero)
  sigma  tabpfn: (q84 - q16) / 2;  rf_desc: std over the trees
Every sigma is floored at 0.01 (analyze_icl.SIGMA_FLOOR) before scoring; n_sigma_clipped counts the
floored values (UC3: identical repeats and spreads < 0.01).

Per cell (dataset x model/method x signal x size [x subset]):
  (truth, prediction, sigma rounded to 9 decimals first: see as_decimal)
  n, RMSE, MAE, rho_sigma_err = Spearman(sigma, |err|) (raw, unfloored sigma)
  PICP68 / PICP90 / MPIW / NLL / CRPS   analyze_icl.gaussian_metrics (z = 1.0 / 1.6449, as cells.csv)
  ECE, miscal_area                       uq.quantile_calibration (PIT on a 21-point grid, mean |gap|)
  ENCE, sigma_CV                         baselines/analyze_baselines.ence_stable (v1 ENCE, stable sort)
  s                                      uq.variance_scaling_factor = sqrt(mean(z^2)), z = err / sigma,
                                         on the full cell: s > 1 over-confident, s < 1 under-confident
  s_robust                               sqrt(median(z^2) / 0.4549): the same scale from the median,
                                         insensitive to a few near-zero sigmas (local addition)
  PICP68_scaled_in / PICP90_scaled_in / ECE_scaled_in   with s * sigma in-sample (shape check)
  Recalibration on a 50/50 calibration / test split (uq.recalibrate_cell; its module RNG and
  numpy's global RNG are re-seeded to 42 before every call). All cells of a dataset are put in the
  order of the split's test_ids first, so every signal of every model is split into the SAME
  75 calibration / 75 test molecules:
    s_cal                         variance-scaling factor fitted on the calibration half
    PICP68_test / PICP90_test / NLL_test           raw sigma on the test half
    PICP68_scaled / PICP90_scaled / NLL_scaled     s_cal * sigma on the test half (out-of-sample)
    ECE_test, ECE_scaled, ECE_isotonic             uq.recalibrate_cell (test half)
    picp90_varscale_v1                             uq's own PICP90 after scaling (z = 1.64485)
    conformal90_cov / conformal90_width            uq split conformal on |residual| (sigma-free:
                                                   identical for UC1 and UC3 of the same model)
    conformal90n_cov / conformal90n_width          local addition: normalised split conformal on
                                                   |residual| / sigma (q90 of the calibration
                                                   scores, interval mu +- q * sigma), i.e. how much
                                                   the signal sharpens a guaranteed interval
  tabpfn only: PICP68_q / PICP90_q from the raw quantiles [q16, q84] / [q05, q95].
ESOL additionally per subset `recalled` / `never_recalled` (permolecule recalled_zeroshot True /
False; None = label unscorable at 3 figures, excluded from both); no split recalibration there.

Paired bootstrap (1000 resamples over molecules, numpy default_rng(0) per dataset, the same index
matrix for both arms) at the top rung: Sonnet 5.5 UC1 minus TabPFN for mean |err|, RMSE and ECE.

Outputs: results/icl/calibration_cells.csv, results/icl/calibration_bootstrap.csv,
         docs/RESULTS_CALIBRATION_R3.md

    python src/analyze_calibration.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "v1reuse"))
sys.path.insert(0, os.path.join(HERE, "baselines"))
from common import DATASET_KEYS, RESULTS, ROOT, load_split  # noqa: E402
from analyze_icl import SIGMA_FLOOR, gaussian_metrics  # noqa: E402
from analyze_baselines import ence_stable  # noqa: E402
from report_icl import ORDER, PRETTY  # noqa: E402
import uq_methods_eval as uq  # noqa: E402

warnings.filterwarnings("ignore")

OUT_CSV = os.path.join(RESULTS, "icl", "calibration_cells.csv")
OUT_BOOT = os.path.join(RESULTS, "icl", "calibration_bootstrap.csv")
OUT_MD = os.path.join(ROOT, "docs", "RESULTS_CALIBRATION_R3.md")
REASONING = "low"
SIZES = [30, 60, 120, 250, 500]
TOP = {"lipophilicity": 1000, "bace": 1000, "antiviral": 844, "esol": 976}
BASE_METHODS = ["tabpfn", "rf_desc"]
SPLIT_SEED = 42
BOOT_B, BOOT_SEED = 1000, 0
MED_CHI2_1 = float(stats.chi2.ppf(0.5, 1))   # 0.4549
MAIN_COLS = ["dataset", "model", "signal", "size", "subset", "n", "ECE", "ENCE", "s", "PICP68", "PICP90",
             "PICP68_scaled", "PICP90_scaled", "conformal90_cov", "conformal90_width", "NLL"]


# ---------------------------------------------------------------------------------------------
# per-cell metrics
# ---------------------------------------------------------------------------------------------

def split_indices(n):
    """The calibration / test halves exactly as uq.recalibrate_cell draws them after re-seeding."""
    idx = np.random.default_rng(SPLIT_SEED).permutation(n)
    return idx[: n // 2], idx[n // 2:]


def as_decimal(*arrays):
    """Round to 9 decimals. The labels and answers are 2-3-decimal numbers, but the repeat means and
    the CSV round trip add 1-ulp noise, which otherwise decides (i) whether an exactly recalled
    value (residual 0, PIT = 0.5) counts as PIT <= 0.5 in the ECE curve (ESOL: 54 of 211 zero
    residuals carry such noise), (ii) how tied |err| / sigma values are ranked (Spearman) and
    (iii) how tied sigmas are binned (ENCE). Rounding makes decimal ties exact ties."""
    out = []
    for a in arrays:
        a = np.round(np.asarray(a, float), 9)
        assert np.isfinite(a).all()
        out.append(a)
    return out


def cell_metrics(y, mu, sig_raw, recal=True, q=None):
    y, mu, sig_raw = as_decimal(y, mu, sig_raw)
    n = len(y)
    sig = np.clip(sig_raw, SIGMA_FLOOR, None)
    err = np.round(np.abs(mu - y), 9)
    c = dict(n=n, n_sigma_zero=int(np.sum(sig_raw == 0)), share_sigma_zero=float(np.mean(sig_raw == 0)),
             n_sigma_clipped=int(np.sum(sig_raw < SIGMA_FLOOR)),
             RMSE=float(np.sqrt(np.mean(err ** 2))), MAE=float(np.mean(err)), mean_sigma=float(np.mean(sig)),
             rho_sigma_err=float(stats.spearmanr(sig_raw, err).correlation) if np.ptp(sig_raw) > 0 else np.nan)
    gm = gaussian_metrics(y, mu, sig)
    c.update({k: gm[k] for k in ["PICP68", "PICP90", "MPIW68", "MPIW90", "NLL", "CRPS"]})
    qc = uq.quantile_calibration(mu, y, sig)
    c["ECE"], c["miscal_area"] = (qc[2], qc[3]) if qc else (np.nan, np.nan)
    c["ENCE"], c["sigma_CV"] = ence_stable(mu, y, sig)
    z2 = (err / sig) ** 2
    c["s"] = uq.variance_scaling_factor(mu, y, sig)
    c["s_robust"] = float(np.sqrt(np.median(z2) / MED_CHI2_1))
    gs = gaussian_metrics(y, mu, c["s"] * sig)
    c["PICP68_scaled_in"], c["PICP90_scaled_in"] = gs["PICP68"], gs["PICP90"]
    qs = uq.quantile_calibration(mu, y, c["s"] * sig)
    c["ECE_scaled_in"] = qs[2] if qs else np.nan
    if q is not None:   # tabpfn raw quantiles
        for p, lo, hi in [(68, "q16", "q84"), (90, "q05", "q95")]:
            L, H = q[lo].to_numpy(float), q[hi].to_numpy(float)
            c[f"PICP{p}_q"] = float(np.mean((y >= L) & (y <= H)))
            c[f"MPIW{p}_q"] = float(np.mean(H - L))
    if not recal or n < 20:
        return c
    # --- v1 recalibration on the fixed 50/50 split (seeded before every call) ---
    np.random.seed(SPLIT_SEED)
    uq.RNG = np.random.default_rng(SPLIT_SEED)
    rc = uq.recalibrate_cell(mu, y, sig)
    cal, tst = split_indices(n)
    s_cal = uq.variance_scaling_factor(mu[cal], y[cal], sig[cal])
    assert abs(s_cal - rc["var_scale_factor"]) < 1e-12, "split replica differs from uq.recalibrate_cell"
    c["s_cal"] = s_cal
    gt = gaussian_metrics(y[tst], mu[tst], sig[tst])
    gts = gaussian_metrics(y[tst], mu[tst], s_cal * sig[tst])
    c.update(PICP68_test=gt["PICP68"], PICP90_test=gt["PICP90"], NLL_test=gt["NLL"],
             PICP68_scaled=gts["PICP68"], PICP90_scaled=gts["PICP90"], NLL_scaled=gts["NLL"],
             MPIW90_scaled=gts["MPIW90"],
             ECE_test=rc["ece_raw"], ECE_scaled=rc["ece_varscale"], ECE_isotonic=rc["ece_isotonic"],
             picp90_varscale_v1=rc["picp90_varscale"],
             conformal90_cov=rc["picp90_conformal"], conformal90_width=rc["mpiw90_conformal"])
    # normalised split conformal (local addition; same split, same quantile rule as uq)
    sc_cal = err[cal] / sig[cal]
    lvl = min(1.0, np.ceil((len(cal) + 1) * 0.90) / len(cal))
    qn = float(np.quantile(sc_cal, lvl, method="higher"))
    c["conformal90n_cov"] = float(np.mean(err[tst] <= qn * sig[tst]))
    c["conformal90n_width"] = float(np.mean(2 * qn * sig[tst]))
    return c


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------

def load_llm(ds):
    P = pd.read_csv(os.path.join(RESULTS, "icl", f"{ds}_permolecule.csv"), dtype={"mol_id": str})
    P = P[P.reasoning == REASONING].copy()
    if "draw" in P.columns:
        P = P[P.draw == 1].copy()          # main split only (context draws 2/3: src/report_icl.py --draws)
    P["uc3_sigma"] = np.sqrt(P.uc3)
    return P


def load_base(ds, method, size):
    return pd.read_csv(os.path.join(RESULTS, "baselines", f"{method}__{ds}__N{size}.csv"), dtype={"mol_id": str})


def ordered(D, test_ids):
    """Cell rows in the order of the split's test_ids (identical 50/50 split for every signal)."""
    assert set(D.mol_id) <= set(test_ids) and not D.mol_id.duplicated().any()
    pos = {m: i for i, m in enumerate(test_ids)}
    return D.assign(_pos=D.mol_id.map(pos)).sort_values("_pos").drop(columns="_pos").reset_index(drop=True)


def all_cells():
    rows, aligned = [], {}
    for ds in DATASET_KEYS:
        test_ids = [str(m) for m in load_split(ds)["test_ids"]]
        P = load_llm(ds)
        flags = P.drop_duplicates("mol_id").set_index("mol_id").recalled_zeroshot
        for size in SIZES + [TOP[ds]]:
            cells = []
            for model in ORDER:
                D = ordered(P[(P.model == model) & (P["size"] == size)], test_ids)
                for sig_name, col in [("UC1", "uc1"), ("UC3", "uc3_sigma")]:
                    cells.append((model, "LLM", sig_name, D, D[col], None))
            for method in BASE_METHODS:
                B = ordered(load_base(ds, method, size), test_ids)
                assert len(B) == len(test_ids) and (B.seed == 0).all()
                cells.append((method, "baseline", "sigma", B.rename(columns={"prediction": "pred"}), B.uncertainty,
                              B if method == "tabpfn" else None))
            for model, fam, sig_name, D, sig, qd in cells:
                key = dict(dataset=ds, model=model, family=fam, signal=sig_name, size=size)
                rows.append(dict(key, subset="all", **cell_metrics(D.truth, D.pred, sig, q=qd)))
                aligned[(ds, model, sig_name, size)] = (D, sig)
                if ds == "esol":
                    fl = D.mol_id.map(flags)
                    for sub, val in [("recalled", True), ("never_recalled", False)]:
                        m = fl.eq(val).to_numpy()
                        rows.append(dict(key, subset=sub, **cell_metrics(
                            D.truth[m], D.pred[m], sig[m], recal=False, q=None if qd is None else qd[m])))
    C = pd.DataFrame(rows)
    C = C[MAIN_COLS[:5] + ["family"] + MAIN_COLS[5:] + [c for c in C.columns if c not in MAIN_COLS + ["family"]]]
    return C, aligned


# ---------------------------------------------------------------------------------------------
# paired bootstrap: Sonnet 5.5 UC1 vs TabPFN at the top rung
# ---------------------------------------------------------------------------------------------

def ece_of(mu, y, sig):
    return uq.quantile_calibration(mu, y, sig)[2]


def bootstrap(aligned):
    out = []
    for ds in DATASET_KEYS:
        A, sa = aligned[(ds, "sonnet55", "UC1", TOP[ds])]
        B, sb = aligned[(ds, "tabpfn", "sigma", TOP[ds])]
        assert (A.mol_id.values == B.mol_id.values).all() and np.allclose(A.truth, B.truth)
        y, ma, mb, sa, sb = as_decimal(A.truth, A.pred, B.pred, sa, sb)
        sa, sb = np.clip(sa, SIGMA_FLOOR, None), np.clip(sb, SIGMA_FLOOR, None)
        ea, eb = np.abs(ma - y), np.abs(mb - y)
        n = len(y)
        idx = np.random.default_rng(BOOT_SEED).integers(0, n, (BOOT_B, n))
        stats_ = {
            "mean_abs_err": (lambda i: ea[i].mean() - eb[i].mean()),
            "RMSE": (lambda i: np.sqrt(np.mean(ea[i] ** 2)) - np.sqrt(np.mean(eb[i] ** 2))),
            "ECE": (lambda i: ece_of(ma[i], y[i], sa[i]) - ece_of(mb[i], y[i], sb[i])),
        }
        full = np.arange(n)
        for name, f in stats_.items():
            d = np.array([f(i) for i in idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append(dict(dataset=ds, size=TOP[ds], metric=name, n=n, sonnet=None, tabpfn=None,
                            diff=float(f(full)), ci_lo=float(lo), ci_hi=float(hi),
                            excludes_zero=bool(lo > 0 or hi < 0)))
            if name == "mean_abs_err":
                out[-1].update(sonnet=float(ea.mean()), tabpfn=float(eb.mean()))
            elif name == "RMSE":
                out[-1].update(sonnet=float(np.sqrt(np.mean(ea ** 2))), tabpfn=float(np.sqrt(np.mean(eb ** 2))))
            else:
                out[-1].update(sonnet=float(ece_of(ma, y, sa)), tabpfn=float(ece_of(mb, y, sb)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------------------------

def label(model, signal):
    return f"{PRETTY.get(model, model)} {signal}" if signal != "sigma" else PRETTY.get(model, model)


def size_table(C, ds, value, signal="UC1", fmt="{:.2f}", subset="all"):
    d = C[(C.dataset == ds) & (C.subset == subset)]
    sizes = SIZES + [TOP[ds]]
    lines = ["| model | " + " | ".join(f"N={s}" for s in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for model, sig in [(m, signal) for m in ORDER] + [(b, "sigma") for b in BASE_METHODS]:
        r = d[(d.model == model) & (d.signal == sig)].set_index("size")[value]
        lines.append(f"| {label(model, sig)} | " + " | ".join(fmt.format(r[s]) for s in sizes) + " |")
    return "\n".join(lines)


def signal_rows(d):
    order = [(m, s) for m in ORDER for s in ("UC1", "UC3")] + [(b, "sigma") for b in BASE_METHODS]
    for model, sig in order:
        r = d[(d.model == model) & (d.signal == sig)]
        if len(r):
            yield model, sig, r.iloc[0]


def f2(v, fmt="{:.2f}"):
    return "" if pd.isna(v) else fmt.format(v)


def top_table(d, recal=True):
    cols = [("n", "{:.0f}"), ("ECE", "{:.3f}"), ("ENCE", "{:.2f}"), ("s", "{:.2f}"), ("s_robust", "{:.2f}"),
            ("PICP68", "{:.2f}"), ("PICP90", "{:.2f}"), ("rho_sigma_err", "{:.2f}"), ("NLL", "{:.2f}"),
            ("share_sigma_zero", "{:.2f}")]
    lines = ["| signal | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for model, sig, r in signal_rows(d):
        lines.append(f"| {label(model, sig)} | " + " | ".join(f2(r[c], f) for c, f in cols) + " |")
    if not recal:
        return "\n".join(lines)
    cols2 = [("s_cal", "{:.2f}"), ("PICP68_test", "{:.2f}"), ("PICP68_scaled", "{:.2f}"),
             ("PICP90_test", "{:.2f}"), ("PICP90_scaled", "{:.2f}"), ("ECE_test", "{:.3f}"),
             ("ECE_scaled", "{:.3f}"), ("ECE_isotonic", "{:.3f}"), ("NLL_test", "{:.2f}"), ("NLL_scaled", "{:.2f}"),
             ("MPIW90_scaled", "{:.2f}"), ("conformal90_cov", "{:.2f}"), ("conformal90_width", "{:.2f}"),
             ("conformal90n_cov", "{:.2f}"), ("conformal90n_width", "{:.2f}")]
    lines += ["", "After recalibration (fitted on 75 calibration molecules, scored on the other 75; same split "
              "for every signal):", "",
              "| signal | " + " | ".join(c for c, _ in cols2) + " |", "|---|" + "---|" * len(cols2)]
    for model, sig, r in signal_rows(d):
        lines.append(f"| {label(model, sig)} | " + " | ".join(f2(r[c], f) for c, f in cols2) + " |")
    return "\n".join(lines)


def boot_table(BT):
    lines = ["| dataset | N | metric | Sonnet 5.5 UC1 | TabPFN | diff (Sonnet - TabPFN) | 95 % CI | CI excludes 0 |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in BT.iterrows():
        lines.append(f"| {r.dataset} | {r['size']} | {r.metric} | {r.sonnet:.3f} | {r.tabpfn:.3f} | {r['diff']:+.3f} | "
                     f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {'yes' if r.excludes_zero else 'no'} |")
    return "\n".join(lines)


def _rng(x, fmt="{:.2f}"):
    return f"{fmt.format(x.min())}-{fmt.format(x.max())}"


def findings_md(C, BT):
    """Findings section. Every number is computed here from C / BT; the wording (over- vs
    under-confident, better / worse) was written against the outputs of 2026-10-12 (main run final)."""
    A = C[C.subset == "all"].copy()
    clean = A.dataset != "esol"
    U1, U3 = A[A.signal == "UC1"], A[A.signal == "UC3"]
    c1u = clean[U1.index]
    TP = A[A.model == "tabpfn"]
    L = []
    # (1) over-/under-confidence of UC1
    parts, meds = [], []
    for m in ORDER:
        x = U1[c1u & (U1.model == m)].s
        parts.append(f"{PRETTY[m]} {x.median():.2f} ({_rng(x)}, s > 1 in {int((x > 1).sum())}/{len(x)})")
        meds.append(x.median())
    tb = TP[TP.dataset != "esol"].s
    es = U1[U1.dataset == "esol"]
    srob = U1[(U1.dataset == "esol") & (U1["size"] == TOP["esol"])].set_index("model").s_robust

    def s_of(ds, m, size):
        return U1[(U1.dataset == ds) & (U1.model == m) & (U1["size"] == size)].s.iloc[0]

    L += ["1. **Self-reported sigma (UC1) is mildly over-confident on the clean sets, under-confident on ESOL.** "
          "Variance-scaling factor s on lipophilicity / BACE / antiviral (18 cells per model, median and range): "
          + "; ".join(parts) + f". TabPFN for comparison: {tb.median():.2f} ({_rng(tb)}). So the LLM sigmas are "
          f"too narrow by a median {100 * (min(meds) - 1):.0f}-{100 * (max(meds) - 1):.0f} %, most for GPT-6 Luna, Gemini 3.8 Flash and GLM 5.3 Flash; Haiku 5.5 is "
          "the exception on lipophilicity (s "
          f"{_rng(U1[(U1.dataset == 'lipophilicity') & (U1.model == 'haiku55')].s)}, too wide). Over-confidence "
          f"tends to shrink with N (e.g. Sonnet antiviral {s_of('antiviral', 'sonnet55', 30):.2f} -> "
          f"{s_of('antiviral', 'sonnet55', TOP['antiviral']):.2f}, BACE {s_of('bace', 'sonnet55', 30):.2f} -> "
          f"{s_of('bace', 'sonnet55', TOP['bace']):.2f}). On ESOL every LLM has s {_rng(es.s)} (under-confident "
          "in the RMSE sense), and the median-based s_robust at the top rung is much lower ("
          + ", ".join(f"{PRETTY[m]} {srob[m]:.2f}" for m in ORDER)
          + "): for the typical ESOL molecule the stated sigma is too wide, and a minority of large errors keeps "
          "the mean-square s near 1 (point 5)."]
    # (2) does variance scaling fix the coverage?
    c1 = U1[c1u]
    e1 = U1[~c1u]

    def dev(v, nom):
        return (v - nom).abs().median()

    L += ["2. **Variance scaling fixes the coverage on the clean sets, not on ESOL; it does not fix the shape.** "
          f"Held-out halves of the {len(c1)} clean UC1 cells: PICP90 median {c1.PICP90_test.median():.2f} -> "
          f"{c1.PICP90_scaled.median():.2f} after scaling (median |PICP90 - 0.90| {dev(c1.PICP90_test, 0.90):.3f} -> "
          f"{dev(c1.PICP90_scaled, 0.90):.3f}), PICP68 {c1.PICP68_test.median():.2f} -> "
          f"{c1.PICP68_scaled.median():.2f} (|dev| {dev(c1.PICP68_test, 0.6827):.3f} -> "
          f"{dev(c1.PICP68_scaled, 0.6827):.3f}; binomial SE at n = 75 is 0.035 / 0.054). The PIT-ECE hardly moves "
          f"({c1.ECE_test.median():.3f} -> {c1.ECE_scaled.median():.3f}; isotonic {c1.ECE_isotonic.median():.3f}): "
          f"one scalar repairs the width, not the distribution shape. ESOL ({len(e1)} cells): PICP68 "
          f"{e1.PICP68_test.median():.2f} -> {e1.PICP68_scaled.median():.2f}, PICP90 {e1.PICP90_test.median():.2f} -> "
          f"{e1.PICP90_scaled.median():.2f}, ECE {e1.ECE_test.median():.3f} -> {e1.ECE_scaled.median():.3f}: a single "
          "factor cannot repair a mixture of too-wide (recalled) and too-narrow (never-recalled) intervals."]
    # (3) UC3 vs UC1
    M = U1.merge(U3, on=["dataset", "model", "size"], suffixes=("_1", "_3"))
    Me = M[M.dataset == "esol"]

    def sh(c, D=M):
        return f"{int((D[c + '_3'] > D[c + '_1']).sum())}/{len(D)}"

    z = U3.groupby("model").share_sigma_zero.median()
    st = M[(M.dataset == "esol") & (M.model == "sonnet55") & (M["size"] == TOP["esol"])].iloc[0]
    L += ["3. **UC3 (repeat spread) is not better than UC1, before or after scaling.** With R = 3 the spread is "
          f"exactly zero for {_rng(U3.share_sigma_zero, '{:.1%}')} of the molecules of a cell (median per model: "
          + ", ".join(f"{PRETTY[m]} {z[m]:.1%}" for m in ORDER)
          + f"; max {U3.share_sigma_zero.max():.1%}, Sonnet on ESOL), and where it is not zero it is far too small: "
          f"s {_rng(U3.s, '{:.1f}')}, s_robust {_rng(U3.s_robust, '{:.1f}')}, held-out PICP68 median "
          f"{U3.PICP68_test.median():.2f}. After scaling it over-covers (PICP90 median "
          f"{U3.PICP90_scaled.median():.2f}) with much wider intervals: of the {len(M)} model x dataset x size cells "
          f"UC3 has the higher scaled NLL in {sh('NLL_scaled')} (median {M.NLL_scaled_3.median():.2f} vs "
          f"{M.NLL_scaled_1.median():.2f}), the wider normalised-conformal 90 % interval in "
          f"{sh('conformal90n_width')} (median {M.conformal90n_width_3.median():.2f} vs "
          f"{M.conformal90n_width_1.median():.2f}) and the higher scaled ECE in {sh('ECE_scaled')}. As a ranking "
          f"signal it beats UC1 in {sh('rho_sigma_err')} cells (rho median {M.rho_sigma_err_3.median():.2f} vs "
          f"{M.rho_sigma_err_1.median():.2f}), mostly on ESOL ({sh('rho_sigma_err', Me)}; Sonnet top rung "
          f"{st.rho_sigma_err_3:.2f} vs {st.rho_sigma_err_1:.2f})."]
    # (4) LLM UC1 vs TabPFN after recalibration
    T = TP[["dataset", "size", "NLL_scaled", "ECE_scaled", "conformal90n_width"]]
    MT = U1.merge(T, on=["dataset", "size"], suffixes=("", "_t"))
    cnt = MT.assign(nll=MT.NLL_scaled < MT.NLL_scaled_t, w=MT.conformal90n_width < MT.conformal90n_width_t,
                    ece=MT.ECE_scaled < MT.ECE_scaled_t).groupby("dataset")[["nll", "w", "ece"]].sum()
    A["sharp"] = A.conformal90n_width / A.conformal90_width
    shp = A.groupby(["dataset", "model", "signal"]).sharp.median()

    def llm_sh(ds):
        return shp.loc[[(ds, m, "UC1") for m in ORDER]]

    bt = BT.set_index(["dataset", "metric"])

    def ci(ds, met):
        r = bt.loc[(ds, met)]
        return f"{ds} {r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"

    L += ["4. **Against TabPFN after recalibration the LLMs win on NLL / width through accuracy, not through a "
          "better sigma.** Number of the 42 LLM UC1 cells per dataset (7 models x 6 sizes) that beat TabPFN on the "
          "same cell, held-out half after variance scaling -- scaled NLL / normalised-conformal 90 % width / scaled "
          "ECE: " + "; ".join(f"{ds} {int(cnt.loc[ds, 'nll'])} / {int(cnt.loc[ds, 'w'])} / {int(cnt.loc[ds, 'ece'])}"
                             for ds in DATASET_KEYS)
          + ". The calibration shape (ECE) is better for TabPFN in most cells; NLL and width favour the LLMs where "
          "their errors are smaller (lipophilicity, ESOL). How much a signal sharpens a guaranteed interval "
          "(normalised / plain split-conformal width, median over sizes; < 1 = the sigma helps): "
          + "; ".join(f"{ds} LLM UC1 {_rng(llm_sh(ds))}, TabPFN {shp.loc[(ds, 'tabpfn', 'sigma')]:.2f}"
                      for ds in DATASET_KEYS)
          + f". On the clean sets no sigma narrows the guaranteed interval by more than "
          f"{100 * (1 - min(llm_sh(ds).min() for ds in DATASET_KEYS[:3])):.0f} % (median); on ESOL the LLM sigma "
          f"does (Sonnet {shp.loc[('esol', 'sonnet55', 'UC1')]:.2f}) and TabPFN's widens it. "
          "TabPFN's raw quantile intervals at the top rung cover PICP68_q / PICP90_q "
          + ", ".join(f"{ds} {TP[TP['size'] == TOP[ds]].set_index('dataset').loc[ds, 'PICP68_q']:.2f} / "
                      f"{TP[TP['size'] == TOP[ds]].set_index('dataset').loc[ds, 'PICP90_q']:.2f}" for ds in DATASET_KEYS)
          + ". Paired bootstrap at the top rung (Sonnet 5.5 UC1 minus TabPFN, 95 % CI): mean |err| "
          + ", ".join(ci(ds, "mean_abs_err") for ds in DATASET_KEYS) + "; ECE "
          + ", ".join(ci(ds, "ECE") for ds in DATASET_KEYS)
          + ". Sonnet is significantly more accurate on lipophilicity and ESOL; its ECE is not distinguishable "
          "from TabPFN's on the clean sets and significantly worse on ESOL."]
    # (5) ESOL recalled vs never recalled
    E = C[(C.dataset == "esol") & (C.signal == "UC1")]
    er, en = E[E.subset == "recalled"], E[E.subset == "never_recalled"]
    tr = C[(C.dataset == "esol") & (C.model == "tabpfn") & (C["size"] == TOP["esol"])].set_index("subset")
    per = [f"{PRETTY[m]} {_rng(er[er.model == m].s)} vs {_rng(en[en.model == m].s)}" for m in ORDER]
    L += ["5. **ESOL: the recalled subset is under-confident, the never-recalled subset over-confident.** UC1 s, "
          f"recalled ({int(er.n.iloc[0])} molecules) vs never-recalled scorable ({int(en.n.iloc[0])}), range over "
          "the 6 sizes: " + "; ".join(per) + f". PICP68 recalled {_rng(er.PICP68)} vs never-recalled "
          f"{_rng(en.PICP68)}; ECE {_rng(er.ECE, '{:.3f}')} vs {_rng(en.ECE, '{:.3f}')}. The heavy recallers "
          "(Sonnet, Sol, Gemini) state sigmas 2-4x wider than their errors on the molecules they recall and slightly "
          "too narrow ones on the rest: the stated sigma does not track whether the value is recalled. TabPFN at "
          f"the top rung: s {tr.loc['recalled', 's']:.2f} vs {tr.loc['never_recalled', 's']:.2f}, ECE "
          f"{tr.loc['recalled', 'ECE']:.3f} vs {tr.loc['never_recalled', 'ECE']:.3f} -- the never-recalled "
          "molecules are harder for TabPFN too (same direction), but TabPFN is not under-confident on the "
          "recalled ones. The pooled ESOL numbers "
          "(s near 1, PICP90 near nominal) are calibrated for the wrong reason, as in v1. Caveat: 51-62 molecules "
          "per subset (ENCE then has 5-6 molecules per bin)."]
    return ["## Findings", "",
            "Numbers computed by the script from the tables below; held-out = the 75-molecule test half of the "
            "fixed 50/50 split; clean sets = lipophilicity, BACE, antiviral.", ""] + L


def write_md(C, BT):
    A = C[C.subset == "all"]
    out = ["# Calibration of the ICL uncertainty signals (reasoning `low`, R = 3)", "",
           "Generated by `src/analyze_calibration.py` from `results/icl/<ds>_permolecule.csv` (reasoning `low`) and "
           "the seed-0 baseline cells; full table `results/icl/calibration_cells.csv`, bootstrap "
           "`results/icl/calibration_bootstrap.csv`. Every signal is scored as N(prediction, sigma^2), sigma floored "
           "at 0.01. UC1 = self-reported sigma (mean of 3 repeats), UC3 = sqrt of the repeat variance "
           "(0 when the 3 repeats are identical), TabPFN sigma = (q84 - q16)/2, RF (descriptors) sigma = tree std.",
           "",
           "Definitions: ECE = mean |observed - nominal| of the PIT calibration curve on a 21-point grid "
           "(v1 `quantile_calibration`; 0 = calibrated); ENCE = mean |RMV - RMSE| / RMV over 10 sigma bins; "
           "s = sqrt(mean((err/sigma)^2)), the factor that makes s*sigma RMSE-calibrated (s > 1 over-confident, "
           "s < 1 under-confident); s_robust = the same from the median of (err/sigma)^2; PICP68 / PICP90 = "
           "coverage of +-1.0 / +-1.645 sigma (as `cells.csv`); `_scaled` = coverage on the 75 held-out molecules "
           "after scaling sigma by s fitted on the other 75 (v1 `recalibrate_cell`, seed 42, identical split for "
           "all signals); conformal90 = v1 split-conformal interval on |residual| (does not use sigma, so it is "
           "the same for UC1 and UC3 of one model); conformal90n = normalised split conformal on |residual|/sigma "
           "(local addition: how much the signal sharpens a guaranteed interval); NLL = Gaussian negative "
           "log-likelihood; rho = Spearman(sigma, |err|).", ""]
    out += findings_md(C, BT)
    out += ["", "## Paired bootstrap: Sonnet 5.5 UC1 vs TabPFN at the top rung", "",
            f"{BOOT_B} resamples over the 150 test molecules (numpy default_rng({BOOT_SEED}) per dataset; the same "
            "resample for both arms); percentile 95 % CI of the difference; negative = Sonnet lower (better).", "",
            boot_table(BT), ""]
    for ds in DATASET_KEYS:
        out += [f"## {ds}", "",
                "**(a) ECE of UC1 (baselines: their sigma)**", "", size_table(A, ds, "ECE", fmt="{:.3f}"), "",
                "**(b) variance-scaling factor s of UC1 (s > 1 over-confident)**", "", size_table(A, ds, "s"), "",
                "s of UC3 (repeat spread):", "", size_table(A, ds, "s", signal="UC3"), "",
                f"**(c) top rung (N = {TOP[ds]}): all signals**", "",
                top_table(A[(A.dataset == ds) & (A["size"] == TOP[ds])]), ""]
        if ds == "esol":
            for sub, txt in [("recalled", "recalled (reproduced to 3 significant figures by >= 1 panel model in "
                                          "the zero-shot check)"),
                             ("never_recalled", "never recalled, scorable (>= 3-figure label, no panel model "
                                                "reproduced it)")]:
                d = C[(C.dataset == ds) & (C["size"] == TOP[ds]) & (C.subset == sub)]
                out += [f"**(d) ESOL top rung, {txt}; n = {int(d.n.iloc[0])}**", "", top_table(d, recal=False), ""]
            out += ["ESOL UC1 ECE and s per size, recalled / never-recalled:", ""]
            for sub in ["recalled", "never_recalled"]:
                d = C[(C.dataset == ds) & (C.subset == sub)]
                out += [f"{sub} -- ECE:", "", size_table(d, ds, "ECE", fmt="{:.3f}", subset=sub), "",
                        f"{sub} -- s:", "", size_table(d, ds, "s", subset=sub), ""]
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")


def main():
    C, aligned = all_cells()
    C.to_csv(OUT_CSV, index=False, float_format="%.10g")
    BT = bootstrap(aligned)
    BT.to_csv(OUT_BOOT, index=False, float_format="%.10g")
    write_md(C, BT)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    top = C[(C.subset == "all") & (C["size"] == C.dataset.map(TOP))]
    print(top[["dataset", "model", "signal", "n", "ECE", "ENCE", "s", "s_robust", "PICP68", "PICP90",
               "PICP68_scaled", "PICP90_scaled", "ECE_scaled", "NLL", "NLL_scaled", "conformal90n_width",
               "share_sigma_zero"]].round(3).to_string(index=False))
    print(BT.round(3).to_string(index=False))
    print(f"{len(C)} rows -> {OUT_CSV}\n-> {OUT_BOOT}\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
