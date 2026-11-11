"""
Mixture sigma of the R = 3 repeated calls (review round of 2026-10-28, Appendix E): does combining the
stated sigma and the repeat spread rank the errors better than the stated sigma alone?

Per molecule, from the per-call records results/icl/<ds>__<model>__N<size>__low[__d<draw>].jsonl (reasoning
`low`, main run = draw 1 and the context draws 2-3; duplicated (mol_id, repeat) pairs dropped keeping the last
record and only parse_ok records used, exactly as src/analyze_icl.py builds results/icl/<ds>_permolecule.csv):
  pred   = mean_r yhat_r                                          (permolecule `pred`)
  e      = pred - truth
  UC1    = mean_r sigma_r (raw stated sigmas)                     (permolecule `uc1`)
  UC3    = SD of the repeat predictions, ddof = 1 (sqrt of permolecule `uc3`); undefined (NaN) for a molecule
           with a single parsed repeat, which is then left out of rho(UC3) (analyze_icl: nan_policy omit)
  sigma_mix^2 = mean_r(s_r^2 + yhat_r^2) - (mean_r yhat_r)^2 = mean_r s_r^2 + var_r(yhat_r) (ddof = 0),
           s_r = max(sigma_r, 0.01) (SIGMA_FLOOR of analyze_icl, applied per repeat), mean over the parsed
           repeats of the molecule (a molecule with fewer than three parsed repeats keeps its one or two:
           as for pred and UC1, every molecule with >= 1 parsed repeat is kept)
Per cell (dataset, model, size, draw): Spearman rho of UC1, UC3 and sigma_mix with |e|, every sigma rounded
to 10 decimals before ranking (selective_prediction.TIE_DECIMALS, average ranks for ties), |e| = |pred - truth|
of results/icl/<ds>_permolecule.csv (as selective_prediction.py ranks it); share_uc3_zero =
share of the molecules with >= 2 parsed repeats whose UC3 is exactly 0 (all repeats identical).

Cross-checks (printed; the script stops if one fails):
  pred, truth, uc1, n_rep per molecule equal results/icl/<ds>_permolecule.csv (|diff| <= 1e-9)
  rho_uc1 equals selective_prediction.csv `rho` (same rounding, same |e|) within 1e-6; with the in-memory |e|
  and unrounded sigmas, rho(UC1) and rho(UC3) equal cells.csv rho_UC1_err / rho_UC3_err within 1e-6
  (the output rho_uc1 / rho_uc3 may differ from the unrounded cells.csv values where float noise splits
  ties in sigma or |e|; the largest difference is printed)

Output: results/icl/mixture_sigma.csv (dataset, model, size, draw, n, rho_uc1, rho_uc3, rho_mix, share_uc3_zero)
No API call.

    python src/mixture_sigma.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASET_KEYS, RESULTS            # noqa: E402
from analyze_icl import SIGMA_FLOOR, draw_of        # noqa: E402
from selective_prediction import TIE_DECIMALS       # noqa: E402

ICL = os.path.join(RESULTS, "icl")
OUT = os.path.join(ICL, "mixture_sigma.csv")
FILE_RE = re.compile(r"^(?P<ds>[a-z]+)__(?P<model>[a-z0-9]+)__N(?P<size>\d+)__low(?:__d(?P<draw>\d+))?\.jsonl$")
TOL = 1e-6


def rho(sig, abserr, rounded=True):
    """Spearman rho(sig, |e|); molecules with NaN sigma left out; NaN if sigma is constant."""
    sig, abserr = np.asarray(sig, float), np.asarray(abserr, float)
    ok = np.isfinite(sig) & np.isfinite(abserr)
    sig, abserr = sig[ok], abserr[ok]
    if rounded:
        sig = np.round(sig, TIE_DECIMALS)
    if len(sig) <= 5 or np.ptp(sig) == 0:
        return np.nan
    return float(stats.spearmanr(sig, abserr).correlation)


def per_molecule(path):
    R = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf8") if l.strip()])
    R = R.drop_duplicates(["mol_id", "repeat"], keep="last")
    ok = R[R.parse_ok == True].copy()                                      # noqa: E712
    ok["mol_id"] = ok.mol_id.astype(str)
    ok["s2"] = np.clip(ok.uncertainty.astype(float), SIGMA_FLOOR, None) ** 2
    g = ok.groupby("mol_id").agg(truth=("truth", "first"), pred=("prediction", "mean"), uc1=("uncertainty", "mean"),
                                 uc3var=("prediction", lambda x: x.var(ddof=1) if len(x) > 1 else np.nan),
                                 var0=("prediction", lambda x: x.var(ddof=0)), ms2=("s2", "mean"),
                                 n_rep=("prediction", "size")).reset_index()
    g["uc3"] = np.sqrt(g.uc3var)
    g["mix"] = np.sqrt(g.ms2 + g.var0)
    g["abserr"] = (g.pred - g.truth).abs()
    return g


def main():
    PM = {}
    for ds in DATASET_KEYS:
        P = pd.read_csv(os.path.join(ICL, f"{ds}_permolecule.csv"), dtype={"mol_id": str})
        PM[ds] = P[P.reasoning == "low"].set_index(["model", "size", "draw", "mol_id"]).sort_index()
    C = pd.read_csv(os.path.join(ICL, "cells.csv"))
    C = C[C.reasoning == "low"].set_index(["dataset", "model", "size", "draw"])
    S = pd.read_csv(os.path.join(ICL, "selective_prediction.csv")).rename(columns={"ds": "dataset"})
    S = S[(S.family == "LLM") & (S.reasoning == "low")].set_index(["dataset", "model", "size", "draw"])
    rows, worst = [], dict(pm=0.0, sel=0.0, c1=0.0, c3=0.0, r1=0.0, r3=0.0)
    for f in sorted(glob.glob(os.path.join(ICL, "*__low*.jsonl"))):
        m = FILE_RE.match(os.path.basename(f))
        if not m or m["ds"] not in DATASET_KEYS:
            continue
        ds, model, size, draw = m["ds"], m["model"], int(m["size"]), draw_of(f)
        g = per_molecule(f)
        # (1) per-molecule values = permolecule.csv
        ref = PM[ds].loc[(model, size, draw)].reindex(g.mol_id)
        assert ref.pred.notna().all() and len(ref) == len(PM[ds].loc[(model, size, draw)]), (f, "molecule sets differ")
        d = max(np.max(np.abs(ref[c].values - g[c].values)) for c in ("pred", "truth", "uc1", "n_rep"))
        d = max(d, np.nanmax(np.abs(ref.uc3.values - g.uc3var.values)) if g.uc3var.notna().any() else 0.0)
        worst["pm"] = max(worst["pm"], d)
        # |e| as in the permolecule file (the values every table and figure ranks: |pred - truth| read back from
        # the CSV, whose float parser may move the last bit and so split ties in |e| differently from the
        # in-memory values of analyze_icl); the in-memory |e| reproduces the unrounded cells.csv values
        ae = np.abs(ref.pred.values - ref.truth.values)
        r1, r3, rm = rho(g.uc1, ae), rho(g.uc3, ae), rho(g.mix, ae)
        r1u, r3u = rho(g.uc1, g.abserr, rounded=False), rho(g.uc3var, g.abserr, rounded=False)
        key = (ds, model, size, draw)
        # (2) rho(UC1) = selective_prediction.csv rho (same rounding); unrounded = cells.csv
        worst["sel"] = max(worst["sel"], abs(r1 - S.loc[key, "rho"]))
        worst["c1"] = max(worst["c1"], abs(r1u - C.loc[key, "rho_UC1_err"]))
        worst["c3"] = max(worst["c3"], abs(r3u - C.loc[key, "rho_UC3_err"]))
        worst["r1"] = max(worst["r1"], abs(r1 - C.loc[key, "rho_UC1_err"]))
        worst["r3"] = max(worst["r3"], abs(r3 - C.loc[key, "rho_UC3_err"]))
        multi = g.uc3.notna()
        rows.append(dict(dataset=ds, model=model, size=size, draw=draw, n=len(g), rho_uc1=r1, rho_uc3=r3, rho_mix=rm,
                         share_uc3_zero=float((np.round(g.uc3[multi], TIE_DECIMALS) == 0).mean())))
    R = pd.DataFrame(rows).sort_values(["dataset", "model", "size", "draw"]).reset_index(drop=True)
    print(f"cells: {len(R)} (expected {len(C)} `low` cells in cells.csv)")
    print(f"check per molecule vs permolecule.csv: max |diff| {worst['pm']:.2e}")
    print(f"check rho_uc1 vs selective_prediction.csv rho: max |diff| {worst['sel']:.2e}")
    print(f"check unrounded rho(UC1) vs cells.csv rho_UC1_err: max |diff| {worst['c1']:.2e}")
    print(f"check unrounded rho(UC3) vs cells.csv rho_UC3_err: max |diff| {worst['c3']:.2e}")
    print(f"info: output rho_uc1 / rho_uc3 (sigma rounded, |e| from the CSV) vs the unrounded cells.csv values: max |diff| "
          f"{worst['r1']:.4f} / {worst['r3']:.4f} (ties split by float noise)")
    assert len(R) == len(C), "cell count differs from cells.csv"
    assert worst["pm"] <= 1e-9 and worst["sel"] <= TOL and worst["c1"] <= TOL and worst["c3"] <= TOL, "cross-check failed"
    R.to_csv(OUT, index=False)
    print(f"-> {os.path.relpath(OUT, os.path.dirname(RESULTS))}")
    d1 = R[R.draw == 1]
    for lab, x in (("clean", d1[d1.dataset != "esol"]), ("esol", d1[d1.dataset == "esol"])):
        dm, d3 = x.rho_mix - x.rho_uc1, x.rho_uc3 - x.rho_uc1
        print(f"draw 1 {lab:5s}: rho_mix > rho_uc1 in {int((dm > 0).sum())} of {len(x)} (equal {int((dm == 0).sum())}); "
              f"median rho_mix - rho_uc1 {dm.median():+.3f} (range {dm.min():+.3f} .. {dm.max():+.3f}); "
              f"rho_uc3 > rho_uc1 in {int((d3 > 0).sum())} of {len(x)}; share_uc3_zero {x.share_uc3_zero.min():.3f}-{x.share_uc3_zero.max():.3f}")


if __name__ == "__main__":
    main()
