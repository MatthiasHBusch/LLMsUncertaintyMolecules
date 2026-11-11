"""
First-pass analysis of the ICL runs (results/icl/*.jsonl): per cell (dataset x model x size x
reasoning) accuracy, uncertainty quality, token and cost accounting.

Per test molecule: prediction = mean over parsed repeats, UC1 = mean self-reported sigma,
UC3 = variance of the per-repeat predictions (ddof=1, needs >= 2 repeats), truth.
Per cell: RMSE, MAE, Spearman(truth, pred); rho(UC1, |err|), rho(UC3, |err|); Gaussian
interval coverage with sigma = UC1 at 68 % and 90 % (PICP), mean interval width, NLL, CRPS;
parse rate, truncation rate, median reasoning tokens, prompt-cache share, cost.
Contamination layer: joins the per-molecule corpus flags and zero-shot hit flags where present,
and reports RMSE / rho(UC1) on the never-recalled subset as extra columns.

Context draws (2026-10-13): the draw is parsed from the file name (`..__<effort>__d<draw>.jsonl`;
no suffix = draw 1, the main split) and stored as column `draw` in cells.csv (after `reasoning`)
and in the per-molecule CSVs (after `reasoning`). A cell is (dataset, model, size, reasoning, draw).
Consumers that want the main experiment only filter draw == 1.

Outputs: results/icl/cells.csv, results/icl/<ds>_permolecule.csv
The full calibration suite (PIT curves, ECE, ENCE, recalibration, Shapley) lives in
src/v1reuse/uq_methods_eval.py and is applied in a second step once cells are complete.

    python src/analyze_icl.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, CONTAM  # noqa: E402

OUT = os.path.join(RESULTS, "icl")
N_TEST = 150
# PICP68 = share of |truth - prediction| <= 1 x sigma (nominal 0.6827 for a Gaussian), PICP90 at
# 1.645 sigma. z = 1.0 (not 0.9945) so that the table means literally "within +/- the reported
# sigma"; a 1e-9 relative tolerance absorbs float artefacts of 2-decimal values (review 2026-10-09).
Z = {68: 1.0, 90: 1.6449}
SIGMA_FLOOR = 0.01       # resolution of the reported values; clipped sigmas are counted per cell
DRAW_RE = re.compile(r"__d(\d+)\.jsonl$")


def draw_of(path):
    """Context draw of a result file: `..__d<k>.jsonl` -> k, no suffix -> 1 (main split)."""
    m = DRAW_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 1


def gaussian_metrics(y, mu, sig):
    n_clipped = int(np.sum(sig < SIGMA_FLOOR))
    sig = np.clip(sig, SIGMA_FLOOR, None)
    z = (y - mu) / sig
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * z ** 2))
    crps = float(np.mean(sig * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))))
    out = dict(NLL=nll, CRPS=crps, n_sigma_clipped=n_clipped)
    for p, zc in Z.items():
        out[f"PICP{p}"] = float(np.mean(np.abs(y - mu) <= zc * sig * (1 + 1e-9)))
        out[f"MPIW{p}"] = float(np.mean(2 * zc * sig))
    return out


def recalled_flags():
    """(dataset, mol_id) -> True if any panel model reproduced the value to 3 significant figures in
    the zero-shot check, False if none did, None if the label is not scorable at 3 figures
    (n_models_hit3 is NaN: labels with < 3 significant figures). None must NOT count as
    'never recalled' (review 2026-10-09: 37 of ESOL's 150 labels are unscorable)."""
    flags = {}
    for f in glob.glob(os.path.join(CONTAM, "zeroshot", "*_permolecule.csv")):
        ds = os.path.basename(f).split("_")[0]
        P = pd.read_csv(f)
        for _, r in P.iterrows():
            v = r.get("n_models_hit3", np.nan)
            flags[(ds, str(r.mol_id))] = None if pd.isna(v) else bool(v > 0)
    return flags


def main():
    files = sorted(glob.glob(os.path.join(OUT, "*.jsonl")))
    if not files:
        sys.exit("no ICL result files yet")
    recalled = recalled_flags()
    cells, per_mol_frames = [], {}
    for f in files:
        rows = [json.loads(l) for l in open(f, encoding="utf8") if l.strip()]
        if not rows:
            continue
        R = pd.DataFrame(rows)
        n_raw = len(R)
        R = R.drop_duplicates(["mol_id", "repeat"], keep="last")   # a gap-fill may append a pair twice
        m = R.iloc[0]
        draw = draw_of(f)
        if "draw" in R.columns and R["draw"].notna().any():
            assert set(R["draw"].dropna().astype(int)) == {draw}, f"{f}: draw field != file-name draw {draw}"
        ok = R[R.parse_ok == True]
        g = ok.groupby("mol_id").agg(truth=("truth", "first"), pred=("prediction", "mean"),
                                     uc1=("uncertainty", "mean"),
                                     uc3=("prediction", lambda x: x.var(ddof=1) if len(x) > 1 else np.nan),
                                     n_rep=("prediction", "size")).reset_index()
        g["abs_err"] = (g.pred - g.truth).abs()
        g["recalled_zeroshot"] = [recalled.get((m.dataset, str(x)), None) for x in g.mol_id]
        y, mu = g.truth.values, g.pred.values
        n_rep_expected = int(R.repeat.max()) + 1
        c = dict(dataset=m.dataset, model=m.model, size=int(m["size"]), reasoning=m.reasoning,  # m.size is Series.size
                 draw=draw, n_calls=len(R), n_duplicates_dropped=n_raw - len(R), n_mol=len(g),
                 coverage=len(g) / N_TEST,                       # molecules answered / 150 (failed calls are absent)
                 parse_rate=len(ok) / len(R),                    # of the stored records
                 call_success_rate=len(R) / (N_TEST * n_rep_expected),
                 trunc_rate=float((R.finish == "length").mean()),
                 RMSE=float(np.sqrt(np.mean((mu - y) ** 2))), MAE=float(np.mean(np.abs(mu - y))),
                 spearman_pred=stats.spearmanr(y, mu).correlation if len(g) > 5 else np.nan,
                 rho_UC1_err=stats.spearmanr(g.uc1, g.abs_err).correlation if len(g) > 5 else np.nan,
                 rho_UC3_err=stats.spearmanr(g.uc3, g.abs_err, nan_policy="omit").correlation
                 if g.uc3.notna().sum() > 5 else np.nan,
                 mean_UC1=float(g.uc1.mean()),
                 **gaussian_metrics(y, mu, g.uc1.values),
                 median_reasoning_tokens=float(R.reasoning_tokens.median()),
                 median_completion_tokens=float(R.completion_tokens.median()),
                 cache_share=float(R.cached_tokens.sum() / max(R.prompt_tokens.sum(), 1)),
                 cost_usd=float(R.cost.sum()))
        nr = g[g.recalled_zeroshot.eq(False)]          # scorable AND never recalled (None excluded)
        rc = g[g.recalled_zeroshot.eq(True)]
        c["n_unscorable_zeroshot"] = int(g.recalled_zeroshot.isna().sum())
        c["n_recalled"] = len(rc)
        if len(nr) > 10:
            c["RMSE_never_recalled"] = float(np.sqrt(np.mean((nr.pred - nr.truth) ** 2)))
            c["rho_UC1_err_never_recalled"] = stats.spearmanr(nr.uc1, nr.abs_err).correlation
            c["PICP68_never_recalled"] = gaussian_metrics(nr.truth.values, nr.pred.values, nr.uc1.values)["PICP68"]
            c["n_never_recalled"] = len(nr)
        if len(rc) > 10:
            c["RMSE_recalled"] = float(np.sqrt(np.mean((rc.pred - rc.truth) ** 2)))
            c["PICP68_recalled"] = gaussian_metrics(rc.truth.values, rc.pred.values, rc.uc1.values)["PICP68"]
        cells.append(c)
        g.insert(0, "draw", draw)
        g.insert(0, "reasoning", m.reasoning)
        g.insert(0, "size", int(m["size"]))
        g.insert(0, "model", m.model)
        per_mol_frames.setdefault(m.dataset, []).append(g)
    C = pd.DataFrame(cells).sort_values(["dataset", "model", "reasoning", "size", "draw"])
    C.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    for ds, frames in per_mol_frames.items():
        pd.concat(frames).to_csv(os.path.join(OUT, f"{ds}_permolecule.csv"), index=False)
    pd.set_option("display.width", 250)
    print(C[["dataset", "model", "size", "reasoning", "draw", "n_mol", "parse_rate", "RMSE", "spearman_pred",
             "rho_UC1_err", "rho_UC3_err", "PICP68", "PICP90", "NLL", "median_reasoning_tokens",
             "cache_share", "cost_usd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
