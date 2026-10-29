"""
Digit-level memorization scoring for zero-shot answers, following the memorization paper
(project 45: memodetect.py + label_floor.py + classify.py), applied to an arbitrary molecule
subset (e.g. a 300-molecule test set).

Per (truth, prediction) pair:
  usable   the LABEL carries >= 3 significant figures (denominator of hit3)
  m1/m2/m3 nested matches at 1, 2, 3 significant figures (m3 requires m1 and m2)
Per cell (model x dataset x subset):
  hit3 = m3 / n_usable, R12 = m2 / m1, R23 = m3 / m2 (retentions)
  one-sided exact binomial test of each retention against the label-only floor
  (mode_R12 / mode_R23 from 45/results/label_floor.csv, copied to src/memo45), tested only if
  the conditioning count is >= MIN_COND; Benjamini-Hochberg over all tests in one family;
  verdict: untestable (m1 < MIN_COND) | clean | contaminated, regime trace / partial / heavy.

The floors are those of the full label distribution; for a 300-molecule subset they are a
close approximation (the floor depends on the digit distribution of the labels, which the
representativeness table shows to be stable for Lipophilicity and FreeSolv).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memo45"))
from memodetect import sig_round, sig_figs, benjamini_hochberg  # noqa: E402

MIN_COND = 15
ALPHA = 0.05
HEAVY_HIT3 = 10.0
FLOOR_FILE = os.path.join(os.path.dirname(__file__), "memo45", "label_floor_45.csv")


def floors(dataset):
    f = pd.read_csv(FLOOR_FILE).set_index("dataset").loc[dataset]
    return dict(R12=float(f.mode_R12), R23=float(f.mode_R23), hit3=float(f.floor_hit3),
                hit1=float(f.floor_hit1))


# Dataset-specific answer-unit rules (project decisions 2026-10-15; NOT part of the 45 protocol, which
# scores the parsed number verbatim; everything else of the 45 protocol is unchanged). ESOL labels are
# logS = log10(solubility / mol/L); two panel models (Sonnet 5.5, Haiku 5.5) often report the solubility
# itself in mol/L (e.g. Chloramphenicol: label -2.111, answer 0.00774 = 10^-2.111). The reply text marks
# the unit too inconsistently to decide it per reply, so the rule is a label-sign plausibility rule:
#   `log_units: positive answers are mol/L where the label is negative`
# on ESOL a parsed answer > 0 is converted with log10 ONLY when the molecule's label is negative
# (logS < 0); for the 10 test molecules with a positive label (logS > 0) the answer is kept as stated,
# since the models answer those in logS (e.g. GPT-6 Astra 2-Hydroxypyridine 1.02, Sonnet Pyrimidine 1.1).
# Answers <= 0 and unparsed answers are unchanged. The clean datasets (Lipophilicity logD, BACE and
# antiviral pIC50) have no unit rule. (First version of the same day converted every positive answer;
# its outputs were superseded.)
UNIT_RULES = {"esol": "log_units: positive answers are mol/L where the label is negative"}


def apply_unit_rule(dataset, value, truth):
    """Return (value on the label scale, converted?) for one parsed answer under UNIT_RULES."""
    if (dataset in UNIT_RULES and value is not None and np.isfinite(value) and value > 0
            and truth is not None and truth < 0):
        return float(np.log10(value)), True
    return value, False


def match_flags(truth, pred):
    """Per-pair flags. pred may be None/NaN (unparsed -> miss)."""
    usable = sig_figs(truth) >= 3
    if pred is None or not np.isfinite(pred):
        return dict(usable=usable, parsed=False, m1=False, m2=False, m3=False, abs_err=np.nan)
    m1 = sig_round(pred, 1) == sig_round(truth, 1)
    m2 = m1 and sig_round(pred, 2) == sig_round(truth, 2)
    m3 = m2 and sig_round(pred, 3) == sig_round(truth, 3)
    return dict(usable=usable, parsed=True, m1=m1, m2=m2, m3=m3, abs_err=abs(pred - truth))


def binom_p(k, n, p0):
    if n < 1:
        return np.nan
    p0 = float(min(max(p0 / 100.0, 1e-6), 1 - 1e-6))
    return float(binom.sf(int(k) - 1, int(n), p0))


def cell_stats(df, dataset):
    """df: one row per molecule with columns truth, pred (float or NaN), usable, m1, m2, m3."""
    fl = floors(dataset)
    n = len(df)
    u = df[df.usable]
    n_usable = len(u)
    parsed = int(df.pred.notna().sum())
    m1, m2, m3 = int(u.m1.sum()), int(u.m2.sum()), int(u.m3.sum())
    ok = df.pred.notna()
    rho = spearmanr(df.truth[ok], df.pred[ok]).correlation if ok.sum() > 5 else np.nan
    med_ae = float(np.nanmedian(np.abs(df.pred - df.truth))) if ok.any() else np.nan
    r = dict(n=n, n_parsed=parsed, parse_rate=parsed / max(n, 1), n_usable=n_usable,
             m1=m1, m2=m2, m3=m3,
             hit1=100 * m1 / max(n_usable, 1), hit3=100 * m3 / max(n_usable, 1),
             R12=100 * m2 / m1 if m1 else np.nan, R23=100 * m3 / m2 if m2 else np.nan,
             floor_R12=fl["R12"], floor_R23=fl["R23"], floor_hit3=fl["hit3"],
             p_R12=binom_p(m2, m1, fl["R12"]) if m1 >= MIN_COND else np.nan,
             p_R23=binom_p(m3, m2, fl["R23"]) if m2 >= MIN_COND else np.nan,
             spearman=rho, medAE=med_ae)
    return r


def add_verdicts(cells):
    """BH over all p-values of the table (one family), then verdict/regime per row."""
    cells = cells.copy()
    p = np.concatenate([cells.p_R12.values, cells.p_R23.values])
    q = benjamini_hochberg(p)
    k = len(cells)
    cells["q_R12"], cells["q_R23"] = q[:k], q[k:]
    verdict, regime = [], []
    for _, r in cells.iterrows():
        if r.m1 < MIN_COND:
            verdict.append("untestable" if r.hit1 > r.get("floor_hit1", 0) else "no-signal")
            regime.append("")
            continue
        s12 = np.isfinite(r.q_R12) and r.q_R12 < ALPHA
        s23 = np.isfinite(r.q_R23) and r.q_R23 < ALPHA
        if s12 or s23:
            verdict.append("contaminated")
            if s12 and r.hit3 >= HEAVY_HIT3:
                regime.append("heavy")
            elif s12:
                regime.append("partial")
            else:
                regime.append("trace")
        else:
            verdict.append("clean")
            regime.append("")
    cells["verdict"], cells["regime"] = verdict, regime
    return cells
