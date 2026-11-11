"""
Score the zero-shot test-set runs (results/contamination/zeroshot/*.json) with the memorization
paper's statistic and write per-cell verdicts and per-molecule recall flags.

Outputs (results/contamination/zeroshot/):
  cells.csv                 model x dataset x arm x rule: n, parse rate, truncation, n_usable,
                            m1/m2/m3, hit3, R12, R23, q-values, verdict, regime, medAE, Spearman,
                            unit_rule / n_unit_converted (answers converted by the dataset's unit rule)
  <ds>_<rule>_permolecule.csv   per molecule: truth, per-model prediction (after the unit rule) and hit3
                            flag; for datasets with a unit rule also the raw parsed answer (_pred_raw)

Unit rule (score_zeroshot.UNIT_RULES, decision 2026-10-15): on ESOL a positive parsed answer is read
as a solubility in mol/L and converted with log10 before scoring when the molecule's label is negative
(logS < 0); answers to the molecules with a positive label are kept as stated (label-sign plausibility
rule). Everything else follows the 45 protocol unchanged.

    python src/analyze_zeroshot_testset.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import CONTAM  # noqa: E402
from score_zeroshot import match_flags, cell_stats, add_verdicts, apply_unit_rule, UNIT_RULES  # noqa: E402

OUT = os.path.join(CONTAM, "zeroshot")


def main():
    files = sorted(glob.glob(os.path.join(OUT, "*.json")))
    if not files:
        sys.exit("no zero-shot result files yet")
    cells, per_mol = [], {}
    for f in files:
        d = json.load(open(f, encoding="utf8"))
        m = d["meta"]
        calls = pd.DataFrame(d["calls"])
        if calls.empty:
            continue
        # dataset-specific unit rule per call (ESOL: positive answer, negative label -> log10), before averaging
        calls["value_raw"] = calls["value"].astype(float)
        conv = [apply_unit_rule(m["dataset"], v, t) for v, t in zip(calls.value_raw, calls.truth)]
        calls["value"] = [c[0] for c in conv]
        calls["unit_converted"] = [c[1] for c in conv]
        # per molecule: mean of parsed values over iterations (1 iteration in the standard arm)
        g = calls.groupby("mol_id").agg(truth=("truth", "first"), pred=("value", "mean"),
                                        pred_raw=("value_raw", "mean"),
                                        n_iter=("value", "size"),
                                        truncated=("truncated", "mean")).reset_index()
        flags = pd.DataFrame([match_flags(t, p) for t, p in zip(g.truth, g.pred)])
        g = pd.concat([g, flags], axis=1)
        st = cell_stats(g, m["dataset"])
        cells.append(dict(dataset=m["dataset"], model=m["tag"], arm=m["arm"], rule=m["rule"],
                          reasoning=m["reasoning"], median_reasoning_tokens=m.get("median_reasoning_tokens"),
                          n_calls=len(calls), n_truncated=int(calls.truncated.sum()),
                          cost_usd=m.get("cost"), unit_rule=UNIT_RULES.get(m["dataset"], ""),
                          n_unit_converted=int(calls.unit_converted.sum()), **st))
        key = (m["dataset"], m["rule"])
        P = per_mol.setdefault(key, g[["mol_id", "truth"]].copy().set_index("mol_id"))
        P[f"{m['arm']}__{m['tag']}_pred"] = g.set_index("mol_id").pred
        if m["dataset"] in UNIT_RULES:
            P[f"{m['arm']}__{m['tag']}_pred_raw"] = g.set_index("mol_id").pred_raw
        P[f"{m['arm']}__{m['tag']}_hit3"] = g.set_index("mol_id").apply(
            lambda r: float(r.m3) if r.usable else np.nan, axis=1)
    C = add_verdicts(pd.DataFrame(cells))
    C.to_csv(os.path.join(OUT, "cells.csv"), index=False)
    for (ds, rule), P in per_mol.items():
        hit_cols = [c for c in P.columns if c.endswith("_hit3")]
        P["n_models_hit3"] = P[hit_cols].sum(axis=1, min_count=1)
        P.reset_index().to_csv(os.path.join(OUT, f"{ds}_{rule}_permolecule.csv"), index=False)
    cols = ["dataset", "model", "arm", "rule", "n", "parse_rate", "n_truncated", "n_usable", "m1",
            "m2", "m3", "hit3", "R12", "R23", "q_R12", "q_R23", "verdict", "regime", "medAE",
            "spearman", "n_unit_converted", "cost_usd"]
    print(C[cols].to_string(index=False))


if __name__ == "__main__":
    main()
