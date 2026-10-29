"""
Summarise the corpus search from the answer cache (results/contamination/corpus/cache.jsonl),
per dataset x index x subset (whole dataset, last300, random300), at any point of the run.

Per molecule and index:
  complete_primary   verbatim and canonical-isomeric forms both answered
  present_verbatim   published SMILES occurs >= 1 time
  present_smiles     any SMILES form (verbatim / canonical isomeric / canonical non-isomeric)
  present_inchikey   InChIKey occurs (database-dump channel)
  present_name       FreeSolv only: IUPAC name occurs
  row_cooccur        a hit string and the published value co-occur in one document (lower bound)
  max_count_smiles   largest count over SMILES forms (popularity score)
Rates are computed over molecules whose primary forms are complete for that index.

Outputs results/contamination/corpus/
  permolecule_flags.csv        one row per (dataset, molecule, index)
  summary_by_subset.csv        coverage and presence rates per dataset x index x subset
  popularity_by_position.csv   presence rate by decile of file position
  headers.csv                  header / filename string counts per dataset x index

    python src/summarize_corpus.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASET_KEYS, CONTAM, load_dataset, load_test_set  # noqa: E402
from corpus_search import (INDEXES, HEADERS, load_cache, molecule_forms, fmt_value,  # noqa: E402
                           PRIMARY_FORMS)

OUT = os.path.join(CONTAM, "corpus")
SMILES_FORMS = {"verbatim", "canonical_iso", "canonical_noiso"}


def main():
    cache = load_cache()
    print(f"cache: {len(cache)} answers")
    rows, hdr_rows = [], []
    for ds in DATASET_KEYS:
        df = load_dataset(ds)
        df["mol_id"] = df.mol_id.astype(str)
        members = {"v2test150": set(load_test_set(ds, "v2test150").mol_id.astype(str))}
        try:
            members["random300"] = set(load_test_set(ds, "random300").mol_id.astype(str))
        except FileNotFoundError:
            members["random300"] = set(df.mol_id.iloc[np.sort(np.random.default_rng(20260907).permutation(len(df))[:300])])
        members["last300"] = set(df.mol_id.iloc[-300:])
        for idx in INDEXES:
            for h in HEADERS.get(ds, []):
                hdr_rows.append(dict(dataset=ds, index=idx, header=h, count=cache.get((idx, "count", h))))
        for _, r in df.iterrows():
            forms = molecule_forms(ds, r)
            for idx in INDEXES:
                ans = {f: cache.get((idx, "count", s)) for f, s in forms.items()}
                prim = [f for f in PRIMARY_FORMS if f in forms]
                complete = all(ans.get(f) is not None for f in prim) and len(prim) > 0
                smi = {f: ans[f] for f in forms if f in SMILES_FORMS and ans.get(f) is not None}
                hits = [forms[f] for f, c in ans.items() if c]
                row = any(cache.get((idx, "find_cnf", f"{s} AND {fmt_value(r.value)}")) for s in hits)
                rows.append(dict(
                    dataset=ds, mol_id=r.mol_id, row_index=r.row_index, index=idx,
                    in_last300=r.mol_id in members["last300"], in_random300=r.mol_id in members["random300"],
                    in_v2test150=r.mol_id in members["v2test150"],
                    too_short="verbatim" not in forms, complete_primary=complete,
                    n_forms_answered=sum(v is not None for v in ans.values()), n_forms=len(forms),
                    present_verbatim=(ans.get("verbatim") or 0) > 0 if ans.get("verbatim") is not None else np.nan,
                    present_smiles=any(c > 0 for c in smi.values()) if smi else np.nan,
                    max_count_smiles=max(smi.values()) if smi else np.nan,
                    present_inchikey=(ans.get("inchikey") or 0) > 0 if ans.get("inchikey") is not None else np.nan,
                    present_name=(ans.get("name") or 0) > 0 if ans.get("name") is not None else np.nan,
                    row_cooccur=row, value=r.value, smiles=r.smiles))
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(OUT, "permolecule_flags.csv"), index=False)
    pd.DataFrame(hdr_rows).to_csv(os.path.join(OUT, "headers.csv"), index=False)

    summary, deciles = [], []
    for (ds, idx), G in F.groupby(["dataset", "index"]):
        Gq = G[~G.too_short]
        for subset, S in [("all", Gq), ("v2test150", Gq[Gq.in_v2test150]),
                          ("random300", Gq[Gq.in_random300]), ("last300", Gq[Gq.in_last300])]:
            C = S[S.complete_primary]
            summary.append(dict(
                dataset=ds, index=idx, subset=subset, n_mol=len(S), n_complete=len(C),
                pct_complete=100 * len(C) / max(len(S), 1),
                pct_present_verbatim=100 * C.present_verbatim.mean() if len(C) else np.nan,
                pct_present_smiles=100 * C.present_smiles.mean() if len(C) else np.nan,
                n_inchikey_answered=int(C.present_inchikey.notna().sum()),
                pct_present_inchikey=100 * C.present_inchikey.mean() if C.present_inchikey.notna().any() else np.nan,
                pct_present_name=100 * C.present_name.mean() if C.present_name.notna().any() else np.nan,
                pct_row_cooccur=100 * C.row_cooccur.mean() if len(C) else np.nan,
                median_count_if_present=float(C.loc[C.present_smiles == True, "max_count_smiles"].median())
                if (C.present_smiles == True).any() else np.nan))
        C = Gq[Gq.complete_primary].copy()
        if len(C) >= 50:
            C["decile"] = pd.qcut(C.row_index, 10, labels=False, duplicates="drop")
            for d, S in C.groupby("decile"):
                deciles.append(dict(dataset=ds, index=idx, decile=int(d), n=len(S),
                                    pct_present_smiles=100 * S.present_smiles.mean()))
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(OUT, "summary_by_subset.csv"), index=False)
    D = pd.DataFrame(deciles)
    D.to_csv(os.path.join(OUT, "popularity_by_position.csv"), index=False)
    pd.set_option("display.width", 250)
    show = S[S.n_complete > 0]
    print(show[["dataset", "index", "subset", "n_mol", "n_complete", "pct_present_verbatim",
                "pct_present_smiles", "pct_present_inchikey", "pct_present_name", "pct_row_cooccur"]]
          .round(1).to_string(index=False))
    if len(D):
        print()
        print(D.pivot_table(index=["dataset", "index"], columns="decile", values="pct_present_smiles")
              .round(1).to_string())


if __name__ == "__main__":
    main()
