"""
Build the candidate test sets and quantify how representative each is of its dataset.

Two rules per dataset:
  last300    the last 300 rows in source-file order (the rule proposed for v2: "late molecules
             are less popular")
  random300  300 rows drawn with numpy default_rng(SEED) -- the neutral alternative

For every rule the script compares the test set with the remaining molecules on: target mean /
SD (Welch t and KS test), SMILES length, heavy-atom count, share of labels with >= 3 significant
figures (needed for the 3-sig-fig memorization test), and the Tanimoto nearest-neighbour
similarity of each test molecule to the remainder (does the tail sit in a different chemical
region?). Output:

  data/test_sets/<ds>_<rule>.csv           mol_id, row_index, smiles, value, smiles_len, n_heavy,
                                           sig_figs, canonical_iso
  results/contamination/test_set_representativeness.csv

    python src/build_test_sets.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memo45"))
from common import DATASET_KEYS, TEST_SETS, CONTAM, load_dataset  # noqa: E402
from memodetect import sig_figs  # noqa: E402

SEED = 20260907
N_TEST = 300


def annotate(df):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    df = df.copy()
    df["smiles_len"] = df.smiles.str.len()
    df["n_heavy"] = [m.GetNumHeavyAtoms() if m else np.nan for m in mols]
    df["sig_figs"] = [sig_figs(v) for v in df.value]
    df["canonical_iso"] = [Chem.MolToSmiles(m) if m else "" for m in mols]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None for m in mols]
    return df, fps


def nn_similarity(fps_test, fps_rest):
    from rdkit import DataStructs
    rest = [f for f in fps_rest if f is not None]
    out = []
    for f in fps_test:
        if f is None:
            out.append(np.nan)
            continue
        out.append(max(DataStructs.BulkTanimotoSimilarity(f, rest)))
    return np.array(out)


def compare(test, rest, fps_t, fps_r):
    r = {}
    for col in ["value", "smiles_len", "n_heavy"]:
        a, b = test[col].dropna().values, rest[col].dropna().values
        r[f"{col}_test_mean"] = a.mean()
        r[f"{col}_rest_mean"] = b.mean()
        r[f"{col}_test_sd"] = a.std(ddof=1)
        r[f"{col}_rest_sd"] = b.std(ddof=1)
        r[f"{col}_ks_p"] = stats.ks_2samp(a, b).pvalue
    r["value_welch_p"] = stats.ttest_ind(test.value, rest.value, equal_var=False).pvalue
    r["frac_sigfigs_ge3_test"] = (test.sig_figs >= 3).mean()
    r["frac_sigfigs_ge3_rest"] = (rest.sig_figs >= 3).mean()
    r["n_sigfigs_ge3_test"] = int((test.sig_figs >= 3).sum())
    nn = nn_similarity(fps_t, fps_r)
    r["nn_tanimoto_test_to_rest_median"] = np.nanmedian(nn)
    r["nn_tanimoto_test_to_rest_mean"] = np.nanmean(nn)
    return r, nn


def main():
    os.makedirs(TEST_SETS, exist_ok=True)
    os.makedirs(CONTAM, exist_ok=True)
    rows = []
    for key in DATASET_KEYS:
        df = load_dataset(key)
        df, fps = annotate(df)
        n = len(df)
        rho_val = stats.spearmanr(df.row_index, df.value).correlation
        rho_len = stats.spearmanr(df.row_index, df.smiles_len).correlation
        is_sorted_smiles = bool((df.smiles.values == np.sort(df.smiles.values)).all())
        sel = {
            "last300": np.arange(n - N_TEST, n),
            "random300": np.sort(np.random.default_rng(SEED).permutation(n)[:N_TEST]),
        }
        for rule, idx in sel.items():
            mask = np.zeros(n, bool)
            mask[idx] = True
            test, rest = df[mask], df[~mask]
            fps_t = [fps[i] for i in np.where(mask)[0]]
            fps_r = [fps[i] for i in np.where(~mask)[0]]
            r, nn = compare(test, rest, fps_t, fps_r)
            out = test[["mol_id", "row_index", "smiles", "value", "smiles_len", "n_heavy",
                        "sig_figs", "canonical_iso"]].copy()
            out["nn_tanimoto_to_rest"] = nn
            f = os.path.join(TEST_SETS, f"{key}_{rule}.csv")
            out.to_csv(f, index=False)
            rows.append(dict(dataset=key, rule=rule, n_total=n, n_test=len(test),
                             spearman_index_value=rho_val, spearman_index_smileslen=rho_len,
                             file_sorted_by_smiles=is_sorted_smiles, **r))
            print(f"{key:14s} {rule:10s} value {r['value_test_mean']:7.3f} vs {r['value_rest_mean']:7.3f} "
                  f"(KS p={r['value_ks_p']:.2g})  len {r['smiles_len_test_mean']:5.1f} vs "
                  f"{r['smiles_len_rest_mean']:5.1f}  >=3sf {r['frac_sigfigs_ge3_test']:.2f} vs "
                  f"{r['frac_sigfigs_ge3_rest']:.2f}  NN-Tanimoto med {r['nn_tanimoto_test_to_rest_median']:.2f}")
        print(f"{'':14s} file order: rho(index,value)={rho_val:+.3f} rho(index,len)={rho_len:+.3f} "
              f"sorted_by_smiles={is_sorted_smiles}")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(CONTAM, "test_set_representativeness.csv"), index=False)
    print("saved", os.path.join(CONTAM, "test_set_representativeness.csv"))


if __name__ == "__main__":
    main()
