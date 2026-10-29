"""
Molecular features for the classical baselines (TabPFN, RF) on the v2 splits.

(a) RDKit 2D descriptors -- all of `Descriptors.descList` (an in-house recipe, without its
    quantile normalisation; TabPFN does its own preprocessing). Raw values are computed once
    per dataset for every molecule (label-free) and cached; failures and +-inf become NaN, values
    are clipped to +-1e8. Column selection and imputation are done PER CELL on the TRAINING rows
    only (`cell_descriptors`): a column is dropped if it is constant (or all-NaN) on the N
    training rows or NaN in more than half of them; remaining NaNs (train and test) are filled
    with the column median of the training rows. No label is involved in any of this, and no
    statistic of the test molecules enters the features of a cell.
(b) Morgan fingerprints, radius 2, 2048 bits (bit vector, uint8) for the RF, as in v1.
    The descriptors of (a) also feed `rf_desc` (RF on the TabPFN features).

Caches: results/baselines/cache/<ds>__rdkit2d__rdkit<version>.npz and
        results/baselines/cache/<ds>__morgan_r2_2048__rdkit<version>.npz
(rows in data/<ds>.csv order; `mol_id` stored alongside and checked on load).
The rdkit version is part of both cache names: descList grows between releases (base env 2023.03
vs uc44 env 2026.03), and fingerprints are kept per version for the same provenance reason.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import RESULTS, load_dataset  # noqa: E402

CACHE = os.path.join(RESULTS, "baselines", "cache")
FP_RADIUS, FP_NBITS = 2, 2048
CLIP = 1e8


def _rdkit_version():
    import rdkit
    return rdkit.__version__


def _ids(df):
    return np.array([str(m) for m in df.mol_id], dtype=str)   # unicode, not object (np.load w/o pickle)


def _check_ids(z, df, path):
    ids = z["mol_id"].astype(str)
    if len(ids) != len(df) or not np.array_equal(ids, _ids(df)):
        raise RuntimeError(f"cache {path} does not match data/<ds>.csv row order; delete it")


def descriptor_matrix(ds):
    """Raw RDKit 2D descriptor matrix of all molecules of `ds` (float64, NaN for failures).

    Returns (X, names, mol_ids) in data/<ds>.csv row order."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    df = load_dataset(ds)
    path = os.path.join(CACHE, f"{ds}__rdkit2d__rdkit{_rdkit_version()}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        _check_ids(z, df, path)
        return z["X"], [str(n) for n in z["names"]], z["mol_id"].astype(str)
    names = [n for n, _ in Descriptors.descList]
    X = np.full((len(df), len(names)), np.nan)
    for i, s in enumerate(df.smiles.astype(str)):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        for k, (_, fn) in enumerate(Descriptors.descList):
            try:
                X[i, k] = float(fn(mol))
            except Exception:
                pass
    X = np.where(np.isfinite(X), X, np.nan)
    X = np.clip(X, -CLIP, CLIP)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, X=X, names=np.array(names), mol_id=_ids(df))
    return X, names, _ids(df)


def morgan_matrix(ds):
    """Morgan r=2, 2048-bit fingerprints of all molecules of `ds` (uint8), data/<ds>.csv order."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    df = load_dataset(ds)
    path = os.path.join(CACHE, f"{ds}__morgan_r{FP_RADIUS}_{FP_NBITS}__rdkit{_rdkit_version()}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        _check_ids(z, df, path)
        return z["X"], z["mol_id"].astype(str)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)
    X = np.zeros((len(df), FP_NBITS), dtype=np.uint8)
    for i, s in enumerate(df.smiles.astype(str)):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            X[i] = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, X=X, mol_id=_ids(df))
    return X, _ids(df)


def cell_descriptors(X, names, train_rows, test_rows, max_nan_frac=0.5):
    """Column selection + median imputation fitted on the training rows of one cell.

    X: raw descriptor matrix (all molecules); train_rows / test_rows: integer row indices.
    Returns (X_train, X_test, kept_names). Uses no labels and no test-row statistics."""
    Xtr, Xte = X[train_rows], X[test_rows]
    nan_frac = np.mean(np.isnan(Xtr), axis=0)
    with warnings.catch_warnings():          # all-NaN columns -> NaN stats, dropped below
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(Xtr, axis=0)
        med = np.nanmedian(Xtr, axis=0)
    keep = (nan_frac <= max_nan_frac) & np.isfinite(sd) & (sd > 0)
    Xtr, Xte, med = Xtr[:, keep], Xte[:, keep], med[keep]
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    return Xtr, Xte, [n for n, k in zip(names, keep) if k]


if __name__ == "__main__":
    from common import DATASET_KEYS
    for ds in (sys.argv[1:] or DATASET_KEYS):
        X, names, _ = descriptor_matrix(ds)
        F, _ = morgan_matrix(ds)
        print(f"{ds}: descriptors {X.shape} (NaN cells {int(np.isnan(X).sum())}), morgan {F.shape}, "
              f"rdkit {_rdkit_version()}")
