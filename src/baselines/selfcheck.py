"""
Self-check of the classical baselines (no API calls, seconds; TabPFN part only in the uc44 env).

1. Cell definition, lipophilicity N=30: training rows are exactly train_order_ids[:30] (same
   order), test rows exactly test_ids, disjoint; labels handed to fit are the 30 training labels.
2. Label poisoning: every label OUTSIDE the cell's 30 training rows (the 150 test labels and all
   unused pool labels) is replaced by random junk and the lipophilicity N=30 cell is refitted
   (trivial, rf, rf_desc, gp_tanimoto, knn_tanimoto, mlp_ens; tabpfn with --tabpfn, chemprop_ens
   with --chemprop). Predictions and sigmas must equal the clean refit bit for bit (1e-6 for
   tabpfn): no test label reaches a fit. The clean refit must also reproduce the stored cell CSV
   (to 1e-9, CSV round trip) when the library versions recorded in its .meta.json match the
   running env (STORED_VERSION_KEYS); otherwise the difference is only reported.
3. Every stored cell (results/baselines/*.csv): 150 rows on exactly test_ids, finite prediction
   and uncertainty, .meta.json training ids == train_order_ids[:size]; reports missing cells of
   the full grid (methods x datasets x ladder sizes, seed 0).

    python src/baselines/selfcheck.py [--tabpfn]          # uc44 env (tabpfn needs it)
    python src/baselines/selfcheck.py --chemprop --only     # base env (chemprop 2.0.0 lives there):
                                                            # poisoning test of chemprop_ens only
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from common import DATASET_KEYS, load_dataset, load_split  # noqa: E402
import run_baselines as rb  # noqa: E402


def check_cell_definition():
    s = load_split("lipophilicity")
    c = rb.Cell("lipophilicity", 30)
    assert c.train_ids == [str(m) for m in s["train_order_ids"][:30]]
    assert c.test_ids == [str(m) for m in s["test_ids"]] and len(c.test_ids) == 150
    assert set(c.df.mol_id.astype(str).values[c.train_rows]) == set(s["train_order_ids"][:30])
    assert set(c.df.mol_id.astype(str).values[c.test_rows]) == set(s["test_ids"])
    assert not set(c.train_ids) & set(c.test_ids)
    assert np.array_equal(c.y_train, c.df.value.to_numpy(float)[c.train_rows])
    print("[ok] lipophilicity N=30: train == train_order_ids[:30], test == test_ids (150), disjoint")


# meta keys whose value must equal the running env for a bitwise stored-cell comparison
STORED_VERSION_KEYS = {"rf": ["sklearn"], "rf_desc": ["sklearn"],
                       "gp_tanimoto": ["rdkit_fp", "numpy", "scipy"], "knn_tanimoto": ["rdkit_fp"],
                       "mlp_ens": ["rdkit_fp", "torch"], "chemprop_ens": ["chemprop", "torch"]}


def _env_versions():
    out = {"numpy": np.__version__}
    for mod, key in [("sklearn", "sklearn"), ("scipy", "scipy"), ("torch", "torch"), ("chemprop", "chemprop"),
                     ("rdkit", "rdkit_fp")]:
        try:
            out[key] = __import__(mod).__version__
        except Exception:
            out[key] = None
    return out


def check_poisoning(methods):
    ds, n = "lipophilicity", 30
    df, split = load_dataset(ds), load_split(ds)
    clean = rb.Cell(ds, n, df, split)
    bad = df.copy()
    outside = np.setdiff1d(np.arange(len(df)), clean.train_rows)   # test rows AND unused pool rows
    bad.loc[outside, "value"] = np.random.default_rng(1).normal(1e3, 1e2, len(outside))
    poisoned = rb.Cell(ds, n, bad, split, check_truth=False)
    assert np.array_equal(poisoned.y_train, clean.y_train)
    for m in methods:
        ref, _ = rb.run_cell(m, clean, 0)
        D, _ = rb.run_cell(m, poisoned, 0)
        tol = 1e-6 if m == "tabpfn" else 0.0
        dp = float(np.max(np.abs(D.prediction.values - ref.prediction.values)))
        du = float(np.max(np.abs(D.uncertainty.values - ref.uncertainty.values)))
        assert list(D.mol_id) == list(ref.mol_id)
        assert dp <= tol and du <= tol, f"{m}: predictions change when test labels change ({dp}, {du})"
        msg = f"[ok] {m}: all non-training labels poisoned -> predictions unchanged (max diff {dp:.1e} / {du:.1e})"
        path = rb.cell_path(m, ds, n, 0)
        if os.path.exists(path):              # and the stored cell is reproduced (CSV round-trip)
            S = pd.read_csv(path, dtype={"mol_id": str})
            ds_ = float(np.max(np.abs(S.prediction.values - ref.prediction.values)))
            meta = json.load(open(path[:-4] + ".meta.json", encoding="utf8"))
            env = _env_versions()
            diff = {k: (meta.get(k), env.get(k)) for k in STORED_VERSION_KEYS.get(m, []) if meta.get(k) != env.get(k)}
            if diff:                                  # fits differ across releases (RF draws, BLAS, fingerprints)
                msg += (f"; stored cell from other versions {diff} (stored, here): "
                        f"not compared bitwise (max diff {ds_:.1e})")
            else:
                assert ds_ <= max(tol, 1e-9), f"{m}: stored cell not reproduced ({ds_})"
                msg += f"; stored cell reproduced (max diff {ds_:.1e})"
        print(msg)


def check_stored_cells():
    files = sorted(glob.glob(os.path.join(rb.OUT, "*__*__N*.csv")))
    seen = set()
    for f in files:
        D = pd.read_csv(f, dtype={"mol_id": str})
        r = D.iloc[0]
        meta = json.load(open(f[:-4] + ".meta.json", encoding="utf8"))
        split = load_split(r.dataset, int(meta.get("draw", 1)))
        test = [str(m) for m in split["test_ids"]]
        assert len(D) == 150 and list(D.mol_id) == test, f"{f}: test ids"
        assert np.isfinite(D.prediction).all() and np.isfinite(D.uncertainty).all(), f"{f}: non-finite"
        assert (D.uncertainty > 0).all(), f"{f}: non-positive sigma"
        assert meta["train_ids"] == [str(m) for m in split["train_order_ids"][:int(r["size"])]], f"{f}: train ids"
        assert not set(meta["train_ids"]) & set(test)
        if int(meta.get("draw", 1)) == 1:
            seen.add((r.method, r.dataset, int(r["size"]), int(r.seed)))
    grid = {(m, d, int(s), 0) for m in rb.METHODS for d in DATASET_KEYS for s in load_split(d)["sizes"]}
    missing = sorted(grid - seen)
    print(f"[ok] {len(files)} stored cells: 150 finite predictions on test_ids, train ids == "
          f"train_order_ids[:N] of the cell's context draw; draw-1 seed-0 grid missing: {len(missing)}"
          + (f" e.g. {missing[:5]}" if missing else ""))


if __name__ == "__main__":
    methods = ["trivial", "rf", "rf_desc", "gp_tanimoto", "knn_tanimoto", "mlp_ens"]
    if "--only" in sys.argv:
        methods = []
    methods += (["tabpfn"] if "--tabpfn" in sys.argv else []) + (["chemprop_ens"] if "--chemprop" in sys.argv else [])
    check_cell_definition()
    check_poisoning(methods)
    check_stored_cells()
    print("selfcheck passed")
