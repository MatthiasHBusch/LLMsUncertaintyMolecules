"""
v2 splits: a fixed test set of 150 random molecules per dataset and a completely separate,
ordered training pool (no cross-validation), so every in-context prompt of a dataset x size
cell shares one identical training block (prompt caching).

Test-set rule (decision 2026-09-08):
  * random order = numpy default_rng(SEED).permutation(n), SEED = 20260907 (the same
    permutation that produced the earlier `random300` candidates, so the corpus search already
    covers the first 300 positions);
  * Lipophilicity, BACE, antiviral: walk the permutation and take the first 150 molecules that
    are ABSENT from every searched pre-training corpus (all required infini-gram answers are 0).
    Molecules whose corpus queries are not yet complete are skipped (not eligible yet), so the
    set is provisional until the required tiers of the corpus search have finished;
  * FreeSolv, ESOL (contaminated counterparts): first 150 of the permutation, no corpus filter
    (a corpus-absent subset does not exist there in a meaningful size).
  * The remaining molecules form the training pool, in permutation order. The in-context set of
    size N is the first N of the pool, so the sets are nested across the ladder.
  * Training-pool dedup (decision 2026-10-08, project lead): after the test set is fixed, every
    pool molecule whose InChIKey connectivity block (first 14 characters of the RDKit InChIKey of
    the canonical SMILES) equals that of ANY test molecule is removed from the pool -- exact
    duplicates under another name/SMILES and stereoisomers of test molecules alike. The pool
    order is otherwise unchanged (same seed, same permutation, duplicates dropped). Removed pairs:
    results/contamination/train_pool_dedup.csv. Applies to all four datasets and to every
    consumer of the split (LLM runs and baselines).
  * Ladder: 30 / 60 / 120 / 250 / 500 / 1000, cut at the pool size, plus the full pool as the
    top rung when 1000 does not fit (after dedup: see the printed ladders / split JSONs).
  * Test sets are FINAL (2026-09-09): if data/splits/<ds>_v2.json / data/test_sets/<ds>_v2test150.csv
    exist, the regenerated test_ids must equal the stored ones and the regenerated test-set CSV
    must be byte-identical to the stored file, otherwise the script aborts without writing.
    Since 2026-10-13 the main split (draw 1) is never rewritten either: if data/splits/<ds>_v2.json
    exists, the regenerated split must equal it field by field (abort otherwise); it is only
    written when absent. (The pre-dedup splits of 2026-10-08 are kept as
    data/splits/<ds>_v2_pre_dedup_2026-10-08.json.) Default --require is therefore `all` (the
    final rule); `main4_primary` reproduces the provisional sets of 2026-09-08 and now aborts.
  * Additional context draws (decision 2026-10-13, `--draws 3`): draw 1 = the main split above.
    Draw d = 2, 3 is written to data/splits/<ds>_v2_draw<d>.json with the SAME test_ids, sizes,
    n_train_pool and deduplicated training pool (same molecule set, so the InChIKey-14 dedup
    against the test set holds exactly as for draw 1; it is re-checked), but train_order_ids =
    an independent permutation of that pool:
        seed_draw = int.from_bytes(sha256(f"{SEED}:draw{d}".encode()).digest()[:4], "big")
        order_d   = [pool_1[i] for i in numpy.random.default_rng(seed_draw).permutation(len(pool_1))]
    with pool_1 = draw 1's train_order_ids. Extra fields: "draw", "seed_draw", "draw_rule".
    The first N of order_d are the in-context set of size N (nested across the ladder, as draw 1).
    Existing draw files are asserted identical to the regenerated ones (never rewritten). The
    overlap of the first 30 / 60 / 120 molecules between the draws is printed and written to
    results/meta/context_draw_overlap.csv.

Outputs: data/splits/<ds>_v2.json  {test_ids, train_order_ids, sizes, n_train_pool, meta}
         data/splits/<ds>_v2_draw<d>.json   (--draws >= 2)
         data/test_sets/<ds>_v2test150.csv   (only written if absent; else asserted identical)
         results/contamination/test_set_v2_eligibility.csv   (rewritten only if its content changes)
         results/contamination/train_pool_dedup.csv          (rewritten only if its content changes)
         results/meta/context_draw_overlap.csv               (--draws >= 2)

    python src/make_splits.py                  # final rule (--require all): checks draw 1, writes nothing new
    python src/make_splits.py --draws 3        # + context draws 2 and 3
"""
import argparse
import filecmp
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import ALL_KEYS, DATA, TEST_SETS, CONTAM, load_dataset  # noqa: E402
from corpus_search import (load_cache, molecule_forms, MAIN_INDEXES, INDEXES,  # noqa: E402
                           PRIMARY_FORMS)

SEED = 20260907
N_TEST = 150
LADDER = [30, 60, 120, 250, 500, 1000]
CORPUS_FILTERED = {"lipophilicity", "bace", "antiviral"}
SPLITS = os.path.join(DATA, "splits")
DRAW_RULE = ("train_order_ids = [pool_1[i] for i in numpy.random.default_rng(seed_draw).permutation(len(pool_1))], "
             "pool_1 = draw 1's deduplicated train_order_ids, seed_draw = int.from_bytes(sha256(f'{SEED}:draw{d}')"
             ".digest()[:4], 'big')")
OVERLAP_SIZES = [30, 60, 120]


def draw_seed(draw):
    """Deterministic, documented seed of context draw d >= 2 (sha256 of '20260907:draw<d>', first 4 bytes)."""
    return int.from_bytes(hashlib.sha256(f"{SEED}:draw{int(draw)}".encode()).digest()[:4], "big")


def make_draw(split, draw, df):
    """Context draw `draw` >= 2 of a draw-1 split: same everything, independently permuted training pool."""
    pool = list(split["train_order_ids"])
    seed_d = draw_seed(draw)
    perm = np.random.default_rng(seed_d).permutation(len(pool))
    order = [pool[i] for i in perm]
    out = {}
    for k, v in split.items():
        out[k] = v
        if k == "seed":
            out.update(draw=int(draw), seed_draw=seed_d, draw_rule=DRAW_RULE)
    out["train_order_ids"] = order
    # guarantees: same pool (set and size), no test molecule, no InChIKey-14 collision with the test set
    test = set(split["test_ids"])
    assert len(order) == len(pool) == split["n_train_pool"] and len(set(order)) == len(order)
    assert set(order) == set(pool), "draw pool != draw-1 pool"
    assert not set(order) & test, "draw pool overlaps the test set"
    smi = dict(zip(df.mol_id.astype(str), df.smiles))
    test_keys = {inchikey14(smi[m]) for m in split["test_ids"]} - {None}
    assert not any(inchikey14(smi[m]) in test_keys for m in order), "InChIKey-14 collision with the test set"
    assert all(n <= len(order) for n in out["sizes"])
    return out


def write_if_changed(frame, path):
    """Write a CSV only if its content differs from the file on disk (unchanged outputs stay untouched)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "t.csv")
        frame.to_csv(tmp, index=False)
        if os.path.exists(path) and filecmp.cmp(tmp, path, shallow=False):
            return False
        shutil.copyfile(tmp, path)
        return True


def inchikey14(smiles):
    """Connectivity block of the RDKit InChIKey of the canonical SMILES (None if unparsable)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    m = Chem.MolFromSmiles(Chem.MolToSmiles(m))
    k = Chem.MolToInchiKey(m) if m is not None else ""
    return k[:14] if k else None


def dedup_pool(key, df, test_idx, train_order):
    """Drop pool molecules sharing an InChIKey-14 with any test molecule; keep the order."""
    test_keys = {}
    for i in test_idx:
        k = inchikey14(df.smiles.iloc[i])
        if k:
            test_keys.setdefault(k, []).append(i)
    kept, removed = [], []
    for pos, i in enumerate(train_order):          # pos = 0-based position in the pre-dedup pool
        k = inchikey14(df.smiles.iloc[i])
        if k and k in test_keys:
            for j in test_keys[k]:
                removed.append(dict(dataset=key, train_mol_id=df.mol_id.iloc[i], train_smiles=df.smiles.iloc[i],
                                    test_mol_id=df.mol_id.iloc[j], test_smiles=df.smiles.iloc[j], inchikey14=k,
                                    same_label=bool(np.isclose(df.value.iloc[i], df.value.iloc[j])),
                                    train_value=float(df.value.iloc[i]), test_value=float(df.value.iloc[j]),
                                    pre_dedup_pool_position=pos,
                                    same_smiles=df.smiles.iloc[i] == df.smiles.iloc[j]))
            continue
        kept.append(int(i))
    return kept, removed


def corpus_status(cache, key, row, require):
    """'absent' | 'present' | 'unknown' for one molecule under the requirement level."""
    forms = molecule_forms(key, row)
    if require == "main4_primary":
        idxs, fs = MAIN_INDEXES, [f for f in forms if f in PRIMARY_FORMS]
    else:
        idxs, fs = INDEXES, list(forms)
    if not fs:
        return "unknown"            # SMILES too short to be searched
    answers = [cache.get((i, "count", forms[f])) for i in idxs for f in fs]
    if any(a is None for a in answers):
        return "unknown"
    return "present" if any(a > 0 for a in answers) else "absent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", default="all", choices=["main4_primary", "all"])
    ap.add_argument("--datasets", default=",".join(ALL_KEYS))
    ap.add_argument("--draws", type=int, default=1,
                    help="number of context draws: 1 = main split only; 3 = also <ds>_v2_draw2/3.json")
    args = ap.parse_args()
    assert args.draws >= 1
    os.makedirs(SPLITS, exist_ok=True)
    cache = load_cache()
    elig_rows, dedup_rows, overlap_rows = [], [], []
    for key in args.datasets.split(","):
        df = load_dataset(key)
        df["mol_id"] = df.mol_id.astype(str)
        perm = np.random.default_rng(SEED).permutation(len(df))
        test_idx, checked, n_present, n_unknown = [], 0, 0, 0
        for i in perm:
            if len(test_idx) == N_TEST:
                break
            checked += 1
            if key in CORPUS_FILTERED:
                st = corpus_status(cache, key, df.iloc[i], args.require)
                if st == "present":
                    n_present += 1
                    continue
                if st == "unknown":
                    n_unknown += 1
                    continue
            test_idx.append(int(i))
        complete = len(test_idx) == N_TEST
        test_set = set(test_idx)
        train_order = [int(i) for i in perm if i not in test_set]
        n_pool_pre = len(train_order)
        train_order, removed = dedup_pool(key, df, test_idx, train_order)
        dedup_rows += removed
        pool = len(train_order)
        sizes = [n for n in LADDER if n <= pool]
        if pool < LADDER[-1] and pool not in sizes:
            sizes.append(pool)
        split = dict(dataset=key, seed=SEED, n_total=len(df), n_test=len(test_idx),
                     corpus_filter=key in CORPUS_FILTERED, corpus_requirement=args.require,
                     provisional=(key in CORPUS_FILTERED and args.require != "all") or not complete,
                     positions_checked=checked, excluded_present=n_present,
                     skipped_unknown=n_unknown, sizes=sizes, n_train_pool=pool,
                     train_pool_dedup="inchikey14 vs test set (2026-10-08)",
                     n_train_pool_pre_dedup=n_pool_pre, n_removed_dedup=n_pool_pre - pool,
                     test_ids=[df.mol_id.iloc[i] for i in test_idx],
                     train_order_ids=[df.mol_id.iloc[i] for i in train_order])
        T = df.iloc[test_idx][["mol_id", "row_index", "smiles", "value"]]
        split_path = os.path.join(SPLITS, f"{key}_v2.json")
        test_path = os.path.join(TEST_SETS, f"{key}_v2test150.csv")
        # final test sets: regenerate, then require identity with what is on disk
        if os.path.exists(split_path):
            old_ids = json.load(open(split_path, encoding="utf8"))["test_ids"]
            assert [str(m) for m in old_ids] == split["test_ids"], f"{key}: test_ids changed -- abort"
        if os.path.exists(test_path):
            with tempfile.TemporaryDirectory() as td:
                tmp = os.path.join(td, "t.csv")
                T.to_csv(tmp, index=False)
                assert filecmp.cmp(tmp, test_path, shallow=False), f"{key}: test-set CSV not byte-identical -- abort"
        else:
            T.to_csv(test_path, index=False)
        # draw 1 (main split): never rewritten -- the regenerated split must equal the stored one
        if os.path.exists(split_path):
            stored = json.load(open(split_path, encoding="utf8"))
            diff = sorted(k for k in set(stored) | set(split) if stored.get(k) != split.get(k))
            assert not diff, f"{key}: regenerated draw-1 split differs from {split_path} in {diff} -- abort"
            d1_status = "unchanged (asserted identical)"
        else:
            json.dump(split, open(split_path, "w"), indent=1)
            d1_status = "written"
        orders = {1: split["train_order_ids"]}
        for d in range(2, args.draws + 1):
            sd = make_draw(split, d, df)
            path_d = os.path.join(SPLITS, f"{key}_v2_draw{d}.json")
            if os.path.exists(path_d):
                stored = json.load(open(path_d, encoding="utf8"))
                assert stored == sd, f"{key}: regenerated draw {d} differs from {path_d} -- abort"
                st_d = "unchanged (asserted identical)"
            else:
                json.dump(sd, open(path_d, "w"), indent=1)
                st_d = "written"
            orders[d] = sd["train_order_ids"]
            print(f"{key:14s} draw {d}: seed_draw {sd['seed_draw']}  {os.path.basename(path_d)} {st_d}  "
                  f"first 5: {', '.join(sd['train_order_ids'][:5])}")
        for a_ in sorted(orders):
            for b_ in sorted(orders):
                if b_ <= a_:
                    continue
                row = dict(dataset=key, draw_a=a_, draw_b=b_, n_train_pool=pool)
                for n in OVERLAP_SIZES:
                    row[f"overlap_first{n}"] = len(set(orders[a_][:n]) & set(orders[b_][:n]))
                    row[f"expected_first{n}"] = round(n * n / pool, 2)   # hypergeometric mean n^2 / pool
                overlap_rows.append(row)
        elig_rows.append({k: v for k, v in split.items() if k not in ("test_ids", "train_order_ids")})
        print(f"{key:14s} test {len(test_idx):3d}{'' if complete else ' (INCOMPLETE)'}  "
              f"checked {checked}  excluded(present) {n_present}  skipped(unknown) {n_unknown}  "
              f"pool {pool} (dedup removed {n_pool_pre - pool})  ladder {sizes}  provisional={split['provisional']}  "
              f"draw 1: {d1_status}")
    cols = ["dataset", "train_mol_id", "train_smiles", "test_mol_id", "test_smiles", "inchikey14", "same_label",
            "same_smiles", "train_value", "test_value", "pre_dedup_pool_position"]
    if set(args.datasets.split(",")) == set(ALL_KEYS):       # a partial run never shrinks the all-dataset tables
        for name, frame in [("test_set_v2_eligibility.csv", pd.DataFrame(elig_rows)),
                            ("train_pool_dedup.csv", pd.DataFrame(dedup_rows, columns=cols))]:
            changed = write_if_changed(frame, os.path.join(CONTAM, name))
            print(f"{name}: {'REWRITTEN (content changed)' if changed else 'unchanged'}")
    if overlap_rows:
        O = pd.DataFrame(overlap_rows)
        meta_dir = os.path.join(os.path.dirname(CONTAM), "meta")
        os.makedirs(meta_dir, exist_ok=True)
        O.to_csv(os.path.join(meta_dir, "context_draw_overlap.csv"), index=False)
        print("\noverlap of the first N training molecules between context draws (expected n^2 / pool):")
        print(O.to_string(index=False))


if __name__ == "__main__":
    main()
