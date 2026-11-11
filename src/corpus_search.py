"""
Search open pre-training corpora for every molecule of the four datasets (infini-gram API,
https://infini-gram.io, free, no key, per-IP throttled to roughly one request per second).

Extends 45/src/corpus_prevalence.py from "60 random molecules, verbatim SMILES, 3 indexes" to
"every molecule, several string forms, 7 indexes", as a PRIORITISED, RESUMABLE job queue:

  indexes   v4_olmo-2-0325-32b-instruct_llama  the COMPLETE pre-, mid- and post-training data
                                               of OLMo-2 32B Instruct: for that open model this
                                               is a direct training-set membership test
            v4_dolma-v1_7_llama (Dolma 1.7, 2.6T tok), v4_dclm-baseline_llama (DCLM, 4.3T),
            v4_olmo-mix-1124_llama (OLMo-2 pre-training mix, 4.6T)
            v4_piletrain_llama (The Pile, 380B), v4_rpj_llama_s4 (RedPajama, 1.4T),
            v4_c4train_llama (C4, 170B)   -- older crawls, lower priority
  forms     verbatim SMILES as published; RDKit canonical isomeric SMILES; RDKit canonical
            non-isomeric SMILES; InChIKey (database-dump channel); FreeSolv additionally the
            IUPAC name (FreeSolv is distributed as name + value tables). Identical strings are
            queried once; strings shorter than MIN_LEN characters are skipped.
  row       after every hit on a SMILES/name form: CNF query "<string> AND <value>" (document
            co-occurrence with the published value). Under-counts (tokenisation), lower bound.
  header    dataset header / filename strings, once per dataset and index.

Priority tiers (a tier finishes before the next starts):
  1  test-set candidates (first 300 of the split permutation), verbatim + canonical isomeric,
     4 main indexes
  2  test-set molecules, InChIKey + canonical non-isomeric, 4 main indexes
  3  test-set molecules, all forms, Pile / RedPajama / C4
  4  remaining molecules, verbatim + canonical isomeric, 4 main indexes
  5  remaining molecules, everything else
About 163k queries in total; at the observed throttle this takes days, tiers 1-3 (all test
molecules, all indexes) roughly 15-20 h. Every answer is cached in
results/contamination/corpus/cache.jsonl; the per-molecule tables are rebuilt from the cache by
src/summarize_corpus.py at any time, so partial progress is always usable.

    python src/corpus_search.py                 # all tiers, adaptive rate
    python src/corpus_search.py --tiers 1,2,3   # test sets only
    python src/corpus_search.py --plan          # count the jobs per tier, no API calls
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASET_KEYS, CONTAM, load_dataset, canonical_forms  # noqa: E402

API = "https://api.infini-gram.io/"
MAIN_INDEXES = ["v4_olmo-2-0325-32b-instruct_llama", "v4_dolma-v1_7_llama",
                "v4_dclm-baseline_llama", "v4_olmo-mix-1124_llama"]
OLD_INDEXES = ["v4_piletrain_llama", "v4_rpj_llama_s4", "v4_c4train_llama"]
INDEXES = MAIN_INDEXES + OLD_INDEXES
INDEX_SIZE_TOK = {"v4_dolma-v1_7_llama": "2.6T", "v4_dclm-baseline_llama": "4.3T",
                  "v4_olmo-mix-1124_llama": "4.6T",
                  "v4_olmo-2-0325-32b-instruct_llama": "full OLMo-2 32B Instruct training data",
                  "v4_piletrain_llama": "0.38T", "v4_rpj_llama_s4": "1.4T",
                  "v4_c4train_llama": "0.17T"}
HEADERS = {   # same strings as project 45
    "freesolv": ["FreeSolv", "SAMPL.csv"],
    "lipophilicity": ["CMPD_CHEMBLID", "Lipophilicity.csv"],
    "bace": ["bace.csv", "ChiralCenterCountAllPossible"],
    "antiviral": ["ASAP Discovery", "antiviral_potency.csv"],
    "esol": ["measured log solubility in mols per litre", "delaney-processed"],
}
SPLIT_SEED = 20260907        # same permutation as make_splits.py
N_CANDIDATES = 300           # first positions of the permutation = test-set candidates
PRIMARY_FORMS = ["verbatim", "canonical_iso"]
SECONDARY_FORMS = ["inchikey", "canonical_noiso", "name"]
MIN_LEN = 8
OUT = os.path.join(CONTAM, "corpus")
CACHE = os.path.join(OUT, "cache.jsonl")

_cache = {}
_stats = {"api_calls": 0, "errors": 0, "t0": time.time()}
_sleep = {"v": 0.3}


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    _cache[(d["index"], d["qt"], d["q"])] = d["r"]
                except Exception:
                    pass
    return _cache


def _store(index, qt, q, r):
    _cache[(index, qt, q)] = r
    with open(CACHE, "a", encoding="utf8") as f:
        f.write(json.dumps(dict(index=index, qt=qt, q=q, r=r)) + "\n")


def query(index, qt, q, max_tries=40):
    """One API call with adaptive pacing: speed up on success, back off on 403/errors."""
    k = (index, qt, q)
    if k in _cache:
        return _cache[k]
    payload = {"index": index, "query_type": qt, "query": q}
    for attempt in range(max_tries):
        try:
            req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            _stats["api_calls"] += 1
            _sleep["v"] = max(0.15, _sleep["v"] * 0.97)
            time.sleep(_sleep["v"])
            if isinstance(r, dict) and "error" in r:
                _store(index, qt, q, None)
                return None
            val = int(r.get("count", r.get("cnt", 0)))
            _store(index, qt, q, val)
            return val
        except Exception:
            _stats["errors"] += 1
            _sleep["v"] = min(30.0, _sleep["v"] * 1.5 + 0.5)
            time.sleep(_sleep["v"])
    return None       # not cached: retried next run


def fmt_value(v):
    return f"{float(v):.10g}"


def molecule_forms(key, row):
    forms = canonical_forms(str(row.smiles))
    if key in ("freesolv", "esol"):       # distributed as name + value tables
        forms["name"] = str(row.mol_id).strip()
    return {f: s for f, s in forms.items() if len(s) >= MIN_LEN}


def build_jobs():
    """[(tier, dataset, mol_id, row_index, value, index, form, string)] in execution order."""
    jobs = []
    for key in DATASET_KEYS:
        df = load_dataset(key)
        import numpy as np
        cand = set(np.random.default_rng(SPLIT_SEED).permutation(len(df))[:N_CANDIDATES])
        df["is_test"] = [i in cand for i in range(len(df))]
        forms = {r.mol_id: molecule_forms(key, r) for _, r in df.iterrows()}
        for _, r in df.iterrows():
            F = forms[r.mol_id]
            prim = [(f, s) for f, s in F.items() if f in PRIMARY_FORMS]
            sec = [(f, s) for f, s in F.items() if f in SECONDARY_FORMS]
            if r.is_test:
                for idx in MAIN_INDEXES:
                    jobs += [(1, key, r.mol_id, r.row_index, r.value, idx, f, s) for f, s in prim]
                    jobs += [(2, key, r.mol_id, r.row_index, r.value, idx, f, s) for f, s in sec]
                for idx in OLD_INDEXES:
                    jobs += [(3, key, r.mol_id, r.row_index, r.value, idx, f, s) for f, s in prim + sec]
            else:
                for idx in MAIN_INDEXES:
                    jobs += [(4, key, r.mol_id, r.row_index, r.value, idx, f, s) for f, s in prim]
                    jobs += [(5, key, r.mol_id, r.row_index, r.value, idx, f, s) for f, s in sec]
                for idx in OLD_INDEXES:
                    jobs += [(5, key, r.mol_id, r.row_index, r.value, idx, f, s) for f, s in prim + sec]
    # stable sort: tier, then dataset order, then molecule order (already in that order)
    jobs.sort(key=lambda j: j[0])
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="1,2,3,4,5")
    ap.add_argument("--plan", action="store_true", help="count jobs, no API calls")
    ap.add_argument("--no-row", action="store_true", help="skip the CNF co-occurrence queries")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    load_cache()
    tiers = {int(t) for t in args.tiers.split(",")}
    jobs = [j for j in build_jobs() if j[0] in tiers]
    todo = [j for j in jobs if (j[5], "count", j[7]) not in _cache]
    print(f"cache {len(_cache)} answers; jobs {len(jobs)}, not yet answered {len(todo)}", flush=True)
    for t in sorted(tiers):
        n_t = sum(1 for j in jobs if j[0] == t)
        n_d = sum(1 for j in todo if j[0] == t)
        print(f"  tier {t}: {n_t} queries, {n_d} open", flush=True)
    if args.plan:
        return
    # headers first (28 queries)
    for key, hs in HEADERS.items():
        for idx in INDEXES:
            for h in hs:
                query(idx, "count", h)
    t0 = time.time()
    n_done = 0
    for tier, key, mol_id, row_index, value, idx, form, s in todo:
        c = query(idx, "count", s)
        if c and not args.no_row:
            query(idx, "find_cnf", f"{s} AND {fmt_value(value)}")
        n_done += 1
        if n_done % 200 == 0:
            el = time.time() - t0
            print(f"  tier {tier} {key:14s} {idx:34s} {n_done}/{len(todo)}  "
                  f"{_stats['api_calls'] / max(el, 1):.2f} api/s  errors {_stats['errors']}  "
                  f"sleep {_sleep['v']:.2f}s  ETA {(len(todo) - n_done) / max(n_done / el, 1e-9) / 3600:.1f} h",
                  flush=True)
    print(f"done. api calls {_stats['api_calls']}, errors {_stats['errors']}, "
          f"{(time.time() - t0) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
