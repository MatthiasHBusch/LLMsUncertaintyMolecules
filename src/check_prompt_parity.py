"""
Prompt parity: the Julia runner (src/julia/Run_ICL.jl, the one that talks to the API) must
produce exactly the conversation the Python reference (src/icl_common.py) describes.

Runs the Julia dry-run with ICL_DUMP for one cell per dataset and diffs the three messages,
for every context draw given (default 1 and 2: ICL_DRAW=<d> on the Julia side,
icl_common.cell_data(..., draw=d) on the Python side). For draws >= 2 it also asserts that the
training block differs from draw 1's (the draw split is really used) and that the system and task
messages are identical to draw 1's (same test molecule, same wording).

    python src/check_prompt_parity.py                # all four datasets at N = 30, draws 1 and 2
    python src/check_prompt_parity.py --draws 1,2,3
"""
import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from common import ALL_KEYS, ROOT  # noqa: E402
from icl_common import build_messages, cell_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", default="1,2")
    draws = [int(d) for d in ap.parse_args().draws.split(",")]
    ok = True
    for draw in draws:
        for key in ALL_KEYS:
            dump = os.path.join(tempfile.gettempdir(), f"icl_dump_{key}_d{draw}.json")
            env = dict(os.environ, ICL_DRYRUN="1", ICL_DATASETS=key, ICL_SIZES="30", ICL_DUMP=dump,
                       ICL_MODELS="opus55", ICL_DRAW=str(draw), ICL_BUDGET="0")
            r = subprocess.run(["julia", os.path.join(ROOT, "src", "julia", "Run_ICL.jl")], env=env,
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{key} draw {draw}: julia failed\n{r.stderr[-2000:]}")
                ok = False
                continue
            jl = json.load(open(dump, encoding="utf8"))
            test, train = cell_data(key, 30, draw=draw)
            py = build_messages(key, train, test.iloc[0].smiles)
            py_txt = [py[0]["content"], py[1]["content"][0]["text"], py[2]["content"]]
            jl_txt = [m["content"] for m in jl]
            same = py_txt == jl_txt and jl[1]["cache"] == "ephemeral"
            extra = ""
            if draw > 1:
                t1, tr1 = cell_data(key, 30, draw=1)
                p1 = build_messages(key, tr1, t1.iloc[0].smiles)
                differs = p1[1]["content"][0]["text"] != py_txt[1]
                rest_same = p1[0]["content"] == py_txt[0] and p1[2]["content"] == py_txt[2]
                n_shared = len({s for s, _ in tr1} & {s for s, _ in train})
                extra = (f"  training block {'differs from' if differs else 'EQUALS'} draw 1 "
                         f"({n_shared} of 30 molecules shared), system/task {'same' if rest_same else 'DIFFERENT'}")
                ok = ok and differs and rest_same
            print(f"{key:14s} draw {draw}  {'IDENTICAL' if same else 'DIFFERENT'}  (system {len(py_txt[0])} chars, "
                  f"training {len(py_txt[1])}, task {len(py_txt[2])}){extra}")
            if not same:
                ok = False
                for name, a, b in zip(["system", "training", "task"], py_txt, jl_txt):
                    if a != b:
                        print(f"--- {name}: python vs julia ---")
                        print("\n".join(list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""))[:40]))
    print("parity", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
