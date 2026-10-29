"""
Decompress the gzipped raw records to where the analysis scripts read them (and the runners append to
them): results/icl/replies/<cell>.jsonl.gz -> results/icl/<cell>.jsonl, and
results/contamination/corpus/cache.jsonl.gz -> results/contamination/corpus/cache.jsonl.
Run it once before the analysis. Existing files are kept unless --force. The unpacked files are
git-ignored. No API calls.

    python src/unpack_replies.py [--force]
"""
import argparse
import glob
import gzip
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def unpack(src, dst, force):
    if os.path.exists(dst) and not force:
        return 0
    with gzip.open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.replace(dst + ".part", dst)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    a = ap.parse_args()
    icl = os.path.join(ROOT, "results", "icl")
    files = sorted(glob.glob(os.path.join(icl, "replies", "*.jsonl.gz")))
    n = sum(unpack(f, os.path.join(icl, os.path.basename(f)[:-3]), a.force) for f in files)
    print(f"results/icl: {n} of {len(files)} reply files unpacked")
    cache = os.path.join(ROOT, "results", "contamination", "corpus", "cache.jsonl.gz")
    if os.path.exists(cache):
        print(f"corpus cache: {unpack(cache, cache[:-3], a.force)} file unpacked")


if __name__ == "__main__":
    main()
