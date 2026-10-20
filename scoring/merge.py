"""Merge per-method score files (from score.py) into one table for evaluate.py.

    python scoring/merge.py scores/*.csv --out my_scores.csv.gz [--reference data/scores.csv.gz]

Methods missing from the given files are taken from --reference (e.g. SelfMAD++), so that a subset of the methods
can be rescored and evaluated together with the released scores of the others."""
import argparse
import csv
import gzip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--reference", default="data/scores.csv.gz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with gzip.open(a.reference, "rt") as f:
        ref = list(csv.DictReader(f))
    cols = [k for k in ref[0] if k not in ("dataset", "subset", "image")]
    table = {(r["dataset"], r["subset"], r["image"]): dict(r) for r in ref}
    for path in a.files:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        m = [k for k in rows[0] if k not in ("dataset", "subset", "image")][0]
        for r in rows:
            table[(r["dataset"], r["subset"], r["image"])][m] = r[m]
        print(f"{m}: {len(rows)} scores from {path}")
    with gzip.open(a.out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "subset", "image"] + cols)
        w.writeheader(); w.writerows(table.values())


if __name__ == "__main__":
    main()
