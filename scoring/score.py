"""Score the evaluated images with one FIQA method and write its native scores.

    python scoring/score.py --method diffiqa_r --repo /path/to/DifFIQA --weights "/path/to/diffiqa(r).pth" \
        --images /path/to/aligned --out scores/diffiqa_r.csv

--images is a folder with the aligned 112x112 images as <images>/<dataset>/<subset>/<image>, with the dataset, subset,
and image names of data/scores.csv.gz (e.g. FRLL/MorDIFF/morph_001_03_and_010_03.png). Use scoring/merge.py to combine
the per-method files into one table for evaluate.py."""
import argparse
import csv
import gzip
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import LOADERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(LOADERS))
    ap.add_argument("--repo", required=True, help="clone of the official repository of the method")
    ap.add_argument("--weights", help="checkpoint file (or PFE model folder) of the method")
    ap.add_argument("--clip-rn50", help="CLIB-FIQA only: OpenAI CLIP RN50.pt")
    ap.add_argument("--gpu", type=int, default=0, help="SER-FIQ only: GPU index, -1 for CPU")
    ap.add_argument("--images", required=True)
    ap.add_argument("--list", default=str(Path(__file__).resolve().parents[1] / "data/scores.csv.gz"))
    ap.add_argument("--dataset", help="score only this dataset (FRLL or FEI)")
    ap.add_argument("--subset", help="score only this subset")
    ap.add_argument("--limit", type=int, help="score only the first N images of each subset (quick check)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with gzip.open(a.list, "rt") as f:
        items = [(r["dataset"], r["subset"], r["image"]) for r in csv.DictReader(f)
                 if (a.dataset in (None, r["dataset"])) and (a.subset in (None, r["subset"]))]
    if a.limit:
        seen = {}
        items = [it for it in items if seen.setdefault(it[:2], []).append(it) or len(seen[it[:2]]) <= a.limit]
    paths = [Path(a.images).resolve() / d / s / n for d, s, n in items]
    out = Path(a.out).resolve()                                         # some loaders change the working directory
    extra = {"clip_rn50": a.clip_rn50} if a.method == "clib_fiqa" else {"gpu": a.gpu} if a.method == "ser_fiq" else {}
    score = LOADERS[a.method](str(Path(a.repo).resolve()), a.weights and str(Path(a.weights).resolve()), **extra)
    values = score(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "subset", "image", a.method])
        w.writerows([[d, s, n, repr(float(v))] for (d, s, n), v in zip(items, values)])
    print(f"{a.method}: {len(items)} images -> {out}")


if __name__ == "__main__":
    main()
