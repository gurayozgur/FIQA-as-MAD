"""Compute all results of the paper from a score table.
All values in results/*.csv are percentages (or S, rho, p) with six decimals; round them once for reporting.

    python evaluate.py                       # uses data/scores.csv.gz and data/fiqa_pauc.csv, writes results/
    python evaluate.py --no-bootstrap        # skip the confidence intervals (about 2 minutes)
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from fiqa_as_mad.data import Scores
from fiqa_as_mad.methods import FIQA, NAME, ATTACKS
from fiqa_as_mad import metrics as M


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/scores.csv.gz")
    ap.add_argument("--pauc", default="data/fiqa_pauc.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-bootstrap", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = Scores(a.scores)
    ref = [m for m in D.methods if m not in FIQA]                      # SelfMAD++ (reference only)
    bf = lambda m, c: D.q(m, c, "BonaFide")

    # ---- per-attack D-EER, S, MACER; means (Tabs. III and IV)
    R = {}
    for c, A in ATTACKS.items():
        for m in FIQA + ref:
            t = M.threshold(bf(m, c))
            for s in A:
                at = D.q(m, c, s)
                R[(m, c, s)] = dict(deer=M.deer(bf(m, c), at), S=M.separation(bf(m, c), at)[1], macer=M.macer(at, t))
            R[(m, c, "mean")] = {k: np.mean([R[(m, c, s)][k] for s in A]) for k in ("deer", "macer")}
        rows = [[NAME[m]] + [f"{R[(m, c, s)]['deer']:.6f}" for s in A] + [f"{R[(m, c, s)]['S']:.6f}" for s in A]
                + [f"{R[(m, c, 'mean')]['deer']:.6f}", f"{R[(m, c, 'mean')]['macer']:.6f}"] for m in ref + FIQA]
        write(out / f"table_{c.lower()}.csv", ["method"] + [f"D-EER {s}" for s in A] + [f"S {s}" for s in A]
              + ["mean D-EER", "mean MACER"], rows)
        write(out / f"macer_per_attack_{c.lower()}.csv", ["method"] + A + ["mean"],
              [[NAME[m]] + [f"{R[(m, c, s)]['macer']:.6f}" for s in A + ["mean"]] for m in ref + FIQA])

    # ---- direction selected with the attack labels (comparison only, not training-free)
    rows = []
    for m in FIQA:
        row = [NAME[m]]
        for c, A in ATTACKS.items():
            plus = [R[(m, c, s)]["deer"] for s in A]
            minus = [M.deer(-bf(m, c), -D.q(m, c, s)) for s in A]
            row += [f"{np.mean(plus):.6f}", f"{min(np.mean(plus), np.mean(minus)):.6f}", int(np.mean(minus) < np.mean(plus)),
                    f"{np.mean(np.minimum(plus, minus)):.6f}"]
        rows.append(row)
    write(out / "direction_selection.csv", ["method"] + [f"{c} {k}" for c in ATTACKS for k in
          ("fixed", "per-method", "per-method inverted", "per-attack")], rows)

    # ---- threshold transfer between the datasets
    rows = []
    for m in FIQA:
        row = [NAME[m]]
        for src, dst in (("FRLL", "FEI"), ("FEI", "FRLL")):
            t = M.threshold(bf(m, src))
            row += [f"{M.bscer(bf(m, dst), t):.6f}", f"{np.mean([M.macer(D.q(m, dst, s), t) for s in ATTACKS[dst]]):.6f}",
                    f"{R[(m, dst, 'mean')]['macer']:.6f}"]
        rows.append(row)
    write(out / "threshold_transfer.csv", ["method"] + [f"{d} {k}" for d in ("FRLL->FEI", "FEI->FRLL")
          for k in ("BSCER", "mean MACER", "in-dataset mean MACER")], rows)

    # ---- per-attack summary
    rows = []
    for c, A in ATTACKS.items():
        for s in A:
            best = min(FIQA, key=lambda m: R[(m, c, s)]["deer"])
            rows.append([c, s, sum(R[(m, c, s)]["S"] < 0 for m in FIQA), NAME[best], f"{R[(best, c, s)]['deer']:.6f}",
                         f"{np.median([R[(m, c, s)]['deer'] for m in FIQA]):.6f}"])
    write(out / "per_attack_summary.csv", ["dataset", "attack", "reversals (S<0)", "lowest D-EER method", "lowest D-EER",
                                           "median D-EER"], rows)

    # ---- FIQA ranking vs MAD ranking (pAUC of the error-versus-discard curve, 14 methods)
    if Path(a.pauc).exists():
        P = list(csv.DictReader(open(a.pauc)))
        meth = sorted({r["method"] for r in P}); cols = [k for k in P[0] if k not in ("fr_model", "method")]
        ranks = {m: [] for m in meth}
        for fr in sorted({r["fr_model"] for r in P}):
            for col in cols:
                v = [(float(r[col]), r["method"]) for r in P if r["fr_model"] == fr and r[col] not in ("", None)]
                if len(v) == len(meth):                                  # combinations with results for all methods
                    for k, (_, m) in enumerate(sorted(v), 1): ranks[m].append(k)
        avg = {m: np.mean(v) for m, v in ranks.items()}
        M14 = [m for m in FIQA if m in avg]
        fiqa_rank = {m: i + 1 for i, m in enumerate(sorted(M14, key=avg.get))}
        rows, summary = [], []
        mad_rank = {c: {m: i + 1 for i, m in enumerate(sorted(M14, key=lambda m: R[(m, c, "mean")]["deer"]))} for c in ATTACKS}
        loo = {}
        for c in ATTACKS:
            rho, p = spearmanr([avg[m] for m in M14], [R[(m, c, "mean")]["deer"] for m in M14])
            summary.append([c, f"{rho:.6f}", f"{p:.6f}", len(ranks[M14[0]])])
            for drop in M14:
                k = [m for m in M14 if m != drop]
                loo[(drop, c)] = spearmanr([avg[m] for m in k], [R[(m, c, "mean")]["deer"] for m in k])[1]
        for m in M14:
            rows.append([NAME[m], f"{avg[m]:.6f}", fiqa_rank[m], mad_rank["FRLL"][m], mad_rank["FEI"][m],
                         f"{loo[(m, 'FRLL')]:.6f}", f"{loo[(m, 'FEI')]:.6f}"])
        write(out / "fiqa_vs_mad_ranking.csv", ["method", "avg FIQA rank", "FIQA rank", "MAD rank FRLL", "MAD rank FEI",
                                                "p without method FRLL", "p without method FEI"], rows)
        write(out / "fiqa_vs_mad_spearman.csv", ["dataset", "rho", "p", "combinations"], summary)

    # ---- bootstrap confidence intervals of the mean D-EER
    if not a.no_bootstrap:
        ci = M.bootstrap_mean_deer(D, FIQA + ref)
        write(out / "confidence_intervals.csv", ["method", "FRLL mean D-EER", "FRLL CI low", "FRLL CI high",
                                                 "FEI mean D-EER", "FEI CI low", "FEI CI high"],
              [[NAME[m]] + sum([[f"{R[(m, c, 'mean')]['deer']:.6f}", f"{ci[(m, c)][0]:.6f}", f"{ci[(m, c)][1]:.6f}"]
                                for c in ATTACKS], []) for m in ref + FIQA])

    # ---- console summary
    for c in ATTACKS:
        best = sorted(FIQA, key=lambda m: R[(m, c, "mean")]["deer"])[:3]
        print(f"{c}: lowest mean D-EER " + ", ".join(f"{NAME[m]} {R[(m, c, 'mean')]['deer']:.2f}%" for m in best))
    print(f"results written to {out}/")


if __name__ == "__main__":
    main()
