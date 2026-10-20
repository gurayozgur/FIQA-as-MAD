"""Metrics of the paper (Sec. III). All functions take direction-normalized scores q = pi_m * s_m, where a larger q
means higher predicted utility, and an image is classified as attack if q <= tau."""
import math
import re
import numpy as np
from sklearn.metrics import roc_curve


def deer(bf, at, w_bf=None, w_at=None):
    """D-EER (%): ROC point closest to MACER = BSCER, i.e. argmin |tpr + fpr - 1| on the default sklearn ROC
    (bona fide = positive class). Returns the attack error at that point. Optional sample weights (bootstrap)."""
    y = np.r_[np.ones(len(bf)), np.zeros(len(at))]
    s = np.r_[bf, at]
    w = None if w_bf is None else np.r_[w_bf, w_at]
    if w is not None:
        keep = w > 0
        y, s, w = y[keep], s[keep], w[keep]
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, sample_weight=w)
    return 100 * fpr[np.nanargmin(np.abs(tpr + fpr - 1.0))]


def threshold(bf, bscer=0.2):
    """tau_m: lowest bona fide score at which BSCER reaches 20% (41/204 on FRLL, 40/200 on FEI)."""
    return np.sort(bf)[int(round(bscer * len(bf))) - 1]


def macer(at, tau):
    """MACER (%): proportion of morphs classified as bona fide (q > tau)."""
    return 100 * np.mean(at > tau)


def bscer(bf, tau):
    """BSCER (%): proportion of bona fide images classified as attack (q <= tau)."""
    return 100 * np.mean(bf <= tau)


def separation(bf, at):
    """Fisher discriminant ratio and the signed separation S = sign(mu_BF - mu_A) * sqrt(FDR) (Eq. 1)."""
    fdr = (bf.mean() - at.mean()) ** 2 / (bf.var() + at.var())
    return fdr, np.sign(bf.mean() - at.mean()) * math.sqrt(fdr)


def identities(dataset, subset, name):
    """Subject identities of an image: one for a bona fide image, the two parents for a morph."""
    if dataset == "FRLL":
        if subset == "BonaFide":
            return (name[:3],)
        m = (re.match(r"(\d{3})_\d\d-vs-(\d{3})", name) or re.match(r"M_(\d{3})_\d\d_(\d{3})_", name)
             or re.match(r"morph_(\d{3})_\d\d_and_(\d{3})", name))
    else:
        if subset == "BonaFide":
            return (name.split("-")[0],)
        m = re.match(r"M_(\d+)-\d+_(\d+)-", name)
    return m.groups()


def bootstrap_mean_deer(data, methods, n_rep=1000, seed=20261013):
    """95% intervals of the mean D-EER over the attacks of each dataset (identity-level bootstrap, Sec. III).
    Each replicate draws identities with replacement, weights a bona fide image by the draws of its identity and a
    morph by the product of the draws of its two identities, and recomputes the D-EER of every attack and their mean.
    The datasets are processed in the order FRLL, FEI with one random generator, as in the paper."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in ("FRLL", "FEI"):
        attacks = data.attacks[c]
        ids = {s: [identities(c, s, n) for n in data.names[(c, s)]] for s in ["BonaFide"] + attacks}
        pool = sorted({i for t in ids["BonaFide"] for i in t})
        reps = []
        for _ in range(n_rep):
            draws = dict(zip(pool, np.bincount(rng.choice(len(pool), len(pool), replace=True), minlength=len(pool))))
            w_bf = np.array([draws[t[0]] for t in ids["BonaFide"]], float)
            w_at = {s: np.array([draws.get(t[0], 0) * draws.get(t[1], 0) for t in ids[s]], float) for s in attacks}
            reps.append([np.mean([deer(data.q(m, c, "BonaFide"), data.q(m, c, s), w_bf, w_at[s])
                                  for s in attacks if w_at[s].sum() > 0]) for m in methods])
        reps = np.array(reps)
        for k, m in enumerate(methods):
            out[(m, c)] = tuple(np.percentile(reps[:, k], [2.5, 97.5]))
    return out
