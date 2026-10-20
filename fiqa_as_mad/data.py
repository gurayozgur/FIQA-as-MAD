"""Score table: one row per image (dataset, subset, image) and one column per method with its NATIVE score."""
import csv
import gzip
from collections import defaultdict
import numpy as np
from .methods import ATTACKS, PI


class Scores:
    def __init__(self, path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", newline="") as f:
            reader = csv.DictReader(f)
            self.methods = [k for k in reader.fieldnames if k not in ("dataset", "subset", "image")]
            native = defaultdict(dict)
            for r in reader:
                for m in self.methods:
                    if r[m] != "":
                        native[(m, r["dataset"], r["subset"])][r["image"]] = float(r[m])
        self.attacks = ATTACKS
        # evaluated images: those with a score for every method (Sec. IV), in sorted order of the file names
        keys = {(c, s) for _, c, s in native}
        self.names = {k: sorted(set.intersection(*[set(native[(m,) + k]) for m in self.methods])) for k in keys}
        self._q = {(m,) + k: PI[m] * np.array([native[(m,) + k][n] for n in self.names[k]])
                   for m in self.methods for k in keys}

    def q(self, method, dataset, subset):
        """Direction-normalized scores q = pi_m * s_m of one method on one subset."""
        return self._q[(method, dataset, subset)]
