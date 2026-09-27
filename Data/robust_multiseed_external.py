from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import kruskal

DATA_DIR = Path(__file__).resolve().parent
LABEL_DIR = DATA_DIR / "robust_seed_stability"
OUT_DIR = DATA_DIR / "robust_primary_article"
SEEDS = [0, 7, 42, 123, 2024]
METHODS = ["DEC", "IDEC"]
KEYS = ["IDCNTRY", "IDSCHOOL", "IDSTUD"]
PVS = [f"ASRREA0{i}" for i in range(1, 6)]

student, _ = pyreadstat.read_sav(str(DATA_DIR / "ASGMARR5.sav"), usecols=KEYS + ["TOTWGT"] + PVS)
for c in KEYS:
    student[c] = pd.to_numeric(student[c], errors="coerce").astype("Int64")
for c in ["TOTWGT"] + PVS:
    student[c] = pd.to_numeric(student[c], errors="coerce")


def weighted_eta(y, g, w):
    f = pd.DataFrame({"y": y, "g": g, "w": w}).dropna()
    f = f[f.w > 0]
    grand = np.average(f.y, weights=f.w)
    total = np.sum(f.w * (f.y - grand) ** 2)
    between = 0.0
    for _, p in f.groupby("g"):
        mean = np.average(p.y, weights=p.w)
        between += p.w.sum() * (mean - grand) ** 2
    return between / total


rows = []
for method in METHODS:
    for seed in SEEDS:
        labels = pd.read_csv(LABEL_DIR / method / f"seed_{seed}" / "ids_with_clusters.csv")
        for c in KEYS:
            labels[c] = pd.to_numeric(labels[c], errors="coerce").astype("Int64")
        d = labels.merge(student, on=KEYS, validate="one_to_one")
        for pv in PVS:
            h, p = kruskal(*[x[pv].dropna().to_numpy() for _, x in d.groupby("cluster")])
            rows.append({
                "method": method, "seed": seed, "pv": pv, "H": h, "p": p,
                "weighted_eta2": weighted_eta(d[pv], d.cluster, d.TOTWGT),
            })
results = pd.DataFrame(rows)
summary = results.groupby(["method", "seed"], as_index=False).agg(
    eta2_mean=("weighted_eta2", "mean"),
    eta2_min=("weighted_eta2", "min"),
    eta2_max=("weighted_eta2", "max"),
    H_min=("H", "min"), H_max=("H", "max"), p_max=("p", "max"),
)
results.to_csv(OUT_DIR / "reading_all_seeds.csv", index=False)
summary.to_csv(OUT_DIR / "reading_all_seeds_summary.csv", index=False)
print(summary.to_string(index=False))

