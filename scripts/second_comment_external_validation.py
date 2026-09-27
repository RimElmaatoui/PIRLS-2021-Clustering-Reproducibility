from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2_contingency, kruskal


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = DATA_DIR / "validation_prioritaire_clusters"
OUT_DIR.mkdir(exist_ok=True)
METHODS = ["DEC", "IDEC"]
PRIMARY_SEED = 0
PVS = [f"ASRREA0{i}" for i in range(1, 6)]
KEYS = ["IDCNTRY", "IDSCHOOL", "IDSTUD"]


def weighted_mean(values, weights):
    valid = values.notna() & weights.notna() & (weights > 0)
    return np.average(values[valid], weights=weights[valid])


def weighted_eta_squared(values, groups, weights):
    valid = values.notna() & groups.notna() & weights.notna() & (weights > 0)
    y, g, w = values[valid], groups[valid], weights[valid]
    grand_mean = np.average(y, weights=w)
    total = np.sum(w * (y - grand_mean) ** 2)
    between = 0.0
    for cluster in sorted(g.unique()):
        mask = g == cluster
        cluster_mean = np.average(y[mask], weights=w[mask])
        between += w[mask].sum() * (cluster_mean - grand_mean) ** 2
    return between / total


required = KEYS + ["TOTWGT", "ASBGHRL", "ASBG03"] + PVS
student, _ = pyreadstat.read_sav(str(DATA_DIR / "ASGMARR5.sav"), usecols=required)
for key in KEYS:
    student[key] = pd.to_numeric(student[key], errors="coerce").astype("Int64")
for variable in ["TOTWGT", "ASBGHRL", "ASBG03"] + PVS:
    student[variable] = pd.to_numeric(student[variable], errors="coerce")

merged_data = {}
profile_rows = []
test_rows = []

for method in METHODS:
    label_path = (
        DATA_DIR
        / f"{method}_results_MAROC_multi_seeds"
        / f"seed_{PRIMARY_SEED}"
        / "ids_with_clusters.csv"
    )
    labels = pd.read_csv(label_path)
    for key in KEYS:
        labels[key] = pd.to_numeric(labels[key], errors="coerce").astype("Int64")
    labels["cluster"] = pd.to_numeric(labels["cluster"], errors="raise").astype(int)
    data = labels[KEYS + ["cluster"]].merge(student, on=KEYS, validate="one_to_one")
    data["reading_pv1"] = data["ASRREA01"]
    data["cluster_display"] = data["cluster"] + 1
    merged_data[method] = data

    total_weight = data["TOTWGT"].sum()
    for cluster, group in data.groupby("cluster"):
        language = group[["ASBG03", "TOTWGT"]].dropna()
        language_weights = language.groupby("ASBG03")["TOTWGT"].sum()
        language_total = language_weights.sum()
        profile_rows.append(
            {
                "method": method,
                "cluster": cluster + 1,
                "n": len(group),
                "weighted_share_pct": 100 * group["TOTWGT"].sum() / total_weight,
                "reading_weighted_mean": np.mean(
                    [weighted_mean(group[pv], group["TOTWGT"]) for pv in PVS]
                ),
                "home_resources_weighted_mean": weighted_mean(
                    group["ASBGHRL"], group["TOTWGT"]
                ),
                "test_language_always_almost_pct": 100
                * (language_weights.get(1.0, 0) + language_weights.get(2.0, 0))
                / language_total,
                "test_language_sometimes_pct": 100
                * language_weights.get(3.0, 0)
                / language_total,
                "test_language_never_pct": 100
                * language_weights.get(4.0, 0)
                / language_total,
            }
        )

    reading_h = []
    reading_p = []
    reading_eta = []
    for pv in PVS:
        groups = [g[pv].dropna().to_numpy() for _, g in data.groupby("cluster")]
        h_stat, p_value = kruskal(*groups)
        reading_h.append(h_stat)
        reading_p.append(p_value)
        reading_eta.append(weighted_eta_squared(data[pv], data["cluster"], data["TOTWGT"]))
    test_rows.append(
        {
            "method": method,
            "outcome": "Overall reading (five plausible values)",
            "test": "Kruskal-Wallis (exploratory)",
            "statistic_range": f"{min(reading_h):.2f}--{max(reading_h):.2f}",
            "p_value_range": f"{min(reading_p):.2e}--{max(reading_p):.2e}",
            "effect_size": f"weighted eta2={np.mean(reading_eta):.3f}",
        }
    )

    hrl_groups = [g["ASBGHRL"].dropna().to_numpy() for _, g in data.groupby("cluster")]
    hrl_h, hrl_p = kruskal(*hrl_groups)
    hrl_eta = weighted_eta_squared(data["ASBGHRL"], data["cluster"], data["TOTWGT"])
    test_rows.append(
        {
            "method": method,
            "outcome": "Home resources for learning",
            "test": "Kruskal-Wallis (exploratory)",
            "statistic_range": f"{hrl_h:.2f}",
            "p_value_range": f"{hrl_p:.2e}",
            "effect_size": f"weighted eta2={hrl_eta:.3f}",
        }
    )

    language = data[["cluster", "ASBG03"]].dropna()
    contingency = pd.crosstab(language["cluster"], language["ASBG03"])
    chi2, p_value, _, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    cramers_v = np.sqrt(
        chi2 / (n * min(contingency.shape[0] - 1, contingency.shape[1] - 1))
    )
    test_rows.append(
        {
            "method": method,
            "outcome": "Test language spoken at home",
            "test": "Chi-square (exploratory)",
            "statistic_range": f"{chi2:.2f}",
            "p_value_range": f"{p_value:.2e}",
            "effect_size": f"Cramer's V={cramers_v:.3f}",
        }
    )


profiles = pd.DataFrame(profile_rows)
tests = pd.DataFrame(test_rows)
profiles.to_csv(OUT_DIR / "cluster_profiles_comment2.csv", index=False)
tests.to_csv(OUT_DIR / "external_tests_comment2.csv", index=False)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for axis, method in zip(axes, METHODS):
    data = merged_data[method]
    values = [
        data.loc[data["cluster_display"] == cluster, "reading_pv1"].dropna().to_numpy()
        for cluster in range(1, 6)
    ]
    axis.boxplot(values, labels=range(1, 6), showfliers=False, patch_artist=True)
    for patch in axis.artists:
        patch.set_facecolor("#9ecae1")
    axis.set_title(method)
    axis.set_xlabel("Cluster")
    axis.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("Overall reading score (PV1)")
fig.suptitle("Held-out reading achievement by cluster (descriptive visualization)")
fig.tight_layout()
fig.savefig(OUT_DIR / "reading_boxplots_comment2.png", dpi=300, bbox_inches="tight")
plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
language_labels = ["Always", "Almost always", "Sometimes", "Never"]
colors = ["#2166ac", "#67a9cf", "#fddbc7", "#b2182b"]
for axis, method in zip(axes, METHODS):
    data = merged_data[method][["cluster_display", "ASBG03", "TOTWGT"]].dropna()
    weighted = data.pivot_table(
        index="cluster_display", columns="ASBG03", values="TOTWGT", aggfunc="sum", fill_value=0
    )
    weighted = weighted.reindex(index=range(1, 6), columns=[1.0, 2.0, 3.0, 4.0], fill_value=0)
    proportions = 100 * weighted.div(weighted.sum(axis=1), axis=0)
    bottom = np.zeros(5)
    for category, label, color in zip(proportions.columns, language_labels, colors):
        values = proportions[category].to_numpy()
        axis.bar(range(1, 6), values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_title(method)
    axis.set_xlabel("Cluster")
    axis.set_xticks(range(1, 6))
axes[0].set_ylabel("Weighted proportion (%)")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
fig.suptitle("Frequency of speaking the test language at home")
fig.tight_layout(rect=[0, 0.1, 1, 1])
fig.savefig(OUT_DIR / "home_language_by_cluster_comment2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(profiles.round(2).to_string(index=False))
print("\n", tests.to_string(index=False))
