"""Baselines de reduction de dimension demandes par le reviewer 4.

Ce script compare PCA + KMeans (baseline obligatoire) et UMAP + KMeans
(baseline optionnelle) aux methodes deja calculees. Toutes les comparaisons
utilisent la meme matrice robuste standardisee et K=5.

Sorties: data/dimensionality_reduction_baselines/
"""

from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
INPUT_DIR = DATA_DIR / "robust_primary_article"
LABEL_DIR = DATA_DIR / "Robust_imputation_comparison"
OUT_DIR = DATA_DIR / "dimensionality_reduction_baselines"
ARTICLE_IMAGES = PROJECT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
ARTICLE_IMAGES.mkdir(exist_ok=True)

K = 5
N_COMPONENTS = 16  # meme dimension que l'espace latent AE/DEC/IDEC
SEEDS = [0, 7, 42, 123, 2024]
PRIMARY_SEED = 42
KEYS = ["IDCNTRY", "IDSCHOOL", "IDSTUD"]
PVS = [f"ASRREA0{i}" for i in range(1, 6)]
EXTERNAL_CONTINUOUS = ["ASBGHRL"]
EXTERNAL_CATEGORICAL = ["ASBG03"]


def clustering_metrics(space, labels):
    """Trois indices internes calcules dans un espace donne."""
    return {
        "Silhouette": silhouette_score(space, labels),
        "Calinski_Harabasz": calinski_harabasz_score(space, labels),
        "Davies_Bouldin": davies_bouldin_score(space, labels),
    }


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return np.average(values[valid], weights=weights[valid]) if valid.any() else np.nan


def weighted_eta_squared(values, groups, weights):
    frame = pd.DataFrame({"value": values, "group": groups, "weight": weights}).dropna()
    frame = frame[frame.weight > 0]
    grand = weighted_mean(frame.value, frame.weight)
    total = np.sum(frame.weight * (frame.value - grand) ** 2)
    between = sum(
        part.weight.sum() * (weighted_mean(part.value, part.weight) - grand) ** 2
        for _, part in frame.groupby("group")
    )
    return between / total if total > 0 else np.nan


def cramers_v(table):
    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return chi2, p_value, np.sqrt(chi2 / denominator) if denominator else np.nan


def fit_kmeans_across_seeds(method, reduced_space, original_space):
    """Calcule les metriques propres et communes, puis retourne les labels."""
    rows, fitted_labels = [], {}
    for seed in SEEDS:
        labels = KMeans(n_clusters=K, n_init=20, random_state=seed).fit_predict(reduced_space)
        fitted_labels[seed] = labels
        rows.append({
            "method": method,
            "seed": seed,
            "n_components": reduced_space.shape[1],
            **{f"reduced_{k}": v for k, v in clustering_metrics(reduced_space, labels).items()},
            **{f"original_{k}": v for k, v in clustering_metrics(original_space, labels).items()},
        })
    return pd.DataFrame(rows), fitted_labels


def pairwise_seed_stability(method, fitted_labels):
    rows = []
    seeds = sorted(fitted_labels)
    for pos, seed_a in enumerate(seeds):
        for seed_b in seeds[pos + 1:]:
            rows.append({
                "method": method,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "ARI": adjusted_rand_score(fitted_labels[seed_a], fitted_labels[seed_b]),
            })
    return rows


def load_deep_labels(ids):
    """Charge les labels robustes deja produits pour une comparaison directe."""
    output = {}
    filenames = {
        "AE+KMeans": "labels_robust_median_mode_AE_KMeans.csv",
        "DEC": "labels_robust_median_mode_DEC.csv",
        "IDEC": "labels_robust_median_mode_IDEC.csv",
    }
    for method, filename in filenames.items():
        frame = pd.read_csv(LABEL_DIR / filename)
        joined = ids[["IDSTUD"]].merge(
            frame[["IDSTUD", "cluster"]], on="IDSTUD", validate="one_to_one"
        )
        if len(joined) != len(ids):
            raise ValueError(f"Alignement incomplet pour {method}")
        output[method] = joined.cluster.to_numpy(dtype=int)
    return output


def external_validation(method, ids, labels, student):
    """Validation par variables exclues du clustering: PV, HRL et langue."""
    clustered = ids.copy()
    for key in KEYS:
        clustered[key] = pd.to_numeric(clustered[key], errors="coerce").astype("Int64")
    clustered["cluster"] = labels
    merged = clustered.merge(student, on=KEYS, validate="one_to_one")

    profile_rows = []
    total_weight = merged.TOTWGT.sum()
    for cluster, part in merged.groupby("cluster"):
        language = part[["ASBG03", "TOTWGT"]].dropna()
        language_weights = language.groupby("ASBG03").TOTWGT.sum()
        language_total = language_weights.sum()
        profile_rows.append({
            "method": method,
            "cluster": int(cluster) + 1,
            "n": len(part),
            "weighted_share_pct": 100 * part.TOTWGT.sum() / total_weight,
            "reading_mean_5PV": np.mean([
                weighted_mean(part[pv], part.TOTWGT) for pv in PVS
            ]),
            "HRL_mean": weighted_mean(part.ASBGHRL, part.TOTWGT),
            "language_always_almost_pct": 100 * (
                language_weights.get(1.0, 0) + language_weights.get(2.0, 0)
            ) / language_total,
            "language_sometimes_pct": 100 * language_weights.get(3.0, 0) / language_total,
            "language_never_pct": 100 * language_weights.get(4.0, 0) / language_total,
        })

    reading_h, reading_p, reading_eta = [], [], []
    for pv in PVS:
        groups = [part[pv].dropna().to_numpy() for _, part in merged.groupby("cluster")]
        h_value, p_value = kruskal(*groups)
        reading_h.append(h_value)
        reading_p.append(p_value)
        reading_eta.append(weighted_eta_squared(merged[pv], merged.cluster, merged.TOTWGT))

    hrl_groups = [part.ASBGHRL.dropna().to_numpy() for _, part in merged.groupby("cluster")]
    hrl_h, hrl_p = kruskal(*hrl_groups)
    hrl_eta = weighted_eta_squared(merged.ASBGHRL, merged.cluster, merged.TOTWGT)

    language_valid = merged[["cluster", "ASBG03"]].dropna()
    language_table = pd.crosstab(language_valid.cluster, language_valid.ASBG03)
    language_chi2, language_p, language_v = cramers_v(language_table)

    tests = pd.DataFrame([
        {
            "method": method, "outcome": "Reading (5 plausible values)",
            "statistic_min": min(reading_h), "statistic_max": max(reading_h),
            "p_max": max(reading_p), "effect_size": np.mean(reading_eta),
            "effect_name": "weighted_eta_squared",
        },
        {
            "method": method, "outcome": "Home Resources for Learning",
            "statistic_min": hrl_h, "statistic_max": hrl_h,
            "p_max": hrl_p, "effect_size": hrl_eta,
            "effect_name": "weighted_eta_squared",
        },
        {
            "method": method, "outcome": "Language spoken at home",
            "statistic_min": language_chi2, "statistic_max": language_chi2,
            "p_max": language_p, "effect_size": language_v,
            "effect_name": "Cramers_V",
        },
    ])
    return merged, pd.DataFrame(profile_rows), tests


# 1. Charger la matrice commune robuste deja construite.
X = np.load(INPUT_DIR / "X_robust.npy")
ids = pd.read_csv(INPUT_DIR / "ids.csv")
if len(X) != len(ids):
    raise ValueError("X_robust.npy et ids.csv ne sont pas alignes")

# 2. PCA + KMeans: 16 composantes, comme l'espace latent profond.
pca = PCA(n_components=N_COMPONENTS, svd_solver="full")
X_pca = pca.fit_transform(X)
pca_variance = pd.DataFrame({
    "component": np.arange(1, N_COMPONENTS + 1),
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
})
pca_variance.to_csv(OUT_DIR / "pca_explained_variance.csv", index=False)
np.save(OUT_DIR / "X_pca_16.npy", X_pca)

metric_frames = []
stability_rows = []
all_labels = {}
pca_metrics, pca_labels = fit_kmeans_across_seeds("PCA+KMeans", X_pca, X)
metric_frames.append(pca_metrics)
stability_rows.extend(pairwise_seed_stability("PCA+KMeans", pca_labels))
all_labels["PCA+KMeans"] = pca_labels

# 3. UMAP + KMeans est execute seulement si umap-learn est disponible.
umap_available = importlib.util.find_spec("umap") is not None
if umap_available:
    import umap

    umap_labels = {}
    umap_metric_rows = []
    for seed in SEEDS:
        reducer = umap.UMAP(
            n_components=N_COMPONENTS,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        )
        X_umap = reducer.fit_transform(X)
        labels = KMeans(n_clusters=K, n_init=20, random_state=seed).fit_predict(X_umap)
        umap_labels[seed] = labels
        umap_metric_rows.append({
            "method": "UMAP+KMeans", "seed": seed, "n_components": N_COMPONENTS,
            **{f"reduced_{k}": v for k, v in clustering_metrics(X_umap, labels).items()},
            **{f"original_{k}": v for k, v in clustering_metrics(X, labels).items()},
        })
        if seed == PRIMARY_SEED:
            np.save(OUT_DIR / "X_umap_16_seed42.npy", X_umap)
    metric_frames.append(pd.DataFrame(umap_metric_rows))
    stability_rows.extend(pairwise_seed_stability("UMAP+KMeans", umap_labels))
    all_labels["UMAP+KMeans"] = umap_labels

pd.DataFrame([{
    "baseline": "UMAP+KMeans",
    "available": umap_available,
    "message": "executed" if umap_available else "umap-learn is not installed; PCA baseline executed normally",
}]).to_csv(OUT_DIR / "umap_status.csv", index=False)

# 4. Sauvegarder metriques, labels et stabilite multi-graines.
all_metrics = pd.concat(metric_frames, ignore_index=True)
all_metrics.to_csv(OUT_DIR / "baseline_metrics_all_seeds.csv", index=False)

# Tableau directement comparable: toutes les methodes dans le meme espace X.
existing_common_metrics = pd.read_csv(INPUT_DIR / "primary_input_metrics.csv").rename(columns={
    "Silhouette": "original_Silhouette",
    "Calinski_Harabasz": "original_Calinski_Harabasz",
    "Davies_Bouldin": "original_Davies_Bouldin",
})
baseline_common_metrics = all_metrics.loc[
    all_metrics.seed == PRIMARY_SEED,
    [
        "method", "original_Silhouette", "original_Calinski_Harabasz",
        "original_Davies_Bouldin",
    ],
]
pd.concat([existing_common_metrics, baseline_common_metrics], ignore_index=True).to_csv(
    OUT_DIR / "comparison_all_methods_original_space_seed42.csv", index=False
)

pd.DataFrame([{
    "K": K,
    "n_components": N_COMPONENTS,
    "seeds": ",".join(map(str, SEEDS)),
    "kmeans_n_init": 20,
    "pca_solver": "full",
    "umap_n_neighbors": 30,
    "umap_min_dist": 0.1,
    "umap_available": umap_available,
}]).to_csv(OUT_DIR / "analysis_configuration.csv", index=False)

stability = pd.DataFrame(stability_rows)
stability.to_csv(OUT_DIR / "baseline_pairwise_seed_ari.csv", index=False)
stability.groupby("method").ARI.agg(["mean", "std", "min", "max"]).reset_index().to_csv(
    OUT_DIR / "baseline_seed_stability_summary.csv", index=False
)

for method, labels_by_seed in all_labels.items():
    safe_name = method.lower().replace("+", "_").replace(" ", "_")
    for seed, labels in labels_by_seed.items():
        out = ids.copy()
        out["cluster"] = labels
        out.to_csv(OUT_DIR / f"labels_{safe_name}_seed{seed}.csv", index=False)

# 5. Comparer les partitions PCA/UMAP aux partitions profondes existantes.
deep_labels = load_deep_labels(ids)
partition_rows = []
for baseline, labels_by_seed in all_labels.items():
    labels = labels_by_seed[PRIMARY_SEED]
    for deep_method, deep_partition in deep_labels.items():
        partition_rows.append({
            "baseline": baseline,
            "deep_method": deep_method,
            "seed": PRIMARY_SEED,
            "ARI": adjusted_rand_score(labels, deep_partition),
        })
pd.DataFrame(partition_rows).to_csv(OUT_DIR / "ari_baselines_vs_deep.csv", index=False)

# 6. Validation externe sur le resultat principal (seed 42).
needed = KEYS + ["TOTWGT"] + PVS + EXTERNAL_CONTINUOUS + EXTERNAL_CATEGORICAL
student, _ = pyreadstat.read_sav(str(DATA_DIR / "ASGMARR5.sav"), usecols=needed)
for key in KEYS:
    student[key] = pd.to_numeric(student[key], errors="coerce").astype("Int64")
for column in ["TOTWGT"] + PVS + EXTERNAL_CONTINUOUS + EXTERNAL_CATEGORICAL:
    student[column] = pd.to_numeric(student[column], errors="coerce")

profile_frames, test_frames, external_frames = [], [], {}
for method, labels_by_seed in all_labels.items():
    merged, profiles, tests = external_validation(
        method, ids, labels_by_seed[PRIMARY_SEED], student
    )
    external_frames[method] = merged
    profile_frames.append(profiles)
    test_frames.append(tests)

pd.concat(profile_frames, ignore_index=True).to_csv(
    OUT_DIR / "baseline_external_profiles.csv", index=False
)
pd.concat(test_frames, ignore_index=True).to_csv(
    OUT_DIR / "baseline_external_tests.csv", index=False
)

# 7. Figure principale PCA: variance, projection et score de lecture.
pca_primary_labels = pca_labels[PRIMARY_SEED]
pca_external = external_frames["PCA+KMeans"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].plot(
    pca_variance.component,
    100 * pca_variance.cumulative_explained_variance_ratio,
    marker="o",
)
axes[0].set_xlabel("Number of PCA components")
axes[0].set_ylabel("Cumulative explained variance (%)")
axes[0].grid(alpha=0.25)

axes[1].scatter(
    X_pca[:, 0], X_pca[:, 1], c=pca_primary_labels,
    s=5, cmap="tab10", alpha=0.65, linewidths=0,
)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("PCA+KMeans (K=5)")

reading_values = [
    pca_external.loc[pca_external.cluster == cluster, "ASRREA01"].dropna().to_numpy()
    for cluster in range(K)
]
axes[2].boxplot(reading_values, tick_labels=range(1, K + 1), showfliers=False)
axes[2].set_xlabel("Cluster")
axes[2].set_ylabel("Reading score (PV1)")
axes[2].set_title("Held-out reading achievement")
axes[2].grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(ARTICLE_IMAGES / "pca_kmeans_baseline_validation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Input matrix: {X.shape}")
print(f"PCA retained components: {N_COMPONENTS}")
print(f"PCA cumulative explained variance: {pca_variance.explained_variance_ratio.sum():.4f}")
print(f"UMAP available: {umap_available}")
print("\nPRIMARY BASELINE RESULTS")
print(all_metrics[all_metrics.seed == PRIMARY_SEED].round(5).to_string(index=False))
print(f"\nOutputs written to: {OUT_DIR}")
