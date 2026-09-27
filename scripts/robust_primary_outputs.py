from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyreadstat
from scipy.stats import kruskal, chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
ARTICLE_IMAGES = PROJECT_DIR / "figures"
OUT_DIR = DATA_DIR / "robust_primary_article"
OUT_DIR.mkdir(exist_ok=True)
ARTICLE_IMAGES.mkdir(exist_ok=True)

SEED = 42
K = 5
THRESHOLD = 0.50
LOW_CARDINALITY_MAX = 10
MISSING_CODES = [-99, -98, -97, -96, -95, 999, 9999]
STU_MAX, TCH_MAX, SCH_MAX = 200, 120, 120
METHOD_FILE = {
    "KMeans": "KMeans",
    "Agglomerative": "Agglomerative",
    "GMM": "GMM",
    "AE+KMeans": "AE_KMeans",
    "DEC": "DEC",
    "IDEC": "IDEC",
}
KEYS = ["IDCNTRY", "IDSCHOOL", "IDSTUD"]
PVS = [f"ASRREA0{i}" for i in range(1, 6)]
CONTINUOUS_EXTERNAL = ["ASBGHRL", "ASBGSLR", "ASBGSCR", "ASBGSSB", "ASBGSB"]
CATEGORICAL_EXTERNAL = ["ASBG03", "ASBG06"]


def replace_missing_codes(df):
    out = df.copy()
    features = [c for c in out if c not in {"IDCNTRY", "IDSCHOOL", "IDSTUD", "IDTEACH"}]
    out[features] = out[features].replace(MISSING_CODES, np.nan)
    return out


def is_pv(name):
    u = str(name).upper()
    return bool(re.match(r"^(ASRREA|ASRLIT|ASRINF)0[1-5]$", u)) or "PLAUSIBLE" in u


def select_robust(df, exclude, prefix, cap):
    cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude and not is_pv(c)
    ]
    cols = [
        c for c in cols
        if df[c].isna().mean() <= THRESHOLD and df[c].nunique(dropna=True) > 1
    ][:cap]
    block = df[cols].copy()
    block.columns = [prefix + c for c in cols]
    return block


def infer_discrete(block):
    result = []
    for c in block:
        x = block[c].dropna().to_numpy(float)
        if len(x) and np.all(np.isclose(x, np.round(x))) and pd.Series(x).nunique() <= LOW_CARDINALITY_MAX:
            result.append(c)
    return result


def first_mode(s):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else np.nan


def aggregate_school(part, features, discrete):
    rules = {c: (first_mode if c in discrete else "mean") for c in features}
    return part.groupby(["IDCNTRY", "IDSCHOOL"], as_index=False).agg(rules)


def prepare_robust_matrix():
    stu = replace_missing_codes(pd.read_csv(DATA_DIR / "ASPMARR5.csv", low_memory=False))
    tch = replace_missing_codes(pd.read_csv(DATA_DIR / "ATGMARR5.csv", low_memory=False))
    sch = replace_missing_codes(pd.read_csv(DATA_DIR / "ACGMARR5.csv", low_memory=False))

    stu_b = select_robust(stu, {"IDCNTRY", "IDSCHOOL", "IDSTUD"}, "STU_", STU_MAX)
    tch_b = select_robust(tch, {"IDCNTRY", "IDSCHOOL", "IDTEACH"}, "TCH_", TCH_MAX)
    sch_b = select_robust(sch, {"IDCNTRY", "IDSCHOOL"}, "SCH_", SCH_MAX)
    discrete = infer_discrete(stu_b) + infer_discrete(tch_b) + infer_discrete(sch_b)

    stu_p = pd.concat([stu[KEYS].reset_index(drop=True), stu_b.reset_index(drop=True)], axis=1)
    tch_p = pd.concat([tch[["IDCNTRY", "IDSCHOOL"]].reset_index(drop=True), tch_b.reset_index(drop=True)], axis=1)
    sch_p = pd.concat([sch[["IDCNTRY", "IDSCHOOL"]].reset_index(drop=True), sch_b.reset_index(drop=True)], axis=1)
    tch_s = aggregate_school(tch_p, list(tch_b), infer_discrete(tch_b))
    sch_s = aggregate_school(sch_p, list(sch_b), infer_discrete(sch_b))
    merged = stu_p.merge(tch_s, on=["IDCNTRY", "IDSCHOOL"], how="left").merge(
        sch_s, on=["IDCNTRY", "IDSCHOOL"], how="left"
    )
    ids = merged[KEYS].copy()
    raw = merged.drop(columns=KEYS)
    keep = [
        c for c in raw
        if raw[c].isna().mean() <= THRESHOLD and raw[c].nunique(dropna=True) > 1
    ]
    raw = raw[keep].copy()
    discrete = [c for c in discrete if c in raw]
    continuous = [c for c in raw if c not in discrete]
    completed = raw.copy()
    completed[continuous] = SimpleImputer(strategy="median").fit_transform(completed[continuous])
    completed[discrete] = SimpleImputer(strategy="most_frequent").fit_transform(completed[discrete])
    X = StandardScaler().fit_transform(completed).astype(np.float32)
    assert not np.isnan(X).any()
    assert not any(is_pv(c.replace("STU_", "").replace("TCH_", "").replace("SCH_", "")) for c in raw)
    return ids, raw, completed, X, discrete, continuous


def metrics(X, labels):
    return {
        "Silhouette": silhouette_score(X, labels),
        "Calinski_Harabasz": calinski_harabasz_score(X, labels),
        "Davies_Bouldin": davies_bouldin_score(X, labels),
    }


def weighted_mean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return np.average(x[ok], weights=w[ok]) if ok.any() else np.nan


def weighted_eta_squared(y, groups, weights):
    f = pd.DataFrame({"y": y, "g": groups, "w": weights}).dropna()
    f = f[f.w > 0]
    grand = weighted_mean(f.y, f.w)
    total = np.sum(f.w * (f.y - grand) ** 2)
    between = 0.0
    for _, p in f.groupby("g"):
        between += p.w.sum() * (weighted_mean(p.y, p.w) - grand) ** 2
    return between / total if total else np.nan


def pooled_pv_ci(d, method, reps=300, random_state=2026):
    rng = np.random.default_rng(random_state)
    schools = d.IDSCHOOL.dropna().unique()
    clusters = sorted(d.cluster.unique())
    q = {(c, pv): weighted_mean(g[pv], g.TOTWGT) for c, g in d.groupby("cluster") for pv in PVS}
    boot = {(c, pv): [] for c in clusters for pv in PVS}
    for _ in range(reps):
        sampled = rng.choice(schools, size=len(schools), replace=True)
        counts = pd.Series(sampled).value_counts()
        mult = d.IDSCHOOL.map(counts).fillna(0).to_numpy()
        wb = d.TOTWGT.to_numpy() * mult
        for c in clusters:
            mask = d.cluster.to_numpy() == c
            for pv in PVS:
                boot[(c, pv)].append(weighted_mean(d.loc[mask, pv].to_numpy(), wb[mask]))
    rows = []
    for c in clusters:
        est = np.array([q[(c, pv)] for pv in PVS])
        within = np.array([np.nanvar(boot[(c, pv)], ddof=1) for pv in PVS])
        mean = est.mean()
        total_var = within.mean() + (1 + 1 / len(PVS)) * est.var(ddof=1)
        se = np.sqrt(total_var)
        part = d[d.cluster == c]
        rows.append({
            "method": method, "cluster": c + 1, "n": len(part),
            "reading_mean": mean, "ci95_low": mean - 1.96 * se,
            "ci95_high": mean + 1.96 * se,
        })
    return pd.DataFrame(rows)


ids, raw, completed, X, discrete, continuous = prepare_robust_matrix()
np.save(OUT_DIR / "X_robust.npy", X)
ids.to_csv(OUT_DIR / "ids.csv", index=False)
pd.DataFrame({
    "variable": raw.columns,
    "type": ["discrete" if c in discrete else "continuous" for c in raw],
    "missing_rate": raw.isna().mean().values,
}).to_csv(OUT_DIR / "variables.csv", index=False)
print(f"Robust matrix: {X.shape}; discrete={len(discrete)}; continuous={len(continuous)}")

labels = {}
label_root = DATA_DIR / "Robust_imputation_comparison"
for method, file_method in METHOD_FILE.items():
    p = label_root / f"labels_robust_median_mode_{file_method}.csv"
    d = pd.read_csv(p)
    joined = ids[["IDSTUD"]].merge(d[["IDSTUD", "cluster"]], on="IDSTUD", validate="one_to_one")
    assert len(joined) == len(ids)
    labels[method] = joined.cluster.to_numpy(int)

metric_rows = []
profile_rows = []
size_rows = []
for method, lab in labels.items():
    row = {"method": method, **metrics(X, lab)}
    metric_rows.append(row)
    for c in sorted(np.unique(lab)):
        part = X[lab == c]
        profile_rows.append({
            "method": method, "cluster": c + 1,
            "signed_mean": part.mean(), "absolute_mean": np.abs(part).mean(),
        })
        size_rows.append({"method": method, "cluster": c + 1, "n": int((lab == c).sum())})
pd.DataFrame(metric_rows).to_csv(OUT_DIR / "primary_input_metrics.csv", index=False)
try:
    pd.DataFrame(profile_rows).to_csv(OUT_DIR / "primary_feature_profiles.csv", index=False)
except PermissionError:
    # Keep the complete run reproducible even when the main CSV is open in Excel/VS Code.
    pd.DataFrame(profile_rows).to_csv(OUT_DIR / "primary_feature_profiles_current.csv", index=False)
    print("primary_feature_profiles.csv is locked; wrote primary_feature_profiles_current.csv instead.")
pd.DataFrame(size_rows).to_csv(OUT_DIR / "primary_cluster_sizes.csv", index=False)

# Classical K sensitivity and configuration sensitivity on the robust primary matrix.
ksweep = []
for k in [3, 4, 5]:
    fitted = {
        "KMeans": KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(X),
        "Agglomerative-average": AgglomerativeClustering(n_clusters=k, linkage="average", metric="euclidean").fit_predict(X),
        "GMM-full": GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4, max_iter=500, random_state=SEED).fit_predict(X.astype(float)),
    }
    for method, lab in fitted.items():
        ksweep.append({"method": method, "K": k, **metrics(X, lab)})
pd.DataFrame(ksweep).to_csv(OUT_DIR / "classical_k_sweep.csv", index=False)

variants = []
for linkage in ["average", "complete", "ward"]:
    agg = AgglomerativeClustering(n_clusters=K, linkage=linkage, metric="euclidean")
    lab = agg.fit_predict(X)
    variants.append({"method": "Agglomerative", "configuration": linkage, **metrics(X, lab)})
for cov in ["full", "diag", "spherical", "tied"]:
    lab = GaussianMixture(n_components=K, covariance_type=cov, reg_covar=1e-4, max_iter=500, random_state=SEED).fit_predict(X.astype(float))
    variants.append({"method": "GMM", "configuration": cov, **metrics(X, lab)})
pd.DataFrame(variants).to_csv(OUT_DIR / "classical_variants.csv", index=False)

# One common t-SNE of X, colored by each method.
embedding = TSNE(n_components=2, perplexity=35, random_state=SEED, init="pca", learning_rate="auto").fit_transform(X)
np.save(OUT_DIR / "tsne_X.npy", embedding)
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
display_names = {"KMeans": "KMeans", "Agglomerative": "Agglomerative", "GMM": "GMM",
                 "AE+KMeans": "AE+k-means", "DEC": "DEC", "IDEC": "IDEC"}
for methods, filename in [
    (["KMeans", "Agglomerative", "GMM"], "robust_tsne_classical.png"),
    (["DEC", "IDEC", "AE+KMeans"], "robust_tsne_deep_input.png"),
]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, method in zip(axes, methods):
        ax.set_facecolor("white")
        for cluster in range(K):
            mask = labels[method] == cluster
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=9,
                       color=cluster_colors[cluster], alpha=.82,
                       linewidths=0, label=f"C{cluster}")
        name = display_names[method]
        ax.set_title(f"{name} (t-SNE)", fontsize=15, fontweight="bold", pad=23)
        ax.text(.5, 1.01, f"t-SNE — PIRLS Morocco ({name})", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("t-SNE 1", fontsize=8)
        ax.set_ylabel("t-SNE 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(title="Cluster", loc="upper right", fontsize=6, title_fontsize=7,
                  markerscale=1.6, frameon=True)
        for spine in ax.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(.8)
    fig.tight_layout(pad=2.2, w_pad=2.5)
    fig.savefig(ARTICLE_IMAGES / filename, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        get_ipython()
    except NameError:
        pass
    else:
        from IPython.display import display
        display(fig)
    plt.close(fig)

# External held-out variables.
required = KEYS + ["TOTWGT"] + PVS + CONTINUOUS_EXTERNAL + CATEGORICAL_EXTERNAL
student, _ = pyreadstat.read_sav(str(DATA_DIR / "ASGMARR5.sav"), usecols=required)
for c in KEYS:
    student[c] = pd.to_numeric(student[c], errors="coerce").astype("Int64")
for c in ["TOTWGT"] + PVS + CONTINUOUS_EXTERNAL + CATEGORICAL_EXTERNAL:
    student[c] = pd.to_numeric(student[c], errors="coerce")

external_data = {}
ci_frames = []
external_profiles = []
external_tests = []
continuous_effects = []
categorical_tests = []
for method in ["DEC", "IDEC"]:
    d = ids.copy()
    for c in KEYS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int64")
    d["cluster"] = labels[method]
    d = d.merge(student, on=KEYS, validate="one_to_one")
    external_data[method] = d
    ci_frames.append(pooled_pv_ci(d, method))
    total_w = d.TOTWGT.sum()
    for c, g in d.groupby("cluster"):
        lang = g[["ASBG03", "TOTWGT"]].dropna()
        lw = lang.groupby("ASBG03").TOTWGT.sum()
        lt = lw.sum()
        external_profiles.append({
            "method": method, "cluster": c + 1, "n": len(g),
            "weighted_share_pct": 100 * g.TOTWGT.sum() / total_w,
            "reading_mean": np.mean([weighted_mean(g[pv], g.TOTWGT) for pv in PVS]),
            "hrl_mean": weighted_mean(g.ASBGHRL, g.TOTWGT),
            "always_almost_pct": 100 * (lw.get(1.0, 0) + lw.get(2.0, 0)) / lt,
            "sometimes_pct": 100 * lw.get(3.0, 0) / lt,
            "never_pct": 100 * lw.get(4.0, 0) / lt,
        })
    hs, ps, etas = [], [], []
    for pv in PVS:
        h, p = kruskal(*[g[pv].dropna().to_numpy() for _, g in d.groupby("cluster")])
        hs.append(h); ps.append(p); etas.append(weighted_eta_squared(d[pv], d.cluster, d.TOTWGT))
    external_tests.append({
        "method": method, "outcome": "Reading", "stat_min": min(hs), "stat_max": max(hs),
        "p_max": max(ps), "effect": np.mean(etas),
    })
    for variable in CONTINUOUS_EXTERNAL:
        h, p = kruskal(*[g[variable].dropna().to_numpy() for _, g in d.groupby("cluster")])
        eta = weighted_eta_squared(d[variable], d.cluster, d.TOTWGT)
        continuous_effects.append({"method": method, "variable": variable, "H": h, "p": p, "weighted_eta2": eta})
    for variable in CATEGORICAL_EXTERNAL:
        valid = d[["cluster", variable]].dropna()
        ct = pd.crosstab(valid.cluster, valid[variable])
        chi2, p, _, _ = chi2_contingency(ct)
        n = ct.to_numpy().sum()
        denom = n * min(ct.shape[0] - 1, ct.shape[1] - 1)
        v = np.sqrt(chi2 / denom) if denom else np.nan
        categorical_tests.append({"method": method, "variable": variable, "chi2": chi2, "p": p, "cramers_v": v, "n": n})

ci = pd.concat(ci_frames, ignore_index=True)
profiles = pd.DataFrame(external_profiles)
tests = pd.DataFrame(external_tests)
cont = pd.DataFrame(continuous_effects)
cat = pd.DataFrame(categorical_tests)
ci.to_csv(OUT_DIR / "reading_pooled_ci.csv", index=False)
profiles.to_csv(OUT_DIR / "external_profiles.csv", index=False)
tests.to_csv(OUT_DIR / "reading_tests.csv", index=False)
cont.to_csv(OUT_DIR / "external_continuous_effects.csv", index=False)
cat.to_csv(OUT_DIR / "external_categorical_tests.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, method in zip(axes, ["DEC", "IDEC"]):
    d = external_data[method]
    values = [d.loc[d.cluster == c, "ASRREA01"].dropna().to_numpy() for c in range(K)]
    ax.boxplot(values, labels=range(1, K + 1), showfliers=False, patch_artist=True)
    ax.set_title(method); ax.set_xlabel("Cluster"); ax.grid(axis="y", alpha=.25)
axes[0].set_ylabel("Overall reading score (PV1)")
fig.suptitle("Held-out reading achievement by robust primary cluster")
fig.tight_layout()
fig.savefig(ARTICLE_IMAGES / "robust_reading_boxplots.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
language_labels = ["Always", "Almost always", "Sometimes", "Never"]
colors = ["#2166ac", "#67a9cf", "#fddbc7", "#b2182b"]
for ax, method in zip(axes, ["DEC", "IDEC"]):
    d = external_data[method][["cluster", "ASBG03", "TOTWGT"]].dropna()
    weighted = d.pivot_table(index="cluster", columns="ASBG03", values="TOTWGT", aggfunc="sum", fill_value=0)
    weighted = weighted.reindex(index=range(K), columns=[1.0, 2.0, 3.0, 4.0], fill_value=0)
    props = 100 * weighted.div(weighted.sum(axis=1), axis=0)
    bottom = np.zeros(K)
    for category, label, color in zip(props.columns, language_labels, colors):
        vals = props[category].to_numpy()
        ax.bar(range(1, K + 1), vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_title(method); ax.set_xlabel("Cluster"); ax.set_xticks(range(1, K + 1))
axes[0].set_ylabel("Weighted proportion (%)")
handles, legend_labels = axes[1].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
fig.suptitle("Frequency of speaking the test language at home")
fig.tight_layout(rect=[0, .1, 1, 1])
fig.savefig(ARTICLE_IMAGES / "robust_home_language.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nPRIMARY METRICS\n", pd.DataFrame(metric_rows).round(5).to_string(index=False))
print("\nPROFILES\n", profiles.round(2).to_string(index=False))
print("\nREADING TESTS\n", tests.to_string(index=False))
print("\nCONTINUOUS EFFECTS\n", cont.round(5).to_string(index=False))
print("\nCATEGORICAL TESTS\n", cat.round(5).to_string(index=False))
print("\nOutputs:", OUT_DIR)
