"""Core analysis for the PIRLS cross-country validation notebook.

Countries: Morocco (MAR), France (FRA), Singapore (SGP), South Africa (ZAF).
The same harmonized variables and modelling protocol are used in every country.
"""

from pathlib import Path
from itertools import combinations
import copy
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "cross_country_validation"
OUT_DIR.mkdir(exist_ok=True)

COUNTRIES = {"MAR": "Morocco", "FRA": "France", "SGP": "Singapore", "ZAF": "South Africa"}
KEYS = ["IDCNTRY", "IDSCHOOL", "IDSTUD"]
PVS = [f"ASRREA0{i}" for i in range(1, 6)]
EXTERNAL = ["ASBGHRL", "ASBG03"]
MISSING_CODES = [-99, -98, -97, -96, -95, 999, 9999]
EXCLUDE = set(KEYS + ["IDTEACH", "IDCLASS", "IDBOOK", "TOTWGT"] + PVS + EXTERNAL)

K = 5
LATENT_DIM = 16
SEED = 42
SEEDS_BASELINE = [0, 7, 42, 123, 2024]
PRETRAIN_EPOCHS = 100
REFINE_STEPS = 2000
BATCH_SIZE = 256
LR = 1e-3
MISSING_THRESHOLD = 0.50
SILHOUETTE_SAMPLE = 3000
CAPS = {"ASP": 160, "ATG": 80, "ACG": 80}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def replace_missing(df):
    out = df.copy()
    cols = [c for c in out.columns if c not in {"IDCNTRY", "IDSCHOOL", "IDSTUD", "IDTEACH"}]
    out[cols] = out[cols].replace(MISSING_CODES, np.nan)
    return out


def is_pv(name):
    return bool(re.match(r"^(ASRREA|ASRLIT|ASRINF)0[1-5]$", str(name).upper()))


def read_family(country, family):
    path = DATA_DIR / f"{family}{country}R5.sav"
    if not path.exists():
        raise FileNotFoundError(path)
    frame, _ = pyreadstat.read_sav(str(path))
    return replace_missing(frame)


def common_columns(frames, cap):
    common = set.intersection(*(set(df.columns) for df in frames.values()))
    ordered = [c for c in frames["MAR"].columns if c in common]
    selected = []
    for col in ordered:
        if col in EXCLUDE or is_pv(col):
            continue
        if not all(pd.api.types.is_numeric_dtype(df[col]) for df in frames.values()):
            continue
        if not all(df[col].isna().mean() <= MISSING_THRESHOLD for df in frames.values()):
            continue
        if not all(df[col].nunique(dropna=True) > 1 for df in frames.values()):
            continue
        selected.append(col)
        if len(selected) == cap:
            break
    return selected


def infer_discrete(frames, columns):
    discrete = []
    for col in columns:
        values = pd.concat([df[col].dropna().sample(min(2000, df[col].notna().sum()), random_state=SEED)
                            for df in frames.values() if df[col].notna().any()], ignore_index=True)
        if len(values) and np.all(np.isclose(values, np.round(values))) and values.nunique() <= 10:
            discrete.append(col)
    return discrete


def first_mode(series):
    modes = series.mode(dropna=True)
    return modes.iloc[0] if len(modes) else np.nan


def harmonize_data():
    families = {family: {c: read_family(c, family) for c in COUNTRIES} for family in CAPS}
    selected = {family: common_columns(families[family], CAPS[family]) for family in CAPS}
    discrete_base = {family: infer_discrete(families[family], selected[family]) for family in CAPS}
    merged = {}

    for country in COUNTRIES:
        stu = families["ASP"][country]
        tch = families["ATG"][country]
        sch = families["ACG"][country]

        stu_part = stu[KEYS + selected["ASP"]].copy()
        tch_rules = {c: (first_mode if c in discrete_base["ATG"] else "mean") for c in selected["ATG"]}
        sch_rules = {c: (first_mode if c in discrete_base["ACG"] else "mean") for c in selected["ACG"]}
        tch_part = tch[["IDCNTRY", "IDSCHOOL"] + selected["ATG"]].groupby(
            ["IDCNTRY", "IDSCHOOL"], as_index=False
        ).agg(tch_rules)
        sch_part = sch[["IDCNTRY", "IDSCHOOL"] + selected["ACG"]].groupby(
            ["IDCNTRY", "IDSCHOOL"], as_index=False
        ).agg(sch_rules)

        stu_part = stu_part.rename(columns={c: f"STU_{c}" for c in selected["ASP"]})
        tch_part = tch_part.rename(columns={c: f"TCH_{c}" for c in selected["ATG"]})
        sch_part = sch_part.rename(columns={c: f"SCH_{c}" for c in selected["ACG"]})
        merged[country] = stu_part.merge(tch_part, on=["IDCNTRY", "IDSCHOOL"], how="left").merge(
            sch_part, on=["IDCNTRY", "IDSCHOOL"], how="left"
        )

    candidate = [c for c in merged["MAR"].columns if c not in KEYS]
    keep = [c for c in candidate if all(
        merged[country][c].isna().mean() <= MISSING_THRESHOLD
        and merged[country][c].nunique(dropna=True) > 1 for country in COUNTRIES
    )]
    discrete = set()
    for family, cols in discrete_base.items():
        prefix = {"ASP": "STU_", "ATG": "TCH_", "ACG": "SCH_"}[family]
        discrete.update(prefix + c for c in cols if prefix + c in keep)

    prepared = {}
    audit_rows = []
    for country, frame in merged.items():
        raw = frame[keep].copy()
        disc = [c for c in keep if c in discrete]
        cont = [c for c in keep if c not in discrete]
        completed = raw.copy()
        if cont:
            completed[cont] = SimpleImputer(strategy="median").fit_transform(completed[cont])
        if disc:
            completed[disc] = SimpleImputer(strategy="most_frequent").fit_transform(completed[disc])
        X = StandardScaler().fit_transform(completed).astype(np.float32)
        prepared[country] = {"ids": frame[KEYS].copy(), "X": X, "raw": raw}
        audit_rows.append({
            "country": country, "country_name": COUNTRIES[country], "n_students": len(frame),
            "n_features": len(keep), "continuous": len(cont), "discrete": len(disc),
            "mean_missing_before_imputation": raw.isna().mean().mean(),
        })

    variable_rows = []
    for col in keep:
        variable_rows.append({"variable": col, "type": "discrete" if col in discrete else "continuous"})
    pd.DataFrame(variable_rows).to_csv(OUT_DIR / "common_variables.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT_DIR / "country_data_summary.csv", index=False)
    return prepared, pd.DataFrame(audit_rows)


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, LATENT_DIM))
        self.decoder = nn.Sequential(nn.Linear(LATENT_DIM, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def pretrain_autoencoder(X, seed=SEED):
    set_seed(seed)
    model = Autoencoder(X.shape[1]).to(DEVICE)
    tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for _ in range(PRETRAIN_EPOCHS):
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = torch.mean((reconstructed - batch) ** 2)
            loss.backward()
            optimizer.step()
    return model


def q_student(z, centers):
    distances = torch.sum((z[:, None, :] - centers[None, :, :]) ** 2, dim=2)
    q = (1 + distances) ** -1
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


def target_distribution(q):
    weights = q ** 2 / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


def refine_deep(X, pretrained, method, seed=SEED):
    set_seed(seed)
    model = copy.deepcopy(pretrained).to(DEVICE)
    tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        initial_z = model.encoder(tensor).cpu().numpy()
    km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(initial_z)
    centers = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEVICE))
    optimizer = optim.Adam(list(model.parameters()) + [centers], lr=LR)
    for step in range(1, REFINE_STEPS + 1):
        optimizer.zero_grad()
        if method == "DEC":
            z = model.encoder(tensor)
            reconstructed = None
        else:
            reconstructed, z = model(tensor)
        q = q_student(z, centers)
        p = target_distribution(q.detach())
        kl = torch.sum(p * torch.log((p + 1e-12) / (q + 1e-12))) / len(X)
        loss = kl if method == "DEC" else kl + torch.mean((reconstructed - tensor) ** 2)
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"  {method}: {step}/{REFINE_STEPS}", flush=True)
    with torch.no_grad():
        z = model.encoder(tensor)
        labels = q_student(z, centers).argmax(dim=1).cpu().numpy()
    return labels, z.cpu().numpy(), step


def internal_metrics(space, labels):
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"Silhouette": np.nan, "Calinski_Harabasz": np.nan, "Davies_Bouldin": np.nan}
    sample = min(SILHOUETTE_SAMPLE, len(space))
    return {
        "Silhouette": silhouette_score(space, labels, sample_size=sample, random_state=SEED),
        "Calinski_Harabasz": calinski_harabasz_score(space, labels),
        "Davies_Bouldin": davies_bouldin_score(space, labels),
    }


def fit_country_models(country, X):
    labels, representations, steps = {}, {}, {}
    labels["KMeans"] = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit_predict(X)
    representations["KMeans"] = X
    pca = PCA(n_components=LATENT_DIM, svd_solver="full")
    representations["PCA+KMeans"] = pca.fit_transform(X)
    labels["PCA+KMeans"] = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit_predict(representations["PCA+KMeans"])

    autoencoder = pretrain_autoencoder(X)
    with torch.no_grad():
        z_ae = autoencoder.encoder(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    representations["AE+KMeans"] = z_ae
    labels["AE+KMeans"] = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit_predict(z_ae)
    for method in ["DEC", "IDEC"]:
        labels[method], representations[method], steps[method] = refine_deep(X, autoencoder, method)
    return labels, representations, steps, pca.explained_variance_ratio_.sum()


def weighted_mean(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return np.average(values[ok], weights=weights[ok]) if ok.any() else np.nan


def weighted_eta_squared(values, groups, weights):
    frame = pd.DataFrame({"v": values, "g": groups, "w": weights}).dropna()
    frame = frame[frame.w > 0]
    grand = weighted_mean(frame.v, frame.w)
    total = np.sum(frame.w * (frame.v - grand) ** 2)
    between = sum(part.w.sum() * (weighted_mean(part.v, part.w) - grand) ** 2 for _, part in frame.groupby("g"))
    return between / total if total else np.nan


def external_validation(country, ids, method, labels):
    required = KEYS + ["TOTWGT"] + PVS + EXTERNAL
    student, _ = pyreadstat.read_sav(str(DATA_DIR / f"ASG{country}R5.sav"), usecols=required)
    for key in KEYS:
        student[key] = pd.to_numeric(student[key], errors="coerce").astype("Int64")
        ids[key] = pd.to_numeric(ids[key], errors="coerce").astype("Int64")
    for col in ["TOTWGT"] + PVS + EXTERNAL:
        student[col] = pd.to_numeric(student[col], errors="coerce")
    clustered = ids.copy()
    clustered["cluster"] = labels
    merged = clustered.merge(student, on=KEYS, validate="one_to_one")
    reading_effects = [weighted_eta_squared(merged[pv], merged.cluster, merged.TOTWGT) for pv in PVS]
    hrl_effect = weighted_eta_squared(merged.ASBGHRL, merged.cluster, merged.TOTWGT)
    valid = merged[["cluster", "ASBG03"]].dropna()
    table = pd.crosstab(valid.cluster, valid.ASBG03)
    chi2, _, _, _ = chi2_contingency(table)
    denom = table.to_numpy().sum() * min(table.shape[0] - 1, table.shape[1] - 1)
    language_v = np.sqrt(chi2 / denom) if denom else np.nan
    result = {
        "country": country, "country_name": COUNTRIES[country], "method": method,
        "reading_eta2_mean_5PV": np.mean(reading_effects), "hrl_eta2": hrl_effect,
        "home_language_cramers_v": language_v,
    }
    profiles = []
    total_weight = merged.TOTWGT.sum()
    for cluster, part in merged.groupby("cluster"):
        profiles.append({
            "country": country, "method": method, "cluster": int(cluster) + 1, "n": len(part),
            "weighted_share_pct": 100 * part.TOTWGT.sum() / total_weight,
            "reading_mean_5PV": np.mean([weighted_mean(part[pv], part.TOTWGT) for pv in PVS]),
            "hrl_mean": weighted_mean(part.ASBGHRL, part.TOTWGT),
        })
    return result, profiles


def baseline_stability(country, X, pca_space):
    rows = []
    for method, space in [("KMeans", X), ("PCA+KMeans", pca_space)]:
        partitions = {seed: KMeans(n_clusters=K, n_init=20, random_state=seed).fit_predict(space) for seed in SEEDS_BASELINE}
        values = [adjusted_rand_score(partitions[a], partitions[b]) for a, b in combinations(SEEDS_BASELINE, 2)]
        rows.append({"country": country, "method": method, "mean_ARI": np.mean(values), "min_ARI": np.min(values), "max_ARI": np.max(values)})
    return rows


def run_analysis(force=False):
    sentinel = OUT_DIR / "cross_country_metrics.csv"
    if sentinel.exists() and not force:
        return load_results()

    prepared, data_summary = harmonize_data()
    metric_rows, external_rows, profile_rows, stability_rows, run_rows = [], [], [], [], []
    for country in COUNTRIES:
        print(f"\n===== {COUNTRIES[country]} ({country}) =====", flush=True)
        X, ids = prepared[country]["X"], prepared[country]["ids"]
        labels, reps, steps, pca_variance = fit_country_models(country, X)
        stability_rows.extend(baseline_stability(country, X, reps["PCA+KMeans"]))
        run_rows.append({"country": country, "pca_variance_16": pca_variance, "DEC_steps": steps["DEC"], "IDEC_steps": steps["IDEC"]})
        for method in labels:
            own = internal_metrics(reps[method], labels[method])
            common = internal_metrics(X, labels[method])
            metric_rows.append({
                "country": country, "country_name": COUNTRIES[country], "method": method,
                **{f"representation_{k}": v for k, v in own.items()},
                **{f"original_{k}": v for k, v in common.items()},
            })
            external, profiles = external_validation(country, ids.copy(), method, labels[method])
            external_rows.append(external)
            profile_rows.extend(profiles)
            label_frame = ids.copy()
            label_frame["cluster"] = labels[method]
            label_frame.to_csv(OUT_DIR / f"labels_{country}_{method.replace('+', '_')}.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    external = pd.DataFrame(external_rows)
    profiles = pd.DataFrame(profile_rows)
    stability = pd.DataFrame(stability_rows)
    runs = pd.DataFrame(run_rows)
    metrics.to_csv(OUT_DIR / "cross_country_metrics.csv", index=False)
    external.to_csv(OUT_DIR / "cross_country_external_validation.csv", index=False)
    profiles.to_csv(OUT_DIR / "cross_country_cluster_profiles.csv", index=False)
    stability.to_csv(OUT_DIR / "cross_country_baseline_stability.csv", index=False)
    runs.to_csv(OUT_DIR / "cross_country_run_details.csv", index=False)

    summary = metrics.merge(external, on=["country", "country_name", "method"])
    summary.to_csv(OUT_DIR / "cross_country_comparison_summary.csv", index=False)
    generalizability = summary.groupby("method").agg(
        countries=("country", "nunique"),
        mean_silhouette_X=("original_Silhouette", "mean"),
        min_silhouette_X=("original_Silhouette", "min"),
        max_silhouette_X=("original_Silhouette", "max"),
        mean_reading_eta2=("reading_eta2_mean_5PV", "mean"),
        mean_hrl_eta2=("hrl_eta2", "mean"),
        mean_language_v=("home_language_cramers_v", "mean"),
    ).reset_index()
    generalizability.to_csv(OUT_DIR / "generalizability_summary.csv", index=False)

    methods = ["KMeans", "PCA+KMeans", "AE+KMeans", "DEC", "IDEC"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, value, title in [
        (axes[0], "original_Silhouette", "Common-space Silhouette"),
        (axes[1], "reading_eta2_mean_5PV", "Held-out reading effect"),
        (axes[2], "hrl_eta2", "Home-resources effect"),
    ]:
        pivot = summary.pivot(index="country_name", columns="method", values=value).reindex(columns=methods)
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(title); ax.set_xlabel(""); ax.grid(axis="y", alpha=.25); ax.tick_params(axis="x", rotation=25)
        if ax is not axes[2]:
            ax.get_legend().remove()
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_country_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame([{
        "countries": ",".join(COUNTRIES), "K": K, "latent_dim": LATENT_DIM,
        "pretrain_epochs": PRETRAIN_EPOCHS, "refine_steps_max": REFINE_STEPS,
        "primary_seed": SEED, "silhouette_sample_size": SILHOUETTE_SAMPLE,
        "device": str(DEVICE),
    }]).to_csv(OUT_DIR / "analysis_configuration.csv", index=False)
    return load_results()


def load_results():
    names = {
        "data_summary": "country_data_summary.csv",
        "metrics": "cross_country_metrics.csv",
        "external": "cross_country_external_validation.csv",
        "profiles": "cross_country_cluster_profiles.csv",
        "stability": "cross_country_baseline_stability.csv",
        "summary": "cross_country_comparison_summary.csv",
        "generalizability": "generalizability_summary.csv",
        "runs": "cross_country_run_details.csv",
    }
    return {key: pd.read_csv(OUT_DIR / filename) for key, filename in names.items()}


if __name__ == "__main__":
    results = run_analysis(force=True)
    print("\nDATA SUMMARY\n", results["data_summary"].to_string(index=False))
    print("\nGENERALIZABILITY SUMMARY\n", results["generalizability"].round(4).to_string(index=False))
    print("\nOutputs:", OUT_DIR)
