from pathlib import Path
from itertools import combinations
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

DATA_DIR = Path(__file__).resolve().parent
PRIMARY_DIR = DATA_DIR / "robust_primary_article"
LABEL_DIR = DATA_DIR / "Robust_imputation_comparison"
OUT_DIR = DATA_DIR / "robust_seed_stability"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 7, 42, 123, 2024]
PRIMARY_SEED = 42
K = 5
PRETRAIN_EPOCHS = 100
MAX_ITER = 2000
BATCH_SIZE = 256
LATENT_DIM = 16
HIDDEN_DIMS = [256, 64]
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = np.load(PRIMARY_DIR / "X_robust.npy")
ids = pd.read_csv(PRIMARY_DIR / "ids.csv")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def pretrain(seed):
    set_seed(seed)
    model = Autoencoder(X.shape[1]).to(DEVICE)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    opt = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = torch.mean((recon - xb) ** 2)
            loss.backward(); opt.step()
            total += loss.item() * len(xb)
        if epoch == 1 or epoch % 20 == 0:
            print(f"seed={seed} AE {epoch}/{PRETRAIN_EPOCHS} recon={total/len(X):.6f}", flush=True)
    return model


def q_student(z, centers):
    dist = torch.sum((z[:, None, :] - centers[None, :, :]) ** 2, dim=2)
    q = (1 + dist) ** -1
    return q / q.sum(dim=1, keepdim=True)


def target(q):
    w = q ** 2 / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)


def refine(pretrained, seed, idec):
    set_seed(seed)
    model = copy.deepcopy(pretrained).to(DEVICE)
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad(): z0 = model.encoder(xt).cpu().numpy()
    km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(z0)
    centers = nn.Parameter(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=DEVICE))
    opt = optim.Adam(list(model.parameters()) + [centers], lr=LR)
    for step in range(1, MAX_ITER + 1):
        opt.zero_grad()
        recon, z = model(xt)
        q = q_student(z, centers)
        p = target(q.detach())
        kl = torch.sum(p * torch.log((p + 1e-12) / (q + 1e-12))) / len(X)
        rec = torch.mean((recon - xt) ** 2)
        loss = kl + rec if idec else kl
        loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"seed={seed} {'IDEC' if idec else 'DEC'} {step}/{MAX_ITER}", flush=True)
    with torch.no_grad():
        z = model.encoder(xt)
        labels = q_student(z, centers).argmax(dim=1).cpu().numpy()
    return labels, z.cpu().numpy()


def metric_row(method, seed, labels, Z):
    return {
        "method": method, "seed": seed,
        "silhouette_Z": silhouette_score(Z, labels),
        "CH_Z": calinski_harabasz_score(Z, labels),
        "DB_Z": davies_bouldin_score(Z, labels),
        "silhouette_X": silhouette_score(X, labels),
        "CH_X": calinski_harabasz_score(X, labels),
        "DB_X": davies_bouldin_score(X, labels),
    }


assignments = {}
metric_rows = []
# The saved seed-42 assignments define the robust primary solution used elsewhere.
for method, filename in [("DEC", "DEC"), ("IDEC", "IDEC")]:
    d = pd.read_csv(LABEL_DIR / f"labels_robust_median_mode_{filename}.csv")
    joined = ids[["IDSTUD"]].merge(d[["IDSTUD", "cluster"]], on="IDSTUD", validate="one_to_one")
    assignments[(method, PRIMARY_SEED)] = joined.cluster.to_numpy(int)

for seed in [s for s in SEEDS if s != PRIMARY_SEED]:
    print(f"\n===== SEED {seed} =====", flush=True)
    ae = pretrain(seed)
    for method, idec in [("DEC", False), ("IDEC", True)]:
        lab, Z = refine(ae, seed, idec)
        assignments[(method, seed)] = lab
        metric_rows.append(metric_row(method, seed, lab, Z))
        out = ids.copy(); out["cluster"] = lab
        method_dir = OUT_DIR / method / f"seed_{seed}"
        method_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(method_dir / "ids_with_clusters.csv", index=False)
        np.save(method_dir / "latent.npy", Z)

# Seed 42 input-space metrics; latent values come from the verified primary run.
primary_metrics = pd.read_csv(DATA_DIR / "Robust_imputation_comparison" / "all_clustering_metrics.csv")
for method in ["DEC", "IDEC"]:
    zrow = primary_metrics[(primary_metrics.scenario == "robust_median_mode") & (primary_metrics.method == method) & (primary_metrics.space == "latent")].iloc[-1]
    xrow = primary_metrics[(primary_metrics.scenario == "robust_median_mode") & (primary_metrics.method == method) & (primary_metrics.space == "standardized_input")].iloc[-1]
    metric_rows.append({
        "method": method, "seed": PRIMARY_SEED,
        "silhouette_Z": zrow.Silhouette, "CH_Z": zrow.Calinski_Harabasz, "DB_Z": zrow.Davies_Bouldin,
        "silhouette_X": xrow.Silhouette, "CH_X": xrow.Calinski_Harabasz, "DB_X": xrow.Davies_Bouldin,
    })
    out = ids.copy(); out["cluster"] = assignments[(method, PRIMARY_SEED)]
    method_dir = OUT_DIR / method / f"seed_{PRIMARY_SEED}"
    method_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(method_dir / "ids_with_clusters.csv", index=False)

metrics = pd.DataFrame(metric_rows).sort_values(["method", "seed"])
metrics.to_csv(OUT_DIR / "seed_metrics.csv", index=False)

ari_rows = []
for method in ["DEC", "IDEC"]:
    for a, b in combinations(SEEDS, 2):
        ari_rows.append({"method": method, "seed_a": a, "seed_b": b, "ARI": adjusted_rand_score(assignments[(method, a)], assignments[(method, b)])})
ari = pd.DataFrame(ari_rows)
summary = ari.groupby("method").ARI.agg(["mean", "std", "min", "max"]).reset_index()
ari.to_csv(OUT_DIR / "pairwise_ari.csv", index=False)
summary.to_csv(OUT_DIR / "ari_summary.csv", index=False)
print("\nSEED METRICS\n", metrics.to_string(index=False))
print("\nARI SUMMARY\n", summary.to_string(index=False))

