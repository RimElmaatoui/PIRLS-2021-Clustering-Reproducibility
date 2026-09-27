"""Targeted DEC/IDEC hyperparameter ablation on the final robust Moroccan matrix."""
from pathlib import Path
import copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "hyperparameter_ablation_robust"
OUT_DIR.mkdir(exist_ok=True)
X_FULL = np.load(DATA_DIR / "robust_primary_article" / "X_robust.npy")
SEED, K, BATCH, PRE_EPOCHS = 42, 5, 256, 30
# Fixed resource-aware screening sample; final models use all 7,017 students, 100/2,000.
sample_rng = np.random.default_rng(SEED)
SCREEN_INDEX = np.sort(sample_rng.choice(len(X_FULL), size=3000, replace=False))
X = X_FULL[SCREEN_INDEX]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class AE(nn.Module):
    def __init__(self, d, hidden, latent):
        super().__init__()
        enc=[]; prev=d
        for h in hidden: enc += [nn.Linear(prev,h),nn.ReLU()]; prev=h
        enc += [nn.Linear(prev,latent)]
        dec=[]; prev=latent
        for h in hidden[::-1]: dec += [nn.Linear(prev,h),nn.ReLU()]; prev=h
        dec += [nn.Linear(prev,d)]
        self.encoder=nn.Sequential(*enc); self.decoder=nn.Sequential(*dec)
    def forward(self,x):
        z=self.encoder(x); return self.decoder(z),z

def pretrain(hidden, latent, lr=1e-3):
    seed_all(); model=AE(X.shape[1],hidden,latent).to(DEVICE)
    loader=DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32)),batch_size=BATCH,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    opt=optim.Adam(model.parameters(),lr=lr)
    for _ in range(PRE_EPOCHS):
        for (xb,) in loader:
            xb=xb.to(DEVICE); opt.zero_grad(); rec,_=model(xb); loss=((rec-xb)**2).mean(); loss.backward(); opt.step()
    return model

def qfun(z,c):
    q=(1+((z[:,None,:]-c[None,:,:])**2).sum(2))**-1
    return q/q.sum(1,keepdim=True).clamp_min(1e-12)

def target(q):
    w=q**2/q.sum(0,keepdim=True).clamp_min(1e-12)
    return w/w.sum(1,keepdim=True).clamp_min(1e-12)

def metrics(space, labels):
    return silhouette_score(space,labels,sample_size=min(3000,len(space)),random_state=SEED),calinski_harabasz_score(space,labels),davies_bouldin_score(space,labels)

def refine(pretrained, method, lr=1e-3, steps=2000, recon_weight=1.0):
    seed_all(); model=copy.deepcopy(pretrained).to(DEVICE); xt=torch.tensor(X,dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): z0=model.encoder(xt).cpu().numpy()
    km=KMeans(n_clusters=K,n_init=20,random_state=SEED).fit(z0)
    centers=nn.Parameter(torch.tensor(km.cluster_centers_,dtype=torch.float32,device=DEVICE))
    opt=optim.Adam(list(model.parameters())+[centers],lr=lr)
    for _ in range(steps):
        opt.zero_grad(); rec,z=model(xt); q=qfun(z,centers); p=target(q.detach())
        kl=(p*torch.log((p+1e-12)/(q+1e-12))).sum()/len(X)
        loss=kl if method=="DEC" else kl+recon_weight*((rec-xt)**2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        z=model.encoder(xt); labels=qfun(z,centers).argmax(1).cpu().numpy(); z=z.cpu().numpy()
    return labels,z

cache={}
def get_ae(hidden,latent):
    key=(tuple(hidden),latent)
    if key not in cache:
        print("pretrain",key,flush=True); cache[key]=pretrain(hidden,latent)
    return cache[key]

configs=[]
for latent in [8,16,32]:
    for method in ["DEC","IDEC"]: configs.append(("latent_dim",str(latent),method,[256,64],latent,1e-3,500,1.0))
for name,hidden in [("shallow","[128]"),("base","[256,64]"),("deep","[512,256,64]")]:
    h={"[128]":[128],"[256,64]":[256,64],"[512,256,64]":[512,256,64]}[hidden]
    for method in ["DEC","IDEC"]: configs.append(("architecture",name,method,h,16,1e-3,500,1.0))
for weight in [0.1,1.0,10.0]: configs.append(("reconstruction_weight",str(weight),"IDEC",[256,64],16,1e-3,500,weight))
for steps in [100,250,500]:
    for method in ["DEC","IDEC"]: configs.append(("iterations",str(steps),method,[256,64],16,1e-3,steps,1.0))
for lr in [1e-4,1e-3,3e-3]:
    for method in ["DEC","IDEC"]: configs.append(("learning_rate",str(lr),method,[256,64],16,lr,500,1.0))

# Remove exact duplicate fits while retaining all ablation labels.
fit_cache={}; records=[]; label_map={}
for i,(factor,setting,method,hidden,latent,lr,steps,rw) in enumerate(configs,1):
    fit_key=(method,tuple(hidden),latent,lr,steps,rw)
    print(f"[{i}/{len(configs)}] {factor}={setting} {method}",flush=True)
    if fit_key not in fit_cache:
        fit_cache[fit_key]=refine(get_ae(hidden,latent),method,lr,steps,rw)
    labels,z=fit_cache[fit_key]; label_map[(factor,setting,method)]=labels
    sz,chz,dbz=metrics(z,labels); sx,chx,dbx=metrics(X,labels)
    records.append({"factor":factor,"setting":setting,"method":method,"hidden":str(hidden),"latent_dim":latent,"lr":lr,"iterations":steps,"reconstruction_weight":rw,
                    "Silhouette_Z":sz,"CH_Z":chz,"DB_Z":dbz,"Silhouette_X":sx,"CH_X":chx,"DB_X":dbx})

df=pd.DataFrame(records)
for method in ["DEC","IDEC"]:
    primary=label_map[("latent_dim","16",method)]
    mask=df.method.eq(method)
    for idx,row in df[mask].iterrows():
        df.loc[idx,"ARI_vs_primary"]=adjusted_rand_score(primary,label_map[(row.factor,row.setting,method)])
df.to_csv(OUT_DIR/"hyperparameter_ablation.csv",index=False)
summary=df.groupby(["factor","method"]).agg(Silhouette_X_min=("Silhouette_X","min"),Silhouette_X_max=("Silhouette_X","max"),ARI_min=("ARI_vs_primary","min"),ARI_max=("ARI_vs_primary","max")).reset_index()
summary.to_csv(OUT_DIR/"ablation_range_summary.csv",index=False)
pd.DataFrame([{"screening_n":len(X),"full_n":len(X_FULL),"pretrain_epochs":PRE_EPOCHS,"max_refinement_iterations":500,"seed":SEED}]).to_csv(OUT_DIR/"ablation_configuration.csv",index=False)
print(df.round(5).to_string(index=False)); print("\n",summary.round(5).to_string(index=False))
