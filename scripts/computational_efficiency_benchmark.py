"""Isolated-process timing and peak-memory benchmark for clustering methods."""
from pathlib import Path
import argparse, ctypes, json, os, platform, subprocess, sys, threading, time
import numpy as np
import pandas as pd

DATA_DIR=Path(__file__).resolve().parents[1]/"data"
OUT_DIR=DATA_DIR/"computational_efficiency"; OUT_DIR.mkdir(exist_ok=True)
X_PATH=DATA_DIR/"robust_primary_article"/"X_robust.npy"
METHODS=["KMeans","Agglomerative","GMM","PCA+KMeans","AE+KMeans","DEC","IDEC"]
SIZES=[1000,3000,7017]

def working_set_bytes():
    class PMC(ctypes.Structure):
        _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong),("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
    counters=PMC(); counters.cb=ctypes.sizeof(PMC)
    get_process=ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype=ctypes.c_void_p
    get_memory=ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes=[ctypes.c_void_p,ctypes.POINTER(PMC),ctypes.c_ulong]
    get_memory.restype=ctypes.c_int
    if not get_memory(get_process(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError()
    return int(counters.WorkingSetSize)

def worker(method,n):
    import random
    from sklearn.cluster import KMeans,AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader,TensorDataset
    seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(min(8,torch.get_num_threads()))
    Xall=np.load(X_PATH,mmap_mode="r"); idx=np.sort(np.random.default_rng(seed).choice(len(Xall),n,replace=False)); X=np.asarray(Xall[idx],dtype=np.float32)
    baseline=working_set_bytes(); peak=[baseline]; running=[True]
    def monitor():
        while running[0]: peak[0]=max(peak[0],working_set_bytes()); time.sleep(.01)
    th=threading.Thread(target=monitor,daemon=True); th.start()
    phases={}; start=time.perf_counter(); labels=None
    if method=="KMeans":
        labels=KMeans(n_clusters=5,n_init=20,random_state=seed).fit_predict(X)
    elif method=="Agglomerative":
        labels=AgglomerativeClustering(n_clusters=5,linkage="average",metric="euclidean").fit_predict(X)
    elif method=="GMM":
        labels=GaussianMixture(n_components=5,covariance_type="full",reg_covar=1e-4,max_iter=500,random_state=seed).fit_predict(X.astype(float))
    elif method=="PCA+KMeans":
        t=time.perf_counter(); Z=PCA(n_components=16,svd_solver="full").fit_transform(X); phases["representation_seconds"]=time.perf_counter()-t
        t=time.perf_counter(); labels=KMeans(n_clusters=5,n_init=20,random_state=seed).fit_predict(Z); phases["clustering_seconds"]=time.perf_counter()-t
    else:
        class AE(nn.Module):
            def __init__(self,d):
                super().__init__(); self.encoder=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,16)); self.decoder=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,256),nn.ReLU(),nn.Linear(256,d))
            def forward(self,x): z=self.encoder(x); return self.decoder(z),z
        device=torch.device("cpu"); model=AE(X.shape[1]).to(device); tensor=torch.tensor(X,dtype=torch.float32)
        loader=DataLoader(TensorDataset(tensor),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed)); opt=optim.Adam(model.parameters(),lr=1e-3)
        t=time.perf_counter()
        for _ in range(100):
            for (xb,) in loader:
                opt.zero_grad(); rec,_=model(xb); loss=((rec-xb)**2).mean(); loss.backward(); opt.step()
        phases["pretraining_seconds"]=time.perf_counter()-t
        with torch.no_grad(): Z=model.encoder(tensor).numpy()
        if method=="AE+KMeans":
            t=time.perf_counter(); labels=KMeans(n_clusters=5,n_init=20,random_state=seed).fit_predict(Z); phases["clustering_seconds"]=time.perf_counter()-t
        else:
            centers=nn.Parameter(torch.tensor(KMeans(n_clusters=5,n_init=20,random_state=seed).fit(Z).cluster_centers_,dtype=torch.float32)); opt=optim.Adam(list(model.parameters())+[centers],lr=1e-3)
            def qfun(z):
                q=(1+((z[:,None,:]-centers[None,:,:])**2).sum(2))**-1; return q/q.sum(1,keepdim=True).clamp_min(1e-12)
            t=time.perf_counter()
            for _ in range(2000):
                opt.zero_grad(); rec,z=model(tensor); q=qfun(z); w=q.detach()**2/q.detach().sum(0,keepdim=True).clamp_min(1e-12); p=w/w.sum(1,keepdim=True).clamp_min(1e-12); kl=(p*torch.log((p+1e-12)/(q+1e-12))).sum()/n; loss=kl if method=="DEC" else kl+((rec-tensor)**2).mean(); loss.backward(); opt.step()
            phases["refinement_seconds"]=time.perf_counter()-t
            with torch.no_grad(): labels=qfun(model.encoder(tensor)).argmax(1).numpy()
    elapsed=time.perf_counter()-start; running[0]=False; th.join(timeout=.1); peak[0]=max(peak[0],working_set_bytes())
    return {"method":method,"n":n,"features":X.shape[1],"total_seconds":elapsed,"baseline_mb":baseline/2**20,"peak_working_set_mb":peak[0]/2**20,"incremental_peak_mb":max(0,peak[0]-baseline)/2**20,"clusters":len(np.unique(labels)),**phases}

def orchestrate():
    rows=[]
    for n in SIZES:
        for method in METHODS:
            print(f"benchmark {method} n={n}",flush=True)
            cmd=[sys.executable,str(Path(__file__).resolve()),"--worker","--method",method,"--n",str(n)]
            done=subprocess.run(cmd,capture_output=True,text=True,check=True); row=json.loads(done.stdout.strip().splitlines()[-1]); rows.append(row); pd.DataFrame(rows).to_csv(OUT_DIR/"computational_benchmark.csv",index=False)
    df=pd.DataFrame(rows); base=df[df.n==1000][["method","total_seconds"]].rename(columns={"total_seconds":"seconds_n1000"}); large=df[df.n==7017][["method","total_seconds"]].rename(columns={"total_seconds":"seconds_n7017"}); scale=base.merge(large,on="method"); scale["empirical_time_ratio_7017_to_1000"]=scale.seconds_n7017/scale.seconds_n1000; scale.to_csv(OUT_DIR/"scalability_summary.csv",index=False)
    pd.DataFrame([{"platform":platform.platform(),"python":sys.version.split()[0],"logical_cpus":os.cpu_count(),"device":"CPU","features":203,"deep_pretrain_epochs":100,"deep_refinement_iterations":2000}]).to_csv(OUT_DIR/"benchmark_environment.csv",index=False)
    print(df.round(3).to_string(index=False)); print("\n",scale.round(2).to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--worker",action="store_true"); ap.add_argument("--method",choices=METHODS); ap.add_argument("--n",type=int); args=ap.parse_args()
    if args.worker: print(json.dumps(worker(args.method,args.n)))
    else: orchestrate()
