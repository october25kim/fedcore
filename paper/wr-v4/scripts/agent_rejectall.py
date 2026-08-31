"""Sec 5.3 reject-all split: freeze proposal-only selector per cell, run 1000
reservoir audits (exact binomial simulation), report validity split."""
import csv, gzip, os, sys
import numpy as np
from scipy.stats import beta
root="/w"; ALPHA=0.2; DELTA=0.05; NDRAW=1000
rng_master=np.random.default_rng(20260830)

def cp_upper(K,A,eps):
    if A==0: return 1.0
    if K>=A: return 1.0
    return float(beta.ppf(1.0-eps,K+1,A-K))

# candidates at alpha=0.2: sid -> list of (score,slot,gamma,threshold,feas)
cands={}
with gzip.open(root+"/results/theorem_aligned_wr_450_v3/primary_run/primary_candidate_counts.csv.gz","rt") as f:
    for row in csv.DictReader(f):
        if abs(float(row["alpha"])-0.2)>1e-12 or int(row["client"])!=0: continue
        cands.setdefault(row["semantic_id"],[]).append(
            (row["score"],row["slot"],float(row["gamma"]),float(row["threshold"]),int(row["proposal_feasible"])))
pins={}
with open(root+"/results/theorem_aligned_wr_450_v3/governing/raw_input_pins.csv") as f:
    for row in csv.DictReader(f): pins[row["semantic_id"]]=row["path"]
def resolve(p):
    for c in (root+"/"+p, root+"/results/confirmatory_400r/"+p):
        if os.path.exists(c): return c
    raise FileNotFoundError(p)

def softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def fold(z,pref):
    if pref+"__known_logits" in z.files:  # style A
        unk=z[pref+"__known_or_unknown"]=="unknown"
        cl=z[pref+"__client_id"].astype(int)
        pred=z[pref+"__predicted_known_index"].astype(int)
        true=z[pref+"__true_known_class_index_or_neg1"].astype(int)
        sc={"native":np.asarray(z[pref+"__native_score"],float),
            "energy":np.asarray(z[pref+"__energy_score"],float),
            "margin":np.asarray(z[pref+"__known_margin_score"],float)}
    else:
        p2={"proposal":"prop","certification":"cert"}[pref]
        L=z[p2+"_logits"]; y=z[p2+"_y_open"].astype(int)
        unk=y==-1; cl=z[p2+"_client"].astype(int)
        pred=L.argmax(axis=1); true=y
        P=softmax(L); srt=np.sort(P,axis=1)
        sc={"native":P.max(axis=1),
            "energy":np.log(np.exp(L-L.max(axis=1,keepdims=True)).sum(axis=1))+L.max(axis=1),
            "margin":srt[:,-1]-srt[:,-2]}
    err_if_acc = unk | (pred!=true)
    return sc,unk,cl,err_if_acc

out=[]
for sid,cl_ in sorted(cands.items()):
    z=np.load(resolve(pins[sid]),allow_pickle=True)
    scP,unkP,clP,errP=fold(z,"proposal")
    scC,unkC,clC,errC=fold(z,"certification")
    # 5.3 rule: among feasible cands with prop acceptance>=0.01: max acceptance,
    # ties: smaller prop risk, lexicographic score name, smaller gamma
    best=None
    for score,slot,gamma,thr,feas in cl_:
        if not feas: continue
        key="native" if slot=="native" else score
        s=scP[key]; acc=s>=thr
        pa=float(acc.mean())
        if pa<0.01: continue
        pr=float(errP[acc].mean()) if acc.any() else 0.0
        item=(-pa,pr,score,gamma,key,thr)
        if best is None or item<best: best=item
    J=int(clC.max())+1
    rec={"sid":sid,"rejectall":int(best is None)}
    if best is None:
        rec.update(validity=1.0,target=0.0)
    else:
        key,thr=best[4],best[5]
        accC=scC[key]>=thr
        a=np.zeros(J); r=np.zeros(J); nres=np.zeros(J,int)
        for j in range(J):
            m=clC==j; nres[j]=m.sum()
            aj=accC&m
            a[j]=aj.mean() if m.any() else 0
            r[j]=errC[aj].mean() if aj.any() else 0.0
        posmask=a>0
        target=float(r[posmask].max()) if posmask.any() else 0.0
        rng=np.random.default_rng(rng_master.integers(0,2**63))
        ok=0
        A=rng.binomial(nres[None,:].repeat(NDRAW,0), a[None,:])
        K=rng.binomial(A, r[None,:])
        for t in range(NDRAW):
            u=max(cp_upper(int(K[t,j]),int(A[t,j]),DELTA) for j in range(J))
            ok+= (u>=target-1e-12)
        rec.update(validity=ok/NDRAW,target=target)
    out.append(rec)

va=[o["validity"] for o in out]; pos=[o for o in out if not o["rejectall"]]
vp=[o["validity"] for o in pos]
print("cells:",len(out),"reject-all:",sum(o["rejectall"] for o in out),"positive:",len(pos))
def stats(v): 
    v=np.array(v); return round(v.min(),4), round(np.percentile(v,5),4), round(np.median(v),4)
print("ALL      min/p5/median:",stats(va))
print("POSITIVE min/p5/median:",stats(vp))
print("positive cells below 0.95:", int((np.array(vp)<0.95).sum()))
with open(root+"/results/agent_extract/rejectall_validity.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["sid","rejectall","validity","target"]); w.writeheader(); w.writerows(out)
