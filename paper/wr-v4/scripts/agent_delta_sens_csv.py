"""Emit delta-sensitivity CSV (H/S/B x joint confidence) from released count tensor."""
import csv, gzip
import numpy as np
from scipy.stats import beta, binom
root="/w"; ALPHA=0.2; M=12
def cp_upper(K,A,eps):
    if A==0: return 1.0
    if K>=A: return 1.0
    return float(beta.ppf(1.0-eps, K+1, A-K))
def cp_lower(A,n,eps):
    if A==0: return 0.0
    return float(beta.ppf(eps, A, n-A+1))
cells={}
with gzip.open(root+"/results/theorem_aligned_wr_450_v3/primary_run/primary_candidate_counts.csv.gz","rt") as f:
    for row in csv.DictReader(f):
        if abs(float(row["alpha"])-ALPHA)>1e-12: continue
        sid=row["semantic_id"]; ci=int(row["candidate_index"])
        d=cells.setdefault(sid,{}).setdefault(ci,{"A":{},"K":{},"n":{},"feas":int(row["proposal_feasible"])})
        cl=int(row["client"])
        d["A"][cl]=int(row["A"]); d["K"][cl]=int(row["K"]); d["n"][cl]=int(row["n"])
out=[["joint_confidence","delta_r","delta_c","procedure","certified_cells","effective_certified_acceptance"]]
for dd in (0.05,0.025,0.01,0.005):
    res={t:[0,[]] for t in ("H","S","B")}
    for sid,cands in cells.items():
        bestS=None; bestB=None; passH=[]
        for ci,d in sorted(cands.items()):
            if not d["feas"]: continue
            ks=sorted(d["A"]); J=len(ks)
            A=[d["A"][c] for c in ks]; K=[d["K"][c] for c in ks]; n=[d["n"][c] for c in ks]
            u=max(cp_upper(k,a,dd/M) for a,k in zip(A,K))
            l=min(cp_lower(a,nn,dd/M) for a,nn in zip(A,n))
            if u<=ALPHA and l>0 and (bestS is None or l>bestS): bestS=l
            uB=max(cp_upper(k,a,dd/(M*J)) for a,k in zip(A,K))
            lB=min(cp_lower(a,nn,dd/(M*J)) for a,nn in zip(A,n))
            if uB<=ALPHA and lB>0 and (bestB is None or lB>bestB): bestB=lB
            p=max((float(binom.cdf(k,a,ALPHA)) if a>0 else 1.0) for a,k in zip(A,K))
            passH.append((p,ci,l))
        while len(passH)<M: passH.append((1.0,-1,0.0))
        passH.sort()
        rej=[]
        for i,(p,ci,l) in enumerate(passH):
            if p<=dd/(M-i): rej.append(l)
            else: break
        bestH=max((l for l in rej if l>0), default=None)
        for t,b in (("S",bestS),("H",bestH),("B",bestB)):
            res[t][1].append(b if b else 0.0)
            if b: res[t][0]+=1
    for t in ("H","S","B"):
        out.append([f"{1-2*dd:.2f}", dd, dd, t, res[t][0], round(float(np.mean(res[t][1])),4)])
with open(root+"/results/agent_extract/delta_sensitivity.csv","w",newline="") as f:
    csv.writer(f).writerows(out)
print("written", len(out)-1, "rows")
