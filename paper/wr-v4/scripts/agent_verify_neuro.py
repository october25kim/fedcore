"""Verify neuro-version numbers: (1) delta sweep incl 0.98 + B procedure; (2) AUROC stats."""
import csv, gzip
import numpy as np
from scipy.stats import beta, binom, spearmanr, pearsonr
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
        sid=row["semantic_id"]; ci=int(row["candidate_index"]); cl=int(row["client"])
        d=cells.setdefault(sid,{}).setdefault(ci,{"A":{},"K":{},"n":{},"feas":int(row["proposal_feasible"])})
        d["A"][cl]=int(row["A"]); d["K"][cl]=int(row["K"]); d["n"][cl]=int(row["n"])

def run(delta_r, delta_c):
    out={}
    for tag in ("S","H","B"): out[tag]=[0,[]]
    for sid,cands in cells.items():
        bestS=None; bestB=None; passH=[]
        for ci,d in sorted(cands.items()):
            if not d["feas"]: continue
            ks=sorted(d["A"]); J=len(ks)
            A=[d["A"][c] for c in ks]; K=[d["K"][c] for c in ks]; n=[d["n"][c] for c in ks]
            u=max(cp_upper(k,a,delta_r/M) for a,k in zip(A,K))
            lcb=min(cp_lower(a,nn,delta_c/M) for a,nn in zip(A,n))
            if u<=ALPHA and lcb>0 and (bestS is None or lcb>bestS): bestS=lcb
            uB=max(cp_upper(k,a,delta_r/(M*J)) for a,k in zip(A,K))
            lcbB=min(cp_lower(a,nn,delta_c/(M*J)) for a,nn in zip(A,n))
            if uB<=ALPHA and lcbB>0 and (bestB is None or lcbB>bestB): bestB=lcbB
            p=max((float(binom.cdf(k,a,ALPHA)) if a>0 else 1.0) for a,k in zip(A,K))
            passH.append((p,ci,lcb))
        while len(passH)<M: passH.append((1.0,-1,0.0))
        passH.sort()
        rejected=[]
        for i,(p,ci,lcb) in enumerate(passH):
            if p <= delta_r/(M-i): rejected.append((ci,lcb))
            else: break
        bestH=max((l for c,l in rejected if l>0), default=None)
        for tag,best in (("S",bestS),("H",bestH),("B",bestB)):
            out[tag][1].append(best if best else 0.0)
            if best: out[tag][0]+=1
    line=f"dr=dc={delta_r}"
    for tag in ("H","S","B"):
        line+=" | %s %d/%.4f" % (tag,out[tag][0],float(np.mean(out[tag][1])))
    print(line)

for d in (0.05, 0.025, 0.01, 0.005):
    run(d,d)

print("=== AUROC ===")
rows=list(csv.DictReader(open(root+"/results/agent_extract/auroc_vs_cert.csv")))
au=np.array([float(r["best_auroc"]) for r in rows])
ec=np.array([float(r["H_ecc"]) for r in rows])
ds=np.array([r["dataset"] for r in rows])
cert=np.array([int(r["H_certified"]) for r in rows])
print("N=",len(rows),"Spearman=%.4f Pearson=%.4f"%(spearmanr(au,ec).statistic, pearsonr(au,ec).statistic))
# top decile refused
k=int(np.ceil(len(au)*0.10)); idx=np.argsort(-au)[:45]
print("top-45 AUROC refused:", int((cert[idx]==0).sum()), "of", len(idx))
med=np.median(au); below=au<=med
print("at/below median: n=", int(below.sum()), "certified:", int(cert[below].sum()))
for dname in ("cifar10","cifar100","officehome","pathmnist"):
    m=ds==dname; a=au[m]; c=cert[m]
    q1=np.quantile(a,0.25); q3=np.quantile(a,0.75)
    lo=a<=q1; hi=a>=q3
    print(dname, "n=",int(m.sum()), "lowQ %d/%d"%(int(c[lo].sum()),int(lo.sum())), "hiQ %d/%d"%(int(c[hi].sum()),int(hi.sum())), "maxECA=%.3f"%ec[m].max())
