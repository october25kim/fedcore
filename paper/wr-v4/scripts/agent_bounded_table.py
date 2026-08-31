import csv, gzip, os, sys
import numpy as np
sys.path.insert(0,"/w")
from fedcore.mixture import traffic_mixture_confidence_box
from fedcore.certificate.joint import joint_conditional_certificate
root="/w"; M=12; J=4; DL,DR,DC_=0.02,0.04,0.04
ALPHAS=[0.10,0.15,0.20,0.25,0.30]; MGRID=[250,500,1000,2000]
cells={}
with gzip.open(root+"/results/theorem_aligned_wr_450_v3/primary_run/primary_candidate_counts.csv.gz","rt") as f:
    for row in csv.DictReader(f):
        if row["dataset"]!="officehome": continue
        a=float(row["alpha"]); sid=row["semantic_id"]; ci=int(row["candidate_index"]); cl=int(row["client"])
        d=cells.setdefault((sid,round(a,2)),{}).setdefault(ci,{"A":[0]*J,"K":[0]*J,"n":[0]*J,"feas":int(row["proposal_feasible"])})
        d["A"][cl]=int(row["A"]); d["K"][cl]=int(row["K"]); d["n"][cl]=int(row["n"])
pins={}
with open(root+"/results/theorem_aligned_wr_450_v3/governing/raw_input_pins.csv") as f:
    for row in csv.DictReader(f):
        if "officehome" in row["semantic_id"]: pins[row["semantic_id"]]=row["path"]
def resolve(p):
    for c in (root+"/"+p, root+"/results/confirmatory_400r/"+p):
        if os.path.exists(c): return c
    raise FileNotFoundError(p)
boxes={}
for sid in pins:
    z=np.load(resolve(pins[sid]),allow_pickle=True)
    tc=np.asarray(z["traffic_client"]); seed=int(z["traffic_draw_seed"])
    boxes[sid]={}
    for m in MGRID:
        rng=np.random.default_rng(np.random.SeedSequence([seed,m]))
        counts=np.bincount(tc[rng.integers(0,len(tc),size=m)],minlength=J)
        b=traffic_mixture_confidence_box(counts.tolist(),delta=DL).mixture
        boxes[sid][m]=(b.lower.tolist(),b.upper.tolist())
def fam(cands,alpha,eps,low,up=None,lo=None,hi=None):
    best=None
    for ci in sorted(cands):
        dd=cands[ci]
        if not dd["feas"]: continue
        try:
            c=joint_conditional_certificate(dd["A"],dd["K"],dd["n"],alpha=alpha,risk_eps=eps,
                acceptance_lower_eps=low,acceptance_upper_eps=up,lambda_lower=lo,lambda_upper=hi)
        except Exception: continue
        if c.certified and (best is None or c.coverage_lcb>best): best=c.coverage_lcb
    return best
def sweep(tag,fn):
    out=[tag]
    for a in ALPHAS:
        C=0;E=[]
        for sid in pins:
            cands=cells.get((sid,a))
            b=fn(sid,cands,a) if cands else None
            E.append(b or 0.0); C+= b is not None
        out.append("%d/%.3f"%(C,float(np.mean(E))))
    print(" | ".join(out))
sweep("FS 0.05/0.05", lambda s,c,a: fam(c,a,[0.05/M]*J,[0.05/M]*J))
sweep("FS 0.04/0.04", lambda s,c,a: fam(c,a,[0.04/M]*J,[0.04/M]*J))
for m in MGRID:
    sweep("BOX m=%d"%m, lambda s,c,a,m=m: fam(c,a,[DR/2/J/M]*J,[DC_/J/M]*J,[DR/2/J/M]*J,*boxes[s][m]))
