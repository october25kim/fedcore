"""Bounded-Lambda Office-Home campaign (agent, predeclared protocol).
Primary: m=1000 traffic draw, delta_lambda/r/c = 0.02/0.04/0.04, M=12 family.
Controls: full simplex at headline (0.05/0.05) and budget-matched (0.04/0.04).
"""
import numpy as np, csv, gzip, os, sys, json
sys.path.insert(0,"/w")
from fedcore.mixture import traffic_mixture_confidence_box
from fedcore.certificate.joint import joint_conditional_certificate

root="/w"
ALPHA=0.2; M=12; J=4
DL, DR, DC_=0.02, 0.04, 0.04
M_GRID=(250,500,1000,2000); M_PRIMARY=1000

# --- load count tensor at alpha=0.2, officehome ---
cells={}  # sid -> {cand_idx: (A[4],K[4],n[4])}
with gzip.open(root+"/results/theorem_aligned_wr_450_v3/primary_run/primary_candidate_counts.csv.gz","rt") as f:
    for row in csv.DictReader(f):
        if row["dataset"]!="officehome" or float(row["alpha"])!=ALPHA: continue
        sid=row["semantic_id"]; ci=int(row["candidate_index"]); cl=int(row["client"])
        d=cells.setdefault(sid,{}).setdefault(ci,{"A":[0]*J,"K":[0]*J,"n":[0]*J,"feas":int(row["proposal_feasible"])})
        d["A"][cl]=int(row["A"]); d["K"][cl]=int(row["K"]); d["n"][cl]=int(row["n"])
print("OH cells:",len(cells), file=sys.stderr)

# --- traffic reservoirs from pins ---
pins={}
with open(root+"/results/theorem_aligned_wr_450_v3/governing/raw_input_pins.csv") as f:
    for row in csv.DictReader(f):
        if "officehome" in row["semantic_id"]: pins[row["semantic_id"]]=row["path"]
def resolve(p):
    for c in (root+"/"+p, root+"/results/confirmatory_400r/"+p):
        if os.path.exists(c): return c
    raise FileNotFoundError(p)

def certify_family(A,K,n,feas,risk_eps,low_eps,up_eps=None,lo=None,hi=None):
    best=None; ncert=0
    for ci in sorted(A):
        if not A[ci]["feas"]: continue
        try:
            c=joint_conditional_certificate(A[ci]["A"],A[ci]["K"],A[ci]["n"],alpha=ALPHA,
                risk_eps=risk_eps,acceptance_lower_eps=low_eps,
                acceptance_upper_eps=up_eps,lambda_lower=lo,lambda_upper=hi)
        except Exception as e:
            continue
        if c.certified:
            ncert+=1
            if best is None or c.coverage_lcb>best[1]: best=(ci,c.coverage_lcb,c.risk_ucb)
    return ncert,best

rows=[]
for sid,cands in sorted(cells.items()):
    z=np.load(resolve(pins[sid]),allow_pickle=True)
    tc=np.asarray(z["traffic_client"]); seed=int(z["traffic_draw_seed"])
    res={"sid":sid}
    # full simplex controls (per-member eps, no J division)
    for tag,(dr,dc) in (("FS_headline",(0.05,0.05)),("FS_matched",(0.04,0.04))):
        eps=[dr/M]*J; low=[dc/M]*J
        nc,best=certify_family(cands,None,None,None,eps,low)
        res[tag+"_cert"]=1 if best else 0; res[tag+"_lcb"]=best[1] if best else 0.0
    # bounded
    for m in M_GRID:
        rng=np.random.default_rng(np.random.SeedSequence([seed,m]))
        draw=tc[rng.integers(0,len(tc),size=m)]
        counts=np.bincount(draw,minlength=J)
        box=traffic_mixture_confidence_box(counts.tolist(),delta=DL).mixture
        lo,hi=box.lower.tolist(),box.upper.tolist()
        eps=[DR/2.0/J/M]*J; low=[DC_/J/M]*J; up=[DR/2.0/J/M]*J
        nc,best=certify_family(cands,None,None,None,eps,low,up,lo,hi)
        res[f"BOX{m}_cert"]=1 if best else 0
        res[f"BOX{m}_lcb"]=best[1] if best else 0.0
        res[f"BOX{m}_ucb"]=best[2] if best else float("nan")
        res[f"BOX{m}_width"]=float(np.mean(np.array(hi)-np.array(lo)))
    rows.append(res)

out=root+"/results/agent_extract/bounded_lambda_cells.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# summary
def summ(tag):
    c=sum(r[tag+"_cert"] for r in rows); e=np.mean([r[tag+"_lcb"] for r in rows])
    return c,round(float(e),4)
print("cells:",len(rows))
for tag in ["FS_headline","FS_matched"]+[f"BOX{m}" for m in M_GRID]:
    print(tag, summ(tag))
