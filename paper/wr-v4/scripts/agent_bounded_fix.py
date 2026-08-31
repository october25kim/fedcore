"""Theorem-aligned bounded-mixture recomputation (review correction).
Risk-side endpoints at eps_r = dr/(3*J*M); coverage LCB at eps_c = dc/(J*M).
Two-call structure because joint_conditional_certificate shares alow between
the robust-ratio domain and the coverage infimum."""
import csv, os, sys
import numpy as np
from fedcore.certificate.joint import joint_conditional_certificate
from fedcore.mixture import traffic_mixture_confidence_box

root="/w"; J=4; M=12
DL, DR, DC_=0.02, 0.04, 0.04
M_GRID=(250,500,1000,2000)
ALPHAS=(0.10,0.15,0.20,0.25,0.30)

import gzip
cells={}
with gzip.open(root+"/results/theorem_aligned_wr_450_v3/primary_run/primary_candidate_counts.csv.gz","rt") as f:
    for row in csv.DictReader(f):
        if "officehome" not in row["semantic_id"]: continue
        al=round(float(row["alpha"]),2)
        sid=row["semantic_id"]; ci=int(row["candidate_index"]); cl=int(row["client"])
        d=cells.setdefault(al,{}).setdefault(sid,{}).setdefault(ci,{"A":[0]*J,"K":[0]*J,"n":[0]*J,"feas":int(row["proposal_feasible"])})
        d["A"][cl]=int(row["A"]); d["K"][cl]=int(row["K"]); d["n"][cl]=int(row["n"])
print("alphas:",sorted(cells), "cells:",len(cells[0.2]), file=sys.stderr)

pins={}
with open(root+"/results/theorem_aligned_wr_450_v3/governing/raw_input_pins.csv") as f:
    for row in csv.DictReader(f):
        if "officehome" in row["semantic_id"]: pins[row["semantic_id"]]=row["path"]
def resolve(p):
    for c in (root+"/"+p, root+"/results/confirmatory_400r/"+p):
        if os.path.exists(c): return c
    raise FileNotFoundError(p)

def certify_bounded(d, alpha, lo, hi):
    er=DR/(3.0*J*M); ec=DC_/(J*M)
    try:
        c1=joint_conditional_certificate(d["A"],d["K"],d["n"],alpha=alpha,
            risk_eps=[er]*J,acceptance_lower_eps=[er]*J,acceptance_upper_eps=[er]*J,
            lambda_lower=lo,lambda_upper=hi)
        c2=joint_conditional_certificate(d["A"],d["K"],d["n"],alpha=alpha,
            risk_eps=[er]*J,acceptance_lower_eps=[ec]*J,acceptance_upper_eps=[er]*J,
            lambda_lower=lo,lambda_upper=hi)
    except Exception:
        return None
    ok = c1.certified and c2.coverage_lcb>0
    return c2.coverage_lcb if ok else None

def certify_fs(d, alpha, dr, dc):
    try:
        c=joint_conditional_certificate(d["A"],d["K"],d["n"],alpha=alpha,
            risk_eps=[dr/M]*J,acceptance_lower_eps=[dc/M]*J)
    except Exception:
        return None
    return c.coverage_lcb if c.certified else None

results={}
for sid in sorted(cells[0.2]):
    z=np.load(resolve(pins[sid]),allow_pickle=True)
    tc=np.asarray(z["traffic_client"]); seed=int(z["traffic_draw_seed"])
    boxes={}
    for m in M_GRID:
        rng=np.random.default_rng(np.random.SeedSequence([seed,m]))
        draw=tc[rng.integers(0,len(tc),size=m)]
        counts=np.bincount(draw,minlength=J)
        box=traffic_mixture_confidence_box(counts.tolist(),delta=DL).mixture
        boxes[m]=(box.lower.tolist(),box.upper.tolist())
    for alpha in ALPHAS:
        cands=cells[round(alpha,2)][sid]
        for tag,fn in [("FS_headline",lambda d: certify_fs(d,alpha,0.05,0.05)),
                       ("FS_matched",lambda d: certify_fs(d,alpha,0.04,0.04))]:
            best=None
            for ci,d in sorted(cands.items()):
                if not d["feas"]: continue
                v=fn(d)
                if v is not None and (best is None or v>best): best=v
            key=(tag,alpha); s=results.setdefault(key,[0,0.0])
            if best: s[0]+=1; s[1]+=best
        for m in M_GRID:
            lo,hi=boxes[m]; best=None
            for ci,d in sorted(cands.items()):
                if not d["feas"]: continue
                v=certify_bounded(d,alpha,lo,hi)
                if v is not None and (best is None or v>best): best=v
            key=(f"BOX{m}",alpha); s=results.setdefault(key,[0,0.0])
            if best: s[0]+=1; s[1]+=best

N=len(cells[0.2])
out=[["procedure","alpha","certified","eca"]]
for (tag,alpha),(c,tot) in sorted(results.items()):
    out.append([tag,alpha,c,round(tot/N,4)])
with open(root+"/results/agent_extract/bounded_lambda_fixed_summary.csv","w",newline="") as f:
    csv.writer(f).writerows(out)
for row in out[1:]:
    print(row)
