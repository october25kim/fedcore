import csv, numpy as np
root="/w"
# H rows at alpha 0.2
cert={}
with open(root+"/results/theorem_aligned_wr_450_v3/primary_run/primary_per_cell_procedures.csv") as f:
    for r in csv.DictReader(f):
        if r["procedure"]=="H" and abs(float(r["alpha"])-0.2)<1e-9:
            cert[r["semantic_id"]]={"certified":int(r["certified"]),"ecc":float(r["effective_certified_coverage"] or 0),
                                     "score":r["selected_score"]}
au={}
with open(root+"/results/agent_extract/auroc_cert_fold.csv") as f:
    for r in csv.DictReader(f):
        au.setdefault(r["semantic_id"],{})[r["score"]]=(float(r["auroc"]),float(r["fpr_at_95tpr"]))
rows=[]
for sid,c in cert.items():
    if sid not in au: continue
    best_auroc=max(v[0] for v in au[sid].values())
    ds=sid.split("_")[0] if not sid.startswith("officehome") else "officehome"
    rows.append((sid,ds,best_auroc,min(v[1] for v in au[sid].values()),c["certified"],c["ecc"]))
import statistics
print("cells joined:",len(rows))
for ds in ["cifar10","cifar100","officehome","resnext29"]:
    sub=[r for r in rows if r[1].startswith(ds[:6])] if ds!="resnext29" else [r for r in rows if "pathmnist" in r[0]]
    if not sub: 
        sub=[r for r in rows if ds[:4] in r[0]]
    if not sub: continue
    cert1=[r for r in sub if r[4]==1]; cert0=[r for r in sub if r[4]==0]
    print(ds, "n=",len(sub), "| cert:",len(cert1), "| AUROC cert:", round(statistics.mean(r[2] for r in cert1),3) if cert1 else "-",
          "| AUROC refused:", round(statistics.mean(r[2] for r in cert0),3) if cert0 else "-")
# key mismatch exemplars
hi_ref=sorted([r for r in rows if r[4]==0], key=lambda r:-r[2])[:3]
lo_cert=sorted([r for r in rows if r[4]==1], key=lambda r:r[2])[:3]
print("HIGH-AUROC but REFUSED:"); [print("  ",r[0][:50],round(r[2],3)) for r in hi_ref]
print("LOW-AUROC but CERTIFIED:"); [print("  ",r[0][:50],round(r[2],3),"ecc",round(r[5],3)) for r in lo_cert]
# overall correlation
import math
xs=[r[2] for r in rows]; ys=[r[5] for r in rows]
mx,my=statistics.mean(xs),statistics.mean(ys)
r_p=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
print("Pearson(bestAUROC, ECC) overall:",round(r_p,3))
# within-dataset correlation
for ds in ["cifar10_","cifar100","officehome","pathmnist"]:
    sub=[r for r in rows if ds in r[0]]
    if len(sub)<5: continue
    xs=[r[2] for r in sub]; ys=[r[5] for r in sub]
    mx,my=statistics.mean(xs),statistics.mean(ys)
    den=math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    print("Pearson within",ds,round(sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den,3) if den>0 else "nan")
with open(root+"/results/agent_extract/auroc_vs_cert.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["semantic_id","dataset","best_auroc","best_fpr95","H_certified","H_ecc"]); w.writerows(rows)
