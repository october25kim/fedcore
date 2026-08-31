import numpy as np, csv, os, json
root="/data/workspace/sanghoon/fedcore2"
pins=root+"/results/theorem_aligned_wr_450_v3/governing/raw_input_pins.csv"
outdir=root+"/results/agent_extract"; os.makedirs(outdir,exist_ok=True)

def resolve(p):
    for cand in (root+"/"+p, root+"/results/confirmatory_400r/"+p):
        if os.path.exists(cand): return cand
    raise FileNotFoundError(p)

def auroc(sk, su):
    # P(known score > unknown score) + 0.5 ties, rank-based
    s=np.concatenate([sk,su]); r=np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort").astype(float)+1.0
    # handle ties via average ranks
    order=np.argsort(s, kind="mergesort"); ranks=np.empty(len(s)); ranks[order]=np.arange(1,len(s)+1)
    sv=s[order]; i=0
    while i<len(sv):
        j=i
        while j+1<len(sv) and sv[j+1]==sv[i]: j+=1
        if j>i: ranks[order[i:j+1]]=(i+1+j+1)/2.0
        i=j+1
    Rk=ranks[:len(sk)].sum()
    return (Rk-len(sk)*(len(sk)+1)/2.0)/(len(sk)*len(su))

def fpr_at_95tpr(sk,su):
    t=np.quantile(sk,0.05)  # accept if score>=t keeps ~95% of knowns
    return float((su>=t).mean())

def softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

arows=[]; trows=[]
with open(pins) as f:
    for row in csv.DictReader(f):
        sid=row["semantic_id"]; path=resolve(row["path"]) 
        z=np.load(path, allow_pickle=True)
        if "certification__known_logits" in z.files:  # cifar / pathmnist style
            unk=z["certification__known_or_unknown"]=="unknown"
            scores={"native":z["certification__native_score"],
                    "energy":z["certification__energy_score"],
                    "margin":z["certification__known_margin_score"]}
            ds=row["kind"]
        else:  # officehome style
            L=z["cert_logits"]; unk=z["cert_y_open"]==-1
            P=softmax(L)
            srt=np.sort(P,axis=1)
            scores={"native":P.max(axis=1),
                    "energy":np.log(np.exp(L-L.max(axis=1,keepdims=True)).sum(axis=1))+L.max(axis=1),
                    "margin":srt[:,-1]-srt[:,-2]}
            ds="officehome"
            tc=np.bincount(z["traffic_client"],minlength=4)
            trows.append([sid,str(z["pipeline_name"]),str(z["split_id"]),int(z["train_rep"]),
                          int(z["traffic_draw_seed"]),len(z["traffic_client"])]+tc.tolist())
        for name,s in scores.items():
            s=np.asarray(s,dtype=float)
            sk,su=s[~unk],s[unk]
            arows.append([sid,ds,name,len(sk),len(su),round(auroc(sk,su),6),round(fpr_at_95tpr(sk,su),6)])

with open(outdir+"/auroc_cert_fold.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["semantic_id","dataset","score","n_known","n_unknown","auroc","fpr_at_95tpr"]); w.writerows(arows)
with open(outdir+"/officehome_traffic.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["semantic_id","arm","split_id","train_rep","traffic_draw_seed","traffic_N","c0","c1","c2","c3"]); w.writerows(trows)
print("auroc rows:",len(arows),"traffic rows:",len(trows))
