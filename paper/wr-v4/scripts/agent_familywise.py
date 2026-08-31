"""Count-level familywise repeated-audit validation (Sec 5.3 extension).
Per cell: all 12 frozen members, NDRAW audits from true reservoir rates;
per rep apply Holm/IUT (delta_r=0.05, M=12 padded) and S (delta_r/M, delta_c/M).
Report familywise false-certification rates and selected-member acceptance-LCB coverage."""
import csv, gzip, os, sys
import numpy as np
from scipy.stats import beta, binom
root="/w"; ALPHA=0.2; DR=0.05; DC=0.05; M=12; NDRAW=1000
rng_master=np.random.default_rng(20260831)

def cp_upper_vec(K,A,eps):
    out=np.ones_like(K,dtype=float)
    m=(A>0)&(K<A)
    out[m]=beta.ppf(1.0-eps,K[m]+1,A[m]-K[m])
    out[(A>0)&(K>=A)]=1.0
    return out
def cp_lower_vec(A,n,eps):
    out=np.zeros_like(A,dtype=float)
    m=A>0
    out[m]=beta.ppf(eps,A[m],n[m]-A[m]+1)
    return out

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
    if pref+"__known_logits" in z.files:
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
    err=unk|(pred!=true)
    return sc,cl,err

out=[]
for ci_,(sid,cl_) in enumerate(sorted(cands.items())):
    z=np.load(resolve(pins[sid]),allow_pickle=True)
    scC,clC,errC=fold(z,"certification")
    J=int(clC.max())+1
    nres=np.array([(clC==j).sum() for j in range(J)])
    # per-member true rates
    A_p=[]; R_p=[]; unsafe=[]; amin=[]
    for score,slot,gamma,thr,feas in cl_:
        if not feas:
            A_p.append(np.zeros(J)); R_p.append(np.zeros(J)); unsafe.append(False); amin.append(0.0); continue
        key="native" if slot=="native" else score
        acc=scC[key]>=thr
        a=np.zeros(J); r=np.zeros(J)
        for j in range(J):
            m=clC==j
            a[j]=acc[m].mean() if m.any() else 0.0
            aj=acc&m
            r[j]=errC[aj].mean() if aj.any() else 0.0
        A_p.append(a); R_p.append(r)
        pos=a>0
        tgt=r[pos].max() if pos.any() else 0.0
        unsafe.append(bool(tgt>=ALPHA))
        amin.append(a[pos].min() if pos.any() else 0.0)
    A_p=np.array(A_p); R_p=np.array(R_p); unsafe=np.array(unsafe); amin=np.array(amin)
    Mreal=len(cl_)
    rng=np.random.default_rng(rng_master.integers(0,2**63))
    # simulate: shape (NDRAW, Mreal, J)
    A=rng.binomial(np.broadcast_to(nres,(NDRAW,Mreal,J)), np.broadcast_to(A_p,(NDRAW,Mreal,J)))
    K=rng.binomial(A, np.broadcast_to(R_p,(NDRAW,Mreal,J)))
    # p-values per member: max over strata of BinCDF(K,A,alpha); A=0 -> 1
    P=np.where(A>0, binom.cdf(K,np.maximum(A,1),ALPHA), 1.0).max(axis=2)  # (NDRAW,Mreal)
    if Mreal<M: P=np.concatenate([P,np.ones((NDRAW,M-Mreal))],axis=1)
    # Holm
    idx=np.argsort(P,axis=1); Ps=np.take_along_axis(P,idx,axis=1)
    thresh=DR/(M-np.arange(M))
    passseq=Ps<=thresh[None,:]
    cum=np.cumprod(passseq,axis=1).astype(bool)  # step-down: reject prefix
    rejected=np.zeros_like(P,dtype=bool)
    np.put_along_axis(rejected,idx,cum,axis=1)
    rejected=rejected[:,:Mreal]
    fw_H=(rejected & unsafe[None,:]).any(axis=1).mean()
    # S: per-member simultaneous
    U=cp_upper_vec(K.reshape(-1,J),A.reshape(-1,J),DR/M).reshape(NDRAW,Mreal,J).max(axis=2)
    L=cp_lower_vec(A.reshape(-1,J),np.broadcast_to(nres,(NDRAW,Mreal,J)).reshape(-1,J),DC/M).reshape(NDRAW,Mreal,J).min(axis=2)
    certS=(U<=ALPHA)&(L>0)
    fw_S=(certS & unsafe[None,:]).any(axis=1).mean()
    # selected member (S rule: max acceptance LCB among certified) acceptance-LCB coverage
    Lm=np.where(certS,L,-1.0)
    sel=Lm.argmax(axis=1); has=certS.any(axis=1)
    selL=Lm[np.arange(NDRAW),sel]
    cover_fail=((selL>amin[sel]+1e-12)&has).mean()
    out.append({"sid":sid,"J":J,"members":Mreal,"n_unsafe_members":int(unsafe.sum()),
                "fw_false_cert_H":fw_H,"fw_false_cert_S":fw_S,"sel_accLCB_fail":cover_fail})
    if (ci_+1)%50==0: print(ci_+1,"cells", file=sys.stderr)

with open(root+"/results/agent_extract/familywise_validity.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
fH=np.array([o["fw_false_cert_H"] for o in out]); fS=np.array([o["fw_false_cert_S"] for o in out])
cf=np.array([o["sel_accLCB_fail"] for o in out]); nu=np.array([o["n_unsafe_members"] for o in out])
print("cells:",len(out),"| cells with >=1 unsafe member:",int((nu>0).sum()))
print("Holm fw false-cert: max %.4f mean %.5f cells>0: %d cells>0.05: %d"%(fH.max(),fH.mean(),(fH>0).sum(),(fH>0.05).sum()))
print("S    fw false-cert: max %.4f mean %.5f cells>0.05: %d"%(fS.max(),fS.mean(),(fS>0.05).sum()))
print("sel accept-LCB fail: max %.4f mean %.5f cells>0.05: %d"%(cf.max(),cf.mean(),(cf>0.05).sum()))
