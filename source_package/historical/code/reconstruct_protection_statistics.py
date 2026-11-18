"""Independent vectorized risk-statistic reconstruction from original count triples.

Does not import the production certifier, its preprocessing, or its evaluators.
"""
from pathlib import Path
from collections import defaultdict
import csv, gzip, hashlib, json, math, time
import numpy as np
from scipy.stats import beta, binom
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
COUNT=ROOT/'inputs/results/theorem_aligned_wr_450_v3/primary_run/primary_candidate_counts.csv.gz'
DIAG=ROOT/'evidence/protection_recheck/results/candidate_diagnostics.csv.gz'
OUT=ROOT/'evidence/protection_statistics_reconstruction'
OUT.mkdir(exist_ok=True)
def rows(p):
    with gzip.open(p,'rt') as f:yield from csv.DictReader(f)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lo(k,n,q):
    out=np.zeros(k.shape);mask=k>0
    out[mask]=beta.ppf(q,k[mask],n[mask]-k[mask]+1)
    return out
def hi(k,n,q):
    out=np.ones(k.shape);mask=(n>0)&(k<n)
    out[mask]=beta.isf(q,k[mask]+1,n[mask]-k[mask]);return out

start=time.monotonic()
raw=defaultdict(dict)
for r in rows(COUNT):
    key=(r['semantic_id'],float(r['alpha']),int(r['candidate_index']))
    j=int(r['client']);assert j not in raw[key]
    raw[key][j]=(int(r['A']),int(r['K']),int(r['n']))
assert len(raw)==32400
arrays={key:np.array([r[j] for j in sorted(r)]) for key,r in raw.items()}
groups=defaultdict(list)
for r in rows(DIAG):groups[(float(r['alpha']),float(r['t']),int(r['J']))].append(r)
assert sum(map(len,groups.values()))==162000
maxerr=defaultdict(float);checks=defaultdict(int);boundary=[]
cache={}
for (alpha,t,J),rs in sorted(groups.items()):
    rs.sort(key=lambda r:(r['semantic_id'],int(r['candidate_index'])))
    ks=[(r['semantic_id'],alpha,int(r['candidate_index'])) for r in rs]
    triples=np.stack([arrays[k] for k in ks])
    A,K,N=(triples[:,:,i] for i in range(3))
    assert all(np.array_equal(np.array(json.loads(r[name])),triples[i,:,c]) for i,r in enumerate(rs) for name,c in [('A',0),('K',1),('n',2)])
    checks['original_count_rows_matched']+=len(rs)
    cachekey=(alpha,J)
    if cachekey not in cache:
        tail=.05/(3*J*12)
        rc=hi(K,A,tail)-alpha
        al=lo(A,N,tail);au=hi(A,N,tail)
        # Fixed-target maxima over rectangular conditional-risk regions.
        ratio=np.maximum(rc*al,rc*au)
        mass=(1-alpha)*hi(K,N,.05/(2*J*12))-alpha*lo(A-K,N,.05/(2*J*12))
        cp=binom.cdf(K,A,alpha);cp[A==0]=1
        cache[cachekey]=(lo(A,N,.05/(J*12)),al,ratio,mass,cp.max(1))
    lower,al,ratio,mass,pcp=cache[cachekey]
    vmax=lambda a:(1-t)*a.mean(1)+t*a.max(1)
    vmin=lambda a:(1-t)*a.mean(1)+t*a.min(1)
    vs=np.ones((1,J))/J if t==0 else (1-t)*np.ones((J,J))/J+t*np.eye(J)
    contrast=(K-alpha*A)/N
    vertexlogs=[];vertexscales=[];hps=[]
    L=np.maximum(1,np.ceil(np.log2(N.sum(1))).astype(int))
    for v in vs:
        w=v/N
        scale=w/((1-alpha)*w.max(1)[:,None])
        def stake_log(s):
            z=s*scale
            return (K*np.log1p(-(1-alpha)*z)+(A-K)*np.log1p(alpha*z)).sum(1)
        logs=np.stack([stake_log(s) for s in [.01,.02,.05,.10,.20,.40,.60,.80]])
        lb=logsumexp(logs,axis=0)-math.log(8)
        ls=np.empty_like(lb)
        for length in sorted(set(L)):
            mask=L==length
            dyad=np.stack([stake_log(2.**(-ell))[mask]-math.log(ell*(ell+1)) for ell in range(1,length+1)])
            dyad=np.vstack([dyad,np.full((1,mask.sum()),-math.log(length+1))])
            ls[mask]=np.logaddexp(lb[mask],logsumexp(dyad,axis=0))-math.log(2)
        vertexlogs.append(lb);vertexscales.append(ls)
        d=(contrast*v).sum(1);s=(v*v/N).sum(1)
        hps.append(np.exp(-2*np.minimum(d,0)**2/s))
    logb=np.min(vertexlogs,axis=0);logs=np.min(vertexscales,axis=0)
    pb=np.exp(np.minimum(0,-logb));ps=np.exp(np.minimum(0,-logs))
    computed={'common_acceptance_lcb':vmin(lower),'ratio_D':vmax(ratio),'ratio_min_denominator':vmin(al),
              'mass_cp_D':vmax(mass),'hoeffding_p':np.max(hps,axis=0),'betting_p':pb,'scale_p':ps,
              'cp_iut_p':pcp,'composed_p':np.minimum(1,2*np.minimum(pcp,pb)),
              'betting_min_log_e':logb,'scale_min_log_e':logs,'worst_empirical_contrast':vmax(contrast)}
    for name,actual in computed.items():
        expected=np.array([float(r[name]) for r in rs])
        err=np.abs(actual-expected);mx=float(err.max())
        maxerr[name]=max(maxerr[name],mx)
        assert np.allclose(actual,expected,atol=2e-11,rtol=2e-13),(alpha,t,J,name,mx)
        checks[name]+=len(rs)
    masks={'RATIO':(computed['ratio_D']<=0)&(computed['ratio_min_denominator']>0),
           'MASS_CP':computed['mass_cp_D']<=0}
    for name,pname in [('HOEFFDING_H','hoeffding_p'),('DIRECT_BETTING_H','betting_p'),('SCALE_H','scale_p'),('CP_IUT_H','cp_iut_p'),('COMPOSED_CP_BETTING_H','composed_p')]:
        p=computed[pname].reshape(-1,12)
        order=np.argsort(p,axis=1,kind='stable')
        sp=np.take_along_axis(p,order,axis=1)
        prefix=np.cumprod(sp<=(.05/np.arange(12,0,-1)),axis=1).astype(bool)
        mask=np.zeros_like(prefix);np.put_along_axis(mask,order,prefix,axis=1)
        masks[name]=mask.ravel()
    for name,mask in masks.items():
        expected=np.array([int(r['pass_'+name]) for r in rs],bool)
        assert np.array_equal(mask,expected),(alpha,t,J,name,np.sum(mask!=expected))
        checks['candidate_pass_flags']+=len(rs)
report={'status':'PASS','scope':'All seven risk constructions independently recomputed from original client triples, including non-CP statistics, common acceptance bounds, and candidate-level pass flags; no training, inference, or new audit draws.',
        'checks':dict(checks),'max_absolute_difference_by_quantity':dict(maxerr),'original_candidate_triples':len(raw),
        'candidate_protection_rows':162000,'inputs':{str(p.relative_to(ROOT)):digest(p) for p in [COUNT,DIAG]},
        'script_sha256':digest(Path(__file__)),'elapsed_seconds':time.monotonic()-start,
        'limits':'Numerical reconstruction establishes code-to-formula agreement on archived inputs. It does not establish historical sample independence, unseen-population transfer, or fresh empirical superiority.'}
(OUT/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
