"""Frozen V1 count-only certificate adapter. No image, label, or outcome input."""
import math
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, binom

METHODS = ('RATIO','MASS_CP','HOEFFDING_H','DIRECT_BETTING_H','SCALE_H','CP_IUT_H','COMPOSED_CP_BETTING_H')
STAKES = (.01,.02,.05,.10,.20,.40,.60,.80)

def lower(k, n, q):
    k,n=np.broadcast_arrays(k,n);out=np.zeros(k.shape,dtype=float);m=k>0
    out[m]=beta.ppf(q,k[m],n[m]-k[m]+1)
    return out

def upper(k,n,q):
    k,n=np.broadcast_arrays(k,n);out=np.ones(k.shape,dtype=float);m=(n>0)&(k<n)
    out[m]=beta.isf(q,k[m]+1,n[m]-k[m])
    return out

def holm(p, delta):
    p=np.asarray(p,float);order=np.argsort(p,axis=-1,kind='stable')
    sorted_p=np.take_along_axis(p,order,axis=-1)
    prefix=np.logical_and.accumulate(sorted_p<=delta/np.arange(p.shape[-1],0,-1),axis=-1)
    out=np.zeros(p.shape,bool);np.put_along_axis(out,order,prefix,axis=-1)
    return out

def certify(n,A,K,t,alpha=.2,dr=.05,dc=.05):
    """Arrays end in (12 candidates, J domains); arbitrary leading batch dims."""
    n,A,K=np.broadcast_arrays(np.asarray(n),np.asarray(A),np.asarray(K))
    if A.ndim<2 or A.shape[-2]!=12 or not (0<=t<=1): raise ValueError('Invalid frozen family or t')
    if not all(np.isfinite(x).all() and np.equal(x,np.floor(x)).all() for x in (n,A,K)):raise ValueError('Counts must be finite integers')
    if np.any(n<=0) or np.any(K<0) or np.any(K>A) or np.any(A>n):raise ValueError('Invalid count ordering')
    J=A.shape[-1];M=12
    vmax=lambda a:(1-t)*a.mean(-1)+t*a.max(-1)
    vmin=lambda a:(1-t)*a.mean(-1)+t*a.min(-1)
    common=vmin(lower(A,n,dc/(J*M)))
    tail=dr/(3*J*M);r=upper(K,A,tail)-alpha;al=lower(A,n,tail);au=upper(A,n,tail)
    ratio=vmax(np.maximum(r*al,r*au));den=vmin(al)
    mass=vmax((1-alpha)*upper(K,n,dr/(2*J*M))-alpha*lower(A-K,n,dr/(2*J*M)))
    pcp=binom.cdf(K,A,alpha);pcp[A==0]=1;pcp=pcp.max(-1)
    vertices=np.full((1,J),1/J) if t==0 else (1-t)*np.ones((J,J))/J+t*np.eye(J)
    L=np.maximum(1,np.ceil(np.log2(n.sum(-1))).astype(int))
    blog=[];slog=[];hp=[]
    for v in vertices:
        w=v/n;scale=w/((1-alpha)*w.max(-1,keepdims=True))
        def le(stake):
            z=stake*scale
            return (K*np.log1p(-(1-alpha)*z)+(A-K)*np.log1p(alpha*z)).sum(-1)
        lb=logsumexp(np.stack([le(s) for s in STAKES]),axis=0)-math.log(len(STAKES))
        ls=np.empty_like(lb)
        for length in np.unique(L):
            mask=L==length
            terms=[le(2.**(-ell))[mask]-math.log(ell*(ell+1)) for ell in range(1,int(length)+1)]
            terms.append(np.full(mask.sum(),-math.log(length+1)))
            ls[mask]=np.logaddexp(lb[mask],logsumexp(np.stack(terms),axis=0))-math.log(2)
        blog.append(lb);slog.append(ls)
        d=((K-alpha*A)/n*v).sum(-1);ss=(v*v/n).sum(-1)
        hp.append(np.exp(-2*np.minimum(d,0)**2/ss))
    logb=np.min(blog,axis=0);logs=np.min(slog,axis=0)
    pb=np.exp(np.minimum(0,-logb));ps=np.exp(np.minimum(0,-logs))
    stats={'common_acceptance_lcb':common,'ratio_D':ratio,'ratio_min_denominator':den,'mass_cp_D':mass,
           'hoeffding_p':np.max(hp,axis=0),'betting_p':pb,'scale_p':ps,'cp_iut_p':pcp,
           'composed_p':np.minimum(1,2*np.minimum(pcp,pb)),
           'betting_min_log_e':logb,'scale_min_log_e':logs,'worst_empirical_contrast':vmax((K-alpha*A)/n)}
    passed={'RATIO':(ratio<=0)&(den>0),'MASS_CP':mass<=0}
    for method,key in zip(METHODS[2:],['hoeffding_p','betting_p','scale_p','cp_iut_p','composed_p']):
        passed[method]=holm(stats[key],dr)
    if not all(np.isfinite(x).all() for x in stats.values()):raise ArithmeticError('Nonfinite certificate result')
    return stats,passed

def choose(stats,passed,eligible):
    """One model. Return candidate index or None; never drop multiplicity slots."""
    scores=['msp','energy','margin'];gammas=[.3,.5,.7,1.]
    if np.asarray(stats['common_acceptance_lcb']).shape!=(12,):raise ValueError('One complete family required')
    result={}
    for method,flags in passed.items():
        ids=[i for i in range(12) if eligible[i] and flags[i] and stats['common_acceptance_lcb'][i]>0]
        result[method]=min(ids,key=lambda i:(-stats['common_acceptance_lcb'][i],gammas[i%4],scores[i//4],i)) if ids else None
    return result
