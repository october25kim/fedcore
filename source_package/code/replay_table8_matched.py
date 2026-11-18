"""Portable, count-only Table 8 comparison. See the frozen PROTOCOL.json."""
import csv,gzip,json,math,hashlib,sys,itertools
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from scipy.stats import beta,binom
from scipy.special import logsumexp
from scipy.optimize import linprog
import mpmath as mp
from frozen_v1_certifier import holm,METHODS,STAKES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
mp.mp.dps=60
S=10**11
alpha=F(1,5);er=F(1,3600);ec=F(1,1200);em=F(1,2400);et=F(1,400)

def cdf_num(k,n,a,b):
    """Exact numerator of Bin(n,a/b) CDF with denominator b**n."""
    if k<0:return 0,b**n
    if k>=n:return b**n,b**n
    if a==0:return b**n,b**n
    if a==b:return 0,b**n
    c=b-a;term=c**n;total=term
    for i in range(k):
        term=term*(n-i)*a//((i+1)*c)
        total+=term
    return total,b**n

@lru_cache(None)
def cp(k,n,q,side):
    k,n=int(k),int(n)
    if side=='lower':
        if k==0:return F(0)
        p=F(math.floor(float(beta.ppf(float(q),k,n-k+1))*S),S)
        a,b=cdf_num(n-k,n,(1-p).numerator,(1-p).denominator)
    else:
        if n==0 or k==n:return F(1)
        p=F(math.ceil(float(beta.isf(float(q),k+1,n-k))*S),S)
        a,b=cdf_num(k,n,p.numerator,p.denominator)
    assert a*q.denominator<=b*q.numerator,(k,n,q,side,p)
    return p

def vertices(lo,hi):
    vs=set();J=len(lo)
    for free in range(J):
        fixed=[j for j in range(J) if j!=free]
        for bits in itertools.product((0,1),repeat=J-1):
            v=[F(0)]*J
            for j,b in zip(fixed,bits):v[j]=hi[j] if b else lo[j]
            v[free]=1-sum(v)
            if lo[free]<=v[free]<=hi[free]:vs.add(tuple(v))
    assert vs
    return sorted(vs)

def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def fm(v):return mp.mpf(v.numerator)/v.denominator

def exact_holm(ps):
    order=sorted(range(12),key=lambda i:(ps[i],i));flags=[False]*12
    for rank,i in enumerate(order):
        if ps[i]<=mp.mpf('.04')/(12-rank):flags[i]=True
        else:break
    return flags

def main():
    proto=json.loads((ROOT/'PROTOCOL.json').read_text())
    for name,h in proto['source_hashes'].items():assert hashlib.sha256((ROOT/'input'/name).read_bytes()).hexdigest()==h
    cells={}
    with gzip.open(ROOT/'input/primary_candidate_counts.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            if r['dataset']!='officehome' or float(r['alpha'])!=.2:continue
            sid=r['semantic_id'];ci=int(r['candidate_index']);j=int(r['client'])
            d=cells.setdefault(sid,{}).setdefault(ci,{'n':[0]*4,'A':[0]*4,'K':[0]*4,'feasible':int(r['proposal_feasible']),'score':r.get('score',''),'gamma':r.get('gamma',''),'metadata':r})
            for key in ('n','A','K'):d[key][j]=int(r[key])
    assert len(cells)==100 and all(len(c)==12 for c in cells.values())
    traffic={r['semantic_id']:r for r in json.loads((ROOT/'input/traffic_payload.json').read_text())}
    ref={r['semantic_id']:r for r in json.loads((ROOT/'input/reference_cells.json').read_text())}
    refs={(r['semantic_id'],int(r['candidate_index'])):r for r in csv.DictReader((ROOT/'input/reference_candidates.csv').open())}
    rows=[];candidate_rows=[];solver=[];max_p_diff=0.;mismatches=0;ref_mismatches=0;max_ucb_diff=0.
    for ix,(sid,cands) in enumerate(sorted(cells.items())):
        t=traffic[sid];rng=np.random.default_rng(np.random.SeedSequence([t['traffic_draw_seed'],1000]));tc=np.asarray(t['traffic_client'])
        draw=np.bincount(tc[rng.integers(0,len(tc),size=1000)],minlength=4)
        assert draw.tolist()==ref[sid]['traffic_draw_counts']
        lo=[cp(int(k),1000,et,'lower') for k in draw];hi=[cp(int(k),1000,et,'upper') for k in draw]
        vs=vertices(lo,hi);vf=np.asarray(vs,float)
        n=np.array([cands[i]['n'] for i in range(12)]);A=np.array([cands[i]['A'] for i in range(12)]);K=np.array([cands[i]['K'] for i in range(12)])
        assert np.all(n>0) and np.all((K>=0)&(K<=A)&(A<=n))
        coverage=[];rflags=[];mflags=[];ucs=[];stats=[]
        for i in range(12):
            al=[cp(a,nn,er,'lower') for a,nn in zip(A[i],n[i])];au=[cp(a,nn,er,'upper') for a,nn in zip(A[i],n[i])]
            ru=[cp(k,a,er,'upper') for k,a in zip(K[i],A[i])];cl=[cp(a,nn,ec,'lower') for a,nn in zip(A[i],n[i])]
            common=min(dot(v,cl) for v in vs);coverage.append(common)
            den=min(dot(v,al) for v in vs)
            # A linear-fractional function reaches an extreme at a traffic vertex
            # and an acceptance-box corner when denominators stay positive.
            if den>0:
                u=max(dot(v,[a*r for a,r in zip(aa,ru)])/dot(v,aa) for v in vs for aa in itertools.product(*zip(al,au)))
            else:u=F(1)
            ucs.append(u);rflags.append(den>0 and u<=alpha)
            g=[max((r-alpha)*a,(r-alpha)*b) for r,a,b in zip(ru,al,au)]
            dmax=max(dot(v,g) for v in vs)
            assert (den>0 and dmax<=0)==rflags[-1]
            mass=[(1-alpha)*cp(k,nn,em,'upper')-alpha*cp(a-k,nn,em,'lower') for a,k,nn in zip(A[i],K[i],n[i])]
            md=max(dot(v,mass) for v in vs);mflags.append(md<=0)
            # An independent LP solve validates the rational vertex calculation.
            lp=linprog(-np.asarray(g,float),A_eq=np.ones((1,4)),b_eq=[1],bounds=list(zip(map(float,lo),map(float,hi))),method='highs')
            assert lp.success and abs(-lp.fun-float(dmax))<1e-9
            solver.append({'semantic_id':sid,'candidate_index':i,'vertices':len(vs),'ratio_ucb_exact':str(u),'ratio_D_exact':str(dmax),'min_denominator_exact':str(den),'acceptance_lcb_exact':str(common),'mass_D_exact':str(md),'lp_gap':abs(-lp.fun-float(dmax)),'max_vertex_residual':max(abs(sum(v)-1) for v in vs).__float__()})
            if (sid,i) in refs:
                rr=refs[sid,i];ref_mismatches+=int(rflags[-1]!=bool(int(rr['corrected_pass'])))
                max_ucb_diff=max(max_ucb_diff,abs(float(u)-float(rr['corrected_risk_ucb'])))
        L=np.maximum(1,np.ceil(np.log2(n.sum(-1))).astype(int));bp=[];sp=[];hp=[]
        for v in vf:
            w=v/n;scale=w/(.8*w.max(-1,keepdims=True))
            def le(s):
                z=s*scale
                return (K*np.log1p(-.8*z)+(A-K)*np.log1p(.2*z)).sum(-1)
            lb=logsumexp(np.stack([le(s) for s in STAKES]),axis=0)-math.log(8)
            ls=np.empty(12)
            for i,length in enumerate(L):
                small=[le(2.**(-ell))[i]-math.log(ell*(ell+1)) for ell in range(1,int(length)+1)]+[-math.log(length+1)]
                ls[i]=np.logaddexp(lb[i],logsumexp(small))-math.log(2)
            bp.append(np.exp(np.minimum(0,-lb)));sp.append(np.exp(np.minimum(0,-ls)))
            d=((K-.2*A)/n*v).sum(-1);hp.append(np.exp(-2*np.minimum(d,0)**2/(v*v/n).sum(-1)))
        pcp=binom.cdf(K,A,.2);pcp[A==0]=1;pcp=pcp.max(-1)
        ps={'HOEFFDING_H':np.max(hp,axis=0),'DIRECT_BETTING_H':np.max(bp,axis=0),'SCALE_H':np.max(sp,axis=0),'CP_IUT_H':pcp}
        ps['COMPOSED_CP_BETTING_H']=np.minimum(1,2*np.minimum(pcp,ps['DIRECT_BETTING_H']))
        passed={'RATIO':rflags,'MASS_CP':mflags,**{k:holm(v,.04).tolist() for k,v in ps.items()}}
        # High-precision reconstruction uses products, not log-sum-exp.
        high={k:[] for k in ps}
        for i in range(12):
            hh=[];bb=[];ss=[]
            for v in vs:
                w=[fm(vj)/int(nn) for vj,nn in zip(v,n[i])];wm=max(w)
                def ev(s):return mp.fprod((1-s*wj/wm)**int(k)*(1+s*wj/wm/4)**int(a-k) for wj,a,k in zip(w,A[i],K[i]))
                e=sum(ev(mp.mpf(str(s))) for s in STAKES)/8
                es=(e+sum(ev(mp.mpf(2)**(-ell))/(ell*(ell+1)) for ell in range(1,int(L[i])+1))+mp.mpf(1)/(int(L[i])+1))/2
                bb.append(min(mp.mpf(1),1/e));ss.append(min(mp.mpf(1),1/es))
                d=sum(fm(vj)*(int(k)-fm(alpha)*int(a))/int(nn) for vj,a,k,nn in zip(v,A[i],K[i],n[i]));sv=sum(fm(vj)**2/int(nn) for vj,nn in zip(v,n[i]))
                hh.append(mp.exp(-2*min(0,d)**2/sv))
            pp=[]
            for a,k in zip(A[i],K[i]):
                aa,bbn=cdf_num(int(k),int(a),1,5);pp.append(mp.mpf(aa)/bbn)
            p=max(pp);b=max(bb)
            for key,val in [('HOEFFDING_H',max(hh)),('DIRECT_BETTING_H',b),('SCALE_H',max(ss)),('CP_IUT_H',p),('COMPOSED_CP_BETTING_H',min(1,2*min(p,b)))]:high[key].append(val)
        for key,vals in high.items():
            dif=max(abs(float(v)-p) for v,p in zip(vals,ps[key]));max_p_diff=max(max_p_diff,dif)
            flags=exact_holm(vals);mismatches+=sum(a!=b for a,b in zip(flags,passed[key]))
            assert dif<1e-10 and flags==passed[key]
            # Isolate every decision from floating-point threshold ambiguity.
            order=np.argsort(ps[key],kind='stable')
            assert all(abs(ps[key][i]-.04/(12-rank))>1e-12 for rank,i in enumerate(order))
        for method in METHODS:
            ids=[i for i in range(12) if cands[i]['feasible'] and passed[method][i] and coverage[i]>0]
            chosen=min(ids,key=lambda i:(-coverage[i],[.3,.5,.7,1.][i%4],['msp','energy','margin'][i//4],i)) if ids else None
            rows.append({'semantic_id':sid,'method':method,'selected_candidate':chosen if chosen is not None else -1,'certified':int(chosen is not None),'eca':float(coverage[chosen]) if chosen is not None else 0,'feasible_candidates':sum(c['feasible'] for c in cands.values())})
            if method=='RATIO':
                old=ref[sid]['best']['corrected'];assert (chosen is None)==(old is None)
                if old is not None:assert abs(float(coverage[chosen])-old['lcb'])<1e-8
            for i in range(12):candidate_rows.append({'semantic_id':sid,'candidate_index':i,'method':method,'proposal_feasible':cands[i]['feasible'],'risk_pass':int(passed[method][i]),'acceptance_lcb':float(coverage[i]),'p_value':float(ps[method][i]) if method in ps else '', 'ratio_ucb':float(ucs[i]) if method=='RATIO' else ''})
        if ix%10==9:print(f'{ix+1}/100 cells verified; exact CP endpoint checks {cp.cache_info().currsize}',flush=True)
    assert len(rows)==700 and len(candidate_rows)==8400 and len(solver)==1200
    summ={m:{'cells':100,'certified':sum(r['certified'] for r in rows if r['method']==m),'eca':float(np.mean([r['eca'] for r in rows if r['method']==m])),'sd':float(np.std([r['eca'] for r in rows if r['method']==m],ddof=1))} for m in METHODS}
    for name,data in [('outcomes.csv',rows),('candidate_method_flags.csv',candidate_rows),('solver_verification.csv',solver)]:
        with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    report={'status':'PASS_REPLAY_AND_NUMERICAL_VERIFICATION','scope':'Retrospective reference-setting comparison only; no new training or population evidence','summary':summ,'denominators':{'cells':100,'count_rows':4800,'candidate_states':1200,'flags':8400,'outcomes':700},'exact_CP_endpoints':cp.cache_info().currsize,'high_precision_p_max_abs_difference':max_p_diff,'high_precision_Holm_flag_mismatches':mismatches,'reference_RATIO_flag_mismatches':ref_mismatches,'reference_RATIO_ucb_max_abs_difference':max_ucb_diff,'max_LP_vertex_gap':max(s['lp_gap'] for s in solver),'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),'versions':{'python':sys.version,'numpy':np.__version__,'scipy':__import__('scipy').__version__,'mpmath':mp.__version__}}
    assert ref_mismatches==0
    (OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
