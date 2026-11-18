"""Frozen retrospective TV/direct comparison. No image or future-outcome access."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,csv,gzip,itertools,math
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import logsumexp
from scipy.stats import binom
import mpmath as mp
R=Path(__file__).resolve().parents[1];S=R/'source_package';O=R/'results/tv';O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(S/'code'))
from replay_table8_matched import cp,vertices,dot,fm,exact_holm,er,ec,em,et,STAKES
from frozen_v1_certifier import holm
mp.mp.dps=60
TV=['TV_MASS_CP','TV_HOEFFDING_H','TV_DIRECT_BETTING_H','TV_SCALE_H','TV_COMPOSED_CP_BETTING_H']

def write(name,rows):pd.DataFrame(rows).to_csv(O/name,index=False)
def center_minimax(vs):
 rows=[];b=[]
 for mask in itertools.product((0,1),repeat=4):
  rows.append([-m for m in mask]+[-1]);b.append(-max(float(dot(v,mask)) for v in vs))
 res=linprog([0,0,0,0,1],A_ub=rows,b_ub=b,A_eq=[[1,1,1,1,0]],b_eq=[1],bounds=[(0,1)]*4+[(0,1)],method='highs');assert res.success
 ints=[max(0,int(round(x*10**11))) for x in res.x[:4]];w=[F(x,sum(ints)) for x in ints];eta=max(sum(abs(a-b) for a,b in zip(v,w))/2 for v in vs);gap=float(eta)-res.fun;assert -1e-8<=gap<=1e-8
 return w,eta,gap

def shifted(n,A,K,w,eta):
 wf=np.array(w,float);ef=float(eta);wt=wf/n;wm=wt.max(-1,keepdims=True);den=(.8+ef)*wm
 def le(s):
  z=s*wt/den
  return (K*np.log1p(-z*(.8+ef))+(A-K)*np.log1p(z*(.2-ef))+(n-A)*np.log1p(-z*ef)).sum(-1)
 lb=logsumexp(np.stack([le(s) for s in STAKES]),axis=0)-math.log(8);L=np.maximum(1,np.ceil(np.log2(n.sum(-1))).astype(int));ls=np.empty(12)
 for i,length in enumerate(L):
  terms=[le(2.**(-ell))[i]-math.log(ell*(ell+1)) for ell in range(1,int(length)+1)]+[-math.log(length+1)]
  ls[i]=np.logaddexp(lb[i],logsumexp(terms))-math.log(2)
 dh=((K-.2*A)/n*wf).sum(-1)+ef;sv=(wf*wf/n).sum(-1)
 ps={'TV_HOEFFDING_H':np.exp(-2*np.minimum(dh,0)**2/sv),'TV_DIRECT_BETTING_H':np.exp(np.minimum(0,-lb)),'TV_SCALE_H':np.exp(np.minimum(0,-ls))}
 pc=binom.cdf(K,A,.2);pc[A==0]=1;pc=pc.max(-1);ps['TV_COMPOSED_CP_BETTING_H']=np.minimum(1,2*np.minimum(ps['TV_DIRECT_BETTING_H'],pc))
 # Independent 60-digit products for all twelve candidates and every stake.
 high={k:[] for k in ps}
 for i in range(12):
  z=[fm(v)/int(nn) for v,nn in zip(w,n[i])];zm=max(z);ee=fm(eta);aa=mp.mpf(1)/5
  def ev(s):
   h=s/((1-aa+ee)*zm)
   return mp.fprod((1-h*v*(1-aa+ee))**int(k)*(1+h*v*(aa-ee))**int(a-k)*(1-h*v*ee)**int(nn-a) for v,nn,a,k in zip(z,n[i],A[i],K[i]))
  eb=sum(ev(mp.mpf(str(s))) for s in STAKES)/8
  es=(eb+sum(ev(mp.mpf(2)**(-ell))/(ell*(ell+1)) for ell in range(1,int(L[i])+1))+mp.mpf(1)/(int(L[i])+1))/2
  dd=sum(fm(v)*(int(k)-aa*int(a))/int(nn) for v,nn,a,k in zip(w,n[i],A[i],K[i]))+ee
  vv=sum(fm(v)**2/int(nn) for v,nn in zip(w,n[i]))
  pp=max(mp.fsum(mp.binomial(int(a),j)*aa**j*(1-aa)**(int(a)-j) for j in range(int(k)+1)) if a else mp.mpf(1) for a,k in zip(A[i],K[i]))
  bb=min(mp.mpf(1),1/eb)
  for key,value in [('TV_HOEFFDING_H',mp.exp(-2*min(0,dd)**2/vv)),('TV_DIRECT_BETTING_H',bb),('TV_SCALE_H',min(mp.mpf(1),1/es)),('TV_COMPOSED_CP_BETTING_H',min(1,2*min(bb,pp)))]:high[key].append(value)
 diff=0.;margin=1.
 for k,p in ps.items():
  diff=max(diff,max(abs(float(a)-b) for a,b in zip(high[k],p)));assert diff<1e-10
  assert exact_holm(high[k])==holm(p,.04).tolist()
  order=np.argsort(p,kind='stable');margin=min(margin,min(abs(p[i]-.04/(12-rank)) for rank,i in enumerate(order)));assert margin>1e-12
 return ps,diff,margin

def main():
 proto=json.loads((R/'protocols/TV_PROTOCOL.json').read_text())
 for p,h in proto['input_hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
 cells={}
 with gzip.open(S/'input/primary_candidate_counts.csv.gz','rt') as f:
  for r in csv.DictReader(f):
   if r['dataset']!='officehome' or float(r['alpha'])!=.2:continue
   d=cells.setdefault(r['semantic_id'],{}).setdefault(int(r['candidate_index']),{'n':[0]*4,'A':[0]*4,'K':[0]*4,'feasible':int(r['proposal_feasible'])})
   for k in ['n','A','K']:d[k][int(r['client'])]=int(r[k])
 assert len(cells)==100 and all(len(c)==12 for c in cells.values())
 traffic={r['semantic_id']:r for r in json.loads((S/'input/traffic_payload.json').read_text())};refs={r['semantic_id']:r for r in json.loads((S/'input/reference_cells.json').read_text())}
 archived=pd.read_csv(S/'results/solver_verification.csv').set_index(['semantic_id','candidate_index'])
 rows=[];flags=[];mechs=[];geometry=[];maxdiff=0.;minmargin=1.;maxgap=0.
 for ix,(sid,cands) in enumerate(sorted(cells.items())):
  arm='B' if 'frozen' in sid else 'A';t=traffic[sid];rng=np.random.default_rng(np.random.SeedSequence([t['traffic_draw_seed'],1000]));tc=np.asarray(t['traffic_client']);draw=np.bincount(tc[rng.integers(0,len(tc),size=1000)],minlength=4);assert draw.tolist()==refs[sid]['traffic_draw_counts']
  lo=[cp(k,1000,et,'lower') for k in draw];hi=[cp(k,1000,et,'upper') for k in draw];vs=vertices(lo,hi)
  wc=[F(int(k),1000) for k in draw];eta=max(sum(abs(a-b) for a,b in zip(v,wc))/2 for v in vs);wm,emini,gap=center_minimax(vs);maxgap=max(maxgap,abs(gap))
  n=np.array([cands[i]['n'] for i in range(12)]);A=np.array([cands[i]['A'] for i in range(12)]);K=np.array([cands[i]['K'] for i in range(12)])
  mass=[];cl=[];common=[];empirical=[]
  for i in range(12):
   mass.append([F(4,5)*cp(k,nn,em,'upper')-F(1,5)*cp(a-k,nn,em,'lower') for a,k,nn in zip(A[i],K[i],n[i])]);cl.append([cp(a,nn,ec,'lower') for a,nn in zip(A[i],n[i])]);common.append(min(dot(v,cl[-1]) for v in vs));empirical.append([F(int(k)*5-int(a),int(nn)*5) for a,k,nn in zip(A[i],K[i],n[i])])
   ar=archived.loc[(sid,i)];assert common[-1]==F(ar.acceptance_lcb_exact) and max(dot(v,mass[-1]) for v in vs)==F(ar.mass_D_exact)
  for cname,w,eta in [('empirical',wc,eta),('minimax',wm,emini)]:
   geometry.append({'semantic_id':sid,'arm':arm,'center':cname,'w':json.dumps(list(map(str,w))),'eta_exact':str(eta),'eta':float(eta),'vertices':len(vs),'minimax_LP_gap':gap if cname=='minimax' else ''})
   ps,diff,margin=shifted(n,A,K,w,eta);maxdiff=max(maxdiff,diff);minmargin=min(minmargin,margin)
   tvmass=[dot(w,g)+eta for g in mass];passed={'TV_MASS_CP':[g<=0 for g in tvmass],**{k:holm(p,.04).tolist() for k,p in ps.items()}}
   for i in range(12):
    direct=max(dot(v,mass[i]) for v in vs);demp=max(dot(v,empirical[i]) for v in vs);slack=tvmass[i]-direct;assert slack>=0 and direct-demp>=0
    native=dot(w,cl[i])-eta
    mechs.append({'semantic_id':sid,'arm':arm,'candidate_index':i,'center':cname,'proposal_feasible':cands[i]['feasible'],'n_total':int(sum(n[i])),'accepted_total':int(sum(A[i])),'errors_total':int(sum(K[i])),'empirical_worst_D':float(demp),'mass_endpoint_inflation':float(direct-demp),'direct_mass_D':float(direct),'TV_mass_D':float(tvmass[i]),'TV_geometry_slack':float(slack),'TV_geometry_slack_exact':str(slack),'eta':float(eta),'native_TV_acceptance':float(native),'common_acceptance':float(common[i]),'empirical_infeasible':int(demp>0),'mass_confidence_block':int(demp<=0 and direct>0),'TV_geometry_block':int(direct<=0 and tvmass[i]>0),'acceptance_block':int(common[i]<=0)})
   for method in TV:
    ids=[i for i in range(12) if cands[i]['feasible'] and passed[method][i] and common[i]>0]
    chosen=min(ids,key=lambda i:(-common[i],[.3,.5,.7,1.][i%4],['msp','energy','margin'][i//4],i)) if ids else -1
    rows.append({'semantic_id':sid,'arm':arm,'center':cname,'method':method,'certified':int(chosen>=0),'selected_candidate':chosen,'eca':float(common[chosen]) if chosen>=0 else 0.})
    for i in range(12):flags.append({'semantic_id':sid,'arm':arm,'center':cname,'method':method,'candidate_index':i,'proposal_feasible':cands[i]['feasible'],'risk_pass':int(passed[method][i]),'common_acceptance':float(common[i]),'native_TV_acceptance':float(dot(w,cl[i])-eta),'p_value':float(ps[method][i]) if method in ps else ''})
  if ix%10==9:print(f'{ix+1}/100; max p difference={maxdiff:.3g}',flush=True)
 assert len(rows)==1000 and len(flags)==12000 and len(mechs)==2400
 write('outcomes.csv',rows);write('candidate_flags.csv',flags);write('mechanism.csv',mechs);write('traffic_geometry.csv',geometry)
 df=pd.DataFrame(rows);original=pd.read_csv(S/'results/outcomes.csv');original['arm']=np.where(original.semantic_id.str.contains('frozen'),'B','A');original['center']='direct'
 allrows=pd.concat([original,df],ignore_index=True);allrows.to_csv(O/'all_outcomes.csv',index=False)
 agg=allrows.groupby(['center','method']).agg(cells=('eca','size'),certified=('certified','sum'),eca=('eca','mean')).reset_index();agg.to_csv(O/'summary.csv',index=False)
 allrows.groupby(['center','method','arm']).agg(cells=('eca','size'),certified=('certified','sum'),eca=('eca','mean')).reset_index().to_csv(O/'by_arm.csv',index=False)
 contrasts=[];decisions={}
 for center in ['empirical','minimax']:
  candidates=[]
  for method in proto['comparison_methods']:
   good=True
   for tv in proto['TV_methods']:
    a=original[original.method==method].set_index('semantic_id');b=df[(df.method==tv)&(df.center==center)].set_index('semantic_id');d=a.eca-b.eca;armvals={arm:float(d[a.arm==arm].mean()) for arm in ['A','B']};good &= bool(d.mean()>=.01 and min(armvals.values())>0)
    contrasts.append({'center':center,'direct':method,'TV':tv,'cells':100,'mean_difference':float(d.mean()),'A_difference':armvals['A'],'B_difference':armvals['B'],'wins':int((d>1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'losses':int((d< -1e-12).sum())})
   if good:candidates.append(method)
  besttv=float(df[(df.center==center)&df.method.isin(proto['TV_methods'])].groupby('method').eca.mean().max());bestdirect=float(original[original.method.isin(proto['comparison_methods'])].groupby('method').eca.mean().max())
  decisions[center]={'decision':'PASS_CONDITIONAL_TV_COMPARISON' if candidates else 'KILL_DIRECT_UTILITY_CLAIM' if bestdirect<=besttv else 'HOLD','qualifying_methods':candidates}
 write('contrasts.csv',contrasts)
 report={'status':'VALID','decisions':decisions,'denominators':{'cells':100,'TV_outcomes':1000,'TV_candidate_flags':12000,'candidate_center_mechanism_rows':2400,'historical_outcomes_retained':700},'max_high_precision_p_difference':maxdiff,'minimum_Holm_threshold_distance':minmargin,'max_minimax_radius_LP_gap':maxgap,'exact_CP_endpoint_checks':cp.cache_info().currsize,'protocol_sha256':hashlib.sha256((R/'protocols/TV_PROTOCOL.json').read_bytes()).hexdigest(),'scope':'Fixed counts and traffic frames only. Not a Fed-SRC private/anytime reproduction. Native TV acceptance is separately preserved; primary objective is shared.'}
 (O/'DECISION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
