"""Exhaustive, descriptive count-state mechanism audit. No policy fitting."""
from pathlib import Path
import json,sys,itertools,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'source_package'
sys.path.insert(0,str(SRC/'code'))
import frozen_v1_certifier as cert
OUT=ROOT/'results'/'mechanism';OUT.mkdir(exist_ok=True)
s=pd.read_csv(SRC/'e2/STATES.csv');old=pd.read_csv(SRC/'e2/RANKINGS.csv').set_index('state_id')
stats_rows=[];values={};maxdiff=0.;mismatches=0
for r in s.to_dict('records'):
    n=np.full((12,4),r['n']);a=np.tile([r[f'A{j}'] for j in range(4)],(12,1));k=np.tile([r[f'K{j}'] for j in range(4)],(12,1))
    st,flags=cert.certify(n,a,k,r['t']);v={}
    row={x:r[x] for x in ['state_id','n','f_index','q_index','t']}
    row.update({key:float(val[0]) for key,val in st.items()})
    row.update(accepted_total=int(a[0].sum()),errors_total=int(k[0].sum()),mean_acceptance=float(a[0].mean()/r['n']),empirical_conditional_error=float(k[0].sum()/a[0].sum()) if a[0].sum() else None)
    row['realized_count_key']=json.dumps([r['n'],*a[0].tolist(),*k[0].tolist(),r['t']])
    for m in cert.METHODS:
        assert (flags[m]==flags[m][0]).all()
        v[m]=float(st['common_acceptance_lcb'][0]) if flags[m][0] and st['common_acceptance_lcb'][0]>0 else 0.
        maxdiff=max(maxdiff,abs(v[m]-old.loc[r['state_id'],'ECA_'+m]))
        mismatches+=int(int(flags[m].sum())!=int(old.loc[r['state_id'],'n_pass_'+m]))
        row['ECA_'+m]=v[m]
    values[r['state_id']]=v;stats_rows.append(row)
assert maxdiff<1e-12 and mismatches==0
states=pd.DataFrame(stats_rows);states['duplicate_realized_state']=states.duplicated('realized_count_key',keep=False)
states.to_csv(OUT/'state_statistics.csv',index=False)
pairrows=[];changes=[]
for axis in ['n','f_index','q_index']:
    fixed=[x for x in ['n','f_index','q_index','t'] if x!=axis]
    for _,g in s.groupby(fixed):
        for left,right in itertools.combinations(g.sort_values(axis).to_dict('records'),2):
            a,b=values[left['state_id']],values[right['state_id']]
            strict=0;orderchange=0
            for m1,m2 in itertools.combinations(cert.METHODS,2):
                x=np.sign(a[m1]-a[m2]);y=np.sign(b[m1]-b[m2]);rev=bool(x*y<0);change=bool(x!=y)
                strict+=rev;orderchange+=change
                changes.append(dict(axis=axis,left=left['state_id'],right=right['state_id'],t=left['t'],method1=m1,method2=m2,left_sign=int(x),right_sign=int(y),strict_reversal=rev,order_changed=change))
            pairrows.append(dict(axis=axis,left=left['state_id'],right=right['state_id'],t=left['t'],left_value=left[axis],right_value=right[axis],strict_reversal_pairs=strict,order_changes=orderchange,any_strict_reversal=strict>0,any_order_change=orderchange>0,identical_realized_counts=all(left[k]==right[k] for k in ['n']+[f'{a}{j}' for a in ['A','K'] for j in range(4)])))
pairs=pd.DataFrame(pairrows);details=pd.DataFrame(changes)
assert len(pairs)==1152 and len(details)==24192
pairs.to_csv(OUT/'one_factor_state_pairs.csv',index=False);details.to_csv(OUT/'all_method_pair_comparisons.csv.gz',index=False,compression='gzip')
summary=pairs.groupby(['axis','t']).agg(pairs=('left','size'),strict_reversal_state_pairs=('any_strict_reversal','sum'),order_change_state_pairs=('any_order_change','sum'),identical_realized_pairs=('identical_realized_counts','sum')).reset_index()
summary.to_csv(OUT/'one_factor_summary.csv',index=False)
contr=[]
for t,g in states.groupby('t'):
    for m1,m2 in itertools.combinations(cert.METHODS,2):
        d=g['ECA_'+m1]-g['ECA_'+m2]
        contr.append(dict(t=t,method1=m1,method2=m2,states=len(g),positive=int((d>1e-12).sum()),tie=int((abs(d)<=1e-12).sum()),negative=int((d< -1e-12).sum()),mean_designed_state_difference=float(d.mean())))
pd.DataFrame(contr).to_csv(OUT/'state_contrasts.csv',index=False)
m=pd.read_csv(ROOT/'results/tv/mechanism.csv')
oh=m.groupby(['center','arm']).agg(candidates=('candidate_index','size'),proposal_feasible=('proposal_feasible','sum'),empirical_infeasible=('empirical_infeasible','sum'),endpoint_block=('mass_confidence_block','sum'),TV_block=('TV_geometry_block','sum'),acceptance_block=('acceptance_block','sum'),mean_empirical_D=('empirical_worst_D','mean'),mean_endpoint_inflation=('mass_endpoint_inflation','mean'),mean_TV_slack=('TV_geometry_slack','mean')).reset_index()
oh.to_csv(OUT/'officehome_all_candidates.csv',index=False)
pacs=pd.read_csv(SRC/'pacs/OUTCOMES.csv.gz')
assert len(pacs)==162000 and pacs.model_id.nunique()==30
mod=pacs.query('budget==512').groupby(['model_id','architecture','split','seed','t','procedure']).eca.mean().reset_index()
mod.to_csv(OUT/'pacs_per_model_primary.csv',index=False)
pc=[]
for (arch,t),g in mod.groupby(['architecture','t']):
    p=g.pivot(index='model_id',columns='procedure',values='eca')
    for m1,m2 in itertools.combinations(p.columns,2):
        d=p[m1]-p[m2]
        pc.append(dict(architecture=arch,t=t,method1=m1,method2=m2,models=len(p),positive=int((d>1e-12).sum()),tie=int((abs(d)<=1e-12).sum()),negative=int((d< -1e-12).sum()),mean_delta=float(d.mean()),minimum=float(d.min()),maximum=float(d.max())))
pd.DataFrame(pc).to_csv(OUT/'pacs_model_contrasts.csv',index=False)
report=dict(status='VALID_DESCRIPTIVE_ANALYSIS',designed_states=256,reproduced_flags=256*12*7,flag_mismatches=mismatches,max_ECA_difference=maxdiff,one_factor_state_pairs=len(pairs),method_pair_comparisons=len(details),strict_reversal_state_pairs=int(pairs.any_strict_reversal.sum()),order_change_state_pairs=int(pairs.any_order_change.sum()),unique_realized_states=int(states.realized_count_key.nunique()),PACS_outcomes=len(pacs),PACS_models=30,PACS_primary_outcomes=int((pacs.budget==512).sum()),scope='Designed states are not empirical frequencies. Mechanism labels overlap. PACS outcome comparisons do not establish count-level causation. Existing PACS HOLD preserved.')
(OUT/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));print(summary.to_string(index=False))
