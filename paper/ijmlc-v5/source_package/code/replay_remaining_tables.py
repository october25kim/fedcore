"""Portable outcome-to-table and designed-count replay; no GPU or raw images."""
from pathlib import Path
import csv,gzip,json,hashlib,itertools,sys
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'results';O.mkdir(exist_ok=True)
report={}
# Main Table 3, formerly Table 9. Formula reconstruction is a separate command.
x=pd.read_csv(R/'historical/evidence/protection_recheck/results/per_cell.csv.gz')
assert len(x)==94500 and not x.duplicated(['semantic_id','alpha','t','method']).any()
p=x[(x.alpha==.2)&(x.t==.5)]
rows=[]
for (dataset,method),g in p.groupby(['dataset','method']):
 assert len(g)=={'cifar10':150,'cifar100':150,'officehome':100,'pathmnist':50}[dataset]
 rows.append({'dataset':dataset,'method':method,'cells':len(g),'certified':int(g.certified.sum()),'eca_percent':g.eca.mean()*100,'sd_percent':g.eca.std(ddof=1)*100})
pd.DataFrame(rows).to_csv(O/'TABLE3_REPLAY.csv',index=False)
report['table3']={'all_outcomes':len(x),'reference_outcomes':len(p),'summary_rows':len(rows)}
# Main Table 6, formerly Table 12: reconstruct every contrast and bootstrap.
x=pd.read_csv(R/'pacs/OUTCOMES.csv.gz');keys=['model_id','t','budget','procedure','replicate']
assert len(x)==162000 and not x.duplicated(keys).any() and x.model_id.nunique()==30
assert set(x.budget)=={256,512,1024} and set(x.t)=={.5,1} and set(x.replicate)==set(range(100))
assert (x.groupby(['model_id','t','budget','procedure']).size()==100).all()
assert (x.loc[x.certified==0,'eca']==0).all()
p=x[x.budget==512];assert len(p)==54000
wide=p.pivot(index=['architecture','model_id','t','replicate'],columns='procedure',values='eca')
idx=np.random.Generator(np.random.PCG64(2026091601)).integers(0,100,size=(20000,100))
bs=json.loads((R/'pacs/BOOTSTRAP.json').read_text())
assert hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()==bs['meta']['index_sha256']
cts=pd.read_csv(R/'pacs/CONTRASTS.csv');errors=[];out=[]
for row in cts.to_dict('records'):
 a,c=row['architecture'],row['control'];aw=wide.xs(a)
 d=(aw.V1_SCALE_CP-aw[c]).groupby(level='replicate').mean().to_numpy()
 boot=d[idx].mean(1);vals=[d.mean(),*np.quantile(boot,[.05/32,1-.05/32],method='linear')]
 err=max(abs(vals[i]-row[n]) for i,n in enumerate(['delta_hat','ci_lower','ci_upper']));assert err<1e-12;errors.append(err)
 out.append({'architecture':a,'control':c,'mean_pp':100*vals[0],'lower_pp':100*vals[1],'upper_pp':100*vals[2]})
fixed=cts[cts.primary==1];mins=fixed.groupby('architecture').delta_hat.min()
confirm=bool((mins>=.01).all() and (fixed.ci_lower>0).all());kill=bool((fixed.ci_upper<=0).any() or (mins<.002).all())
assert not confirm and not kill
assert hashlib.sha256(gzip.decompress((R/'pacs/OUTCOMES.csv.gz').read_bytes())).hexdigest()=='435e455d613cc09ac635a984286841f95ef0bf5ede2cb557dbdd410cd1278e79'
pd.DataFrame(out).to_csv(O/'TABLE6_REPLAY.csv',index=False)
report['pacs']={'outcomes':len(x),'primary_outcomes':len(p),'models':30,'contrasts':len(out),'bootstrap_resamples':len(idx),'max_abs_error':max(errors),'decision':'HOLD','level':'Saved outcomes to contrasts; no retraining or logit export replay'}
# E2 independent original formulas, imported without executing its historical main.
sys.path.insert(0,str(R/'e2/code'))
from e2_count_state_audit import certify_one,METHODS
states=pd.read_csv(R/'e2/STATES.csv');expected=pd.read_csv(R/'e2/FLAGS.csv.gz');ranks=pd.read_csv(R/'e2/RANKINGS.csv')
assert len(states)==256 and len(expected)==21504
flags=0;maxdiff=0.
for st in states.to_dict('records'):
 stats,passed,eca,selected=certify_one([st['n']]*4,[st[f'A{j}'] for j in range(4)],[st[f'K{j}'] for j in range(4)],st['t'])
 for method in METHODS:
  sub=expected[(expected.state_id==st['state_id'])&(expected.method==method)]
  assert len(sub)==12
  assert np.array_equal(np.asarray(passed[method]),sub.sort_values('slot').passed.to_numpy(bool));flags+=12
  maxdiff=max(maxdiff,abs(eca[method]-ranks.loc[ranks.state_id==st['state_id'],'ECA_'+method].iloc[0]))
assert maxdiff<1e-12
counts={}
for t,g in ranks.groupby('t'):
 counts[str(t)]=sum(any((a['ECA_'+m]-a['ECA_'+n])*(b['ECA_'+m]-b['ECA_'+n])<0 for m,n in itertools.combinations(METHODS,2)) for a,b in itertools.combinations(g.to_dict('records'),2))
assert counts=={'0.0':25,'0.5':0,'0.75':2,'1.0':0}
report['e2']={'states':len(states),'flags':flags,'max_eca_error':maxdiff,'reversal_state_pairs':counts,'level':'Designed-count formula replay; not deployment-frequency evidence'}
report['status']='PASS'
(O/'REMAINING_TABLE_REPLAY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
