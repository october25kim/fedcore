"""Reaggregate the two archived B.3 simulation targets; no new simulation draws."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
a=pd.read_csv(ROOT/'validity_B1000_per_cell.csv').query("method=='fedcore_noJ'")
b=pd.read_csv(ROOT/'rejectall_validity_fixed.csv')
truth=pd.read_csv(ROOT/'reservoir_source_uniqueness_and_truth.csv')
assert len(a)==len(b)==450
assert (truth.cert_unique_n+truth.test_unique_n-truth.cert_test_unique_overlap==truth.reservoir_unique_n).all()
assert (truth.audit_draw_n==truth.cert_raw_n).all()
t=truth.groupby('semantic_id').agg(risk=('true_risk','max'),acceptance=('true_acceptance','min'),accepted=('unique_accepted_n','max'))
m=a.merge(t,left_on='semantic_id',right_index=True,validate='one_to_one')
assert np.allclose(m.true_worst_client_risk,m.risk,atol=1e-14,rtol=0)
assert np.allclose(m.true_min_client_coverage,m.acceptance,atol=1e-14,rtol=0)
zero=set(m.loc[m.accepted==0,'semantic_id'])
assert len(zero)==7 and zero==set(b.loc[b.rejectall==1,'sid'])
assert (a.loc[a.semantic_id.isin(zero),'empirical_ucb_coverage']==1).all()
assert (b.loc[b.rejectall==1,'validity']==1).all()
assert a.empirical_ucb_coverage.min()==a.loc[~a.semantic_id.isin(zero),'empirical_ucb_coverage'].min()==.953
assert b.validity.min()==b.loc[b.rejectall==0,'validity'].min()==.958
m=a.merge(b,left_on='semantic_id',right_on='sid',validate='one_to_one')
assert ((m.true_worst_client_risk-m.target).abs()>1e-12).sum()==442
print(json.dumps({'status':'PASS','cells':450,'non_reject_all':443,'union_minimum':.953,'certification_only_minimum':.958,'cells_with_different_targets':442,'new_simulation':False},indent=2))
