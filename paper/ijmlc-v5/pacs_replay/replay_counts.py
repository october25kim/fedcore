"""Replay the frozen PACS counts and selections, then match all archived outcomes."""
from pathlib import Path
import csv, gzip, json
import numpy as np
from fedcore_pacs import v1_pacs_certifier as c, v1_pacs_analysis as an

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'source_package/pacs'
k = an.protocol_constants(json.loads((ROOT/'PROTOCOL.json').read_text()))
matrix = list(csv.DictReader((DATA/'MODEL_MATRIX.csv').open()))
assert len(matrix) == 30 and len({r['model_id'] for r in matrix}) == 30
assert {p.stem for p in (ROOT/'audit').glob('*.npz')} == {r['model_id'] for r in matrix}
audits = {}; total_flags = 0; max_error = 0.
for row in matrix:
    mid = row['model_id']
    with np.load(ROOT/'audit'/f'{mid}.npz', allow_pickle=False) as z:
        a = {name: z[name] for name in z.files}
    assert not bool(a['dry_run']) and str(a['model_id']) == mid
    assert a['n'].shape == a['A'].shape == a['K'].shape == (3,100,12,4)
    for ti,t in enumerate(k['ts']):
        st, passed = c.certify(a['n'],a['A'],a['K'],t,k['alpha'],k['dr'],k['dc'])
        for name,v in st.items():
            field = 'lcb' if name == 'common_acceptance_lcb' else 'stat_'+name
            if field in a:
                err = float(np.max(np.abs(v-a[field][ti])))
                assert err < 1e-10, (mid,field,err)
                max_error = max(max_error,err)
        for mi,method in enumerate(c.METHODS):
            assert np.array_equal(passed[method],a['flags'][ti,:,mi]), (mid,method)
            total_flags += passed[method].size
        for bi in range(3):
            for r in range(100):
                selected = c.choose({n:v[bi,r] for n,v in st.items()},
                                    {n:v[bi,r] for n,v in passed.items()}, a['eligible'])
                for mi,method in enumerate(c.METHODS):
                    ci = selected[method]
                    assert int(a['chosen'][ti,bi,mi,r]) == (-1 if ci is None else ci)
                    e = 0. if ci is None else float(st['common_acceptance_lcb'][bi,r,ci])
                    assert abs(e-a['eca'][ti,bi,mi,r]) < 1e-12
    audits[mid] = a
E, chosen = an.assemble(audits,matrix,k)
actual = an.outcome_rows(E,chosen,matrix,k)
with gzip.open(DATA/'OUTCOMES.csv.gz','rt') as f:
    expected = list(csv.DictReader(f))
keys = ['model_id','architecture','split','seed','t','budget','procedure','replicate']
key = lambda row: tuple(str(row[n]) for n in keys)
left = {key(r):r for r in actual}; right = {key(r):r for r in expected}
assert len(left) == len(right) == 162000 and left.keys() == right.keys()
for identity,r in left.items():
    old = right[identity]
    assert r['certified'] == int(old['certified']) and r['candidate_index'] == int(old['candidate_index'])
    assert abs(r['eca']-float(old['eca'])) < 1e-12
report = {'status':'PASS','models':30,'candidate_method_flags':total_flags,
          'outcomes':len(left),'primary_outcomes':54000,'max_statistic_error':max_error,
          'scope':'Saved aggregate counts to candidate decisions, selections and all outcome rows. No training, raw images or logits.'}
assert total_flags == 1512000
print(json.dumps(report,indent=2))
