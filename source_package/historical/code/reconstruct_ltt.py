"""Independent count-to-decision reconstruction; no production certifier imports."""
from pathlib import Path
from collections import defaultdict
import csv, gzip, hashlib, itertools, json, math, time
import numpy as np
from scipy.stats import beta, binom

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / 'inputs/results/theorem_aligned_wr_450_v3'
OUT = BASE / 'evidence/ltt_reconstruction'
OUT.mkdir(parents=True, exist_ok=True)

def rows(p):
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
        yield from csv.DictReader(f)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def upper(k, n, tail):
    ans = np.ones(k.shape, float)
    good = (n > 0) & (k < n)
    ans[good] = beta.isf(tail, k[good] + 1, (n-k)[good])
    return ans

def lower(k, n, tail):
    ans = np.zeros(k.shape, float)
    good = k > 0
    ans[good] = beta.ppf(tail, k[good], (n-k)[good] + 1)
    return ans

start = time.monotonic()
source_files = [SRC/'primary_run/primary_candidate_counts.csv.gz',
                SRC/'primary_run/primary_per_cell_procedures.csv',
                SRC/'governing/family_manifest.csv.gz']
stored = {(r['semantic_id'], float(r['alpha']), r['procedure']): r
          for r in rows(source_files[1])}
family = {(r['semantic_id'], float(r['alpha']), int(r['candidate_index'])): r
          for r in rows(source_files[2])}
assert len(stored) == 8100 and len(family) == 32400
results = []
checks = defaultdict(int)
maxerr = 0.0
roster = defaultdict(set)
for (sid, alpha), it in itertools.groupby(rows(source_files[0]),
                    lambda r: (r['semantic_id'], float(r['alpha']))):
    rr = list(it)
    J = 4 if rr[0]['dataset'] == 'officehome' else 5
    assert len(rr) == 12*J
    A = np.full((12,J), -1, dtype=int)
    K = A.copy(); N = A.copy()
    for r in rr:
        m,j = int(r['candidate_index']),int(r['client'])
        assert A[m,j] == -1
        A[m,j],K[m,j],N[m,j] = int(r['A']),int(r['K']),int(r['n'])
        f = family[sid,alpha,m]
        assert r['selector_sha256'] == f['selector_sha256']
        assert r['proposal_feasible'] == f['proposal_feasible']
    assert np.all((K>=0)&(K<=A)&(A<=N))
    assert np.all(N == N[0])
    payload = {'A': A.tolist(), 'K': K.tolist(), 'n': N[0].tolist()}
    count_hash = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert all(r['counts_tensor_sha256'] == count_hash for r in rr)
    checks['count_tensor_hashes'] += 1
    fs = [family[sid,alpha,m] for m in range(12)]
    feasible = np.array([int(f['proposal_feasible'])==1 for f in fs])
    # This calculation specializes the LTT exact accepted-error binomial test.
    p_stratum = binom.cdf(K, A, alpha)
    p_stratum[A == 0] = 1
    p = p_stratum.max(axis=1)
    ordered = sorted(range(12),key=lambda i:(float(p[i]),i))
    holm_pass = np.zeros(12,bool)
    for rank, m in enumerate(ordered):
        if p[m] > .05/(12-rank):
            break
        holm_pass[m] = True
    adjusted = np.empty(12,float)
    for rank,m in enumerate(ordered):
        adjusted[m] = min(1.,max((12-q)*float(p[ordered[q]]) for q in range(rank+1)))
    C = lower(A,N,.05/12).min(axis=1)
    U = upper(K,A,.05/12).max(axis=1)
    assert np.array_equal(U<=alpha, p<=.05/12)
    checks['cp_iut_candidate_equivalences'] += 12
    assert bool(np.any(U<=alpha)) == bool(np.any(holm_pass))
    masks = {'H':holm_pass & feasible & (C>0), 'S':(U<=alpha)&feasible&(C>0)}
    UB = upper(K,A,.05/(12*J)).max(axis=1)
    CB = lower(A,N,.05/(12*J)).min(axis=1)
    masks['B'] = (UB<=alpha)&feasible&(CB>0)
    assert bool(masks['H'].any()) == bool(masks['S'].any())
    checks['H_S_cell_equivalences'] += 1
    roster[rr[0]['dataset']].add(sid)
    for proc,mask in masks.items():
        cs = CB if proc=='B' else C
        passing = np.flatnonzero(mask)
        pick = min(passing,key=lambda i:(-float(cs[i]),float(fs[i]['gamma']),
                   'msp' if fs[i]['slot']=='native' else fs[i]['slot'],int(i))) if len(passing) else None
        eca = float(cs[pick]) if pick is not None else 0.
        old = stored[sid,alpha,proc]
        assert int(old['certified']) == int(pick is not None)
        assert old['selected_candidate_index'] == (str(pick) if pick is not None else '')
        assert old['counts_tensor_sha256'] == count_hash
        err = abs(eca-float(old['effective_certified_coverage']))
        assert err <= 1e-12
        maxerr=max(maxerr,err)
        if pick is not None:
            expected = old['holm_adjusted_pvalue'] if proc=='H' else old['risk_ucb']
            actual = adjusted[pick] if proc=='H' else (UB if proc=='B' else U)[pick]
            err=abs(float(expected)-actual);assert err<1e-12;maxerr=max(maxerr,err)
            assert old['selected_selector_sha256']==fs[pick]['selector_sha256']
        results.append({'semantic_id':sid,'dataset':rr[0]['dataset'],'alpha':alpha,
                        'procedure':proc,'certified':int(pick is not None),
                        'selected_candidate':pick,'eca':eca})
        checks['archived_decisions_reconstructed'] += 1
assert {k:len(v) for k,v in roster.items()} == {'cifar10':150,'cifar100':150,'officehome':100,'pathmnist':50}
summary=[]
for alpha in [.05,.10,.15,.20,.25,.30]:
    for proc in ['H','S','B']:
        v=[r for r in results if r['alpha']==alpha and r['procedure']==proc]
        summary.append({'alpha':alpha,'procedure':proc,'N':len(v),
                        'certified':sum(r['certified'] for r in v),
                        'eca':sum(r['eca'] for r in v)/len(v)})
for name,data in [('reconstructed_decisions.csv',results),('summary.csv',summary)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
report={'status':'PASS','scope':'Independent arithmetic and LTT-specialization reconstruction from frozen client counts; no new audit samples, inference, or training.',
        'checks':dict(checks),'roster':{k:len(v) for k,v in roster.items()},'max_absolute_numeric_difference':maxerr,
        'inputs':{str(p.relative_to(BASE)):digest(p) for p in source_files},
        'script_sha256':digest(Path(__file__)),'elapsed_seconds':time.monotonic()-start,
        'limitation':'Does not reconstruct raw-image inference, sampling provenance, or independence from count triples alone.'}
(OUT/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
print(json.dumps([r for r in summary if r['alpha']==.2],indent=2))
