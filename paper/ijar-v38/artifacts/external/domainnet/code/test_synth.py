"""Wiring test on SYNTHETIC logits only (no DomainNet image, no checkpoint). Writes to a scratch dir marked DRY."""
import json, sys, os, hashlib
from pathlib import Path
import numpy as np
from common import *
import runner
scratch = Path(sys.argv[1]); scratch.mkdir(parents=True, exist_ok=True); (scratch / 'logits').mkdir(exist_ok=True)
data = scratch / 'data'; data.mkdir(exist_ok=True)
matrix = runner.load_matrix(sys.argv[2])
rng = np.random.Generator(np.random.PCG64(7))
sizes = {'clipart': 360, 'infograph': 640, 'painting': 1030, 'quickdraw': 1500, 'real': 1870, 'sketch': 590}
frames = {}
for d in DOMAINS:
    for ro in ('proposal', 'audit'):
        n = sizes[d]; ident = np.array(sorted(hashlib.sha256(f'{d}{ro}{i}'.encode()).hexdigest() for i in range(n)))
        frames[(d, ro)] = (ident, rng.choice(CLASSES, size=n))
ledger = {'host': 'synthetic', 'models': {}, 'gpu_seconds': 0.0}
for m in matrix:
    lm = label_map(int(m['split'])); payload = {'model_id': np.array(m['model_id'])}
    for (d, ro), (ident, cls) in frames.items():
        lab = np.array([lm[c] for c in cls]); lg = rng.normal(size=(len(cls), 4))
        strength = 4.0 if m['architecture'] == 'convnext_tiny' else 2.5
        for i, l in enumerate(lab):
            if l >= 0:
                lg[i, l] += strength
        payload[f'{ro}__{d}__logits'] = lg; payload[f'{ro}__{d}__identity'] = ident; payload[f'{ro}__{d}__class'] = cls
    p = scratch / 'logits' / f"{m['model_id']}.npz"; np.savez_compressed(p, **payload)
    ledger['models'][m['model_id']] = {'gpu_seconds': 0, 'logits_sha256': sha(p)}
dump(scratch / 'GPU_LEDGER_synthetic.json', ledger)
dump(scratch / 'RUN_BINDING.json', {'status': 'PASS', 'synthetic': True})
dump(data / 'DATA_CHECK.json', {'G2_LIST_FILES': {'status': 'PASS'}, 'G3_DUPLICATES': {'status': 'PASS'}, 'G4_SUPPORT': {'status': 'PASS'}, 'G6_PACS_OVERLAP': {'overlap_count': 0}, 'synthetic': True})
print('synthetic frames written')
