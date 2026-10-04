"""S1 v2 runner. Imports the UNMODIFIED V1 adapter/certifier/analysis (hash-checked) and applies them to DomainNet-6 frames
with J=6 clients, per-domain draws floor(B/6), protocol_id v2 and bootstrap seed 2026092501 (PROTOCOL.json sha256 in common.py).
Stages: binding | drycheck | infer | audit | analyze."""
import argparse, csv, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np
from common import *

FEDCORE_ROOT = os.environ.get('FEDCORE_ROOT', '/home/sanghoon/Desktop/Workspace/Fedcore')
sys.path.insert(0, FEDCORE_ROOT)
from fedcore.experiments import v1_pacs_adapter as a      # noqa: E402
from fedcore.experiments import v1_pacs_certifier as c    # noqa: E402
from fedcore.experiments import v1_pacs_analysis as an    # noqa: E402

J = len(DOMAINS)
BUDGETS = [256, 512, 1024]; PRIMARY_B = 512; TS = [0.5, 1.0]; R = 100


def k_v2(archs):
    """V1 protocol_constants re-specified for v2 (the V1 function hard-codes J=4 and the V1 bootstrap seed)."""
    return {'alpha': 0.2, 'dr': 0.05, 'dc': 0.05, 'J': J, 'ts': TS, 'weights': [0.5, 0.5], 'budgets': BUDGETS, 'primary_budget': PRIMARY_B, 'replicates': R,
            'n_boot': 20000, 'boot_seed': BOOT_SEED, 'n_contrasts': 16, 'rule': {0.5: 'SCALE_H', 1.0: 'CP_IUT_H'}, 'diag': {0.5: 'DIRECT_BETTING_H', 1.0: 'CP_IUT_H'},
            'fixed': list(an.FIXED), 'models': 30, 'archs': archs, 'confirm_g': 0.01, 'kill_g': 0.002}


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def load_matrix(p):
    rows = list(csv.DictReader(open(p)))
    assert len(rows) == 30
    return rows


def load_manifest(data_dir):
    rows = list(csv.DictReader(open(Path(data_dir) / 'MANIFEST.csv')))
    for r in rows:
        r['row'] = int(r['row'])
    return rows


def groups(rows):
    """(domain, role) -> identity-sorted list of manifest rows (V1 loaders are identity-sorted per domain/role)."""
    g = {}
    for d in DOMAINS:
        for ro in ('proposal', 'audit'):
            sub = sorted((r for r in rows if r['domain'] == d and r['role'] == ro), key=lambda r: r['identity'])
            if not sub:
                raise ValueError(f'empty {d}/{ro}')
            if len({r['identity'] for r in sub}) != len(sub):
                raise ValueError('unresolved within-domain duplicate')
            g[(d, ro)] = sub
    return g


class CropDS:
    def __init__(self, crops, idx):
        from torchvision import transforms as T
        self.crops = crops; self.idx = idx
        self.t = T.Compose([T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])  # V1 ops after CenterCrop

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        from PIL import Image
        return self.t(Image.fromarray(np.asarray(self.crops[self.idx[i]]))), 0


def build_frozen(arch, split, seed, ckpt_path, v1_state_sha):
    import torch
    from fedcore.experiments.v1_pacs_runner import state_sha256
    a.deterministic_torch()
    model = a.build_model(arch, V1_PID, split, seed, pretrained=False)
    ck = torch.load(ckpt_path, map_location='cpu')
    if (ck['protocol_id'], ck['architecture'], ck['split'], ck['seed'], ck['completed_rounds']) != (V1_PID, arch, split, seed, 30):
        raise ValueError('checkpoint identity mismatch')
    model.load_state_dict(ck['state'], strict=True)
    if state_sha256(model.state_dict()) != v1_state_sha:
        raise ValueError('state digest != V1 RUN_STATE')
    return model.eval()


def infer_rows(model, crops, rowlist, device, workers):
    import torch
    ds = CropDS(crops, [r['row'] for r in rowlist])
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, drop_last=False, num_workers=workers)
    logits, _ = a.infer(model, loader, device)
    return logits


# ---------------- binding ----------------
def stage_binding(args):
    out = Path(args.run_dir); out.mkdir(parents=True, exist_ok=True)
    code_dir = Path(FEDCORE_ROOT) / 'fedcore/experiments'
    code = {n: sha(code_dir / n) for n in V1_CODE_SHA256}
    ck_expect = {}
    for line in open(args.ckpt_sha):
        h, p = line.split(); ck_expect[p] = h
    ck = {}
    matrix = load_matrix(args.model_matrix)
    for m in matrix:
        p = f"outputs/fedcore_v1_run/checkpoints/{m['model_id']}/latest.pt"
        full = Path(FEDCORE_ROOT) / p
        ck[m['model_id']] = {'path': p, 'exists': full.exists(), 'sha256': sha(full) if full.exists() else None, 'expected': ck_expect.get(p)}
    v1state = json.load(open(args.v1_run_state))
    b = {'protocol_id': PID, 'protocol_sha256_expected': PROTOCOL_SHA256, 'protocol_sha256_actual': sha(args.protocol),
         'v1_code_sha256': code, 'v1_code_match': all(code[n] == V1_CODE_SHA256[n] for n in code),
         'v1_code_match_run_state': all(v1state['code_sha256'][n] == code[n] for n in code),
         'checkpoints': ck, 'checkpoints_present': sum(v['exists'] for v in ck.values()), 'checkpoints_match': sum(v['sha256'] == v['expected'] for v in ck.values()),
         'model_matrix_sha256': sha(args.model_matrix), 'v1_run_state_sha256': sha(args.v1_run_state),
         's1v2_code_sha256': {n: sha(Path(__file__).parent / n) for n in ('common.py', 'prepare.py', 'runner.py')},
         'data_check_sha256': sha(Path(args.data_dir) / 'DATA_CHECK.json'), 'manifest_sha256': sha(Path(args.data_dir) / 'MANIFEST.csv'),
         'crops_sha256': sha(Path(args.data_dir) / 'CROPS_uint8.npy'), 'utc': utc()}
    import torch, torchvision, scipy, PIL
    b['env'] = {'python': sys.version, 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__, 'PIL': PIL.__version__}
    b['status'] = 'PASS' if (b['protocol_sha256_actual'] == PROTOCOL_SHA256 and b['v1_code_match'] and b['v1_code_match_run_state'] and b['checkpoints_present'] == 30 and b['checkpoints_match'] == 30) else 'FAIL'
    dump(out / 'RUN_BINDING.json', b); print(json.dumps({'binding': b['status'], 'ckpt_match': b['checkpoints_match'], 'code': b['v1_code_match']}))
    return 0 if b['status'] == 'PASS' else 2


# ---------------- drycheck (no outcomes) ----------------
def stage_drycheck(args):
    import torch, zipfile, io
    from PIL import Image, ImageOps
    out = Path(args.run_dir); rows = load_manifest(args.data_dir); crops = np.load(Path(args.data_dir) / 'CROPS_uint8.npy', mmap_mode='r')
    matrix = load_matrix(args.model_matrix); v1state = {m['model_id']: m for m in json.load(open(args.v1_run_state))['models']}
    res = {'utc': utc(), 'checks': {}, 'note': 'Wiring only. No ECA, certificate, score threshold, accuracy or acceptance summary is computed or viewed.'}
    t_v1 = a.transforms(False); ds_t = CropDS(crops, []).t
    rng = np.random.Generator(np.random.PCG64(1))
    sub = []
    for d in DOMAINS:
        dr = [r for r in rows if r['domain'] == d]
        pick = sorted(rng.choice(len(dr), size=min(8, len(dr)), replace=False).tolist())
        sub += [dr[i] for i in pick]
    maxdiff = 0.0; zs = {}
    for r in sub:
        z = zs.get(r['domain']) or zs.setdefault(r['domain'], zipfile.ZipFile(Path(args.zips) / f"{r['domain']}.zip"))
        with Image.open(io.BytesIO(z.read(r['path']))) as im:
            x1 = t_v1(ImageOps.exif_transpose(im).convert('RGB'))
        x2 = ds_t(Image.fromarray(np.asarray(crops[r['row']])))
        maxdiff = max(maxdiff, float((x1 - x2).abs().max()))
    res['checks']['cached_transform_equals_v1_transform'] = {'images': len(sub), 'max_abs_diff': maxdiff, 'pass': maxdiff == 0.0}
    # label maps and head order
    lm = {}
    for m in matrix:
        s = int(m['split']); known = m['known_classes'].split(';')
        assert known == sorted(known) == V1_SPLITS[s]['known'], 'head order'
        lm[s] = label_map(s)
    res['checks']['label_maps'] = {str(s): v for s, v in lm.items()}
    res['checks']['head_order_alphabetical_equals_v1_matrix'] = True
    protocol = json.load(open(args.protocol))
    res['checks']['label_maps_equal_protocol'] = all(protocol['dataset']['splits'][s]['label_map'] == lm[s] for s in lm)
    # shapes/finite on <=64 per domain for one model per arch (logits NOT summarised)
    dev = args.device; shp = {}
    per_dom = []
    for d in DOMAINS:
        per_dom += [r for r in rows if r['domain'] == d][:64]
    t0 = time.monotonic()
    for m in matrix[:2]:
        model = build_frozen(m['architecture'], int(m['split']), int(m['seed']), Path(FEDCORE_ROOT) / f"outputs/fedcore_v1_run/checkpoints/{m['model_id']}/latest.pt", v1state[m['model_id']]['state_sha256'])
        lg = infer_rows(model, crops, per_dom, dev, 0)
        shp[m['model_id']] = {'shape': list(lg.shape), 'finite': bool(np.isfinite(lg).all()), 'state_digest_match_v1': True}
        del model; torch.cuda.empty_cache()
    res['gpu_seconds'] = time.monotonic() - t0
    res['checks']['logit_shapes'] = shp
    res['checks']['role_split'] = {f'{d}/{ro}': sum(1 for r in rows if r['domain'] == d and r['role'] == ro) for d in DOMAINS for ro in ('proposal', 'audit')}
    res['checks']['draw_per_domain'] = {B: B // J for B in BUDGETS}
    res['pass'] = res['checks']['cached_transform_equals_v1_transform']['pass'] and res['checks']['label_maps_equal_protocol'] and all(v['shape'][1] == 4 and v['finite'] for v in shp.values())
    dump(out / 'DRYCHECK.json', res); print(json.dumps({'drycheck_pass': res['pass'], 'maxdiff': maxdiff, 'gpu_s': res['gpu_seconds']}))
    return 0 if res['pass'] else 3


# ---------------- infer ----------------
def stage_infer(args):
    import torch
    out = Path(args.run_dir); (out / 'logits').mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.data_dir); g = groups(rows); crops = np.load(Path(args.data_dir) / 'CROPS_uint8.npy', mmap_mode='r')
    matrix = load_matrix(args.model_matrix); v1state = {m['model_id']: m for m in json.load(open(args.v1_run_state))['models']}
    ledger_p = out / f'GPU_LEDGER_{args.host}.json'
    ledger = json.load(open(ledger_p)) if ledger_p.exists() else {'host': args.host, 'models': {}, 'gpu_seconds': 0.0}
    for m in matrix:
        if args.arch and m['architecture'] != args.arch:
            continue
        mid = m['model_id']; lp = out / 'logits' / f'{mid}.npz'
        if mid in ledger['models'] and lp.exists() and sha(lp) == ledger['models'][mid]['logits_sha256']:
            continue
        t0 = time.monotonic()
        model = build_frozen(m['architecture'], int(m['split']), int(m['seed']), Path(FEDCORE_ROOT) / f"outputs/fedcore_v1_run/checkpoints/{mid}/latest.pt", v1state[mid]['state_sha256'])
        payload = {'model_id': np.array(mid), 'protocol_id': np.array(PID)}
        for (d, ro), lst in g.items():
            lg = infer_rows(model, crops, lst, args.device, args.workers)
            payload[f'{ro}__{d}__logits'] = lg.astype(np.float64)
            payload[f'{ro}__{d}__identity'] = np.array([r['identity'] for r in lst])
            payload[f'{ro}__{d}__class'] = np.array([r['class'] for r in lst])
        np.savez_compressed(lp, **payload)
        dt = time.monotonic() - t0
        ledger['models'][mid] = {'gpu_seconds': dt, 'logits_sha256': sha(lp), 'utc': utc(), 'gpu_name': torch.cuda.get_device_name(0)}
        ledger['gpu_seconds'] = sum(v['gpu_seconds'] for v in ledger['models'].values()); dump(ledger_p, ledger)
        print(json.dumps({'model': mid, 'sec': round(dt, 1), 'total_gpu_s': round(ledger['gpu_seconds'], 1)}), flush=True)
        del model; torch.cuda.empty_cache()
        if ledger['gpu_seconds'] / 3600 > args.gpu_hours_cap:
            print('GPU_CAP reached'); return 10
    return 0


# ---------------- audit (V1 audit_model with J=6) ----------------
def audit_model_v2(split, prop, aud):
    budgets, ts = BUDGETS, TS; M = len(an.FIXED)
    logits_all = np.concatenate([prop[d]['logits'] for d in DOMAINS]); labels_all = np.concatenate([prop[d]['labels'] for d in DOMAINS])
    cands = a.proposals(logits_all, labels_all)
    eligible = np.array([bool(x['eligible']) for x in cands]); nB = len(budgets)
    n = np.zeros((nB, R, 12, J), np.int64); A = np.zeros_like(n); K = np.zeros_like(n); unique = np.zeros((nB, R, J), np.int64)
    sizes = np.array([len(aud[d]['labels']) for d in DOMAINS])
    for r in range(R):
        for j, d in enumerate(DOMAINS):
            idx = a.audit_indices(PID, split, d, r, int(sizes[j]))
            for bi, B in enumerate(budgets):
                per = B // J
                if per > len(idx):
                    raise ValueError('budget exceeds frozen draw stream length')
                nn, AA, KK = a.counts_for_draws(aud[d]['logits'], aud[d]['labels'], cands, idx[:per])
                n[bi, r, :, j] = nn; A[bi, r, :, j] = AA; K[bi, r, :, j] = KK; unique[bi, r, j] = len(set(idx[:per].tolist()))
    nT = len(ts)
    flags = np.zeros((nT, nB, M, R, 12), bool); lcb = np.zeros((nT, nB, R, 12)); chosen = np.full((nT, nB, M, R), -1, np.int64); eca = np.zeros((nT, nB, M, R))
    for ti, t in enumerate(ts):
        for bi, B in enumerate(budgets):
            stats, passed = c.certify(n[bi], A[bi], K[bi], t, alpha=0.2, dr=0.05, dc=0.05)
            lcb[ti, bi] = stats['common_acceptance_lcb']
            for mi, m in enumerate(an.FIXED):
                flags[ti, bi, mi] = passed[m]
            for r in range(R):
                st_r = {key: v[r] for key, v in stats.items()}; pd_r = {m: passed[m][r] for m in an.FIXED}
                ch = c.choose(st_r, pd_r, eligible.tolist())
                for mi, m in enumerate(an.FIXED):
                    ci = ch[m]
                    if ci is not None:
                        chosen[ti, bi, mi, r] = ci; eca[ti, bi, mi, r] = lcb[ti, bi, r, ci]
    # post-decision full-frame truth (never fed back)
    acc = np.zeros((12, J)); mass = np.zeros((12, J))
    for j, d in enumerate(DOMAINS):
        full = np.arange(len(aud[d]['labels']))
        nn, AA, KK = a.counts_for_draws(aud[d]['logits'], aud[d]['labels'], cands, full)
        acc[:, j] = AA / nn; mass[:, j] = KK / nn
    failure = np.zeros((nT, nB, M, R), bool); worst_risk = np.zeros((nT, 12)); worst_acc = np.zeros((nT, 12))
    for ti, t in enumerate(ts):
        wr, wa = an.truth_summaries(acc, mass, t); worst_risk[ti] = wr; worst_acc[ti] = wa
        for bi in range(nB):
            failure[ti, bi] = an.joint_failure(flags[ti, bi], lcb[ti, bi], eligible, wr, wa, 0.2)
    return {'n': n, 'A': A, 'K': K, 'unique_draws': unique, 'flags': flags, 'lcb': lcb, 'chosen': chosen, 'eca': eca, 'eligible': eligible,
            'candidates_json': np.array(json.dumps(cands, sort_keys=True)), 'truth_acc': acc, 'truth_mass': mass, 'worst_risk': worst_risk, 'worst_acc': worst_acc,
            'failure': failure, 'audit_frame_sizes': sizes, 'proposal_frame_sizes': np.array([len(prop[d]['labels']) for d in DOMAINS])}


def audit_checks_v2(aud):
    problems = []
    for bi, B in enumerate(BUDGETS):
        if not np.all(aud['n'][bi] == B // J):
            problems.append(f'n != floor(B/6) at B={B}')
    if np.any(aud['K'] > aud['A']) or np.any(aud['A'] > aud['n']) or np.any(aud['K'] < 0):
        problems.append('count ordering violated')
    nT, nB, M, RR = aud['eca'].shape
    for ti in range(nT):
        for bi in range(nB):
            for mi in range(M):
                for r in range(RR):
                    ci = int(aud['chosen'][ti, bi, mi, r])
                    if ci >= 0:
                        if not (aud['flags'][ti, bi, mi, r, ci] and aud['eligible'][ci] and aud['lcb'][ti, bi, r, ci] > 0):
                            problems.append('chosen candidate not passing/eligible/positive'); break
                        if aud['eca'][ti, bi, mi, r] != aud['lcb'][ti, bi, r, ci]:
                            problems.append('ECA != LCB of chosen'); break
                    elif aud['eca'][ti, bi, mi, r] != 0:
                        problems.append('refusal with nonzero ECA'); break
    if aud['flags'].shape != (nT, nB, M, RR, 12) or aud['flags'].size != 2 * 3 * 7 * 100 * 12:
        problems.append('flag tensor shape')
    if not np.isfinite(aud['eca']).all():
        problems.append('nonfinite ECA')
    return sorted(set(problems))


def frames_for(z, split):
    lm = label_map(split); prop = {}; aud = {}
    for d in DOMAINS:
        for ro, tgt in (('proposal', prop), ('audit', aud)):
            ident = z[f'{ro}__{d}__identity']; order = np.argsort(ident, kind='stable')
            if not np.array_equal(order, np.arange(len(ident))):
                raise ValueError('frame not identity-sorted')
            tgt[d] = {'logits': z[f'{ro}__{d}__logits'], 'labels': np.array([lm[x] for x in z[f'{ro}__{d}__class']], np.int64), 'identity': ident}
    return prop, aud


def stage_audit(args):
    out = Path(args.run_dir); (out / 'audit').mkdir(exist_ok=True)
    matrix = load_matrix(args.model_matrix); state_p = out / 'AUDIT_STATE.json'
    st = json.load(open(state_p)) if state_p.exists() else {'models': {}}
    ledgers = [json.load(open(p)) for p in out.glob('GPU_LEDGER_*.json')]
    lsha = {mid: v['logits_sha256'] for L in ledgers for mid, v in L['models'].items()}
    for m in matrix:
        mid = m['model_id']
        if mid in st['models']:
            continue
        lp = out / 'logits' / f'{mid}.npz'
        if sha(lp) != lsha.get(mid):
            raise SystemExit(f'logits hash mismatch {mid}')
        z = np.load(lp, allow_pickle=False); t0 = time.time()
        prop, aud = frames_for(z, int(m['split']))
        res = audit_model_v2(int(m['split']), prop, aud)
        probs = audit_checks_v2(res)
        if probs:
            raise SystemExit(f'audit self-check failed {mid}: {probs}')
        res.update({'model_id': np.array(mid), 'architecture': np.array(m['architecture']), 'split': np.array(int(m['split'])), 'seed': np.array(int(m['seed'])), 'logits_sha256': np.array(lsha[mid])})
        ap = out / 'audit' / f'{mid}.npz'; np.savez_compressed(ap, **res)
        st['models'][mid] = {'audit_sha256': sha(ap), 'seconds': time.time() - t0}; dump(state_p, st)
        print(json.dumps({'audited': mid, 's': round(time.time() - t0, 1)}), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['binding', 'drycheck', 'infer', 'audit', 'analyze'])
    p.add_argument('--run-dir', required=True); p.add_argument('--data-dir', required=True); p.add_argument('--zips', default=None)
    p.add_argument('--protocol', required=True); p.add_argument('--model-matrix', required=True); p.add_argument('--v1-run-state', required=True)
    p.add_argument('--ckpt-sha', default=None); p.add_argument('--device', default='cuda:0'); p.add_argument('--arch', default=None)
    p.add_argument('--host', default='ubuntu-4070'); p.add_argument('--workers', type=int, default=4); p.add_argument('--gpu-hours-cap', type=float, default=8.0)
    p.add_argument('--v1-contrasts', default=os.path.join(FEDCORE_ROOT, 'outputs/fedcore_v1_run/CONTRASTS.csv'))
    args = p.parse_args()
    if sha(args.protocol) != PROTOCOL_SHA256:
        raise SystemExit('INVALID: PROTOCOL.json hash differs from the v2 seal')
    if args.stage == 'analyze':
        from analyze import stage_analyze
        return stage_analyze(args)
    return {'binding': stage_binding, 'drycheck': stage_drycheck, 'infer': stage_infer, 'audit': stage_audit}[args.stage](args)


if __name__ == '__main__':
    sys.exit(main())
