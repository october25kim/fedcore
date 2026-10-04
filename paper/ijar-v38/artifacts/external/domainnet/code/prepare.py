"""S1 v2 data preparation (CPU only, no model): list filter, decode, V1 identity, dedup, roles, V1 inference crop cache,
pre-GPU gates G2/G3/G4/G6. Writes to --out. Never computes any model output."""
import argparse, csv, io, json, os, struct, hashlib, zipfile, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageOps
from common import *

_Z = {}; _ZIPDIR = None; _T = None


def _init(zipdir):
    global _ZIPDIR, _T
    _ZIPDIR = zipdir
    from torchvision import transforms as T
    _T = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True), T.CenterCrop(224)])  # V1 inference ops before ToTensor


def _work(e):
    d, cls, path, lab, sp = e
    z = _Z.get(d)
    if z is None:
        z = _Z[d] = zipfile.ZipFile(os.path.join(_ZIPDIR, d + '.zip'))
    try:
        data = z.read(path)
    except KeyError:
        return (e, 'missing', None, None, None, None)
    try:
        with Image.open(io.BytesIO(data)) as im0:
            im = ImageOps.exif_transpose(im0).convert('RGB')
        w, h = im.size
        ident = hashlib.sha256(struct.pack('>II', w, h) + im.tobytes()).hexdigest()
        crop = np.asarray(_T(im), dtype=np.uint8)
        if crop.shape != (224, 224, 3):
            return (e, 'bad_crop_shape', None, None, None, None)
        return (e, 'ok', ident, w, h, crop)
    except Exception as ex:  # noqa
        return (e, 'undecodable:' + repr(ex)[:200], None, None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lists', required=True); ap.add_argument('--zips', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--v1-fold-manifest', required=True); ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    entries = []; label_by_class = defaultdict(set); listed = Counter()
    for d in DOMAINS:
        for sp in ('train', 'test'):
            for line in open(os.path.join(a.lists, f'{d}_{sp}.txt')):
                parts = line.split()
                if not parts:
                    continue
                path, lab = parts[0], int(parts[1])
                if path.split('/')[0] != d:
                    raise SystemExit(f'list path domain mismatch {path}')
                cls = path.split('/')[1].strip().lower()
                if cls in CLASSES:
                    entries.append((d, cls, path, lab, sp)); label_by_class[(d, cls)].add(lab); listed[(d, cls)] += 1
    label_inconsistent = {f'{d}/{c}': sorted(v) for (d, c), v in label_by_class.items() if len(v) != 1}
    # all domains must agree on the integer label for a class too (report)
    cross = defaultdict(set)
    for (d, c), v in label_by_class.items():
        cross[c] |= v
    entries.sort(key=lambda e: (e[0], e[2]))
    results = []
    with Pool(a.workers, initializer=_init, initargs=(a.zips,)) as pool:
        for r in pool.imap(_work, entries, chunksize=16):
            results.append(r)
    excluded = [{'domain': e[0], 'class': e[1], 'path': e[2], 'list_split': e[4], 'reason': st} for (e, st, *_ ) in results if st != 'ok']
    ok = [r for r in results if r[1] == 'ok']
    # within-domain duplicates / conflicts
    by_dom_ident = defaultdict(list)
    for (e, st, ident, w, h, crop) in ok:
        by_dom_ident[(e[0], ident)].append((e, w, h, crop))
    conflicts = []; duplicates = []; rows = []; crops = []
    for (d, ident), lst in sorted(by_dom_ident.items()):
        classes = sorted({x[0][1] for x in lst})
        if len(classes) > 1:
            conflicts.append({'domain': d, 'identity': ident, 'classes': classes, 'paths': sorted(x[0][2] for x in lst)})
        lst.sort(key=lambda x: x[0][2])
        keep = lst[0]
        for x in lst[1:]:
            duplicates.append({'domain': d, 'identity': ident, 'retained': keep[0][2], 'removed': x[0][2], 'class_retained': keep[0][1], 'class_removed': x[0][1]})
        e, w, h, crop = keep
        rows.append({'row': len(rows), 'domain': d, 'class': e[1], 'path': e[2], 'list_split': e[4], 'list_label': e[3], 'identity': ident, 'width': w, 'height': h, 'role': role_of(ident)})
        crops.append(crop)
    # cross-domain identity: same role automatically; report cross-domain class conflicts descriptively
    cls_by_ident = defaultdict(set); dom_by_ident = defaultdict(set)
    for r in rows:
        cls_by_ident[r['identity']].add(r['class']); dom_by_ident[r['identity']].add(r['domain'])
    cross_domain_conflicts = {i: sorted(v) for i, v in cls_by_ident.items() if len(v) > 1}
    cross_domain_shared = sum(1 for v in dom_by_ident.values() if len(v) > 1)
    arr = np.stack(crops).astype(np.uint8)
    np.save(os.path.join(a.out, 'CROPS_uint8.npy'), arr)
    with open(os.path.join(a.out, 'MANIFEST.csv'), 'w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    for name, lst in (('EXCLUSIONS.csv', excluded), ('DUPLICATES.csv', duplicates)):
        with open(os.path.join(a.out, name), 'w', newline='') as f:
            if lst:
                w_ = csv.DictWriter(f, fieldnames=list(lst[0])); w_.writeheader(); w_.writerows(lst)
            else:
                f.write('none\n')
    # G2
    excl_by_dom = Counter(x['domain'] for x in excluded)
    listed_dom = Counter(); [listed_dom.update({d: n}) for (d, c), n in listed.items()]
    tol = {d: int(np.floor(MISSING_TOL_FRAC * listed_dom[d])) for d in DOMAINS}
    g2_fail = [d for d in DOMAINS if excl_by_dom[d] > tol[d]]
    g2 = {'status': 'FAIL' if (g2_fail or label_inconsistent) else 'PASS', 'listed_per_domain': dict(listed_dom), 'excluded_per_domain': {d: excl_by_dom[d] for d in DOMAINS},
          'tolerance_per_domain': tol, 'domains_over_tolerance': g2_fail, 'label_inconsistent_within_domain': label_inconsistent,
          'class_integer_labels_across_domains': {c: sorted(v) for c, v in cross.items()}, 'listed_per_cell': {f'{d}/{c}': listed[(d, c)] for d in DOMAINS for c in CLASSES}}
    # G3
    g3 = {'status': 'FAIL' if conflicts else 'PASS', 'within_domain_conflicts': conflicts, 'within_class_duplicates_collapsed': len(duplicates),
          'cross_domain_shared_identities': cross_domain_shared, 'cross_domain_class_conflicts_descriptive': len(cross_domain_conflicts)}
    # G4
    cell = Counter((r['domain'], r['class'], r['role']) for r in rows)
    fails = [{'domain': d, 'class': c, 'role': ro, 'n': cell[(d, c, ro)]} for d in DOMAINS for c in CLASSES for ro in ('proposal', 'audit') if cell[(d, c, ro)] < MIN_SUPPORT]
    g4 = {'status': 'FAIL' if fails else 'PASS', 'min_support': MIN_SUPPORT, 'failing_cells': fails,
          'cells': {f'{d}/{c}': {'unique': cell[(d, c, 'proposal')] + cell[(d, c, 'audit')], 'proposal': cell[(d, c, 'proposal')], 'audit': cell[(d, c, 'audit')]} for d in DOMAINS for c in CLASSES},
          'frame_sizes': {f'{d}/{ro}': sum(cell[(d, c, ro)] for c in CLASSES) for d in DOMAINS for ro in ('proposal', 'audit')},
          'min_cell_proposal': min(cell[(d, c, 'proposal')] for d in DOMAINS for c in CLASSES), 'min_cell_audit': min(cell[(d, c, 'audit')] for d in DOMAINS for c in CLASSES)}
    # G6
    v1ids = set()
    with open(a.v1_fold_manifest) as f:
        for r in csv.DictReader(f):
            v1ids.add(r['identity'])
    ov = sorted({r['identity'] for r in rows} & v1ids)
    g6 = {'status': 'OVERLAP_HOLD_VALIDITY' if ov else 'PASS', 'v1_identities': len(v1ids), 'overlap_count': len(ov), 'overlap_identities': ov,
          'overlap_rows': [r for r in rows if r['identity'] in set(ov)]}
    status = 'INVALID' if (g2['status'] == 'FAIL' or g3['status'] == 'FAIL' or g4['status'] == 'FAIL') else ('HOLD_VALIDITY_FLAG' if ov else 'PASS')
    out = {'protocol_id': PID, 'protocol_sha256': PROTOCOL_SHA256, 'status': status, 'G2_LIST_FILES': g2, 'G3_DUPLICATES': g3, 'G4_SUPPORT': g4, 'G6_PACS_OVERLAP': g6,
           'entries_listed_6class': len(entries), 'decoded_ok': len(ok), 'unique_rows': len(rows),
           'crops_shape': list(arr.shape), 'crops_sha256': sha(os.path.join(a.out, 'CROPS_uint8.npy')), 'manifest_sha256': sha(os.path.join(a.out, 'MANIFEST.csv')),
           'wall_seconds_cpu': time.time() - t0, 'model_outputs_computed': False}
    dump(os.path.join(a.out, 'DATA_CHECK.json'), out)
    print(json.dumps({k: (v['status'] if isinstance(v, dict) and 'status' in v else v) for k, v in out.items() if k not in ('G6_PACS_OVERLAP',)} | {'G6': g6['status'], 'G4_min': [g4['min_cell_proposal'], g4['min_cell_audit']]}, default=str))


if __name__ == '__main__':
    main()
