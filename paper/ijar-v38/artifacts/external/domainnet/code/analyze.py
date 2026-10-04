"""S1 v2 analysis: V1 analysis functions (unmodified) with v2 constants; primary + pre-specified secondary; descriptive V1 comparison."""
import csv, json, hashlib
from pathlib import Path
import numpy as np
from common import *
from runner import an, k_v2, load_matrix, BUDGETS, TS, R, J, PRIMARY_B, utc


def secondary_v2(E, audits, matrix, k):
    out = {'eca_mean_by_procedure_t_budget': {}, 'certification_rate_by_procedure_t_budget': {}, 'joint_failure_rate_by_procedure_t_budget': {}, 'unique_draws_mean_by_budget_domain': {}}
    for pi, proc in enumerate(an.PROCEDURES):
        for ti, t in enumerate(k['ts']):
            for bi, B in enumerate(k['budgets']):
                key = f'{proc}|t={t}|B={B}'
                out['eca_mean_by_procedure_t_budget'][key] = float(np.nanmean(E[:, pi, ti, bi, :]))
                out['certification_rate_by_procedure_t_budget'][key] = float(np.nanmean(E[:, pi, ti, bi, :] > 0))
    for mi, m in enumerate(an.FIXED):
        for ti, t in enumerate(k['ts']):
            for bi, B in enumerate(k['budgets']):
                out['joint_failure_rate_by_procedure_t_budget'][f'{m}|t={t}|B={B}'] = float(np.mean([audits[r['model_id']]['failure'][ti, bi, mi].mean() for r in matrix]))
    for bi, B in enumerate(k['budgets']):
        for j, d in enumerate(DOMAINS):
            out['unique_draws_mean_by_budget_domain'][f'B={B}|{d}'] = float(np.mean([audits[r['model_id']]['unique_draws'][bi, :, j].mean() for r in matrix]))
    out['audit_frame_sizes'] = {d: int(audits[matrix[0]['model_id']]['audit_frame_sizes'][j]) for j, d in enumerate(DOMAINS)}
    out['proposal_frame_sizes'] = {d: int(audits[matrix[0]['model_id']]['proposal_frame_sizes'][j]) for j, d in enumerate(DOMAINS)}
    out['eligible_candidates_per_model'] = {r['model_id']: int(audits[r['model_id']]['eligible'].sum()) for r in matrix}
    return out


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def boot_rows(boot, label):
    return [{'analysis': label, 'architecture': a_, 'control': m_, 'contrast': 'V1_SCALE_CP minus control', 'primary': int(m_ in an.FIXED), **v} for (a_, m_), v in boot.items()]


def stage_analyze(args):
    out = Path(args.run_dir); matrix = load_matrix(args.model_matrix)
    k = k_v2(['resnet18', 'convnext_tiny'])
    invalid = []; incomplete = []
    binding = json.load(open(out / 'RUN_BINDING.json'))
    if binding['status'] != 'PASS':
        invalid.append('RUN_BINDING not PASS')
    dc = json.load(open(Path(args.data_dir) / 'DATA_CHECK.json'))
    for gname in ('G2_LIST_FILES', 'G3_DUPLICATES', 'G4_SUPPORT'):
        if dc[gname]['status'] != 'PASS':
            invalid.append(f'{gname} FAIL')
    overlap = dc['G6_PACS_OVERLAP']['overlap_count']
    st = json.load(open(out / 'AUDIT_STATE.json'))
    audits = {}
    for m in matrix:
        mid = m['model_id']; p = out / 'audit' / f'{mid}.npz'
        if mid not in st['models'] or not p.exists():
            incomplete.append(f'{mid} not audited'); continue
        if sha(p) != st['models'][mid]['audit_sha256']:
            invalid.append(f'{mid} audit hash changed'); continue
        z = np.load(p, allow_pickle=False); audits[mid] = {x: z[x] for x in z.files}
        from runner import audit_checks_v2
        invalid += [f'{mid}: {x}' for x in audit_checks_v2(audits[mid])]
    if incomplete:
        dump(out / 'DECISION.json', {'decision': 'INCOMPLETE', 'incomplete_reasons': incomplete, 'invalid_reasons': invalid, 'utc': utc()}); return 10
    E, chosen = an.assemble(audits, matrix, k)
    rows = an.outcome_rows(E, chosen, matrix, k)
    if len(rows) != 162000 or np.isnan(E).any():
        invalid.append(f'outcome rows {len(rows)} != 162000 or NaN')
    for ti, t in enumerate(k['ts']):
        for name, table in ((an.V1_NAME, k['rule']), (an.GEO_NAME, k['diag'])):
            if not np.array_equal(E[:, an.PROCEDURES.index(name), ti], E[:, an.FIXED.index(table[t]), ti]):
                invalid.append(f'{name} rows differ from {table[t]} at t={t}')
    flags_total = sum(int(audits[m['model_id']]['flags'].size) for m in matrix)
    if flags_total != 1512000:
        invalid.append(f'candidate flags {flags_total}')
    primary_rows = sum(r['budget'] == PRIMARY_B for r in rows)
    D = an.contrasts(E, matrix, k); boot, boot_meta = an.bootstrap(D, k)
    groups, thr = an.truth_groups(audits, matrix, k)
    decision, detail = an.decide(boot, D, groups, invalid, [], k)
    if decision not in ('INVALID', 'INCOMPLETE', 'HOLD_VALIDITY') and overlap > 0:
        detail['pre_decision_v1_rule'] = decision; decision = 'HOLD_VALIDITY'; detail['pacs_overlap_count'] = overlap
    detail['pacs_overlap_count'] = overlap
    label = {'CONFIRM': 'TRANSFER', 'KILL': 'NOT_TRANSFER', 'HOLD': 'HOLD (inconclusive transfer)'}.get(decision, decision)
    support_rows, wtl = an.per_model_support(E, matrix, k); sec = secondary_v2(E, audits, matrix, k)
    # ---- write primary ----
    an.write_csv(out / 'OUTCOMES.csv', rows)
    write_csv(out / 'CONTRASTS.csv', [{kk: vv for kk, vv in r.items() if kk != 'analysis'} for r in boot_rows(boot, 'primary')])
    an.write_csv(out / 'VALIDITY_GROUPS.csv', groups); an.write_csv(out / 'PER_MODEL_DELTAS.csv', support_rows)
    dump(out / 'BOOTSTRAP.json', {'meta': boot_meta, 'contrasts': {f'{a_}|{m_}': v for (a_, m_), v in boot.items()},
                                  'per_replicate_D_sha256': hashlib.sha256(np.stack([D[key] for key in sorted(D)]).tobytes()).hexdigest()})
    dec = {'decision': decision, 'transfer_label': label, 'protocol_id': PID, 'protocol_sha256': PROTOCOL_SHA256, 'models': 30, 'rows_total': len(rows), 'primary_rows': primary_rows,
           'candidate_flags': flags_total, 'validity_groups': len(groups), 'validity_threshold_p': thr, 'detail': detail, 'bootstrap_seed': BOOT_SEED, 'J': J,
           'per_domain_draws': {B: B // J for B in BUDGETS}, 'utc': utc(),
           'note': 'S1 v2 DomainNet-6 transfer probe under frozen V1 predictors. Descriptive of transfer only; does not rewrite V1 (V1 remains HOLD).'}
    dump(out / 'DECISION.json', dec)
    # ---- g per arch + margins ----
    g = {}
    for a_ in k['archs']:
        deltas = {m: boot[(a_, m)]['delta_hat'] for m in an.FIXED}
        amin = min(deltas, key=deltas.get)
        g[a_] = {'g_min_mean_diff': deltas[amin], 'argmin_control': amin, 'n_ci_lower_gt0': sum(boot[(a_, m)]['ci_lower'] > 0 for m in an.FIXED),
                 'n_ci_upper_le0': sum(boot[(a_, m)]['ci_upper'] <= 0 for m in an.FIXED), 'models': 15}
    dump(out / 'G_BY_ARCH.json', g)
    # ---- secondary subsets (descriptive) ----
    extra = {}
    for tag, splits in (('SECONDARY_person_unknown_splits_0_3_4', PERSON_UNKNOWN_SPLITS), ('DESCRIPTIVE_person_known_splits_1_2', PERSON_KNOWN_SPLITS)):
        sel = [i for i, m in enumerate(matrix) if int(m['split']) in splits]; msub = [matrix[i] for i in sel]
        Ds = an.contrasts(E[sel], msub, k); bs, _ = an.bootstrap(Ds, k)
        gs = [gr for gr in groups if gr['model_id'] in {m['model_id'] for m in msub}]
        dsub, detsub = an.decide(bs, Ds, gs, [], [], k)
        write_csv(out / f'CONTRASTS_{tag}.csv', boot_rows(bs, tag))
        extra[tag] = {'splits': splits, 'models': len(msub), 'models_per_arch': len(msub) // 2, 'v1_rule_applied_descriptively': dsub, 'detail': detsub,
                      'note': 'descriptive; does not override the primary verdict'}
    # per-t contrasts (descriptive)
    bi = BUDGETS.index(PRIMARY_B); v1 = an.PROCEDURES.index(an.V1_NAME); arch_of = np.array([m['architecture'] for m in matrix])
    rows_t = []
    for a_ in k['archs']:
        s = np.where(arch_of == a_)[0]
        for ti, t in enumerate(TS):
            for name in an.FIXED + [an.GEO_NAME]:
                d = (E[s, v1, ti, bi, :] - E[s, an.PROCEDURES.index(name), ti, bi, :]).mean(axis=0)
                rows_t.append({'architecture': a_, 't': t, 'control': name, 'delta_hat': float(d.mean())})
    write_csv(out / 'CONTRASTS_BY_T_descriptive.csv', rows_t)
    # ECA by arch/procedure/t/budget + cert rate
    eca_rows = []
    for a_ in k['archs']:
        s = np.where(arch_of == a_)[0]
        for pi, proc in enumerate(an.PROCEDURES):
            for ti, t in enumerate(TS):
                for bj, B in enumerate(BUDGETS):
                    x = E[s, pi, ti, bj, :]
                    eca_rows.append({'architecture': a_, 'procedure': proc, 't': t, 'budget': B, 'eca_mean': float(x.mean()), 'certification_rate': float((x > 0).mean()), 'n_model_replicates': int(x.size)})
    write_csv(out / 'ECA_BY_ARCH_PROCEDURE_T_BUDGET.csv', eca_rows)
    # split/seed means
    ss = []
    for i, m in enumerate(matrix):
        for pi, proc in enumerate(an.PROCEDURES):
            ss.append({'model_id': m['model_id'], 'architecture': m['architecture'], 'split': int(m['split']), 'seed': int(m['seed']), 'procedure': proc,
                       'eca_B512_mean_over_t_rep': float(E[i, pi, :, bi, :].mean()), 'cert_rate_B512': float((E[i, pi, :, bi, :] > 0).mean())})
    write_csv(out / 'PER_MODEL_ECA.csv', ss)
    # ---- post-decision diagnostics ----
    pdiag = {}
    for m in matrix:
        if int(m['split']) not in PERSON_KNOWN_SPLITS:
            continue
        z = np.load(out / 'logits' / f"{m['model_id']}.npz", allow_pickle=False)
        for d in DOMAINS:
            lg = z[f'audit__{d}__logits']; pdiag.setdefault(m['architecture'], {}).setdefault(d, []).append(float((lg.argmax(1) == 3).mean()))
    person = {a_: {d: float(np.mean(v)) for d, v in dd.items()} for a_, dd in pdiag.items()}
    dump(out / 'PERSON_DIAGNOSTIC.json', {'metric': 'fraction of full audit-frame images with argmax = person head (index 3), splits 1,2, mean over 6 models per arch', 'values': person, 'post_decision': True})
    dom = {}
    for a_ in k['archs']:
        s = [i for i, m in enumerate(matrix) if m['architecture'] == a_]
        vals = {d: [] for d in DOMAINS}
        for i in s:
            au = audits[matrix[i]['model_id']]
            for ti in range(2):
                for r in range(R):
                    ci = int(au['chosen'][ti, bi, an.FIXED.index(k['rule'][TS[ti]]), r])
                    if ci >= 0:
                        for j, d in enumerate(DOMAINS):
                            vals[d].append(float(au['truth_acc'][ci, j]))
        dom[a_] = {d: {'mean_true_acceptance_of_V1_SCALE_CP_choice': (float(np.mean(v)) if v else None), 'n_certified_model_t_replicates': len(v)} for d, v in vals.items()}
    sec['domain_level_V1_SCALE_CP_B512'] = dom
    dump(out / 'SECONDARY.json', {**sec, 'win_tie_loss': wtl, 'rows': len(rows), 'candidate_flags': flags_total, 'primary_rows': primary_rows, 'subsets': extra})
    # ---- descriptive comparison with V1 ----
    v1c = {(r['architecture'], r['control']): r for r in csv.DictReader(open(args.v1_contrasts))}
    comp = []
    for (a_, m_), v in boot.items():
        o = v1c[(a_, m_)]
        comp.append({'architecture': a_, 'control': m_, 'primary': int(m_ in an.FIXED), 'V1_PACS_delta_hat': float(o['delta_hat']), 'V1_ci_lower': float(o['ci_lower']), 'V1_ci_upper': float(o['ci_upper']),
                     'S1v2_DomainNet6_delta_hat': v['delta_hat'], 'S1v2_ci_lower': v['ci_lower'], 'S1v2_ci_upper': v['ci_upper'],
                     'V1_lower_gt0': int(float(o['ci_lower']) > 0), 'S1v2_lower_gt0': int(v['ci_lower'] > 0)})
    write_csv(out / 'COMPARISON_V1_descriptive.csv', comp)
    dump(out / 'DENOMINATORS.json', {'models': 30, 'rows_total': len(rows), 'rows_expected': 162000, 'primary_rows': primary_rows, 'primary_expected': 54000, 'candidate_flags': flags_total,
                                     'flags_expected': 1512000, 'validity_groups': len(groups), 'contrasts': len(boot), 'per_model_delta_rows': len(support_rows),
                                     'replicates': R, 'budgets': BUDGETS, 'per_domain_draws': {B: B // J for B in BUDGETS}, 'ts': TS, 'secondary_subset_models': {k2: v['models'] for k2, v in extra.items()}})
    print(json.dumps({'decision': decision, 'label': label, 'g': {a_: g[a_]['g_min_mean_diff'] for a_ in g}, 'invalid': invalid[:5]}))
    return 0
