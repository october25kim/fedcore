"""Frozen V1 post-audit analysis: outcome rows, paired contrasts, paired-block bootstrap, post-decision truth, gates.

Pure numpy/scipy functions over the per-model audit arrays written by ``v1_pacs_runner``. Nothing here reads images,
labels or logits; the only inputs are ECA/flag/count tensors and the full-frame truth summaries computed *after* every
decision was fixed. All constants come from PROTOCOL.json (validated against the certifier's hard-coded family).
"""
import csv, hashlib, json
import numpy as np
from scipy.stats import binom
from . import v1_pacs_certifier as c

FIXED = list(c.METHODS)                      # 7 fixed procedures, frozen order
V1_NAME = 'V1_SCALE_CP'; GEO_NAME = 'GEOMETRY_DIRECT_CP'
PROCEDURES = FIXED + [V1_NAME, GEO_NAME]     # 9 procedure rows per (model, t, budget, replicate)
DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']


def protocol_constants(protocol):
    au = protocol['audit']; an = protocol['analysis']; pr = protocol['procedures']
    ts = [float(x) for x in au['protection_t']]
    budgets = sorted([int(au['primary_budget'])] + [int(b) for b in au['secondary_budgets']])
    k = {'alpha': float(au['alpha']), 'dr': float(au['delta_r']), 'dc': float(au['delta_c']), 'J': int(au['J']), 'ts': ts, 'weights': [float(w) for w in au['protection_weights']],
         'budgets': budgets, 'primary_budget': int(au['primary_budget']), 'replicates': int(au['replicates']), 'n_boot': 20000, 'boot_seed': int(an['bootstrap_seed']),
         'n_contrasts': int(an['primary_comparisons']) + int(an['additional_diagnostic_comparisons']),
         'rule': {float(t): pr['primary_rule'][t] for t in ('0.5', '1.0')}, 'diag': {float(t): pr['diagnostic_control'][t] for t in ('0.5', '1.0')},
         'fixed': list(pr['fixed_controls']), 'models': int(protocol['denominators']['models']), 'archs': list(protocol['training']['architectures']),
         'confirm_g': 0.01, 'kill_g': 0.002}
    if k['fixed'] != FIXED or k['n_contrasts'] != 16 or k['J'] != 4 or k['ts'] != [0.5, 1.0] or k['weights'] != [0.5, 0.5]:
        raise ValueError('PROTOCOL procedures/audit block disagrees with the frozen certifier family')
    if '20000' not in an['CI'] or '2026091601' != str(k['boot_seed']):
        raise ValueError('PROTOCOL CI block disagrees with the frozen bootstrap constants')
    return k


def vertices(t, J=4):
    return np.full((1, J), 1 / J) if t == 0 else (1 - t) * np.ones((J, J)) / J + t * np.eye(J)


def truth_summaries(acc, mass, t):
    """acc, mass: (12, J) full-frame acceptance and error-acceptance fractions per candidate/domain.
    Returns worst-mixture risk and worst-mixture (minimum) acceptance over the vertices of Lambda_t."""
    V = vertices(t, acc.shape[1])
    num = mass @ V.T                           # (12, nv): sum_j v_j m_ij  (error-acceptance mass under vertex v)
    den = acc @ V.T                            # (12, nv): sum_j v_j a_ij  (acceptance under vertex v)
    risk = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    return risk.max(axis=1), den.min(axis=1)


def joint_failure(flags, lcb, eligible, worst_risk, worst_acc, alpha):
    """flags (M, R, 12) bool; lcb (R, 12); eligible (12,); worst_risk/worst_acc (12,). Returns (M, R) failure events."""
    dep = flags & eligible[None, None, :] & (lcb[None] > 0)                   # deployment-eligible risk-passing candidates
    risk_fail = (dep & (worst_risk[None, None, :] > alpha)).any(-1)           # (M, R)
    acc_fail = (lcb > worst_acc[None, :]).any(-1)                             # (R,) any candidate LCB above true worst acceptance
    return risk_fail | acc_fail[None, :]


def assemble(audits, matrix, k):
    """audits: dict model_id -> dict of arrays; matrix: list of MODEL_MATRIX rows in frozen order.
    Returns the ECA tensor E[model, procedure(9), t(2), budget(3), replicate(100)] and per-model metadata."""
    nT, nB, R = len(k['ts']), len(k['budgets']), k['replicates']
    E = np.full((len(matrix), len(PROCEDURES), nT, nB, R), np.nan)
    chosen = np.full(E.shape, -2, dtype=np.int64)
    for mi, row in enumerate(matrix):
        aud = audits[row['model_id']]
        eca = aud['eca']; ch = aud['chosen']                                  # (T, B, M, R)
        for fi in range(len(FIXED)):
            E[mi, fi] = eca[:, :, fi]; chosen[mi, fi] = ch[:, :, fi]
        for ti, t in enumerate(k['ts']):
            for name, table in ((V1_NAME, k['rule']), (GEO_NAME, k['diag'])):
                src = FIXED.index(table[t]); pi = PROCEDURES.index(name)
                E[mi, pi, ti] = eca[ti, :, src]; chosen[mi, pi, ti] = ch[ti, :, src]
    return E, chosen


def outcome_rows(E, chosen, matrix, k):
    rows = []
    for mi, row in enumerate(matrix):
        for pi, proc in enumerate(PROCEDURES):
            for ti, t in enumerate(k['ts']):
                src = proc if proc in FIXED else (k['rule'] if proc == V1_NAME else k['diag'])[t]
                for bi, B in enumerate(k['budgets']):
                    for r in range(k['replicates']):
                        ci = int(chosen[mi, pi, ti, bi, r]); e = float(E[mi, pi, ti, bi, r])
                        rows.append({'model_id': row['model_id'], 'architecture': row['architecture'], 'split': int(row['split']), 'seed': int(row['seed']), 't': t, 'budget': B,
                                     'procedure': proc, 'derived': int(proc not in FIXED), 'source_procedure': src, 'replicate': r, 'certified': int(ci >= 0), 'candidate_index': ci, 'eca': e})
    return rows


def contrasts(E, matrix, k):
    """Paired per-replicate differences D[a, control, r] at the primary budget, averaged over 15 models and 2 t."""
    bi = k['budgets'].index(k['primary_budget']); v1 = PROCEDURES.index(V1_NAME); geo = PROCEDURES.index(GEO_NAME)
    arch_of = np.array([row['architecture'] for row in matrix])
    D = {}
    for a in k['archs']:
        sel = np.where(arch_of == a)[0]
        base = E[sel, v1, :, bi, :]                                          # (15, 2, R)
        for fi, name in enumerate(FIXED):
            D[(a, name)] = (base - E[sel, fi, :, bi, :]).mean(axis=(0, 1))    # (R,)
        D[(a, GEO_NAME)] = (base - E[sel, geo, :, bi, :]).mean(axis=(0, 1))
    return D


def bootstrap(D, k):
    rng = np.random.Generator(np.random.PCG64(k['boot_seed']))
    R = k['replicates']; idx = rng.integers(0, R, size=(k['n_boot'], R))
    lo_q = k['dr'] / (2 * k['n_contrasts']); hi_q = 1 - lo_q                # .05/32, 1-.05/32
    out = {}
    for key, d in D.items():
        boot = d[idx].mean(axis=1)
        out[key] = {'delta_hat': float(d.mean()), 'ci_lower': float(np.quantile(boot, lo_q, method='linear')), 'ci_upper': float(np.quantile(boot, hi_q, method='linear')),
                    'boot_mean': float(boot.mean()), 'boot_sd': float(boot.std(ddof=1))}
    return out, {'n_boot': k['n_boot'], 'seed': k['boot_seed'], 'rng': 'numpy PCG64 Generator, integers(0,R,size=(n_boot,R)) shared across all contrasts', 'quantiles': [lo_q, hi_q], 'method': 'linear',
                 'index_sha256': hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()}


def truth_groups(audits, matrix, k):
    """1,260 (model, budget, t, fixed method) groups: joint-family failure count over replicates and exact Bin(R,.10) upper-tail p."""
    rows = []; R = k['replicates']; thr = k['dr'] / 1260
    for row in matrix:
        f = audits[row['model_id']]['failure']                               # (T, B, M, R)
        for ti, t in enumerate(k['ts']):
            for bi, B in enumerate(k['budgets']):
                for mi, m in enumerate(FIXED):
                    cnt = int(f[ti, bi, mi].sum()); p = float(binom.sf(cnt - 1, R, 0.10)) if cnt > 0 else 1.0
                    rows.append({'model_id': row['model_id'], 'architecture': row['architecture'], 't': t, 'budget': B, 'procedure': m, 'failures': cnt, 'replicates': R, 'p_upper_exact': p, 'flag': int(p <= thr)})
    if len(rows) != 1260:
        raise ValueError(f'expected 1260 validity groups, got {len(rows)}')
    return rows, thr


def decide(boot, D, groups, invalid_reasons, incomplete_reasons, k):
    """Frozen precedence: INVALID, INCOMPLETE, HOLD_VALIDITY, KILL, CONFIRM, HOLD."""
    detail = {'invalid_reasons': invalid_reasons, 'incomplete_reasons': incomplete_reasons}
    if invalid_reasons:
        return 'INVALID', detail
    if incomplete_reasons:
        return 'INCOMPLETE', detail
    flagged = [g for g in groups if g['flag']]
    detail['validity_flagged_groups'] = len(flagged); detail['validity_threshold_p'] = k['dr'] / 1260
    if flagged:
        return 'HOLD_VALIDITY', detail
    g = {}; any_upper_le0 = False; all_lower_gt0 = True
    for a in k['archs']:
        deltas = [boot[(a, m)]['delta_hat'] for m in FIXED]; g[a] = float(min(deltas))
        any_upper_le0 |= any(boot[(a, m)]['ci_upper'] <= 0 for m in FIXED)
        all_lower_gt0 &= all(boot[(a, m)]['ci_lower'] > 0 for m in FIXED)
    detail['g_by_arch'] = g; detail['any_fixed_ci_upper_le_0'] = any_upper_le0; detail['all_14_ci_lower_gt_0'] = all_lower_gt0
    if any_upper_le0 or all(g[a] < k['kill_g'] for a in k['archs']):
        return 'KILL', detail
    if all(g[a] >= k['confirm_g'] for a in k['archs']) and all_lower_gt0:
        return 'CONFIRM', detail
    return 'HOLD', detail


def per_model_support(E, matrix, k):
    """Per-model paired deltas vs each control at the primary budget, each t separately and combined; win/tie/loss."""
    bi = k['budgets'].index(k['primary_budget']); v1 = PROCEDURES.index(V1_NAME)
    rows = []; wtl = {}
    for mi, row in enumerate(matrix):
        for name in FIXED + [GEO_NAME]:
            pi = PROCEDURES.index(name)
            d = E[mi, v1, :, bi, :] - E[mi, pi, :, bi, :]                      # (2, R)
            rec = {'model_id': row['model_id'], 'architecture': row['architecture'], 'split': int(row['split']), 'seed': int(row['seed']), 'control': name,
                   'delta_t05': float(d[0].mean()), 'delta_t10': float(d[1].mean()), 'delta_both': float(d.mean())}
            rows.append(rec)
            key = (row['architecture'], name); w = wtl.setdefault(key, [0, 0, 0])
            w[0 if rec['delta_both'] > 0 else (1 if rec['delta_both'] == 0 else 2)] += 1
    return rows, {f'{a}|{n}': {'win': v[0], 'tie': v[1], 'loss': v[2]} for (a, n), v in wtl.items()}


def secondary(E, audits, matrix, k):
    out = {'eca_mean_by_procedure_t_budget': {}, 'certification_rate_by_procedure_t_budget': {}, 'joint_failure_rate_by_procedure_t_budget': {}, 'unique_draws_mean_by_budget_domain': {}, 'domain_frame_sizes': {}}
    for pi, proc in enumerate(PROCEDURES):
        for ti, t in enumerate(k['ts']):
            for bi, B in enumerate(k['budgets']):
                key = f'{proc}|t={t}|B={B}'
                out['eca_mean_by_procedure_t_budget'][key] = float(np.nanmean(E[:, pi, ti, bi, :]))
                out['certification_rate_by_procedure_t_budget'][key] = float(np.nanmean(E[:, pi, ti, bi, :] > 0))
    for mi, m in enumerate(FIXED):
        for ti, t in enumerate(k['ts']):
            for bi, B in enumerate(k['budgets']):
                out['joint_failure_rate_by_procedure_t_budget'][f'{m}|t={t}|B={B}'] = float(np.mean([audits[r['model_id']]['failure'][ti, bi, mi].mean() for r in matrix]))
    for bi, B in enumerate(k['budgets']):
        for j, d in enumerate(DOMAINS):
            out['unique_draws_mean_by_budget_domain'][f'B={B}|{d}'] = float(np.mean([audits[r['model_id']]['unique_draws'][bi, :, j].mean() for r in matrix]))
    out['domain_frame_sizes'] = {r['model_id']: [int(x) for x in audits[r['model_id']]['audit_frame_sizes']] for r in matrix}
    return out


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
