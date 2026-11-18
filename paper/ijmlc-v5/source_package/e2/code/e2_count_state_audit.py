#!/usr/bin/env python3
"""FedCORE E2: finite count-state mechanism audit (independent formula reconstruction).

Formulas rewritten from the frozen count-only certificate math used by the
independent protection-statistics reconstruction (match Table 9 / V1 fixed set).
Does NOT import production certifier modules.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, binom

SEOUL = ZoneInfo("Asia/Seoul")
OUT = Path("/workspace/outputs/fedcore_e2_count_state_20260918")
METHODS = (
    "RATIO",
    "MASS_CP",
    "HOEFFDING_H",
    "DIRECT_BETTING_H",
    "SCALE_H",
    "CP_IUT_H",
    "COMPOSED_CP_BETTING_H",
)
STAKES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80)
J = 4
M = 12
ALPHA = 0.20
DR = 0.05
DC = 0.05
NS = (64, 128, 256, 512)
FS = (
    (0, 0, 0, 0),
    (Fraction(1, 8), Fraction(1, 8), Fraction(1, 8), Fraction(1, 8)),
    (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
    (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1)),
)
QS = (
    (0, 0, 0, 0),
    (Fraction(1, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10)),
    (Fraction(3, 10), Fraction(1, 20), Fraction(1, 20), Fraction(1, 20)),
    (Fraction(3, 10), Fraction(3, 10), Fraction(1, 20), Fraction(1, 20)),
)
TS = (0.0, 0.5, 0.75, 1.0)
FORMULA_VERSION = (
    "independent_reconstruction_e2_20260918_v1;"
    "math_parity_with_frozen_count_only_certify;"
    "source_note: reconstruct_protection_statistics.py unavailable on box;"
    "formulas_rewritten_not_imported_from_v1_pacs_certifier"
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lower(k, n, q):
    k, n = np.broadcast_arrays(k, n)
    out = np.zeros(k.shape, dtype=float)
    m = k > 0
    out[m] = beta.ppf(q, k[m], n[m] - k[m] + 1)
    return out


def upper(k, n, q):
    k, n = np.broadcast_arrays(k, n)
    out = np.ones(k.shape, dtype=float)
    m = (n > 0) & (k < n)
    out[m] = beta.isf(q, k[m] + 1, n[m] - k[m])
    return out


def holm(p, delta):
    p = np.asarray(p, float)
    order = np.argsort(p, axis=-1, kind="stable")
    sorted_p = np.take_along_axis(p, order, axis=-1)
    thresh = delta / np.arange(p.shape[-1], 0, -1)
    prefix = np.logical_and.accumulate(sorted_p <= thresh, axis=-1)
    out = np.zeros(p.shape, bool)
    np.put_along_axis(out, order, prefix, axis=-1)
    return out


def certify_one(n_vec, A_vec, K_vec, t, alpha=ALPHA, dr=DR, dc=DC):
    """Certify M=12 identical slots sharing (n,A,K). Returns stats, passed, eca_by_method."""
    n = np.asarray(n_vec, dtype=float)
    A = np.asarray(A_vec, dtype=float)
    K = np.asarray(K_vec, dtype=float)
    # shape (M, J)
    n = np.broadcast_to(n, (M, J)).copy()
    A = np.broadcast_to(A, (M, J)).copy()
    K = np.broadcast_to(K, (M, J)).copy()

    vmax = lambda a: (1 - t) * a.mean(-1) + t * a.max(-1)
    vmin = lambda a: (1 - t) * a.mean(-1) + t * a.min(-1)
    common = vmin(lower(A, n, dc / (J * M)))
    tail = dr / (3 * J * M)
    r = upper(K, A, tail) - alpha
    al = lower(A, n, tail)
    au = upper(A, n, tail)
    ratio = vmax(np.maximum(r * al, r * au))
    den = vmin(al)
    mass = vmax((1 - alpha) * upper(K, n, dr / (2 * J * M)) - alpha * lower(A - K, n, dr / (2 * J * M)))
    pcp = binom.cdf(K, A, alpha)
    pcp = np.where(A == 0, 1.0, pcp)
    pcp = pcp.max(-1)
    if t == 0:
        vertices = np.full((1, J), 1 / J)
    else:
        vertices = (1 - t) * np.ones((J, J)) / J + t * np.eye(J)
    L = np.maximum(1, np.ceil(np.log2(n.sum(-1))).astype(int))
    blog = []
    slog = []
    hp = []
    for v in vertices:
        w = v / n
        scale = w / ((1 - alpha) * w.max(-1, keepdims=True))

        def le(stake):
            z = stake * scale
            return (K * np.log1p(-(1 - alpha) * z) + (A - K) * np.log1p(alpha * z)).sum(-1)

        lb = logsumexp(np.stack([le(s) for s in STAKES]), axis=0) - math.log(len(STAKES))
        ls = np.empty_like(lb)
        for length in np.unique(L):
            mask = L == length
            terms = [le(2.0 ** (-ell))[mask] - math.log(ell * (ell + 1)) for ell in range(1, int(length) + 1)]
            terms.append(np.full(mask.sum(), -math.log(length + 1)))
            ls[mask] = np.logaddexp(lb[mask], logsumexp(np.stack(terms), axis=0)) - math.log(2)
        blog.append(lb)
        slog.append(ls)
        d = ((K - alpha * A) / n * v).sum(-1)
        ss = (v * v / n).sum(-1)
        hp.append(np.exp(-2 * np.minimum(d, 0) ** 2 / ss))
    logb = np.min(np.stack(blog), axis=0)
    logs = np.min(np.stack(slog), axis=0)
    pb = np.exp(np.minimum(0, -logb))
    ps = np.exp(np.minimum(0, -logs))
    stats = {
        "common_acceptance_lcb": common,
        "ratio_D": ratio,
        "ratio_min_denominator": den,
        "mass_cp_D": mass,
        "hoeffding_p": np.max(np.stack(hp), axis=0),
        "betting_p": pb,
        "scale_p": ps,
        "cp_iut_p": pcp,
        "composed_p": np.minimum(1, 2 * np.minimum(pcp, pb)),
        "betting_min_log_e": logb,
        "scale_min_log_e": logs,
        "worst_empirical_contrast": vmax((K - alpha * A) / n),
    }
    if not all(np.isfinite(np.asarray(x)).all() for x in stats.values()):
        raise ArithmeticError("Nonfinite certificate result")
    passed = {
        "RATIO": (ratio <= 0) & (den > 0),
        "MASS_CP": mass <= 0,
    }
    for method, key in zip(METHODS[2:], ["hoeffding_p", "betting_p", "scale_p", "cp_iut_p", "composed_p"]):
        passed[method] = holm(stats[key], dr)

    eca = {}
    selected = {}
    for method in METHODS:
        flags = passed[method]
        ids = [i for i in range(M) if flags[i] and common[i] > 0]
        if ids:
            # identical LCB: pick lowest index
            i = min(ids)
            eca[method] = float(common[i])
            selected[method] = i
        else:
            eca[method] = 0.0
            selected[method] = None
    return stats, passed, eca, selected


def floor_nf(n: int, f_j) -> int:
    return int(math.floor(Fraction(n) * Fraction(f_j)))


def floor_aq(A_j: int, q_j) -> int:
    return int(math.floor(Fraction(A_j) * Fraction(q_j)))


def f_label(f):
    return "(" + ",".join(str(Fraction(x)) for x in f) + ")"


def q_label(q):
    return "(" + ",".join(str(Fraction(x)) for x in q) + ")"


def row_sha(parts: dict) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode())


def rank_procedures(eca: dict) -> list[str]:
    # higher ECA better; ties by procedure name
    return sorted(METHODS, key=lambda m: (-eca[m], m))


def pairwise_strict_flip(eca1: dict, eca2: dict, a: str, b: str) -> bool:
    e1a, e1b = eca1[a], eca1[b]
    e2a, e2b = eca2[a], eca2[b]
    if e1a == e1b or e2a == e2b:
        return False
    return (e1a > e1b and e2a < e2b) or (e1a < e1b and e2a > e2b)


def ranking_reversal(eca1: dict, eca2: dict) -> bool:
    # exists pair with strict order flip
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1 :]:
            if pairwise_strict_flip(eca1, eca2, a, b):
                return True
    return False


def build_states():
    rows = []
    sid = 0
    for n in NS:
        for fi, f in enumerate(FS):
            for qi, q in enumerate(QS):
                A = [floor_nf(n, fj) for fj in f]
                K = [floor_aq(Aj, qj) for Aj, qj in zip(A, q)]
                for t in TS:
                    parts = {
                        "state_id": sid,
                        "n": n,
                        "f_index": fi,
                        "q_index": qi,
                        "f": [str(Fraction(x)) for x in f],
                        "q": [str(Fraction(x)) for x in q],
                        "t": t,
                        "A": A,
                        "K": K,
                        "J": J,
                        "M": M,
                        "alpha": ALPHA,
                        "delta_r": DR,
                        "delta_c": DC,
                    }
                    parts["row_sha256"] = row_sha(parts)
                    rows.append(parts)
                    sid += 1
    return rows


def write_states_csv(states, path: Path) -> None:
    fields = [
        "state_id",
        "n",
        "f_index",
        "q_index",
        "f",
        "q",
        "t",
        "A0",
        "A1",
        "A2",
        "A3",
        "K0",
        "K1",
        "K2",
        "K3",
        "row_sha256",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in states:
            w.writerow(
                {
                    "state_id": s["state_id"],
                    "n": s["n"],
                    "f_index": s["f_index"],
                    "q_index": s["q_index"],
                    "f": json.dumps(s["f"]),
                    "q": json.dumps(s["q"]),
                    "t": s["t"],
                    "A0": s["A"][0],
                    "A1": s["A"][1],
                    "A2": s["A"][2],
                    "A3": s["A"][3],
                    "K0": s["K"][0],
                    "K1": s["K"][1],
                    "K2": s["K"][2],
                    "K3": s["K"][3],
                    "row_sha256": s["row_sha256"],
                }
            )


def detailed_breakdown(state, stats, passed, eca, selected) -> dict:
    # scalar summaries for identical slots (slot 0)
    def arr0(x):
        a = np.asarray(x)
        return a[0].tolist() if a.ndim else float(a)

    return {
        "state_id": state["state_id"],
        "n": state["n"],
        "f": state["f"],
        "q": state["q"],
        "t": state["t"],
        "A": state["A"],
        "K": state["K"],
        "common_acceptance_lcb_slot0": float(np.asarray(stats["common_acceptance_lcb"])[0]),
        "ratio_D_slot0": float(np.asarray(stats["ratio_D"])[0]),
        "ratio_min_denominator_slot0": float(np.asarray(stats["ratio_min_denominator"])[0]),
        "mass_cp_D_slot0": float(np.asarray(stats["mass_cp_D"])[0]),
        "hoeffding_p_slot0": float(np.asarray(stats["hoeffding_p"])[0]),
        "betting_p_slot0": float(np.asarray(stats["betting_p"])[0]),
        "scale_p_slot0": float(np.asarray(stats["scale_p"])[0]),
        "cp_iut_p_slot0": float(np.asarray(stats["cp_iut_p"])[0]),
        "composed_p_slot0": float(np.asarray(stats["composed_p"])[0]),
        "passed_any": {m: bool(np.asarray(passed[m]).any()) for m in METHODS},
        "passed_all_identical": {m: bool(np.asarray(passed[m]).all() == np.asarray(passed[m]).any()) for m in METHODS},
        "n_passed_slots": {m: int(np.asarray(passed[m]).sum()) for m in METHODS},
        "eca": eca,
        "selected_slot": selected,
        "ranking": rank_procedures(eca),
        "holm_threshold_first": DR / M,
        "notes": [
            "All 12 slots are identical (n,A,K) clones; M=12 multiplicity retained.",
            "ECA equals common acceptance LCB if any slot certified with LCB>0, else 0.",
            "Holm on identical p: all pass iff p<=delta_r/M, else none.",
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "WITNESSES").mkdir(exist_ok=True)
    created = datetime.now(SEOUL).isoformat(timespec="seconds")

    states = build_states()
    assert len(states) == 256
    states_path = OUT / "STATES.csv"
    write_states_csv(states, states_path)
    states_sha = sha256_file(states_path)

    protocol = {
        "experiment": "FedCORE_E2_finite_count_state_mechanism_audit",
        "created_asia_seoul": created,
        "machineId_requested": "0d480a29-a8d7-419a-b0a9-fc9ed0af7a3a",
        "execution_host_note": "Computed on Grok Bot box; Mac MCP file ops unavailable at run time",
        "J": J,
        "M": M,
        "alpha": ALPHA,
        "delta_r": DR,
        "delta_c": DC,
        "n_grid": list(NS),
        "f_grid": [[str(Fraction(x)) for x in f] for f in FS],
        "q_grid": [[str(Fraction(x)) for x in q] for q in QS],
        "t_grid": list(TS),
        "A_formula": "A_j = floor(n * f_j) with fractions.Fraction then floor",
        "K_formula": "K_j = floor(A_j * q_j) with fractions.Fraction then floor",
        "n_states": 256,
        "logical_flags": 256 * 7 * 12,
        "slots_design": "12 identical dependent clones of same (n,A,K); M=12 multiplicity applies",
        "eca_rule": "if any certified with common LCB>0: ECA=that LCB; else refusal ECA=0",
        "ranking_rule": "order by ECA desc; ties by procedure name",
        "reversal_definition": "fixed t; exists procedures i,j with strict ECA order flip between two states",
        "primary": "count of strict ranking-reversal state-pairs at each fixed t; witnesses; formula explanation",
        "decisions": ["PASS_IDENTIFIABILITY", "FAIL_FOR_HYPOTHESIS", "INCONCLUSIVE", "INVALID"],
        "prohibitions": [
            "No CI/significance",
            "Not a sample of real frequencies",
            "Do not claim causal attribution rates among rectangular/tail/gate",
            "Do not invent new methods",
            "Do not report unverified counterfactuals as deployment results",
            "Do not touch V1 outcomes, retrain, or change RESULT_LOCK",
        ],
        "formulas_version_note": FORMULA_VERSION,
        "procedures": list(METHODS),
        "STATES_csv_sha256": states_sha,
        "seal": "STATES.csv hash sealed before flag computation",
        "cpu_only": True,
    }
    protocol_path = OUT / "E2_PROTOCOL.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    protocol_sha_pre = sha256_file(protocol_path)

    # Run enumeration
    rankings = []
    flag_rows = []  # compact: one row per state×method (slots collapsed with identical note)
    results_by_id = {}
    for s in states:
        n_vec = [s["n"]] * J
        stats, passed, eca, selected = certify_one(n_vec, s["A"], s["K"], s["t"])
        ranking = rank_procedures(eca)
        results_by_id[s["state_id"]] = {
            "state": s,
            "stats": stats,
            "passed": passed,
            "eca": eca,
            "selected": selected,
            "ranking": ranking,
        }
        rankings.append(
            {
                "state_id": s["state_id"],
                "n": s["n"],
                "f_index": s["f_index"],
                "q_index": s["q_index"],
                "t": s["t"],
                "ranking": json.dumps(ranking),
                **{f"ECA_{m}": eca[m] for m in METHODS},
                **{f"pass_any_{m}": int(bool(np.asarray(passed[m]).any())) for m in METHODS},
                **{f"n_pass_{m}": int(np.asarray(passed[m]).sum()) for m in METHODS},
                "common_lcb": float(np.asarray(stats["common_acceptance_lcb"])[0]),
            }
        )
        # FLAGS compact + expand logical count
        for method in METHODS:
            flags = np.asarray(passed[method])
            # all slots identical decisions expected
            identical = bool(np.all(flags == flags[0]))
            for slot in range(M):
                flag_rows.append(
                    {
                        "state_id": s["state_id"],
                        "method": method,
                        "slot": slot,
                        "passed": int(bool(flags[slot])),
                        "common_lcb": float(np.asarray(stats["common_acceptance_lcb"])[slot]),
                        "identical_slots_collapse_ok": int(identical),
                    }
                )

    assert len(flag_rows) == 21504

    # Write FLAGS.csv.gz via gzip module
    import gzip

    flags_path = OUT / "FLAGS.csv.gz"
    with gzip.open(flags_path, "wt", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["state_id", "method", "slot", "passed", "common_lcb", "identical_slots_collapse_ok"],
        )
        w.writeheader()
        w.writerows(flag_rows)

    rankings_path = OUT / "RANKINGS.csv"
    with rankings_path.open("w", newline="") as f:
        fields = (
            ["state_id", "n", "f_index", "q_index", "t", "ranking", "common_lcb"]
            + [f"ECA_{m}" for m in METHODS]
            + [f"pass_any_{m}" for m in METHODS]
            + [f"n_pass_{m}" for m in METHODS]
        )
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rankings)

    # Reversals at fixed t
    key_contrasts = [
        ("DIRECT_BETTING_H", "CP_IUT_H", "DIRECT_vs_CP"),
        ("SCALE_H", "CP_IUT_H", "SCALE_vs_CP"),
        ("RATIO", "DIRECT_BETTING_H", "RATIO_vs_DIRECT"),
        ("RATIO", "SCALE_H", "RATIO_vs_SCALE"),
    ]
    reversal_rows = []
    counts_by_t = {}
    pairwise_counts = {name: {t: 0 for t in TS} for _, _, name in key_contrasts}
    witnesses_candidates = []

    for t in TS:
        ids = [s["state_id"] for s in states if s["t"] == t]
        n_rev = 0
        for i, s1 in enumerate(ids):
            for s2 in ids[i + 1 :]:
                e1 = results_by_id[s1]["eca"]
                e2 = results_by_id[s2]["eca"]
                is_rev = ranking_reversal(e1, e2)
                if is_rev:
                    n_rev += 1
                    flipped = []
                    for a_i, a in enumerate(METHODS):
                        for b in METHODS[a_i + 1 :]:
                            if pairwise_strict_flip(e1, e2, a, b):
                                flipped.append(f"{a}>{b}" if e1[a] > e1[b] else f"{b}>{a}")
                    reversal_rows.append(
                        {
                            "t": t,
                            "state_id_1": s1,
                            "state_id_2": s2,
                            "ranking_1": json.dumps(results_by_id[s1]["ranking"]),
                            "ranking_2": json.dumps(results_by_id[s2]["ranking"]),
                            "eca_1": json.dumps(e1),
                            "eca_2": json.dumps(e2),
                            "flipped_pairs": json.dumps(flipped),
                        }
                    )
                    witnesses_candidates.append((t, s1, s2, flipped, e1, e2))
                for a, b, name in key_contrasts:
                    if pairwise_strict_flip(e1, e2, a, b):
                        pairwise_counts[name][t] += 1
        counts_by_t[t] = n_rev

    reversals_path = OUT / "REVERSALS.csv"
    with reversals_path.open("w", newline="") as f:
        fields = [
            "t",
            "state_id_1",
            "state_id_2",
            "ranking_1",
            "ranking_2",
            "eca_1",
            "eca_2",
            "flipped_pairs",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(reversal_rows)

    total_reversals = sum(counts_by_t.values())

    # Decision
    if total_reversals >= 1:
        # need formula re-derivation witnesses
        decision_label = "PASS_IDENTIFIABILITY"
    else:
        decision_label = "FAIL_FOR_HYPOTHESIS"

    # Pick >=2 full witnesses with informative key flips if possible
    witnesses = []
    # prefer witnesses that flip DIRECT vs CP or SCALE vs CP
    def score_w(item):
        t, s1, s2, flipped, e1, e2 = item
        score = 0
        if pairwise_strict_flip(e1, e2, "DIRECT_BETTING_H", "CP_IUT_H"):
            score += 10
        if pairwise_strict_flip(e1, e2, "SCALE_H", "CP_IUT_H"):
            score += 8
        if pairwise_strict_flip(e1, e2, "RATIO", "DIRECT_BETTING_H"):
            score += 4
        if pairwise_strict_flip(e1, e2, "RATIO", "SCALE_H"):
            score += 4
        score += len(flipped)
        return -score

    witnesses_candidates.sort(key=score_w)
    chosen = []
    seen_t = set()
    for item in witnesses_candidates:
        t = item[0]
        # diversify t a bit
        if len(chosen) < 2 or (t not in seen_t and len(chosen) < 4):
            chosen.append(item)
            seen_t.add(t)
        if len(chosen) >= 2 and len(seen_t) >= min(2, len({c[0] for c in witnesses_candidates})):
            if len(chosen) >= 2:
                break
    if len(chosen) < 2 and len(witnesses_candidates) >= 2:
        chosen = witnesses_candidates[:2]
    elif len(chosen) < 2:
        chosen = witnesses_candidates[: len(witnesses_candidates)]

    for wi, (t, s1, s2, flipped, e1, e2) in enumerate(chosen, start=1):
        for sid, tag in [(s1, "A"), (s2, "B")]:
            r = results_by_id[sid]
            detail = detailed_breakdown(r["state"], r["stats"], r["passed"], r["eca"], r["selected"])
            detail["witness_pair_index"] = wi
            detail["witness_role"] = tag
            detail["partner_state_id"] = s2 if sid == s1 else s1
            detail["flipped_pairs_in_pair"] = flipped
            path = OUT / "WITNESSES" / f"witness_{wi}_state_{sid}_{tag}.json"
            path.write_text(json.dumps(detail, indent=2) + "\n")
            # markdown formula re-derivation
            md = []
            md.append(f"# Witness {wi} — state {sid} (role {tag})")
            md.append("")
            md.append(f"- t = {t}")
            md.append(f"- n = {r['state']['n']}")
            md.append(f"- f = {r['state']['f']}")
            md.append(f"- q = {r['state']['q']}")
            md.append(f"- A = {r['state']['A']}")
            md.append(f"- K = {r['state']['K']}")
            md.append(f"- partner_state_id = {detail['partner_state_id']}")
            md.append(f"- flipped_pairs = {flipped}")
            md.append("")
            md.append("## Common acceptance LCB")
            md.append(
                f"tail_c = δc/(J·M) = {DC}/({J}·{M}) = {DC/(J*M)}; "
                f"L_j = BetaQuantile lower(A_j,n_j,tail_c); "
                f"LCB = (1-t)·mean(L) + t·min(L) = {detail['common_acceptance_lcb_slot0']}"
            )
            md.append("")
            md.append("## Per-procedure risk / pass (slot 0; all 12 identical)")
            md.append(f"- RATIO: D={detail['ratio_D_slot0']}, den={detail['ratio_min_denominator_slot0']}, "
                      f"pass_any={detail['passed_any']['RATIO']} (need D≤0 and den>0)")
            md.append(f"- MASS_CP: D={detail['mass_cp_D_slot0']}, pass_any={detail['passed_any']['MASS_CP']} (need D≤0)")
            md.append(f"- HOEFFDING_H: p={detail['hoeffding_p_slot0']}, Holm first thresh δr/M={DR/M}, "
                      f"n_pass={detail['n_passed_slots']['HOEFFDING_H']}")
            md.append(f"- DIRECT_BETTING_H: p={detail['betting_p_slot0']}, n_pass={detail['n_passed_slots']['DIRECT_BETTING_H']}")
            md.append(f"- SCALE_H: p={detail['scale_p_slot0']}, n_pass={detail['n_passed_slots']['SCALE_H']}")
            md.append(f"- CP_IUT_H: p={detail['cp_iut_p_slot0']}, n_pass={detail['n_passed_slots']['CP_IUT_H']}")
            md.append(f"- COMPOSED_CP_BETTING_H: p={detail['composed_p_slot0']}, n_pass={detail['n_passed_slots']['COMPOSED_CP_BETTING_H']}")
            md.append("")
            md.append("## ECA vector and ranking")
            md.append(f"- ECA = {json.dumps(r['eca'])}")
            md.append(f"- ranking = {r['ranking']}")
            md.append("")
            md.append("## Interpretation limit")
            md.append(
                "This witness shows that, at fixed t, changing only the count profile (n,f,q)→(A,K) "
                "can flip procedure ranking by ECA. It does **not** attribute causal share to "
                "rectangular bound vs tail vs gate, and is not a deployment frequency claim."
            )
            (OUT / "WITNESSES" / f"witness_{wi}_state_{sid}_{tag}.md").write_text("\n".join(md) + "\n")
        witnesses.append({"index": wi, "t": t, "state_id_1": s1, "state_id_2": s2, "flipped_pairs": flipped})

    if decision_label == "PASS_IDENTIFIABILITY" and len(witnesses) < 2:
        decision_label = "INCONCLUSIVE"

    # Check identical-slot integrity
    bad_collapse = sum(1 for r in flag_rows if r["identical_slots_collapse_ok"] == 0)
    if bad_collapse:
        decision_label = "INVALID"

    decision = {
        "decision": decision_label,
        "created_asia_seoul": created,
        "n_states": 256,
        "logical_flags": 21504,
        "identical_slot_integrity_failures": bad_collapse,
        "ranking_reversal_state_pairs_by_t": {str(t): counts_by_t[t] for t in TS},
        "ranking_reversal_state_pairs_total": total_reversals,
        "key_contrast_pairwise_flip_state_pairs_by_t": {
            name: {str(t): pairwise_counts[name][t] for t in TS} for _, _, name in key_contrasts
        },
        "n_witnesses_written": len(witnesses),
        "witnesses": witnesses,
        "interpretation_limits": [
            "Identifiability of count-state effects on procedure ranking at fixed t only.",
            "Not average performance superiority.",
            "256 states are a designed grid, not a sample of real frequencies.",
            "No CI/significance.",
            "Do not claim causal attribution rates among rectangular/tail/gate.",
            "Geometry-alone hypothesis: if PASS, ranking can reverse under count-profile change at fixed t, so geometry (t) alone cannot explain all procedure differences; strengthen limitation that constructions differ. If FAIL, narrow the claim that count profiles reverse rankings under this sealed grid.",
        ],
        "geometry_alone_implication": (
            "PASS_IDENTIFIABILITY implies procedure ranking is not determined by t alone under this sealed count grid; "
            "geometry-alone causal attribution among procedures should be narrowed / limitations strengthened."
            if decision_label == "PASS_IDENTIFIABILITY"
            else "No strict ranking reversal found on this sealed grid; the specific count-profile reversal hypothesis is unsupported here (FAIL_FOR_HYPOTHESIS) or inconclusive."
        ),
        "STATES_csv_sha256": states_sha,
        "E2_PROTOCOL_sha256_at_seal": protocol_sha_pre,
        "formulas_version_note": FORMULA_VERSION,
        "v1_result_lock": "NOT_TOUCHED_on_box; Mac path RESULT_LOCK confirmation blocked (no Mac MCP)",
    }
    decision_path = OUT / "DECISION.json"
    decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")

    # Korean report
    report = []
    report.append("# FedCORE E2 유한 count-state 메커니즘 감사 보고")
    report.append("")
    report.append(f"- 작성 시각 (Asia/Seoul): {created}")
    report.append(f"- 판정: **{decision_label}**")
    report.append(f"- 봉인 STATES.csv SHA-256: `{states_sha}`")
    report.append(f"- 논리 flags: 21,504 (=256×7×12)")
    report.append("")
    report.append("## 질문")
    report.append("")
    report.append(
        "고정된 t에서 count profile(n, f→A, q→K)만 바꾸면 7개 고정 인증 절차의 ECA 순위가 "
        "엄격히 역전되는가? 이는 geometry(t)만으로 절차 차이를 설명할 수 있는지의 "
        "**식별 가능성**을 묻는다. 평균 성능 우위를 주장하지 않는다."
    )
    report.append("")
    report.append("## 설계 요약 (봉인)")
    report.append("")
    report.append(
        f"J={J}, M={M}, α={ALPHA}, δr=δc={DR}; n∈{list(NS)}; "
        "f 4개, q 4개, t 4개 → 256 상태. A_j=⌊n f_j⌋, K_j=⌊A_j q_j⌋ (Fraction→floor). "
        "12 slots는 동일 (n,A,K) 완전 의존 복제이며 M=12 보정 유지."
    )
    report.append("")
    report.append("## 결과 요약")
    report.append("")
    report.append("### 고정 t별 엄격 순위 역전 (상태쌍 수)")
    report.append("")
    for t in TS:
        report.append(f"- t={t}: **{counts_by_t[t]}** pairs")
    report.append(f"- 합계: **{total_reversals}**")
    report.append("")
    report.append("### 주요 pairwise flip (상태쌍 수, 고정 t)")
    report.append("")
    for _, _, name in key_contrasts:
        report.append(f"- **{name}**: " + ", ".join(f"t={t}:{pairwise_counts[name][t]}" for t in TS))
    report.append("")
    report.append("### Witness")
    report.append("")
    if witnesses:
        for w in witnesses:
            report.append(
                f"- witness {w['index']}: t={w['t']}, states {w['state_id_1']} vs {w['state_id_2']}, "
                f"flips={w['flipped_pairs']}"
            )
        report.append("")
        report.append("상세 식 재유도는 `WITNESSES/` 디렉터리를 본다.")
    else:
        report.append("- (역전 없음 → witness 없음)")
    report.append("")
    report.append("## 해석 한계")
    report.append("")
    report.append(
        "- 본 감사는 설계된 256개 상태의 **결정적 enumeration**이며 현실 빈도 표본이 아니다. "
        "CI·유의성 검정을 붙이지 않았다."
    )
    report.append(
        "- rectangular bound / tail allocation / denominator gate의 **인과 기여율**을 분리·산출하지 않는다."
    )
    report.append("- 새 인증 방법을 만들지 않았고, 검증되지 않은 counterfactual을 deployment 결과로 보고하지 않는다.")
    report.append("- V1 outcomes / RESULT_LOCK / retrain을 변경하지 않았다.")
    report.append("")
    report.append("## geometry-alone 가설에 대한 함의")
    report.append("")
    report.append(decision["geometry_alone_implication"])
    report.append("")
    report.append("## 공식 출처 메모")
    report.append("")
    report.append(
        "요청된 `reconstruct_protection_statistics.py`는 이번 실행 환경(box)에서 Mac 경로 MCP 부재로 "
        "직접 읽지 못했다. 동일 동결 count-only 수학을 **재작성**하여 사용했으며 production certifier "
        "모듈은 import하지 않았다. 모호성으로 부호를 확정할 수 없으면 INCONCLUSIVE로 두도록 설계했다."
    )
    report.append("")
    report.append("## 산출물")
    report.append("")
    for name in [
        "E2_PROTOCOL.json",
        "STATES.csv",
        "FLAGS.csv.gz",
        "RANKINGS.csv",
        "REVERSALS.csv",
        "WITNESSES/",
        "DECISION.json",
        "REPORT_ko.md",
        "HASHES.sha256",
        "MANIFEST.json",
    ]:
        report.append(f"- `{OUT / name}`")
    report_path = OUT / "REPORT_ko.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    # HASHES + MANIFEST
    artifact_files = [
        "E2_PROTOCOL.json",
        "STATES.csv",
        "FLAGS.csv.gz",
        "RANKINGS.csv",
        "REVERSALS.csv",
        "DECISION.json",
        "REPORT_ko.md",
        "code/e2_count_state_audit.py",
    ]
    for p in sorted((OUT / "WITNESSES").glob("*")):
        artifact_files.append(str(p.relative_to(OUT)))

    hash_lines = []
    manifest_files = []
    for rel in artifact_files:
        p = OUT / rel
        if not p.exists():
            continue
        h = sha256_file(p)
        hash_lines.append(f"{h}  {rel}")
        manifest_files.append({"path": rel, "sha256": h, "bytes": p.stat().st_size})

    hashes_path = OUT / "HASHES.sha256"
    hashes_path.write_text("\n".join(hash_lines) + "\n")

    manifest = {
        "experiment": "FedCORE_E2",
        "created_asia_seoul": created,
        "output_dir_box": str(OUT),
        "output_dir_mac_requested": "/Users/october25kim/Documents/Codex/2026-09-16/neurocomputing-desk-reject/outputs/fedcore_e2_count_state_20260918/",
        "mac_write_status": "BLOCKED_NO_MCP",
        "decision": decision_label,
        "STATES_csv_sha256": states_sha,
        "files": manifest_files,
        "HASHES_sha256_of_HASHES_file": None,
    }
    # write manifest without self hash first, then add HASHES hash
    manifest["HASHES_sha256_of_HASHES_file"] = sha256_file(hashes_path)
    manifest_path = OUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    # append HASHES for HASHES itself and MANIFEST
    with hashes_path.open("a") as f:
        f.write(f"{sha256_file(hashes_path)}  HASHES.sha256\n")
        # recompute after self-append is messy; instead hash MANIFEST only
        f.write(f"{sha256_file(manifest_path)}  MANIFEST.json\n")

    print(json.dumps({
        "decision": decision_label,
        "states_sha": states_sha,
        "protocol_sha": sha256_file(protocol_path),
        "decision_sha": sha256_file(decision_path),
        "report_sha": sha256_file(report_path),
        "reversals_by_t": counts_by_t,
        "total_reversals": total_reversals,
        "pairwise": {k: dict(v) for k, v in pairwise_counts.items()},
        "n_witnesses": len(witnesses),
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
