"""Office-Home selector-rescue post-hoc analysis (Phases S1/S2/S3, no new training).

A NO-NEW-TRAINING follow-up to the frozen Office-Home campaign.  The original
PRIMARY result -- client-level FULL SIMPLEX, preregistered coverage-max selector,
0/50 non-vacuous at every alpha -- is preserved byte-identical; this script never
writes ``results/officehome/final_cell_results.csv`` or any primary output.  It
writes only under ``results/officehome/selector_rescue/`` and two docs.

It answers: does any FINITE-SAMPLE-VALID selector policy turn the 0/50 primary
into a positive x/50 at alpha=0.20 -- and if so, does the finite-family
multiplicity cost eat the gain?

Phases (see docs/agent/officehome_selector_rescue_plan.md):

* S1  Simultaneous selector-family Fed-CORE over the M=12 prospectively frozen
      candidates x J=4 clients.  SAME certification draw as the primary
      (frozen ``audit_draw_seed``), union-bounded at ``delta_r/(MJ)`` +
      ``delta_c/(MJ)``; valid certification-data-dependent argmax.  Plus a
      Holm/LTT risk-only variant (IU p-values, FWER at ``delta_r``; coverage
      simultaneous at ``delta_c``).  delta_r = delta_c = 0.05.
* S2  Fresh-draw most-buffered (gamma=0.3) SINGLE pre-frozen policy + a NEW
      independent cert draw from the SAME frozen reservoirs (seed a function of
      the cell name only, distinct from the audit/traffic streams).  Full-simplex
      certificate at ``delta_r/J`` + ``delta_c/J`` (no multiplicity).
* S3  Canonical Theorem-2 budget recheck of the SECONDARY traffic-Lambda + rho
      certificates only: risk-side ``eps_r = delta_r/(3J)`` over
      {rbar, alow-denominator, ahigh}; coverage ``eps_c = delta_c/J`` on a
      SEPARATE lower acceptance bound.  delta_total 0.10 / delta_lambda 0.02 /
      delta_r 0.04 / delta_c 0.04.  Verifies the 0/50 secondary conclusion.

Every certificate reuses the exact fedcore CP core (``cp_upper`` / ``cp_lower``
via ``fedcore.officehome_rescue`` and ``fedcore.certificate.joint``); no CP is
reimplemented, and no favourable seed/alpha/score/cell is selected.

Entry point: ``python -m fedcore.experiments.officehome_selector_rescue``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.certificate.joint import joint_conditional_certificate
from fedcore.mixture import rho_mixture_box
from fedcore.officehome_selector import select_from_artifact
from fedcore.officehome_traffic_lambda import build_traffic_lambda, draw_traffic_client_counts
from fedcore.scores import compute_score
from fedcore.selector import choose_threshold, open_set_error

from fedcore.officehome_rescue import (
    FAMILY_GAMMAS,
    FAMILY_NGRID,
    FAMILY_SCORES,
    M_CANDIDATES,
    CandidateKey,
    family_keys,
    fresh_draw_seed,
    holm_family_certificate,
    select_family_candidate,
    selector_definition_hash,
    simultaneous_family_certificate,
)

# Reuse the EXACT primary loading / draw / certificate machinery so S1 shares the
# primary's certification draw and full-simplex certificate byte-for-byte.
from fedcore.experiments.officehome_final_analysis import (
    ALPHAS,
    ALPHA_PRIMARY,
    CHECKSUMS,
    DELTA_C,
    DELTA_R,
    DOCDIR,
    J,
    M_GRID,
    M_PRIMARY,
    NJ_GRID,
    NJ_MAX,
    NJ_PRIMARY,
    OUTDIR,
    RHO_GRID,
    T_DELTA_C,
    T_DELTA_LAMBDA,
    T_DELTA_R,
    Cell,
    DrawnCounts,
    certify_full_simplex,
    counts_from_draw,
    eval_client_simplex_risk,
    fold_overlap_row,
    full_simplex_budget,
    git_commit,
    load_cells,
    load_frozen_checksums,
    write_csv,
    _sha256,
)
from fedcore.certificate.allocation import zero_error_floor

RESCUE_DIR = os.path.join(OUTDIR, "selector_rescue")


def _j(value) -> str:
    return json.dumps(value, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Fresh-draw positions (Phase S2): same structure as the primary draw, but a
# cell-name-only seed distinct from the audit / traffic streams.
# --------------------------------------------------------------------------- #
def draw_positions_seeded(cell: Cell, seed: int) -> Dict[int, np.ndarray]:
    """``NJ_MAX`` with-replacement reservoir positions per domain, given ``seed``."""
    cert_client = cell.arrays["cert_client"]
    rng = np.random.default_rng(int(seed))
    draws: Dict[int, np.ndarray] = {}
    for j in range(J):
        res = np.where(cert_client == j)[0]
        if res.size == 0:
            raise RuntimeError(f"FAIL CLOSED: empty cert reservoir domain {j} {cell.name}")
        draws[j] = res[rng.integers(0, res.size, size=NJ_MAX)]
    return draws


# --------------------------------------------------------------------------- #
# Candidate family recomputation (proposal fold only) -> per-candidate counts
# --------------------------------------------------------------------------- #
@dataclass
class FamilyCounts:
    keys: List[CandidateKey]
    A: np.ndarray              # (M, J)
    K: np.ndarray              # (M, J)
    n: np.ndarray              # (J,)
    thresholds: np.ndarray     # (M,)
    scores: List[str]
    gammas: List[float]
    prop_feasible: np.ndarray  # (M,) threshold feasibility (proposal fold only)


def family_counts_for_cell(
    cell: Cell, draws: Dict[int, np.ndarray], alpha: float, n_j: int
) -> FamilyCounts:
    """Recompute the M=12 candidate family from the PROPOSAL fold and evaluate it
    on the given draw (never reads certification/eval labels for the selector)."""
    policy, cands = select_from_artifact(
        cell.arrays, alpha=alpha, scores=FAMILY_SCORES, gammas=FAMILY_GAMMAS
    )
    keys = family_keys(alpha)
    # cands are in scores x gammas order == family_keys order; assert alignment.
    if len(cands) != M_CANDIDATES:
        raise RuntimeError(f"expected {M_CANDIDATES} candidates, got {len(cands)}")
    A = np.zeros((M_CANDIDATES, J), dtype=int)
    K = np.zeros((M_CANDIDATES, J), dtype=int)
    thr = np.zeros(M_CANDIDATES, dtype=float)
    feas = np.zeros(M_CANDIDATES, dtype=bool)
    scores: List[str] = []
    gammas: List[float] = []
    for m, cand in enumerate(cands):
        assert cand.score_name == keys[m].score_name and float(cand.gamma) == keys[m].gamma
        dc = counts_from_draw(
            cell, draws, n_j, cand.score_name, cand.threshold, bool(cand.threshold_feasible)
        )
        A[m] = dc.A
        K[m] = dc.K
        thr[m] = cand.threshold
        feas[m] = bool(cand.threshold_feasible)
        scores.append(cand.score_name)
        gammas.append(float(cand.gamma))
    n = np.full(J, n_j, dtype=int)
    return FamilyCounts(keys, A, K, n, thr, scores, gammas, feas)


# --------------------------------------------------------------------------- #
# Single-selector (no-multiplicity) full-simplex certificate for a count row.
# Used ONLY as a diagnostic to isolate multiplicity cost vs model-family barrier.
# --------------------------------------------------------------------------- #
def single_selector_full_simplex(
    A_row: Sequence[int], K_row: Sequence[int], n: Sequence[int], alpha: float
) -> Tuple[float, float, bool]:
    """(U, C, certified) at ``delta_r/J`` + ``delta_c/J`` -- what a hypothetically
    pre-registered SINGLE selector would obtain (no family multiplicity)."""
    A_row = np.asarray(A_row, int)
    K_row = np.asarray(K_row, int)
    n = np.asarray(n, int)
    eps_r = DELTA_R / J
    eps_c = DELTA_C / J
    rbar = np.array([cp_upper(int(K_row[j]), int(A_row[j]), eps_r) for j in range(J)])
    alow = np.array([cp_lower(int(A_row[j]), int(n[j]), eps_c) for j in range(J)])
    U = float(rbar.max())
    C = float(alow.min())
    certified = bool(U <= alpha and C > 0.0 and np.any(A_row > 0))
    return U, C, certified


# --------------------------------------------------------------------------- #
# Canonical Theorem-2 box certificate (Phase S3): risk eps = delta_r/(3J) over
# {rbar, alow, ahigh}; coverage eps = delta_c/J on a SEPARATE lower bound.
# --------------------------------------------------------------------------- #
def certify_box_canonical(
    dc: DrawnCounts,
    alpha: float,
    lam_lower: Sequence[float],
    lam_upper: Sequence[float],
    *,
    delta_r: float,
    delta_c: float,
):
    """Return (risk_ucb, coverage_lcb, certified, feasible). Reuses the exact core
    twice: once for the risk ratio (denominator alow at delta_r/(3J)) and once for
    the separately-budgeted coverage LCB (alow at delta_c/J)."""
    risk_eps = np.full(J, delta_r / (3.0 * J))
    lower_eps_risk = np.full(J, delta_r / (3.0 * J))
    upper_eps = np.full(J, delta_r / (3.0 * J))
    lower_eps_cov = np.full(J, delta_c / J)

    cert_risk = joint_conditional_certificate(
        dc.A, dc.K, dc.n, alpha=alpha,
        risk_eps=risk_eps, acceptance_lower_eps=lower_eps_risk,
        acceptance_upper_eps=upper_eps,
        lambda_lower=lam_lower, lambda_upper=lam_upper,
    )
    cert_cov = joint_conditional_certificate(
        dc.A, dc.K, dc.n, alpha=alpha,
        risk_eps=risk_eps, acceptance_lower_eps=lower_eps_cov,
        acceptance_upper_eps=upper_eps,
        lambda_lower=lam_lower, lambda_upper=lam_upper,
    )
    risk_ucb = float(cert_risk.risk_ucb)
    coverage_lcb = float(cert_cov.coverage_lcb)
    feasible = bool(cert_risk.feasible)
    certified = bool(feasible and math.isfinite(risk_ucb) and risk_ucb <= alpha and coverage_lcb > 0.0)
    return risk_ucb, coverage_lcb, certified, feasible


# --------------------------------------------------------------------------- #
# Original (as-run) box certificate reused for the S3 diff (delta_r split over
# 2 events, alow shared as denominator + coverage). Mirrors the primary run.
# --------------------------------------------------------------------------- #
def certify_box_original(
    dc: DrawnCounts,
    alpha: float,
    lam_lower: Sequence[float],
    lam_upper: Sequence[float],
    *,
    delta_r: float,
    delta_c: float,
):
    risk_eps = np.full(J, (delta_r / 2.0) / J)      # rbar
    upper_eps = np.full(J, (delta_r / 2.0) / J)     # ahigh
    lower_eps = np.full(J, delta_c / J)             # alow (shared)
    cert = joint_conditional_certificate(
        dc.A, dc.K, dc.n, alpha=alpha,
        risk_eps=risk_eps, acceptance_lower_eps=lower_eps,
        acceptance_upper_eps=upper_eps,
        lambda_lower=lam_lower, lambda_upper=lam_upper,
    )
    return float(cert.risk_ucb), float(cert.coverage_lcb), bool(cert.certified), bool(cert.feasible)


# --------------------------------------------------------------------------- #
# Primary certificate (recomputed in-memory, never written) for pairing.
# --------------------------------------------------------------------------- #
@dataclass
class PrimaryCell:
    policy: object
    dc: DrawnCounts
    cert: object
    certified: bool
    support: bool


def primary_for_cell(cell: Cell, draws: Dict[int, np.ndarray], alpha: float) -> PrimaryCell:
    policy, _ = select_from_artifact(cell.arrays, alpha=alpha, scores=FAMILY_SCORES, gammas=FAMILY_GAMMAS)
    support = bool(policy.feasible)
    dc = counts_from_draw(
        cell, draws, NJ_PRIMARY, policy.score_name if support else "msp", policy.threshold, support
    )
    cert = certify_full_simplex(dc, alpha, full_simplex_budget())
    certified = bool(support and cert.certified)
    return PrimaryCell(policy=policy, dc=dc, cert=cert, certified=certified, support=support)


# --------------------------------------------------------------------------- #
# S2 selector: most-buffered gamma=0.3, max-min per-client acceptance rate
# --------------------------------------------------------------------------- #
@dataclass
class S2Policy:
    feasible: bool
    score_name: str
    gamma: float
    threshold: float
    min_client_accept_rate: float
    max_client_risk: float
    selector_hash: str
    per_client_accept_rate: List[float]
    per_client_risk: List[float]


def s2_select_policy(cell: Cell, alpha: float, gamma: float = 0.3) -> S2Policy:
    """Proposal-only: one threshold per score at gamma=0.3; pick the score with the
    LARGEST minimum per-client proposal acceptance rate. Ties: lower max per-client
    proposal risk, lexicographic score, selector hash. Fail closed if none feasible."""
    logits = cell.arrays["prop_logits"]
    y = cell.arrays["prop_y_open"]
    cl = cell.arrays["prop_client"]
    pred = logits.argmax(axis=-1)
    err = open_set_error(pred, y)
    cand_rows = []
    for score_name in FAMILY_SCORES:
        sv = compute_score(score_name, logits)
        sel = choose_threshold(sv, pred, y, float(gamma), float(alpha), n_grid=FAMILY_NGRID)
        if not sel.feasible:
            continue
        acc = sv >= sel.threshold
        rates = []
        risks = []
        for jj in range(J):
            m = cl == jj
            nj = int(m.sum())
            acc_j = m & acc
            aj = int(acc_j.sum())
            rates.append(aj / nj if nj > 0 else 0.0)
            risks.append(float(err[acc_j].mean()) if aj > 0 else 0.0)
        cand_rows.append(
            {
                "score_name": score_name,
                "gamma": float(gamma),
                "threshold": float(sel.threshold),
                "min_rate": float(min(rates)),
                "max_risk": float(max(risks)),
                "hash": selector_definition_hash(score_name, gamma, alpha),
                "rates": rates,
                "risks": risks,
            }
        )
    if not cand_rows:
        return S2Policy(False, "", float(gamma), float("inf"), 0.0, 0.0, "", [0.0] * J, [0.0] * J)
    best = min(cand_rows, key=lambda c: (-c["min_rate"], c["max_risk"], c["score_name"], c["hash"]))
    return S2Policy(
        True, best["score_name"], best["gamma"], best["threshold"],
        best["min_rate"], best["max_risk"], best["hash"], best["rates"], best["risks"],
    )


# --------------------------------------------------------------------------- #
# PHASE S1
# --------------------------------------------------------------------------- #
def run_s1(cells: List[Cell]) -> Dict[str, List[dict]]:
    sim_rows: List[dict] = []
    holm_rows: List[dict] = []
    selected_rows: List[dict] = []
    paired_rows: List[dict] = []

    count_floor_mult = {a: zero_error_floor(DELTA_R / (M_CANDIDATES * J), a) for a in ALPHAS}
    count_floor_single = {a: zero_error_floor(DELTA_R / J, a) for a in ALPHAS}

    for cell in cells:
        draws = {j: p for j, p in _cached_draws(cell).items()}
        for alpha in ALPHAS:
            prim = primary_for_cell(cell, draws, alpha)
            fc = family_counts_for_cell(cell, draws, alpha, NJ_PRIMARY)
            sim = simultaneous_family_certificate(
                fc.A, fc.K, fc.n, alpha=alpha, delta_r=DELTA_R, delta_c=DELTA_C
            )
            sel_idx = select_family_candidate(fc.keys, sim.certified, sim.C)

            # single-selector (no-multiplicity) diagnostic per candidate
            U_single = np.zeros(M_CANDIDATES)
            C_single = np.zeros(M_CANDIDATES)
            cert_single = np.zeros(M_CANDIDATES, dtype=bool)
            for m in range(M_CANDIDATES):
                U_single[m], C_single[m], cert_single[m] = single_selector_full_simplex(
                    fc.A[m], fc.K[m], fc.n, alpha
                )
            any_single = bool(cert_single.any())
            n_single = int(cert_single.sum())

            family_certified = sel_idx is not None
            sel_score = fc.scores[sel_idx] if sel_idx is not None else ""
            sel_gamma = fc.gammas[sel_idx] if sel_idx is not None else float("nan")
            sel_hash = fc.keys[sel_idx].selector_hash if sel_idx is not None else ""
            sel_U = float(sim.U[sel_idx]) if sel_idx is not None else float("nan")
            sel_C = float(sim.C[sel_idx]) if sel_idx is not None else float("nan")
            sel_U_single = float(U_single[sel_idx]) if sel_idx is not None else float("nan")
            mult_cost = (sel_U - sel_U_single) if sel_idx is not None else float("nan")

            # held-out diagnostic (NOT used for selection): realized eval risk of
            # the selected candidate's operating point.
            heldout = float("nan")
            if sel_idx is not None:
                csr, _r, _ae, _pooled = eval_client_simplex_risk(
                    cell, sel_score, float(fc.thresholds[sel_idx]), True
                )
                heldout = csr

            sim_rows.append({
                "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                "train_rep": cell.train_rep, "alpha": alpha, "n_j": NJ_PRIMARY,
                "M": M_CANDIDATES, "Jclients": J, "delta_r": DELTA_R, "delta_c": DELTA_C,
                "eps_r_multiplicity": sim.eps_r, "eps_c_multiplicity": sim.eps_c,
                "count_floor_multiplicity": count_floor_mult[alpha],
                "count_floor_single": count_floor_single[alpha],
                "n_candidates_certified": int(sim.certified.sum()),
                "family_certified": family_certified,
                "selected_score": sel_score, "selected_gamma": sel_gamma,
                "selected_hash": sel_hash,
                "selected_risk_ucb": sel_U, "selected_coverage_lcb": sel_C,
                "EffectiveCertCov": sel_C if family_certified else 0.0,
                "CondCertCov": sel_C if family_certified else float("nan"),
                "selected_risk_ucb_single_budget": sel_U_single,
                "multiplicity_cost_ucb": mult_cost,
                "any_single_selector_certifies_diag": any_single,
                "n_single_selector_certify_diag": n_single,
                "heldout_eval_worst_client_risk_diag": heldout,
                "primary_certified": prim.certified,
                "primary_risk_ucb": float(prim.cert.risk_ucb),
                "primary_coverage_lcb": float(prim.cert.coverage_lcb),
                "primary_score": prim.policy.score_name, "primary_gamma": prim.policy.gamma,
                "cand_scores": _j(fc.scores), "cand_gammas": _j(fc.gammas),
                "cand_U": _j([float(x) for x in sim.U]),
                "cand_C": _j([float(x) for x in sim.C]),
                "cand_certified": _j([bool(x) for x in sim.certified]),
                "cand_U_single_diag": _j([float(x) for x in U_single]),
                "cand_certified_single_diag": _j([bool(x) for x in cert_single]),
                "cand_A": _j([[int(v) for v in row] for row in fc.A]),
                "cand_K": _j([[int(v) for v in row] for row in fc.K]),
            })

            # ---- Holm variant ----
            holm = holm_family_certificate(
                fc.A, fc.K, fc.n, alpha=alpha, delta_r=DELTA_R, delta_c=DELTA_C
            )
            holm_idx = select_family_candidate(fc.keys, holm.certified, holm.C)
            holm_certified = holm_idx is not None
            hsel_score = fc.scores[holm_idx] if holm_idx is not None else ""
            hsel_gamma = fc.gammas[holm_idx] if holm_idx is not None else float("nan")
            hsel_C = float(holm.C[holm_idx]) if holm_idx is not None else float("nan")
            holm_rows.append({
                "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                "train_rep": cell.train_rep, "alpha": alpha, "n_j": NJ_PRIMARY,
                "M": M_CANDIDATES, "Jclients": J, "fwer_delta_r": DELTA_R, "delta_c": DELTA_C,
                "eps_c_multiplicity": holm.eps_c,
                "n_candidates_holm_reject": int(holm.holm_reject.sum()),
                "n_candidates_certified": int(holm.certified.sum()),
                "holm_certified": holm_certified,
                "selected_score": hsel_score, "selected_gamma": hsel_gamma,
                "selected_coverage_lcb": hsel_C,
                "EffectiveCertCov": hsel_C if holm_certified else 0.0,
                "primary_certified": prim.certified,
                "cand_pvalues": _j([float(x) for x in holm.pvalues]),
                "cand_holm_reject": _j([bool(x) for x in holm.holm_reject]),
                "cand_C": _j([float(x) for x in holm.C]),
                "cand_certified": _j([bool(x) for x in holm.certified]),
            })

            for method, cert_flag, ssc, sgm, shs, scov in (
                ("simultaneous", family_certified, sel_score, sel_gamma, sel_hash, sel_C),
                ("holm", holm_certified, hsel_score, hsel_gamma,
                 fc.keys[holm_idx].selector_hash if holm_idx is not None else "", hsel_C),
            ):
                selected_rows.append({
                    "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                    "train_rep": cell.train_rep, "alpha": alpha, "method": method,
                    "certified": cert_flag,
                    "selected_score": ssc if cert_flag else "NONE",
                    "selected_gamma": sgm if cert_flag else float("nan"),
                    "selected_hash": shs if cert_flag else "",
                    "selected_coverage_lcb": scov,
                })

            paired_rows.append({
                "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                "train_rep": cell.train_rep, "alpha": alpha,
                "primary_certified": prim.certified,
                "primary_risk_ucb": float(prim.cert.risk_ucb),
                "family_simultaneous_certified": family_certified,
                "family_simultaneous_risk_ucb": sel_U,
                "holm_certified": holm_certified,
                "any_single_selector_certifies_diag": any_single,
                "transition_primary_to_simultaneous": f"{int(prim.certified)}->{int(family_certified)}",
                "transition_primary_to_holm": f"{int(prim.certified)}->{int(holm_certified)}",
            })

    return {
        "simultaneous_family_results.csv": sim_rows,
        "holm_family_results.csv": holm_rows,
        "family_selected_candidates.csv": selected_rows,
        "family_vs_primary_paired.csv": paired_rows,
    }


# small per-cell draw cache so S1 and the taxonomy reuse identical positions
_DRAW_CACHE: Dict[str, Dict[int, np.ndarray]] = {}


def _cached_draws(cell: Cell) -> Dict[int, np.ndarray]:
    if cell.name not in _DRAW_CACHE:
        from fedcore.experiments.officehome_final_analysis import draw_positions
        _DRAW_CACHE[cell.name] = draw_positions(cell)
    return _DRAW_CACHE[cell.name]


# --------------------------------------------------------------------------- #
# PHASE S2
# --------------------------------------------------------------------------- #
def run_s2(cells: List[Cell]) -> Dict[str, List[dict]]:
    manifest_rows: List[dict] = []
    result_rows: List[dict] = []
    reservoir_rows: List[dict] = []
    paired_rows: List[dict] = []

    for cell in cells:
        prim_draws = _cached_draws(cell)
        seed = fresh_draw_seed(cell.name)
        # freshness / distinctness assertions (fail closed)
        distinct_audit = int(seed) != int(cell.audit_draw_seed)
        distinct_traffic = int(seed) != int(cell.traffic_draw_seed)
        if not (distinct_audit and distinct_traffic):
            raise RuntimeError(f"FAIL CLOSED: fresh seed collides with a frozen stream for {cell.name}")
        fresh_draws = draw_positions_seeded(cell, seed)
        dnames = [str(x) for x in cell.arrays["domains"]]

        for alpha in ALPHAS:
            pol = s2_select_policy(cell, alpha, gamma=0.3)
            prim = primary_for_cell(cell, prim_draws, alpha)
            manifest_rows.append({
                "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                "train_rep": cell.train_rep, "alpha": alpha, "gamma": 0.3,
                "fresh_draw_seed": int(seed),
                "audit_draw_seed": int(cell.audit_draw_seed),
                "traffic_draw_seed": int(cell.traffic_draw_seed),
                "seed_distinct_from_audit": distinct_audit,
                "seed_distinct_from_traffic": distinct_traffic,
                "seed_rule": "blake2b8(officehome-fresh-draw-v1|cell_name) mod 2**32",
                "policy_feasible": pol.feasible,
                "selected_score": pol.score_name if pol.feasible else "NONE",
                "selected_gamma": pol.gamma, "threshold": pol.threshold,
                "selected_hash": pol.selector_hash,
                "min_client_prop_accept_rate": pol.min_client_accept_rate,
                "max_client_prop_risk": pol.max_client_risk,
                "per_client_prop_accept_rate": _j([float(x) for x in pol.per_client_accept_rate]),
                "per_client_prop_risk": _j([float(x) for x in pol.per_client_risk]),
            })
            for n_j in NJ_GRID:
                if pol.feasible:
                    dc = counts_from_draw(cell, fresh_draws, n_j, pol.score_name, pol.threshold, True)
                    cert = certify_full_simplex(dc, alpha, full_simplex_budget())
                    certified = bool(cert.certified)
                    ucb = float(cert.risk_ucb)
                    cov = float(cert.coverage_lcb)
                    A_list = dc.A.tolist(); K_list = dc.K.tolist()
                else:
                    certified = False; ucb = float("inf"); cov = 0.0
                    A_list = [0] * J; K_list = [0] * J
                result_rows.append({
                    "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                    "train_rep": cell.train_rep, "alpha": alpha, "n_j": n_j,
                    "primary_n_j": bool(n_j == NJ_PRIMARY),
                    "policy_feasible": pol.feasible,
                    "selected_score": pol.score_name if pol.feasible else "NONE",
                    "selected_gamma": pol.gamma, "threshold": pol.threshold,
                    "fresh_draw_seed": int(seed),
                    "certified": certified, "cert_risk_ucb": ucb, "cert_coverage_lcb": cov,
                    "EffectiveCertCov": cov if certified else 0.0,
                    "count_floor": zero_error_floor(DELTA_R / J, alpha),
                    "per_client_A": _j(A_list), "per_client_K": _j(K_list),
                    "per_client_n": _j([n_j] * J),
                })
                if n_j == NJ_PRIMARY:
                    paired_rows.append({
                        "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                        "train_rep": cell.train_rep, "alpha": alpha,
                        "primary_certified": prim.certified,
                        "primary_risk_ucb": float(prim.cert.risk_ucb),
                        "fresh_certified": certified, "fresh_risk_ucb": ucb,
                        "transition_primary_to_fresh": f"{int(prim.certified)}->{int(certified)}",
                    })

        # reservoir accounting for the fresh draw (n_j=500), per domain
        cert_client = cell.arrays["cert_client"]
        overlap_total = fold_overlap_row(cell)["overlap_total"]
        for jj in range(J):
            pos = fresh_draws[jj][:NJ_PRIMARY]
            uniq = int(np.unique(pos).size)
            reservoir_rows.append({
                "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                "train_rep": cell.train_rep, "domain": jj, "domain_name": dnames[jj],
                "reservoir_size": int(np.sum(cert_client == jj)),
                "nominal_draw_count": NJ_PRIMARY, "unique_draws": uniq,
                "duplication_rate": float(1.0 - uniq / NJ_PRIMARY),
                "fresh_draw_seed": int(seed),
                "distinct_from_audit_seed": distinct_audit,
                "all_positions_in_domain_reservoir": bool(np.all(cert_client[pos] == jj)),
                "fold_overlap_total": int(overlap_total),
            })

    return {
        "fresh_draw_manifest.csv": manifest_rows,
        "fresh_draw_results.csv": result_rows,
        "fresh_draw_reservoir_accounting.csv": reservoir_rows,
        "fresh_draw_vs_primary_paired.csv": paired_rows,
    }


# --------------------------------------------------------------------------- #
# PHASE S3
# --------------------------------------------------------------------------- #
def run_s3(cells: List[Cell]) -> Dict[str, List[dict]]:
    lambda_rows: List[dict] = []
    rho_rows: List[dict] = []
    diff_rows: List[dict] = []

    for cell in cells:
        draws = _cached_draws(cell)
        for alpha in ALPHAS:
            prim = primary_for_cell(cell, draws, alpha)
            support = prim.support
            dc = prim.dc  # primary policy + primary n_j=500 draw (same as original secondary)

            # ---- traffic-Lambda ----
            for m in M_GRID:
                counts = draw_traffic_client_counts(
                    cell.arrays["traffic_client"], m, seed=cell.traffic_draw_seed, n_clients=J
                )
                tl = build_traffic_lambda(counts, delta_lambda=T_DELTA_LAMBDA)
                lo = tl.box.mixture.lower; hi = tl.box.mixture.upper
                if support:
                    can_u, can_c, can_cert, can_feas = certify_box_canonical(
                        dc, alpha, lo, hi, delta_r=T_DELTA_R, delta_c=T_DELTA_C
                    )
                    orig_u, orig_c, orig_cert, _ = certify_box_original(
                        dc, alpha, lo, hi, delta_r=T_DELTA_R, delta_c=T_DELTA_C
                    )
                else:
                    can_u = orig_u = float("inf"); can_c = orig_c = 0.0
                    can_cert = orig_cert = False; can_feas = False
                lambda_rows.append({
                    "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                    "train_rep": cell.train_rep, "alpha": alpha, "m": m,
                    "primary_m": bool(m == M_PRIMARY), "proposal_support": support,
                    "convention": "canonical_theorem2",
                    "delta_lambda": T_DELTA_LAMBDA, "delta_r": T_DELTA_R, "delta_c": T_DELTA_C,
                    "eps_r_per_client": T_DELTA_R / (3.0 * J), "eps_c_per_client": T_DELTA_C / J,
                    "canonical_risk_ucb": can_u, "canonical_coverage_lcb": can_c,
                    "canonical_certified": can_cert, "canonical_feasible": can_feas,
                    "original_risk_ucb": orig_u, "original_coverage_lcb": orig_c,
                    "original_certified": orig_cert,
                })
                if m == M_PRIMARY:
                    diff_rows.append(_diff_row(cell, alpha, "traffic", None, m,
                                               can_u, can_c, can_cert, orig_u, orig_c, orig_cert))

            # ---- fixed-rho ----
            center = np.full(J, 1.0 / J)
            for rho in RHO_GRID:
                box = rho_mixture_box(center, rho)
                if support:
                    can_u, can_c, can_cert, can_feas = certify_box_canonical(
                        dc, alpha, box.lower, box.upper, delta_r=T_DELTA_R, delta_c=T_DELTA_C
                    )
                    orig_u, orig_c, orig_cert, _ = certify_box_original(
                        dc, alpha, box.lower, box.upper, delta_r=T_DELTA_R, delta_c=T_DELTA_C
                    )
                else:
                    can_u = orig_u = float("inf"); can_c = orig_c = 0.0
                    can_cert = orig_cert = False; can_feas = False
                rho_rows.append({
                    "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                    "train_rep": cell.train_rep, "alpha": alpha, "rho": rho,
                    "proposal_support": support, "convention": "canonical_theorem2",
                    "delta_r": T_DELTA_R, "delta_c": T_DELTA_C,
                    "eps_r_per_client": T_DELTA_R / (3.0 * J), "eps_c_per_client": T_DELTA_C / J,
                    "canonical_risk_ucb": can_u, "canonical_coverage_lcb": can_c,
                    "canonical_certified": can_cert, "canonical_feasible": can_feas,
                    "original_risk_ucb": orig_u, "original_coverage_lcb": orig_c,
                    "original_certified": orig_cert,
                })
                diff_rows.append(_diff_row(cell, alpha, "rho", rho, None,
                                           can_u, can_c, can_cert, orig_u, orig_c, orig_cert))

    return {
        "canonical_budget_lambda_results.csv": lambda_rows,
        "canonical_budget_rho_results.csv": rho_rows,
        "budget_convention_diff.csv": diff_rows,
    }


def _diff_row(cell, alpha, kind, rho, m, cu, cc, ccert, ou, oc, ocert):
    du = (cu - ou) if math.isfinite(cu) and math.isfinite(ou) else float("nan")
    return {
        "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
        "train_rep": cell.train_rep, "alpha": alpha, "kind": kind,
        "rho": rho if rho is not None else float("nan"),
        "m": m if m is not None else -1,
        "canonical_risk_ucb": cu, "original_risk_ucb": ou, "risk_ucb_diff_canonical_minus_original": du,
        "canonical_coverage_lcb": cc, "original_coverage_lcb": oc,
        "canonical_certified": ccert, "original_certified": ocert,
        "certified_changed": bool(ccert != ocert),
    }


# --------------------------------------------------------------------------- #
# FAILURE TAXONOMY
# --------------------------------------------------------------------------- #
def failure_transition_matrix(
    cells: List[Cell], s1: Dict[str, List[dict]], s2: Dict[str, List[dict]]
) -> List[dict]:
    """Classify each (cell, alpha) barrier and record every method transition."""
    sim_by = {(r["cell"], r["alpha"]): r for r in s1["simultaneous_family_results.csv"]}
    holm_by = {(r["cell"], r["alpha"]): r for r in s1["holm_family_results.csv"]}
    fresh_by = {(r["cell"], r["alpha"]): r for r in s2["fresh_draw_vs_primary_paired.csv"]}
    rows: List[dict] = []
    for cell in cells:
        for alpha in ALPHAS:
            sr = sim_by[(cell.name, alpha)]
            hr = holm_by[(cell.name, alpha)]
            fr = fresh_by[(cell.name, alpha)]
            primary_cert = bool(sr["primary_certified"])
            fam_cert = bool(sr["family_certified"])
            holm_cert = bool(hr["holm_certified"])
            fresh_cert = bool(fr["fresh_certified"])
            any_single = bool(sr["any_single_selector_certifies_diag"])

            cand_A = json.loads(sr["cand_A"])
            cand_certified_single = json.loads(sr["cand_certified_single_diag"])
            cand_U_single = json.loads(sr["cand_U_single_diag"])
            count_floor_single = int(sr["count_floor_single"])

            # barrier classification for the FAMILY extension
            if fam_cert or holm_cert:
                barrier = "certified"
            elif any_single:
                # some single selector would certify at delta/J, but the finite-
                # family multiplicity union bound removes it.
                barrier = "finite-family-selection-multiplicity"
            else:
                # no single selector in the family certifies even without
                # multiplicity -> a model-family limitation. Sub-classify by the
                # best (lowest-U_single) candidate that has full support.
                best_m = int(np.argmin(cand_U_single))
                A_best = cand_A[best_m]
                if all(a == 0 for a in A_best):
                    barrier = "zero-acceptance"
                elif any(a == 0 for a in A_best):
                    barrier = "zero-acceptance"
                elif min(A_best) < count_floor_single:
                    barrier = "count"
                else:
                    # support ok, counts ok -> the operating point's empirical risk
                    # is the barrier; it is common to the whole model family.
                    barrier = "model-family-empirical-risk"

            rows.append({
                "cell": cell.name, "pipeline": cell.pipeline, "split_id": cell.split_id,
                "train_rep": cell.train_rep, "alpha": alpha,
                "primary_certified": primary_cert,
                "family_simultaneous_certified": fam_cert,
                "holm_certified": holm_cert,
                "fresh_draw_certified": fresh_cert,
                "any_single_selector_certifies_diag": any_single,
                "n_single_selector_certify_diag": int(sr["n_single_selector_certify_diag"]),
                "barrier_class": barrier,
                "min_risk_ucb_multiplicity": float(min(json.loads(sr["cand_U"]))),
                "min_risk_ucb_single_diag": float(min(cand_U_single)),
                "count_floor_multiplicity": int(sr["count_floor_multiplicity"]),
                "count_floor_single": count_floor_single,
                "transition_primary_to_simultaneous": f"{int(primary_cert)}->{int(fam_cert)}",
                "transition_primary_to_holm": f"{int(primary_cert)}->{int(holm_cert)}",
                "transition_primary_to_fresh": f"{int(primary_cert)}->{int(fresh_cert)}",
            })
    return rows


# --------------------------------------------------------------------------- #
# CSV field ordering (stable)
# --------------------------------------------------------------------------- #
def _fields(rows: List[dict]) -> List[str]:
    return list(rows[0].keys()) if rows else []


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main() -> None:
    frozen = load_frozen_checksums()
    cells = load_cells(frozen)
    print(f"[integrity] 50/50 frozen logit sha256 verified; numpy {np.__version__}")
    if len(cells) != 50:
        raise RuntimeError(f"expected 50 cells, got {len(cells)}")

    os.makedirs(RESCUE_DIR, exist_ok=True)

    print("[S1] simultaneous selector-family Fed-CORE + Holm ...")
    s1 = run_s1(cells)
    print("[S2] fresh-draw most-buffered (gamma=0.3) ...")
    s2 = run_s2(cells)
    print("[S3] canonical Theorem-2 secondary budget recheck ...")
    s3 = run_s3(cells)
    print("[taxonomy] failure transition matrix ...")
    taxonomy = failure_transition_matrix(cells, s1, s2)

    outputs: Dict[str, List[dict]] = {}
    outputs.update(s1)
    outputs.update(s2)
    outputs.update(s3)
    outputs["failure_transition_matrix.csv"] = taxonomy

    for name, rows in outputs.items():
        write_csv(os.path.join(RESCUE_DIR, name), rows, _fields(rows))

    # manifest + checksums (rescue only; primary manifest untouched)
    check = {name: _sha256(os.path.join(RESCUE_DIR, name)) for name in outputs}
    manifest = {
        "artifact_type": "fedcore.officehome.selector_rescue",
        "generated_by": "fedcore.experiments.officehome_selector_rescue",
        "git_commit": git_commit(),
        "numpy_version": np.__version__,
        "n_cells": len(cells),
        "M_candidates": M_CANDIDATES, "J": J,
        "alphas": list(ALPHAS), "alpha_primary": ALPHA_PRIMARY,
        "s1_budget": {"delta_r": DELTA_R, "delta_c": DELTA_C,
                      "eps_r": DELTA_R / (M_CANDIDATES * J), "eps_c": DELTA_C / (M_CANDIDATES * J),
                      "note": "simultaneous union bound over M*J events"},
        "holm_budget": {"fwer_delta_r": DELTA_R, "coverage_delta_c": DELTA_C,
                        "coverage_eps": DELTA_C / (M_CANDIDATES * J)},
        "s2_budget": {"delta_r": DELTA_R, "delta_c": DELTA_C, "gamma": 0.3,
                      "seed_rule": "blake2b8(officehome-fresh-draw-v1|cell_name) mod 2**32",
                      "note": "single pre-frozen policy; no multiplicity"},
        "s3_budget": {"delta_total": T_DELTA_LAMBDA + T_DELTA_R + T_DELTA_C,
                      "delta_lambda": T_DELTA_LAMBDA, "delta_r": T_DELTA_R, "delta_c": T_DELTA_C,
                      "eps_r": T_DELTA_R / (3.0 * J), "eps_c": T_DELTA_C / J,
                      "note": "risk over {rbar,alow,ahigh}=delta_r/(3J); coverage separate delta_c/J"},
        "certificate_core": "fedcore.certificate.cp (cp_upper/cp_lower), joint (no CP reimplemented)",
        "output_sha256": check,
        "preservation": "primary final_cell_results.csv NOT written by this script",
    }
    with open(os.path.join(RESCUE_DIR, "RESCUE_MANIFEST.json"), "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, sort_keys=True)
        h.write("\n")
    with open(os.path.join(RESCUE_DIR, "RESCUE_CHECKSUMS.sha256"), "w", encoding="utf-8") as h:
        for name in list(outputs) + ["RESCUE_MANIFEST.json"]:
            h.write(f"{_sha256(os.path.join(RESCUE_DIR, name))}  {name}\n")

    _print_headline(cells, s1, s2, s3, taxonomy)
    print("\nOutputs written to", RESCUE_DIR)


def _print_headline(cells, s1, s2, s3, taxonomy) -> None:
    sim = s1["simultaneous_family_results.csv"]
    holm = s1["holm_family_results.csv"]
    fresh = s2["fresh_draw_results.csv"]
    print("\n=== HEADLINE: does a VALID finite-family selector turn 0/50 into x/50? ===")
    for alpha in ALPHAS:
        s = [r for r in sim if r["alpha"] == alpha]
        h = [r for r in holm if r["alpha"] == alpha]
        f = [r for r in fresh if r["alpha"] == alpha and r["n_j"] == NJ_PRIMARY]
        prim = sum(1 for r in s if r["primary_certified"])
        n_sim = sum(1 for r in s if r["family_certified"])
        n_holm = sum(1 for r in h if r["holm_certified"])
        n_fresh = sum(1 for r in f if r["certified"])
        any_single = sum(1 for r in s if r["any_single_selector_certifies_diag"])
        tag = "  <== PRIMARY" if alpha == ALPHA_PRIMARY else ""
        print(f"  alpha={alpha:.2f}: primary {prim}/50 | S1-simultaneous {n_sim}/50 | "
              f"Holm {n_holm}/50 | S2-fresh {n_fresh}/50 | (diag any-single-selector {any_single}/50){tag}")
    print("\n=== S3 canonical secondary recheck (0/50 conclusion?) ===")
    lam = s3["canonical_budget_lambda_results.csv"]
    for alpha in ALPHAS:
        f = [r for r in lam if r["alpha"] == alpha and r["primary_m"]]
        c_can = sum(1 for r in f if r["canonical_certified"])
        c_orig = sum(1 for r in f if r["original_certified"])
        print(f"  alpha={alpha:.2f}: traffic(m=1000) canonical {c_can}/50 | original {c_orig}/50")
    print("\n=== barrier taxonomy (family) at alpha=0.20 ===")
    from collections import Counter
    c = Counter(r["barrier_class"] for r in taxonomy if r["alpha"] == ALPHA_PRIMARY)
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}/50")


if __name__ == "__main__":
    main()
