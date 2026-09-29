"""Pure count-to-decision implementation of the sealed v3 H/S/B comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.experiments.v3_contract import (
    ALPHA,
    DELTA_C,
    DELTA_R,
    J,
    M,
    PRIMARY_BUDGET_TOTAL,
    PRIMARY_REPLICATE_ID,
    PROCEDURES,
    PROTOCOL_ID,
    ContractError,
    ModelCell,
)
from fedcore.experiments.v3_counts import count_matrices, count_tensor_sha256, validate_count_rows
from fedcore.experiments.v3_proposal import ProposalMember
from fedcore.officehome_rescue import (
    candidate_null_pvalue,
    holm_adjusted_pvalues,
    holm_step_down_reject,
)


@dataclass(frozen=True)
class CandidateDecision:
    model_id: str
    candidate_index: int
    procedure: str
    model_evaluable: bool
    proposal_feasible: bool
    checkpoint_sha256: str | None
    proposal_seal_sha256: str | None
    audit_sequence_sha256: str | None
    count_tensor_sha256: str | None
    budget_total: int
    replicate_id: int
    score: str
    gamma: float
    threshold: float | None
    risk_tail: float | None
    coverage_tail: float | None
    raw_max_client_p: float | None
    holm_rank: int | None
    holm_critical_value: float | None
    holm_adjusted_p: float | None
    risk_ucb: float | None
    coverage_lcb: float | None
    risk_pass: bool | None
    coverage_positive: bool | None
    candidate_certified: bool
    selected: bool

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def _validate_members(members: Sequence[ProposalMember]) -> None:
    if len(members) != M:
        raise ContractError(f"expected {M} proposal members")
    for index, member in enumerate(members):
        if member.candidate_index != index:
            raise ContractError("proposal family order drift")
        if member.proposal_feasible and member.threshold is None:
            raise ContractError("feasible proposal member lacks a threshold")
        if not member.proposal_feasible and member.threshold is not None:
            raise ContractError("infeasible proposal placeholder must not have a threshold")


def _select(rows: Sequence[CandidateDecision]) -> int | None:
    eligible = [row for row in rows if row.candidate_certified]
    if not eligible:
        return None
    winner = min(
        eligible,
        key=lambda row: (
            -float(row.coverage_lcb),
            float(row.gamma),
            row.score,
            int(row.candidate_index),
        ),
    )
    return winner.candidate_index


def _holm_arrays(A: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, ...]:
    pvalues = np.asarray(
        [candidate_null_pvalue(A[m], K[m], ALPHA) for m in range(M)],
        dtype=np.float64,
    )
    adjusted, ranks = holm_adjusted_pvalues(pvalues)
    rejected = holm_step_down_reject(pvalues, DELTA_R)
    critical = np.asarray(
        [DELTA_R / (M - int(rank) + 1) for rank in ranks], dtype=np.float64
    )
    return pvalues, adjusted, ranks, rejected, critical


def _bounds(
    A: np.ndarray,
    K: np.ndarray,
    n: np.ndarray,
    *,
    risk_tail: float,
    coverage_tail: float,
) -> tuple[np.ndarray, np.ndarray]:
    risk = np.empty(M, dtype=np.float64)
    coverage = np.empty(M, dtype=np.float64)
    for m in range(M):
        risk[m] = max(cp_upper(int(K[m, j]), int(A[m, j]), risk_tail) for j in range(J))
        coverage[m] = min(
            cp_lower(int(A[m, j]), int(n[j]), coverage_tail) for j in range(J)
        )
    return risk, coverage


def candidate_decisions(
    model: ModelCell,
    members: Sequence[ProposalMember],
    count_rows: Iterable[Mapping[str, object]],
    *,
    checkpoint_sha256: str,
    proposal_seal_sha256: str,
    audit_sequence_sha256: str,
    budget_total: int = PRIMARY_BUDGET_TOTAL,
    replicate_id: int = PRIMARY_REPLICATE_ID,
) -> list[CandidateDecision]:
    """Evaluate H, S, and B from one immutable count tensor.

    The function is intentionally count-only.  It cannot inspect logits, labels,
    PACS images, full-frame truth, or any previously produced scientific result.
    """

    _validate_members(members)
    normalized = validate_count_rows(count_rows, expected_model_id=model.model_id)
    if str(normalized[0]["status"]) == "model_failure":
        return model_failure_candidate_decisions(model, members, budget_total, replicate_id)
    n, A, K = count_matrices(normalized)
    tensor_hash = count_tensor_sha256(normalized)
    pvalues, adjusted, ranks, h_reject, critical = _holm_arrays(A, K)
    h_coverage_tail = DELTA_C / M
    _, h_coverage = _bounds(
        A,
        K,
        n,
        risk_tail=DELTA_R / M,
        coverage_tail=h_coverage_tail,
    )
    s_risk_tail = DELTA_R / M
    s_coverage_tail = DELTA_C / M
    s_risk, s_coverage = _bounds(
        A,
        K,
        n,
        risk_tail=s_risk_tail,
        coverage_tail=s_coverage_tail,
    )
    b_risk_tail = DELTA_R / (M * J)
    b_coverage_tail = DELTA_C / (M * J)
    b_risk, b_coverage = _bounds(
        A,
        K,
        n,
        risk_tail=b_risk_tail,
        coverage_tail=b_coverage_tail,
    )
    rows: list[CandidateDecision] = []
    for procedure in PROCEDURES:
        for m, member in enumerate(members):
            if procedure == "H":
                risk_tail = None
                coverage_tail = h_coverage_tail
                raw_p = float(pvalues[m])
                rank = int(ranks[m])
                holm_critical = float(critical[m])
                holm_adjusted = float(adjusted[m])
                risk_ucb = None
                risk_pass = bool(h_reject[m])
                coverage_lcb = float(h_coverage[m])
            elif procedure == "S":
                risk_tail = s_risk_tail
                coverage_tail = s_coverage_tail
                raw_p = None
                rank = None
                holm_critical = None
                holm_adjusted = None
                risk_ucb = float(s_risk[m])
                risk_pass = risk_ucb <= ALPHA
                coverage_lcb = float(s_coverage[m])
            else:
                risk_tail = b_risk_tail
                coverage_tail = b_coverage_tail
                raw_p = None
                rank = None
                holm_critical = None
                holm_adjusted = None
                risk_ucb = float(b_risk[m])
                risk_pass = risk_ucb <= ALPHA
                coverage_lcb = float(b_coverage[m])
            coverage_positive = coverage_lcb > 0.0
            certified = bool(member.proposal_feasible and risk_pass and coverage_positive)
            rows.append(
                CandidateDecision(
                    model_id=model.model_id,
                    candidate_index=m,
                    procedure=procedure,
                    model_evaluable=True,
                    proposal_feasible=member.proposal_feasible,
                    checkpoint_sha256=checkpoint_sha256,
                    proposal_seal_sha256=proposal_seal_sha256,
                    audit_sequence_sha256=audit_sequence_sha256,
                    count_tensor_sha256=tensor_hash,
                    budget_total=budget_total,
                    replicate_id=replicate_id,
                    score=member.score,
                    gamma=member.gamma,
                    threshold=member.threshold,
                    risk_tail=risk_tail,
                    coverage_tail=coverage_tail,
                    raw_max_client_p=raw_p,
                    holm_rank=rank,
                    holm_critical_value=holm_critical,
                    holm_adjusted_p=holm_adjusted,
                    risk_ucb=risk_ucb,
                    coverage_lcb=coverage_lcb,
                    risk_pass=bool(risk_pass),
                    coverage_positive=bool(coverage_positive),
                    candidate_certified=certified,
                    selected=False,
                )
            )
    selected_by_procedure: dict[str, int | None] = {}
    for procedure in PROCEDURES:
        selected_by_procedure[procedure] = _select(
            [row for row in rows if row.procedure == procedure]
        )
    return [
        replace(
            row,
            selected=selected_by_procedure[row.procedure] == row.candidate_index,
        )
        for row in rows
    ]


def model_failure_candidate_decisions(
    model: ModelCell,
    members: Sequence[ProposalMember],
    budget_total: int = PRIMARY_BUDGET_TOTAL,
    replicate_id: int = PRIMARY_REPLICATE_ID,
) -> list[CandidateDecision]:
    _validate_members(members)
    return [
        CandidateDecision(
            model_id=model.model_id,
            candidate_index=member.candidate_index,
            procedure=procedure,
            model_evaluable=False,
            proposal_feasible=False,
            checkpoint_sha256=None,
            proposal_seal_sha256=None,
            audit_sequence_sha256=None,
            count_tensor_sha256=None,
            budget_total=budget_total,
            replicate_id=replicate_id,
            score=member.score,
            gamma=member.gamma,
            threshold=None,
            risk_tail=None,
            coverage_tail=None,
            raw_max_client_p=None,
            holm_rank=None,
            holm_critical_value=None,
            holm_adjusted_p=None,
            risk_ucb=None,
            coverage_lcb=None,
            risk_pass=None,
            coverage_positive=None,
            candidate_certified=False,
            selected=False,
        )
        for procedure in PROCEDURES
        for member in members
    ]


def _diagnostic_subtype(rows: Sequence[CandidateDecision]) -> str:
    feasible = [row for row in rows if row.proposal_feasible]
    if not feasible:
        return "not_applicable"
    risk_any = any(row.risk_pass for row in feasible)
    positive_any = any(row.coverage_positive for row in feasible if row.risk_pass)
    if not risk_any:
        return "no_risk_pass"
    if not positive_any:
        return "nonpositive_coverage"
    return "mixed"


def procedure_rows(
    model: ModelCell,
    candidates: Sequence[CandidateDecision],
    *,
    candidate_decision_file_sha256: str,
) -> list[dict[str, object]]:
    if len(candidates) != M * len(PROCEDURES):
        raise ContractError("candidate decision denominator drift")
    output: list[dict[str, object]] = []
    for procedure in PROCEDURES:
        rows = [row for row in candidates if row.procedure == procedure]
        if len(rows) != M:
            raise ContractError("missing candidate decisions for a procedure")
        selected = [row for row in rows if row.selected]
        model_evaluable = all(row.model_evaluable for row in rows)
        if not model_evaluable:
            certified = False
            chosen = None
            refusal_reason = "model_failure"
            diagnostic = "not_applicable"
        elif selected:
            if len(selected) != 1:
                raise ContractError("a procedure selected more than one candidate")
            certified = True
            chosen = selected[0]
            refusal_reason = "none"
            diagnostic = "not_applicable"
        else:
            certified = False
            chosen = None
            if not any(row.proposal_feasible for row in rows):
                refusal_reason = "proposal_family_all_infeasible"
            else:
                refusal_reason = "no_candidate_certified"
            diagnostic = _diagnostic_subtype(rows)
        first = rows[0]
        output.append(
            {
                "protocol_id": PROTOCOL_ID,
                "model_id": model.model_id,
                "architecture": model.architecture,
                "split": model.split,
                "nominal_training_seed": model.nominal_training_seed,
                "training_status": "terminal_success" if model_evaluable else "terminal_model_failure",
                "model_evaluable": model_evaluable,
                "failure_stage": "none" if model_evaluable else "training",
                "checkpoint_sha256": first.checkpoint_sha256,
                "proposal_seal_sha256": first.proposal_seal_sha256,
                "audit_sequence_sha256": first.audit_sequence_sha256,
                "count_tensor_sha256": first.count_tensor_sha256,
                "candidate_decision_file_sha256": (
                    candidate_decision_file_sha256 if model_evaluable else None
                ),
                "procedure": procedure,
                "alpha": ALPHA,
                "delta_r": DELTA_R,
                "delta_c": DELTA_C,
                "mixture_set": "Delta^3",
                "budget_total": first.budget_total,
                "replicate_id": first.replicate_id,
                "proposal_feasible_members": (
                    sum(row.proposal_feasible for row in rows) if model_evaluable else None
                ),
                "certified": certified,
                "selected_candidate": chosen.candidate_index if chosen else -1,
                "selected_score": chosen.score if chosen else "",
                "selected_gamma": chosen.gamma if chosen else None,
                "selected_threshold": chosen.threshold if chosen else None,
                "risk_decision": bool(chosen.risk_pass) if chosen else False,
                "risk_ucb": chosen.risk_ucb if chosen else None,
                "raw_p_value": chosen.raw_max_client_p if chosen else None,
                "holm_adjusted_p_value": chosen.holm_adjusted_p if chosen else None,
                "coverage_lcb": float(chosen.coverage_lcb) if chosen else 0.0,
                "ECA": float(chosen.coverage_lcb) if chosen else 0.0,
                "refusal_reason": refusal_reason,
                "diagnostic_failure_subtype": diagnostic,
            }
        )
    return output


def validate_hsb_invariants(candidates: Sequence[CandidateDecision]) -> None:
    """Raise ``ContractError`` for any candidate- or cell-level H/S/B violation."""

    if len(candidates) != M * len(PROCEDURES):
        raise ContractError("candidate denominator drift")
    by_key = {(row.procedure, row.candidate_index): row for row in candidates}
    model_evaluable = all(row.model_evaluable for row in candidates)
    if not model_evaluable:
        if any(row.candidate_certified or row.selected for row in candidates):
            raise ContractError("model-failure placeholder cannot certify or select")
        return
    hashes = {row.count_tensor_sha256 for row in candidates}
    if len(hashes) != 1 or None in hashes:
        raise ContractError("H/S/B did not consume one identical count tensor")
    for m in range(M):
        h, s, b = (by_key[(procedure, m)] for procedure in PROCEDURES)
        if abs(float(h.coverage_lcb) - float(s.coverage_lcb)) > 1e-12:
            raise ContractError("C_H != C_S")
        if float(s.risk_ucb) > float(b.risk_ucb) + 1e-12:
            raise ContractError("U_S > U_B")
        if float(s.coverage_lcb) + 1e-12 < float(b.coverage_lcb):
            raise ContractError("C_S < C_B")
        if s.risk_pass and not h.risk_pass:
            raise ContractError("S risk pass without H risk pass")
        if b.risk_pass and not s.risk_pass:
            raise ContractError("B risk pass without S risk pass")
        if b.candidate_certified and not s.candidate_certified:
            raise ContractError("B certification without S certification")
        if s.candidate_certified and not h.candidate_certified:
            raise ContractError("S certification without H certification")
    selected = {}
    for procedure in PROCEDURES:
        procedure_rows_ = [row for row in candidates if row.procedure == procedure]
        marked = [row for row in procedure_rows_ if row.selected]
        expected = _select(procedure_rows_)
        if len(marked) != (0 if expected is None else 1):
            raise ContractError("selected marker count mismatch")
        if expected is not None and marked[0].candidate_index != expected:
            raise ContractError("selection tie-break mismatch")
        selected[procedure] = marked[0] if marked else None
    cert = {procedure: selected[procedure] is not None for procedure in PROCEDURES}
    if cert["B"] and not cert["S"] or cert["S"] and not cert["H"]:
        raise ContractError("cell-level B=>S=>H invariant failed")
    eca = {
        procedure: 0.0 if selected[procedure] is None else float(selected[procedure].coverage_lcb)
        for procedure in PROCEDURES
    }
    if eca["H"] + 1e-12 < eca["S"] or eca["S"] + 1e-12 < eca["B"]:
        raise ContractError("cell-level E_H>=E_S>=E_B invariant failed")


__all__ = [
    "CandidateDecision",
    "candidate_decisions",
    "model_failure_candidate_decisions",
    "procedure_rows",
    "validate_hsb_invariants",
]
