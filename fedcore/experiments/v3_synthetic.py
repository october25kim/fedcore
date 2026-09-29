"""PACS-free deterministic fixtures for v3 integration and failure gates."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from fedcore.experiments.v3_contract import (
    CLIENTS,
    PRIMARY_DRAWS_PER_CLIENT,
    ModelCell,
    canonical_json_sha256,
    candidate_family,
)
from fedcore.experiments.v3_counts import (
    AuditClient,
    count_audit_records,
    model_failure_count_rows,
)
from fedcore.experiments.v3_proposal import (
    ProposalMember,
    construct_proposal_family,
    proposal_manifest_sha256,
)


SYNTHETIC_MARKERS = {
    "synthetic": True,
    "PACS_images_used": False,
    "PACS_inference": False,
}


def _seed(model: ModelCell, role: str) -> int:
    return int(canonical_json_sha256({"model_id": model.model_id, "role": role})[:16], 16)


def _separated_logits(
    seed: int,
    n: int,
    *,
    unknown_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create easy known/unknown records without loading any PACS asset."""

    if not 0 <= unknown_count < n:
        raise ValueError("unknown_count must lie in [0,n)")
    rng = np.random.default_rng(seed)
    known_count = n - unknown_count
    labels = np.arange(n, dtype=np.int64) % 4
    is_unknown = np.zeros(n, dtype=bool)
    is_unknown[known_count:] = True
    labels[known_count:] = -1
    logits = np.full((n, 4), -2.0, dtype=np.float64)
    for i in range(known_count):
        logits[i, int(labels[i])] = 5.0 + float(rng.uniform(0.0, 0.2))
    logits[:known_count] += rng.normal(0.0, 0.01, size=(known_count, 4))
    logits[known_count:] = rng.normal(0.0, 0.01, size=(unknown_count, 4))
    permutation = rng.permutation(n)
    return logits[permutation], labels[permutation], is_unknown[permutation]


def _all_error_logits(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.0, size=(n, 4)).astype(np.float64)
    labels = np.full(n, -1, dtype=np.int64)
    is_unknown = np.ones(n, dtype=bool)
    return logits, labels, is_unknown


def _failure_members() -> tuple[ProposalMember, ...]:
    return tuple(
        ProposalMember(
            candidate_index=item.candidate_index,
            score=item.score,
            gamma=item.gamma,
            proposal_feasible=False,
            threshold=None,
            proposal_n=0,
            proposal_A=0,
            proposal_K=0,
            proposal_risk=None,
        )
        for item in candidate_family()
    )


def _fixture_count_rows(
    model_id: str,
    *,
    strict_hsb: bool,
) -> list[dict[str, object]]:
    if strict_hsb:
        accepted = (64, 100, 128) + (128,) * 9
        errors = (3, 9, 14) + (15,) * 9
    else:
        accepted = (128,) * 12
        errors = (15,) * 12
    return [
        {
            "model_id": model_id,
            "candidate_index": candidate_index,
            "client": client,
            "status": "ok",
            "n": PRIMARY_DRAWS_PER_CLIENT,
            "A": accepted[candidate_index],
            "K": errors[candidate_index],
        }
        for candidate_index in range(12)
        for client in CLIENTS
    ]


def synthetic_cell_payload(model: ModelCell, ordinal: int) -> dict[str, Any]:
    """Return one deterministic cell payload with registered adverse cases.

    Ordinals 0--26 use the strict ``H>S>B`` count fixture. Ordinal 27 is
    proposal-feasible but every candidate fails the risk decision. Ordinal 28
    is a terminal training failure, and ordinal 29 is evaluable but
    proposal-infeasible. The adverse cases are synthetic wiring checks and are
    never scientific PACS outcomes.
    """

    if ordinal == 28:
        members = _failure_members()
        counts = model_failure_count_rows(model.model_id)
        status = "terminal_model_failure"
        checkpoint_hash = None
        proposal_hash = None
        audit_hash = None
    elif ordinal == 29:
        logits, labels, unknown = _all_error_logits(_seed(model, "proposal"), 256)
        members = construct_proposal_family(logits, labels, unknown)
        if any(member.proposal_feasible for member in members):
            raise RuntimeError("all-error proposal unexpectedly produced a feasible member")
        audits = []
        for client_index, client in enumerate(CLIENTS):
            a_logits, a_labels, a_unknown = _all_error_logits(
                _seed(model, f"audit:{client_index}"), PRIMARY_DRAWS_PER_CLIENT
            )
            audits.append(AuditClient(client, a_logits, a_labels, a_unknown))
        counts = count_audit_records(model.model_id, members, audits)
        status = "terminal_success"
        checkpoint_hash = canonical_json_sha256({"model": model.model_id, "checkpoint": "synthetic"})
        proposal_hash = proposal_manifest_sha256(members)
        audit_hash = canonical_json_sha256({"model": model.model_id, "audit": "synthetic-all-error"})
    else:
        logits, labels, unknown = _separated_logits(
            _seed(model, "proposal"), 256, unknown_count=32
        )
        members = construct_proposal_family(logits, labels, unknown)
        if not any(member.proposal_feasible for member in members):
            raise RuntimeError("separated synthetic proposal has no feasible member")
        counts = _fixture_count_rows(model.model_id, strict_hsb=ordinal < 27)
        status = "terminal_success"
        checkpoint_hash = canonical_json_sha256({"model": model.model_id, "checkpoint": "synthetic"})
        proposal_hash = proposal_manifest_sha256(members)
        audit_hash = canonical_json_sha256({"model": model.model_id, "audit": "synthetic"})
    return {
        **SYNTHETIC_MARKERS,
        "model": asdict(model),
        "ordinal": ordinal,
        "training_status": status,
        "checkpoint_sha256": checkpoint_hash,
        "proposal_seal_sha256": proposal_hash,
        "audit_sequence_sha256": audit_hash,
        "members": [member.as_row() for member in members],
        "counts": counts,
    }


__all__ = ["SYNTHETIC_MARKERS", "synthetic_cell_payload"]
