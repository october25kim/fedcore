"""Count-only audit interface shared by all v3 H/S/B procedures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from fedcore.experiments.v3_contract import (
    CLIENTS,
    J,
    M,
    ContractError,
    canonical_json_sha256,
)
from fedcore.experiments.v3_proposal import ProposalMember, oriented_scores, proposal_error_vector


@dataclass(frozen=True)
class AuditClient:
    client: str
    logits: np.ndarray
    labels: np.ndarray
    is_unknown: np.ndarray


def count_audit_records(
    model_id: str,
    members: Sequence[ProposalMember],
    audits: Sequence[AuditClient],
) -> list[dict[str, object]]:
    """Reduce trusted audit records to the registered ``(n,A,K)`` tensor."""

    if len(members) != M or tuple(item.candidate_index for item in members) != tuple(range(M)):
        raise ContractError("proposal family must contain the ordered 12 members")
    if len(audits) != J or tuple(item.client for item in audits) != CLIENTS:
        raise ContractError(f"audit clients must be ordered as {CLIENTS!r}")
    rows: list[dict[str, object]] = []
    for audit in audits:
        logits = np.asarray(audit.logits, dtype=np.float64)
        scores = oriented_scores(logits)
        errors = proposal_error_vector(logits, audit.labels, audit.is_unknown)
        n = int(logits.shape[0])
        for member in members:
            if not member.proposal_feasible:
                a = 0
                k = 0
            else:
                if member.threshold is None:
                    raise ContractError("feasible proposal member has no threshold")
                accepted = scores[member.score] >= float(member.threshold)
                a = int(accepted.sum())
                k = int(np.count_nonzero(accepted & errors))
            rows.append(
                {
                    "model_id": model_id,
                    "candidate_index": member.candidate_index,
                    "client": audit.client,
                    "status": "ok",
                    "n": n,
                    "A": a,
                    "K": k,
                }
            )
    return sorted(rows, key=lambda row: (int(row["candidate_index"]), CLIENTS.index(str(row["client"]))))


def model_failure_count_rows(model_id: str) -> list[dict[str, object]]:
    return [
        {
            "model_id": model_id,
            "candidate_index": candidate_index,
            "client": client,
            "status": "model_failure",
            "n": None,
            "A": None,
            "K": None,
        }
        for candidate_index in range(M)
        for client in CLIENTS
    ]


def validate_count_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_model_id: str | None = None,
    expected_n: int | None = None,
) -> list[dict[str, object]]:
    normalized = [dict(row) for row in rows]
    if len(normalized) != M * J:
        raise ContractError(f"expected {M * J} count rows per model, observed {len(normalized)}")
    keys: set[tuple[int, str]] = set()
    statuses: set[str] = set()
    for row in normalized:
        if expected_model_id is not None and row.get("model_id") != expected_model_id:
            raise ContractError("count row model_id mismatch")
        index = int(row["candidate_index"])
        client = str(row["client"])
        if index not in range(M) or client not in CLIENTS:
            raise ContractError("count row key outside registered family/client set")
        key = (index, client)
        if key in keys:
            raise ContractError(f"duplicate count key: {key!r}")
        keys.add(key)
        status = str(row["status"])
        statuses.add(status)
        if status == "model_failure":
            if any(row.get(name) not in (None, "") for name in ("n", "A", "K")):
                raise ContractError("model-failure counts must be explicit nulls")
            continue
        if status != "ok":
            raise ContractError(f"unexpected count status: {status}")
        n, a, k = int(row["n"]), int(row["A"]), int(row["K"])
        if not 0 <= k <= a <= n:
            raise ContractError("count ordering must satisfy 0 <= K <= A <= n")
        if expected_n is not None and n != expected_n:
            raise ContractError(f"audit n drift: expected {expected_n}, observed {n}")
    if statuses not in ({"ok"}, {"model_failure"}):
        raise ContractError("a model cannot mix successful and failure count rows")
    return sorted(normalized, key=lambda row: (int(row["candidate_index"]), CLIENTS.index(str(row["client"]))))


def count_tensor_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    normalized = validate_count_rows(rows)
    payload = [
        {
            "candidate_index": int(row["candidate_index"]),
            "client": str(row["client"]),
            "status": str(row["status"]),
            "n": None if row.get("n") in (None, "") else int(row["n"]),
            "A": None if row.get("A") in (None, "") else int(row["A"]),
            "K": None if row.get("K") in (None, "") else int(row["K"]),
        }
        for row in normalized
    ]
    return canonical_json_sha256(payload)


def count_matrices(
    rows: Iterable[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = validate_count_rows(rows)
    if str(normalized[0]["status"]) == "model_failure":
        raise ContractError("model-failure placeholders have no count matrices")
    n = np.zeros(J, dtype=np.int64)
    A = np.zeros((M, J), dtype=np.int64)
    K = np.zeros((M, J), dtype=np.int64)
    for row in normalized:
        m = int(row["candidate_index"])
        j = CLIENTS.index(str(row["client"]))
        current_n = int(row["n"])
        if m == 0:
            n[j] = current_n
        elif n[j] != current_n:
            raise ContractError("n must be candidate-invariant within a client")
        A[m, j] = int(row["A"])
        K[m, j] = int(row["K"])
    return n, A, K


__all__ = [
    "AuditClient",
    "count_audit_records",
    "count_matrices",
    "count_tensor_sha256",
    "model_failure_count_rows",
    "validate_count_rows",
]
