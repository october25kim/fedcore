"""Proposal-only selector construction for the sealed PACS H/S/B v3 family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from fedcore.experiments.v3_contract import (
    ALPHA,
    M,
    Candidate,
    candidate_family,
    canonical_json_sha256,
)


@dataclass(frozen=True)
class ProposalMember:
    candidate_index: int
    score: str
    gamma: float
    proposal_feasible: bool
    threshold: float | None
    proposal_n: int
    proposal_A: int
    proposal_K: int
    proposal_risk: float | None

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def _validate_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("known-class logits must have shape (N, 4)")
    if values.shape[0] == 0 or np.any(~np.isfinite(values)):
        raise ValueError("logits must be non-empty and finite")
    return values


def oriented_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Return the three registered higher-is-more-confident score arrays."""

    values = _validate_logits(logits)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    ordered = np.sort(probabilities, axis=1)
    return {
        "msp": probabilities.max(axis=1),
        "energy_lse": logsumexp(values, axis=1),
        "margin": ordered[:, -1] - ordered[:, -2],
    }


def proposal_error_vector(
    logits: np.ndarray,
    labels: Sequence[int],
    is_unknown: Sequence[bool],
) -> np.ndarray:
    values = _validate_logits(logits)
    y = np.asarray(labels, dtype=np.int64)
    unknown = np.asarray(is_unknown, dtype=bool)
    if y.shape != (values.shape[0],) or unknown.shape != (values.shape[0],):
        raise ValueError("labels and is_unknown must be length N")
    prediction = np.argmax(values, axis=1)
    return unknown | (prediction != y)


def _eligible_exact(k: int, a: int, gamma: float, alpha: float) -> bool:
    if a <= 0:
        return False
    target = Fraction(str(gamma)) * Fraction(str(alpha))
    return k * target.denominator <= a * target.numerator


def proposal_risk_eligible(k: int, a: int, gamma: float, alpha: float = ALPHA) -> bool:
    """Public exact-arithmetic form of the registered proposal comparison."""

    if k < 0 or a < 0 or k > a:
        raise ValueError("proposal counts must satisfy 0 <= K <= A")
    return _eligible_exact(k, a, gamma, alpha)


def _threshold_grid(scores: np.ndarray, n_grid: int = 300) -> np.ndarray:
    if n_grid <= 1:
        raise ValueError("n_grid must exceed one")
    return np.unique(
        np.quantile(
            np.asarray(scores, dtype=np.float64),
            np.linspace(0.0, 1.0, n_grid, dtype=np.float64),
            method="linear",
        )
    )


def construct_proposal_family(
    logits: np.ndarray,
    labels: Sequence[int],
    is_unknown: Sequence[bool],
    *,
    alpha: float = ALPHA,
    family: Sequence[Candidate] | None = None,
) -> tuple[ProposalMember, ...]:
    """Freeze all 12 selectors using proposal records only.

    The function never receives certification records.  For every registered
    score/gamma pair it maximizes accepted proposal count subject to the exact
    integer risk comparison and breaks an acceptance-count tie using the smaller
    threshold.  Infeasible members remain ordered reject-all placeholders.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    values = _validate_logits(logits)
    errors = proposal_error_vector(values, labels, is_unknown)
    scores_by_name = oriented_scores(values)
    frozen_family = tuple(candidate_family() if family is None else family)
    if len(frozen_family) != M:
        raise ValueError(f"registered family must contain exactly {M} members")
    members: list[ProposalMember] = []
    for candidate in frozen_family:
        scores = scores_by_name[candidate.score]
        best: tuple[int, float, int] | None = None
        for threshold in _threshold_grid(scores):
            accepted = scores >= threshold
            a = int(accepted.sum())
            k = int(np.count_nonzero(errors & accepted))
            if not _eligible_exact(k, a, candidate.gamma, alpha):
                continue
            key = (a, -float(threshold), k)
            if best is None or key > (best[0], -best[1], best[2]):
                best = (a, float(threshold), k)
        if best is None:
            members.append(
                ProposalMember(
                    candidate_index=candidate.candidate_index,
                    score=candidate.score,
                    gamma=candidate.gamma,
                    proposal_feasible=False,
                    threshold=None,
                    proposal_n=int(values.shape[0]),
                    proposal_A=0,
                    proposal_K=0,
                    proposal_risk=None,
                )
            )
            continue
        a, threshold, k = best
        members.append(
            ProposalMember(
                candidate_index=candidate.candidate_index,
                score=candidate.score,
                gamma=candidate.gamma,
                proposal_feasible=True,
                threshold=threshold,
                proposal_n=int(values.shape[0]),
                proposal_A=a,
                proposal_K=k,
                proposal_risk=float(k / a),
            )
        )
    return tuple(members)


def proposal_manifest_sha256(members: Iterable[ProposalMember]) -> str:
    rows = [member.as_row() for member in members]
    return canonical_json_sha256(rows)


def members_from_rows(rows: Iterable[Mapping[str, object]]) -> tuple[ProposalMember, ...]:
    members = tuple(
        ProposalMember(
            candidate_index=int(row["candidate_index"]),
            score=str(row["score"]),
            gamma=float(row["gamma"]),
            proposal_feasible=bool(row["proposal_feasible"]),
            threshold=(None if row.get("threshold") in (None, "") else float(row["threshold"])),
            proposal_n=int(row.get("proposal_n", 0)),
            proposal_A=int(row.get("proposal_A", 0)),
            proposal_K=int(row.get("proposal_K", 0)),
            proposal_risk=(
                None
                if row.get("proposal_risk") in (None, "")
                else float(row["proposal_risk"])
            ),
        )
        for row in rows
    )
    if len(members) != M or tuple(m.candidate_index for m in members) != tuple(range(M)):
        raise ValueError("proposal rows do not preserve the registered 12-member ordering")
    return members


__all__ = [
    "ProposalMember",
    "construct_proposal_family",
    "members_from_rows",
    "oriented_scores",
    "proposal_error_vector",
    "proposal_manifest_sha256",
    "proposal_risk_eligible",
]
