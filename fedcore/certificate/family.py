"""Theorem-aligned simple simultaneous certificates for frozen families.

This module covers the numerical-UCB family branch. Holm/IUT is a separate
full-simplex fixed-alpha decision implemented in :mod:`fedcore.officehome_rescue`.
Every family member must be frozen before its certification counts are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from fedcore.counts import strict_count_vector
from fedcore.mixture import BoundedSimplex, solve_coverage_infimum, solve_robust_ratio

from .cp import cp_lower, cp_upper


@dataclass(frozen=True)
class SimpleFamilyMemberCertificate:
    """Numerical risk and coverage certificate for one frozen family member."""

    risk_ucb: float
    coverage_lcb: float
    risk_pass: bool
    coverage_pass: bool
    certified: bool
    rbar: np.ndarray
    risk_acceptance_lower: Optional[np.ndarray]
    risk_acceptance_upper: Optional[np.ndarray]
    coverage_acceptance_lower: np.ndarray
    solver_status: str
    solver_certificate_valid: bool
    failure_reason: str
    solver_diagnostics: dict[str, object]


@dataclass(frozen=True)
class SimpleFamilyCertificate:
    """Simultaneous result for a proposal-frozen numerical-UCB family."""

    members: tuple[SimpleFamilyMemberCertificate, ...]
    risk_ucb: np.ndarray
    coverage_lcb: np.ndarray
    risk_pass: np.ndarray
    coverage_pass: np.ndarray
    certified: np.ndarray
    risk_tail: float
    coverage_tail: float
    risk_acceptance_tail: Optional[float]
    mixture_target: str
    M: int
    S: int


def _family_counts(
    A: Sequence[Sequence[int]],
    K: Sequence[Sequence[int]],
    n: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accepted = np.asarray(A)
    errors = np.asarray(K)
    if accepted.ndim != 2 or errors.shape != accepted.shape:
        raise ValueError("A and K must be aligned (M, S) matrices")
    if accepted.shape[0] < 1 or accepted.shape[1] < 1:
        raise ValueError("the frozen family and stratum set must be non-empty")
    rows_a = np.vstack(
        [strict_count_vector(row, f"A[{m}]") for m, row in enumerate(accepted)]
    )
    rows_k = np.vstack(
        [strict_count_vector(row, f"K[{m}]") for m, row in enumerate(errors)]
    )
    totals = strict_count_vector(n, "n")
    if totals.shape != (accepted.shape[1],):
        raise ValueError("n must be a length-S vector")
    if np.any(rows_k > rows_a) or np.any(rows_a > totals[None, :]):
        raise ValueError("counts must satisfy 0 <= K <= A <= n")
    return rows_a, rows_k, totals


def simple_simultaneous_family_certificate(
    A: Sequence[Sequence[int]],
    K: Sequence[Sequence[int]],
    n: Sequence[int],
    *,
    alpha: float,
    delta_r: float,
    delta_c: float,
    mixture_target: str = "simplex",
    lambda_lower: Optional[Sequence[float]] = None,
    lambda_upper: Optional[Sequence[float]] = None,
) -> SimpleFamilyCertificate:
    """Certify every member of a proposal-frozen finite family.

    For the full simplex, each member uses ``delta_r/M`` and ``delta_c/M``
    with no additional stratum penalty. For a strict coordinate-bounded mixture,
    the three risk-side endpoint families use ``delta_r/(3*S*M)`` and the
    separate coverage endpoint family uses ``delta_c/(S*M)``.

    A bounded solver failure is never promoted to an approximate certificate.
    It returns ``risk_ucb=inf``, ``coverage_lcb=0``, and ``certified=False``.
    """
    accepted, errors, totals = _family_counts(A, K, n)
    M, S = accepted.shape
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not (0.0 < float(delta_r) < 1.0 and 0.0 < float(delta_c) < 1.0):
        raise ValueError("delta_r and delta_c must lie in (0, 1)")
    if mixture_target not in {"simplex", "bounded"}:
        raise ValueError("mixture_target must be 'simplex' or 'bounded'")

    if mixture_target == "simplex":
        if lambda_lower is not None or lambda_upper is not None:
            raise ValueError("simplex mode does not accept coordinate bounds")
        beta_r = float(delta_r) / M
        beta_c = float(delta_c) / M
        risk_acceptance_tail: Optional[float] = None
        mixture = None
        configuration_error = ""
    else:
        if lambda_lower is None or lambda_upper is None:
            raise ValueError("bounded mode requires lambda_lower and lambda_upper")
        beta_r = float(delta_r) / (3.0 * S * M)
        beta_c = float(delta_c) / (S * M)
        risk_acceptance_tail = beta_r
        try:
            mixture = BoundedSimplex(lambda_lower, lambda_upper).tightened()
            if mixture.dimension != S:
                raise ValueError("mixture bounds must have length S")
            configuration_error = ""
        except (ValueError, RuntimeError) as exc:
            mixture = None
            configuration_error = f"configuration-infeasible:{type(exc).__name__}:{exc}"

    members: list[SimpleFamilyMemberCertificate] = []
    for m in range(M):
        rbar = np.array(
            [cp_upper(int(errors[m, s]), int(accepted[m, s]), beta_r) for s in range(S)],
            dtype=float,
        )
        coverage_lower = np.array(
            [cp_lower(int(accepted[m, s]), int(totals[s]), beta_c) for s in range(S)],
            dtype=float,
        )

        if mixture_target == "simplex":
            risk_ucb = float(np.max(rbar))
            coverage_lcb = float(np.min(coverage_lower))
            risk_lower = None
            risk_upper = None
            solver_valid = True
            solver_status = "closed_form_full_simplex"
            diagnostics: dict[str, object] = {
                "solver_status": solver_status,
                "solver_certificate_valid": True,
                "risk_solver_status": solver_status,
                "risk_solver_certificate_valid": True,
                "coverage_solver_status": solver_status,
                "coverage_solver_certificate_valid": True,
            }
            reason = ""
        elif mixture is None:
            risk_ucb = float("inf")
            coverage_lcb = 0.0
            risk_lower = np.array(
                [cp_lower(int(accepted[m, s]), int(totals[s]), beta_r) for s in range(S)]
            )
            risk_upper = np.array(
                [cp_upper(int(accepted[m, s]), int(totals[s]), beta_r) for s in range(S)]
            )
            solver_valid = False
            solver_status = "configuration-infeasible"
            reason = configuration_error
            diagnostics = {
                "solver_status": solver_status,
                "solver_certificate_valid": False,
                "risk_solver_status": "not_run",
                "risk_solver_certificate_valid": False,
                "coverage_solver_status": "not_run",
                "coverage_solver_certificate_valid": False,
                "solver_reason": reason,
            }
        else:
            risk_lower = np.array(
                [cp_lower(int(accepted[m, s]), int(totals[s]), beta_r) for s in range(S)],
                dtype=float,
            )
            risk_upper = np.array(
                [cp_upper(int(accepted[m, s]), int(totals[s]), beta_r) for s in range(S)],
                dtype=float,
            )
            ratio = solve_robust_ratio(rbar, risk_lower, risk_upper, mixture)
            coverage = solve_coverage_infimum(coverage_lower, mixture)
            solver_valid = bool(ratio.certificate_valid and coverage.certificate_valid)
            solver_status = f"risk:{ratio.status};coverage:{coverage.status}"
            diagnostics = {
                **ratio.diagnostics("risk_solver"),
                **coverage.diagnostics("coverage_solver"),
                "solver_status": solver_status,
                "solver_certificate_valid": solver_valid,
            }
            if solver_valid and ratio.feasible:
                risk_ucb = float(ratio.value)
                coverage_lcb = float(coverage.value)
                reason = ratio.reason or coverage.reason or ""
            else:
                risk_ucb = float("inf")
                coverage_lcb = 0.0
                reason = ratio.reason or coverage.reason or "solver-validation-failed"

        risk_pass = bool(solver_valid and np.isfinite(risk_ucb) and risk_ucb <= alpha)
        coverage_pass = bool(solver_valid and coverage_lcb > 0.0)
        certified = bool(risk_pass and coverage_pass)
        members.append(
            SimpleFamilyMemberCertificate(
                risk_ucb=risk_ucb,
                coverage_lcb=coverage_lcb,
                risk_pass=risk_pass,
                coverage_pass=coverage_pass,
                certified=certified,
                rbar=rbar,
                risk_acceptance_lower=risk_lower,
                risk_acceptance_upper=risk_upper,
                coverage_acceptance_lower=coverage_lower,
                solver_status=solver_status,
                solver_certificate_valid=solver_valid,
                failure_reason=reason,
                solver_diagnostics=diagnostics,
            )
        )

    return SimpleFamilyCertificate(
        members=tuple(members),
        risk_ucb=np.array([member.risk_ucb for member in members], dtype=float),
        coverage_lcb=np.array([member.coverage_lcb for member in members], dtype=float),
        risk_pass=np.array([member.risk_pass for member in members], dtype=bool),
        coverage_pass=np.array([member.coverage_pass for member in members], dtype=bool),
        certified=np.array([member.certified for member in members], dtype=bool),
        risk_tail=beta_r,
        coverage_tail=beta_c,
        risk_acceptance_tail=risk_acceptance_tail,
        mixture_target=mixture_target,
        M=M,
        S=S,
    )


def select_simple_family_member(
    certificate: SimpleFamilyCertificate,
    proposal_order: Optional[Sequence[int]] = None,
) -> Optional[int]:
    """Choose maximum certified coverage with a proposal-frozen tie order."""
    if proposal_order is None:
        order = list(range(certificate.M))
    else:
        order = [int(value) for value in proposal_order]
        if sorted(order) != list(range(certificate.M)):
            raise ValueError("proposal_order must be a permutation of 0..M-1")
    eligible = [index for index in order if certificate.certified[index]]
    if not eligible:
        return None
    priority = {index: rank for rank, index in enumerate(order)}
    return min(
        eligible,
        key=lambda index: (-float(certificate.coverage_lcb[index]), priority[index]),
    )


__all__ = [
    "SimpleFamilyCertificate",
    "SimpleFamilyMemberCertificate",
    "select_simple_family_member",
    "simple_simultaneous_family_certificate",
]
