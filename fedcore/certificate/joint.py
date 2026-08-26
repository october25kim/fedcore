"""Conservative joint risk/coverage certificate with frozen stratum budgets.

This is the one-shot campaign API. The legacy APIs remain available for golden
reproduction, while this path accounts for every simultaneous confidence event
and uses deterministic bounded-simplex optimization with one-sided endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.counts import strict_count_vector
from fedcore.mixture import BoundedSimplex, solve_coverage_infimum, solve_robust_ratio


def _counts(A, K, n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        strict_count_vector(value, name) for value, name in zip((A, K, n), "AKn")
    )
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("A, K, and n must align")
    accepted, errors, totals = arrays
    if np.any(errors > accepted) or np.any(accepted > totals):
        raise ValueError("counts must satisfy 0 <= K <= A <= n")
    return accepted, errors, totals


def _eps(values: Sequence[float], dimension: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (dimension,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {dimension}")
    if np.any(array <= 0.0) or np.any(array >= 1.0):
        raise ValueError(f"every {name} value must lie in (0, 1)")
    return array


@dataclass(frozen=True)
class JointCertificate:
    risk_ucb: float
    coverage_lcb: float
    certified: bool
    feasible: bool
    reason: str
    rbar: np.ndarray
    alow: np.ndarray
    ahigh: Optional[np.ndarray]
    risk_eps: np.ndarray
    acceptance_lower_eps: np.ndarray
    acceptance_upper_eps: Optional[np.ndarray]
    lambda_lower: np.ndarray
    lambda_upper: np.ndarray
    min_denominator: float
    lambda_star: Optional[np.ndarray]
    acceptance_star: Optional[np.ndarray]
    solver_status: str
    solver_certificate_valid: bool
    solver_diagnostics: dict[str, object] = field(default_factory=dict)


def joint_conditional_certificate(
    A: Sequence[int],
    K: Sequence[int],
    n: Sequence[int],
    *,
    alpha: float,
    risk_eps: Sequence[float],
    acceptance_lower_eps: Sequence[float],
    acceptance_upper_eps: Optional[Sequence[float]] = None,
    lambda_lower: Optional[Sequence[float]] = None,
    lambda_upper: Optional[Sequence[float]] = None,
) -> JointCertificate:
    """Certify risk and coverage jointly over full or coordinate-bounded simplex."""

    accepted, errors, totals = _counts(A, K, n)
    J = len(accepted)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    eps_risk = _eps(risk_eps, J, "risk_eps")
    eps_lower = _eps(acceptance_lower_eps, J, "acceptance_lower_eps")
    rbar = np.array(
        [
            cp_upper(int(errors[j]), int(accepted[j]), float(eps_risk[j]))
            for j in range(J)
        ]
    )
    alow = np.array(
        [
            cp_lower(int(accepted[j]), int(totals[j]), float(eps_lower[j]))
            for j in range(J)
        ]
    )

    full_simplex = lambda_lower is None and lambda_upper is None
    if (lambda_lower is None) != (lambda_upper is None):
        raise ValueError("lambda_lower and lambda_upper must be supplied together")
    if full_simplex:
        mixture = BoundedSimplex(np.zeros(J), np.ones(J))
        ahigh = None
        eps_upper = None
        risk_ucb = float(np.max(rbar))
        min_denominator = float(np.min(alow))
        risk_feasible = min_denominator > 0.0
        reason = "ok" if risk_feasible else "vanishing_denominator"
        lambda_star = None
        acceptance_star = None
        risk_status = "closed_form_full_simplex"
        risk_valid = True
        risk_diagnostics: dict[str, object] = {
            "risk_solver_status": risk_status,
            "risk_solver_certificate_valid": True,
            "risk_solver_tolerance": 0.0,
            "risk_solver_iterations": 0,
        }
    else:
        if acceptance_upper_eps is None:
            raise ValueError("bounded mixture requires acceptance_upper_eps")
        eps_upper = _eps(acceptance_upper_eps, J, "acceptance_upper_eps")
        ahigh = np.array(
            [
                cp_upper(int(accepted[j]), int(totals[j]), float(eps_upper[j]))
                for j in range(J)
            ]
        )
        try:
            mixture = BoundedSimplex(lambda_lower, lambda_upper).tightened()
        except (ValueError, RuntimeError) as exc:
            lower_raw = np.asarray(lambda_lower, dtype=float)
            upper_raw = np.asarray(lambda_upper, dtype=float)
            status = "infeasible_mixture_set"
            diagnostics = {
                "solver_status": f"risk:{status};coverage:not_run",
                "solver_certificate_valid": False,
                "risk_solver_status": status,
                "risk_solver_certificate_valid": False,
                "risk_solver_reason": f"{type(exc).__name__}:{exc}",
                "coverage_solver_status": "not_run",
                "coverage_solver_certificate_valid": False,
            }
            return JointCertificate(
                risk_ucb=float("inf"),
                coverage_lcb=0.0,
                certified=False,
                feasible=False,
                reason=status,
                rbar=rbar,
                alow=alow,
                ahigh=ahigh,
                risk_eps=eps_risk,
                acceptance_lower_eps=eps_lower,
                acceptance_upper_eps=eps_upper,
                lambda_lower=lower_raw,
                lambda_upper=upper_raw,
                min_denominator=0.0,
                lambda_star=None,
                acceptance_star=None,
                solver_status=diagnostics["solver_status"],
                solver_certificate_valid=False,
                solver_diagnostics=diagnostics,
            )
        ratio = solve_robust_ratio(rbar, alow, ahigh, mixture)
        risk_ucb = float(ratio.value)
        min_denominator = float(ratio.min_denominator)
        risk_valid = bool(ratio.certificate_valid)
        risk_feasible = bool(ratio.feasible and risk_valid)
        reason = ratio.reason or "ok"
        lambda_star = ratio.lambda_star
        acceptance_star = ratio.acceptance_star
        risk_status = ratio.status
        risk_diagnostics = ratio.diagnostics("risk_solver")

    coverage = solve_coverage_infimum(alow, mixture)
    coverage_lcb = float(coverage.value)
    coverage_valid = bool(coverage.certificate_valid)
    solver_valid = bool(risk_valid and coverage_valid)
    solver_status = f"risk:{risk_status};coverage:{coverage.status}"
    solver_diagnostics = {
        **risk_diagnostics,
        **coverage.diagnostics("coverage_solver"),
        "solver_status": solver_status,
        "solver_certificate_valid": solver_valid,
    }
    if not solver_valid:
        risk_feasible = False
        reason = (
            ratio.reason
            if not full_simplex and not ratio.certificate_valid and ratio.reason
            else coverage.reason or "solver_validation_failed"
        )
    if not np.any(accepted):
        risk_feasible = False
        reason = "zero_accepted_coverage"
    elif coverage_lcb <= 0.0:
        risk_feasible = False
        reason = "vanishing_denominator"
    certified = bool(risk_feasible and np.isfinite(risk_ucb) and risk_ucb <= alpha)
    return JointCertificate(
        risk_ucb=risk_ucb,
        coverage_lcb=coverage_lcb,
        certified=certified,
        feasible=risk_feasible,
        reason=reason,
        rbar=rbar,
        alow=alow,
        ahigh=ahigh,
        risk_eps=eps_risk,
        acceptance_lower_eps=eps_lower,
        acceptance_upper_eps=eps_upper,
        lambda_lower=mixture.lower,
        lambda_upper=mixture.upper,
        min_denominator=min_denominator,
        lambda_star=lambda_star,
        acceptance_star=acceptance_star,
        solver_status=solver_status,
        solver_certificate_valid=solver_valid,
        solver_diagnostics=solver_diagnostics,
    )
