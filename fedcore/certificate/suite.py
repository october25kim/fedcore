"""Historical pre-registered allocation-variant suite.

This module reproduces the campaign's conservative clientwise union-bound
allocation experiments.  It is not the current no-stratum-penalty Theorem-1
API; use :func:`fedcore.certificate.full_simplex_fixed_member_certificate` for
the theorem-facing full-simplex calculation.  Keeping the distinction explicit
prevents archived sensitivity rows from being mistaken for current-theorem
recertification.

One call to :func:`certify_variant` produces one canonical row: a single
(run, threshold policy, allocation rule, mixture target, alpha) cell of the
pre-registered 2x2 x targets design of WP-A/B/C.

Budget composition (Corollary 1). The total failure budget ``delta`` splits as
``delta_r + delta_c <= delta``; the headline is ``delta_r = delta_c = delta / 2``.
The risk budget ``delta_r`` is spent as follows:

* ``simplex`` historical branch -- one event per client. Client ``j`` spends its whole
  ``eps_j`` on ``rbar_j = U(K_j, A_j; eps_j)``; the certificate is
  ``max_j rbar_j``. Uniform ``eps_j = delta_r / J`` is a valid conservative
  legacy union-bound calculation, not the sharper current Theorem 1.
* ``box`` target -- three events per client: ``rbar_j`` and the two ends of the
  acceptance box. Client ``j`` therefore spends ``eps_j / 3`` on each, so the
  union over all ``3J`` events is still ``sum_j eps_j <= delta_r``. With the
  uniform allocation this reproduces the brief's ``eps_r = delta_r / (3J)``
  exactly; with an allocated ``eps`` it is the natural union-bound composition of
  the archived proposal allocation with the bounded-mixture certificate. The
  certificate is a deterministic global supremum
  over ``Lambda_G(rho)`` with a conservative numerical upper endpoint
  (:mod:`fedcore.certificate.lambda_sets`), never a sampled max.

The coverage bound spends ``delta_c`` as ``delta_c / J`` per client on
``L(A_j, n_j; delta_c / J)`` and reports
``inf_{lam in Lambda} sum_j lam_j L_j``. Deploy iff ``UCB <= alpha`` and
``coverage_lcb > 0``.

Fold hygiene. The threshold policy AND the allocation are functions of the
proposal fold only. The certification fold supplies ``(A_j, K_j, n_j)`` and
nothing else. Test-fold labels are used only for the held-out realized risk that
WP-F checks the certificate against; they never enter selection or certification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from fedcore.policy import ThresholdPolicy, choose_threshold_policy, policy_counts
from fedcore.selector import empirical_risk_coverage, open_set_error

from .allocation import (
    ALLOCATION_RULES,
    R3_KAPPA,
    allocate,
    zero_error_floor,
    zero_error_frontier,
)
from .cp import cp_lower, cp_upper
from .lambda_sets import (
    NormalizedBox,
    solve_normalized_box_coverage,
    solve_normalized_box_risk,
    uniform_box,
)

__all__ = [
    "VariantResult",
    "certify_variant",
    "FAILURE_MODES",
    "MIXTURE_TARGETS",
]

FAILURE_MODES = (
    "",  # certified
    "proposal-infeasible",
    "observed-risk",
    "zero-error-count",
    "margin-limited",
    "solver-failure",
)

MIXTURE_TARGETS = ("simplex", "box")


@dataclass
class VariantResult:
    """One canonical certificate row."""

    certified: bool
    cert_risk_ucb: float
    cert_coverage_lcb: float
    failure_mode: str
    frontier: float
    rbar: np.ndarray
    eps: np.ndarray
    floors: np.ndarray
    A: np.ndarray
    K: np.ndarray
    n: np.ndarray
    thresholds: np.ndarray
    prop_coverage: float
    prop_risk: float
    test_coverage: float
    test_risk: float
    meta: Dict[str, object] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        """Flatten to the canonical CSV schema (per-client fields as lists)."""
        row: Dict[str, object] = {
            "certified": bool(self.certified),
            "cert_risk_ucb": float(self.cert_risk_ucb),
            "cert_coverage_lcb": float(self.cert_coverage_lcb),
            "cert_n": int(np.sum(self.A)),
            "cert_k": int(np.sum(self.K)),
            "failure_mode": self.failure_mode,
            "frontier": float(self.frontier),
            "prop_coverage": float(self.prop_coverage),
            "prop_risk": float(self.prop_risk),
            "test_coverage": float(self.test_coverage),
            "test_risk": float(self.test_risk),
            "per_client_n": np.asarray(self.n).tolist(),
            "per_client_A": np.asarray(self.A).tolist(),
            "per_client_K": np.asarray(self.K).tolist(),
            "per_client_eps": np.asarray(self.eps).tolist(),
            "per_client_floor": np.asarray(self.floors).tolist(),
            "per_client_rbar": np.asarray(self.rbar).tolist(),
            "thresholds": np.asarray(self.thresholds).tolist(),
        }
        row.update(self.meta)
        return row


def _scaled_proposal_counts(
    A_prop: np.ndarray, K_prop: np.ndarray, n_prop: np.ndarray, n_cert: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project proposal counts onto the certification fold's size.

    ``A_hat_j = A_prop_j * n_cert_j / n_prop_j`` per the brief's R1 definition.
    A client with an empty proposal fold projects to zero, which R1 reads as
    "maximally scarce" and hands the largest budget -- the conservative reading.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(n_prop > 0, n_cert / np.maximum(n_prop, 1), 0.0)
    return A_prop * scale, K_prop * scale


def _failure_mode(
    U: float,
    alpha: float,
    A: np.ndarray,
    K: np.ndarray,
    policy_feasible: bool,
) -> str:
    """Classify why a cell did not certify. Order matters: most specific first."""
    if not policy_feasible:
        return "proposal-infeasible"
    if U <= alpha:
        return ""
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = np.where(A > 0, K / np.maximum(A, 1), np.inf)
    if np.any(observed > alpha):
        # The data itself says some client is unsafe; no sample size fixes this.
        return "observed-risk"
    if np.all(K == 0):
        # Perfect accepted sets, still too few of them: a pure counting failure,
        # which is exactly what the archived allocation frontier diagnoses.
        return "zero-error-count"
    return "margin-limited"


def certify_variant(
    prop_view: Dict[str, np.ndarray],
    cert_view: Dict[str, np.ndarray],
    test_view: Dict[str, np.ndarray],
    *,
    n_clients: int,
    alpha: float,
    gamma: float,
    delta_r: float,
    delta_c: float,
    threshold_policy: str = "global",
    allocation_rule: str = "uniform",
    target: str = "simplex",
    rho: float = 0.15,
    mixture_box: Optional[NormalizedBox] = None,
    kappa: float = R3_KAPPA,
    score_name: str = "msp",
) -> VariantResult:
    """Certify one variant cell. See the module docstring for budget composition.

    ``prop_view``/``cert_view``/``test_view`` are dicts with aligned arrays
    ``score``, ``pred``, ``y_open``, ``client``. ``mixture_box`` overrides the
    default ``Lambda_G(rho)`` (used for the traffic-derived box ``Lambda_hat`` of
    WP-D); it is ignored when ``target == 'simplex'``.
    """
    if allocation_rule not in ALLOCATION_RULES:
        raise ValueError(f"unknown allocation rule {allocation_rule!r}")
    if target not in MIXTURE_TARGETS:
        raise ValueError(f"unknown mixture target {target!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if delta_r <= 0.0 or delta_c <= 0.0:
        raise ValueError("delta_r and delta_c must be strictly positive")

    # (1) SELECTOR -- proposal fold only.
    policy = choose_threshold_policy(
        prop_view["score"],
        prop_view["pred"],
        prop_view["y_open"],
        prop_view["client"],
        gamma=gamma,
        alpha=alpha,
        n_clients=n_clients,
        policy=threshold_policy,
    )
    policy_feasible = bool(np.any(policy.feasible))

    # (2) ALLOCATION -- proposal fold only, frozen before the certification fold
    #     is touched. This ordering is required by the proposal-allocated rule.
    A_prop, K_prop, n_prop = policy_counts(
        prop_view["score"],
        prop_view["pred"],
        prop_view["y_open"],
        prop_view["client"],
        policy,
        n_clients,
    )
    n_cert = np.array(
        [int(np.sum(np.asarray(cert_view["client"]) == j)) for j in range(n_clients)]
    )
    A_hat, K_hat = _scaled_proposal_counts(
        A_prop.astype(float),
        K_prop.astype(float),
        n_prop.astype(float),
        n_cert.astype(float),
    )
    eps = allocate(
        allocation_rule,
        A_hat=A_hat,
        K_hat=K_hat,
        J=n_clients,
        delta_r=delta_r,
        alpha=alpha,
        kappa=kappa,
    )

    # (3) CERTIFICATION fold -- supplies counts only.
    A, K, n = policy_counts(
        cert_view["score"],
        cert_view["pred"],
        cert_view["y_open"],
        cert_view["client"],
        policy,
        n_clients,
    )

    # (4) The certificate.
    if target == "simplex":
        eps_risk = eps
        rbar = np.array(
            [
                cp_upper(int(K[j]), int(A[j]), float(eps_risk[j]))
                for j in range(n_clients)
            ]
        )
        U = float(min(rbar.max(), 1.0))
        feasible = True
        risk_solver_status = "closed_form_full_simplex"
        risk_solver_valid = True
        solver_meta: Dict[str, object] = {
            "risk_solver_status": risk_solver_status,
            "risk_solver_certificate_valid": True,
            "risk_solver_tolerance": 0.0,
            "risk_solver_iterations": 0,
        }
    else:
        # Three events per client (rbar, alow, ahigh) -> eps_j / 3 each.
        eps_risk = eps / 3.0
        rbar = np.array(
            [
                cp_upper(int(K[j]), int(A[j]), float(eps_risk[j]))
                for j in range(n_clients)
            ]
        )
        alow = np.array(
            [
                cp_lower(int(A[j]), int(n[j]), float(eps_risk[j]))
                for j in range(n_clients)
            ]
        )
        ahigh = np.array(
            [
                cp_upper(int(A[j]), int(n[j]), float(eps_risk[j]))
                for j in range(n_clients)
            ]
        )
        box = mixture_box if mixture_box is not None else uniform_box(n_clients, rho)
        risk_solver = solve_normalized_box_risk(rbar, alow, ahigh, box)
        U = float(risk_solver.value)
        risk_solver_valid = bool(
            risk_solver.domain_feasible and risk_solver.certificate_valid
        )
        feasible = risk_solver_valid
        risk_solver_status = risk_solver.status
        solver_meta = risk_solver.diagnostics("risk_solver")

    # (5) Corollary 1 coverage LCB at the separate budget delta_c.
    eps_cov = delta_c / n_clients
    L = np.array([cp_lower(int(A[j]), int(n[j]), eps_cov) for j in range(n_clients)])
    if target == "simplex":
        # inf over the full simplex is the worst single client.
        coverage_lcb = float(L.min())
        coverage_solver_valid = True
        coverage_solver_status = "closed_form_full_simplex"
        solver_meta.update(
            {
                "coverage_solver_status": coverage_solver_status,
                "coverage_solver_certificate_valid": True,
                "coverage_solver_tolerance": 0.0,
                "coverage_solver_iterations": 0,
            }
        )
    else:
        box = mixture_box if mixture_box is not None else uniform_box(n_clients, rho)
        coverage_solver = solve_normalized_box_coverage(L, box)
        coverage_lcb = float(coverage_solver.value)
        coverage_solver_valid = bool(
            coverage_solver.domain_feasible and coverage_solver.certificate_valid
        )
        coverage_solver_status = coverage_solver.status
        solver_meta.update(coverage_solver.diagnostics("coverage_solver"))

    solver_valid = bool(risk_solver_valid and coverage_solver_valid)
    feasible = bool(feasible and solver_valid)
    solver_meta.update(
        {
            "solver_status": (
                f"risk:{risk_solver_status};coverage:{coverage_solver_status}"
            ),
            "solver_certificate_valid": solver_valid,
        }
    )

    # (6) Empirical proposal / test diagnostics. Test labels are read HERE ONLY,
    #     after the certificate is fixed, and never feed back into it.
    prop_accept = policy.accept(prop_view["score"], prop_view["client"])
    prop_err = open_set_error(prop_view["pred"], prop_view["y_open"])
    prop_coverage = float(np.mean(prop_accept))
    prop_risk = float(np.mean(prop_err[prop_accept])) if np.any(prop_accept) else 0.0

    test_accept = policy.accept(test_view["score"], test_view["client"])
    test_err = open_set_error(test_view["pred"], test_view["y_open"])
    test_coverage = float(np.mean(test_accept))
    test_risk = float(np.mean(test_err[test_accept])) if np.any(test_accept) else 0.0

    certified = bool(feasible and policy_feasible and U <= alpha and coverage_lcb > 0.0)
    mode = _failure_mode(U if feasible else np.inf, alpha, A, K, policy_feasible)
    if not solver_valid:
        mode = "solver-failure"
    if certified:
        mode = ""
    elif feasible and policy_feasible and U <= alpha and coverage_lcb <= 0.0:
        # Risk certifies but no accepted mass can be guaranteed: not deployable.
        mode = "zero-error-count"

    floors = np.array(
        [zero_error_floor(float(e), alpha) for e in eps_risk], dtype=float
    )

    return VariantResult(
        certified=certified,
        cert_risk_ucb=float(U),
        cert_coverage_lcb=float(coverage_lcb),
        failure_mode=mode,
        frontier=zero_error_frontier(A, alpha),
        rbar=rbar,
        eps=eps,
        floors=floors,
        A=A,
        K=K,
        n=n,
        thresholds=policy.thresholds,
        prop_coverage=prop_coverage,
        prop_risk=prop_risk,
        test_coverage=test_coverage,
        test_risk=test_risk,
        meta={
            "score_name": score_name,
            "gamma": float(gamma),
            "alpha": float(alpha),
            "delta": float(delta_r + delta_c),
            "delta_r": float(delta_r),
            "delta_c": float(delta_c),
            "Lambda": target if target == "simplex" else f"box(rho={rho})",
            "rho": float(rho) if target == "box" else float("nan"),
            "threshold_policy": threshold_policy,
            "allocation_rule": allocation_rule,
            "n_clients": int(n_clients),
            "budget_spent": float(np.sum(eps)),
            "theorem_contract": "legacy-conservative-stratum-allocation",
            **solver_meta,
        },
    )
