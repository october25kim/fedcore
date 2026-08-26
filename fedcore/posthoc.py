"""One-shot post-hoc evaluation on immutable scored observations."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Optional, Sequence

import numpy as np

from fedcore.budget import FailureBudget, allocate_failure_budget
from fedcore.certificate.joint import joint_conditional_certificate
from fedcore.policy import choose_threshold_policy, policy_counts, policy_risk_coverage
from fedcore.mixture import (
    BoundedSimplex,
    intersect_mixture_boxes,
    rho_mixture_box,
    traffic_mixture_confidence_box,
)


def _ids_digest(sample_ids: Optional[Sequence[object]]) -> str:
    if sample_ids is None:
        return ""
    h = hashlib.sha256()
    for value in np.asarray(sample_ids).astype(str):
        encoded = value.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
    return h.hexdigest()


def evaluate_four_policies(
    prop_view: Mapping[str, np.ndarray],
    cert_view: Mapping[str, np.ndarray],
    test_view: Mapping[str, np.ndarray],
    *,
    score_name: str,
    gamma: float,
    alpha: float,
    budget: FailureBudget,
    n_clients: int,
    dirichlet_alpha: float,
    Lambda: str,
    lambda_lower: Optional[Sequence[float]] = None,
    lambda_upper: Optional[Sequence[float]] = None,
    cert_sample_ids: Optional[Sequence[object]] = None,
) -> list[dict[str, object]]:
    """Evaluate global/client thresholds x uniform/informed allocations.

    All returned rows echo one certification-ID digest. Thresholds and confidence
    allocations are frozen from proposal observations before certification counts
    are computed.
    """

    if gamma not in {0.5, 0.7, 1.0}:
        raise ValueError("gamma must be one of {0.5, 0.7, 1.0}")
    if Lambda not in {"simplex", "bounded"}:
        raise ValueError("Lambda must be 'simplex' or 'bounded'")
    if Lambda == "bounded" and budget.acceptance_upper <= 0.0:
        raise ValueError("bounded Lambda requires an acceptance_upper budget")
    common_digest = _ids_digest(cert_sample_ids)
    if cert_sample_ids is not None and len(cert_sample_ids) != len(cert_view["score"]):
        raise ValueError("cert_sample_ids must align with certification observations")

    rows = []
    for threshold_name in ("global", "client_specific"):
        threshold = choose_threshold_policy(
            prop_view["score"],
            prop_view["pred"],
            prop_view["y_open"],
            prop_view["client"],
            gamma=gamma,
            alpha=alpha,
            n_clients=n_clients,
            policy=threshold_name,
        )
        Ap, Kp, _ = policy_counts(
            prop_view["score"],
            prop_view["pred"],
            prop_view["y_open"],
            prop_view["client"],
            threshold,
            n_clients,
        )
        prop_coverage, prop_risk = policy_risk_coverage(
            prop_view["score"],
            prop_view["pred"],
            prop_view["y_open"],
            prop_view["client"],
            threshold,
        )
        for allocation_name in ("uniform", "proposal_informed"):
            allocations = allocate_failure_budget(
                budget, Ap, Kp, policy=allocation_name
            )
            A, K, n = policy_counts(
                cert_view["score"],
                cert_view["pred"],
                cert_view["y_open"],
                cert_view["client"],
                threshold,
                n_clients,
            )
            cert = joint_conditional_certificate(
                A,
                K,
                n,
                alpha=alpha,
                risk_eps=allocations["conditional_risk"],
                acceptance_lower_eps=allocations["acceptance_lower"],
                acceptance_upper_eps=allocations.get("acceptance_upper"),
                lambda_lower=(lambda_lower if Lambda == "bounded" else None),
                lambda_upper=(lambda_upper if Lambda == "bounded" else None),
            )
            test_coverage, test_risk = policy_risk_coverage(
                test_view["score"],
                test_view["pred"],
                test_view["y_open"],
                test_view["client"],
                threshold,
            )
            proposal_feasible = (
                bool(np.all(threshold.feasible))
                if threshold.client_specific
                else bool(threshold.feasible[0])
            )
            rows.append(
                {
                    "score_name": score_name,
                    "gamma": gamma,
                    "alpha": alpha,
                    "delta": budget.total,
                    "Lambda": Lambda,
                    "dirichlet_alpha": dirichlet_alpha,
                    "n_clients": n_clients,
                    "certified": bool(proposal_feasible and cert.certified),
                    "cert_risk_ucb": cert.risk_ucb,
                    "cert_coverage_lcb": cert.coverage_lcb,
                    "cert_n": int(A.sum()),
                    "cert_k": int(K.sum()),
                    "prop_coverage": prop_coverage,
                    "prop_risk": prop_risk,
                    "test_coverage": test_coverage,
                    "test_risk": test_risk,
                    "threshold_policy": threshold_name,
                    "allocation_policy": allocation_name,
                    "proposal_feasible": proposal_feasible,
                    "certificate_feasible": cert.feasible,
                    "certificate_reason": cert.reason,
                    "solver_status": cert.solver_status,
                    "solver_certificate_valid": cert.solver_certificate_valid,
                    "thresholds_json": json.dumps(
                        threshold.thresholds.tolist(), separators=(",", ":")
                    ),
                    "risk_eps_json": json.dumps(
                        cert.risk_eps.tolist(), separators=(",", ":")
                    ),
                    "acceptance_lower_eps_json": json.dumps(
                        cert.acceptance_lower_eps.tolist(), separators=(",", ":")
                    ),
                    "acceptance_upper_eps_json": json.dumps(
                        (
                            cert.acceptance_upper_eps.tolist()
                            if cert.acceptance_upper_eps is not None
                            else []
                        ),
                        separators=(",", ":"),
                    ),
                    "lambda_lower_json": json.dumps(
                        cert.lambda_lower.tolist(), separators=(",", ":")
                    ),
                    "lambda_upper_json": json.dumps(
                        cert.lambda_upper.tolist(), separators=(",", ":")
                    ),
                    "cert_sample_ids_sha256": common_digest,
                    **cert.solver_diagnostics,
                    **budget.as_dict(),
                }
            )
    return rows


def mixture_set_from_traffic(
    traffic_client_ids: Sequence[int],
    *,
    n_clients: int,
    mixture_delta: float,
    rho: Optional[float] = None,
    center: Optional[Sequence[float]] = None,
) -> BoundedSimplex:
    """Build ``Lambda_hat`` from traffic identities, optionally intersecting rho."""
    clients = np.asarray(traffic_client_ids, dtype=int)
    if clients.ndim != 1 or np.any(clients < 0) or np.any(clients >= n_clients):
        raise ValueError("traffic client identities are outside the client roster")
    counts = np.bincount(clients, minlength=n_clients)
    traffic = traffic_mixture_confidence_box(counts, mixture_delta).mixture
    if rho is None:
        return traffic
    if center is None:
        center = (
            counts / counts.sum()
            if counts.sum()
            else np.full(n_clients, 1.0 / n_clients)
        )
    return intersect_mixture_boxes(traffic, rho_mixture_box(center, rho))
