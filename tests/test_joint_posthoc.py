"""Exact joint certificate and four-policy common-sample tests."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.budget import make_failure_budget  # noqa: E402
from fedcore.certificate.joint import joint_conditional_certificate  # noqa: E402
from fedcore.posthoc import (
    evaluate_four_policies,
    mixture_set_from_traffic,
)  # noqa: E402


def test_zero_coverage_and_vanishing_denominator_are_not_certificates():
    cert = joint_conditional_certificate(
        [0, 0],
        [0, 0],
        [100, 100],
        alpha=0.1,
        risk_eps=[0.01, 0.01],
        acceptance_lower_eps=[0.01, 0.01],
    )
    assert not cert.certified and not cert.feasible
    assert cert.reason in {"zero_accepted_coverage", "vanishing_denominator"}


def _view(seed, n=600):
    rng = np.random.default_rng(seed)
    client = np.repeat(np.arange(3), n // 3)
    y = rng.integers(0, 2, n)
    pred = y.copy()
    pred[rng.random(n) < (0.01 + 0.01 * client)] ^= 1
    score = rng.normal(0.8 - 0.05 * client, 0.12, n)
    return {"score": score, "pred": pred, "y_open": y, "client": client}


def test_four_rows_share_identical_certification_ids_and_complete_budget():
    prop, cert, test = _view(1), _view(2), _view(3)
    ids = np.array([f"fixture:{i}" for i in range(len(cert["score"]))])
    budget = make_failure_budget(
        0.1, include_mixture=False, include_acceptance_box=False
    )
    rows = evaluate_four_policies(
        prop,
        cert,
        test,
        score_name="msp",
        gamma=0.7,
        alpha=0.1,
        budget=budget,
        n_clients=3,
        dirichlet_alpha=0.5,
        Lambda="simplex",
        cert_sample_ids=ids,
    )
    assert len(rows) == 4
    assert len({row["cert_sample_ids_sha256"] for row in rows}) == 1
    assert {(row["threshold_policy"], row["allocation_policy"]) for row in rows} == {
        ("global", "uniform"),
        ("global", "proposal_informed"),
        ("client_specific", "uniform"),
        ("client_specific", "proposal_informed"),
    }
    assert all(np.isclose(row["delta_spent"], 0.1) for row in rows)


def test_bounded_exact_path_runs():
    prop, cert, test = _view(4), _view(5), _view(6)
    budget = make_failure_budget(
        0.1, include_mixture=False, include_acceptance_box=True
    )
    rows = evaluate_four_policies(
        prop,
        cert,
        test,
        score_name="msp",
        gamma=0.5,
        alpha=0.1,
        budget=budget,
        n_clients=3,
        dirichlet_alpha=0.5,
        Lambda="bounded",
        lambda_lower=[0.2, 0.2, 0.2],
        lambda_upper=[0.5, 0.5, 0.5],
    )
    assert len(rows) == 4
    assert all(np.isfinite(row["cert_risk_ucb"]) for row in rows)
    assert all(row["solver_certificate_valid"] for row in rows)
    for row in rows:
        for key in (
            "risk_solver_status",
            "risk_solver_tolerance",
            "risk_solver_iterations",
            "risk_solver_bracket_lower",
            "risk_solver_bracket_upper",
            "risk_solver_residual_lower",
            "risk_solver_residual_upper",
            "coverage_solver_status",
            "coverage_solver_tolerance",
            "coverage_solver_iterations",
            "coverage_solver_bracket_lower",
            "coverage_solver_bracket_upper",
            "coverage_solver_residual_lower",
            "coverage_solver_residual_upper",
        ):
            assert key in row


def test_traffic_lambda_hat_uses_client_identity_counts():
    box = mixture_set_from_traffic(
        [0] * 40 + [1] * 35 + [2] * 25,
        n_clients=3,
        mixture_delta=0.02,
        rho=0.2,
        center=[0.4, 0.35, 0.25],
    )
    assert box.dimension == 3
    assert box.lower.sum() <= 1.0 <= box.upper.sum()


def main():
    test_zero_coverage_and_vanishing_denominator_are_not_certificates()
    test_four_rows_share_identical_certification_ids_and_complete_budget()
    test_bounded_exact_path_runs()
    test_traffic_lambda_hat_uses_client_identity_counts()
    print("joint/posthoc tests: PASS")


if __name__ == "__main__":
    main()
