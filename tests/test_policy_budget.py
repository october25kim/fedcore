"""Regression tests for proposal-only policies and complete failure budgets."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.budget import (  # noqa: E402
    FailureBudget,
    allocate_failure_budget,
    make_failure_budget,
)
from fedcore.policy import choose_threshold_policy, policy_counts  # noqa: E402


def _proposal():
    rng = np.random.default_rng(7)
    n = 240
    client = np.repeat(np.arange(3), n // 3)
    y = rng.integers(0, 2, size=n)
    pred = y.copy()
    pred[rng.random(n) < np.choose(client, [0.03, 0.08, 0.15])] ^= 1
    score = rng.normal(1.0 - 0.15 * client, 0.2, size=n)
    return score, pred, y, client


def test_budget_sum_and_reuse_contract():
    b = make_failure_budget(0.10, include_mixture=True, include_acceptance_box=True)
    assert np.isclose(b.spent, 0.10)
    assert np.isclose(b.slack, 0.0)
    try:
        FailureBudget(0.1, mixture=0.04, conditional_risk=0.04, acceptance_lower=0.04)
    except ValueError:
        pass
    else:
        raise AssertionError("overspent failure budget was accepted")


def test_four_policies_are_proposal_only_and_deterministic():
    score, pred, y, client = _proposal()
    frozen = {}
    for threshold_name in ("global", "client_specific"):
        p = choose_threshold_policy(
            score,
            pred,
            y,
            client,
            gamma=0.7,
            alpha=0.10,
            n_clients=3,
            policy=threshold_name,
        )
        Ap, Kp, _ = policy_counts(score, pred, y, client, p, 3)
        for allocation_name in ("uniform", "proposal_informed"):
            b = make_failure_budget(
                0.10, include_mixture=False, include_acceptance_box=True
            )
            alloc = allocate_failure_budget(b, Ap, Kp, policy=allocation_name)
            key = (threshold_name, allocation_name)
            frozen[key] = (p.thresholds.copy(), {k: v.copy() for k, v in alloc.items()})

    # Arbitrarily changing would-be certification labels cannot affect objects that
    # have already been frozen from the proposal fold.
    cert_y_a = y.copy()
    cert_y_b = np.full_like(y, -1)
    assert not np.array_equal(cert_y_a, cert_y_b)
    for key, (thresholds, alloc) in frozen.items():
        threshold_name, allocation_name = key
        p2 = choose_threshold_policy(
            score,
            pred,
            y,
            client,
            gamma=0.7,
            alpha=0.10,
            n_clients=3,
            policy=threshold_name,
        )
        Ap, Kp, _ = policy_counts(score, pred, y, client, p2, 3)
        b = make_failure_budget(
            0.10, include_mixture=False, include_acceptance_box=True
        )
        alloc2 = allocate_failure_budget(b, Ap, Kp, policy=allocation_name)
        assert np.array_equal(thresholds, p2.thresholds)
        for component in alloc:
            assert np.array_equal(alloc[component], alloc2[component])
            assert np.isclose(alloc2[component].sum(), getattr(b, component))


def test_missing_client_is_explicitly_non_deployable():
    score, pred, y, client = _proposal()
    keep = client != 2
    p = choose_threshold_policy(
        score[keep],
        pred[keep],
        y[keep],
        client[keep],
        gamma=0.5,
        alpha=0.1,
        n_clients=3,
        policy="client_specific",
    )
    assert not p.feasible[2]
    assert np.isinf(p.thresholds[2])
    assert not p.accept(np.array([10.0]), np.array([2]))[0]


def main():
    test_budget_sum_and_reuse_contract()
    test_four_policies_are_proposal_only_and_deterministic()
    test_missing_client_is_explicitly_non_deployable()
    print("policy/budget tests: PASS")


if __name__ == "__main__":
    main()
