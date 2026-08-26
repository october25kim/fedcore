"""Exact full-simplex certificate regression for the Office-Home primary cert.

Primary certificate: client-level FULL SIMPLEX (J=4), alpha=0.20,
delta_r=delta_c=0.05, computed with the exact fedcore certificate core.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedcore.certificate import conditional_risk_certificate
from fedcore.certificate.cp import cp_upper
from fedcore.certificate.suite import certify_variant

J = 4
ALPHA = 0.20
DELTA_R = 0.05
DELTA_C = 0.05


def test_full_simplex_closed_form_is_max_rbar():
    A = [500, 500, 500, 500]
    K = [10, 20, 5, 0]
    eps = DELTA_R
    rbar = [cp_upper(K[j], A[j], eps) for j in range(J)]
    cert = conditional_risk_certificate(A, K, A, DELTA_R, Lambda="simplex")
    assert cert.eps == pytest.approx(eps)
    assert cert.U == pytest.approx(max(rbar))
    assert cert.U <= ALPHA  # certifies at alpha=0.20


def test_zero_accepted_client_is_non_deployable():
    # A client with A_j == 0 yields rbar_j == 1.0 -> U == 1.0 (never dropped).
    cert = conditional_risk_certificate([500, 0, 500, 500], [1, 0, 1, 1], [500, 500, 500, 500], DELTA_R, Lambda="simplex")
    assert cert.U == 1.0
    assert cert.U > ALPHA


def test_high_observed_risk_client_refuses_certificate():
    # K_j / A_j well above alpha on one client dominates the simplex max.
    cert = conditional_risk_certificate([400, 400, 400, 400], [1, 2, 1, 200], [400, 400, 400, 400], DELTA_R, Lambda="simplex")
    assert cert.U > ALPHA


def _view(score, pred, y_open, client):
    return {
        "score": np.asarray(score, dtype=float),
        "pred": np.asarray(pred),
        "y_open": np.asarray(y_open),
        "client": np.asarray(client),
    }


def _three_folds(rng, n_per_client=400, unknown_frac=0.25, known_acc=0.98):
    rows = []
    for _ in range(3):  # prop / cert / test folds, same generating law
        score, pred, y_open, client = [], [], [], []
        for j in range(J):
            for _ in range(n_per_client):
                if rng.random() < unknown_frac:
                    y = -1
                    s = rng.uniform(0.0, 0.4)  # unknowns score low
                    p = int(rng.integers(0, 10))
                else:
                    y = int(rng.integers(0, 10))
                    correct = rng.random() < known_acc
                    p = y if correct else int((y + 1) % 10)
                    s = rng.uniform(0.6, 1.0)  # confident knowns
                score.append(s); pred.append(p); y_open.append(y); client.append(j)
        rows.append(_view(score, pred, y_open, client))
    return rows


def test_certify_variant_simplex_end_to_end_certifies_with_buffer():
    prop_view, cert_view, test_view = _three_folds(np.random.default_rng(0))
    result = certify_variant(
        prop_view, cert_view, test_view,
        n_clients=J, alpha=ALPHA, gamma=0.3, delta_r=DELTA_R, delta_c=DELTA_C,
        threshold_policy="global", allocation_rule="uniform", target="simplex",
        score_name="msp",
    )
    assert result.certified
    assert result.cert_risk_ucb <= ALPHA
    assert result.cert_coverage_lcb > 0.0
    row = result.as_row()
    assert row["Lambda"] == "simplex"
    assert row["n_clients"] == J
    assert row["budget_spent"] == pytest.approx(DELTA_R)  # uniform spends exactly delta_r


def test_risk_buffer_matters_gamma_one_hugs_alpha():
    # gamma=1.0 lets the coverage-maximizing selector accept up to the full risk
    # budget alpha, so the CP upper bound exceeds alpha (CLAUDE.md section 6): the
    # buffered gamma certifies where gamma=1.0 does not.
    rng = np.random.default_rng(0)
    prop_view, cert_view, test_view = _three_folds(rng)
    loose = certify_variant(
        prop_view, cert_view, test_view,
        n_clients=J, alpha=ALPHA, gamma=1.0, delta_r=DELTA_R, delta_c=DELTA_C,
        allocation_rule="uniform", target="simplex",
    )
    tight = certify_variant(
        prop_view, cert_view, test_view,
        n_clients=J, alpha=ALPHA, gamma=0.3, delta_r=DELTA_R, delta_c=DELTA_C,
        allocation_rule="uniform", target="simplex",
    )
    assert loose.cert_risk_ucb > tight.cert_risk_ucb
    assert not loose.certified and tight.certified


def test_historical_suite_uniform_allocation_matches_legacy_union_bound():
    # The archived variant suite intentionally preserves the conservative
    # clientwise union-bound calculation; it is not the current theorem API.
    rng = np.random.default_rng(1)
    n = 300
    score, pred, y_open, client = [], [], [], []
    for j in range(J):
        for _ in range(n):
            y = int(rng.integers(0, 8))
            score.append(rng.uniform(0.7, 1.0)); pred.append(y); y_open.append(y); client.append(j)
    view = _view(score, pred, y_open, client)
    result = certify_variant(
        view, view, view,
        n_clients=J, alpha=ALPHA, gamma=1.0, delta_r=DELTA_R, delta_c=DELTA_C,
        allocation_rule="uniform", target="simplex",
    )
    A, K = result.A, result.K
    eps = DELTA_R / J
    expected = max(cp_upper(int(K[j]), int(A[j]), eps) for j in range(J))
    assert result.cert_risk_ucb == pytest.approx(min(expected, 1.0))
