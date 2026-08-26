"""Artifact-free tests for the theorem-aligned simple family branch."""

from __future__ import annotations

import numpy as np

from fedcore.certificate import (
    select_simple_family_member,
    simple_simultaneous_family_certificate,
    zero_error_count_threshold,
)


def _counts():
    A = np.array([[100, 110, 120], [80, 90, 100]])
    K = np.array([[0, 0, 0], [1, 2, 1]])
    n = np.array([200, 210, 220])
    return A, K, n


def test_full_simplex_family_divides_by_M_only() -> None:
    A, K, n = _counts()
    result = simple_simultaneous_family_certificate(
        A, K, n, alpha=0.20, delta_r=0.05, delta_c=0.05
    )
    assert result.risk_tail == 0.05 / 2
    assert result.coverage_tail == 0.05 / 2
    assert result.risk_acceptance_tail is None
    assert result.S == 3 and result.M == 2


def test_strict_bounded_family_uses_all_simultaneous_factors() -> None:
    A, K, n = _counts()
    result = simple_simultaneous_family_certificate(
        A,
        K,
        n,
        alpha=0.20,
        delta_r=0.06,
        delta_c=0.04,
        mixture_target="bounded",
        lambda_lower=[0.10, 0.10, 0.10],
        lambda_upper=[0.80, 0.80, 0.80],
    )
    assert result.risk_tail == 0.06 / (3 * 3 * 2)
    assert result.risk_acceptance_tail == result.risk_tail
    assert result.coverage_tail == 0.04 / (3 * 2)
    assert all(member.solver_certificate_valid for member in result.members)
    assert np.all(np.isfinite(result.risk_ucb))


def test_infeasible_bounded_configuration_fails_closed() -> None:
    A, K, n = _counts()
    result = simple_simultaneous_family_certificate(
        A,
        K,
        n,
        alpha=0.20,
        delta_r=0.05,
        delta_c=0.05,
        mixture_target="bounded",
        lambda_lower=[0.8, 0.8, 0.8],
        lambda_upper=[0.9, 0.9, 0.9],
    )
    assert np.all(np.isinf(result.risk_ucb))
    assert np.all(result.coverage_lcb == 0.0)
    assert not np.any(result.certified)
    assert all(not member.solver_certificate_valid for member in result.members)


def test_selection_uses_coverage_then_frozen_order() -> None:
    A, K, n = _counts()
    result = simple_simultaneous_family_certificate(
        A, K, n, alpha=0.20, delta_r=0.05, delta_c=0.05
    )
    expected = min(
        np.flatnonzero(result.certified),
        key=lambda index: -float(result.coverage_lcb[index]),
    )
    assert select_simple_family_member(result) == int(expected)


def test_zero_error_release_floors() -> None:
    from fedcore.certificate.allocation import zero_error_floor

    assert zero_error_count_threshold(0.20, 0.05) == 14
    assert zero_error_floor(0.05, 0.20) == 14
    assert zero_error_floor(0.05 / 5, 0.20) == 21
    assert zero_error_floor(0.05 / 12, 0.20) == 25
    assert zero_error_floor(0.05 / 60, 0.20) == 32


def test_public_holm_api_is_decision_only_and_full_simplex_only() -> None:
    from fedcore.certificate.holm import holm_family_certificate

    A, K, n = _counts()
    result = holm_family_certificate(
        A, K, n, alpha=0.20, delta_r=0.05, delta_c=0.05
    )
    assert result.risk_ucb is None
    assert result.eps_c == 0.05 / 2
    assert np.all(result.adjusted_pvalues >= result.pvalues)

    import pytest

    with pytest.raises(ValueError, match="full-simplex"):
        holm_family_certificate(
            A,
            K,
            n,
            alpha=0.20,
            delta_r=0.05,
            delta_c=0.05,
            mixture_target="bounded",
        )
