"""Adversarial tests for the strict-mixture numerical certificate contract.

The tests deliberately check one-sided validity, not merely approximate
equality.  A risk result must dominate an enumerated feasible supremum and a
coverage result must not exceed an enumerated feasible infimum.  Raw primal
witnesses are diagnostics only.
"""

from __future__ import annotations

from decimal import Decimal
import itertools
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fedcore.certificate import conditional_risk_certificate
from fedcore.certificate.joint import joint_conditional_certificate
from fedcore.certificate.lambda_sets import (
    NormalizedBox,
    solve_normalized_box_coverage,
    solve_normalized_box_risk,
    vertex_enumeration_reference,
)
from fedcore.certificate.theorem1 import stratified_certificate
from fedcore.mixture import (
    BoundedSimplex,
    LinearExtremum,
    solve_coverage_infimum,
    solve_robust_ratio,
)


def _normalized_coverage_reference(values, box: NormalizedBox) -> float:
    values = np.asarray(values, dtype=float)
    best = math.inf
    for raw_weights in box.vertices():
        denominator = math.fsum(float(value) for value in raw_weights)
        if denominator <= 0.0:
            continue
        numerator = math.fsum(
            float(weight) * float(value)
            for weight, value in zip(raw_weights, values, strict=True)
        )
        best = min(best, numerator / denominator)
    return best


def _bounded_simplex_vertices(box: BoundedSimplex) -> list[np.ndarray]:
    """Enumerate small bounded-simplex vertices for a test-only oracle."""
    candidates: list[np.ndarray] = []
    dimension = box.dimension
    for free in range(dimension):
        fixed = [index for index in range(dimension) if index != free]
        for endpoints in itertools.product((0, 1), repeat=len(fixed)):
            weights = np.empty(dimension, dtype=float)
            for index, endpoint in zip(fixed, endpoints, strict=True):
                weights[index] = (box.lower[index], box.upper[index])[endpoint]
            weights[free] = 1.0 - math.fsum(
                float(weights[index]) for index in fixed
            )
            if (
                box.lower[free] - 1e-13
                <= weights[free]
                <= box.upper[free] + 1e-13
            ):
                weights[free] = min(
                    box.upper[free], max(box.lower[free], weights[free])
                )
                if not any(
                    np.allclose(weights, old, atol=1e-13, rtol=0.0)
                    for old in candidates
                ):
                    candidates.append(weights.copy())
    assert candidates
    return candidates


def _bounded_risk_reference(risk, alow, ahigh, box: BoundedSimplex) -> float:
    risk = np.asarray(risk, dtype=float)
    best = -math.inf
    for weights in _bounded_simplex_vertices(box):
        for endpoints in itertools.product((0, 1), repeat=box.dimension):
            acceptance = np.array(
                [
                    (alow[index], ahigh[index])[endpoint]
                    for index, endpoint in enumerate(endpoints)
                ],
                dtype=float,
            )
            denominator = math.fsum(
                float(weight) * float(value)
                for weight, value in zip(weights, acceptance, strict=True)
            )
            assert denominator > 0.0
            numerator = math.fsum(
                float(weight) * float(value) * float(rate)
                for weight, value, rate in zip(
                    weights, acceptance, risk, strict=True
                )
            )
            best = max(best, numerator / denominator)
    return best


def _bounded_coverage_reference(values, box: BoundedSimplex) -> float:
    return min(
        math.fsum(
            float(weight) * float(value)
            for weight, value in zip(weights, values, strict=True)
        )
        for weights in _bounded_simplex_vertices(box)
    )


def test_fixed_normalized_risk_rounding_counterexample_stays_conservative():
    box = NormalizedBox(
        [0.16756113660065686, 0.15357721240539815],
        [0.3422644388435838, 0.26005532641630535],
    )
    result = solve_normalized_box_risk(
        [0.9063147462756882, 0.7963915196438551],
        [0.5640456857734877, 0.3552306443303402],
        [0.7475824287789344, 0.9331788371499324],
        box,
    )
    high_precision_reference = Decimal(
        "0.886996473871393294447326218899475854"
    )
    legacy_midpoint = Decimal("0.8869964738713443")
    boundary_alpha = Decimal("0.88699647387136")

    assert legacy_midpoint <= boundary_alpha < high_precision_reference
    assert result.certificate_valid and result.status == "converged"
    assert Decimal.from_float(result.value) >= high_precision_reference
    assert Decimal.from_float(result.value) > boundary_alpha
    assert result.value >= result.bracket_upper
    assert result.witness_value <= result.value
    assert result.residual_lower >= 0.0
    assert result.residual_upper <= 0.0


def test_fixed_normalized_coverage_rounding_counterexample_stays_conservative():
    box = NormalizedBox(
        [0.11443386596784103, 0.1317555599572049, 0.16120370352214453],
        [0.6120103431014071, 0.38241038035620134, 0.28242594406565225],
    )
    result = solve_normalized_box_coverage(
        [0.6383929740324413, 0.3752175413055333, 0.20027068383534147],
        box,
    )
    high_precision_reference = Decimal(
        "0.338719644568293182828441085618478611"
    )
    legacy_midpoint = Decimal("0.3387196445683423")

    assert legacy_midpoint > high_precision_reference
    assert result.certificate_valid and result.status == "converged"
    assert Decimal.from_float(result.value) <= high_precision_reference
    assert result.value <= result.bracket_lower
    assert result.value <= result.witness_value
    assert result.residual_lower >= 0.0
    assert result.residual_upper <= 0.0


def test_normalized_box_random_vertex_stress_is_one_sided_and_deterministic():
    rng = np.random.default_rng(20260825)
    for dimension in range(2, 6):
        for _ in range(40):
            lo = rng.uniform(0.0, 0.2, dimension)
            hi = lo + rng.uniform(0.02, 0.5, dimension)
            box = NormalizedBox(lo, hi)
            risk = rng.uniform(0.0, 1.0, dimension)
            alow = rng.uniform(0.03, 0.7, dimension)
            ahigh = rng.uniform(alow, 1.0)

            first = solve_normalized_box_risk(risk, alow, ahigh, box)
            second = solve_normalized_box_risk(risk, alow, ahigh, box)
            exact_risk = vertex_enumeration_reference(risk, alow, ahigh, box)
            coverage = solve_normalized_box_coverage(alow, box)
            exact_coverage = _normalized_coverage_reference(alow, box)

            assert first.certificate_valid and coverage.certificate_valid
            assert first.value >= exact_risk
            assert coverage.value <= exact_coverage
            assert first.value == second.value
            assert first.bracket_lower == second.bracket_lower
            assert first.bracket_upper == second.bracket_upper


def test_normalized_box_nonconvergence_and_denominator_fail_closed():
    box = NormalizedBox([0.1, 0.1], [0.9, 0.9])
    risk = solve_normalized_box_risk(
        [0.1, 0.9], [0.2, 0.3], [0.8, 0.9], box,
        tolerance=1e-15, max_iterations=1,
    )
    coverage = solve_normalized_box_coverage(
        [0.2, 0.8], box, tolerance=1e-15, max_iterations=1
    )
    denominator = solve_normalized_box_risk(
        [0.1, 0.9], [0.0, 0.0], [0.0, 0.0], box
    )

    assert not risk.certificate_valid and risk.status == "nonconverged"
    assert math.isinf(risk.value)
    assert not coverage.certificate_valid and coverage.status == "nonconverged"
    assert coverage.value == 0.0
    assert not denominator.certificate_valid
    assert denominator.status == "infeasible_no_positive_denominator"
    assert math.isinf(denominator.value)


def test_normalized_box_subnormal_arithmetic_fails_closed():
    risk = solve_normalized_box_risk(
        [0.0, 1.0],
        [2.5e-323, 3.5e-323],
        [2.5e-323, 3.5e-323],
        NormalizedBox([0.1, 0.1], [0.1, 0.1]),
    )
    # The exact ratio of the represented float endpoints is 7/(5+7)=7/12;
    # the retired float products rounded it to 1/2 and falsely returned valid.
    assert not risk.certificate_valid
    assert risk.status == "numerical_failure"
    assert math.isinf(risk.value)

    coverage = solve_normalized_box_coverage(
        [0.0, 0.7],
        NormalizedBox([5e-324, 5e-324], [5e-324, 5e-324]),
    )
    # The normalized singleton has exact coverage 0.35; the retired arithmetic
    # returned about 0.5 because its raw-weight products were subnormal.
    assert not coverage.certificate_valid
    assert coverage.status == "numerical_failure"
    assert coverage.value == 0.0


def test_normalized_coverage_huge_raw_box_overflow_returns_fail_closed_result():
    result = solve_normalized_box_coverage(
        [0.2, 0.8],
        NormalizedBox([1e308, 1e308], [1e308, 1e308]),
    )
    assert not result.certificate_valid
    assert result.status == "numerical_failure"
    assert result.reason == "maximum_denominator_error:OverflowError"
    assert result.value == 0.0


def test_bounded_simplex_random_vertex_stress_is_one_sided():
    rng = np.random.default_rng(1701)
    for dimension in range(2, 5):
        for _ in range(30):
            center = rng.dirichlet(np.ones(dimension))
            radius = float(rng.uniform(0.05, 0.3))
            box = BoundedSimplex(
                np.maximum(0.0, center - radius),
                np.minimum(1.0, center + radius),
            ).tightened()
            risk = rng.uniform(0.0, 1.0, dimension)
            alow = rng.uniform(0.05, 0.7, dimension)
            ahigh = rng.uniform(alow, 1.0)

            risk_result = solve_robust_ratio(
                risk, alow, ahigh, box, tolerance=1e-13
            )
            coverage_result = solve_coverage_infimum(
                alow, box, tolerance=1e-13
            )
            exact_risk = _bounded_risk_reference(risk, alow, ahigh, box)
            exact_coverage = _bounded_coverage_reference(alow, box)

            assert risk_result.certificate_valid and risk_result.feasible
            assert coverage_result.certificate_valid
            assert risk_result.value >= exact_risk
            assert coverage_result.value <= exact_coverage
            assert risk_result.residual_lower >= 0.0
            assert risk_result.residual_upper <= 0.0
            assert coverage_result.residual_lower >= 0.0
            assert coverage_result.residual_upper <= 0.0


def test_bounded_simplex_nonconvergence_fails_closed_in_both_directions():
    box = BoundedSimplex([0.0, 0.0], [1.0, 1.0])
    risk = solve_robust_ratio(
        [0.1, 0.9], [0.2, 0.3], [0.8, 0.9], box,
        tolerance=1e-15, max_iterations=1,
    )
    coverage = solve_coverage_infimum(
        [0.2, 0.8], box, tolerance=1e-15, max_iterations=1
    )

    assert risk.status == "nonconverged" and not risk.certificate_valid
    assert math.isinf(risk.value)
    assert coverage.status == "nonconverged" and not coverage.certificate_valid
    assert coverage.value == 0.0


def test_bounded_simplex_subnormal_arithmetic_fails_closed():
    fixed = BoundedSimplex([0.5, 0.5], [0.5, 0.5])
    risk = solve_robust_ratio(
        [1.0, 0.0], [5e-324, 1e-323], [5e-324, 1e-323], fixed
    )
    # Exact represented-float ratio is 1/(1+2)=1/3. The retired solver rounded
    # both witness and residual arithmetic toward zero and returned ~9e-13.
    assert not risk.certificate_valid
    assert risk.status == "numerical_failure"
    assert math.isinf(risk.value)

    coverage = solve_coverage_infimum([5e-324, 1e-323], fixed)
    assert not coverage.certificate_valid
    assert coverage.status == "numerical_failure"
    assert coverage.value == 0.0


def test_bounded_coverage_numerical_residual_failure_fails_closed():
    class CorruptPrimalObjective(BoundedSimplex):
        def minimize(self, coefficients):  # noqa: D102 - deliberate fault injector
            return LinearExtremum(
                value=0.9, weights=np.array([0.5, 0.5]), sense="min"
            )

    result = solve_coverage_infimum(
        [0.2, 0.8], CorruptPrimalObjective([0.0, 0.0], [1.0, 1.0])
    )
    assert not result.certificate_valid
    assert result.status == "numerical_residual_failure"
    assert result.reason == "greedy_solution_validation_failed"
    assert result.value == 0.0


def test_certificate_apis_propagate_solver_evidence_and_fail_closed():
    counts = dict(A=[120, 95, 140], K=[3, 2, 4], n=[300, 280, 320])
    normalized = conditional_risk_certificate(
        **counts, delta=0.05, Lambda="box", box=0.15
    )
    singleton = conditional_risk_certificate(
        **counts, delta=0.05, Lambda="known", lam=[0.2, 0.3, 0.5]
    )
    for result in (normalized, singleton):
        assert result.feasible and result.solver_certificate_valid
        assert result.solver_status == "converged"
        assert result.solver_iterations > 0
        assert result.solver_bracket_lower <= result.solver_bracket_upper
        assert result.solver_residual_lower >= 0.0
        assert result.solver_residual_upper <= 0.0

    joint = joint_conditional_certificate(
        **counts,
        alpha=0.2,
        risk_eps=[0.005] * 3,
        acceptance_lower_eps=[0.005] * 3,
        acceptance_upper_eps=[0.005] * 3,
        lambda_lower=[0.1, 0.1, 0.1],
        lambda_upper=[0.8, 0.8, 0.8],
    )
    assert joint.solver_certificate_valid
    assert "risk_solver_bracket_upper" in joint.solver_diagnostics
    assert "coverage_solver_bracket_lower" in joint.solver_diagnostics
    assert joint.lambda_star is not None
    assert math.isclose(math.fsum(joint.lambda_star), 1.0, abs_tol=1e-12)

    empty = joint_conditional_certificate(
        **counts,
        alpha=0.2,
        risk_eps=[0.005] * 3,
        acceptance_lower_eps=[0.005] * 3,
        acceptance_upper_eps=[0.005] * 3,
        lambda_lower=[0.5, 0.5, 0.5],
        lambda_upper=[0.8, 0.8, 0.8],
    )
    assert not empty.solver_certificate_valid and not empty.certified
    assert math.isinf(empty.risk_ucb) and empty.coverage_lcb == 0.0
    assert empty.reason == "infeasible_mixture_set"


def test_officehome_canonical_summary_combines_risk_and_coverage_validity():
    from fedcore.experiments import officehome_selector_rescue as rescue

    risk_certificate = SimpleNamespace(
        risk_ucb=0.08,
        coverage_lcb=0.2,
        certified=True,
        feasible=True,
        solver_diagnostics={
            "risk_solver_status": "converged",
            "risk_solver_certificate_valid": True,
            "coverage_solver_status": "converged",
            "coverage_solver_certificate_valid": True,
        },
    )
    failed_coverage_certificate = SimpleNamespace(
        risk_ucb=0.08,
        coverage_lcb=0.0,
        certified=False,
        feasible=True,
        solver_diagnostics={
            "risk_solver_status": "converged",
            "risk_solver_certificate_valid": True,
            "coverage_solver_status": "numerical_failure",
            "coverage_solver_certificate_valid": False,
        },
    )
    counts = SimpleNamespace(
        A=np.array([100, 100, 100, 100]),
        K=np.array([1, 1, 1, 1]),
        n=np.array([200, 200, 200, 200]),
    )
    with patch.object(
        rescue,
        "joint_conditional_certificate",
        side_effect=[risk_certificate, failed_coverage_certificate],
    ):
        summary = rescue.certify_box_canonical(
            counts,
            0.1,
            [0.1, 0.1, 0.1, 0.1],
            [0.7, 0.7, 0.7, 0.7],
            delta_r=0.04,
            delta_c=0.04,
        )

    assert not summary.feasible and not summary.certified
    assert not summary.solver_diagnostics["solver_certificate_valid"]
    assert "coverage:numerical_failure" in summary.solver_diagnostics["solver_status"]
    # Historical four-value unpacking remains source-compatible.
    assert tuple(summary) == (0.08, 0.0, False, False)


def test_historical_sampled_box_baseline_is_retired_fail_closed():
    try:
        stratified_certificate(
            [20, 20], [1, 1], [50, 50], 0.05, Lambda="box", box=0.1
        )
    except RuntimeError as exc:
        assert "retired sampled" in str(exc)
    else:  # pragma: no cover - explicit failure message without pytest dependency
        raise AssertionError("sampled legacy box unexpectedly returned a certificate")
