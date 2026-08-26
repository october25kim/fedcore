"""Reference tests for exact bounded-simplex mixture optimization.

The file is both pytest-compatible and directly runnable:

``python tests/test_mixture.py``

The reference solvers intentionally enumerate all small-dimensional vertices.
Production code in :mod:`fedcore.mixture` does not use enumeration or random
mixture samples.
"""

from __future__ import annotations

import inspect
import itertools
import math
import os
import sys
from typing import Callable, Iterable

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.mixture import (  # noqa: E402
    BoundedSimplex,
    coverage_infimum,
    intersect_mixture_boxes,
    maximize_linear,
    minimize_linear,
    robust_ratio_supremum,
    rho_mixture_box,
    solve_coverage_infimum,
    solve_robust_ratio,
    traffic_mixture_confidence_box,
)


def _assert_raises(exception: type[BaseException], function: Callable, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return
    except Exception as exc:  # pragma: no cover - produces a useful script failure
        raise AssertionError(
            f"expected {exception.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exception.__name__}, but no exception was raised")


def _bounded_simplex_vertices(lower: Iterable[float], upper: Iterable[float]):
    """Enumerate vertices by leaving each coordinate free in turn."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    dimension = len(lower)
    candidates = []
    for free in range(dimension):
        fixed = [index for index in range(dimension) if index != free]
        for endpoints in itertools.product((0, 1), repeat=len(fixed)):
            weights = np.empty(dimension, dtype=float)
            for index, endpoint in zip(fixed, endpoints, strict=True):
                weights[index] = (lower[index], upper[index])[endpoint]
            weights[free] = 1.0 - math.fsum(float(weights[index]) for index in fixed)
            if lower[free] - 1e-13 <= weights[free] <= upper[free] + 1e-13:
                weights[free] = min(upper[free], max(lower[free], weights[free]))
                if not any(
                    np.allclose(weights, old, atol=1e-13, rtol=0.0)
                    for old in candidates
                ):
                    candidates.append(weights.copy())
    if not candidates:
        raise AssertionError("reference vertex enumerator found no feasible point")
    return candidates


def _ratio_vertex_supremum(risk, alow, ahigh, lambda_lower, lambda_upper):
    """Exponential reference for the joint linear-fractional problem."""
    risk = np.asarray(risk, dtype=float)
    vertices = _bounded_simplex_vertices(lambda_lower, lambda_upper)
    best = -math.inf
    best_pair = None
    for weights in vertices:
        for endpoints in itertools.product((0, 1), repeat=len(risk)):
            acceptance = np.array(
                [
                    (alow[index], ahigh[index])[endpoint]
                    for index, endpoint in enumerate(endpoints)
                ],
                dtype=float,
            )
            denominator = float(np.dot(weights, acceptance))
            if denominator <= 0.0:
                return math.inf, None
            value = float(np.dot(weights * acceptance, risk) / denominator)
            if value > best:
                best = value
                best_pair = (weights.copy(), acceptance.copy())
    return best, best_pair


def test_bounded_simplex_validation_and_tightening():
    box = BoundedSimplex([0.1, 0.4, 0.0], [0.6, 0.7, 0.8])
    tight = box.tightened()
    np.testing.assert_allclose(tight.lower, [0.1, 0.4, 0.0], atol=1e-15)
    np.testing.assert_allclose(tight.upper, [0.6, 0.7, 0.5], atol=1e-15)
    assert math.fsum(tight.lower) <= 1.0 <= math.fsum(tight.upper)

    _assert_raises(ValueError, BoundedSimplex, [], [])
    _assert_raises(ValueError, BoundedSimplex, [0.0, 0.0], [1.0])
    _assert_raises(ValueError, BoundedSimplex, [-0.1, 0.2], [0.8, 1.0])
    _assert_raises(ValueError, BoundedSimplex, [0.4, 0.7], [0.6, 0.9])
    _assert_raises(ValueError, BoundedSimplex, [0.0, 0.0], [0.4, 0.5])
    _assert_raises(ValueError, BoundedSimplex, [0.0, np.nan], [1.0, 1.0])


def test_linear_greedy_matches_complete_vertex_enumeration():
    lower = np.array([0.05, 0.10, 0.15, 0.00])
    upper = np.array([0.55, 0.45, 0.65, 0.40])
    vertices = _bounded_simplex_vertices(lower, upper)
    rng = np.random.default_rng(1729)
    for _ in range(50):
        coefficients = rng.normal(size=4)
        reference_values = np.array(
            [np.dot(coefficients, weights) for weights in vertices]
        )
        maximum = maximize_linear(coefficients, lower, upper)
        minimum = minimize_linear(coefficients, lower, upper)
        np.testing.assert_allclose(maximum.value, reference_values.max(), atol=2e-14)
        np.testing.assert_allclose(minimum.value, reference_values.min(), atol=2e-14)
        np.testing.assert_allclose(maximum.weights.sum(), 1.0, atol=2e-15)
        np.testing.assert_allclose(minimum.weights.sum(), 1.0, atol=2e-15)
        assert np.all(maximum.weights >= lower) and np.all(maximum.weights <= upper)
        assert np.all(minimum.weights >= lower) and np.all(minimum.weights <= upper)


def test_linear_ties_are_stable_and_deterministic():
    box = BoundedSimplex([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    first = box.maximize([1.0, 1.0, 1.0])
    second = box.maximize([1.0, 1.0, 1.0])
    np.testing.assert_array_equal(first.weights, [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.value == second.value == 1.0


def test_robust_ratio_matches_joint_vertex_enumeration():
    lambda_lower = np.array([0.05, 0.10, 0.15, 0.00])
    lambda_upper = np.array([0.55, 0.45, 0.65, 0.40])
    risk = np.array([0.08, 0.31, 0.16, 0.24])
    alow = np.array([0.15, 0.30, 0.10, 0.25])
    ahigh = np.array([0.80, 0.65, 0.90, 0.55])
    expected, _ = _ratio_vertex_supremum(risk, alow, ahigh, lambda_lower, lambda_upper)

    result = solve_robust_ratio(
        risk,
        alow,
        ahigh,
        BoundedSimplex(lambda_lower, lambda_upper),
        tolerance=1e-13,
    )
    assert result.feasible
    assert result.reason is None
    assert result.attained_value <= expected + 2e-14
    assert result.value >= expected - 2e-14
    np.testing.assert_allclose(result.value, expected, atol=2e-13, rtol=0.0)
    np.testing.assert_allclose(result.attained_value, expected, atol=2e-13, rtol=0.0)
    assert result.optimality_gap <= 2e-13
    assert result.lambda_star is not None and result.acceptance_star is not None
    attained = np.dot(result.lambda_star * result.acceptance_star, risk) / np.dot(
        result.lambda_star, result.acceptance_star
    )
    np.testing.assert_allclose(attained, result.attained_value, atol=2e-14)

    scalar = robust_ratio_supremum(
        risk, alow, ahigh, lambda_lower, lambda_upper, tolerance=1e-13
    )
    assert scalar == result.value


def test_robust_ratio_random_reference_cells_and_determinism():
    rng = np.random.default_rng(314159)
    lambda_lower = np.array([0.05, 0.10, 0.00])
    lambda_upper = np.array([0.75, 0.65, 0.80])
    box = BoundedSimplex(lambda_lower, lambda_upper)
    for _ in range(20):
        risk = rng.uniform(0.0, 1.0, size=3)
        alow = rng.uniform(0.05, 0.5, size=3)
        ahigh = rng.uniform(alow, 1.0)
        expected, _ = _ratio_vertex_supremum(
            risk, alow, ahigh, lambda_lower, lambda_upper
        )
        first = solve_robust_ratio(risk, alow, ahigh, box, tolerance=1e-13)
        second = solve_robust_ratio(risk, alow, ahigh, box, tolerance=1e-13)
        np.testing.assert_allclose(first.value, expected, atol=2e-13, rtol=0.0)
        np.testing.assert_allclose(first.attained_value, expected, atol=2e-13, rtol=0.0)
        assert first.value == second.value
        assert first.attained_value == second.attained_value
        np.testing.assert_array_equal(first.lambda_star, second.lambda_star)
        np.testing.assert_array_equal(first.acceptance_star, second.acceptance_star)


def test_robust_ratio_full_simplex_and_fixed_mixture_special_cases():
    risk = np.array([0.1, 0.4, 0.2])
    alow = np.array([0.2, 0.3, 0.4])
    ahigh = np.array([0.8, 0.9, 0.7])
    full = solve_robust_ratio(
        risk, alow, ahigh, BoundedSimplex(np.zeros(3), np.ones(3))
    )
    np.testing.assert_allclose(full.value, risk.max(), atol=2e-12)

    weights = np.array([0.2, 0.5, 0.3])
    fixed = solve_robust_ratio(
        risk, alow, ahigh, BoundedSimplex(weights, weights), tolerance=1e-13
    )
    expected, _ = _ratio_vertex_supremum(risk, alow, ahigh, weights, weights)
    np.testing.assert_allclose(fixed.value, expected, atol=2e-13)


def test_positive_tiny_denominator_and_nearly_degenerate_box_remain_feasible():
    lambda_lower = np.array([0.4999999999989, 0.5000000000009])
    lambda_upper = np.array([0.4999999999991, 0.5000000000011])
    risk = np.array([0.0, 1.0])
    alow = np.array([1e-14, 3e-14])
    ahigh = np.array([2e-14, 8e-14])
    expected, _ = _ratio_vertex_supremum(risk, alow, ahigh, lambda_lower, lambda_upper)
    result = solve_robust_ratio(
        risk,
        alow,
        ahigh,
        BoundedSimplex(lambda_lower, lambda_upper),
        tolerance=1e-13,
    )
    assert result.feasible
    assert result.min_denominator > 0.0
    assert result.attained_value <= expected + 2e-14
    assert result.value >= expected - 2e-14
    np.testing.assert_allclose(result.value, expected, atol=2e-13, rtol=0.0)


def test_vanishing_denominator_is_explicitly_infeasible():
    result = solve_robust_ratio(
        [0.1, 0.2],
        [0.0, 0.3],
        [0.8, 0.9],
        BoundedSimplex([0.0, 0.0], [1.0, 1.0]),
    )
    assert not result.feasible
    assert result.reason == "vanishing_denominator"
    assert result.min_denominator == 0.0
    assert math.isinf(result.value)
    assert result.lambda_star is None and result.acceptance_star is None
    assert math.isinf(
        robust_ratio_supremum(
            [0.1, 0.2], [0.0, 0.3], [0.8, 0.9], [0.0, 0.0], [1.0, 1.0]
        )
    )


def test_solver_validation_and_nonconvergence_are_loud():
    box = BoundedSimplex([0.0, 0.0], [1.0, 1.0])
    _assert_raises(ValueError, solve_robust_ratio, [0.1], [0.2, 0.2], [0.8, 0.8], box)
    _assert_raises(
        ValueError,
        solve_robust_ratio,
        [0.1, 1.2],
        [0.2, 0.2],
        [0.8, 0.8],
        box,
    )
    _assert_raises(
        ValueError,
        solve_robust_ratio,
        [0.1, 0.2],
        [0.5, 0.2],
        [0.4, 0.8],
        box,
    )
    _assert_raises(
        ValueError,
        solve_robust_ratio,
        [0.1, 0.2],
        [0.2, 0.2],
        [0.8, 0.8],
        box,
        tolerance=0.0,
    )
    nonconverged = solve_robust_ratio(
        [0.1, 0.2],
        [0.2, 0.2],
        [0.8, 0.8],
        box,
        tolerance=1e-15,
        max_iterations=1,
    )
    assert not nonconverged.feasible
    assert not nonconverged.certificate_valid
    assert nonconverged.status == "nonconverged"
    assert math.isinf(nonconverged.value)


def test_coverage_infimum_matches_vertex_enumeration():
    lower = np.array([0.05, 0.10, 0.15, 0.00])
    upper = np.array([0.55, 0.45, 0.65, 0.40])
    acceptance_lower = np.array([0.7, 0.2, 0.5, 0.4])
    expected = min(
        float(np.dot(weights, acceptance_lower))
        for weights in _bounded_simplex_vertices(lower, upper)
    )
    result = solve_coverage_infimum(
        acceptance_lower, BoundedSimplex(lower, upper), tolerance=1e-13
    )
    assert result.certificate_valid
    assert result.value <= expected
    assert result.value <= result.raw_value
    assert result.residual_lower >= 0.0 and result.residual_upper <= 0.0
    np.testing.assert_allclose(result.value, expected, atol=2e-13)
    np.testing.assert_allclose(
        coverage_infimum(
            acceptance_lower, lower, upper, tolerance=1e-13
        ),
        expected,
        atol=2e-13,
    )


def test_rho_box_and_intersection_are_exact():
    rho = rho_mixture_box([0.2, 0.3, 0.5], 0.1)
    np.testing.assert_allclose(rho.lower, [0.1, 0.2, 0.4], atol=1e-15)
    np.testing.assert_allclose(rho.upper, [0.3, 0.4, 0.6], atol=1e-15)
    traffic = BoundedSimplex([0.15, 0.1, 0.35], [0.4, 0.35, 0.7])
    both = intersect_mixture_boxes(rho, traffic)
    for vertex in _bounded_simplex_vertices(both.lower, both.upper):
        assert np.all(vertex >= rho.lower) and np.all(vertex <= rho.upper)
        assert np.all(vertex >= traffic.lower) and np.all(vertex <= traffic.upper)
    _assert_raises(ValueError, rho_mixture_box, [0.2, 0.2], 0.1)
    _assert_raises(ValueError, intersect_mixture_boxes)


def test_traffic_box_is_bonferroni_cp_and_label_free():
    counts = np.array([50, 30, 20])
    delta = 0.1
    result = traffic_mixture_confidence_box(counts, delta)
    epsilon = delta / (2 * len(counts))
    expected_lower = np.array(
        [beta.ppf(epsilon, count, counts.sum() - count + 1) for count in counts]
    )
    expected_upper = np.array(
        [beta.ppf(1 - epsilon, count + 1, counts.sum() - count) for count in counts]
    )
    np.testing.assert_allclose(result.raw_lower, expected_lower, atol=1e-15)
    np.testing.assert_allclose(result.raw_upper, expected_upper, atol=1e-15)
    assert result.total == 100
    assert result.tail_probability == epsilon
    assert math.fsum(result.lower) <= 1.0 <= math.fsum(result.upper)
    assert np.all(result.lower >= result.raw_lower - 2e-15)
    assert np.all(result.upper <= result.raw_upper + 2e-15)
    empirical = counts / counts.sum()
    assert np.all(empirical >= result.lower) and np.all(empirical <= result.upper)

    parameters = inspect.signature(traffic_mixture_confidence_box).parameters
    assert tuple(parameters) == ("client_counts", "delta")
    assert "labels" not in parameters and "y" not in parameters
    _assert_raises(
        TypeError,
        traffic_mixture_confidence_box,
        counts,
        delta,
        labels=np.array([0, 1, 0]),
    )


def test_traffic_box_zero_sample_single_client_and_bad_counts():
    empty_traffic = traffic_mixture_confidence_box([0, 0, 0], 0.1)
    np.testing.assert_array_equal(empty_traffic.lower, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(empty_traffic.upper, [1.0, 1.0, 1.0])

    single = traffic_mixture_confidence_box([0], 0.1)
    np.testing.assert_array_equal(single.lower, [1.0])
    np.testing.assert_array_equal(single.upper, [1.0])

    _assert_raises(ValueError, traffic_mixture_confidence_box, [], 0.1)
    _assert_raises(ValueError, traffic_mixture_confidence_box, [-1, 2], 0.1)
    _assert_raises(ValueError, traffic_mixture_confidence_box, [1.5, 2], 0.1)
    _assert_raises(ValueError, traffic_mixture_confidence_box, [True, False], 0.1)
    _assert_raises(ValueError, traffic_mixture_confidence_box, [1, 2], 0.0)
    _assert_raises(ValueError, traffic_mixture_confidence_box, [1, 2], 1.0)


def test_traffic_box_replay_is_bitwise_deterministic():
    first = traffic_mixture_confidence_box([7, 0, 19, 3], 0.025)
    second = traffic_mixture_confidence_box([7, 0, 19, 3], 0.025)
    np.testing.assert_array_equal(first.raw_lower, second.raw_lower)
    np.testing.assert_array_equal(first.raw_upper, second.raw_upper)
    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    print("bounded-simplex mixture verification")
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print(f"ALL PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
