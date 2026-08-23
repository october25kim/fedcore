"""Traffic-derived Lambda_hat theorem tests (Office-Home §6.3)."""

from __future__ import annotations

import numpy as np
import pytest

from fedcore.officehome_traffic_lambda import (
    DELTA_LAMBDA,
    M_GRID,
    N_DOMAIN_CLIENTS,
    TrafficLambdaError,
    assert_traffic_identity_disjoint,
    box_total_width,
    build_traffic_lambda,
    coordinate_tail_report,
    covers,
    draw_traffic_client_counts,
    traffic_lambda_box,
)
from fedcore.experiments.officehome_theorem_tests import run_theorem_suite


def test_exact_coordinate_tail_accounting():
    box = traffic_lambda_box([100, 200, 300, 400], DELTA_LAMBDA)
    report = coordinate_tail_report(box)
    assert report["n_tails"] == 2 * N_DOMAIN_CLIENTS
    assert report["per_side_tail"] == pytest.approx(DELTA_LAMBDA / (2 * N_DOMAIN_CLIENTS))
    assert report["union_bound"] == pytest.approx(DELTA_LAMBDA)


def test_label_and_score_free_construction():
    ids = [0, 1, 2, 3] * 100
    a = draw_traffic_client_counts(ids, 500, seed=3)
    b = draw_traffic_client_counts(ids, 500, seed=3)
    assert np.array_equal(a, b)
    assert int(a.sum()) == 500


def test_deterministic_replay():
    ids = [0, 0, 1, 2, 3, 3] * 50
    counts = draw_traffic_client_counts(ids, 400, seed=9)
    box_a = traffic_lambda_box(counts, DELTA_LAMBDA)
    box_b = traffic_lambda_box(
        draw_traffic_client_counts(ids, 400, seed=9), DELTA_LAMBDA
    )
    assert np.array_equal(box_a.raw_lower, box_b.raw_lower)
    assert np.array_equal(box_a.raw_upper, box_b.raw_upper)


def test_width_decreases_with_m_in_aggregate():
    lam = np.array([0.25, 0.25, 0.25, 0.25])
    rng = np.random.default_rng(0)
    widths = {}
    for m in M_GRID:
        ws = [
            box_total_width(traffic_lambda_box(rng.multinomial(m, lam), DELTA_LAMBDA))
            for _ in range(300)
        ]
        widths[m] = float(np.mean(ws))
    ordered = sorted(M_GRID)
    assert all(widths[ordered[i]] > widths[ordered[i + 1]] for i in range(len(ordered) - 1))


def test_fail_closed_on_identity_overlap():
    assert_traffic_identity_disjoint(["a", "b", "c"], ["d", "e"])  # ok
    with pytest.raises(TrafficLambdaError):
        assert_traffic_identity_disjoint(["a", "b"], ["b", "c"])
    with pytest.raises(TrafficLambdaError):
        assert_traffic_identity_disjoint(["a", "a"])  # internal duplicate


def test_full_simplex_fallback_on_empty_traffic():
    res = build_traffic_lambda([0, 0, 0, 0], DELTA_LAMBDA)
    assert res.fell_back_to_simplex
    assert np.array_equal(res.box.raw_lower, np.zeros(N_DOMAIN_CLIENTS))
    assert np.array_equal(res.box.raw_upper, np.ones(N_DOMAIN_CLIENTS))


def test_covers_boundary_mixture():
    # A boundary mixture with a zero coordinate is covered by construction.
    lam = np.array([0.0, 0.34, 0.33, 0.33])
    rng = np.random.default_rng(1)
    hits = sum(
        covers(traffic_lambda_box(rng.multinomial(1000, lam), DELTA_LAMBDA), lam)
        for _ in range(300)
    )
    assert hits / 300 >= 1.0 - DELTA_LAMBDA - 0.02


def test_theorem_suite_all_properties_pass():
    # Fast MC run; the coverage guarantee is analytic, the study confirms it.
    result = run_theorem_suite(n_trials=6000, write=False)
    assert result["all_pass"], result["properties"]
    for name, ok in result["properties"].items():
        assert ok, name
