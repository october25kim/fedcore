"""Exact mixture and nested-sampling checks."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.sampling import (  # noqa: E402
    nested_without_replacement_indices,
    sample_group_mixture,
    sample_traffic_clients,
    stratified_nested_without_replacement_indices,
)


def _draw(n):
    return sample_group_mixture(
        {0: ["a0", "a1"], 1: ["b0", "b1", "b2"], 2: ["c0"]},
        client_to_group=[0, 0, 1],
        group_probabilities=[0.7, 0.3],
        client_probabilities_given_group={0: [0.25, 0.75], 1: [1.0]},
        n=n,
        seed=42,
    )


def test_exact_law_and_prefix_replay():
    small, large = _draw(200), _draw(50_000)
    # The per-observation implementation must replay the same literal prefix.
    assert np.array_equal(small.sample_id, large.sample_id[:200])
    assert np.array_equal(small.client_id, large.client_id[:200])
    group_freq = np.bincount(large.group_id, minlength=2) / len(large.group_id)
    client_freq = np.bincount(large.client_id, minlength=3) / len(large.client_id)
    assert np.max(np.abs(group_freq - [0.7, 0.3])) < 0.01
    assert np.max(np.abs(client_freq - [0.175, 0.525, 0.3])) < 0.01


def test_traffic_is_identity_only_and_nested():
    a = sample_traffic_clients([0.2, 0.3, 0.5], 50, 9)
    b = sample_traffic_clients([0.2, 0.3, 0.5], 500, 9)
    assert np.array_equal(a, b[:50])


def test_nested_without_replacement():
    draws = nested_without_replacement_indices(100, [10, 25, 50], 3)
    assert np.array_equal(draws[10], draws[25][:10])
    assert np.array_equal(draws[25], draws[50][:25])
    assert len(np.unique(draws[50])) == 50


def test_stratified_nested_fractions_use_common_prefixes():
    strata = np.repeat(np.arange(3), [11, 17, 23])
    draws = stratified_nested_without_replacement_indices(
        strata,
        [0.25, 0.5, 1.0],
        seed=808,
    )
    for value in (0.25, 0.5):
        assert set(draws[value]).issubset(set(draws[1.0]))
    assert set(draws[0.25]).issubset(set(draws[0.5]))
    for fraction, indices in draws.items():
        for client in range(3):
            expected = int(np.floor(fraction * np.sum(strata == client)))
            assert int(np.sum(strata[indices] == client)) == expected
    replay = stratified_nested_without_replacement_indices(strata, [0.5], seed=808)
    np.testing.assert_array_equal(draws[0.5], replay[0.5])


def main():
    test_exact_law_and_prefix_replay()
    test_traffic_is_identity_only_and_nested()
    test_nested_without_replacement()
    test_stratified_nested_fractions_use_common_prefixes()
    print("sampling tests: PASS")


if __name__ == "__main__":
    main()
