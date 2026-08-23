"""Regression tests for fail-closed integer count validation."""

from __future__ import annotations

from decimal import Decimal
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcore.budget import allocate_stratum_budget
from fedcore.certificate.joint import joint_conditional_certificate
from fedcore.counts import strict_count_vector


INT64_MAX = int(np.iinfo(np.int64).max)
INVALID_COUNT_VECTORS = (
    [True, False],
    [1, np.bool_(False)],
    [1.0, 2.0],
    [1.5, 2],
    [float("nan"), 1],
    [float("inf"), 1],
    [float("-inf"), 1],
    [1 + 0j, 2 + 0j],
    ["1", "2"],
    [Decimal("1"), Decimal("2")],
    [None, 1],
    [-1, 0],
    [INT64_MAX + 1, 0],
    np.array([INT64_MAX + 1, 0], dtype=np.uint64),
)


def test_strict_count_vector_preserves_valid_integer_counts():
    counts = strict_count_vector([np.int32(0), np.uint64(2), INT64_MAX], "counts")

    np.testing.assert_array_equal(counts, [0, 2, INT64_MAX])
    assert counts.dtype == np.int64
    assert not counts.flags.writeable


@pytest.mark.parametrize("values", INVALID_COUNT_VECTORS)
def test_strict_count_vector_rejects_every_non_count_value(values):
    with pytest.raises(ValueError):
        strict_count_vector(values, "counts")


@pytest.mark.parametrize("values", INVALID_COUNT_VECTORS)
def test_joint_certificate_never_silently_casts_invalid_counts(values):
    size = len(values)
    with pytest.raises(ValueError):
        joint_conditional_certificate(
            values,
            np.zeros(size, dtype=np.int64),
            np.full(size, 2, dtype=np.int64),
            alpha=0.1,
            risk_eps=np.full(size, 0.01),
            acceptance_lower_eps=np.full(size, 0.01),
        )


@pytest.mark.parametrize("position", ("A", "K", "n"))
def test_joint_certificate_validates_each_count_vector(position):
    values = {
        "A": [1, 1],
        "K": [0, 0],
        "n": [2, 2],
    }
    values[position] = [1.0, 1.0]

    with pytest.raises(ValueError):
        joint_conditional_certificate(
            values["A"],
            values["K"],
            values["n"],
            alpha=0.1,
            risk_eps=[0.01, 0.01],
            acceptance_lower_eps=[0.01, 0.01],
        )


@pytest.mark.parametrize("values", INVALID_COUNT_VECTORS)
def test_budget_allocation_never_silently_casts_invalid_counts(values):
    size = len(values)
    with pytest.raises(ValueError):
        allocate_stratum_budget(0.05, values, np.zeros(size, dtype=np.int64))


def test_budget_allocation_validates_proposal_error_counts():
    with pytest.raises(ValueError):
        allocate_stratum_budget(0.05, [2, 2], [0.0, 1.0])
