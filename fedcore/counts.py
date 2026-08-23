"""Strict validation helpers for statistical count vectors."""

from __future__ import annotations

import operator
from typing import Sequence

import numpy as np


_INT64_MAX = int(np.iinfo(np.int64).max)


def strict_count_vector(values: Sequence[int], name: str) -> np.ndarray:
    """Return a non-empty, one-dimensional, read-only ``int64`` count vector.

    Count inputs are deliberately not coerced: booleans, floats (including
    integer-valued floats), strings, NaN/Inf, negative integers, and integers
    outside the non-negative ``int64`` range are rejected before conversion.
    """

    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a non-empty one-dimensional integer sequence"
        ) from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional integer sequence")

    counts: list[int] = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must contain integer counts, not booleans")
        try:
            count = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain only integer counts") from exc
        if count < 0:
            raise ValueError(f"{name} must contain non-negative integer counts")
        if count > _INT64_MAX:
            raise ValueError(f"{name} contains a count too large for int64")
        counts.append(count)

    result = np.asarray(counts, dtype=np.int64)
    result.setflags(write=False)
    return result
