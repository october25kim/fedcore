"""Occupancy arithmetic for with-replacement reservoir draws.

The classical occupancy result: drawing ``n`` times WITH replacement, uniformly and
independently, from a reservoir of ``M`` distinct items, the expected number of
distinct items seen is

    E[unique] = M * (1 - (1 - 1/M)^n).

This is the yardstick for "how many distinct labelled examples actually support a
number that was computed from n nominal draws".

It applies ONLY to uniform with-replacement draws. It is deliberately NOT defined
for the headline repartition, which is a permutation (without replacement) and whose
unique count is exactly ``n`` by construction -- callers get NaN there rather than a
misapplied formula (see ``draws.DrawMode``).
"""

from __future__ import annotations

import numpy as np


def expected_unique_count(M: int, n: int) -> float:
    """``M * (1 - (1 - 1/M)^n)`` for uniform with-replacement draws.

    Computed in log space so large ``n`` does not lose precision to (1-1/M)^n
    underflow.
    """
    if M <= 0 or n <= 0:
        return 0.0
    # (1 - 1/M)^n == exp(n * log1p(-1/M)); log1p keeps precision for large M.
    return float(M * (-np.expm1(n * np.log1p(-1.0 / M))))


def expected_max_multiplicity(M: int, n: int) -> float:
    """Expected maximum multiplicity is NOT in closed form; report NaN, not a guess.

    Kept as an explicit, documented NaN so a reader of the schema knows the column's
    observed value has no analytic companion, rather than wondering why it is absent.
    """
    return float("nan")


def unique_and_multiplicity(sampled_ids: np.ndarray) -> tuple:
    """``(unique_count, max_multiplicity, id_to_multiplicity)`` for a draw."""
    ids, counts = np.unique(np.asarray(sampled_ids), return_counts=True)
    if len(ids) == 0:
        return 0, 0, {}
    return int(len(ids)), int(counts.max()), dict(zip(ids.tolist(), counts.tolist()))
