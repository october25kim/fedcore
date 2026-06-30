"""Proposition 3 pooled CP (subordinate) + ground-truth selective-risk helper.

Code moved verbatim from the original flat ``certificates.py`` -- behaviour unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .cp import cp_upper


# --------------------------------------------------------------------------- #
# Proposition 3 -- pooled CP (subordinate) and ground-truth helpers
# --------------------------------------------------------------------------- #
def pooled_cp(A: Sequence[int], K: Sequence[int], delta: float) -> float:
    """Proposition 3: pooled selective-risk bound ``U+(sum K, sum A; delta)``.

    Valid only under matched-mixture i.i.d. calibration. Invalid under
    heterogeneity (pooled accepted-error count is Poisson-binomial). Subordinate
    to Theorem 1/1'.
    """
    A = np.asarray(A)
    K = np.asarray(K)
    return cp_upper(int(K.sum()), int(A.sum()), delta)


def true_selective_risk(
    a: Sequence[float], r: Sequence[float], lam: Sequence[float]
) -> float:
    """Ground-truth ``R_sel(lambda) = sum(lam a r) / sum(lam a)`` (for sims)."""
    a = np.asarray(a, dtype=float)
    r = np.asarray(r, dtype=float)
    lam = np.asarray(lam, dtype=float)
    denom = float(np.sum(lam * a))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(lam * a * r) / denom)
