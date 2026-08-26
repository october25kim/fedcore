"""Pooled CP under its narrow matched-mixture i.i.d. audit contract.

Ordinary binomial CP is not a heterogeneous Poisson-binomial certificate.  The
public API therefore requires the caller to acknowledge the i.i.d. audit model;
counterexample scripts can use the explicitly non-certifying diagnostic helper.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .cp import cp_upper


# --------------------------------------------------------------------------- #
# Matched-mixture pooled CP (subordinate) and ground-truth helpers
# --------------------------------------------------------------------------- #
def pooled_cp(
    A: Sequence[int],
    K: Sequence[int],
    delta: float,
    *,
    matched_mixture_iid: bool,
) -> float:
    """Theorem 3: pooled selective-risk bound ``U+(sum K, sum A; delta)``.

    Valid only under matched-mixture i.i.d. calibration. Invalid under
    heterogeneity (pooled accepted-error count is Poisson-binomial). Subordinate
    to the stratified full-simplex and strict bounded-mixture certificates.
    """
    if not matched_mixture_iid:
        raise ValueError(
            "pooled_cp is certifying only for a matched-mixture i.i.d. audit; "
            "use pooled_cp_diagnostic for a deliberately non-certifying comparison"
        )
    return pooled_cp_diagnostic(A, K, delta)


def pooled_cp_diagnostic(A: Sequence[int], K: Sequence[int], delta: float) -> float:
    """Compute the pooled number without attaching a validity claim."""
    A = np.asarray(A)
    K = np.asarray(K)
    if A.ndim != 1 or K.shape != A.shape or A.size == 0:
        raise ValueError("A and K must be aligned non-empty vectors")
    try:
        A_float = A.astype(float)
        K_float = K.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("A and K must contain integer counts") from exc
    if (
        np.any(~np.isfinite(A_float))
        or np.any(~np.isfinite(K_float))
        or np.any(A_float != np.floor(A_float))
        or np.any(K_float != np.floor(K_float))
        or np.any(K_float < 0)
        or np.any(A_float < 0)
        or np.any(K_float > A_float)
    ):
        raise ValueError("counts must satisfy 0 <= K <= A")
    return cp_upper(int(K_float.sum()), int(A_float.sum()), delta)


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
