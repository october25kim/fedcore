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
    """Proposition 3: pooled selective-risk bound ``U+(sum K, sum A; delta)``.

    Valid only under matched-mixture i.i.d. calibration. Invalid under
    heterogeneity (pooled accepted-error count is Poisson-binomial). Subordinate
    to Theorem 1/1'.
    """
    if not matched_mixture_iid:
        raise ValueError(
            "pooled_cp is certifying only for a matched-mixture i.i.d. audit; "
            "use pooled_cp_diagnostic for a deliberately non-certifying comparison"
        )
    return pooled_cp_diagnostic(A, K, delta)


def pooled_cp_diagnostic(A: Sequence[int], K: Sequence[int], delta: float) -> float:
    """Compute the pooled number without attaching a validity claim."""
    accepted = np.asarray(A)
    errors = np.asarray(K)
    if accepted.ndim != 1 or errors.ndim != 1 or accepted.shape != errors.shape:
        raise ValueError("A and K must be aligned one-dimensional count vectors")
    if accepted.size == 0:
        raise ValueError("A and K must be non-empty")
    try:
        accepted_float = accepted.astype(float)
        errors_float = errors.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("A and K must contain integer counts") from exc
    if (
        np.any(~np.isfinite(accepted_float))
        or np.any(~np.isfinite(errors_float))
        or np.any(accepted_float != np.floor(accepted_float))
        or np.any(errors_float != np.floor(errors_float))
        or np.any(errors_float < 0.0)
        or np.any(accepted_float < 0.0)
        or np.any(errors_float > accepted_float)
    ):
        raise ValueError("counts must satisfy 0 <= K <= A")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return cp_upper(int(errors_float.sum()), int(accepted_float.sum()), float(delta))


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
