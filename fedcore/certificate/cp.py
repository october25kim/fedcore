"""Clopper-Pearson confidence limits + lambda-box sampling helpers.

Pure CP primitives shared by the Theorem 1/1', stratified, and pooled certificates.
Depends only on numpy + scipy (no torch). Code moved verbatim from the original flat
``certificates.py`` during the structure-only refactor -- behaviour unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from scipy.stats import beta as _beta


# --------------------------------------------------------------------------- #
# Clopper-Pearson one-sided confidence limits
# --------------------------------------------------------------------------- #
def cp_upper(k: int, n: int, eps: float) -> float:
    """One-sided Clopper-Pearson UPPER limit for a Binomial(n, p) success rate.

    Returns ``p_hi`` such that ``P(Bin(n, p_hi) <= k) = eps`` (coverage at level
    ``1 - eps``). Conventions: ``1.0`` if ``n <= 0`` (no data => no information)
    and ``1.0`` if ``k >= n`` (saturated).
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - eps, k + 1, n - k))


def cp_lower(k: int, n: int, eps: float) -> float:
    """One-sided Clopper-Pearson LOWER limit for a Binomial(n, p) success rate.

    Returns ``p_lo`` such that ``P(Bin(n, p_lo) >= k) = eps``. Conventions:
    ``0.0`` if ``n <= 0`` and ``0.0`` if ``k <= 0``.
    """
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    return float(_beta.ppf(eps, k, n - k + 1))


# --------------------------------------------------------------------------- #
# lambda-box sampling helper
# --------------------------------------------------------------------------- #
def _sample_lambdas(
    J: int,
    radius: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample mixture weights from a box around the uniform mixture.

    Each component is drawn uniformly from ``[1/J - radius, 1/J + radius]``,
    clipped to be non-negative, then renormalized to sum to 1. The uniform
    mixture itself is always included as the first sample.
    """
    u = 1.0 / J
    lo = max(0.0, u - radius)
    hi = u + radius
    samples = np.empty((n_samples + 1, J), dtype=float)
    samples[0] = u
    raw = rng.uniform(lo, hi, size=(n_samples, J))
    samples[1:] = raw
    samples = np.clip(samples, 0.0, None)
    samples = samples / samples.sum(axis=1, keepdims=True)
    return samples


def _resolve_box_radius(box: Optional[Union[float, Sequence[float]]]) -> float:
    """Interpret the ``box`` argument as an additive radius around uniform."""
    if box is None:
        return 0.15
    if isinstance(box, (int, float)):
        return float(box)
    # tuple/sequence (lo, hi): convert to half-width
    box = tuple(box)
    return float((box[1] - box[0]) / 2.0)
