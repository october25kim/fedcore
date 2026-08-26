"""Accepted-count planning formulas.

The current manuscript's Theorem 4 gives the exact zero-error threshold for one
fixed full-simplex selector. The historical ``thm2_floor`` symbol is retained
only for archived clientwise analyses that divided the tail by ``J``.
"""

from __future__ import annotations

import math

import numpy as np


def zero_error_count_threshold(alpha: float, delta_r: float) -> int:
    """Theorem 4: smallest ``A`` with ``U+(0, A; delta_r) <= alpha``.

    This is the fixed-selector, no-stratum-penalty threshold
    ``ceil(log(1/delta_r) / -log(1-alpha))``.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < float(delta_r) < 1.0:
        raise ValueError("delta_r must lie in (0, 1)")
    return int(
        math.ceil(math.log(1.0 / float(delta_r)) / -math.log1p(-float(alpha)))
    )


def thm2_floor(J: int, delta: float, alpha: float) -> float:
    """Archived unrounded clientwise floor ``log(J/delta)/-log(1-alpha)``.

    The function name and return type are kept for compatibility with historical
    campaign artifacts. It is not the current manuscript's Theorem 4 API. Use
    :func:`zero_error_count_threshold` for a fixed full-simplex selector.
    """
    if isinstance(J, bool) or int(J) != J or J <= 0:
        raise ValueError("J must be a positive integer")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(np.log(int(J) / float(delta)) / (-np.log1p(-float(alpha))))


__all__ = ["thm2_floor", "zero_error_count_threshold"]
