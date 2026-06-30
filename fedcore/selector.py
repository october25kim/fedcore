"""Risk-buffered accept/reject selector over an open-set score.

Open-set label convention (``y_open``): a known class id in ``[0, C)`` for an
in-distribution point, or ``-1`` for an unknown (out-of-distribution) point.

A prediction is an *open-set error* if the point is unknown (must be rejected,
so any acceptance is wrong) or if the predicted known class is wrong. The
selector accepts a point when its score is at least a threshold; the threshold
is chosen on the PROPOSAL fold only, with a risk buffer ``gamma * alpha``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def open_set_error(pred: np.ndarray, y_open: np.ndarray) -> np.ndarray:
    """Boolean error mask: unknown point, or wrong known-class prediction."""
    pred = np.asarray(pred)
    y_open = np.asarray(y_open)
    return (y_open < 0) | (pred != y_open)


@dataclass
class Selector:
    """An accept-if-score-at-least-threshold rule."""

    threshold: float
    feasible: bool

    def accept(self, score: np.ndarray) -> np.ndarray:
        """Boolean accept mask for a vector of scores."""
        return np.asarray(score) >= self.threshold


def empirical_risk_coverage(
    score: np.ndarray, err: np.ndarray, t: float
) -> Tuple[float, float]:
    """Coverage and selective risk at threshold ``t``.

    Coverage = fraction accepted. Selective risk = error rate among accepted
    points (``0.0`` if nothing is accepted, which is vacuously safe).
    """
    score = np.asarray(score)
    err = np.asarray(err, dtype=bool)
    accept = score >= t
    n = len(score)
    n_acc = int(accept.sum())
    coverage = n_acc / n if n > 0 else 0.0
    risk = float(err[accept].mean()) if n_acc > 0 else 0.0
    return coverage, risk


def choose_threshold(
    score: np.ndarray,
    pred: np.ndarray,
    y_open: np.ndarray,
    gamma: float,
    alpha: float,
    n_grid: int = 300,
) -> Selector:
    """Pick the coverage-maximizing threshold whose risk respects the buffer.

    Searches candidate thresholds (score quantiles) and returns the one with the
    largest coverage among those with ``empirical_risk <= gamma * alpha`` and
    positive coverage. If none qualifies, returns a reject-everything selector
    (``threshold = +inf``, ``feasible = False``).
    """
    score = np.asarray(score, dtype=float)
    err = open_set_error(pred, y_open)
    budget = gamma * alpha

    qs = np.quantile(score, np.linspace(0.0, 1.0, n_grid))
    candidates = np.unique(qs)

    best_t = np.inf
    best_cov = -1.0
    for t in candidates:
        cov, risk = empirical_risk_coverage(score, err, t)
        if cov > 0.0 and risk <= budget and cov > best_cov:
            best_cov = cov
            best_t = float(t)

    if best_cov < 0.0:
        return Selector(threshold=np.inf, feasible=False)
    return Selector(threshold=best_t, feasible=True)


def counts_per_client(
    score: np.ndarray,
    pred: np.ndarray,
    y_open: np.ndarray,
    client: np.ndarray,
    selector: Selector,
    n_clients: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-client ``(A, K, n)`` counts under a fixed selector.

    ``A_j`` accepted, ``K_j`` accepted-and-wrong, ``n_j`` total -- for each
    client ``j in [0, n_clients)``.
    """
    score = np.asarray(score)
    client = np.asarray(client)
    err = open_set_error(pred, y_open)
    accept = selector.accept(score)

    A = np.zeros(n_clients, dtype=int)
    K = np.zeros(n_clients, dtype=int)
    n = np.zeros(n_clients, dtype=int)
    for j in range(n_clients):
        mask = client == j
        n[j] = int(mask.sum())
        acc_j = mask & accept
        A[j] = int(acc_j.sum())
        K[j] = int((acc_j & err).sum())
    return A, K, n
