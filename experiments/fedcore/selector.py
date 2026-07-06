"""Risk-buffered proposal selector and open-set accepted/error counting.

Open-set error semantics
------------------------
A point is described by (score, pred, y_open):
  * ``pred``   : argmax over known classes (in [0, C)).
  * ``y_open`` : ground-truth open-set label; a known class in [0, C), or -1 for
                 an unknown-class point.
Acceptance: ``A(x) = 1`` iff ``score(x) >= t``.
Error (among accepted): the prediction is wrong, which in open-set means either
the true class is unknown (any known prediction is wrong) OR the predicted known
class is incorrect:
        E(x) = A(x) * 1{ y_open == -1  OR  pred != y_open }.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def open_set_error(pred: np.ndarray, y_open: np.ndarray) -> np.ndarray:
    """Boolean error indicator (ignores acceptance)."""
    return (y_open < 0) | (pred != y_open)


@dataclass
class Selector:
    threshold: float
    feasible: bool          # whether the risk-buffer constraint was satisfiable

    def accept(self, score: np.ndarray) -> np.ndarray:
        return score >= self.threshold


def empirical_risk_coverage(
    score: np.ndarray, err: np.ndarray, t: float
) -> tuple[float, float]:
    """(coverage, selective_risk) at threshold ``t``."""
    acc = score >= t
    n_acc = int(acc.sum())
    cov = n_acc / len(score) if len(score) else 0.0
    risk = float(err[acc].mean()) if n_acc > 0 else 0.0
    return cov, risk


def choose_threshold(
    score: np.ndarray,
    pred: np.ndarray,
    y_open: np.ndarray,
    gamma: float,
    alpha: float,
    n_grid: int = 300,
) -> Selector:
    """Risk-buffered proposal: maximize coverage s.t. empirical risk <= gamma*alpha.

    Scans candidate thresholds (score quantiles). Returns the feasible threshold
    with the largest coverage; if none is feasible, returns a selector that
    accepts nothing (threshold = +inf, feasible=False).
    """
    err = open_set_error(pred, y_open)
    buffer = gamma * alpha
    qs = np.linspace(0.0, 1.0, n_grid)
    cands = np.unique(np.quantile(score, qs))
    best_t, best_cov, feasible = np.inf, -1.0, False
    for t in cands:
        cov, risk = empirical_risk_coverage(score, err, t)
        if risk <= buffer and cov > best_cov:
            best_cov, best_t, feasible = cov, float(t), True
    return Selector(threshold=best_t, feasible=feasible)


def counts_per_client(
    score: np.ndarray,
    pred: np.ndarray,
    y_open: np.ndarray,
    client: np.ndarray,
    selector: Selector,
    n_clients: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-client (A_j, K_j, n_j) under the given selector.

    A_j = accepted count, K_j = accepted-error count, n_j = points at client j.
    """
    acc = selector.accept(score)
    err = open_set_error(pred, y_open)
    A = np.zeros(n_clients, dtype=int)
    K = np.zeros(n_clients, dtype=int)
    n = np.zeros(n_clients, dtype=int)
    for j in range(n_clients):
        mask = client == j
        n[j] = int(mask.sum())
        aj = acc & mask
        A[j] = int(aj.sum())
        K[j] = int((aj & err).sum())
    return A, K, n
