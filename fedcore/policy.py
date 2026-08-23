"""Proposal-only global and client-specific threshold policies.

The functions here deliberately accept one fold at a time.  Policy construction
consumes proposal arrays; applying a frozen policy to certification or test arrays
does not expose their labels to threshold selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fedcore.selector import choose_threshold, open_set_error


@dataclass(frozen=True)
class ThresholdPolicy:
    """A frozen accept/reject rule with either one or one-per-client threshold."""

    name: str
    thresholds: np.ndarray
    feasible: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.thresholds, dtype=float)
        f = np.asarray(self.feasible, dtype=bool)
        if t.ndim != 1 or len(t) == 0 or f.shape != t.shape:
            raise ValueError(
                "thresholds and feasible must be aligned non-empty vectors"
            )
        object.__setattr__(self, "thresholds", t.copy())
        object.__setattr__(self, "feasible", f.copy())

    @property
    def client_specific(self) -> bool:
        return len(self.thresholds) > 1

    def accept(self, score: Sequence[float], client: Sequence[int]) -> np.ndarray:
        score_arr = np.asarray(score, dtype=float)
        client_arr = np.asarray(client, dtype=int)
        if score_arr.shape != client_arr.shape:
            raise ValueError("score and client must be aligned")
        if self.client_specific:
            if np.any(client_arr < 0) or np.any(client_arr >= len(self.thresholds)):
                raise ValueError("client id outside frozen threshold vector")
            threshold = self.thresholds[client_arr]
            usable = self.feasible[client_arr]
        else:
            threshold = np.full(score_arr.shape, self.thresholds[0])
            usable = np.full(score_arr.shape, self.feasible[0])
        return usable & (score_arr >= threshold)

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold_policy": self.name,
            "thresholds": self.thresholds.tolist(),
            "threshold_feasible": self.feasible.tolist(),
        }


def choose_threshold_policy(
    score: Sequence[float],
    pred: Sequence[int],
    y_open: Sequence[int],
    client: Sequence[int],
    *,
    gamma: float,
    alpha: float,
    n_clients: int,
    policy: str,
    n_grid: int = 300,
) -> ThresholdPolicy:
    """Choose a global or per-client policy using proposal observations only."""

    score_arr = np.asarray(score, dtype=float)
    pred_arr = np.asarray(pred)
    y_arr = np.asarray(y_open)
    client_arr = np.asarray(client, dtype=int)
    if not (score_arr.shape == pred_arr.shape == y_arr.shape == client_arr.shape):
        raise ValueError("proposal score/pred/y_open/client arrays must align")
    if n_clients <= 0 or np.any(client_arr < 0) or np.any(client_arr >= n_clients):
        raise ValueError("invalid n_clients or proposal client id")

    if policy == "global":
        sel = choose_threshold(score_arr, pred_arr, y_arr, gamma, alpha, n_grid=n_grid)
        return ThresholdPolicy(
            name="global",
            thresholds=np.array([sel.threshold]),
            feasible=np.array([sel.feasible]),
        )
    if policy != "client_specific":
        raise ValueError(f"unknown threshold policy {policy!r}")

    thresholds = np.full(n_clients, np.inf, dtype=float)
    feasible = np.zeros(n_clients, dtype=bool)
    for j in range(n_clients):
        mask = client_arr == j
        if not np.any(mask):
            continue
        sel = choose_threshold(
            score_arr[mask], pred_arr[mask], y_arr[mask], gamma, alpha, n_grid=n_grid
        )
        thresholds[j] = sel.threshold
        feasible[j] = sel.feasible
    return ThresholdPolicy("client_specific", thresholds, feasible)


def policy_counts(
    score: Sequence[float],
    pred: Sequence[int],
    y_open: Sequence[int],
    client: Sequence[int],
    threshold_policy: ThresholdPolicy,
    n_clients: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-client ``A, K, n`` under one already-frozen threshold policy."""

    score_arr = np.asarray(score, dtype=float)
    pred_arr = np.asarray(pred)
    y_arr = np.asarray(y_open)
    client_arr = np.asarray(client, dtype=int)
    if not (score_arr.shape == pred_arr.shape == y_arr.shape == client_arr.shape):
        raise ValueError("score/pred/y_open/client arrays must align")
    accept = threshold_policy.accept(score_arr, client_arr)
    error = open_set_error(pred_arr, y_arr)
    A = np.zeros(n_clients, dtype=int)
    K = np.zeros(n_clients, dtype=int)
    n = np.zeros(n_clients, dtype=int)
    for j in range(n_clients):
        mask = client_arr == j
        n[j] = int(mask.sum())
        accepted = mask & accept
        A[j] = int(accepted.sum())
        K[j] = int((accepted & error).sum())
    return A, K, n


def policy_risk_coverage(
    score: Sequence[float],
    pred: Sequence[int],
    y_open: Sequence[int],
    client: Sequence[int],
    threshold_policy: ThresholdPolicy,
) -> tuple[float, float]:
    """Empirical coverage and accepted risk for evaluation only."""

    score_arr = np.asarray(score, dtype=float)
    client_arr = np.asarray(client, dtype=int)
    accept = threshold_policy.accept(score_arr, client_arr)
    error = open_set_error(np.asarray(pred), np.asarray(y_open))
    coverage = float(accept.mean()) if len(accept) else 0.0
    risk = float(error[accept].mean()) if np.any(accept) else 0.0
    return coverage, risk
