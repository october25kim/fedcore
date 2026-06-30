"""Synthetic heterogeneous client populations for the CPU experiments.

A :class:`ClientPopulation` is the ground truth of a federated calibration
setup: per-client acceptance probability ``a_j`` and conditional error rate
``r_j``. :func:`draw_counts` samples the observed counts ``(A_j, K_j)`` from a
total of ``n_j`` calibration points per client, matching the conditional law
``K_j | A_j ~ Bin(A_j, r_j)`` used by Theorem 1/1'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ClientPopulation:
    """Ground-truth per-client acceptance / error rates.

    Parameters
    ----------
    a : per-client acceptance probability ``a_j = P_j(accept)``.
    r : per-client conditional error rate ``r_j = P_j(error | accept)``.
    """

    a: np.ndarray
    r: np.ndarray

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float)
        self.r = np.asarray(self.r, dtype=float)
        if self.a.shape != self.r.shape:
            raise ValueError("a and r must have the same shape")

    @property
    def m(self) -> np.ndarray:
        """Accepted-error mass ``m_j = a_j * r_j = P_j(accept & error)``."""
        return self.a * self.r

    @property
    def J(self) -> int:
        """Number of clients."""
        return int(self.a.shape[0])


def draw_counts(
    pop: ClientPopulation, n, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample observed ``(A, K)`` counts for each client.

    For client ``j`` with ``n_j`` calibration points: each point is accepted
    with probability ``a_j``; among the ``A_j`` accepted points, each is wrong
    with probability ``r_j`` (so ``K_j <= A_j``).

    Parameters
    ----------
    n : int or array-like of per-client totals ``n_j``.
    """
    J = pop.J
    n_arr = np.full(J, n, dtype=int) if np.isscalar(n) else np.asarray(n, dtype=int)
    A = np.empty(J, dtype=int)
    K = np.empty(J, dtype=int)
    for j in range(J):
        accepts = rng.random(n_arr[j]) < pop.a[j]
        A_j = int(accepts.sum())
        errs = rng.random(A_j) < pop.r[j]
        A[j] = A_j
        K[j] = int(errs.sum())
    return A, K


def heterogeneous_population(
    n_good: int = 4,
    a_good: float = 0.7,
    r_good: float = 0.02,
    a_bad: float = 0.5,
    r_bad: float = 0.3,
) -> ClientPopulation:
    """Canonical ``n_good`` low-risk clients + one high-risk client.

    The high-risk client accepts less often (``a_bad``) and is wrong far more
    often when it does (``r_bad``); this is what breaks naive pooling under a
    deployment-mixture shift toward the bad client.
    """
    a = np.array([a_good] * n_good + [a_bad], dtype=float)
    r = np.array([r_good] * n_good + [r_bad], dtype=float)
    return ClientPopulation(a=a, r=r)
