"""Synthetic heterogeneous client generator for Fed-CORE certificate experiments.

Each client j is summarized by an acceptance rate ``a_j`` and a conditional
selective risk ``r_j``. A certification-fold draw of ``n_j`` i.i.d. points yields
the secure-aggregatable counts ``A_j`` (accepted) and ``K_j`` (accepted-errors).

The point-level draw respects ``K_j <= A_j``: a point is accepted with prob a_j,
and -- if accepted -- is an error with prob r_j.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientPopulation:
    """Ground-truth per-client parameters."""

    a: np.ndarray   # acceptance rates  a_j  in (0, 1]
    r: np.ndarray   # conditional selective risks r_j in [0, 1]

    @property
    def J(self) -> int:
        return len(self.a)

    @property
    def m(self) -> np.ndarray:
        """Accepted-error mass m_j = a_j r_j."""
        return self.a * self.r


def draw_counts(
    pop: ClientPopulation,
    n: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw certification-fold counts (A_j, K_j) for each client.

    Returns
    -------
    A, K : integer arrays of length J.
    """
    J = pop.J
    A = np.empty(J, dtype=int)
    K = np.empty(J, dtype=int)
    for j in range(J):
        acc = rng.random(int(n[j])) < pop.a[j]
        nacc = int(acc.sum())
        A[j] = nacc
        # among accepted points, error with prob r_j
        K[j] = int((rng.random(nacc) < pop.r[j]).sum())
    return A, K


def heterogeneous_population(
    n_good: int = 4,
    a_good: float = 0.70,
    r_good: float = 0.02,
    a_bad: float = 0.50,
    r_bad: float = 0.30,
) -> ClientPopulation:
    """A common FL pathology: many low-risk clients + one high-risk minority client.

    This is the configuration under which a deployment mixture that overweights
    the high-risk client breaks the naive pooled certificate.
    """
    a = np.array([a_good] * n_good + [a_bad], dtype=float)
    r = np.array([r_good] * n_good + [r_bad], dtype=float)
    return ClientPopulation(a=a, r=r)
