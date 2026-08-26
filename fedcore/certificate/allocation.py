"""Archived proposal-allocated clientwise sensitivity: rules R1-R3.

These rules implement a valid conservative union-bound allocation across
clients and are retained to reproduce the historical campaign. They are not the
current full-simplex fixed-member theorem, whose scalar-extremum API has no
client-count penalty. See ``full_simplex_fixed_member_certificate``.

This module implements the *allocated* branch of the Fed-CORE certificate, in which
the per-client error budgets ``eps_j`` are chosen on the PROPOSAL fold only and
frozen before the certification fold is touched.

Contract (see the pre-registered brief, section 0):

* **Historical allocated certificate.** For any allocation
  ``eps = (eps_1, ..., eps_J)`` with ``sum_j eps_j <= delta_r`` that is measurable
  w.r.t. the proposal fold and fixed before the certification fold is observed,
  ``rbar_j = U(K_j, A_j; eps_j)`` and ``Ubar = max_j rbar_j`` is a valid
  ``1 - delta_r`` upper bound on ``R_sel(lambda)`` for every ``lambda`` with
  positive accepted mass.  Validity is a union bound over the ``J`` per-client
  events; uniform allocation ``eps_j = delta_r / J`` is the conservative legacy
  special case.

* **Archived zero-error allocation frontier.** With ``K_j = 0`` for all ``j``, an
  allocation meeting ``max_j rbar_j <= alpha`` exists **iff**
  ``sum_j (1 - alpha)^{A_j} <= delta_r``.  This is strictly weaker than the
  legacy uniform requirement ``max_j (1 - alpha)^{A_j} <= delta_r / J``
  whenever the accepted counts ``A_j`` are heterogeneous.

* **Allocation rules** (predeclared; never selected post hoc):
  ``R1`` zero-error water-filling, ``R2`` general water-filling, ``R3`` reserve
  safeguard (the headline rule, ``kappa = 0.2``).

The key inversion used throughout: for the zero-error Clopper-Pearson upper limit
``U(0, A, eps) = 1 - eps^{1/A}``,

    ``U(0, A, eps) <= alpha``  <=>  ``eps >= (1 - alpha)^A``,

so a client with accepted count ``A`` needs at least ``(1 - alpha)^A`` of the
budget, and the per-client zero-error count floor at budget ``eps`` is
``ceil(ln(1/eps) / (-ln(1 - alpha)))``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from scipy.stats import binom as _binom

from .cp import cp_upper

__all__ = [
    "AllocatedCertificate",
    "allocated_risk_certificate",
    "zero_error_frontier",
    "zero_error_floor",
    "uniform_allocation",
    "allocate_R1",
    "allocate_R2",
    "allocate_R3",
    "allocate",
    "ALLOCATION_RULES",
    "R3_KAPPA",
]


# The reserve fraction of rule R3, fixed by pre-registration. Never tune this.
R3_KAPPA = 0.2

ALLOCATION_RULES = ("uniform", "R1", "R2", "R3")


# --------------------------------------------------------------------------- #
# Feasibility arithmetic
# --------------------------------------------------------------------------- #
def zero_error_floor(eps: float, alpha: float) -> int:
    """Smallest accepted count ``A`` with ``U(0, A, eps) <= alpha``.

    Inverts ``1 - eps^{1/A} <= alpha`` to ``A >= ln(1/eps) / (-ln(1 - alpha))``.
    With ``eps = delta_r / J`` this is the archived clientwise uniform floor
    ``ceil(ln(J / delta_r) / (-ln(1 - alpha)))``.
    """
    if not 0.0 < eps <= 1.0:
        raise ValueError(f"eps must lie in (0, 1], got {eps!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    if eps >= 1.0:
        return 0
    return int(math.ceil(math.log(1.0 / eps) / (-math.log1p(-alpha))))


def _zero_error_floor_real(eps: float, alpha: float) -> float:
    """Un-rounded floor, used to verify that R1 equalizes the slack exactly."""
    return math.log(1.0 / eps) / (-math.log1p(-alpha))


def zero_error_frontier(A: Sequence[int], alpha: float) -> float:
    """Archived allocation-frontier value ``sum_j (1 - alpha)^{A_j}``.

    A zero-error allocation certifying at level ``alpha`` exists iff this value is
    ``<= delta_r``. Returned even when some ``K_j > 0`` (as a diagnostic); the
    equivalence itself only holds in the all-zero-error regime.
    """
    counts = np.asarray(A, dtype=float)
    if np.any(counts < 0):
        raise ValueError("accepted counts must be non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    return float(np.sum(np.power(1.0 - alpha, counts)))


# --------------------------------------------------------------------------- #
# Allocation rules (proposal-fold measurable)
# --------------------------------------------------------------------------- #
def _validate_budget(delta_r: float, J: int) -> None:
    if not 0.0 < delta_r < 1.0:
        raise ValueError(f"delta_r must lie in (0, 1), got {delta_r!r}")
    if J <= 0:
        raise ValueError("need at least one client")


def uniform_allocation(J: int, delta_r: float) -> np.ndarray:
    """Archived conservative allocation ``eps_j = delta_r / J``."""
    _validate_budget(delta_r, J)
    return np.full(J, delta_r / J, dtype=float)


def allocate_R1(A_hat: Sequence[float], delta_r: float, alpha: float) -> np.ndarray:
    """R1 -- zero-error water-filling.

    ``eps_j = delta_r * (1 - alpha)^{A_hat_j} / sum_l (1 - alpha)^{A_hat_l}``.

    Equalizes the per-client slack ``floor_j - A_hat_j``: writing
    ``S = sum_l (1 - alpha)^{A_hat_l}`` and ``c = -ln(1 - alpha) > 0``,

        ``floor_j = ln(S / delta_r) / c + A_hat_j``

    so ``floor_j - A_hat_j = ln(S / delta_r) / c`` is constant in ``j``. Clients
    already rich in accepted mass therefore surrender budget to the scarce ones,
    which is exactly the heterogeneous regime where the uniform rule fails.
    """
    counts = np.asarray(A_hat, dtype=float)
    _validate_budget(delta_r, counts.size)
    if np.any(counts < 0):
        raise ValueError("proposal accepted counts must be non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

    # Work in log space and subtract the max before exponentiating: the raw
    # weight (1 - alpha)^A underflows for large A (at alpha=0.1, A=9000 gives
    # ~1e-412), and even the shifted weight underflows once the count spread is
    # wide enough. Any coordinate that still reaches 0 is clamped to the smallest
    # positive normal double, then the vector is rescaled to spend exactly
    # delta_r. Clamping UP is the safe direction: it keeps sum(eps) == delta_r,
    # so the archived allocation union bound still holds, and a larger eps_j only tightens
    # rbar_j. Without the clamp a count-rich client would receive eps_j = 0,
    # hence rbar_j = 1, and the cell would be refused despite being certifiable.
    log_w = counts * math.log1p(-alpha)
    log_w -= log_w.max()
    w = np.exp(log_w)
    total = w.sum()
    if not np.isfinite(total) or total <= 0.0:
        return uniform_allocation(counts.size, delta_r)
    eps = delta_r * w / total
    tiny = float(np.finfo(float).tiny)
    if np.any(eps < tiny):
        eps = np.maximum(eps, tiny)
        eps = eps * (delta_r / eps.sum())
    return eps


def allocate_R2(
    A_hat: Sequence[float],
    K_hat: Sequence[float],
    delta_r: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> np.ndarray:
    """R2 -- general water-filling.

    ``eps_j(t)`` is the smallest ``eps`` with ``U(K_hat_j, A_hat_j, eps) <= t``;
    bisect on the common bound ``t`` until ``sum_j eps_j(t) <= delta_r``.

    The inversion is exact rather than numerical: ``U(K, A, eps) <= t`` iff
    ``eps >= P(Bin(A, t) <= K)``, so ``eps_j(t) = BinomCDF(K_j; A_j, t)``, which
    is non-increasing in ``t``. Hence ``sum_j eps_j(t)`` is non-increasing and the
    smallest feasible ``t`` -- the tightest achievable uniform UCB -- is found by
    bisection on ``[0, 1]``. Reduces to R1's behaviour when every ``K_hat_j = 0``,
    where ``eps_j(t) = (1 - t)^{A_hat_j}``.
    """
    counts = np.asarray(A_hat, dtype=float)
    errors = np.asarray(K_hat, dtype=float)
    _validate_budget(delta_r, counts.size)
    if counts.size != errors.size:
        raise ValueError("A_hat and K_hat must have the same length")
    if np.any(errors < 0) or np.any(errors > counts):
        raise ValueError("need 0 <= K_hat_j <= A_hat_j")

    def eps_of_t(t: float) -> np.ndarray:
        # P(Bin(A, t) <= K); A_j == 0 carries no information -> full budget need.
        out = np.ones(counts.size, dtype=float)
        live = counts > 0
        if np.any(live):
            out[live] = _binom.cdf(errors[live], counts[live], t)
        return out

    lo, hi = 0.0, 1.0
    if float(eps_of_t(hi).sum()) > delta_r:
        # Even t = 1 cannot fit the budget (only possible with A_j == 0 clients).
        return uniform_allocation(counts.size, delta_r)
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        if float(eps_of_t(mid).sum()) <= delta_r:
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance:
            break
    eps = eps_of_t(hi)
    total = float(eps.sum())
    if total > delta_r and total > 0.0:  # numerical guard: never overspend
        eps = eps * (delta_r / total)
    return eps


def allocate_R3(
    A_hat: Sequence[float],
    delta_r: float,
    alpha: float,
    *,
    kappa: float = R3_KAPPA,
) -> np.ndarray:
    """R3 -- reserve safeguard (the pre-registered headline rule).

    ``eps_j = kappa * delta_r / J + (1 - kappa) * delta_r * w_j`` with ``w_j``
    the R1 weights. The reserve keeps every ``eps_j >= kappa * delta_r / J``, so
    the worst-case floor inflation relative to uniform is bounded by
    ``ln(J / (kappa * delta_r)) / ln(J / delta_r)`` -- R1's tail risk (a client
    whose proposal count badly over-predicts its certification count receives an
    exponentially small budget) is capped, at the cost of a small constant factor.
    """
    counts = np.asarray(A_hat, dtype=float)
    J = counts.size
    _validate_budget(delta_r, J)
    if not 0.0 <= kappa <= 1.0:
        raise ValueError(f"kappa must lie in [0, 1], got {kappa!r}")
    w = allocate_R1(counts, delta_r, alpha) / delta_r
    return kappa * delta_r / J + (1.0 - kappa) * delta_r * w


def allocate(
    rule: str,
    *,
    A_hat: Optional[Sequence[float]] = None,
    K_hat: Optional[Sequence[float]] = None,
    J: Optional[int] = None,
    delta_r: float,
    alpha: float,
    kappa: float = R3_KAPPA,
) -> np.ndarray:
    """Dispatch to a predeclared allocation rule by name."""
    if rule not in ALLOCATION_RULES:
        raise ValueError(
            f"unknown allocation rule {rule!r}; expected {ALLOCATION_RULES}"
        )
    if rule == "uniform":
        size = J if J is not None else len(np.asarray(A_hat))
        return uniform_allocation(int(size), delta_r)
    if A_hat is None:
        raise ValueError(f"rule {rule!r} requires proposal accepted counts A_hat")
    if rule == "R1":
        return allocate_R1(A_hat, delta_r, alpha)
    if rule == "R2":
        if K_hat is None:
            raise ValueError("rule 'R2' requires proposal accepted-error counts K_hat")
        return allocate_R2(A_hat, K_hat, delta_r)
    return allocate_R3(A_hat, delta_r, alpha, kappa=kappa)


# --------------------------------------------------------------------------- #
# Archived proposal-allocated clientwise certificate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AllocatedCertificate:
    """Result of the archived proposal-allocated clientwise certificate."""

    U: float
    rbar: np.ndarray
    eps: np.ndarray
    delta_r: float
    rule: str
    feasible: bool
    budget_spent: float
    frontier: float

    @property
    def budget_respected(self) -> bool:
        """Whether the allocation honours ``sum_j eps_j <= delta_r``."""
        return bool(self.budget_spent <= self.delta_r * (1.0 + 1e-9))


def allocated_risk_certificate(
    A: Sequence[int],
    K: Sequence[int],
    eps: Sequence[float],
    delta_r: float,
    *,
    rule: str = "custom",
    alpha: Optional[float] = None,
) -> AllocatedCertificate:
    """Certify ``max_j U(K_j, A_j; eps_j)`` with an archived clientwise allocation.

    ``eps`` must have been computed from the proposal fold only and frozen before
    ``(A, K)`` were observed -- this function cannot verify that, and it is the
    caller's pre-registration duty. Overspending the budget raises rather than
    returning a silently invalid bound.

    A client with ``A_j == 0`` contributes ``rbar_j = 1`` (``cp_upper`` returns
    1.0 with no data), which correctly makes the certificate non-deployable
    rather than dropping the client from the mixture.
    """
    accepted = np.asarray(A, dtype=float)
    errors = np.asarray(K, dtype=float)
    budget = np.asarray(eps, dtype=float)
    J = accepted.size
    if not (errors.size == J and budget.size == J):
        raise ValueError("A, K and eps must have the same length")
    if np.any(budget <= 0.0):
        raise ValueError("every eps_j must be strictly positive")
    spent = float(budget.sum())
    if spent > delta_r * (1.0 + 1e-9):
        raise ValueError(
            f"allocation overspends the risk budget: sum(eps)={spent!r} > delta_r={delta_r!r}"
        )

    rbar = np.array(
        [
            cp_upper(int(errors[j]), int(accepted[j]), float(budget[j]))
            for j in range(J)
        ],
        dtype=float,
    )
    U = float(min(np.max(rbar), 1.0)) if J else 1.0
    frontier = (
        zero_error_frontier(accepted, alpha) if alpha is not None else float("nan")
    )
    return AllocatedCertificate(
        U=U,
        rbar=rbar,
        eps=budget,
        delta_r=float(delta_r),
        rule=rule,
        feasible=bool(np.isfinite(U)),
        budget_spent=spent,
        frontier=frontier,
    )
