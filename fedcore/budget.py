"""Failure-budget plans and proposal-only stratum allocations.

The legacy certification path historically spent ``delta`` on the risk bound and
then another ``delta / 2`` on a coverage bound.  That is fine for the risk-only
deployment decision, but it is not a joint ``1-delta`` risk-and-coverage claim.
This module makes every simultaneous component explicit and validates the sum.

Allocations may depend on the independent proposal fold.  They must never depend
on certification labels.  Callers therefore pass proposal counts explicitly and
receive immutable per-stratum epsilon vectors to freeze in an artifact manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from fedcore.counts import strict_count_vector


_COMPONENTS = (
    "mixture",
    "conditional_risk",
    "acceptance_lower",
    "acceptance_upper",
)


@dataclass(frozen=True)
class FailureBudget:
    """A complete simultaneous failure-budget declaration.

    ``acceptance_lower`` can be reused for both the robust risk denominator and
    the reported coverage LCB because it is the same confidence event.  It is
    declared and spent once.  Unused components should be exactly zero.
    """

    total: float
    mixture: float = 0.0
    conditional_risk: float = 0.0
    acceptance_lower: float = 0.0
    acceptance_upper: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < self.total < 1.0:
            raise ValueError("total must lie strictly between zero and one")
        if self.spent > self.total + 1e-15:
            raise ValueError(
                f"failure-budget components sum to {self.spent:.17g}, "
                f"which exceeds total={self.total:.17g}"
            )

    @property
    def spent(self) -> float:
        return float(sum(getattr(self, name) for name in _COMPONENTS))

    @property
    def slack(self) -> float:
        return float(self.total - self.spent)

    def as_dict(self) -> dict[str, float]:
        out = {"delta_total": float(self.total)}
        out.update(
            {f"delta_{name}": float(getattr(self, name)) for name in _COMPONENTS}
        )
        out["delta_spent"] = self.spent
        out["delta_slack"] = self.slack
        return out


def make_failure_budget(
    total: float,
    *,
    include_mixture: bool,
    include_acceptance_box: bool,
) -> FailureBudget:
    """Return a conservative equal-component default for a joint claim.

    The exact fractions are experiment policy and should normally be supplied by
    the governing run plan.  This helper exists for smoke tests and safe defaults:
    it divides ``total`` equally over precisely the confidence-event families that
    are used, never spending an undeclared extra budget.
    """

    names = ["conditional_risk", "acceptance_lower"]
    if include_acceptance_box:
        names.append("acceptance_upper")
    if include_mixture:
        names.append("mixture")
    share = float(total) / len(names)
    values = {name: (share if name in names else 0.0) for name in _COMPONENTS}
    return FailureBudget(total=float(total), **values)


def _validate_proposal_counts(
    A: Sequence[int], K: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    A_arr = strict_count_vector(A, "proposal A")
    K_arr = strict_count_vector(K, "proposal K")
    if K_arr.shape != A_arr.shape:
        raise ValueError("proposal A and K must be aligned vectors")
    if np.any(K_arr > A_arr):
        raise ValueError("proposal counts must satisfy 0 <= K_j <= A_j")
    return A_arr, K_arr


def allocate_stratum_budget(
    component_delta: float,
    proposal_A: Sequence[int],
    proposal_K: Sequence[int],
    *,
    policy: str = "uniform",
    floor_fraction: float = 0.10,
) -> np.ndarray:
    """Freeze per-stratum epsilons using proposal counts only.

    ``proposal_informed`` assigns more failure probability (hence a tighter CP
    upper bound) to proposal-side bottlenecks: strata with high smoothed error or
    small accepted count.  A uniform floor keeps every simultaneous event active.
    The returned vector sums to ``component_delta`` up to floating-point roundoff.
    """

    A, K = _validate_proposal_counts(proposal_A, proposal_K)
    J = len(A)
    delta = float(component_delta)
    if not np.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("component_delta must lie strictly between zero and one")
    if not 0.0 <= floor_fraction <= 1.0:
        raise ValueError("floor_fraction must lie in [0, 1]")

    if policy == "uniform":
        weights = np.full(J, 1.0 / J, dtype=float)
    elif policy == "proposal_informed":
        # Jeffreys-smoothed error plus a sparse-count term.  Both are functions of
        # the proposal fold only; no certification observation enters this rule.
        smoothed_risk = (K.astype(float) + 0.5) / (A.astype(float) + 1.0)
        sparse = 1.0 / np.sqrt(A.astype(float) + 1.0)
        hardness = smoothed_risk + sparse
        if not np.any(hardness > 0.0):
            informed = np.full(J, 1.0 / J)
        else:
            informed = hardness / hardness.sum()
        weights = (
            floor_fraction * np.full(J, 1.0 / J) + (1.0 - floor_fraction) * informed
        )
    else:
        raise ValueError(f"unknown allocation policy {policy!r}")

    eps = delta * weights
    # Force an exact declared sum without changing proposal measurability.
    eps[-1] += delta - float(eps.sum())
    if np.any(eps <= 0.0) or not np.isclose(eps.sum(), delta, rtol=0.0, atol=1e-15):
        raise RuntimeError("internal failure-budget allocation error")
    return eps


def allocate_failure_budget(
    budget: FailureBudget,
    proposal_A: Sequence[int],
    proposal_K: Sequence[int],
    *,
    policy: str,
) -> Mapping[str, np.ndarray]:
    """Return every active per-stratum epsilon vector for one frozen policy."""

    A, K = _validate_proposal_counts(proposal_A, proposal_K)
    out: dict[str, np.ndarray] = {}
    for name in ("conditional_risk", "acceptance_lower", "acceptance_upper"):
        value = getattr(budget, name)
        if value > 0.0:
            out[name] = allocate_stratum_budget(value, A, K, policy=policy)
    return out
