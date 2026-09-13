"""Exact optimization over the pre-registered deployment-mixture sets.

The brief defines the group-box mixture set as

    ``Lambda_G(rho) = { lam / sum(lam) : lam in B(rho), lam != 0 }``

the **normalized image** of the componentwise box
``B(rho) = prod_j [c_j - rho, c_j + rho]`` around a predeclared center ``c``.
This is NOT the same set as ``B(rho) & simplex`` (which is what
:func:`fedcore.mixture.rho_mixture_box` builds): for ``J >= 3`` the normalized
image is strictly larger.  With ``J = 3``, ``c = (1/3, 1/3, 1/3)``, ``rho = 0.15``
the vertex ``(0.4833, 0.1833, 0.1833)`` normalizes to ``(0.5686, 0.2157, 0.2157)``,
whose first coordinate exceeds the ``0.4833`` cap that the intersection imposes.
Certifying against the intersection when the declared set is the normalized image
would therefore understate the supremum.  This module targets the normalized
image, matching the brief and matching the (sampled) semantics of the legacy
``_sample_lambdas`` path it replaces.

Why the exact solve is cheap.  Both certified functionals are ratios of linear
forms in ``lam``, hence invariant to positive rescaling of ``lam``:

    ``sup_{lam in Lambda_G(rho)} f(lam) = sup_{lam in B(rho) \\ {0}} f(lam)``

and the supremum of a linear-fractional objective over a box is attained at a
vertex.  Rather than enumerate the ``2^J`` vertices, we use the Dinkelbach
parametric root: the residual ``g(t) = sup_x [N(x) - t D(x)]`` separates across
coordinates, so each evaluation is ``O(J)`` and the root ``g(t*) = 0`` gives
``t* = sup f``.  :func:`vertex_enumeration_reference` provides the ``O(2^J)``
brute-force cross-check that the unit tests pin this against.

Validity note. ``g(t*) = 0`` implies ``N(x) - t* D(x) <= 0`` for every feasible
``x``, so ``N(x)/D(x) <= t*`` whenever ``D(x) > 0``.  The returned value is a
valid upper bound even when the box's closure contains ``D = 0`` points, which is
the conservative direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "NormalizedBox",
    "uniform_box",
    "exact_risk_supremum",
    "exact_coverage_infimum",
    "vertex_enumeration_reference",
]

_TOL = 1e-13
_MAX_ITER = 200


@dataclass(frozen=True)
class NormalizedBox:
    """The mixture set ``{lam / sum(lam) : lo <= lam <= hi, lam != 0}``.

    ``lo``/``hi`` are the RAW box bounds before normalization. ``rho = 0`` yields
    the singleton ``{center}``.
    """

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        lo = np.array(self.lo, dtype=float, copy=True)
        hi = np.array(self.hi, dtype=float, copy=True)
        if lo.ndim != 1 or lo.size == 0 or lo.shape != hi.shape:
            raise ValueError("lo and hi must be non-empty vectors of equal length")
        if np.any(lo < 0.0):
            raise ValueError("raw box lower bounds must be non-negative")
        if np.any(lo > hi):
            raise ValueError("every lower bound must be at most its upper bound")
        if not np.any(hi > 0.0):
            raise ValueError("the box must contain a non-zero mixture")
        lo.setflags(write=False)
        hi.setflags(write=False)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def dimension(self) -> int:
        return int(self.lo.size)

    def vertices(self) -> np.ndarray:
        """All ``2^J`` raw vertices. Only for the reference cross-check."""
        J = self.dimension
        if J > 20:
            raise ValueError(f"refusing to enumerate 2^{J} vertices")
        return np.array(
            [
                [self.hi[j] if bit else self.lo[j] for j, bit in enumerate(mask)]
                for mask in itertools.product((0, 1), repeat=J)
            ],
            dtype=float,
        )


def uniform_box(J: int, rho: float) -> NormalizedBox:
    """``Lambda_G(rho)``: the rho-box around the uniform mixture over ``J`` parts."""
    if J <= 0:
        raise ValueError("need at least one part")
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError(f"rho must be finite and non-negative, got {rho!r}")
    u = 1.0 / J
    return NormalizedBox(
        np.maximum(0.0, np.full(J, u - rho)),
        np.minimum(1.0, np.full(J, u + rho)),
    )


def _dinkelbach_root(residual, lo_t: float, hi_t: float) -> float:
    """Root of a non-increasing residual on ``[lo_t, hi_t]`` by bisection."""
    if residual(lo_t) <= 0.0:
        return lo_t
    if residual(hi_t) >= 0.0:
        return hi_t
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo_t + hi_t)
        if residual(mid) >= 0.0:
            lo_t = mid
        else:
            hi_t = mid
        if hi_t - lo_t < _TOL:
            break
    return 0.5 * (lo_t + hi_t)


def exact_risk_supremum(
    rbar: Sequence[float],
    alow: Sequence[float],
    ahigh: Sequence[float],
    box: NormalizedBox,
) -> Tuple[float, bool]:
    """``sup_{lam in Lambda, a in [alow, ahigh]} (sum lam a rbar) / (sum lam a)``.

    Returns ``(value, feasible)``. ``feasible`` is False -- and the value
    ``+inf`` -- only when no feasible point has a strictly positive denominator,
    i.e. the accepted mass cannot be bounded away from zero; per the brief such a
    cell REFUSES certification rather than dropping a client.
    """
    r = np.asarray(rbar, dtype=float)
    lo_a = np.asarray(alow, dtype=float)
    hi_a = np.asarray(ahigh, dtype=float)
    J = box.dimension
    if not (r.size == lo_a.size == hi_a.size == J):
        raise ValueError("rbar, alow, ahigh and the box must share a dimension")
    if np.any(lo_a < 0.0) or np.any(hi_a > 1.0) or np.any(lo_a > hi_a):
        raise ValueError("acceptance box must satisfy 0 <= alow <= ahigh <= 1")

    # Largest achievable denominator; if it is zero no feasible point has mass.
    if float(np.sum(box.hi * hi_a)) <= 0.0:
        return math.inf, False

    def residual(t: float) -> float:
        # Separable: maximize lam_j * a_j * (rbar_j - t) coordinatewise, with
        # lam_j * a_j ranging over [lo_j * alow_j, hi_j * ahigh_j].
        gain = r - t
        hi_prod = box.hi * hi_a
        lo_prod = box.lo * lo_a
        return float(np.sum(np.where(gain > 0.0, hi_prod * gain, lo_prod * gain)))

    value = _dinkelbach_root(residual, float(r.min()), float(r.max()))
    # The simplex bound max_j rbar_j dominates every mixture; enforce it exactly.
    return float(min(value, float(r.max()), 1.0)), True


def exact_coverage_infimum(
    acceptance_lower: Sequence[float], box: NormalizedBox
) -> float:
    """``inf_{lam in Lambda} sum_j lam_j * alow_j`` (Corollary 1).

    Over the normalized image this is ``inf_{lam in B} (sum lam alow)/(sum lam)``,
    a weighted average of ``alow`` and therefore again linear-fractional; the
    infimum is attained at a raw-box vertex and found by the same parametric root.
    """
    L = np.asarray(acceptance_lower, dtype=float)
    if L.size != box.dimension:
        raise ValueError("acceptance_lower must match the box dimension")
    if np.any(L < 0.0) or np.any(L > 1.0):
        raise ValueError("acceptance_lower must lie in [0, 1]")

    def residual(t: float) -> float:
        # min_lam sum lam_j (L_j - t); negate to reuse the non-increasing root.
        gain = L - t
        return -float(np.sum(np.where(gain < 0.0, box.hi * gain, box.lo * gain)))

    # residual(t) = -min(...) is non-decreasing in t; bisect on its negation.
    def neg(t: float) -> float:
        return -residual(t)

    return float(
        max(min(_dinkelbach_root(neg, float(L.min()), float(L.max())), 1.0), 0.0)
    )


def vertex_enumeration_reference(
    rbar: Sequence[float],
    alow: Sequence[float],
    ahigh: Sequence[float],
    box: NormalizedBox,
) -> float:
    """Brute-force ``O(2^J * 2^J)`` reference supremum, for tests only.

    Enumerates every raw-box vertex against every acceptance-box vertex. This is
    the ground truth that :func:`exact_risk_supremum` is pinned against, and the
    check the brief requires ("LFP solver: box-Lambda_G supremum equals exact
    vertex-enumeration value on random small instances").
    """
    r = np.asarray(rbar, dtype=float)
    lo_a = np.asarray(alow, dtype=float)
    hi_a = np.asarray(ahigh, dtype=float)
    J = box.dimension
    best = -math.inf
    for lam in box.vertices():
        if lam.sum() <= 0.0:
            continue
        for mask in itertools.product((0, 1), repeat=J):
            a = np.where(np.array(mask, dtype=bool), hi_a, lo_a)
            denom = float(np.sum(lam * a))
            if denom <= 0.0:
                continue
            best = max(best, float(np.sum(lam * a * r) / denom))
    return best if math.isfinite(best) else math.inf
