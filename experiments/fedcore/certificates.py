"""Conformal-prediction (CP) primitives and Fed-CORE selective-risk certificates.

This module implements the *certification core* of Fed-CORE. It depends only on
``numpy`` + ``scipy`` (no torch), so it runs on any CPU.

Three certificates live here:

* :func:`conditional_risk_certificate` -- **Theorem 1/1'**, the MAIN result.
  Bounds the per-client conditional error rate ``r_j`` directly via the law
  ``K_j | A_j ~ Bin(A_j, r_j)`` and solves a robust linear-fractional program
  over the deployment mixture ``lambda`` (and, for bounded ``Lambda``, over the
  acceptance-probability box). Uniformly tighter than the mass-ratio baseline.
* :func:`stratified_certificate` -- the **Appendix-C mass-ratio baseline ONLY**.
  Bounds ``m_j = P_j(accept & error)`` and ``a_j = P_j(accept)`` separately and
  forms the ratio. Kept for comparison; do not promote it above Theorem 1/1'.
* :func:`pooled_cp` -- **Proposition 3**, the subordinate pooled bound. Valid
  only under matched-mixture i.i.d. calibration (Poisson-binomial caveat).

Notation: ``A_j`` accepted count, ``K_j`` accepted-and-wrong count, ``n_j`` total
calibration count for client ``j``; ``J`` number of clients; ``delta`` overall
failure probability; ``alpha`` target selective risk.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
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


# --------------------------------------------------------------------------- #
# Theorem 1/1' -- CONDITIONAL selective-risk certificate (MAIN)
# --------------------------------------------------------------------------- #
@dataclass
class ConditionalCertificate:
    """Result of :func:`conditional_risk_certificate`."""

    U: float
    rbar: np.ndarray
    eps: float
    feasible: bool
    Lambda: str = "simplex"
    alow: Optional[np.ndarray] = None
    ahigh: Optional[np.ndarray] = None


def _inner_sup_over_a(
    lam: np.ndarray,
    rbar: np.ndarray,
    alow: np.ndarray,
    ahigh: np.ndarray,
) -> float:
    """Inner ``sup`` of ``(sum lam a rbar)/(sum lam a)`` over the a-box.

    The objective is linear-fractional in ``a``, so its supremum over a box is
    attained at a vertex. Enumerate the ``2^J`` vertices (each ``a_j`` is either
    ``alow_j`` or ``ahigh_j``). A vertex with non-positive denominator means the
    denominator bound vanishes -> the program is infeasible there -> ``+inf``.
    """
    J = len(lam)
    best = -np.inf
    for bits in itertools.product((0, 1), repeat=J):
        a = np.where(np.array(bits, dtype=bool), ahigh, alow)
        denom = float(np.sum(lam * a))
        if denom <= 0.0:
            return np.inf
        ratio = float(np.sum(lam * a * rbar) / denom)
        if ratio > best:
            best = ratio
    return best


def conditional_risk_certificate(
    A: Sequence[int],
    K: Sequence[int],
    n: Sequence[int],
    delta: float,
    Lambda: str = "simplex",
    lam: Optional[Sequence[float]] = None,
    box: Optional[Union[float, Sequence[float]]] = None,
    n_lam_samples: int = 256,
    seed: int = 0,
) -> ConditionalCertificate:
    """Theorem 1/1': conditional selective-risk upper certificate.

    Uses ``K_j | A_j ~ Bin(A_j, r_j)`` so that ``rbar_j = U+(K_j, A_j; eps)``
    upper-bounds the per-client conditional error rate. The certified selective
    risk ``U`` upper-bounds ``R_sel(lambda)`` for every ``lambda in Lambda``.

    Parameters
    ----------
    A, K, n : per-client accepted, accepted-and-wrong, and total counts.
    delta : overall failure probability (split across clients / bounds).
    Lambda : ``'simplex'`` (full simplex, closed form ``max_j rbar_j``),
        ``'known'`` (a single fixed ``lam``), or ``'box'`` (sup over a sampled
        box of mixtures around uniform).
    lam : required for ``Lambda='known'``.
    box : additive radius around the uniform mixture for ``Lambda='box'``.
    """
    A = np.asarray(A, dtype=float)
    K = np.asarray(K, dtype=float)
    n = np.asarray(n, dtype=float)
    J = len(A)

    if Lambda == "simplex":
        eps = delta / J
        # rbar_j = 1.0 when A_j == 0 (cp_upper handles n<=0 -> 1.0).
        rbar = np.array(
            [cp_upper(int(K[j]), int(A[j]), eps) for j in range(J)], dtype=float
        )
        U = float(min(np.max(rbar), 1.0))
        return ConditionalCertificate(
            U=U, rbar=rbar, eps=eps, feasible=np.isfinite(U), Lambda="simplex"
        )

    if Lambda in ("box", "known"):
        eps = delta / (3.0 * J)
        rbar = np.array(
            [cp_upper(int(K[j]), int(A[j]), eps) for j in range(J)], dtype=float
        )
        alow = np.array(
            [cp_lower(int(A[j]), int(n[j]), eps) for j in range(J)], dtype=float
        )
        ahigh = np.array(
            [cp_upper(int(A[j]), int(n[j]), eps) for j in range(J)], dtype=float
        )

        if Lambda == "known":
            if lam is None:
                raise ValueError("Lambda='known' requires `lam`.")
            lam_arr = np.asarray(lam, dtype=float)
            val = _inner_sup_over_a(lam_arr, rbar, alow, ahigh)
        else:  # box
            radius = _resolve_box_radius(box)
            rng = np.random.default_rng(seed)
            lams = _sample_lambdas(J, radius, n_lam_samples, rng)
            val = -np.inf
            for lam_arr in lams:
                v = _inner_sup_over_a(lam_arr, rbar, alow, ahigh)
                if v > val:
                    val = v

        feasible = np.isfinite(val)
        U = float(min(val, 1.0)) if feasible else np.inf
        return ConditionalCertificate(
            U=U, rbar=rbar, eps=eps, feasible=bool(feasible),
            Lambda=Lambda, alow=alow, ahigh=ahigh,
        )

    raise ValueError(f"unknown Lambda={Lambda!r}")


# --------------------------------------------------------------------------- #
# Appendix C -- mass-ratio STRATIFIED certificate (BASELINE ONLY)
# --------------------------------------------------------------------------- #
@dataclass
class StratifiedCertificate:
    """Result of :func:`stratified_certificate` (mass-ratio baseline)."""

    U: float
    mbar: np.ndarray
    alow: np.ndarray
    eps: float
    Lambda: str = "simplex"


def stratified_certificate(
    A: Sequence[int],
    K: Sequence[int],
    n: Sequence[int],
    delta: float,
    Lambda: str = "simplex",
    lam: Optional[Sequence[float]] = None,
    box: Optional[Union[float, Sequence[float]]] = None,
    n_lam_samples: int = 256,
    seed: int = 0,
) -> StratifiedCertificate:
    """Appendix-C mass-ratio baseline (NOT the main certificate).

    Bounds ``m_j = P_j(accept & error)`` by ``mbar_j = U+(K_j, n_j; eps)`` and
    ``a_j = P_j(accept)`` from below by ``alow_j = U-(A_j, n_j; eps)``, then
    forms ``sup_lambda (sum lam mbar)/(sum lam alow)``. The simplex sup has the
    closed form ``max_j mbar_j / alow_j``. Looser than Theorem 1/1'.
    """
    A = np.asarray(A, dtype=float)
    K = np.asarray(K, dtype=float)
    n = np.asarray(n, dtype=float)
    J = len(A)

    eps = delta / (2.0 * J)
    mbar = np.array(
        [cp_upper(int(K[j]), int(n[j]), eps) for j in range(J)], dtype=float
    )
    alow = np.array(
        [cp_lower(int(A[j]), int(n[j]), eps) for j in range(J)], dtype=float
    )

    if Lambda == "simplex":
        ratios = [mbar[j] / alow[j] for j in range(J) if alow[j] > 0.0]
        U = max(ratios) if ratios else np.inf
    elif Lambda == "known":
        if lam is None:
            raise ValueError("Lambda='known' requires `lam`.")
        lam_arr = np.asarray(lam, dtype=float)
        denom = float(np.sum(lam_arr * alow))
        U = float(np.sum(lam_arr * mbar) / denom) if denom > 0 else np.inf
    elif Lambda == "box":
        radius = _resolve_box_radius(box)
        rng = np.random.default_rng(seed)
        lams = _sample_lambdas(J, radius, n_lam_samples, rng)
        U = -np.inf
        for lam_arr in lams:
            denom = float(np.sum(lam_arr * alow))
            v = float(np.sum(lam_arr * mbar) / denom) if denom > 0 else np.inf
            if v > U:
                U = v
    else:
        raise ValueError(f"unknown Lambda={Lambda!r}")

    U = float(min(U, 1.0))
    return StratifiedCertificate(U=U, mbar=mbar, alow=alow, eps=eps, Lambda=Lambda)


# --------------------------------------------------------------------------- #
# Proposition 3 -- pooled CP (subordinate) and ground-truth helpers
# --------------------------------------------------------------------------- #
def pooled_cp(A: Sequence[int], K: Sequence[int], delta: float) -> float:
    """Proposition 3: pooled selective-risk bound ``U+(sum K, sum A; delta)``.

    Valid only under matched-mixture i.i.d. calibration. Invalid under
    heterogeneity (pooled accepted-error count is Poisson-binomial). Subordinate
    to Theorem 1/1'.
    """
    A = np.asarray(A)
    K = np.asarray(K)
    return cp_upper(int(K.sum()), int(A.sum()), delta)


def true_selective_risk(
    a: Sequence[float], r: Sequence[float], lam: Sequence[float]
) -> float:
    """Ground-truth ``R_sel(lambda) = sum(lam a r) / sum(lam a)`` (for sims)."""
    a = np.asarray(a, dtype=float)
    r = np.asarray(r, dtype=float)
    lam = np.asarray(lam, dtype=float)
    denom = float(np.sum(lam * a))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(lam * a * r) / denom)
