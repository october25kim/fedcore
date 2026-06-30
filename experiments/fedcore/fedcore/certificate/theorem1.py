"""Theorem 1/1' conditional selective-risk certificate (MAIN) + the App-C stratified
mass-ratio baseline.

* :func:`conditional_risk_certificate` -- Theorem 1 (simplex) / Theorem 1' (box,
  robust linear-fractional over the acceptance-probability box). THE main result.
* :func:`stratified_certificate` -- Appendix-C mass-ratio baseline ONLY (kept for
  comparison; do not promote above Theorem 1/1').

Code moved verbatim from the original flat ``certificates.py`` -- behaviour unchanged.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .cp import _resolve_box_radius, _sample_lambdas, cp_lower, cp_upper


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
