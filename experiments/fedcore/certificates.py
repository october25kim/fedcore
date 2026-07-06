"""Finite-sample certificates for the federated accepted selective risk (Fed-CORE).

This module implements the certificates analyzed in the Fed-CORE draft:

  * Clopper-Pearson one-sided bounds (exact, distribution-free for a binomial).
  * `stratified_certificate`  -- Theorem 1: a worst-case-mixture upper bound on
    the deployment accepted selective risk, valid under partial exchangeability
    across heterogeneous clients with unknown mixture weights lambda in Lambda.
  * `pooled_cp`               -- the pooled single-binomial bound. Used both as
    the Theorem-3 certificate (valid only under matched-lambda partial
    exchangeability) and as the *naive* baseline whose failure under mixture
    shift demonstrates the non-reducibility of Theorem 1.

Notation (matching the draft):
  a_j = P_{P_j}(accept)                 per-client acceptance rate
  r_j = P_{P_j}(error | accept)         per-client conditional selective risk
  m_j = a_j r_j = P_{P_j}(accept, error) per-client accepted-error mass
  R_sel(lambda) = sum_j lam_j m_j / sum_j lam_j a_j   global accepted selective risk

Observed on the certification fold (secure-aggregatable counts):
  A_j ~ Binomial(n_j, a_j),  K_j ~ Binomial(n_j, m_j),  K_j <= A_j.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import beta


# --------------------------------------------------------------------------- #
# Clopper-Pearson one-sided limits                                            #
# --------------------------------------------------------------------------- #
def cp_upper(k: int, n: int, eps: float) -> float:
    """One-sided upper Clopper-Pearson limit at level ``eps`` for P(success).

    For ``K ~ Binomial(n, p)`` this returns ``U`` with ``P(p <= U) >= 1 - eps``.
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - eps, k + 1, n - k))


def cp_lower(k: int, n: int, eps: float) -> float:
    """One-sided lower Clopper-Pearson limit at level ``eps`` for P(success).

    For ``K ~ Binomial(n, p)`` this returns ``L`` with ``P(p >= L) >= 1 - eps``.
    """
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    return float(beta.ppf(eps, k, n - k + 1))


# --------------------------------------------------------------------------- #
# Theorem 1 -- stratified worst-case-mixture certificate                      #
# --------------------------------------------------------------------------- #
@dataclass
class StratifiedCertificate:
    """Result of the stratified certificate computation."""

    U: float                 # the certificate value  (deploy iff U <= alpha)
    mbar: np.ndarray         # per-client upper bounds on m_j
    alow: np.ndarray         # per-client lower bounds on a_j
    eps: float               # per-event Clopper-Pearson level (delta / 2J)


def stratified_certificate(
    A: np.ndarray,
    K: np.ndarray,
    n: np.ndarray,
    delta: float,
    Lambda: Literal["simplex", "known", "box"] = "simplex",
    lam: np.ndarray | None = None,
    box: tuple[np.ndarray, np.ndarray] | None = None,
    n_box_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> StratifiedCertificate:
    """Theorem 1 certificate on the deployment accepted selective risk.

    Parameters
    ----------
    A, K, n : per-client accepted count, accepted-error count, cert-fold size.
    delta   : certificate failure probability (confidence 1 - delta).
    Lambda  : admissible deployment-weight set.
        - 'simplex' : full simplex  -> U = max_j mbar_j / alow_j   (closed form).
        - 'known'   : a single known weight vector ``lam``.
        - 'box'     : a box [lo, hi] (intersected with the simplex); the
                      worst case is approximated by sampling.
    Returns
    -------
    StratifiedCertificate with the value ``U`` and the per-client limits.
    """
    A = np.asarray(A, dtype=float)
    K = np.asarray(K, dtype=float)
    n = np.asarray(n, dtype=float)
    J = len(A)
    eps = delta / (2.0 * J)

    mbar = np.array([cp_upper(int(K[j]), int(n[j]), eps) for j in range(J)])
    alow = np.array([cp_lower(int(A[j]), int(n[j]), eps) for j in range(J)])

    if Lambda == "simplex":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(alow > 0, mbar / alow, np.inf)
        U = float(np.max(ratios))
    elif Lambda == "known":
        assert lam is not None, "lam required for Lambda='known'"
        lam = np.asarray(lam, dtype=float)
        U = float((lam * mbar).sum() / (lam * alow).sum())
    elif Lambda == "box":
        assert box is not None, "box=(lo, hi) required for Lambda='box'"
        rng = rng or np.random.default_rng(0)
        lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
        best = 0.0
        for _ in range(n_box_samples):
            ll = lo + (hi - lo) * rng.random(J)
            s = ll.sum()
            if s <= 0:
                continue
            ll = ll / s
            denom = (ll * alow).sum()
            if denom > 0:
                best = max(best, (ll * mbar).sum() / denom)
        U = float(best)
    else:  # pragma: no cover
        raise ValueError(f"unknown Lambda={Lambda!r}")

    # a selective risk is a probability: cap the (possibly vacuous) bound at 1.
    U = float(min(U, 1.0))
    return StratifiedCertificate(U=U, mbar=mbar, alow=alow, eps=eps)


# --------------------------------------------------------------------------- #
# Pooled certificate (Theorem 3 / naive baseline)                             #
# --------------------------------------------------------------------------- #
def pooled_cp(A: np.ndarray, K: np.ndarray, delta: float) -> float:
    """Pooled single-binomial Clopper-Pearson upper bound on the accepted risk.

    Computes ``cp_upper(sum_j K_j, sum_j A_j, delta)``.

    Interpretation depends on context:
      * Theorem 3 (valid) when deployment lambda matches the calibration
        accepted-count proportions and Lemma L holds.
      * NAIVE baseline (can be invalid) when deployment lambda shifts away from
        the calibration proportions -- this is the failure the ablation exposes.
    """
    A = np.asarray(A)
    K = np.asarray(K)
    return cp_upper(int(K.sum()), int(A.sum()), delta)


# --------------------------------------------------------------------------- #
# CONDITIONAL selective-risk certificate (the sharper MAIN certificate)       #
# --------------------------------------------------------------------------- #
@dataclass
class ConditionalRiskCertificate:
    """Result of the conditional selective-risk certificate (new Theorem 1)."""

    U: float                 # certificate value (deploy iff U <= alpha; +inf = infeasible)
    rbar: np.ndarray         # per-client upper bounds on the conditional risk r_j
    alow: np.ndarray         # per-client lower bounds on a_j (bounded-Lambda only)
    ahigh: np.ndarray        # per-client upper bounds on a_j (bounded-Lambda only)
    eps: float               # per-event Clopper-Pearson level
    feasible: bool


def conditional_risk_certificate(
    A: np.ndarray,
    K: np.ndarray,
    n: np.ndarray,
    delta: float,
    Lambda: Literal["simplex", "known", "box"] = "simplex",
    lam: np.ndarray | None = None,
    box: tuple[np.ndarray, np.ndarray] | None = None,
    n_lam_samples: int = 4000,
    rng: np.random.Generator | None = None,
) -> ConditionalRiskCertificate:
    """Sharper certificate via the CONDITIONAL law ``K_j | A_j ~ Bin(A_j, r_j)``.

    This bounds the per-client selective risk ``r_j = P_j(error | accept)``
    directly with ``rbar_j = U+(K_j, A_j; eps)`` -- no denominator slack from a
    separate acceptance lower bound. The global risk is a convex combination
    ``R_sel(lambda) = sum_j w_j(lambda) r_j`` with acceptance-reweighted weights
    ``w_j = lambda_j a_j / sum_l lambda_l a_l``, so:

      * Lambda = simplex : worst case concentrates all weight on the worst client
        => ``U = max_j rbar_j``  (eps = delta/J; only ONE event per client).
      * Lambda = box/known : the weights are constrained; we additionally bound
        ``a_j in [alow_j, ahigh_j]`` and solve the robust linear-fractional
        program ``sup_{lambda in Lambda, a in box} (sum lam a rbar)/(sum lam a)``
        (eps = delta/3J: rbar_j, alow_j, ahigh_j). The inner sup over ``a`` is
        attained at a box vertex (enumerated); the outer sup over lambda is sampled.

    Returns ``U = +inf`` (feasible=False) when no client has any accepted points
    under a deployment that the certificate must cover (denominator can vanish).
    """
    A = np.asarray(A, float); K = np.asarray(K, float); n = np.asarray(n, float)
    J = len(A)

    if Lambda == "simplex":
        eps = delta / J
        rbar = np.array([cp_upper(int(K[j]), int(A[j]), eps) if A[j] > 0 else 1.0
                         for j in range(J)])
        U = float(np.max(rbar))
        return ConditionalRiskCertificate(
            U=min(U, 1.0), rbar=rbar, alow=np.zeros(J), ahigh=np.ones(J),
            eps=eps, feasible=True,
        )

    # bounded Lambda (box) or known lambda: need acceptance intervals too
    eps = delta / (3.0 * J)
    rbar = np.array([cp_upper(int(K[j]), int(A[j]), eps) if A[j] > 0 else 1.0
                     for j in range(J)])
    alow = np.array([cp_lower(int(A[j]), int(n[j]), eps) for j in range(J)])
    ahigh = np.array([cp_upper(int(A[j]), int(n[j]), eps) for j in range(J)])

    # enumerate a-vertices (each a_j in {alow_j, ahigh_j}); J is small in FL
    import itertools
    a_vertices = [np.where(np.array(bits, bool), ahigh, alow)
                  for bits in itertools.product([0, 1], repeat=J)]

    def sup_over_a(lam_vec: np.ndarray) -> float:
        best_local, feasible_local = 0.0, False
        for a in a_vertices:
            denom = float((lam_vec * a).sum())
            if denom > 0:
                feasible_local = True
                best_local = max(best_local, float((lam_vec * a * rbar).sum()) / denom)
        return best_local if feasible_local else np.inf

    if Lambda == "known":
        assert lam is not None
        U = sup_over_a(np.asarray(lam, float))
    else:  # box
        assert box is not None
        rng = rng or np.random.default_rng(0)
        lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
        U = 0.0
        for _ in range(n_lam_samples):
            ll = lo + (hi - lo) * rng.random(J)
            s = ll.sum()
            if s > 0:
                U = max(U, sup_over_a(ll / s))
    feasible = np.isfinite(U)
    return ConditionalRiskCertificate(
        U=float(min(U, 1.0)) if feasible else float("inf"),
        rbar=rbar, alow=alow, ahigh=ahigh, eps=eps, feasible=bool(feasible),
    )


# --------------------------------------------------------------------------- #
# Ground-truth deployment risk                                                #
# --------------------------------------------------------------------------- #
def true_selective_risk(a: np.ndarray, r: np.ndarray, lam: np.ndarray) -> float:
    """R_sel(lambda) = sum_j lam_j a_j r_j / sum_j lam_j a_j."""
    a = np.asarray(a, float)
    r = np.asarray(r, float)
    lam = np.asarray(lam, float)
    m = a * r
    return float((lam * m).sum() / (lam * a).sum())
