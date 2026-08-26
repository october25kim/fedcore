"""Current Fed-CORE certificates and a clearly labelled legacy baseline.

The full-simplex theorem has a scalar-extremum/intersection-union structure.  For
one selector fixed independently of the certification fold, every stratum may use
the *member-level* tail budget: there is no additional division by the number of
strata.  A proposal-frozen family of ``M`` members divides by ``M`` only.

The strict bounded-mixture branch is different.  It needs simultaneous risk and
acceptance endpoints, so its risk-side tail is divided across ``3S`` events.

``conditional_risk_certificate`` is kept as the compact risk-only API used by the
older experiment runners.  Its full-simplex default now follows the current
theorem.  Set ``legacy_simplex_union_bound=True`` only to reproduce the archived
``delta/S`` campaign calculation.  The sampled bounded-mixture maximum used by
the old public snapshot has been replaced by a deterministic normalized-box
solve whose numerical output is a conservative bisection endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .cp import _resolve_box_radius, cp_lower, cp_upper
from .lambda_sets import NormalizedBox, solve_normalized_box_risk, uniform_box


# --------------------------------------------------------------------------- #
# Current full-simplex / strict-mixture conditional certificate
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
    method: str = ""
    solver_status: str = "closed_form"
    solver_certificate_valid: bool = True
    solver_tolerance: float = 0.0
    solver_iterations: int = 0
    solver_bracket_lower: float = float("nan")
    solver_bracket_upper: float = float("nan")
    solver_residual_lower: float = float("nan")
    solver_residual_upper: float = float("nan")
    solver_witness_value: float = float("nan")
    solver_reason: str = ""


@dataclass
class FullSimplexCertificate:
    """Risk and coverage bounds for one proposal-frozen family member.

    ``family_size=1`` is the fixed-selector theorem.  For simple simultaneous
    certification of a frozen family, call once per member with the common
    predeclared ``family_size=M``.  The resulting tails are ``delta_r/M`` and
    ``delta_c/M``; they are deliberately *not* divided by the number of strata.
    """

    risk_ucb: float
    coverage_lcb: float
    rbar: np.ndarray
    acceptance_lower: np.ndarray
    risk_tail: float
    coverage_tail: float
    family_size: int
    positive_coverage: bool


def _count_vectors(
    A: Sequence[int], K: Sequence[int], n: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return one-dimensional integer count vectors."""
    arrays = [np.asarray(x) for x in (A, K, n)]
    if any(x.ndim != 1 for x in arrays):
        raise ValueError("A, K, and n must be one-dimensional")
    if not arrays[0].size or not (arrays[0].size == arrays[1].size == arrays[2].size):
        raise ValueError("A, K, and n must be non-empty and have equal length")
    if any(np.any(~np.isfinite(x.astype(float))) for x in arrays):
        raise ValueError("counts must be finite")
    if any(np.any(x.astype(float) != np.floor(x.astype(float))) for x in arrays):
        raise ValueError("counts must be integers")
    A_i, K_i, n_i = (x.astype(np.int64) for x in arrays)
    if np.any(K_i < 0) or np.any(A_i < 0) or np.any(n_i < 0):
        raise ValueError("counts must be non-negative")
    if np.any(K_i > A_i) or np.any(A_i > n_i):
        raise ValueError("counts must satisfy 0 <= K <= A <= n per stratum")
    return A_i, K_i, n_i


def full_simplex_fixed_member_certificate(
    A: Sequence[int],
    K: Sequence[int],
    n: Sequence[int],
    *,
    delta_r: float,
    delta_c: float,
    family_size: int = 1,
) -> FullSimplexCertificate:
    """Theorem-1/Corollary-1 bounds for a proposal-frozen family member.

    The fixed member's risk target is ``max_s r_s`` and its coverage target is
    ``min_s a_s``.  If the maximum UCB misses its scalar target, the marginal
    bound for a fixed true worst stratum must have failed; the analogous argument
    applies to the minimum coverage target.  Hence every stratum uses the full
    member-level tail.  This does *not* produce simultaneous stratumwise
    intervals and does not apply to a strict bounded-mixture program.
    """
    A_i, K_i, n_i = _count_vectors(A, K, n)
    if not (0.0 < delta_r < 1.0 and 0.0 < delta_c < 1.0):
        raise ValueError("delta_r and delta_c must lie in (0, 1)")
    if isinstance(family_size, bool) or int(family_size) != family_size or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    M = int(family_size)
    beta_r = float(delta_r) / M
    beta_c = float(delta_c) / M
    rbar = np.array(
        [cp_upper(int(k), int(a), beta_r) for a, k in zip(A_i, K_i)], dtype=float
    )
    alow = np.array(
        [cp_lower(int(a), int(nn), beta_c) for a, nn in zip(A_i, n_i)], dtype=float
    )
    U = float(min(float(np.max(rbar)), 1.0))
    C = float(max(float(np.min(alow)), 0.0))
    return FullSimplexCertificate(
        risk_ucb=U,
        coverage_lcb=C,
        rbar=rbar,
        acceptance_lower=alow,
        risk_tail=beta_r,
        coverage_tail=beta_c,
        family_size=M,
        positive_coverage=bool(C > 0.0),
    )


def _inner_sup_over_a(
    lam: np.ndarray,
    rbar: np.ndarray,
    alow: np.ndarray,
    ahigh: np.ndarray,
) -> float:
    """Inner ``sup`` of ``(sum lam a rbar)/(sum lam a)`` over the a-box.

    The objective is linear-fractional in ``a``, so its supremum over the box is
    attained at a vertex (each ``a_j`` is either ``alow_j`` or ``ahigh_j``). With
    ``lam_j >= 0`` and ``ahigh_j >= alow_j``, the *maximizing* vertex sets
    ``a_j = ahigh_j`` exactly for the clients whose ``rbar_j`` exceeds the optimal
    ratio (Dinkelbach: at the optimum ``t*``, ``a*`` maximizes
    ``sum lam_j a_j (rbar_j - t*)``, which picks ``ahigh_j`` iff ``rbar_j > t*``).
    Hence the optimal vertex is a *prefix* in ``rbar``-descending order, so only
    the ``J+1`` prefix vertices need be evaluated -- the exact same supremum as
    enumerating all ``2^J`` vertices, in ``O(J log J)`` instead of ``O(2^J)``.

    The minimum-denominator vertex is all-``alow`` (raising any ``a_j`` toward
    ``ahigh`` only increases the denominator); if its denominator is non-positive
    the bound vanishes and the program is infeasible -> ``+inf`` (matching the
    brute-force behavior, which likewise returns ``+inf`` at that vertex).
    """
    order = np.argsort(-rbar, kind="stable")
    a = np.array(alow, dtype=float, copy=True)
    best = -np.inf
    # k = 0 (all alow), then flip the highest-rbar clients to ahigh one at a time.
    for k in range(len(order) + 1):
        if k > 0:
            j = order[k - 1]
            a[j] = ahigh[j]
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
    legacy_simplex_union_bound: bool = False,
) -> ConditionalCertificate:
    """Conditional selective-risk upper certificate.

    Uses ``K_j | A_j ~ Bin(A_j, r_j)`` so that ``rbar_j = U+(K_j, A_j; eps)``
    upper-bounds the per-client conditional error rate. The certified selective
    risk ``U`` upper-bounds ``R_sel(lambda)`` for every ``lambda in Lambda``.

    Parameters
    ----------
    A, K, n : per-client accepted, accepted-and-wrong, and total counts.
    delta : member-level risk tail. The strict-mixture branch divides it across
        its simultaneous endpoints; the full-simplex branch does not divide by S.
    Lambda : ``'simplex'`` (full simplex, closed form ``max_j rbar_j``),
        ``'known'`` (a single fixed ``lam``), or ``'box'`` (deterministic global
        supremum over the normalized image of a componentwise box around uniform,
        reported through a conservative numerical upper endpoint).
    lam : required for ``Lambda='known'``.
    box : additive radius around the uniform mixture for ``Lambda='box'``.
    legacy_simplex_union_bound : reproduce the archived, conservative
        ``delta/S`` calculation.  False by default; the current fixed-member
        full-simplex theorem uses ``delta`` for every stratum.

    ``n_lam_samples`` and ``seed`` are retained for source compatibility but are
    ignored by the deterministic bounded-mixture solver.
    """
    A, K, n = _count_vectors(A, K, n)
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    J = len(A)

    if Lambda == "simplex":
        eps = delta / J if legacy_simplex_union_bound else delta
        # rbar_j = 1.0 when A_j == 0 (cp_upper handles n<=0 -> 1.0).
        rbar = np.array(
            [cp_upper(int(K[j]), int(A[j]), eps) for j in range(J)], dtype=float
        )
        U = float(min(np.max(rbar), 1.0))
        return ConditionalCertificate(
            U=U,
            rbar=rbar,
            eps=eps,
            feasible=np.isfinite(U),
            Lambda="simplex",
            method=(
                "legacy-union-bound-delta-over-S"
                if legacy_simplex_union_bound
                else "theorem1-fixed-member-no-S-penalty"
            ),
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
            if lam_arr.shape != (J,) or np.any(lam_arr < 0.0) or lam_arr.sum() <= 0.0:
                raise ValueError("lam must be a non-negative length-S vector with positive sum")
            lam_arr = lam_arr / lam_arr.sum()
            # A singleton mixture is still a strict-mixture numerical program.
            # Route it through the same conservative bracket contract as a box;
            # the historical prefix optimum is retained only as a test helper.
            solver = solve_normalized_box_risk(
                rbar, alow, ahigh, NormalizedBox(lam_arr, lam_arr)
            )
            val = float(solver.value)
            witness = float(solver.witness_value)
            solver_valid = bool(
                solver.domain_feasible and solver.certificate_valid
            )
            solver_status = solver.status
            solver_tolerance = solver.tolerance
            solver_iterations = solver.iterations
            solver_lower = solver.bracket_lower
            solver_upper = solver.bracket_upper
            residual_lower = solver.residual_lower
            residual_upper = solver.residual_upper
            solver_reason = solver.reason or ""
        else:  # box
            radius = _resolve_box_radius(box)
            solver = solve_normalized_box_risk(
                rbar, alow, ahigh, uniform_box(J, radius)
            )
            val = float(solver.value)
            witness = float(solver.witness_value)
            solver_valid = bool(
                solver.domain_feasible and solver.certificate_valid
            )
            solver_status = solver.status
            solver_tolerance = solver.tolerance
            solver_iterations = solver.iterations
            solver_lower = solver.bracket_lower
            solver_upper = solver.bracket_upper
            residual_lower = solver.residual_lower
            residual_upper = solver.residual_upper
            solver_reason = solver.reason or ""

        feasible = bool(np.isfinite(val) and solver_valid)
        U = float(min(val, 1.0)) if feasible else np.inf
        return ConditionalCertificate(
            U=U, rbar=rbar, eps=eps, feasible=bool(feasible),
            Lambda=Lambda, alow=alow, ahigh=ahigh,
            method=(
                "theorem2-conservative-normalized-box"
                if Lambda == "box"
                else "theorem2-singleton-mixture"
            ),
            solver_status=solver_status,
            solver_certificate_valid=solver_valid,
            solver_tolerance=float(solver_tolerance),
            solver_iterations=int(solver_iterations),
            solver_bracket_lower=float(solver_lower),
            solver_bracket_upper=float(solver_upper),
            solver_residual_lower=float(residual_lower),
            solver_residual_upper=float(residual_upper),
            solver_witness_value=float(witness),
            solver_reason=solver_reason,
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
    closed form ``max_j mbar_j / alow_j``. Retained only as a legacy baseline.

    The historical ``Lambda='box'`` implementation sampled interior mixtures
    and could understate a supremum.  That branch is retired and fails closed;
    strict bounded-mixture certification must use
    :func:`conditional_risk_certificate` or the joint certificate API.
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
        raise RuntimeError(
            "retired sampled Lambda='box' legacy baseline cannot certify; "
            "use conditional_risk_certificate(..., Lambda='box')"
        )
    else:
        raise ValueError(f"unknown Lambda={Lambda!r}")

    U = float(min(U, 1.0))
    return StratifiedCertificate(U=U, mbar=mbar, alow=alow, eps=eps, Lambda=Lambda)
