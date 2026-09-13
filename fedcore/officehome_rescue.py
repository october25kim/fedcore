"""Post-primary selector-rescue theorems for the Office-Home arm (no new training).

This module is the *pure* (artifact-free, numpy/scipy-only) core of the
Office-Home selector-rescue follow-up.  It implements two finite-sample-valid
extensions of the frozen full-simplex certificate over a **prospectively frozen**
finite family of selector candidates, plus the deterministic bookkeeping the
driver needs (selector-definition hash, fresh-draw seed rule).

Nothing here is pre-registered as the primary result.  The primary Office-Home
claim remains the client-level FULL SIMPLEX, coverage-max selector, 0/50
certificate (``results/officehome/final_cell_results.csv``), which is never
recomputed or overwritten by this module.

The candidate family (Phase S0, ``FAMILY_FROZEN_PASS``) is

    score in {msp, energy, margin}  x  gamma in {0.3, 0.5, 0.7, 1.0}  =  M = 12

candidates per ``(cell, alpha)``.  Each candidate's threshold is a deterministic
function of the PROPOSAL fold only, so the family is measurable w.r.t. the
proposal fold and fixed before any certification outcome is read.  That is
exactly the condition under which the two theorems below are valid.

Theorem S1 (simultaneous selector-family certificate).
    Fix the family before the certification fold.  Union-bound over all
    ``M * J`` per-candidate/per-client risk events at ``delta_r / (M J)`` each
    and all ``M * J`` coverage events at ``delta_c / (M J)`` each.  Then
    *simultaneously for every candidate* ``m``,

        rbar_jm = U+(K_jm, A_jm; delta_r / (M J)),   U_m   = max_j rbar_jm
        alow_jm = U-(A_jm, n_j;  delta_c / (M J)),    C_m   = min_j alow_jm

    are valid full-simplex risk-UCB / coverage-LCB.  Because the bounds hold
    simultaneously, a certification-data-dependent choice
    ``m_hat in argmax{ C_m : U_m <= alpha, C_m > 0 }`` still yields a valid
    ``(1 - delta_r - delta_c)`` certificate for the selected candidate.  The
    price is the multiplicity factor ``M`` inside every ``eps``.

Theorem S1' (Holm / LTT risk-only variant).
    Test each per-client null ``H_jm: r_jm > alpha`` with the least-favorable
    one-sided binomial p-value ``p_jm = P(Bin(A_jm, alpha) <= K_jm)``.  The
    candidate null ``H_m: max_j r_jm > alpha`` is an intersection-union null, so
    ``p_m = max_j p_jm`` is a valid p-value for it.  Holm step-down over the
    ``M`` candidate p-values controls the family-wise error rate at ``delta_r``.
    Coverage is still certified by the simultaneous union bound at ``delta_c``.

Both extensions keep the client FULL-SIMPLEX target of the primary (``U_m`` is a
worst-client bound valid for every deployment mixture).  Any positive result is
a "post-primary finite-family extension over a prospectively frozen candidate
set" -- NOT preregistered.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binom as _binom

from fedcore.certificate.cp import cp_lower, cp_upper
from fedcore.officehome_selector import DEFAULT_GAMMAS, DEFAULT_KAPPA, DEFAULT_SCORES

# The frozen candidate family (Phase S0). Predeclared code constants, identical
# to the ones the original primary selector uses.
FAMILY_SCORES: Tuple[str, ...] = tuple(DEFAULT_SCORES)          # (msp, energy, margin)
FAMILY_GAMMAS: Tuple[float, ...] = tuple(DEFAULT_GAMMAS)        # (0.3, 0.5, 0.7, 1.0)
FAMILY_NGRID: int = 300
FAMILY_KAPPA: float = DEFAULT_KAPPA
M_CANDIDATES: int = len(FAMILY_SCORES) * len(FAMILY_GAMMAS)    # 12

# Provenance salts. Frozen strings; changing them changes the derived tokens.
SELECTOR_HASH_SALT = "officehome-selector-rescue-v1"
FRESH_DRAW_SALT = "officehome-fresh-draw-v1"


# --------------------------------------------------------------------------- #
# Deterministic provenance tokens
# --------------------------------------------------------------------------- #
def selector_definition_hash(score_name: str, gamma: float, alpha: float) -> str:
    """Deterministic 16-hex selector-definition token (tie-break + round trip).

    Depends only on the selector DEFINITION (score, gamma, alpha, kappa, grid,
    buffer rule) -- never on the cell, the threshold, or any outcome.  This is a
    local provenance/tie-break stamp for the rescue analysis; it is documented as
    distinct from the opaque Phase-S0 audit stamp, which the driver carries
    forward separately by re-verifying the full recomputed selector content.
    """
    payload = (
        f"{SELECTOR_HASH_SALT}|score={score_name}|gamma={float(gamma):.6f}"
        f"|alpha={float(alpha):.6f}|kappa={FAMILY_KAPPA:.6f}"
        f"|ngrid={FAMILY_NGRID}|buffer=gamma*alpha"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fresh_draw_seed(cell_name: str) -> int:
    """Fresh-draw semantic seed for Phase S2 (outcome-independent, per cell).

    A function of the cell name ONLY -- never of alpha, score, gamma, the solver,
    or any certification/evaluation outcome -- so it is frozen before outcomes are
    read.  The salt makes the stream distinct from the canonical audit draw and
    from any smoke draw.  Returns a uint32-range integer seed.
    """
    digest = hashlib.blake2b(
        f"{FRESH_DRAW_SALT}|{cell_name}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**32)


@dataclass(frozen=True)
class CandidateKey:
    """Identity of one frozen selector candidate (proposal-fold measurable)."""

    score_name: str
    gamma: float
    alpha: float

    @property
    def selector_hash(self) -> str:
        return selector_definition_hash(self.score_name, self.gamma, self.alpha)

    def tiebreak_key(self) -> Tuple[float, str, str]:
        """Deterministic tie-break: smaller gamma, lexicographic score, hash."""
        return (float(self.gamma), self.score_name, self.selector_hash)


def family_keys(alpha: float) -> List[CandidateKey]:
    """The M=12 candidate keys for one alpha, in a fixed deterministic order."""
    keys: List[CandidateKey] = []
    for score_name in FAMILY_SCORES:
        for gamma in FAMILY_GAMMAS:
            keys.append(CandidateKey(score_name, float(gamma), float(alpha)))
    return keys


# --------------------------------------------------------------------------- #
# Theorem S1 -- simultaneous selector-family certificate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FamilyCertificate:
    """Per-candidate simultaneous full-simplex certificate over the M-family."""

    rbar: np.ndarray            # (M, J) risk UCB per candidate/client
    alow: np.ndarray            # (M, J) acceptance LCB per candidate/client
    U: np.ndarray               # (M,) worst-client risk UCB
    C: np.ndarray               # (M,) worst-client coverage LCB
    risk_ok: np.ndarray         # (M,) U_m <= alpha
    cov_ok: np.ndarray          # (M,) C_m > 0
    certified: np.ndarray       # (M,) risk_ok & cov_ok
    eps_r: float                # delta_r / (M J)
    eps_c: float                # delta_c / (M J)
    M: int
    J: int


def simultaneous_family_certificate(
    A: Sequence[Sequence[int]],
    K: Sequence[Sequence[int]],
    n: Sequence[int],
    *,
    alpha: float,
    delta_r: float,
    delta_c: float,
) -> FamilyCertificate:
    """Theorem S1. ``A``/``K`` are ``(M, J)``; ``n`` is ``(J,)``.

    Uses the exact fedcore CP core (``cp_upper`` / ``cp_lower``) with the
    multiplicity-corrected budgets ``delta_r / (M J)`` and ``delta_c / (M J)``.
    """
    A_arr = np.asarray(A, dtype=int)
    K_arr = np.asarray(K, dtype=int)
    n_arr = np.asarray(n, dtype=int)
    if A_arr.ndim != 2 or A_arr.shape != K_arr.shape:
        raise ValueError("A and K must be aligned (M, J) integer matrices")
    M, J = A_arr.shape
    if n_arr.shape != (J,):
        raise ValueError("n must be a length-J vector")
    if np.any(K_arr < 0) or np.any(K_arr > A_arr) or np.any(A_arr > n_arr[None, :]):
        raise ValueError("counts must satisfy 0 <= K <= A <= n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if delta_r <= 0.0 or delta_c <= 0.0:
        raise ValueError("delta_r and delta_c must be positive")

    eps_r = float(delta_r) / (M * J)
    eps_c = float(delta_c) / (M * J)
    rbar = np.empty((M, J), dtype=float)
    alow = np.empty((M, J), dtype=float)
    for m in range(M):
        for j in range(J):
            rbar[m, j] = cp_upper(int(K_arr[m, j]), int(A_arr[m, j]), eps_r)
            alow[m, j] = cp_lower(int(A_arr[m, j]), int(n_arr[j]), eps_c)
    U = rbar.max(axis=1)
    C = alow.min(axis=1)
    risk_ok = U <= alpha
    cov_ok = C > 0.0
    certified = risk_ok & cov_ok
    return FamilyCertificate(
        rbar=rbar, alow=alow, U=U, C=C,
        risk_ok=risk_ok, cov_ok=cov_ok, certified=certified,
        eps_r=eps_r, eps_c=eps_c, M=M, J=J,
    )


def select_family_candidate(
    keys: Sequence[CandidateKey],
    certified: Sequence[bool],
    C: Sequence[float],
) -> Optional[int]:
    """Valid post-selection: argmax C_m among certified; documented tie-break.

    Ties: smaller gamma, then lexicographic score, then selector hash. Returns
    the selected index, or ``None`` if no candidate certifies.
    """
    idx = [m for m in range(len(keys)) if bool(certified[m])]
    if not idx:
        return None
    return min(idx, key=lambda m: (-float(C[m]),) + keys[m].tiebreak_key())


# --------------------------------------------------------------------------- #
# Theorem S1' -- Holm / LTT risk-only variant
# --------------------------------------------------------------------------- #
def candidate_null_pvalue(A_row: Sequence[int], K_row: Sequence[int], alpha: float) -> float:
    """IU p-value for ``H_m: max_j r_jm > alpha``.

    ``p_jm = P(Bin(A_jm, alpha) <= K_jm)`` is the least-favorable one-sided
    p-value for ``H_jm: r_jm > alpha`` (small K is evidence that ``r_jm <= alpha``).
    The intersection-union p-value for the candidate null is ``max_j p_jm``.
    A client with ``A_jm = 0`` yields ``p_jm = 1`` (no evidence), which correctly
    blocks certification of that candidate.
    """
    A_row = np.asarray(A_row, dtype=int)
    K_row = np.asarray(K_row, dtype=int)
    p = np.empty(A_row.shape[0], dtype=float)
    for j in range(A_row.shape[0]):
        a = int(A_row[j])
        k = int(K_row[j])
        if a <= 0:
            p[j] = 1.0
        else:
            p[j] = float(_binom.cdf(k, a, alpha))
    return float(np.max(p))


def holm_step_down_reject(pvalues: Sequence[float], fwer: float) -> np.ndarray:
    """Holm step-down rejections controlling FWER at ``fwer``.

    Sort p-values ascending; reject ranks ``1..i`` where ``i`` is the last rank
    with ``p_(k) <= fwer / (M - k + 1)`` for every ``k <= i`` (stop at the first
    failure). Returns a boolean mask over the original ordering.
    """
    p = np.asarray(pvalues, dtype=float)
    M = p.shape[0]
    reject = np.zeros(M, dtype=bool)
    order = np.argsort(p, kind="stable")
    for rank, m in enumerate(order):  # rank is 0-based
        threshold = float(fwer) / (M - rank)
        if p[m] <= threshold:
            reject[m] = True
        else:
            break
    return reject


@dataclass(frozen=True)
class HolmFamilyCertificate:
    """Holm risk-only + simultaneous-coverage certificate over the M-family."""

    pvalues: np.ndarray         # (M,) IU candidate-null p-values
    holm_reject: np.ndarray     # (M,) risk rejection (r-certified)
    alow: np.ndarray            # (M, J) coverage LCB (simultaneous at delta_c)
    C: np.ndarray               # (M,) worst-client coverage LCB
    cov_ok: np.ndarray          # (M,) C_m > 0
    certified: np.ndarray       # (M,) holm_reject & cov_ok
    eps_c: float
    fwer: float
    M: int
    J: int


def holm_family_certificate(
    A: Sequence[Sequence[int]],
    K: Sequence[Sequence[int]],
    n: Sequence[int],
    *,
    alpha: float,
    delta_r: float,
    delta_c: float,
) -> HolmFamilyCertificate:
    """Theorem S1'. Holm FWER on risk at ``delta_r``; coverage union bound at ``delta_c``."""
    A_arr = np.asarray(A, dtype=int)
    K_arr = np.asarray(K, dtype=int)
    n_arr = np.asarray(n, dtype=int)
    if A_arr.ndim != 2 or A_arr.shape != K_arr.shape:
        raise ValueError("A and K must be aligned (M, J) integer matrices")
    M, J = A_arr.shape
    if n_arr.shape != (J,):
        raise ValueError("n must be a length-J vector")
    if np.any(K_arr < 0) or np.any(K_arr > A_arr) or np.any(A_arr > n_arr[None, :]):
        raise ValueError("counts must satisfy 0 <= K <= A <= n")

    pvalues = np.array(
        [candidate_null_pvalue(A_arr[m], K_arr[m], alpha) for m in range(M)]
    )
    reject = holm_step_down_reject(pvalues, delta_r)

    eps_c = float(delta_c) / (M * J)
    alow = np.empty((M, J), dtype=float)
    for m in range(M):
        for j in range(J):
            alow[m, j] = cp_lower(int(A_arr[m, j]), int(n_arr[j]), eps_c)
    C = alow.min(axis=1)
    cov_ok = C > 0.0
    certified = reject & cov_ok
    return HolmFamilyCertificate(
        pvalues=pvalues, holm_reject=reject, alow=alow, C=C, cov_ok=cov_ok,
        certified=certified, eps_c=eps_c, fwer=float(delta_r), M=M, J=J,
    )


# --------------------------------------------------------------------------- #
# Monte-Carlo helpers (validity tests only; no artifact I/O)
# --------------------------------------------------------------------------- #
def simulate_conditional_counts(
    true_a: np.ndarray, true_r: np.ndarray, n: Sequence[int], rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``A_jm ~ Bin(n_j, a_jm)`` then ``K_jm ~ Bin(A_jm, r_jm)`` (the model law)."""
    true_a = np.asarray(true_a, dtype=float)
    true_r = np.asarray(true_r, dtype=float)
    n_arr = np.asarray(n, dtype=int)
    M, J = true_a.shape
    A = rng.binomial(np.broadcast_to(n_arr, (M, J)), true_a)
    K = rng.binomial(A, true_r)
    return A.astype(int), K.astype(int)


__all__ = [
    "FAMILY_SCORES",
    "FAMILY_GAMMAS",
    "FAMILY_NGRID",
    "FAMILY_KAPPA",
    "M_CANDIDATES",
    "SELECTOR_HASH_SALT",
    "FRESH_DRAW_SALT",
    "CandidateKey",
    "FamilyCertificate",
    "HolmFamilyCertificate",
    "candidate_null_pvalue",
    "family_keys",
    "fresh_draw_seed",
    "holm_family_certificate",
    "holm_step_down_reject",
    "select_family_candidate",
    "selector_definition_hash",
    "simulate_conditional_counts",
    "simultaneous_family_certificate",
]
