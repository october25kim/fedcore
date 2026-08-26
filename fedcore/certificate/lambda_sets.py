"""Deterministic global optimization over declared deployment-mixture sets.

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

Why the global solve is cheap.  Both certified functionals are ratios of linear
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
the conservative direction. Numerically, certification uses a validated bracket
endpoint rounded outward; a midpoint or feasible primal value is never promoted
to a bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "NormalizedBox",
    "ConservativeExtremumResult",
    "uniform_box",
    "solve_normalized_box_risk",
    "solve_normalized_box_coverage",
    "exact_risk_supremum",
    "exact_coverage_infimum",
    "vertex_enumeration_reference",
]

_TOL = 1e-13
_MAX_ITER = 200


@dataclass(frozen=True)
class ConservativeExtremumResult:
    """Fail-closed numerical certificate for one normalized-box extremum.

    ``value`` is always conservative in ``sense``: a risk supremum uses the
    bisection bracket's upper endpoint, while a coverage infimum uses its lower
    endpoint.  ``witness_value`` is an independently evaluated feasible vertex
    and is diagnostic only; it is never returned as the certified bound.

    Any nonconvergence, invalid bracket, non-finite residual, infeasible domain,
    or failed witness/bracket check sets ``certificate_valid=False`` and returns
    ``+inf`` for a supremum or ``0`` for an infimum.
    """

    value: float
    witness_value: float
    sense: str
    domain_feasible: bool
    certificate_valid: bool
    status: str
    iterations: int
    tolerance: float
    max_iterations: int
    bracket_lower: float
    bracket_upper: float
    residual_lower: float
    residual_upper: float
    validation_tolerance: float
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.sense not in {"sup", "inf"}:
            raise ValueError("sense must be 'sup' or 'inf'")

    @property
    def bracket_width(self) -> float:
        if not (
            math.isfinite(self.bracket_lower)
            and math.isfinite(self.bracket_upper)
        ):
            return math.inf
        return max(0.0, float(self.bracket_upper - self.bracket_lower))

    def diagnostics(self, prefix: str) -> dict[str, object]:
        """Flatten solver evidence for CSV/JSON result rows."""
        return {
            f"{prefix}_status": self.status,
            f"{prefix}_certificate_valid": bool(self.certificate_valid),
            f"{prefix}_domain_feasible": bool(self.domain_feasible),
            f"{prefix}_tolerance": float(self.tolerance),
            f"{prefix}_iterations": int(self.iterations),
            f"{prefix}_max_iterations": int(self.max_iterations),
            f"{prefix}_bracket_lower": float(self.bracket_lower),
            f"{prefix}_bracket_upper": float(self.bracket_upper),
            f"{prefix}_bracket_width": float(self.bracket_width),
            f"{prefix}_residual_lower": float(self.residual_lower),
            f"{prefix}_residual_upper": float(self.residual_upper),
            f"{prefix}_validation_tolerance": float(self.validation_tolerance),
            f"{prefix}_witness_value": float(self.witness_value),
            f"{prefix}_reason": self.reason or "",
        }


@dataclass(frozen=True)
class _BisectionBracket:
    status: str
    valid: bool
    iterations: int
    lower: float
    upper: float
    residual_lower: float
    residual_upper: float
    reason: Optional[str] = None


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
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("raw box bounds must be finite")
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


def _solver_controls(tolerance: float, max_iterations: int) -> tuple[float, int]:
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations:
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    return tolerance, max_iterations


def _dinkelbach_bracket(
    residual,
    lo_t: float,
    hi_t: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> _BisectionBracket:
    """Bracket a root of a non-increasing residual without using its midpoint.

    For both robust risk and coverage, the lower endpoint must have a
    non-negative residual and the upper endpoint a non-positive residual.  The
    caller chooses the conservative endpoint appropriate to its optimization
    sense.  A failed sign invariant is a numerical failure, never a usable
    approximate optimum.
    """
    lo_t = float(lo_t)
    hi_t = float(hi_t)
    if not (math.isfinite(lo_t) and math.isfinite(hi_t) and lo_t <= hi_t):
        return _BisectionBracket(
            "invalid_bracket", False, 0, lo_t, hi_t, math.nan, math.nan,
            "nonfinite_or_reversed_initial_bracket",
        )
    try:
        r_lo = float(residual(lo_t))
        r_hi = float(residual(hi_t))
    except ArithmeticError as exc:
        return _BisectionBracket(
            "numerical_failure", False, 0, lo_t, hi_t, math.nan, math.nan,
            f"endpoint_residual_error:{type(exc).__name__}",
        )
    if not (math.isfinite(r_lo) and math.isfinite(r_hi)):
        return _BisectionBracket(
            "numerical_failure", False, 0, lo_t, hi_t, r_lo, r_hi,
            "nonfinite_endpoint_residual",
        )
    if r_lo < 0.0 or r_hi > 0.0:
        return _BisectionBracket(
            "invalid_bracket", False, 0, lo_t, hi_t, r_lo, r_hi,
            "endpoint_residual_sign_failure",
        )

    iterations = 0
    while iterations < max_iterations:
        scale = max(1.0, abs(lo_t), abs(hi_t))
        if hi_t - lo_t <= tolerance * scale:
            break
        mid = lo_t + (hi_t - lo_t) / 2.0
        try:
            r_mid = float(residual(mid))
        except ArithmeticError as exc:
            return _BisectionBracket(
                "numerical_failure", False, iterations, lo_t, hi_t, r_lo, r_hi,
                f"midpoint_residual_error:{type(exc).__name__}",
            )
        if not math.isfinite(r_mid):
            return _BisectionBracket(
                "numerical_failure", False, iterations, lo_t, hi_t, r_lo, r_hi,
                "nonfinite_midpoint_residual",
            )
        iterations += 1
        if r_mid >= 0.0:
            lo_t, r_lo = mid, r_mid
        else:
            hi_t, r_hi = mid, r_mid

    scale = max(1.0, abs(lo_t), abs(hi_t))
    if hi_t - lo_t > tolerance * scale:
        return _BisectionBracket(
            "nonconverged", False, iterations, lo_t, hi_t, r_lo, r_hi,
            "maximum_iterations_exhausted",
        )

    # Re-evaluate independently from the cached loop values.  The strict sign
    # check is deliberate: an endpoint with a residual on the wrong side is not
    # promoted to a certificate merely because the discrepancy is small.
    try:
        check_lo = float(residual(lo_t))
        check_hi = float(residual(hi_t))
    except ArithmeticError as exc:
        return _BisectionBracket(
            "numerical_failure", False, iterations, lo_t, hi_t, math.nan, math.nan,
            f"final_residual_error:{type(exc).__name__}",
        )
    if not (math.isfinite(check_lo) and math.isfinite(check_hi)):
        return _BisectionBracket(
            "numerical_failure", False, iterations, lo_t, hi_t, check_lo, check_hi,
            "nonfinite_final_residual",
        )
    if check_lo < 0.0 or check_hi > 0.0:
        return _BisectionBracket(
            "numerical_residual_failure", False, iterations,
            lo_t, hi_t, check_lo, check_hi, "final_residual_sign_failure",
        )
    return _BisectionBracket(
        "converged", True, iterations, lo_t, hi_t, check_lo, check_hi
    )


def _fail_closed_extremum(
    *,
    sense: str,
    status: str,
    reason: str,
    domain_feasible: bool,
    tolerance: float,
    max_iterations: int,
    bracket: Optional[_BisectionBracket] = None,
    witness_value: float = math.nan,
    validation_tolerance: float = math.nan,
) -> ConservativeExtremumResult:
    bracket = bracket or _BisectionBracket(
        status, False, 0, math.nan, math.nan, math.nan, math.nan, reason
    )
    return ConservativeExtremumResult(
        value=math.inf if sense == "sup" else 0.0,
        witness_value=float(witness_value),
        sense=sense,
        domain_feasible=bool(domain_feasible),
        certificate_valid=False,
        status=status,
        iterations=bracket.iterations,
        tolerance=tolerance,
        max_iterations=max_iterations,
        bracket_lower=bracket.lower,
        bracket_upper=bracket.upper,
        residual_lower=bracket.residual_lower,
        residual_upper=bracket.residual_upper,
        validation_tolerance=validation_tolerance,
        reason=reason,
    )


def _fsum(values: Sequence[float]) -> float:
    return float(math.fsum(float(value) for value in values))


def _safe_multiply(left, right, *, context: str) -> np.ndarray:
    """Multiply floats but reject lost or subnormal non-zero contributions.

    Bracket and primal checks are not independent if both silently round the
    same positive product to zero (or to a very low-precision subnormal).  Such
    arithmetic can make a risk upper endpoint anti-conservative.  We therefore
    fail closed instead of attempting to certify in this unsupported regime.
    """
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        product = np.multiply(left_array, right_array)
    nonzero_operands = (left_array != 0.0) & (right_array != 0.0)
    unsafe = nonzero_operands & (
        (product == 0.0)
        | ((np.abs(product) < np.finfo(float).tiny) & (product != 0.0))
    )
    if np.any(unsafe):
        raise FloatingPointError(f"unsafe_subnormal_product:{context}")
    return np.asarray(product, dtype=float)


def _risk_vertex_witness(
    r: np.ndarray, lo_product: np.ndarray, hi_product: np.ndarray
) -> float:
    """Largest feasible vertex value, used only as a lower witness."""
    product = np.array(lo_product, dtype=float, copy=True)
    order = np.argsort(-r, kind="stable")
    best = -math.inf
    for step in range(len(order) + 1):
        if step:
            index = int(order[step - 1])
            product[index] = hi_product[index]
        denominator = _fsum(product)
        if denominator > 0.0:
            ratio = _fsum(
                _safe_multiply(product, r, context="risk_witness_numerator")
            ) / denominator
            best = max(best, ratio)
    return float(best)


def _coverage_vertex_witness(L: np.ndarray, box: NormalizedBox) -> float:
    """Smallest feasible vertex value, used only as an upper witness."""
    weights = np.array(box.lo, dtype=float, copy=True)
    order = np.argsort(L, kind="stable")
    best = math.inf
    for step in range(len(order) + 1):
        if step:
            index = int(order[step - 1])
            weights[index] = box.hi[index]
        denominator = _fsum(weights)
        if denominator > 0.0:
            ratio = _fsum(
                _safe_multiply(weights, L, context="coverage_witness_numerator")
            ) / denominator
            best = min(best, ratio)
    return float(best)


def solve_normalized_box_risk(
    rbar: Sequence[float],
    alow: Sequence[float],
    ahigh: Sequence[float],
    box: NormalizedBox,
    *,
    tolerance: float = _TOL,
    max_iterations: int = _MAX_ITER,
) -> ConservativeExtremumResult:
    """Conservatively certify the normalized-box robust-risk supremum.

    The certified value is the converged bisection bracket's upper endpoint,
    rounded outward.  The independently evaluated primal vertex is retained only
    as ``witness_value`` and can never make the reported UCB smaller.
    """
    return _solve_normalized_box_risk(
        rbar, alow, ahigh, box,
        tolerance=tolerance, max_iterations=max_iterations,
    )


def _solve_normalized_box_risk(
    rbar: Sequence[float],
    alow: Sequence[float],
    ahigh: Sequence[float],
    box: NormalizedBox,
    *,
    tolerance: float,
    max_iterations: int,
) -> ConservativeExtremumResult:
    tolerance, max_iterations = _solver_controls(tolerance, max_iterations)
    if not isinstance(box, NormalizedBox):
        raise TypeError("box must be a NormalizedBox")
    r = np.asarray(rbar, dtype=float)
    lo_a = np.asarray(alow, dtype=float)
    hi_a = np.asarray(ahigh, dtype=float)
    J = box.dimension
    if not (
        r.ndim == lo_a.ndim == hi_a.ndim == 1
        and r.size == lo_a.size == hi_a.size == J
    ):
        raise ValueError("rbar, alow, ahigh and the box must share a dimension")
    if not (
        np.all(np.isfinite(r))
        and np.all(np.isfinite(lo_a))
        and np.all(np.isfinite(hi_a))
    ):
        raise ValueError("risk and acceptance endpoints must be finite")
    if np.any(r < 0.0) or np.any(r > 1.0):
        raise ValueError("rbar must lie in [0, 1]")
    if np.any(lo_a < 0.0) or np.any(hi_a > 1.0) or np.any(lo_a > hi_a):
        raise ValueError("acceptance box must satisfy 0 <= alow <= ahigh <= 1")

    # Largest achievable denominator; if it is zero no feasible point has mass.
    try:
        lo_product = _safe_multiply(
            box.lo, lo_a, context="risk_lower_weight_acceptance"
        )
        hi_product = _safe_multiply(
            box.hi, hi_a, context="risk_upper_weight_acceptance"
        )
        maximum_denominator = _fsum(hi_product)
    except (FloatingPointError, OverflowError) as exc:
        return _fail_closed_extremum(
            sense="sup", status="numerical_failure",
            reason=f"acceptance_product_error:{type(exc).__name__}",
            domain_feasible=False, tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if not math.isfinite(maximum_denominator):
        return _fail_closed_extremum(
            sense="sup", status="numerical_failure",
            reason="nonfinite_maximum_denominator", domain_feasible=False,
            tolerance=tolerance, max_iterations=max_iterations,
        )
    if maximum_denominator <= 0.0:
        return _fail_closed_extremum(
            sense="sup", status="infeasible_no_positive_denominator",
            reason="no_positive_denominator", domain_feasible=False,
            tolerance=tolerance, max_iterations=max_iterations,
        )

    def residual(t: float) -> float:
        # Separable: maximize lam_j * a_j * (rbar_j - t) coordinatewise, with
        # lam_j * a_j ranging over [lo_j * alow_j, hi_j * ahigh_j].
        gain = r - t
        upper_terms = _safe_multiply(
            hi_product, gain, context="risk_residual_upper"
        )
        lower_terms = _safe_multiply(
            lo_product, gain, context="risk_residual_lower"
        )
        terms = np.where(gain > 0.0, upper_terms, lower_terms)
        return _fsum(terms)

    bracket = _dinkelbach_bracket(
        residual,
        float(r.min()),
        float(r.max()),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    try:
        witness = _risk_vertex_witness(r, lo_product, hi_product)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        return _fail_closed_extremum(
            sense="sup", status="numerical_failure",
            reason=f"witness_error:{type(exc).__name__}", domain_feasible=True,
            tolerance=tolerance, max_iterations=max_iterations, bracket=bracket,
        )
    validation_tolerance = max(
        tolerance * max(1.0, abs(bracket.lower), abs(bracket.upper)),
        128.0 * np.finfo(float).eps * max(1.0, float(J)),
    )
    if not bracket.valid:
        return _fail_closed_extremum(
            sense="sup", status=bracket.status,
            reason=bracket.reason or "bisection_failure", domain_feasible=True,
            tolerance=tolerance, max_iterations=max_iterations, bracket=bracket,
            witness_value=witness, validation_tolerance=validation_tolerance,
        )
    conservative_upper = min(1.0, math.nextafter(bracket.upper, math.inf))
    if not math.isfinite(witness) or witness > conservative_upper:
        return _fail_closed_extremum(
            sense="sup", status="numerical_residual_failure",
            reason="primal_witness_exceeds_upper_bracket", domain_feasible=True,
            tolerance=tolerance, max_iterations=max_iterations, bracket=bracket,
            witness_value=witness, validation_tolerance=validation_tolerance,
        )

    # The upper bracket, not the primal witness or midpoint, is the certificate.
    return ConservativeExtremumResult(
        value=conservative_upper,
        witness_value=witness,
        sense="sup",
        domain_feasible=True,
        certificate_valid=True,
        status="converged",
        iterations=bracket.iterations,
        tolerance=tolerance,
        max_iterations=max_iterations,
        bracket_lower=bracket.lower,
        bracket_upper=bracket.upper,
        residual_lower=bracket.residual_lower,
        residual_upper=bracket.residual_upper,
        validation_tolerance=validation_tolerance,
    )


def exact_risk_supremum(
    rbar: Sequence[float],
    alow: Sequence[float],
    ahigh: Sequence[float],
    box: NormalizedBox,
    *,
    tolerance: float = _TOL,
    max_iterations: int = _MAX_ITER,
) -> Tuple[float, bool]:
    """Compatibility wrapper returning a conservative UCB and validity flag."""
    result = _solve_normalized_box_risk(
        rbar, alow, ahigh, box,
        tolerance=tolerance, max_iterations=max_iterations,
    )
    return float(result.value), bool(
        result.domain_feasible and result.certificate_valid
    )


def solve_normalized_box_coverage(
    acceptance_lower: Sequence[float],
    box: NormalizedBox,
    *,
    tolerance: float = _TOL,
    max_iterations: int = _MAX_ITER,
) -> ConservativeExtremumResult:
    """Conservatively certify normalized-box accepted-coverage infimum.

    Over the normalized image this is ``inf_{lam in B} (sum lam alow)/(sum lam)``,
    a weighted average of ``alow`` and therefore again linear-fractional; the
    infimum is attained at a raw-box vertex.  The certified LCB is the bisection
    bracket's lower endpoint rounded outward, never the raw primal vertex value.
    """
    return _solve_normalized_box_coverage(
        acceptance_lower, box,
        tolerance=tolerance, max_iterations=max_iterations,
    )


def _solve_normalized_box_coverage(
    acceptance_lower: Sequence[float],
    box: NormalizedBox,
    *,
    tolerance: float,
    max_iterations: int,
) -> ConservativeExtremumResult:
    tolerance, max_iterations = _solver_controls(tolerance, max_iterations)
    if not isinstance(box, NormalizedBox):
        raise TypeError("box must be a NormalizedBox")
    L = np.asarray(acceptance_lower, dtype=float)
    if L.ndim != 1 or L.size != box.dimension:
        raise ValueError("acceptance_lower must match the box dimension")
    if not np.all(np.isfinite(L)):
        raise ValueError("acceptance_lower must be finite")
    if np.any(L < 0.0) or np.any(L > 1.0):
        raise ValueError("acceptance_lower must lie in [0, 1]")

    try:
        maximum_denominator = _fsum(box.hi)
    except OverflowError as exc:
        return _fail_closed_extremum(
            sense="inf", status="numerical_failure",
            reason=f"maximum_denominator_error:{type(exc).__name__}",
            domain_feasible=False, tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if not math.isfinite(maximum_denominator):
        return _fail_closed_extremum(
            sense="inf", status="numerical_failure",
            reason="nonfinite_maximum_denominator", domain_feasible=False,
            tolerance=tolerance, max_iterations=max_iterations,
        )
    if maximum_denominator <= 0.0:
        return _fail_closed_extremum(
            sense="inf", status="infeasible_no_positive_denominator",
            reason="no_positive_denominator", domain_feasible=False,
            tolerance=tolerance, max_iterations=max_iterations,
        )

    def residual(t: float) -> float:
        # q(t) = min_lam sum lam_j (L_j - t) is non-increasing.  It is
        # non-negative below the infimum and non-positive above it.
        gain = L - t
        upper_terms = _safe_multiply(
            box.hi, gain, context="coverage_residual_upper"
        )
        lower_terms = _safe_multiply(
            box.lo, gain, context="coverage_residual_lower"
        )
        terms = np.where(gain < 0.0, upper_terms, lower_terms)
        return _fsum(terms)

    bracket = _dinkelbach_bracket(
        residual,
        float(L.min()),
        float(L.max()),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    try:
        witness = _coverage_vertex_witness(L, box)
    except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
        return _fail_closed_extremum(
            sense="inf", status="numerical_failure",
            reason=f"witness_error:{type(exc).__name__}", domain_feasible=True,
            tolerance=tolerance, max_iterations=max_iterations, bracket=bracket,
        )
    validation_tolerance = max(
        tolerance * max(1.0, abs(bracket.lower), abs(bracket.upper)),
        128.0 * np.finfo(float).eps * max(1.0, float(box.dimension)),
    )
    if not bracket.valid:
        return _fail_closed_extremum(
            sense="inf", status=bracket.status,
            reason=bracket.reason or "bisection_failure", domain_feasible=True,
            tolerance=tolerance, max_iterations=max_iterations, bracket=bracket,
            witness_value=witness, validation_tolerance=validation_tolerance,
        )
    conservative_lower = max(0.0, math.nextafter(bracket.lower, -math.inf))
    if not math.isfinite(witness) or conservative_lower > witness:
        return _fail_closed_extremum(
            sense="inf", status="numerical_residual_failure",
            reason="lower_bracket_exceeds_primal_witness", domain_feasible=True,
            tolerance=tolerance, max_iterations=max_iterations, bracket=bracket,
            witness_value=witness, validation_tolerance=validation_tolerance,
        )
    return ConservativeExtremumResult(
        value=conservative_lower,
        witness_value=witness,
        sense="inf",
        domain_feasible=True,
        certificate_valid=True,
        status="converged",
        iterations=bracket.iterations,
        tolerance=tolerance,
        max_iterations=max_iterations,
        bracket_lower=bracket.lower,
        bracket_upper=bracket.upper,
        residual_lower=bracket.residual_lower,
        residual_upper=bracket.residual_upper,
        validation_tolerance=validation_tolerance,
    )


def exact_coverage_infimum(
    acceptance_lower: Sequence[float],
    box: NormalizedBox,
    *,
    tolerance: float = _TOL,
    max_iterations: int = _MAX_ITER,
) -> float:
    """Compatibility wrapper returning a conservative coverage LCB."""
    return float(
        _solve_normalized_box_coverage(
            acceptance_lower, box,
            tolerance=tolerance, max_iterations=max_iterations,
        ).value
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
