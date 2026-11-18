r"""Exact optimization over coordinate-bounded deployment mixtures.

This module contains the deterministic optimization primitives used by the
bounded-mixture version of the Fed-CORE certificate.  A deployment-mixture set
has the form

.. math::

    \Lambda(l, u) = \{\lambda: l_j \leq \lambda_j \leq u_j,
    \ \sum_j \lambda_j = 1\}.

There is no random sampling in this module.  Linear objectives are solved by a
greedy fill, and the robust linear-fractional objective is solved by monotone
parametric bisection.  The latter is exact up to the requested numerical
tolerance: every bisection subproblem is itself solved exactly by the greedy
linear solver.

Traffic-derived mixture boxes deliberately accept client counts only.  They do
not accept outcomes, correctness indicators, labels, scores, or predictions.
Each multinomial coordinate is bounded through its marginal binomial law with
Bonferroni-adjusted, two-sided Clopper--Pearson limits.  Intersecting the raw
box with the simplex then gives an exact coordinate-hull tightening.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from scipy.stats import beta as _beta


_DEFAULT_TOLERANCE = 1e-12


def _readonly_vector(values: Sequence[float], name: str) -> np.ndarray:
    """Return a finite, one-dimensional, non-empty, read-only float vector."""
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_int_vector(values: Sequence[int], name: str) -> np.ndarray:
    """Return a validated, non-negative, read-only integer count vector."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer counts, not booleans")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer counts") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite counts")
    if np.any(numeric < 0.0) or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} must contain non-negative integer counts")
    # Converting through Python integers avoids silent wraparound from unsigned
    # integer dtypes on platforms where their range exceeds int64.
    counts_list = [int(value) for value in raw.tolist()]
    if any(value > np.iinfo(np.int64).max for value in counts_list):
        raise ValueError(f"{name} contains a count too large for int64")
    counts = np.asarray(counts_list, dtype=np.int64)
    counts.setflags(write=False)
    return counts


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return value


def _fsum(values: np.ndarray) -> float:
    """Accurately and deterministically sum a one-dimensional float array."""
    return float(math.fsum(float(value) for value in values))


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    """Accurately and deterministically compute a one-dimensional dot product."""
    return float(
        math.fsum(float(x) * float(y) for x, y in zip(left, right, strict=True))
    )


@dataclass(frozen=True)
class LinearExtremum:
    """Value and deterministic optimizer of a bounded-simplex linear problem."""

    value: float
    weights: np.ndarray
    sense: str

    def __post_init__(self) -> None:
        weights = np.array(self.weights, dtype=float, copy=True)
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        if self.sense not in {"min", "max"}:
            raise ValueError("sense must be either 'min' or 'max'")


@dataclass(frozen=True)
class BoundedSimplex:
    """A non-empty simplex intersected with coordinate bounds.

    The constructor validates ``0 <= lower <= upper <= 1`` and the necessary
    and sufficient non-emptiness condition
    ``sum(lower) <= 1 <= sum(upper)``.
    """

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = _readonly_vector(self.lower, "lower")
        upper = _readonly_vector(self.upper, "upper")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(lower < 0.0) or np.any(upper > 1.0):
            raise ValueError("mixture bounds must lie in [0, 1]")
        if np.any(lower > upper):
            raise ValueError("every lower bound must be at most its upper bound")
        sum_lower = _fsum(lower)
        sum_upper = _fsum(upper)
        if sum_lower > 1.0 or sum_upper < 1.0:
            raise ValueError(
                "bounded simplex is empty: require sum(lower) <= 1 <= sum(upper)"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        """Number of mixture coordinates."""
        return int(self.lower.size)

    def _linear_extremum(
        self, coefficients: Sequence[float], *, maximize: bool
    ) -> LinearExtremum:
        coefficients_array = _readonly_vector(coefficients, "coefficients")
        if coefficients_array.shape != self.lower.shape:
            raise ValueError(
                "coefficients must have the same length as the mixture bounds"
            )

        weights = np.array(self.lower, dtype=float, copy=True)
        residual = 1.0 - _fsum(weights)
        # Stable sorting fixes all tie behavior by original coordinate index.
        if maximize:
            order = np.argsort(-coefficients_array, kind="stable")
            sense = "max"
        else:
            order = np.argsort(coefficients_array, kind="stable")
            sense = "min"

        last_changed: Optional[int] = None
        for index_raw in order:
            if residual <= 0.0:
                break
            index = int(index_raw)
            capacity = float(self.upper[index] - weights[index])
            addition = min(capacity, residual)
            if addition > 0.0:
                weights[index] += addition
                residual -= addition
                last_changed = index

        # Validation guarantees enough total capacity.  A tiny residual may
        # remain only through floating-point subtraction; absorb it without
        # changing the mathematically selected greedy vertex.
        if residual > 0.0:
            if residual > 32.0 * np.finfo(float).eps:
                raise RuntimeError("failed to fill a validated bounded simplex")
            for index_raw in order:
                index = int(index_raw)
                capacity = float(self.upper[index] - weights[index])
                if capacity >= residual:
                    weights[index] += residual
                    last_changed = index
                    residual = 0.0
                    break

        correction = 1.0 - _fsum(weights)
        if correction != 0.0:
            candidates = ([last_changed] if last_changed is not None else []) + [
                int(index) for index in order
            ]
            for index in candidates:
                if index is None:
                    continue
                candidate = weights[index] + correction
                if self.lower[index] <= candidate <= self.upper[index]:
                    weights[index] = candidate
                    break

        weights.setflags(write=False)
        return LinearExtremum(
            value=_dot(coefficients_array, weights),
            weights=weights,
            sense=sense,
        )

    def maximize(self, coefficients: Sequence[float]) -> LinearExtremum:
        """Exactly maximize a linear functional by greedy residual fill."""
        return self._linear_extremum(coefficients, maximize=True)

    def minimize(self, coefficients: Sequence[float]) -> LinearExtremum:
        """Exactly minimize a linear functional by greedy residual fill."""
        return self._linear_extremum(coefficients, maximize=False)

    def tightened(self) -> "BoundedSimplex":
        """Return the exact coordinate hull after enforcing ``sum(lambda)=1``.

        For each coordinate, the simplex constraint implies

        ``lambda[j] >= 1 - sum(upper[k] for k != j)`` and
        ``lambda[j] <= 1 - sum(lower[k] for k != j)``.

        Taking these implications coordinatewise preserves exactly the same
        feasible set while removing portions of the surrounding box that can
        never participate in a feasible mixture.
        """
        sum_lower = _fsum(self.lower)
        sum_upper = _fsum(self.upper)
        lower = np.maximum(self.lower, 1.0 - (sum_upper - self.upper))
        upper = np.minimum(self.upper, 1.0 - (sum_lower - self.lower))
        # The arithmetic above can produce harmless endpoint noise.
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
        close = lower > upper
        if np.any(close):
            gaps = lower[close] - upper[close]
            if np.any(gaps > 64.0 * np.finfo(float).eps):
                raise RuntimeError("simplex tightening produced inconsistent bounds")
            midpoint = (lower[close] + upper[close]) / 2.0
            lower[close] = midpoint
            upper[close] = midpoint
        return BoundedSimplex(lower, upper)


def maximize_linear(
    coefficients: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> LinearExtremum:
    """Exactly maximize ``dot(coefficients, lambda)`` over bounded mixtures."""
    return BoundedSimplex(lower, upper).maximize(coefficients)


def minimize_linear(
    coefficients: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> LinearExtremum:
    """Exactly minimize ``dot(coefficients, lambda)`` over bounded mixtures."""
    return BoundedSimplex(lower, upper).minimize(coefficients)


@dataclass(frozen=True)
class RobustRatioResult:
    """Result of the robust accepted-risk linear-fractional program.

    ``value`` is a numerical upper endpoint for the true supremum and
    ``attained_value`` is the value at the returned feasible optimizer.  Their
    difference is at most the requested bisection tolerance (up to floating
    point roundoff).  When the denominator can vanish, ``feasible`` is false,
    ``value`` is ``+inf``, and no optimizer is returned.
    """

    value: float
    attained_value: float
    feasible: bool
    min_denominator: float
    lambda_star: Optional[np.ndarray]
    acceptance_star: Optional[np.ndarray]
    iterations: int
    tolerance: float
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("lambda_star", "acceptance_star"):
            value = getattr(self, field_name)
            if value is not None:
                array = np.array(value, dtype=float, copy=True)
                array.setflags(write=False)
                object.__setattr__(self, field_name, array)

    @property
    def optimality_gap(self) -> float:
        """Certified numerical gap between upper and attained values."""
        if not self.feasible:
            return math.inf
        return max(0.0, float(self.value - self.attained_value))


def _validate_acceptance_box(
    acceptance_lower: Sequence[float],
    acceptance_upper: Sequence[float],
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower = _readonly_vector(acceptance_lower, "acceptance_lower")
    upper = _readonly_vector(acceptance_upper, "acceptance_upper")
    if lower.size != dimension or upper.size != dimension:
        raise ValueError("acceptance bounds must match the mixture dimension")
    if np.any(lower < 0.0) or np.any(upper > 1.0):
        raise ValueError("acceptance bounds must lie in [0, 1]")
    if np.any(lower > upper):
        raise ValueError("every acceptance lower bound must be at most its upper bound")
    return lower, upper


def solve_robust_ratio(
    risk_upper: Sequence[float],
    acceptance_lower: Sequence[float],
    acceptance_upper: Sequence[float],
    mixture: BoundedSimplex,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
    max_iterations: int = 256,
) -> RobustRatioResult:
    r"""Solve the robust accepted-risk ratio over ``lambda`` and ``a``.

    The optimized quantity is

    .. math::

        \sup_{\lambda\in\Lambda,\ a\in[\underline a,\bar a]}
        \frac{\sum_j \lambda_j a_j \bar r_j}
             {\sum_j \lambda_j a_j}.

    For a trial ratio ``t``, the Dinkelbach residual is

    ``max sum(lambda[j] * a[j] * (risk_upper[j] - t))``.

    Since ``lambda[j]`` is non-negative, its maximizing acceptance endpoint is
    ``acceptance_upper[j]`` for a non-negative coefficient and
    ``acceptance_lower[j]`` otherwise.  What remains is an exact bounded-simplex
    linear maximization.  The residual is monotone in ``t``, so bisection on
    ``[min(risk_upper), max(risk_upper)]`` gives the global optimum, not a local
    optimum or a sample approximation.
    """
    if not isinstance(mixture, BoundedSimplex):
        raise TypeError("mixture must be a BoundedSimplex")
    risk = _readonly_vector(risk_upper, "risk_upper")
    if risk.size != mixture.dimension:
        raise ValueError("risk_upper must match the mixture dimension")
    if np.any(risk < 0.0) or np.any(risk > 1.0):
        raise ValueError("risk_upper must lie in [0, 1]")
    alow, ahigh = _validate_acceptance_box(
        acceptance_lower, acceptance_upper, mixture.dimension
    )
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations:
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    denominator_minimum = mixture.minimize(alow)
    min_denominator = max(0.0, float(denominator_minimum.value))
    if min_denominator <= 0.0:
        return RobustRatioResult(
            value=math.inf,
            attained_value=math.nan,
            feasible=False,
            min_denominator=min_denominator,
            lambda_star=None,
            acceptance_star=None,
            iterations=0,
            tolerance=tolerance,
            reason="vanishing_denominator",
        )

    def maximize_residual(
        trial: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        centered = risk - trial
        acceptance = np.where(centered >= 0.0, ahigh, alow)
        coefficients = acceptance * centered
        linear = mixture.maximize(coefficients)
        weights = linear.weights
        denominator = _dot(weights, acceptance)
        numerator = _dot(weights * acceptance, risk)
        ratio = numerator / denominator
        residual = _dot(weights, coefficients)
        return residual, ratio, weights, acceptance

    low = float(np.min(risk))
    high = float(np.max(risk))
    initial_residual, best_ratio, best_weights, best_acceptance = maximize_residual(low)
    del initial_residual

    if high == low:
        return RobustRatioResult(
            value=high,
            attained_value=best_ratio,
            feasible=True,
            min_denominator=min_denominator,
            lambda_star=best_weights,
            acceptance_star=best_acceptance,
            iterations=0,
            tolerance=tolerance,
        )

    iterations = 0
    while iterations < max_iterations:
        scale = max(1.0, abs(low), abs(high))
        if high - low <= tolerance * scale:
            break
        trial = low + (high - low) / 2.0
        residual, ratio, weights, acceptance = maximize_residual(trial)
        iterations += 1
        if ratio > best_ratio:
            best_ratio = ratio
            best_weights = weights
            best_acceptance = acceptance
        if residual > 0.0:
            low = trial
        else:
            high = trial

    scale = max(1.0, abs(low), abs(high))
    if high - low > tolerance * scale:
        raise RuntimeError(
            "robust ratio bisection did not converge within max_iterations"
        )

    # The attained optimizer is also a valid lower bracket and guards against a
    # final bracket that is one ulp below it through cancellation.
    low = max(low, best_ratio)
    high = max(high, low)
    return RobustRatioResult(
        value=high,
        attained_value=best_ratio,
        feasible=True,
        min_denominator=min_denominator,
        lambda_star=best_weights,
        acceptance_star=best_acceptance,
        iterations=iterations,
        tolerance=tolerance,
    )


def robust_ratio_supremum(
    risk_upper: Sequence[float],
    acceptance_lower: Sequence[float],
    acceptance_upper: Sequence[float],
    lambda_lower: Sequence[float],
    lambda_upper: Sequence[float],
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
    max_iterations: int = 256,
) -> float:
    """Return the robust ratio supremum, or ``+inf`` if it is infeasible."""
    result = solve_robust_ratio(
        risk_upper,
        acceptance_lower,
        acceptance_upper,
        BoundedSimplex(lambda_lower, lambda_upper),
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return float(result.value)


def solve_coverage_infimum(
    acceptance_lower: Sequence[float], mixture: BoundedSimplex
) -> LinearExtremum:
    """Exactly minimize certified accepted coverage over a mixture set."""
    if not isinstance(mixture, BoundedSimplex):
        raise TypeError("mixture must be a BoundedSimplex")
    lower = _readonly_vector(acceptance_lower, "acceptance_lower")
    if lower.size != mixture.dimension:
        raise ValueError("acceptance_lower must match the mixture dimension")
    if np.any(lower < 0.0) or np.any(lower > 1.0):
        raise ValueError("acceptance_lower must lie in [0, 1]")
    return mixture.minimize(lower)


def coverage_infimum(
    acceptance_lower: Sequence[float],
    lambda_lower: Sequence[float],
    lambda_upper: Sequence[float],
) -> float:
    """Return ``inf_lambda sum(lambda[j] * acceptance_lower[j])`` exactly."""
    return float(
        solve_coverage_infimum(
            acceptance_lower, BoundedSimplex(lambda_lower, lambda_upper)
        ).value
    )


def rho_mixture_box(center: Sequence[float], rho: float) -> BoundedSimplex:
    """Coordinate ``rho`` sensitivity set around a predeclared center mixture."""
    weights = _readonly_vector(center, "center")
    if (
        np.any(weights < 0.0)
        or np.any(weights > 1.0)
        or not math.isclose(_fsum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("center must be a probability vector")
    radius = float(rho)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("rho must be finite and non-negative")
    return BoundedSimplex(
        np.maximum(0.0, weights - radius),
        np.minimum(1.0, weights + radius),
    ).tightened()


def intersect_mixture_boxes(*boxes: BoundedSimplex) -> BoundedSimplex:
    """Exact intersection of coordinate-bounded simplex sets."""
    if not boxes or any(not isinstance(box, BoundedSimplex) for box in boxes):
        raise ValueError("at least one BoundedSimplex is required")
    dimension = boxes[0].dimension
    if any(box.dimension != dimension for box in boxes):
        raise ValueError("all mixture boxes must have the same dimension")
    lower = np.maximum.reduce([box.lower for box in boxes])
    upper = np.minimum.reduce([box.upper for box in boxes])
    return BoundedSimplex(lower, upper).tightened()


@dataclass(frozen=True)
class TrafficMixtureConfidenceBox:
    """Simultaneous multinomial coordinate confidence box from traffic counts."""

    counts: np.ndarray
    total: int
    delta: float
    tail_probability: float
    raw_lower: np.ndarray
    raw_upper: np.ndarray
    mixture: BoundedSimplex

    def __post_init__(self) -> None:
        counts = np.array(self.counts, dtype=np.int64, copy=True)
        counts.setflags(write=False)
        raw_lower = np.array(self.raw_lower, dtype=float, copy=True)
        raw_upper = np.array(self.raw_upper, dtype=float, copy=True)
        raw_lower.setflags(write=False)
        raw_upper.setflags(write=False)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "raw_lower", raw_lower)
        object.__setattr__(self, "raw_upper", raw_upper)

    @property
    def lower(self) -> np.ndarray:
        """Tightened coordinate lower bounds."""
        return self.mixture.lower

    @property
    def upper(self) -> np.ndarray:
        """Tightened coordinate upper bounds."""
        return self.mixture.upper


def traffic_mixture_confidence_box(
    client_counts: Sequence[int], delta: float
) -> TrafficMixtureConfidenceBox:
    """Build a label-free simultaneous mixture box from traffic client counts.

    If ``N_j`` is the number of unlabeled traffic observations from client
    ``j`` and ``N = sum_j N_j``, then marginally
    ``N_j ~ Binomial(N, lambda_j)``.  Each lower and upper coordinate limit uses
    one-sided Clopper--Pearson tail probability ``delta / (2 * J)``.  A union
    bound over the ``2 * J`` tails therefore gives simultaneous coverage at
    least ``1 - delta`` without assuming independent coordinates.

    All-zero counts are allowed and correctly return the full simplex (or the
    singleton ``[1, 1]`` when there is only one client).  The function accepts
    no labels or label-derived inputs.
    """
    counts = _readonly_int_vector(client_counts, "client_counts")
    delta = _validate_probability(delta, "delta")
    dimension = int(counts.size)
    total = int(sum(int(value) for value in counts))
    tail_probability = delta / (2.0 * dimension)

    raw_lower = np.zeros(dimension, dtype=float)
    raw_upper = np.ones(dimension, dtype=float)
    if total > 0:
        for index, count_raw in enumerate(counts):
            count = int(count_raw)
            if count > 0:
                raw_lower[index] = float(
                    _beta.ppf(tail_probability, count, total - count + 1)
                )
            if count < total:
                raw_upper[index] = float(
                    _beta.ppf(1.0 - tail_probability, count + 1, total - count)
                )

    raw_box = BoundedSimplex(raw_lower, raw_upper)
    tightened = raw_box.tightened()
    return TrafficMixtureConfidenceBox(
        counts=counts,
        total=total,
        delta=delta,
        tail_probability=tail_probability,
        raw_lower=raw_lower,
        raw_upper=raw_upper,
        mixture=tightened,
    )


def multinomial_mixture_confidence_box(
    client_counts: Sequence[int], delta: float
) -> TrafficMixtureConfidenceBox:
    """Alias with an explicit statistical name for the traffic-count API."""
    return traffic_mixture_confidence_box(client_counts, delta)


__all__ = [
    "BoundedSimplex",
    "LinearExtremum",
    "RobustRatioResult",
    "TrafficMixtureConfidenceBox",
    "coverage_infimum",
    "maximize_linear",
    "minimize_linear",
    "multinomial_mixture_confidence_box",
    "robust_ratio_supremum",
    "solve_coverage_infimum",
    "solve_robust_ratio",
    "traffic_mixture_confidence_box",
]
