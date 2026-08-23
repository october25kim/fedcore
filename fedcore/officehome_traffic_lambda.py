"""Traffic-derived deployment-mixture set for the Office-Home arm (Hybrid Draft 5).

The deployment mixture over the ``J = 4`` domain-clients is unknown. This module
estimates a *simultaneous confidence set* ``Lambda_hat`` for it from the TRAFFIC
fold using domain (client) identities ONLY -- never a label, score, prediction,
or model output. It is a thin, documented theorem layer over
:func:`fedcore.mixture.traffic_mixture_confidence_box`.

Statistical statement. Draw ``m`` traffic observations; let ``N_j`` be the count
from client ``j`` and ``N = sum_j N_j``. Marginally ``N_j ~ Binomial(N, lam_j)``
for the true mixture ``lam``. Each coordinate gets a two-sided Clopper-Pearson
interval at tail ``delta_lambda / (2J)`` per side; a union bound over the ``2J``
tails gives simultaneous coverage

    ``P( lam in Lambda_hat ) >= 1 - delta_lambda``.

The raw coordinate box is then intersected with the simplex (exact
coordinate-hull tightening) for use by the certificate's box target.

Budgets (frozen): ``delta_lambda / delta_r / delta_c = 0.02 / 0.04 / 0.04``;
``m = 1000`` primary, sensitivity grid ``{250, 500, 1000, 2000}``.

Fail-closed: identity overlap between the traffic identities and any other role
(:func:`assert_traffic_identity_disjoint`); empty traffic returns the full-simplex
fallback rather than a spurious tight set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from fedcore.mixture import BoundedSimplex, TrafficMixtureConfidenceBox, traffic_mixture_confidence_box


DELTA_LAMBDA: float = 0.02
DELTA_R: float = 0.04
DELTA_C: float = 0.04
M_PRIMARY: int = 1000
M_GRID: tuple[int, ...] = (250, 500, 1000, 2000)
N_DOMAIN_CLIENTS: int = 4


class TrafficLambdaError(ValueError):
    """Raised on a fail-closed traffic-mixture construction violation."""


def assert_traffic_identity_disjoint(
    traffic_ids: Sequence[str], *other_id_sets: Iterable[str]
) -> None:
    """Fail closed if any traffic identity also appears in another role."""

    traffic = list(traffic_ids)
    traffic_set = set(traffic)
    if len(traffic_set) != len(traffic):
        raise TrafficLambdaError("traffic identities contain duplicates")
    for other in other_id_sets:
        overlap = traffic_set & set(other)
        if overlap:
            raise TrafficLambdaError(
                f"traffic identities overlap another role: {sorted(overlap)[:3]}"
            )


def draw_traffic_client_counts(
    client_ids: Sequence[int], m: int, *, seed: int, n_clients: int = N_DOMAIN_CLIENTS
) -> np.ndarray:
    """Draw ``m`` traffic rows with replacement and tally per client.

    Uses ONLY the per-row client identity (no label/score/prediction is read),
    so the construction is label- and score-free. Deterministic given ``seed``.
    """

    ids = np.asarray(client_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size == 0:
        raise TrafficLambdaError("client_ids must be a non-empty 1-D array")
    if np.any(ids < 0) or np.any(ids >= n_clients):
        raise TrafficLambdaError("client_ids must lie in [0, n_clients)")
    if int(m) <= 0:
        raise TrafficLambdaError("m must be a positive integer")
    rng = np.random.default_rng(int(seed))
    drawn = rng.integers(0, ids.size, size=int(m))
    return np.bincount(ids[drawn], minlength=n_clients).astype(np.int64)


def traffic_lambda_box(
    client_counts: Sequence[int], delta_lambda: float = DELTA_LAMBDA
) -> TrafficMixtureConfidenceBox:
    """Build the simultaneous mixture confidence box from client counts."""

    return traffic_mixture_confidence_box(client_counts, float(delta_lambda))


def full_simplex_box(n_clients: int = N_DOMAIN_CLIENTS) -> BoundedSimplex:
    """The full-simplex fallback ``[0, 1]^J`` intersected with the simplex."""

    return BoundedSimplex(np.zeros(n_clients), np.ones(n_clients)).tightened()


def coordinate_tail_report(box: TrafficMixtureConfidenceBox) -> dict:
    """Exact per-coordinate tail accounting for the union bound.

    Returns the per-side tail probability ``delta_lambda / (2J)``, the number of
    tails ``2J``, and the union bound ``2J * tail = delta_lambda`` -- the exact
    statement of simultaneous coverage.
    """

    J = int(box.counts.size)
    tail = float(box.tail_probability)
    return {
        "n_clients": J,
        "n_tails": 2 * J,
        "per_side_tail": tail,
        "expected_per_side_tail": float(box.delta / (2.0 * J)),
        "union_bound": 2.0 * J * tail,
        "delta_lambda": float(box.delta),
    }


def covers(box: TrafficMixtureConfidenceBox, lambda_star: Sequence[float]) -> bool:
    """Whether ``lambda_star`` lies in every RAW two-sided CP coordinate interval.

    Coverage is stated on the RAW marginal-binomial box (the object the union
    bound certifies). Because ``lambda_star`` sums to 1, raw-box membership and
    tightened-box membership coincide, but the raw box is the honest CP object.
    """

    lam = np.asarray(lambda_star, dtype=float)
    if lam.size != box.raw_lower.size:
        raise TrafficLambdaError("lambda_star dimension mismatch")
    tol = 1e-12
    return bool(
        np.all(lam >= box.raw_lower - tol) and np.all(lam <= box.raw_upper + tol)
    )


def box_total_width(box: TrafficMixtureConfidenceBox) -> float:
    """Total raw coordinate width ``sum_j (upper_j - lower_j)`` (a size proxy)."""

    return float(np.sum(box.raw_upper - box.raw_lower))


@dataclass(frozen=True)
class TrafficLambdaResult:
    """The constructed mixture set plus its tail accounting and provenance."""

    counts: np.ndarray
    total: int
    delta_lambda: float
    box: TrafficMixtureConfidenceBox
    tail_report: dict
    fell_back_to_simplex: bool

    def as_dict(self) -> dict:
        return {
            "counts": self.counts.tolist(),
            "total": int(self.total),
            "delta_lambda": float(self.delta_lambda),
            "raw_lower": self.box.raw_lower.tolist(),
            "raw_upper": self.box.raw_upper.tolist(),
            "tightened_lower": self.box.lower.tolist(),
            "tightened_upper": self.box.upper.tolist(),
            "total_width": box_total_width(self.box),
            "tail_report": self.tail_report,
            "fell_back_to_simplex": bool(self.fell_back_to_simplex),
        }


def build_traffic_lambda(
    client_counts: Sequence[int], delta_lambda: float = DELTA_LAMBDA
) -> TrafficLambdaResult:
    """Assemble the mixture set and its accounting from client counts."""

    box = traffic_lambda_box(client_counts, delta_lambda)
    total = int(box.total)
    return TrafficLambdaResult(
        counts=np.asarray(box.counts, dtype=np.int64),
        total=total,
        delta_lambda=float(delta_lambda),
        box=box,
        tail_report=coordinate_tail_report(box),
        fell_back_to_simplex=(total == 0),
    )


__all__ = [
    "DELTA_C",
    "DELTA_LAMBDA",
    "DELTA_R",
    "M_GRID",
    "M_PRIMARY",
    "N_DOMAIN_CLIENTS",
    "TrafficLambdaError",
    "TrafficLambdaResult",
    "assert_traffic_identity_disjoint",
    "box_total_width",
    "build_traffic_lambda",
    "coordinate_tail_report",
    "covers",
    "draw_traffic_client_counts",
    "full_simplex_box",
    "traffic_lambda_box",
]
