"""Exact mixture draws with immutable IDs and common random numbers.

These are source-level i.i.d. draws *with replacement*.  They are distinct from
the legacy fixed-quota fold construction and from without-replacement audit-budget
subsets.  A draw first samples a public group, then a client conditional on that
group, then one immutable reservoir ID conditional on the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _probability_vector(values: Sequence[float], name: str) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or np.any(~np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError(f"{name} must be a non-empty finite non-negative vector")
    if not np.isclose(p.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must sum to one")
    # Normalize only roundoff, never silently repair a malformed declaration.
    return p / p.sum()


@dataclass(frozen=True)
class GroupMixtureDraw:
    sample_id: np.ndarray
    group_id: np.ndarray
    client_id: np.ndarray
    reservoir_position: np.ndarray
    seed: int

    def prefix(self, n: int) -> "GroupMixtureDraw":
        if not 0 <= n <= len(self.sample_id):
            raise ValueError("prefix size outside draw")
        return GroupMixtureDraw(
            sample_id=self.sample_id[:n].copy(),
            group_id=self.group_id[:n].copy(),
            client_id=self.client_id[:n].copy(),
            reservoir_position=self.reservoir_position[:n].copy(),
            seed=self.seed,
        )


def sample_group_mixture(
    reservoir_ids: Mapping[int, Sequence[object]],
    client_to_group: Sequence[int],
    group_probabilities: Sequence[float],
    client_probabilities_given_group: Mapping[int, Sequence[float]],
    n: int,
    seed: int,
) -> GroupMixtureDraw:
    """Draw exactly from the declared categorical-then-reservoir law.

    ``client_probabilities_given_group[g]`` is aligned with the sorted clients
    assigned to group ``g``.  No label or prediction array is accepted by this API.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    client_group = np.asarray(client_to_group, dtype=int)
    if client_group.ndim != 1 or len(client_group) == 0 or np.any(client_group < 0):
        raise ValueError("client_to_group must be a non-empty non-negative vector")
    group_p = _probability_vector(group_probabilities, "group_probabilities")
    G = len(group_p)
    if np.any(client_group >= G):
        raise ValueError("client_to_group references an undeclared group")

    group_clients: dict[int, np.ndarray] = {}
    conditional: dict[int, np.ndarray] = {}
    reservoirs: dict[int, np.ndarray] = {}
    for client in range(len(client_group)):
        ids = np.asarray(reservoir_ids.get(client, ()), dtype=object)
        reservoirs[client] = ids
    for group in range(G):
        clients = np.flatnonzero(client_group == group)
        if len(clients) == 0 and group_p[group] > 0.0:
            raise ValueError(f"positive-probability group {group} has no clients")
        group_clients[group] = clients
        raw = client_probabilities_given_group.get(group)
        if raw is None:
            raise ValueError(f"missing client probabilities for group {group}")
        p = _probability_vector(raw, f"client_probabilities_given_group[{group}]")
        if len(p) != len(clients):
            raise ValueError(f"group {group} probability vector has wrong length")
        for client, probability in zip(clients, p):
            if probability > 0.0 and len(reservoirs[int(client)]) == 0:
                raise ValueError(
                    f"positive-probability client {client} has an empty reservoir"
                )
        conditional[group] = p

    rng = np.random.default_rng(int(seed))
    groups = np.empty(n, dtype=int)
    clients = np.empty(n, dtype=int)
    positions = np.empty(n, dtype=int)
    ids = np.empty(n, dtype=object)
    # Per-observation categorical construction is intentionally literal: it is
    # easy to audit and the first n draws are invariant when a larger budget is run.
    for i in range(n):
        group = int(rng.choice(G, p=group_p))
        groups[i] = group
        candidates = group_clients[group]
        client = int(rng.choice(candidates, p=conditional[group]))
        position = int(rng.integers(0, len(reservoirs[client])))
        clients[i] = client
        positions[i] = position
        ids[i] = reservoirs[client][position]
    return GroupMixtureDraw(ids, groups, clients, positions, int(seed))


def sample_traffic_clients(
    client_probabilities: Sequence[float], n: int, seed: int
) -> np.ndarray:
    """Draw site identities for ``Lambda_hat``; labels cannot enter this API."""
    if n < 0:
        raise ValueError("n must be non-negative")
    p = _probability_vector(client_probabilities, "client_probabilities")
    return np.random.default_rng(int(seed)).choice(len(p), size=n, p=p).astype(int)


def nested_without_replacement_indices(
    reservoir_size: int, sample_sizes: Sequence[int], seed: int
) -> dict[int, np.ndarray]:
    """One frozen permutation whose prefixes implement nested audit budgets."""
    sizes = sorted(set(int(value) for value in sample_sizes))
    if reservoir_size < 0 or any(
        value < 0 or value > reservoir_size for value in sizes
    ):
        raise ValueError("sample sizes must lie in [0, reservoir_size]")
    perm = np.random.default_rng(int(seed)).permutation(reservoir_size)
    return {size: perm[:size].copy() for size in sizes}


def stratified_nested_without_replacement_indices(
    strata: Sequence[int], sample_fractions: Sequence[float], seed: int
) -> dict[float, np.ndarray]:
    """Nested audit-budget prefixes within every declared stratum.

    A stratum-specific RNG stream prevents a large client from changing another
    client's permutation. Fractions affect prefix length only, never the stream.
    Returned indices are sorted back into immutable artifact order.
    """

    stratum = np.asarray(strata)
    if stratum.ndim != 1 or stratum.dtype.kind == "b":
        raise ValueError("strata must be a one-dimensional integer vector")
    numeric = stratum.astype(float)
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError("strata must contain finite integers")
    stratum = numeric.astype(np.int64)
    if np.any(stratum < 0):
        raise ValueError("strata must be non-negative")
    fractions = sorted(set(float(value) for value in sample_fractions))
    if not fractions or any(
        not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in fractions
    ):
        raise ValueError("sample fractions must be unique values in (0, 1]")
    selected: dict[float, list[np.ndarray]] = {value: [] for value in fractions}
    for value in np.unique(stratum):
        positions = np.flatnonzero(stratum == value)
        # SeedSequence is deterministic and makes the stratum ID a domain
        # component without relying on Python's salted hash().
        child_seed = np.random.SeedSequence([int(seed), int(value)]).generate_state(
            1, dtype=np.uint32
        )[0]
        permutation = positions[
            np.random.default_rng(child_seed).permutation(len(positions))
        ]
        for fraction in fractions:
            count = int(np.floor(fraction * len(positions)))
            selected[fraction].append(permutation[:count])
    return {
        fraction: (
            np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
            if parts
            else np.empty(0, dtype=np.int64)
        )
        for fraction, parts in selected.items()
    }
