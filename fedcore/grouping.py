"""Shared grouping, repartition, and scored-view helpers.

These utilities keep grouped-stratified certification code out of experiment
entry points. The leading-underscore names remain as compatibility aliases for
older scripts.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from fedcore.scores import scored_views


def make_group_map(n_clients: int, G: int) -> np.ndarray:
    """Public, data-independent client-to-group map using balanced CONTIGUOUS blocks.

    LIMITATION -- read before using for a declared partition. This builds contiguous
    blocks (``c*G//n_clients``). It therefore CANNOT express a non-contiguous declared
    partition: the sealed Fed-ISIC ``group_partition_G2``
    ``{HAM_derived: [1,2,3,5], other: [0,4]}`` is NOT contiguous, and this function
    returns ``[0,0,0,1,1,1]`` for it -- the WRONG grouping, silently.

    Use :func:`group_map_from_partition` whenever a partition is declared. This function
    is retained because contiguous blocks are the right map for the synthetic/CIFAR
    arms whose partitions ARE contiguous, and because golden regressions pin it.
    """
    return np.array([c * G // n_clients for c in range(n_clients)], dtype=int)


def group_map_from_partition(
    partition, n_clients: int | None = None
) -> np.ndarray:
    """Client-to-group map from an EXPLICIT declared partition. Order is deterministic.

    Accepts either the sealed CIFAR form -- a list of client lists,
    ``[[0,1,2],[3,4]]`` -- or the sealed Fed-ISIC form, a mapping
    ``{"HAM_derived": [1,2,3,5], "other": [0,4]}``. For the mapping form the group INDEX
    is the position of the group name in SORTED name order, so the coordinate order is
    stable across processes and matches
    ``fedcore.medical.group_mixture.group_names``.

    Unlike :func:`make_group_map` this expresses non-contiguous partitions, which the
    declared Fed-ISIC grouping requires. Every client must appear EXACTLY once: a
    missing client would silently leave a point ungrouped, and a duplicated one would
    put a single audit unit in two strata and break per-group independence.
    """
    if isinstance(partition, Mapping):
        groups = [list(partition[name]) for name in sorted(partition)]
    else:
        groups = [list(g) for g in partition]

    members = [int(c) for g in groups for c in g]
    if len(members) != len(set(members)):
        dupes = sorted({c for c in members if members.count(c) > 1})
        raise ValueError(f"partition assigns client(s) {dupes} to more than one group")
    if n_clients is None:
        n_clients = max(members) + 1
    missing = set(range(n_clients)) - set(members)
    if missing:
        raise ValueError(f"partition leaves client(s) {sorted(missing)} ungrouped")

    vec = np.full(int(n_clients), -1, dtype=int)
    for index, group in enumerate(groups):
        for client in group:
            vec[int(client)] = index
    return vec


def repartition_trusted_pool(pool, cert_frac, test_frac, seed):
    """Split pooled trusted points into disjoint prop/cert/test folds.

    Every key present in ``pool`` is carried through to each fold, so a caller that
    tags points with an identifier (e.g. the accounting layer's ``sample_id``) can
    follow individual points across the repartition. Callers passing the usual
    ``logits``/``y_open``/``client`` pool get byte-identical output.
    """
    rng = np.random.default_rng(seed)
    n = len(pool["y_open"])
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_cert = int(round(n * cert_frac))
    idx = {
        "test": perm[:n_test],
        "cert": perm[n_test : n_test + n_cert],
        "prop": perm[n_test + n_cert :],
    }
    out = {}
    for fold, ix in idx.items():
        out[fold] = {k: np.asarray(v)[ix] for k, v in pool.items()}
    return out


def views_from_parts(parts, score):
    """Build scored fold views from repartitioned logits/y_open/client parts."""
    return {
        fn: scored_views(
            parts[fn]["logits"], parts[fn]["y_open"], parts[fn]["client"], [score]
        )[score]
        for fn in ("prop", "cert", "test")
    }


_group_map = make_group_map
_repartition = repartition_trusted_pool
_views_from_parts = views_from_parts
