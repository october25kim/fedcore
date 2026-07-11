"""Shared grouping, repartition, and scored-view helpers.

These utilities keep grouped-stratified certification code out of experiment
entry points. The leading-underscore names remain as compatibility aliases for
older scripts.
"""

from __future__ import annotations

import numpy as np

from fedcore.scores import scored_views


def make_group_map(n_clients: int, G: int) -> np.ndarray:
    """Public, data-independent client-to-group map using balanced blocks."""
    return np.array([c * G // n_clients for c in range(n_clients)], dtype=int)


def repartition_trusted_pool(pool, cert_frac, test_frac, seed):
    """Split pooled trusted points into disjoint prop/cert/test folds."""
    rng = np.random.default_rng(seed)
    n = len(pool["y_open"])
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_cert = int(round(n * cert_frac))
    idx = {
        "test": perm[:n_test],
        "cert": perm[n_test:n_test + n_cert],
        "prop": perm[n_test + n_cert:],
    }
    out = {}
    for fold, ix in idx.items():
        out[fold] = {k: pool[k][ix] for k in ("logits", "y_open", "client")}
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
